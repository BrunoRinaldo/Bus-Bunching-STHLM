"""Phase 6 - diagnostics (report task D).

Primary diagnostic model: HistGradientBoostingClassifier at k=3 (the core
RQ3 horizon), since it was confirmed the strongest tabular model (doc 3 §5)
and running every diagnostic across all 5 horizons would be excessive scope
for this phase - a deliberate, stated choice, not an oversight.

Covers all five items in the plan's Phase 6:
1. Performance broken out by line, hour of day, season, weather.
2. Ablation for H2: delay-only vs dwell-only vs both feature groups.
3. Robustness: drop the leader's features (simulates a lost-vehicle feed).
4. Transfer: train on a subset of lines, test on held-out lines.
5. Scalability: inference latency per prediction.
Plus: false-negative characterisation.
"""
import json
import time
import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight

import models as m  # reuses load(), prep_hgb_frame(), NUMERIC/BOOLEAN/CATEGORICAL

K = 3
DS = m.DS

DELAY_FEATURES = ["observed_arrival_delay", "observed_departure_delay",
                   "leader_arrival_delay", "leader_delay_growth", "cum_delay_growth_3"]
DWELL_FEATURES = ["dwell", "dwell_rel_median", "leader_dwell"]
LEADER_FEATURES = ["leader_dwell", "leader_arrival_delay", "leader_delay_growth"]

BASE = [c for c in m.ALL_COLS if c not in DELAY_FEATURES + DWELL_FEATURES]


def fit_hgb(Xtr, ytr, cols):
    Xt = m.prep_hgb_frame(Xtr[cols])
    cat_mask = [c in m.CATEGORICAL for c in cols]
    clf = HistGradientBoostingClassifier(categorical_features=cat_mask, max_iter=200, random_state=0)
    sw = compute_sample_weight("balanced", ytr)
    clf.fit(Xt, ytr, sample_weight=sw)
    return clf


def score(clf, X, y, cols):
    Xt = m.prep_hgb_frame(X[cols])
    proba = clf.predict_proba(Xt)[:, 1]
    return average_precision_score(y, proba), proba


def main():
    out = {}
    t0 = time.time()
    Xtr, ytr = m.load(K, "train")
    Xval, yval = m.load(K, "val")
    Xte, yte = m.load(K, "test")
    print(f"loaded train={len(ytr)} val={len(yval)} test={len(yte)} in {time.time()-t0:.1f}s")

    # full model (reuse the already-trained one for consistency with doc 3 §5)
    import joblib
    full_clf = joblib.load("models/hgb_k3.joblib")
    full_pr_auc, full_proba = score(full_clf, Xte, yte, m.ALL_COLS)
    print("full model test PR-AUC (sanity check, should match doc 3):", full_pr_auc)

    # ---------- 1. breakdown by line / hour / month / weather ----------
    con = duckdb.connect()
    con.sql("PRAGMA threads=8")
    meta = con.sql(f"""
        SELECT LineNumber, hour_of_day, month_of_year, wx_group, is_precip
        FROM '{DS}' WHERE split='test' AND y_k{K} IS NOT NULL
    """).df()
    assert len(meta) == len(yte)
    meta["y"] = yte
    meta["proba"] = full_proba
    meta["peak"] = meta.hour_of_day.isin([7, 8, 16, 17])

    def group_pr_auc(df, col):
        rows = []
        for val, g in df.groupby(col, observed=True):
            if g.y.sum() < 20:
                continue
            rows.append({col: str(val), "n": len(g), "base_rate_pct": round(100 * g.y.mean(), 3),
                         "pr_auc": round(average_precision_score(g.y, g.proba), 4)})
        return sorted(rows, key=lambda r: -r["pr_auc"])

    out["by_line"] = group_pr_auc(meta, "LineNumber")
    out["by_peak"] = group_pr_auc(meta, "peak")
    out["by_month"] = group_pr_auc(meta, "month_of_year")
    out["by_weather"] = group_pr_auc(meta, "wx_group")
    print("breakdown done")

    # ---------- 2. ablation: delay-only vs dwell-only vs both ----------
    for name, extra in [("delay_only", DELAY_FEATURES), ("dwell_only", DWELL_FEATURES),
                         ("both", DELAY_FEATURES + DWELL_FEATURES)]:
        cols = BASE + extra
        t0 = time.time()
        clf = fit_hgb(Xtr, ytr, cols)
        pr_auc, _ = score(clf, Xte, yte, cols)
        print(f"ablation {name}: PR-AUC={pr_auc:.4f} ({time.time()-t0:.1f}s, {len(cols)} features)")
        out.setdefault("ablation", {})[name] = {"pr_auc": round(pr_auc, 4), "n_features": len(cols)}
    out["ablation"]["base_only_no_delay_no_dwell"] = None  # not run: base alone isn't a plan-specified arm

    # ---------- 3. robustness: drop leader features ----------
    cols_no_leader = [c for c in m.ALL_COLS if c not in LEADER_FEATURES]
    t0 = time.time()
    clf_no_leader = fit_hgb(Xtr, ytr, cols_no_leader)
    pr_auc_no_leader, _ = score(clf_no_leader, Xte, yte, cols_no_leader)
    print(f"robustness (no leader features): PR-AUC={pr_auc_no_leader:.4f} ({time.time()-t0:.1f}s)")
    out["robustness_no_leader"] = {
        "pr_auc": round(pr_auc_no_leader, 4),
        "pr_auc_full": round(full_pr_auc, 4),
        "degradation_pct": round(100 * (full_pr_auc - pr_auc_no_leader) / full_pr_auc, 2),
    }

    # ---------- 4. transfer: train on high-volume lines, test on the rest ----------
    train_lines = ["4", "541", "474", "607"]
    test_lines = ["116", "117", "179", "401"]
    fetch_cols = [c for c in m.ALL_COLS if c != "temperature_2m_sq"]  # computed post-fetch, not a real column
    con_tr = con.sql(f"""
        SELECT {", ".join(fetch_cols)}, y_k{K} AS y FROM '{DS}'
        WHERE split='train' AND y_k{K} IS NOT NULL AND LineNumber IN ({",".join(repr(l) for l in train_lines)})
    """).df()
    con_te_heldout = con.sql(f"""
        SELECT {", ".join(fetch_cols)}, y_k{K} AS y FROM '{DS}'
        WHERE split='test' AND y_k{K} IS NOT NULL AND LineNumber IN ({",".join(repr(l) for l in test_lines)})
    """).df()
    for b in m.BOOLEAN:
        con_tr[b] = con_tr[b].astype("float64"); con_te_heldout[b] = con_te_heldout[b].astype("float64")
    for c in m.CATEGORICAL:
        con_tr[c] = con_tr[c].astype(str); con_te_heldout[c] = con_te_heldout[c].astype(str)
    con_tr["temperature_2m_sq"] = con_tr["temperature_2m"] ** 2
    con_te_heldout["temperature_2m_sq"] = con_te_heldout["temperature_2m"] ** 2
    y_tr_transfer = con_tr["y"].astype(int).values
    y_te_heldout = con_te_heldout["y"].astype(int).values

    t0 = time.time()
    clf_transfer = fit_hgb(con_tr, y_tr_transfer, m.ALL_COLS)
    pr_auc_heldout_lines, _ = score(clf_transfer, con_te_heldout, y_te_heldout, m.ALL_COLS)
    # same transfer model, scored on its OWN lines' test rows, for a same-model apples-to-apples comparison
    con_te_ownlines = con.sql(f"""
        SELECT {", ".join(fetch_cols)}, y_k{K} AS y FROM '{DS}'
        WHERE split='test' AND y_k{K} IS NOT NULL AND LineNumber IN ({",".join(repr(l) for l in train_lines)})
    """).df()
    for b in m.BOOLEAN:
        con_te_ownlines[b] = con_te_ownlines[b].astype("float64")
    for c in m.CATEGORICAL:
        con_te_ownlines[c] = con_te_ownlines[c].astype(str)
    con_te_ownlines["temperature_2m_sq"] = con_te_ownlines["temperature_2m"] ** 2
    y_te_ownlines = con_te_ownlines["y"].astype(int).values
    pr_auc_ownlines, _ = score(clf_transfer, con_te_ownlines, y_te_ownlines, m.ALL_COLS)
    print(f"transfer: trained on {train_lines}, PR-AUC on own lines={pr_auc_ownlines:.4f}, "
          f"on held-out lines {test_lines}={pr_auc_heldout_lines:.4f} ({time.time()-t0:.1f}s)")
    out["transfer"] = {
        "train_lines": train_lines, "test_lines_heldout": test_lines,
        "pr_auc_on_own_lines": round(pr_auc_ownlines, 4),
        "pr_auc_on_heldout_lines": round(pr_auc_heldout_lines, 4),
        "degradation_pct": round(100 * (pr_auc_ownlines - pr_auc_heldout_lines) / pr_auc_ownlines, 2),
    }

    # ---------- 5. scalability: inference latency ----------
    Xt_full = m.prep_hgb_frame(Xte[m.ALL_COLS])
    sample = Xt_full.iloc[:20000]
    t0 = time.time()
    full_clf.predict_proba(sample)
    batch_time = time.time() - t0
    t0 = time.time()
    for i in range(200):
        full_clf.predict_proba(Xt_full.iloc[[i]])
    single_time = (time.time() - t0) / 200
    out["scalability"] = {
        "batch_20000_rows_total_s": round(batch_time, 4),
        "batch_per_row_ms": round(1000 * batch_time / len(sample), 5),
        "single_row_predict_ms": round(1000 * single_time, 4),
    }
    print("scalability:", out["scalability"])

    # ---------- false negatives ----------
    thresh = 0.5
    pred = (full_proba >= thresh).astype(int)
    fn_mask = (yte == 1) & (pred == 0)
    tp_mask = (yte == 1) & (pred == 1)
    fn_df = meta[fn_mask.values if hasattr(fn_mask, "values") else fn_mask]
    tp_df = meta[tp_mask.values if hasattr(tp_mask, "values") else tp_mask]
    out["false_negatives"] = {
        "n_false_negatives": int(fn_mask.sum()), "n_true_positives": int(tp_mask.sum()),
        "fn_by_line_pct": (fn_df.LineNumber.value_counts(normalize=True) * 100).round(1).to_dict(),
        "tp_by_line_pct": (tp_df.LineNumber.value_counts(normalize=True) * 100).round(1).to_dict(),
        "fn_peak_share_pct": round(100 * fn_df.peak.mean(), 1),
        "tp_peak_share_pct": round(100 * tp_df.peak.mean(), 1),
        "fn_mean_proba": round(float(full_proba[fn_mask].mean()), 4) if fn_mask.sum() else None,
    }
    print("false negatives:", out["false_negatives"])

    with open("reports/phase6_diagnostics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote reports/phase6_diagnostics.json")


if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)
    main()
