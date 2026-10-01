"""Phase 6 diagnostics re-run on the tuned gradient boosting models, k=3 and k=5.

Same diagnostics as src/diagnostics.py (which used the default model at k=3),
on the best tuned HGB per horizon:
  k=3  models/tuned_hgb_k3.joblib  (round 1, docs/03 §6.2; test PR-AUC 0.856)
  k=5  models/long_hgb_k5.joblib   (long-horizon search, docs/03 §6.6; 0.776)
Every diagnostic that trains a new model (ablation, no-leader, transfer) uses
that horizon's tuned configuration and training procedure: internal early
stopping for k=3 (as in src/tuning.py), early stopping on the validation
split for k=5 (as in src/tuning_long_k.py).

Additions against the original: 95% day-block bootstrap intervals for every
group PR-AUC (B = 1000 over test days) and permutation importance for the
tuned model (60,000 test rows, PR-AUC, 5 repeats).

Output (aggregates only): reports/diagnostics_tuned.json.
Usage: .venv/bin/python src/diagnostics_tuned.py
"""
import json
import time

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

import models as m
import tuning as t
from bootstrap_perk import metrics_weighted, prepare
from diagnostics import DELAY_FEATURES, DWELL_FEATURES, LEADER_FEATURES

OUT = "reports/diagnostics_tuned.json"
B = 1000
TRAIN_LINES = ["4", "541", "474", "607"]
HELDOUT_LINES = ["116", "117", "179", "401"]
con = duckdb.connect()
con.sql("PRAGMA threads=8")


def config(k):
    if k == 3:
        params = json.load(open("reports/tuning_stage1.json"))["hgb_best"]["params"]
        return {"model": joblib.load("models/tuned_hgb_k3.joblib"), "features": list(m.ALL_COLS),
                "params": params, "early_stopping": "internal 10% of training rows",
                "source": "models/tuned_hgb_k3.joblib"}
    lk = json.load(open("reports/tuning_long_k.json"))[f"k{k}"]
    params = next(r for r in lk["refits"] if r["name"] == lk["chosen"])["params"]
    obj = joblib.load(f"models/long_hgb_k{k}.joblib")
    return {"model": obj["clf"], "features": obj["features"], "params": params,
            "early_stopping": "validation split (Oct)", "source": f"models/long_hgb_k{k}.joblib"}


def fit(cfg, cols, X, y, Xv, yv):
    if cfg["early_stopping"].startswith("internal"):
        return t.fit_hgb(X[cols], y, cfg["params"])
    clf = HistGradientBoostingClassifier(
        categorical_features=[c in m.CATEGORICAL for c in cols], early_stopping=True,
        n_iter_no_change=20, random_state=0, **cfg["params"])
    clf.fit(m.prep_hgb_frame(X[cols]), y, sample_weight=compute_sample_weight("balanced", y),
            X_val=m.prep_hgb_frame(Xv[cols]), y_val=yv, sample_weight_val=compute_sample_weight("balanced", yv))
    return clf


def proba(clf, cols, X):
    return clf.predict_proba(m.prep_hgb_frame(X[cols]))[:, 1]


def meta(k, split):
    return con.sql(f"SELECT LineNumber, hour_of_day, month_of_year, wx_group, date, y_k{k} AS y FROM '{m.DS}' "
                   f"WHERE split='{split}' AND y_k{k} IS NOT NULL").df()


def boot_ci(y, p, dates):
    days, idx = np.unique(dates, return_inverse=True)
    order, last = prepare(p)
    rng = np.random.RandomState(0)
    v = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.randint(0, len(days), len(days)), minlength=len(days))[idx].astype(float)
        v[b] = metrics_weighted(order, last, y, w)[0]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def by_group(md, col):
    rows = []
    for val, g in md.groupby(col, observed=True):
        y, p = g["y"].to_numpy(int), g["proba"].to_numpy()
        if y.sum() < 20:
            continue
        rows.append({col: str(val), "n": int(len(g)), "base_rate_pct": round(100 * y.mean(), 3),
                     "pr_auc": round(float(average_precision_score(y, p)), 4),
                     "ci95": boot_ci(y, p, g["date"].to_numpy())})
    return sorted(rows, key=lambda r: -r["pr_auc"])


def run(k):
    t0 = time.time()
    cfg = config(k)
    cols = cfg["features"]
    Xtr, ytr = m.load(k, "train")
    Xva, yva = m.load(k, "val")
    Xte, yte = m.load(k, "test")
    md = meta(k, "test")
    assert (md["y"].astype(int).values == yte).all(), "row order mismatch"
    mv = meta(k, "val")
    assert (mv["y"].astype(int).values == yva).all(), "row order mismatch"
    print(f"=== k={k}: loaded in {time.time()-t0:.0f}s ===", flush=True)
    out = {"model": cfg["source"], "params": cfg["params"], "early_stopping": cfg["early_stopping"]}

    p_full = proba(cfg["model"], cols, Xte)
    full = float(average_precision_score(yte, p_full))
    out["full_pr_auc"] = round(full, 4)
    out["full_ci95"] = boot_ci(yte, p_full, md["date"].to_numpy())
    print(f"  full model test PR-AUC {full:.4f} {out['full_ci95']}", flush=True)

    # 1. breakdowns
    md["proba"] = p_full
    md["peak"] = md["hour_of_day"].isin([7, 8, 16, 17])
    for name, col in [("by_line", "LineNumber"), ("by_peak", "peak"), ("by_month", "month_of_year"),
                      ("by_weather", "wx_group")]:
        out[name] = by_group(md, col)
    print("  breakdowns done", flush=True)

    # 2. ablation (H2); 'both' = all current delay + dwell features = the full feature list
    base = [c for c in cols if c not in DELAY_FEATURES + DWELL_FEATURES]
    out["ablation"] = {}
    for name, extra in [("delay_only", DELAY_FEATURES), ("dwell_only", DWELL_FEATURES)]:
        t1 = time.time()
        c = base + extra
        clf = fit(cfg, c, Xtr, ytr, Xva, yva)
        v = float(average_precision_score(yte, proba(clf, c, Xte)))
        out["ablation"][name] = {"pr_auc": round(v, 4), "n_features": len(c), "n_iter": int(clf.n_iter_)}
        print(f"  ablation {name}: {v:.4f} ({time.time()-t1:.0f}s)", flush=True)
    out["ablation"]["both"] = {"pr_auc": round(full, 4), "n_features": len(cols), "note": "the full tuned model"}

    # 3. robustness: no leader features
    t1 = time.time()
    c = [x for x in cols if x not in LEADER_FEATURES]
    clf = fit(cfg, c, Xtr, ytr, Xva, yva)
    v = float(average_precision_score(yte, proba(clf, c, Xte)))
    out["robustness_no_leader"] = {"pr_auc": round(v, 4), "pr_auc_full": round(full, 4),
                                   "degradation_pct": round(100 * (full - v) / full, 2)}
    print(f"  no leader: {v:.4f} ({time.time()-t1:.0f}s)", flush=True)

    # 4. transfer: train on 4 lines, score own vs held-out lines
    t1 = time.time()
    tr_m = Xtr["LineNumber"].isin(TRAIN_LINES).to_numpy()
    va_m = Xva["LineNumber"].isin(TRAIN_LINES).to_numpy()
    clf = fit(cfg, cols, Xtr[tr_m].reset_index(drop=True), ytr[tr_m],
              Xva[va_m].reset_index(drop=True), yva[va_m])
    pt = proba(clf, cols, Xte)
    own = Xte["LineNumber"].isin(TRAIN_LINES).to_numpy()
    held = Xte["LineNumber"].isin(HELDOUT_LINES).to_numpy()
    a_own = float(average_precision_score(yte[own], pt[own]))
    a_held = float(average_precision_score(yte[held], pt[held]))
    out["transfer"] = {"train_lines": TRAIN_LINES, "heldout_lines": HELDOUT_LINES,
                       "pr_auc_on_own_lines": round(a_own, 4), "pr_auc_on_heldout_lines": round(a_held, 4),
                       "own_ci95": boot_ci(yte[own], pt[own], md["date"].to_numpy()[own]),
                       "heldout_ci95": boot_ci(yte[held], pt[held], md["date"].to_numpy()[held]),
                       "full_model_on_heldout_lines": round(float(average_precision_score(yte[held], p_full[held])), 4),
                       "degradation_pct": round(100 * (a_own - a_held) / a_own, 2)}
    print(f"  transfer: own {a_own:.4f}, held-out {a_held:.4f} ({time.time()-t1:.0f}s)", flush=True)

    # 5. scalability
    Xt = m.prep_hgb_frame(Xte[cols])
    t1 = time.time()
    cfg["model"].predict_proba(Xt.iloc[:20000])
    batch = time.time() - t1
    t1 = time.time()
    for i in range(200):
        cfg["model"].predict_proba(Xt.iloc[[i]])
    out["scalability"] = {"batch_per_row_ms": round(1000 * batch / 20000, 5),
                          "single_row_predict_ms": round(1000 * (time.time() - t1) / 200, 3)}

    # 6. false negatives at threshold 0.5
    pred = p_full >= 0.5
    fn, tp = (yte == 1) & ~pred, (yte == 1) & pred
    out["false_negatives"] = {
        "n_false_negatives": int(fn.sum()), "n_true_positives": int(tp.sum()),
        "recall_at_0.5": round(float(tp.sum() / (yte == 1).sum()), 4),
        "fn_by_line_pct": (md[fn]["LineNumber"].value_counts(normalize=True) * 100).round(1).to_dict(),
        "tp_by_line_pct": (md[tp]["LineNumber"].value_counts(normalize=True) * 100).round(1).to_dict(),
        "fn_peak_share_pct": round(100 * float(md[fn]["peak"].mean()), 1),
        "tp_peak_share_pct": round(100 * float(md[tp]["peak"].mean()), 1),
        "fn_mean_proba": round(float(p_full[fn].mean()), 4)}

    # 7. permutation importance, 60k test rows
    t1 = time.time()
    s = np.random.RandomState(0).choice(len(yte), 60_000, replace=False)
    pi = permutation_importance(cfg["model"], Xt.iloc[s], yte[s], scoring="average_precision",
                                n_repeats=5, random_state=0)
    imp = pd.DataFrame({"feature": cols, "mean": pi.importances_mean, "std": pi.importances_std})
    imp = imp.sort_values("mean", ascending=False)
    out["permutation_importance"] = [{"feature": r.feature, "mean": round(float(r.mean), 4),
                                      "std": round(float(r.std), 4)} for r in imp.itertuples()]
    print(f"  permutation importance ({time.time()-t1:.0f}s): "
          f"{[(r['feature'], r['mean']) for r in out['permutation_importance'][:6]]}", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main():
    res = {}
    for k in [3, 5]:
        res[str(k)] = run(k)
        with open(OUT, "w") as f:
            json.dump(res, f, indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
