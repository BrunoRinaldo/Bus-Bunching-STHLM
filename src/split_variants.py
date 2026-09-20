"""Compare three train/val/test split strategies on the same model and
horizon, to test whether the plan's "never split randomly" warning and the
cold-weather validation-coverage gap (docs/03 §1.1) actually matter
empirically, not just in principle.

Three splits, same underlying data (data/processed/model_dataset.parquet),
same model (HistGradientBoostingClassifier), same horizon (k=3, the core
RQ3 horizon, same choice as Phase 6 diagnostics):

1. temporal (primary, already trained) - train Jan-Sep, val Oct, test Nov-Dec.
   Reused from models/phase5_results.json, not retrained here.
2. random (trip-level) - hash(date, trip_id) % 100 assigns whole trips to
   train/val/test, scattered across all of 2024. Still avoids the worst
   leakage (splitting a single trip's own stops across train and test), but
   lets very similar nearby-in-time conditions appear on both sides of the
   split, unlike the temporal split.
3. weekly - ISO week number round-robin (7 of every 10 weeks -> train, 1 ->
   val, 2 -> test), so every split gets a representative mix of seasons
   instead of one solid block. Same block-based avoidance of within-trip
   leakage as the temporal split, but distributes val/test across the year.

IMPORTANT CAVEAT reported alongside every number below: these three splits
evaluate on fundamentally different populations (temporal test = Nov-Dec
only; random test = a scattered whole-year sample; weekly test = a
scattered whole-year sample with block structure). A PR-AUC difference
between them reflects BOTH leakage differences AND different underlying
test-set difficulty (e.g. summer months have looser schedules and different
bunching dynamics - Phase 3 Figure 3). Do not read a higher number as purely
"less leakage" or "better split" without accounting for this.
"""
import json
import time
import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

import models as m

K = 3
DS = m.DS
FETCH_COLS = [c for c in m.ALL_COLS if c != "temperature_2m_sq"]

con = duckdb.connect()
con.sql("PRAGMA threads=8")

SPLIT_CTE = """
WITH base AS (
    SELECT *,
        hash(date::VARCHAR || '-' || trip_id::VARCHAR) % 100 AS rand_bucket,
        (date_diff('day', DATE '2024-01-01',
                    strptime(CAST(date AS VARCHAR), '%Y%m%d')) / 7)::INT AS week_num
    FROM '{DS}'
),
assigned AS (
    SELECT *,
        CASE WHEN rand_bucket < 75 THEN 'train'
             WHEN rand_bucket < 85 THEN 'val'
             ELSE 'test' END AS split_random,
        CASE WHEN week_num % 10 < 7 THEN 'train'
             WHEN week_num % 10 = 7 THEN 'val'
             ELSE 'test' END AS split_weekly
    FROM base
)
""".format(DS=DS)


def load_variant(split_col, split_val):
    cols_sql = ", ".join(FETCH_COLS + [f"y_k{K} AS y"])
    q = f"""
        {SPLIT_CTE}
        SELECT {cols_sql} FROM assigned
        WHERE {split_col} = '{split_val}' AND y_k{K} IS NOT NULL
    """
    df = con.sql(q).df()
    df["temperature_2m_sq"] = df["temperature_2m"] ** 2
    for b in m.BOOLEAN:
        df[b] = df[b].astype("float64")
    for c in m.CATEGORICAL:
        df[c] = df[c].astype(str)
    X = df[m.ALL_COLS]
    y = df["y"].astype(int).values
    return X, y


def weather_coverage(split_col, split_val):
    q = f"""
        {SPLIT_CTE}
        SELECT round(min(temperature_2m),1) min_t, round(avg(temperature_2m),1) mean_t,
               round(100.0*sum(is_snowing::INT)/count(*),2) pct_snowing,
               round(100.0*sum(ice_risk::INT)/count(*),2) pct_ice_risk,
               count(*) n
        FROM assigned WHERE {split_col} = '{split_val}' AND temperature_2m IS NOT NULL
    """
    r = con.sql(q).fetchone()
    return {"min_temp": r[0], "mean_temp": r[1], "pct_snowing": r[2], "pct_ice_risk": r[3], "n": r[4]}


def fit_hgb(Xtr, ytr):
    Xt = m.prep_hgb_frame(Xtr)
    cat_mask = [c in m.CATEGORICAL for c in m.ALL_COLS]
    clf = HistGradientBoostingClassifier(categorical_features=cat_mask, max_iter=200, random_state=0)
    sw = compute_sample_weight("balanced", ytr)
    clf.fit(Xt, ytr, sample_weight=sw)
    return clf


def score(clf, X, y):
    Xt = m.prep_hgb_frame(X)
    proba = clf.predict_proba(Xt)[:, 1]
    return average_precision_score(y, proba)


def main():
    out = {}

    # reused, not retrained
    with open("models/phase5_results.json") as f:
        phase5 = json.load(f)
    out["temporal"] = {
        "test_pr_auc": phase5["3"]["hgb_test"]["pr_auc"],
        "test_n": phase5["3"]["hgb_test"]["n"],
        "test_base_rate_pct": phase5["3"]["hgb_test"]["base_rate_pct"],
        "note": "reused from the primary run, not retrained here",
    }
    out["temporal"]["weather"] = weather_coverage("split", "val")  # primary split's val (Oct)
    out["temporal"]["weather_test"] = weather_coverage("split", "test")

    for name, col in [("random", "split_random"), ("weekly", "split_weekly")]:
        t0 = time.time()
        Xtr, ytr = load_variant(col, "train")
        Xval, yval = load_variant(col, "val")
        Xte, yte = load_variant(col, "test")
        print(f"{name}: train={len(ytr)} val={len(yval)} test={len(yte)} loaded in {time.time()-t0:.1f}s")

        clf = fit_hgb(Xtr, ytr)
        pr_auc = score(clf, Xte, yte)
        print(f"{name}: test PR-AUC={pr_auc:.4f} (fit+score {time.time()-t0:.1f}s total)")
        import joblib
        joblib.dump(clf, f"models/hgb_k3_split_{name}.joblib")

        # same-population re-score: this split's model, evaluated ONLY on
        # Nov-Dec rows (whatever this split called them), to isolate the
        # leakage effect from the population-mix effect (temporal's test is
        # Nov-Dec only, so a raw whole-year-vs-Nov-Dec comparison confounds
        # "different split" with "different evaluation population").
        cols_sql = ", ".join(FETCH_COLS + [f"y_k{K} AS y"])
        q_novdec = f"""
            {SPLIT_CTE}
            SELECT {cols_sql} FROM assigned
            WHERE {col} = 'test' AND y_k{K} IS NOT NULL
              AND (date/100)::INT % 100 IN (11, 12)
        """
        df_nd = con.sql(q_novdec).df()
        df_nd["temperature_2m_sq"] = df_nd["temperature_2m"] ** 2
        for b in m.BOOLEAN:
            df_nd[b] = df_nd[b].astype("float64")
        for c in m.CATEGORICAL:
            df_nd[c] = df_nd[c].astype(str)
        X_nd, y_nd = df_nd[m.ALL_COLS], df_nd["y"].astype(int).values
        pr_auc_novdec = score(clf, X_nd, y_nd) if len(y_nd) > 100 else None
        print(f"{name}: Nov-Dec-only subset of its own test set: n={len(y_nd)}, PR-AUC={pr_auc_novdec}")

        out[name] = {
            "test_pr_auc": round(pr_auc, 4),
            "test_n": len(yte),
            "test_base_rate_pct": round(100 * yte.mean(), 3),
            "test_pr_auc_novdec_subset_only": round(pr_auc_novdec, 4) if pr_auc_novdec else None,
            "test_n_novdec_subset": len(y_nd),
            "train_n": len(ytr), "val_n": len(yval),
            "weather": weather_coverage(col, "val"),
            "weather_test": weather_coverage(col, "test"),
        }
        print(f"{name} val weather coverage: {out[name]['weather']}")

    with open("reports/split_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
