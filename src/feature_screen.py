"""Step 3 screening: which candidate feature groups help HGB at k=5 and k=8?

For each horizon, a fixed HGB configuration (the per-horizon winner from
data/results/tuning_per_k.json) is fit on the same 800k-row training subsample
with each feature set, early-stopped on the validation split (Oct), and
scored on the validation split. The test split is not touched here.

Feature sets: current features; minus month_of_year (train has months 1-9,
val 10 and test 11-12, so the model never sees the test months); then
minus month plus each group from src/extra_features.py, and plus all groups.
Results: data/results/feature_screen.json.

Usage: .venv/bin/python src/feature_screen.py
"""
import json
import time

import duckdb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

import models as m
from extra_features import GROUPS, OUT as EXTRA

HORIZONS = [5, 8]
SUB_N = 800_000
OUT = "data/results/feature_screen.json"
EXTRA_COLS = [c for g in GROUPS.values() for c in g]
con = duckdb.connect()
con.sql("PRAGMA threads=8")


def load(k, split):
    base = [c for c in m.ALL_COLS if c != "temperature_2m_sq"]
    q = f"""
        SELECT {", ".join("d." + c for c in base)}, {", ".join("e." + c for c in EXTRA_COLS)},
               d.date, d.y_k{k} AS y
        FROM '{m.DS}' d JOIN '{EXTRA}' e USING (date, trip_id, stop_sequence)
        WHERE d.split = '{split}' AND d.y_k{k} IS NOT NULL
        ORDER BY d.date, d.trip_id, d.stop_sequence
    """
    df = con.sql(q).df()
    df["temperature_2m_sq"] = df["temperature_2m"] ** 2
    for b in m.BOOLEAN:
        df[b] = df[b].astype("float64")
    for c in m.CATEGORICAL:
        df[c] = df[c].astype(str)
    return df


def fit_score(cols, tr, ytr, va, yva, params):
    clf = HistGradientBoostingClassifier(
        categorical_features=[c in m.CATEGORICAL for c in cols], early_stopping=True,
        n_iter_no_change=15, random_state=0, **params)
    clf.fit(m.prep_hgb_frame(tr[cols]), ytr, sample_weight=compute_sample_weight("balanced", ytr),
            X_val=m.prep_hgb_frame(va[cols]), y_val=yva,
            sample_weight_val=compute_sample_weight("balanced", yva))
    p = clf.predict_proba(m.prep_hgb_frame(va[cols]))[:, 1]
    return float(average_precision_score(yva, p)), int(clf.n_iter_)


def main():
    pk = json.load(open("data/results/tuning_per_k.json"))
    res = {}
    for k in HORIZONS:
        t0 = time.time()
        tr, va = load(k, "train"), load(k, "val")
        idx = np.random.RandomState(0).choice(len(tr), SUB_N, replace=False)
        tr = tr.iloc[idx].reset_index(drop=True)
        ytr, yva = tr["y"].to_numpy(int), va["y"].to_numpy(int)
        params = dict(pk[f"k{k}"]["hgb_final"]["params"], max_iter=2000)
        base = list(m.ALL_COLS)
        no_month = [c for c in base if c != "month_of_year"]
        sets = {"current": base, "no_month": no_month}
        for g, cols in GROUPS.items():
            sets[f"no_month+{g}"] = no_month + cols
        sets["no_month+all"] = no_month + EXTRA_COLS
        res[str(k)] = {"params": params, "sets": {}}
        print(f"=== k={k} (loaded in {time.time()-t0:.0f}s) ===", flush=True)
        for name, cols in sets.items():
            t1 = time.time()
            v, n_iter = fit_score(cols, tr, ytr, va, yva, params)
            res[str(k)]["sets"][name] = {"val_pr_auc": round(v, 4), "n_iter": n_iter, "n_features": len(cols),
                                         "fit_s": round(time.time() - t1, 1)}
            print(f"  {name:22s} val={v:.4f} n_iter={n_iter} ({time.time()-t1:.0f}s)", flush=True)
            with open(OUT, "w") as f:
                json.dump(res, f, indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
