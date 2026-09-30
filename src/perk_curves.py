"""Test-split curves for the per-horizon tuned models (models/perk_*).

Scores the test split (Nov-Dec) with every per-k model and stores only
aggregates in reports/perk_curves.json: the PR curve sampled at fixed recall
levels, a 10-bin reliability curve, and (tabular models) PR-AUC per line.
Nothing row-level is written, so the output can be shared.

Two environments, as for training:
  .venv/bin/python    src/perk_curves.py          HGB + logistic regression
  .venv312/bin/python src/perk_curves.py --keras  Keras sequence models
Each run merges its models into the JSON and leaves the others untouched.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

HORIZONS = [1, 2, 3, 5, 8]
OUT = "reports/perk_curves.json"
RECALL_GRID = np.round(np.linspace(0, 1, 101), 2)
N_BINS = 10


def pr_curve(y, p):
    prec, rec, _ = precision_recall_curve(y, p)
    # interpolated precision: best precision reachable at recall >= r
    out = [float(prec[rec >= r].max()) if (rec >= r).any() else None for r in RECALL_GRID]
    return {"recall": RECALL_GRID.tolist(), "precision": [None if v is None else round(v, 4) for v in out]}


def reliability(y, p):
    edges = np.linspace(0, 1, N_BINS + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, N_BINS - 1)
    rows = []
    for i in range(N_BINS):
        s = b == i
        if s.sum():
            rows.append({"bin": i, "mean_pred": round(float(p[s].mean()), 4),
                         "obs_rate": round(float(y[s].mean()), 4), "n": int(s.sum())})
    return rows


def summary(y, p):
    return {"n": int(len(y)), "base_rate": round(float(y.mean()), 5),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "pr_curve": pr_curve(y, p), "reliability": reliability(y, p)}


def per_line(y, p, lines):
    out = {}
    for ln in sorted(set(lines)):
        s = lines == ln
        if y[s].sum() > 0:
            out[str(ln)] = {"n": int(s.sum()), "base_rate": round(float(y[s].mean()), 5),
                            "pr_auc": round(float(average_precision_score(y[s], p[s])), 4)}
    return out


def tabular(res):
    import joblib
    import models as m
    import tuning as t

    for k in HORIZONS:
        X, y = m.load(k, "test")
        lines = X["LineNumber"].to_numpy()
        hgb = joblib.load(f"models/perk_hgb_k{k}.joblib")
        p = t.hgb_proba(hgb, X)
        res.setdefault("hgb", {})[str(k)] = dict(summary(y, p), per_line=per_line(y, p, lines))
        lr = joblib.load(f"models/perk_logreg_k{k}.joblib")
        Z = t.tchunks(lr["pre"], X, lr["features"])
        if lr["pca"] is not None:
            Z = lr["pca"].transform(Z)
        p = lr["clf"].predict_proba(Z)[:, 1]
        res.setdefault("logreg", {})[str(k)] = dict(summary(y, p), per_line=per_line(y, p, lines))
        print(f"k={k}: hgb {res['hgb'][str(k)]['pr_auc']}  logreg {res['logreg'][str(k)]['pr_auc']}", flush=True)
        del X, y, Z


def sequence(res):
    import keras
    import tuning_keras as tk

    for k in HORIZONS:
        _, _, (Xte, yte) = tk.load_k(k)
        model = keras.saving.load_model(f"models/perk_sequence_k{k}.keras")
        p = model.predict(Xte, batch_size=8192, verbose=0).ravel().astype("float64")
        res.setdefault("keras", {})[str(k)] = summary(yte, p)
        print(f"k={k}: keras {res['keras'][str(k)]['pr_auc']}", flush=True)
        keras.backend.clear_session()


def main():
    res = {}
    (sequence if "--keras" in sys.argv else tabular)(res)
    # re-read just before writing so a concurrent run for the other models is not overwritten
    merged = json.load(open(OUT)) if os.path.exists(OUT) else {}
    merged.update(res)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=1)
    os.replace(tmp, OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
