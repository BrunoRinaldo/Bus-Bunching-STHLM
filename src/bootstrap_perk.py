"""Paired day-block bootstrap: per-horizon tuned models vs. the k=3-tuned configuration.

For every horizon, both models score the same test rows (Nov-Dec). Rows
within a service day share weather, disruptions and traffic, so days, not
rows, are resampled (B = 1000, with replacement). Each resample gives the
PR-AUC and recall at precision 0.80 of both models on identical data; the
2.5 / 97.5 percentiles of the difference are the 95% interval.

Comparison models (the fig 9 setup, searched at k=3 and reused elsewhere):
  HGB     stage-1 winner refit on the full training split, as in tuning.py
          stage2 (models/tuned_hgb_k3.joblib at k=3)
  LogReg  engineered features + PCA 45 + C=0.1, as in tuning.py stage2
  Keras   models/tuned_sequence_k{k}.keras
The refits reproduce the stage-2 test PR-AUC exactly (seeded); this is
asserted. Only aggregates are written, to data/results/perk_bootstrap.json.

  .venv/bin/python    src/bootstrap_perk.py          HGB + logistic regression
  .venv312/bin/python src/bootstrap_perk.py --keras  Keras
"""
import json
import os
import sys
import time

import duckdb
import numpy as np
from sklearn.metrics import average_precision_score

HORIZONS = [1, 2, 3, 5, 8]
B = 1000
OUT = "data/results/perk_bootstrap.json"
con = duckdb.connect()


def test_dates(path, k):
    return con.sql(f"SELECT date, y_k{k} AS y FROM '{path}' "
                   f"WHERE split='test' AND y_k{k} IS NOT NULL").df()


def metrics_weighted(order, last, y, w):
    """Weighted PR-AUC (sklearn's average precision) and recall at precision 0.80.
    order: indices sorting scores descending; last: True at the last row of each tied score."""
    tp = np.cumsum((w * y)[order])[last]
    fp = np.cumsum((w * (1 - y))[order])[last]
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / tp[-1]
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    ok = prec >= 0.80
    return ap, float(rec[ok].max()) if ok.any() else 0.0


def prepare(p):
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    last = np.ones(len(ps), dtype=bool)
    last[:-1] = ps[1:] != ps[:-1]
    return order, last


def recall_at_precision(y, p, target):
    """Highest-recall threshold with precision >= target, cutting only between distinct scores."""
    order, last = prepare(p)
    tp = np.cumsum(y[order])
    prec = tp / np.arange(1, len(tp) + 1)
    ok = np.nonzero((prec >= target) & last)[0]
    if not len(ok):
        return None
    i = int(ok.max())
    return {"recall": round(float(tp[i]) / int(y.sum()), 4), "precision": round(float(prec[i]), 4),
            "threshold": round(float(p[order][i]), 4)}


def paired_bootstrap(y, p_new, p_old, dates):
    days, day_idx = np.unique(dates, return_inverse=True)
    prep = [prepare(p_new), prepare(p_old)]

    def score(w):
        return [metrics_weighted(order, last, y, w) for order, last in prep]

    (a_new, r_new), (a_old, r_old) = score(np.ones(len(y)))
    assert abs(a_new - average_precision_score(y, p_new)) < 1e-6
    rng = np.random.RandomState(0)
    d_auc, d_r80 = np.empty(B), np.empty(B)
    for b in range(B):
        counts = np.bincount(rng.randint(0, len(days), len(days)), minlength=len(days))
        (an, rn), (ao, ro) = score(counts[day_idx].astype(float))
        d_auc[b], d_r80[b] = an - ao, rn - ro
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"n_rows": int(len(y)), "n_days": int(len(days)), "B": B,
            "pr_auc_new": round(a_new, 4), "pr_auc_old": round(a_old, 4),
            "diff": round(a_new - a_old, 4), "ci95": q(d_auc), "share_positive": round(float((d_auc > 0).mean()), 3),
            "r80_new": round(r_new, 4), "r80_old": round(r_old, 4),
            "r80_diff": round(r_new - r_old, 4), "r80_ci95": q(d_r80),
            "r80_share_positive": round(float((d_r80 > 0).mean()), 3)}


def tabular(res):
    import joblib
    from sklearn.decomposition import PCA
    import models as m
    import tuning as t

    s1 = json.load(open("data/results/tuning_stage1.json"))
    s2 = json.load(open("data/results/tuning_stage2.json"))
    lr_cfg = s1["logreg_best_cfg"]["eng_pca"]
    for k in HORIZONS:
        t0 = time.time()
        Xte, yte = m.load(k, "test")
        d = test_dates(m.DS, k)
        assert (d["y"].astype(int).values == yte).all(), "row order mismatch"
        dates = d["date"].values

        new = joblib.load(f"models/perk_hgb_k{k}.joblib")
        if k == 3:
            old = joblib.load("models/tuned_hgb_k3.joblib")
            Xtr = ytr = None
        else:
            Xtr, ytr = m.load(k, "train")
            old = t.fit_hgb(Xtr, ytr, s1["hgb_best"]["params"])
        p_new, p_old = t.hgb_proba(new, Xte), t.hgb_proba(old, Xte)
        ref = s1["hgb_final_full"]["test_pr_auc"] if k == 3 else s2[str(k)]["hgb_tuned"]["test_pr_auc"]
        assert abs(t.pr_auc(yte, p_old) - ref) < 5e-4, (k, t.pr_auc(yte, p_old), ref)
        res.setdefault("hgb", {})[str(k)] = paired_bootstrap(yte, p_new, p_old, dates)
        print(f"k={k} HGB {res['hgb'][str(k)]} ({time.time()-t0:.0f}s)", flush=True)

        lr = joblib.load(f"models/perk_logreg_k{k}.joblib")
        Z = t.tchunks(lr["pre"], Xte, lr["features"])
        p_new = lr["clf"].predict_proba(lr["pca"].transform(Z) if lr["pca"] is not None else Z)[:, 1]
        if k == 3:
            o = joblib.load("models/tuned_logreg_k3_eng_pca.joblib")
            pre, pca, clf = o["pre"], o.get("pca"), o["clf"]
        else:
            if Xtr is None:
                Xtr, ytr = m.load(k, "train")
            idx = np.random.RandomState(0).choice(len(ytr), t.SUB_N, replace=False)
            pre = t.make_pre(t.numeric_for("eng")).fit(t.prep(Xtr.iloc[idx], "eng"))
            Ztr = t.tchunks(pre, Xtr, "eng")
            pca = PCA(n_components=lr_cfg["n_pca"], random_state=0).fit(Ztr[idx])
            clf = t.fit_lr(pca.transform(Ztr), ytr, lr_cfg["C"])
            del Ztr
        Zt = t.tchunks(pre, Xte, "eng")
        p_old = clf.predict_proba(pca.transform(Zt) if pca is not None else Zt)[:, 1]
        ref = s1["logreg_final_full"]["eng_pca"]["test_pr_auc"] if k == 3 else s2[str(k)]["logreg_eng_pca"]["test_pr_auc"]
        assert abs(t.pr_auc(yte, p_old) - ref) < 5e-4, (k, t.pr_auc(yte, p_old), ref)
        res.setdefault("logreg", {})[str(k)] = paired_bootstrap(yte, p_new, p_old, dates)
        print(f"k={k} LR {res['logreg'][str(k)]} ({time.time()-t0:.0f}s)", flush=True)
        save(res)
        del Xtr, ytr, Xte, Z, Zt


def sequence(res):
    import keras
    import sequence_model as sm
    import tuning_keras as tk

    for k in HORIZONS:
        _, _, (Xte, yte) = tk.load_k(k)
        d = test_dates(sm.SEQ, k)
        assert (d["y"].astype(int).values == yte).all(), "row order mismatch"
        preds = []
        for path in [f"models/perk_sequence_k{k}.keras", f"models/tuned_sequence_k{k}.keras"]:
            preds.append(keras.saving.load_model(path).predict(Xte, batch_size=8192, verbose=0).ravel().astype("float64"))
            keras.backend.clear_session()
        res.setdefault("keras", {})[str(k)] = paired_bootstrap(yte, preds[0], preds[1], d["date"].values)
        print(f"k={k} Keras {res['keras'][str(k)]}", flush=True)
        save(res)


def save(res):
    # re-read just before writing so a concurrent run for the other models is not overwritten
    merged = json.load(open(OUT)) if os.path.exists(OUT) else {}
    merged.update(res)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=1)
    os.replace(tmp, OUT)


def main():
    res = {}
    (sequence if "--keras" in sys.argv else tabular)(res)
    save(res)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
