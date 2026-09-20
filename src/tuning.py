"""Hyperparameter tuning (HGB, logistic regression) and PCA experiments.

Selection is done on the VALIDATION split only (Oct); the test split is
scored only for the final chosen configs. Caveat carried from docs/03 §1.1:
validation has no cold-weather rows, so hyperparameters were not selected
against cold conditions - the test set (Nov-Dec) is the check on that.

Stage 1 (k=3): search on an 800k-row training subsample, refit winners on
the full training split, report val + test. Stage 2: refit the winning
configs at the other horizons (no re-search).

Logistic-regression arms:
  base_C      - original features, tuned C
  base_pca    - original features -> PCA(n) -> logreg (n and C tuned)
  eng_C       - original + |headway_ratio|-style features, tuned C
  eng_pca     - engineered features -> PCA(n) -> logreg
PCA is linear, so it cannot add modelling capacity to a linear classifier;
the engineered arms test the more likely bottleneck (bunching depends on
|headway_ratio| being small, which no single linear coefficient on the
signed value can express).
"""
import json
import random
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import models as m

warnings.filterwarnings("ignore")

K = 3
SUB_N = 800_000
STAGE = sys.argv[1] if len(sys.argv) > 1 else "stage1"

EXTRA = ["abs_headway_ratio", "abs_headway_ratio_lag1", "abs_headway_ratio_lag2",
         "abs_headway_ratio_lag3", "near0_headway_ratio", "near0_headway_ratio_lag1",
         "near0_headway_ratio_lag2", "near0_headway_ratio_lag3", "abs_closing_speed_3"]


def numeric_for(fs):
    return m.NUMERIC if fs == "base" else m.NUMERIC + EXTRA


def add_eng(X):
    X = X.copy()
    for c in ["headway_ratio", "headway_ratio_lag1", "headway_ratio_lag2", "headway_ratio_lag3"]:
        X[f"abs_{c}"] = X[c].abs()
        X[f"near0_{c}"] = np.where(X[c].isna(), np.nan, (X[c].abs() < 0.25).astype(float))
    X["abs_closing_speed_3"] = X["closing_speed_3"].abs()
    return X


def prep(X, fs):
    return add_eng(X) if fs == "eng" else X


def make_pre(numeric):
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler())])
    return ColumnTransformer([("num", num_pipe, numeric + m.BOOLEAN),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), m.CATEGORICAL)],
                             sparse_threshold=0.0)


def tchunks(pre, X, fs="base", chunk=500_000):
    first = pre.transform(prep(X.iloc[:chunk], fs)).astype(np.float32)
    out = np.empty((len(X), first.shape[1]), dtype=np.float32)
    out[:len(first)] = first
    for i in range(chunk, len(X), chunk):
        block = pre.transform(prep(X.iloc[i:i + chunk], fs)).astype(np.float32)
        out[i:i + len(block)] = block
    return out


def pr_auc(y, p):
    return float(average_precision_score(y, p))


def recall_at_p80(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    mask = pr[:-1] >= 0.80
    return float(rc[:-1][mask].max()) if mask.any() else None


def fit_lr(Z, y, C):
    clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs")
    clf.fit(Z, y, sample_weight=compute_sample_weight("balanced", y))
    return clf


def fit_hgb(X, y, params):
    cols = list(X.columns)
    clf = HistGradientBoostingClassifier(
        categorical_features=[c in m.CATEGORICAL for c in cols],
        early_stopping=True, n_iter_no_change=15, validation_fraction=0.1,
        random_state=0, **params)
    clf.fit(m.prep_hgb_frame(X), y, sample_weight=compute_sample_weight("balanced", y))
    return clf


def hgb_proba(clf, X):
    return clf.predict_proba(m.prep_hgb_frame(X))[:, 1]


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: None if o is None else float(o))


BASELINE_HGB = dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
                    l2_regularization=0.0, max_depth=None, max_iter=200)
SPACE = dict(learning_rate=[0.03, 0.05, 0.1, 0.2], max_leaf_nodes=[15, 31, 63, 127, 255],
             min_samples_leaf=[20, 50, 100, 300, 1000], l2_regularization=[0.0, 0.1, 1.0, 10.0],
             max_depth=[None, 6, 10])


def stage1():
    out = {"k": K, "sub_n": SUB_N}
    t0 = time.time()
    Xtr, ytr = m.load(K, "train")
    Xval, yval = m.load(K, "val")
    Xte, yte = m.load(K, "test")
    idx = np.random.RandomState(0).choice(len(ytr), SUB_N, replace=False)
    Xs, ys = Xtr.iloc[idx].reset_index(drop=True), ytr[idx]
    print(f"loaded in {time.time()-t0:.0f}s; subsample {len(ys)} rows, base rate {ys.mean():.4f}", flush=True)

    # ---------------- HGB random search ----------------
    rnd = random.Random(1)
    configs = [dict(BASELINE_HGB, name="baseline")]
    for i in range(12):
        c = {k: rnd.choice(v) for k, v in SPACE.items()}
        c["max_iter"] = 500
        c["name"] = f"rs{i}"
        configs.append(c)
    hgb_rows = []
    for c in configs:
        params = {k: v for k, v in c.items() if k != "name"}
        t1 = time.time()
        clf = fit_hgb(Xs, ys, params)
        v = pr_auc(yval, hgb_proba(clf, Xval))
        hgb_rows.append({"name": c["name"], "params": params, "val_pr_auc": round(v, 4),
                         "n_iter": int(clf.n_iter_), "fit_s": round(time.time() - t1, 1)})
        print(f"HGB {c['name']}: val PR-AUC={v:.4f} n_iter={clf.n_iter_} ({time.time()-t1:.0f}s) {params}", flush=True)
        out["hgb_search"] = hgb_rows
        dump(out, "reports/tuning_stage1.json")
    best = max(hgb_rows, key=lambda r: r["val_pr_auc"])
    out["hgb_best"] = best
    print("HGB best:", best, flush=True)

    t1 = time.time()
    clf = fit_hgb(Xtr, ytr, best["params"])
    pv, pt = hgb_proba(clf, Xval), hgb_proba(clf, Xte)
    out["hgb_final_full"] = {"params": best["params"], "val_pr_auc": round(pr_auc(yval, pv), 4),
                             "test_pr_auc": round(pr_auc(yte, pt), 4),
                             "test_recall_at_p80": recall_at_p80(yte, pt), "fit_s": round(time.time() - t1, 1)}
    joblib.dump(clf, f"models/tuned_hgb_k{K}.joblib")
    print("HGB tuned, full data:", out["hgb_final_full"], flush=True)
    dump(out, "reports/tuning_stage1.json")

    # ---------------- logistic regression: C sweep + PCA sweep ----------------
    lr_rows = []
    best_cfg = {}
    for fs in ["base", "eng"]:
        numeric = numeric_for(fs)
        pre = make_pre(numeric).fit(prep(Xs, fs))
        Zs, Zv = tchunks(pre, Xs, fs), tchunks(pre, Xval, fs)
        dim = Zs.shape[1]
        print(f"[{fs}] encoded dim = {dim}", flush=True)
        # no-PCA C sweep
        for C in [0.01, 0.1, 1.0, 10.0]:
            t1 = time.time()
            clf = fit_lr(Zs, ys, C)
            v = pr_auc(yval, clf.predict_proba(Zv)[:, 1])
            lr_rows.append({"features": fs, "n_pca": None, "C": C, "val_pr_auc": round(v, 4),
                            "n_iter": int(clf.n_iter_[0])})
            print(f"LR {fs} C={C} no-PCA: val={v:.4f} n_iter={clf.n_iter_[0]} ({time.time()-t1:.0f}s)", flush=True)
        # PCA sweep
        nmax = min(60, dim - 1)
        pca = PCA(n_components=nmax, random_state=0).fit(Zs)
        evr = np.cumsum(pca.explained_variance_ratio_)
        Zs_p, Zv_p = pca.transform(Zs), pca.transform(Zv)
        for n in [5, 10, 20, 30, 45, nmax]:
            for C in [0.1, 1.0]:
                clf = fit_lr(Zs_p[:, :n], ys, C)
                v = pr_auc(yval, clf.predict_proba(Zv_p[:, :n])[:, 1])
                lr_rows.append({"features": fs, "n_pca": n, "C": C, "val_pr_auc": round(v, 4),
                                "explained_var": round(float(evr[n - 1]), 4), "n_iter": int(clf.n_iter_[0])})
                print(f"LR {fs} PCA n={n} C={C}: val={v:.4f} (explained var {evr[n-1]:.3f})", flush=True)
        out["logreg_search"] = lr_rows
        dump(out, "reports/tuning_stage1.json")
        for arm, rows in [(f"{fs}_C", [r for r in lr_rows if r["features"] == fs and r["n_pca"] is None]),
                          (f"{fs}_pca", [r for r in lr_rows if r["features"] == fs and r["n_pca"] is not None])]:
            best_cfg[arm] = max(rows, key=lambda r: r["val_pr_auc"])
        del Zs, Zv, Zs_p, Zv_p
    out["logreg_best_cfg"] = best_cfg
    dump(out, "reports/tuning_stage1.json")
    print("LR best per arm:", best_cfg, flush=True)

    # ---------------- logistic regression: refit arms on the FULL train split ----------------
    out["logreg_final_full"] = {}
    for fs in ["base", "eng"]:
        numeric = numeric_for(fs)
        pre = make_pre(numeric).fit(prep(Xs, fs))
        Ztr = tchunks(pre, Xtr, fs)
        Zv, Zt = tchunks(pre, Xval, fs), tchunks(pre, Xte, fs)
        for arm in [f"{fs}_C", f"{fs}_pca"]:
            cfg = best_cfg[arm]
            t1 = time.time()
            if cfg["n_pca"]:
                pca = PCA(n_components=cfg["n_pca"], random_state=0).fit(Ztr[np.random.RandomState(0).choice(len(Ztr), SUB_N, replace=False)])
                A, B, Cc = pca.transform(Ztr), pca.transform(Zv), pca.transform(Zt)
            else:
                pca = None
                A, B, Cc = Ztr, Zv, Zt
            clf = fit_lr(A, ytr, cfg["C"])
            pv, pt = clf.predict_proba(B)[:, 1], clf.predict_proba(Cc)[:, 1]
            out["logreg_final_full"][arm] = {"n_pca": cfg["n_pca"], "C": cfg["C"],
                                             "val_pr_auc": round(pr_auc(yval, pv), 4),
                                             "test_pr_auc": round(pr_auc(yte, pt), 4),
                                             "test_recall_at_p80": recall_at_p80(yte, pt),
                                             "fit_s": round(time.time() - t1, 1)}
            print(f"LR final {arm}: {out['logreg_final_full'][arm]}", flush=True)
            joblib.dump({"pre": pre, "pca": pca, "clf": clf, "features": fs}, f"models/tuned_logreg_k{K}_{arm}.joblib")
            dump(out, "reports/tuning_stage1.json")
        del Ztr, Zv, Zt

    with open("models/phase5_results.json") as f:
        p5 = json.load(f)
    out["baseline_reference"] = {"hgb_val": p5[str(K)]["hgb_val"]["pr_auc"], "hgb_test": p5[str(K)]["hgb_test"]["pr_auc"],
                                 "logreg_val": p5[str(K)]["logreg_val"]["pr_auc"], "logreg_test": p5[str(K)]["logreg_test"]["pr_auc"]}
    dump(out, "reports/tuning_stage1.json")
    print("stage1 done", flush=True)


def stage2():
    s1 = json.load(open("reports/tuning_stage1.json"))
    res = {}
    hgb_params = s1["hgb_best"]["params"]
    lr_final = s1["logreg_final_full"]
    best_lr_arm = max(lr_final, key=lambda a: lr_final[a]["val_pr_auc"])
    arms = [best_lr_arm] + ([a for a in ["base_pca"] if a != best_lr_arm])
    cfgs = s1["logreg_best_cfg"]
    for k in [1, 2, 5, 8]:
        t0 = time.time()
        Xtr, ytr = m.load(k, "train")
        Xval, yval = m.load(k, "val")
        Xte, yte = m.load(k, "test")
        idx = np.random.RandomState(0).choice(len(ytr), SUB_N, replace=False)
        res[str(k)] = {}
        clf = fit_hgb(Xtr, ytr, hgb_params)
        pv, pt = hgb_proba(clf, Xval), hgb_proba(clf, Xte)
        res[str(k)]["hgb_tuned"] = {"val_pr_auc": round(pr_auc(yval, pv), 4), "test_pr_auc": round(pr_auc(yte, pt), 4),
                                    "test_recall_at_p80": recall_at_p80(yte, pt)}
        print(f"k={k} HGB tuned: {res[str(k)]['hgb_tuned']}", flush=True)
        for arm in arms:
            fs = arm.split("_")[0]
            cfg = cfgs[arm]
            pre = make_pre(numeric_for(fs)).fit(prep(Xtr.iloc[idx], fs))
            Ztr = tchunks(pre, Xtr, fs)
            Zv, Zt = tchunks(pre, Xval, fs), tchunks(pre, Xte, fs)
            if cfg["n_pca"]:
                pca = PCA(n_components=cfg["n_pca"], random_state=0).fit(Ztr[idx])
                Ztr, Zv, Zt = pca.transform(Ztr), pca.transform(Zv), pca.transform(Zt)
            lr = fit_lr(Ztr, ytr, cfg["C"])
            pv, pt = lr.predict_proba(Zv)[:, 1], lr.predict_proba(Zt)[:, 1]
            res[str(k)][f"logreg_{arm}"] = {"val_pr_auc": round(pr_auc(yval, pv), 4), "test_pr_auc": round(pr_auc(yte, pt), 4),
                                            "test_recall_at_p80": recall_at_p80(yte, pt)}
            print(f"k={k} LR {arm}: {res[str(k)][f'logreg_{arm}']}", flush=True)
        print(f"k={k} done in {time.time()-t0:.0f}s", flush=True)
        dump(res, "reports/tuning_stage2.json")
    print("stage2 done", flush=True)


if __name__ == "__main__":
    {"stage1": stage1, "stage2": stage2}[STAGE]()
