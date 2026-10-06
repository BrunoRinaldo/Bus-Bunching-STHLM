"""Per-horizon hyperparameter tuning for HGB and logistic regression.

tuning.py searched at k=3 only and reused the winners at k=1, 2, 5, 8. This
script runs a separate search at every horizon, so each k gets its own
configuration. Same protocol as tuning.py: search on an 800k-row training
subsample, select on the validation split (Oct), refit the winner on the
full training split, and score the test split only for that winner.

HGB, per k, in two rounds:
  round 1 - baseline, the k=3 winner from tuning_stage1.json, and 14 random
            configurations from the same grid as tuning.py
  round 2 - up to 8 neighbours of the round-1 best (one parameter moved one
            grid step), a local refinement around the best region
Logistic regression, per k: the engineered feature set only (tuning.py
showed the base set is clearly worse), C sweep without PCA plus a PCA(n) x C
grid; the single best by validation PR-AUC is refit on full data.

Resumable: every trial is written to data/results/tuning_per_k.json as soon as it
finishes, and on restart finished trials and finished refits are skipped.
Random configurations are seeded per k, so a restart regenerates the same
ones. Models go to models/perk_*.joblib and do not overwrite the k=3-only
tuned models.

Usage: python3 src/tuning_per_k.py [--smoke]
  --smoke  tiny subsample, 2 trials, 2 horizons, separate output files -
           checks the pipeline end to end in a few minutes
"""
import json
import os
import random
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.decomposition import PCA

import models as m
import tuning as t

warnings.filterwarnings("ignore")

SMOKE = "--smoke" in sys.argv
HORIZONS = [1, 2] if SMOKE else [1, 2, 3, 5, 8]
SUB_N = 20_000 if SMOKE else 800_000
N_RANDOM = 1 if SMOKE else 14
N_NEIGHBOURS = 1 if SMOKE else 8
LR_C = [0.1, 1.0] if SMOKE else [0.01, 0.1, 1.0, 10.0]
LR_PCA_N = [10] if SMOKE else [20, 30, 45, 60]
LR_PCA_C = [1.0] if SMOKE else [0.1, 1.0]
OUT = "data/results/tuning_per_k_smoke.json" if SMOKE else "data/results/tuning_per_k.json"
MODEL_PREFIX = "models/perk_smoke_" if SMOKE else "models/perk_"
HGB_MAX_ITER = 800  # early stopping ends most fits well before this

SPACE = dict(t.SPACE)


def load_state():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {"sub_n": SUB_N, "horizons": HORIZONS}


def save_state(state):
    # write to a temp file and rename, so a crash mid-write never corrupts the resume state
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=lambda o: None if o is None else float(o))
    os.replace(tmp, OUT)


def round1_configs(k):
    rnd = random.Random(100 + k)
    configs = [dict(t.BASELINE_HGB, name="baseline")]
    s1_path = "data/results/tuning_stage1.json"
    if os.path.exists(s1_path):
        prev = dict(json.load(open(s1_path))["hgb_best"]["params"])
        prev["max_iter"] = HGB_MAX_ITER
        configs.append(dict(prev, name="k3_winner"))
    for i in range(N_RANDOM):
        c = {p: rnd.choice(v) for p, v in SPACE.items()}
        c["max_iter"] = HGB_MAX_ITER
        c["name"] = f"rs{i}"
        configs.append(c)
    return configs


def neighbours(best_params, tried):
    out = []
    for p, grid in SPACE.items():
        if best_params.get(p) not in grid:
            continue
        i = grid.index(best_params[p])
        for j in (i - 1, i + 1):
            if 0 <= j < len(grid):
                c = dict(best_params)
                c[p] = grid[j]
                key = json.dumps(c, sort_keys=True)
                if key not in tried:
                    tried.add(key)
                    out.append(c)
    return out[:N_NEIGHBOURS]


def run_hgb_trial(S, name, params, Xs, ys, Xval, yval):
    if any(r["name"] == name for r in S["hgb_search"]):
        return
    t1 = time.time()
    clf = t.fit_hgb(Xs, ys, params)
    v = t.pr_auc(yval, t.hgb_proba(clf, Xval))
    S["hgb_search"].append({"name": name, "params": params, "val_pr_auc": round(v, 4),
                            "n_iter": int(clf.n_iter_), "fit_s": round(time.time() - t1, 1)})
    print(f"  HGB {name}: val PR-AUC={v:.4f} n_iter={clf.n_iter_} ({time.time()-t1:.0f}s) {params}", flush=True)


def tune_hgb(state, k, data):
    Xtr, ytr, Xval, yval, Xte, yte, Xs, ys = data
    S = state[f"k{k}"]
    S.setdefault("hgb_search", [])

    for c in round1_configs(k):
        params = {p: v for p, v in c.items() if p != "name"}
        run_hgb_trial(S, c["name"], params, Xs, ys, Xval, yval)
        save_state(state)

    r1 = [r for r in S["hgb_search"] if not r["name"].startswith("nb")]
    best1 = max(r1, key=lambda r: r["val_pr_auc"])
    # neighbours are derived from the round-1 best only, so a restart regenerates the same list
    tried_r1 = {json.dumps(r["params"], sort_keys=True) for r in r1}
    for i, params in enumerate(neighbours(best1["params"], tried_r1)):
        run_hgb_trial(S, f"nb{i}", params, Xs, ys, Xval, yval)
        save_state(state)

    best = max(S["hgb_search"], key=lambda r: r["val_pr_auc"])
    S["hgb_best"] = best
    print(f"  HGB best at k={k}: {best['name']} val={best['val_pr_auc']}", flush=True)

    if "hgb_final" not in S:
        t1 = time.time()
        clf = t.fit_hgb(Xtr, ytr, best["params"])
        pv, pt = t.hgb_proba(clf, Xval), t.hgb_proba(clf, Xte)
        S["hgb_final"] = {"name": best["name"], "params": best["params"], "n_iter": int(clf.n_iter_),
                          "val_pr_auc": round(t.pr_auc(yval, pv), 4),
                          "test_pr_auc": round(t.pr_auc(yte, pt), 4),
                          "test_recall_at_p80": t.recall_at_p80(yte, pt),
                          "fit_s": round(time.time() - t1, 1)}
        joblib.dump(clf, f"{MODEL_PREFIX}hgb_k{k}.joblib")
        print(f"  HGB final k={k}: {S['hgb_final']}", flush=True)
        save_state(state)


def tune_logreg(state, k, data):
    Xtr, ytr, Xval, yval, Xte, yte, Xs, ys = data
    S = state[f"k{k}"]
    rows = S.setdefault("lr_search", [])
    done = {(r["n_pca"], r["C"]) for r in rows}
    fs = "eng"

    grid = [(None, C) for C in LR_C] + [(n, C) for n in LR_PCA_N for C in LR_PCA_C]
    if any(g not in done for g in grid):
        pre = t.make_pre(t.numeric_for(fs)).fit(t.prep(Xs, fs))
        Zs, Zv = t.tchunks(pre, Xs, fs), t.tchunks(pre, Xval, fs)
        nmax = min(max(LR_PCA_N), Zs.shape[1] - 1)
        pca = PCA(n_components=nmax, random_state=0).fit(Zs)
        evr = np.cumsum(pca.explained_variance_ratio_)
        Zs_p, Zv_p = pca.transform(Zs), pca.transform(Zv)
        for n, C in grid:
            if (n, C) in done:
                continue
            t1 = time.time()
            if n is None:
                clf = t.fit_lr(Zs, ys, C)
                v = t.pr_auc(yval, clf.predict_proba(Zv)[:, 1])
                ev = None
            else:
                n_eff = min(n, nmax)
                clf = t.fit_lr(Zs_p[:, :n_eff], ys, C)
                v = t.pr_auc(yval, clf.predict_proba(Zv_p[:, :n_eff])[:, 1])
                ev = round(float(evr[n_eff - 1]), 4)
            rows.append({"features": fs, "n_pca": n, "C": C, "val_pr_auc": round(v, 4),
                         "explained_var": ev, "n_iter": int(clf.n_iter_[0]),
                         "fit_s": round(time.time() - t1, 1)})
            print(f"  LR n_pca={n} C={C}: val={v:.4f} ({time.time()-t1:.0f}s)", flush=True)
            save_state(state)
        del Zs, Zv, Zs_p, Zv_p

    best = max(rows, key=lambda r: r["val_pr_auc"])
    S["lr_best"] = best
    print(f"  LR best at k={k}: n_pca={best['n_pca']} C={best['C']} val={best['val_pr_auc']}", flush=True)

    if "lr_final" not in S:
        t1 = time.time()
        pre = t.make_pre(t.numeric_for(fs)).fit(t.prep(Xs, fs))
        Ztr, Zv, Zt = t.tchunks(pre, Xtr, fs), t.tchunks(pre, Xval, fs), t.tchunks(pre, Xte, fs)
        pca = None
        if best["n_pca"]:
            n_eff = min(best["n_pca"], Ztr.shape[1] - 1)
            sub = np.random.RandomState(0).choice(len(Ztr), min(SUB_N, len(Ztr)), replace=False)
            pca = PCA(n_components=n_eff, random_state=0).fit(Ztr[sub])
            Ztr, Zv, Zt = pca.transform(Ztr), pca.transform(Zv), pca.transform(Zt)
        clf = t.fit_lr(Ztr, ytr, best["C"])
        pv, pt = clf.predict_proba(Zv)[:, 1], clf.predict_proba(Zt)[:, 1]
        S["lr_final"] = {"features": fs, "n_pca": best["n_pca"], "C": best["C"],
                         "val_pr_auc": round(t.pr_auc(yval, pv), 4),
                         "test_pr_auc": round(t.pr_auc(yte, pt), 4),
                         "test_recall_at_p80": t.recall_at_p80(yte, pt),
                         "fit_s": round(time.time() - t1, 1)}
        joblib.dump({"pre": pre, "pca": pca, "clf": clf, "features": fs}, f"{MODEL_PREFIX}logreg_k{k}.joblib")
        print(f"  LR final k={k}: {S['lr_final']}", flush=True)
        save_state(state)


def load_k(k):
    Xtr, ytr = m.load(k, "train")
    Xval, yval = m.load(k, "val")
    Xte, yte = m.load(k, "test")
    if SMOKE:
        # keep the smoke run fast: shrink every split, not just the search subsample
        def shrink(X, y, n, seed):
            i = np.random.RandomState(seed).choice(len(y), min(n, len(y)), replace=False)
            return X.iloc[i].reset_index(drop=True), y[i]
        Xtr, ytr = shrink(Xtr, ytr, 60_000, 1)
        Xval, yval = shrink(Xval, yval, 30_000, 2)
        Xte, yte = shrink(Xte, yte, 30_000, 3)
    idx = np.random.RandomState(0).choice(len(ytr), min(SUB_N, len(ytr)), replace=False)
    Xs, ys = Xtr.iloc[idx].reset_index(drop=True), ytr[idx]
    return Xtr, ytr, Xval, yval, Xte, yte, Xs, ys


def main():
    os.makedirs("models", exist_ok=True)
    state = load_state()
    t_all = time.time()
    for k in HORIZONS:
        S = state.setdefault(f"k{k}", {})
        if "hgb_final" in S and "lr_final" in S:
            print(f"k={k}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        print(f"=== k={k} ===", flush=True)
        data = load_k(k)
        print(f"  loaded in {time.time()-t0:.0f}s; train {len(data[1])} rows, subsample {len(data[7])}", flush=True)
        tune_hgb(state, k, data)
        tune_logreg(state, k, data)
        S["elapsed_s"] = round(S.get("elapsed_s", 0) + time.time() - t0, 1)
        save_state(state)
        print(f"k={k} done in {time.time()-t0:.0f}s", flush=True)
        del data
    print(f"tuning_per_k done in {time.time()-t_all:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
