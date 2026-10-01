"""Step 2: HGB search for the long horizons (k=5, 8) with the lessons from round 2.

Changes against src/tuning_per_k.py (docs/03 §6.4):
  - early stopping on the validation split (Oct) instead of a random 10% of
    training rows, which are near-duplicates of the rows next to them
  - a 2M-row search subsample instead of 800k
  - the search space extended past the edge where the round-2 winners sat
    (learning rate 0.03, 255 leaves), plus max_features
  - the top REFIT_N configurations are refit on the full training split and
    the choice is made on those refits, not on the subsample
  - feature set per horizon = the best set on validation in
    reports/feature_screen.json, if it beats the current features by at
    least MIN_GAIN; otherwise the current features
The test split is scored once, for the chosen refit, and compared with the
current model (models/perk_hgb_k{k}.joblib) by the paired day-block
bootstrap from src/bootstrap_perk.py.

Resumable: every trial and refit is written to reports/tuning_long_k.json.
Models: models/long_hgb_k{k}.joblib.

Usage: .venv/bin/python src/tuning_long_k.py [--smoke]
"""
import json
import os
import random
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

import models as m
import tuning as t
from bootstrap_perk import paired_bootstrap
from extra_features import GROUPS
from feature_screen import load

SMOKE = "--smoke" in sys.argv
HORIZONS = [5, 8]
SUB_N = 30_000 if SMOKE else 2_000_000
N_RANDOM = 1 if SMOKE else 16
REFIT_N = 1 if SMOKE else 3
MIN_GAIN = 0.005  # a feature set must beat the current one by this much on validation
MAX_ITER = 50 if SMOKE else 3000
OUT = "reports/tuning_long_k_smoke.json" if SMOKE else "reports/tuning_long_k.json"
MODEL_PREFIX = "models/long_smoke_" if SMOKE else "models/long_"
SPACE = dict(learning_rate=[0.02, 0.03, 0.05], max_leaf_nodes=[127, 255, 511, 1023],
             min_samples_leaf=[50, 100, 200, 500], l2_regularization=[0.0, 1.0, 10.0],
             max_depth=[8, 10, 12, None], max_features=[0.5, 0.8, 1.0])


def save(state):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, default=float)
    os.replace(tmp, OUT)


def feature_set(k):
    sets = json.load(open("reports/feature_screen.json"))[str(k)]["sets"]
    name = max(sets, key=lambda s: sets[s]["val_pr_auc"])
    if sets[name]["val_pr_auc"] - sets["current"]["val_pr_auc"] < MIN_GAIN:
        name = "current"
    cols = [c for c in m.ALL_COLS if not (name.startswith("no_month") and c == "month_of_year")]
    extra = name.split("+")[1] if "+" in name else None
    if extra == "all":
        cols += [c for g in GROUPS.values() for c in g]
    elif extra:
        cols += GROUPS[extra]
    return name, cols


def fit(cols, X, y, Xv, yv, params):
    clf = HistGradientBoostingClassifier(
        categorical_features=[c in m.CATEGORICAL for c in cols], early_stopping=True,
        n_iter_no_change=20, random_state=0, **params)
    clf.fit(m.prep_hgb_frame(X[cols]), y, sample_weight=compute_sample_weight("balanced", y),
            X_val=m.prep_hgb_frame(Xv[cols]), y_val=yv, sample_weight_val=compute_sample_weight("balanced", yv))
    return clf


def proba(clf, cols, X):
    return clf.predict_proba(m.prep_hgb_frame(X[cols]))[:, 1]


def configs(k, prev):
    rnd = random.Random(300 + k)
    out = [dict(prev, max_features=1.0, name="round2_winner")]
    for i in range(N_RANDOM):
        out.append(dict({p: rnd.choice(v) for p, v in SPACE.items()}, name=f"rs{i}"))
    return out


def main():
    state = json.load(open(OUT)) if os.path.exists(OUT) else {}
    pk = json.load(open("reports/tuning_per_k.json"))
    for k in HORIZONS:
        S = state.setdefault(f"k{k}", {})
        if "test" in S:
            print(f"k={k}: done, skipping", flush=True)
            continue
        t0 = time.time()
        fs_name, cols = feature_set(k)
        S["feature_set"], S["features"] = fs_name, cols
        tr, va, te = load(k, "train"), load(k, "val"), load(k, "test")
        if SMOKE:
            tr, va, te = (d.sample(n, random_state=1).reset_index(drop=True)
                          for d, n in [(tr, 60_000), (va, 30_000), (te, 30_000)])
        ytr, yva, yte = (d["y"].to_numpy(int) for d in (tr, va, te))
        idx = np.random.RandomState(0).choice(len(tr), min(SUB_N, len(tr)), replace=False)
        sub, ysub = tr.iloc[idx].reset_index(drop=True), ytr[idx]
        print(f"=== k={k}: features '{fs_name}' ({len(cols)}), loaded in {time.time()-t0:.0f}s ===", flush=True)

        prev = {p: v for p, v in pk[f"k{k}"]["hgb_final"]["params"].items() if p != "max_iter"}
        rows = S.setdefault("search", [])
        for c in configs(k, prev):
            if any(r["name"] == c["name"] for r in rows):
                continue
            params = dict({p: v for p, v in c.items() if p != "name"}, max_iter=MAX_ITER)
            t1 = time.time()
            clf = fit(cols, sub, ysub, va, yva, params)
            v = float(average_precision_score(yva, proba(clf, cols, va)))
            rows.append({"name": c["name"], "params": params, "val_pr_auc": round(v, 4),
                         "n_iter": int(clf.n_iter_), "fit_s": round(time.time() - t1, 1)})
            print(f"  {c['name']}: val={v:.4f} n_iter={clf.n_iter_} ({time.time()-t1:.0f}s) {params}", flush=True)
            save(state)

        refits = S.setdefault("refits", [])
        for r in sorted(rows, key=lambda r: -r["val_pr_auc"])[:REFIT_N]:
            if any(f["name"] == r["name"] for f in refits):
                continue
            t1 = time.time()
            clf = fit(cols, tr, ytr, va, yva, r["params"])
            v = float(average_precision_score(yva, proba(clf, cols, va)))
            joblib.dump(clf, f"{MODEL_PREFIX}hgb_k{k}_{r['name']}.joblib")
            refits.append({"name": r["name"], "params": r["params"], "val_pr_auc": round(v, 4),
                           "n_iter": int(clf.n_iter_), "fit_s": round(time.time() - t1, 1)})
            print(f"  refit {r['name']}: val={v:.4f} n_iter={clf.n_iter_} ({time.time()-t1:.0f}s)", flush=True)
            save(state)

        best = max(refits, key=lambda r: r["val_pr_auc"])
        clf = joblib.load(f"{MODEL_PREFIX}hgb_k{k}_{best['name']}.joblib")
        joblib.dump({"clf": clf, "features": cols}, f"{MODEL_PREFIX}hgb_k{k}.joblib")
        p_new = proba(clf, cols, te)
        old = joblib.load(f"models/perk_hgb_k{k}.joblib")
        p_old = t.hgb_proba(old, te[m.ALL_COLS])
        S["chosen"] = best["name"]
        S["test"] = {"pr_auc": round(float(average_precision_score(yte, p_new)), 4),
                     "recall_at_p80": t.recall_at_p80(yte, p_new),
                     "vs_perk_hgb": paired_bootstrap(yte, p_new, p_old, te["date"].to_numpy())}
        S["elapsed_s"] = round(time.time() - t0, 1)
        save(state)
        b = S["test"]["vs_perk_hgb"]
        print(f"  k={k} chosen {best['name']}: test {b['pr_auc_old']} -> {b['pr_auc_new']} diff {b['diff']} "
              f"CI {b['ci95']}; R@P80 {b['r80_old']} -> {b['r80_new']} CI {b['r80_ci95']}", flush=True)
        del tr, va, te, sub
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
