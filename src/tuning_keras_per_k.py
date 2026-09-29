"""Per-horizon hyperparameter tuning for the Keras sequence model (run under .venv312).

tuning_keras.py searched at k=3 only and reused the winner at the other
horizons. This script searches separately at every k: baseline, the k=3
winner from tuning_keras.json, and 12 random configurations from the same
space, each trained on an 800k-row subsample and selected on the validation
split. The winner is refit on the full training split and scored on val +
test.

Resumable: every trial is written to reports/tuning_keras_per_k.json as soon
as it finishes; finished trials and refits are skipped on restart. Random
configurations and weight initialisation are seeded per k and per trial, so
a restart regenerates the same configurations. Models go to
models/perk_sequence_k{k}.keras.

Usage: python3 src/tuning_keras_per_k.py [--smoke]
"""
import json
import os
import random
import sys
import time

import numpy as np
import keras
from sklearn.metrics import average_precision_score

import sequence_model as sm
import tuning_keras as tk

SMOKE = "--smoke" in sys.argv
HORIZONS = [1, 2] if SMOKE else [1, 2, 3, 5, 8]
SUB_N = 20_000 if SMOKE else 800_000
N_RANDOM = 1 if SMOKE else 12
SEARCH_EPOCHS, SEARCH_PATIENCE = (2, 1) if SMOKE else (10, 2)
FINAL_EPOCHS, FINAL_PATIENCE = (2, 1) if SMOKE else (15, 3)
OUT = "reports/tuning_keras_per_k_smoke.json" if SMOKE else "reports/tuning_keras_per_k.json"
MODEL_PREFIX = "models/perk_smoke_" if SMOKE else "models/perk_"


def load_state():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {"sub_n": SUB_N, "horizons": HORIZONS}


def save_state(state):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=float)
    os.replace(tmp, OUT)


def configs_for(k):
    rnd = random.Random(200 + k)
    configs = [dict(tk.BASELINE, name="baseline")]
    if os.path.exists("reports/tuning_keras.json"):
        configs.append(dict(json.load(open("reports/tuning_keras.json"))["best"]["cfg"], name="k3_winner"))
    for i in range(N_RANDOM):
        c = {p: rnd.choice(v) for p, v in tk.SPACE.items()}
        c["name"] = f"rs{i}"
        configs.append(c)
    return configs


def shrink(X, y, n, seed):
    i = np.random.RandomState(seed).choice(len(y), min(n, len(y)), replace=False)
    return X[i], y[i]


def main():
    os.makedirs("models", exist_ok=True)
    state = load_state()
    t_all = time.time()
    for k in HORIZONS:
        S = state.setdefault(f"k{k}", {})
        if "final" in S:
            print(f"k={k}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        print(f"=== k={k} ===", flush=True)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = tk.load_k(k)
        if SMOKE:
            Xtr, ytr = shrink(Xtr, ytr, 60_000, 1)
            Xva, yva = shrink(Xva, yva, 30_000, 2)
            Xte, yte = shrink(Xte, yte, 30_000, 3)
        Xs, ys = shrink(Xtr, ytr, SUB_N, 0)
        print(f"  loaded in {time.time()-t0:.0f}s; train {len(ytr)} rows, subsample {len(ys)}", flush=True)

        rows = S.setdefault("search", [])
        for j, c in enumerate(configs_for(k)):
            if any(r["name"] == c["name"] for r in rows):
                continue
            cfg = {p: v for p, v in c.items() if p != "name"}
            keras.utils.set_random_seed(1000 * k + j)
            t1 = time.time()
            model = tk.fit(cfg, Xs, ys, Xva, yva, epochs=SEARCH_EPOCHS, patience=SEARCH_PATIENCE)
            v = float(average_precision_score(yva, model.predict(Xva, batch_size=8192, verbose=0).ravel()))
            rows.append({"name": c["name"], "cfg": cfg, "val_pr_auc": round(v, 4), "fit_s": round(time.time() - t1, 1)})
            print(f"  Keras {c['name']}: val PR-AUC={v:.4f} ({time.time()-t1:.0f}s) {cfg}", flush=True)
            save_state(state)
            keras.backend.clear_session()

        best = max(rows, key=lambda r: r["val_pr_auc"])
        S["best"] = best
        print(f"  Keras best at k={k}: {best['name']} val={best['val_pr_auc']}", flush=True)

        keras.utils.set_random_seed(1000 * k + 999)
        t1 = time.time()
        model = tk.fit(best["cfg"], Xtr, ytr, Xva, yva, epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE)
        model.save(f"{MODEL_PREFIX}sequence_k{k}.keras")
        S["final"] = {"name": best["name"], "cfg": best["cfg"],
                      "keras_val": sm.evaluate(model, Xva, yva), "keras_test": sm.evaluate(model, Xte, yte),
                      "fit_s": round(time.time() - t1, 1)}
        S["elapsed_s"] = round(time.time() - t0, 1)
        print(f"  Keras final k={k}: val={S['final']['keras_val']['pr_auc']} "
              f"test={S['final']['keras_test']['pr_auc']}", flush=True)
        save_state(state)
        keras.backend.clear_session()
        del Xtr, ytr, Xva, yva, Xte, yte, Xs, ys
        print(f"k={k} done in {time.time()-t0:.0f}s", flush=True)
    print(f"tuning_keras_per_k done in {time.time()-t_all:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
