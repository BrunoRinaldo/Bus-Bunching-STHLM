"""Hyperparameter tuning for the Keras sequence model (run under .venv312).

Random search at k=3 on an 800k-row training subsample, selected on the
validation split; the winner is refit on the full training split and scored
on val + test, then refit (same config, no re-search) at the other horizons.
Same validation-weather caveat as tuning.py (docs/03 §1.1).

Search space includes the pooling choice on purpose: the original model uses
GlobalAveragePooling1D, which averages over the 6 time steps and so throws
away WHICH step is the current one - the most informative one. 'last' keeps
only the final step's causal-conv output, 'flatten' keeps all steps.
"""
import json
import os
import random
import time

import numpy as np
import keras
from keras import layers
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight

import sequence_model as sm

SUB_N = 800_000
SPACE = dict(arch=["cnn", "cnn", "lstm"], n_conv=[1, 2, 3], filters=[16, 32, 64],
             kernel=[2, 3], pool=["avg", "flatten", "last"], dense=[16, 32, 64],
             dropout=[0.1, 0.2, 0.4], lr=[3e-4, 1e-3, 3e-3], batch=[2048, 4096, 8192])
BASELINE = dict(arch="cnn", n_conv=2, filters=24, kernel=3, pool="avg", dense=16,
                dropout=0.2, lr=1e-3, batch=8192)


def build(cfg):
    inp = keras.Input(shape=(sm.SEQ_LEN, 4))
    if cfg["arch"] == "lstm":
        x = layers.LSTM(cfg["filters"])(inp)
    else:
        x = inp
        for _ in range(cfg["n_conv"]):
            x = layers.Conv1D(cfg["filters"], cfg["kernel"], padding="causal", activation="relu")(x)
        if cfg["pool"] == "avg":
            x = layers.GlobalAveragePooling1D()(x)
        elif cfg["pool"] == "last":
            x = layers.Flatten()(layers.Cropping1D((sm.SEQ_LEN - 1, 0))(x))
        else:
            x = layers.Flatten()(x)
    x = layers.Dense(cfg["dense"], activation="relu")(x)
    x = layers.Dropout(cfg["dropout"])(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(cfg["lr"]), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])
    return model


def fit(cfg, Xtr, ytr, Xva, yva, epochs, patience):
    classes = np.unique(ytr)
    cw = compute_class_weight("balanced", classes=classes, y=ytr)
    model = build(cfg)
    es = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=patience, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=cfg["batch"],
              class_weight={int(c): float(w) for c, w in zip(classes, cw)}, callbacks=[es], verbose=0)
    return model


def load_k(k):
    out = {}
    for split in ["train", "val", "test"]:
        hr, dw, dg, y = sm.load(k, split)
        out[split] = (hr, dw, dg, y)
    Xtr, stats = sm.standardize(*out["train"][:3])
    Xva, _ = sm.standardize(*out["val"][:3], stats)
    Xte, _ = sm.standardize(*out["test"][:3], stats)
    return (Xtr, out["train"][3]), (Xva, out["val"][3]), (Xte, out["test"][3])


def main():
    os.makedirs("models", exist_ok=True)
    res = {"sub_n": SUB_N}
    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_k(3)
    idx = np.random.RandomState(0).choice(len(ytr), SUB_N, replace=False)
    Xs, ys = Xtr[idx], ytr[idx]

    rnd = random.Random(2)
    configs = [dict(BASELINE, name="baseline")]
    for i in range(9):
        c = {k: rnd.choice(v) for k, v in SPACE.items()}
        c["name"] = f"rs{i}"
        configs.append(c)
    rows = []
    for c in configs:
        cfg = {k: v for k, v in c.items() if k != "name"}
        t0 = time.time()
        model = fit(cfg, Xs, ys, Xva, yva, epochs=10, patience=2)
        v = float(average_precision_score(yva, model.predict(Xva, batch_size=8192, verbose=0).ravel()))
        rows.append({"name": c["name"], "cfg": cfg, "val_pr_auc": round(v, 4), "fit_s": round(time.time() - t0, 1)})
        print(f"Keras {c['name']}: val PR-AUC={v:.4f} ({time.time()-t0:.0f}s) {cfg}", flush=True)
        res["search"] = rows
        json.dump(res, open("data/results/tuning_keras.json", "w"), indent=2)
    best = max(rows, key=lambda r: r["val_pr_auc"])
    res["best"] = best
    print("Keras best:", best, flush=True)

    res["final"] = {}
    for k in [3, 1, 2, 5, 8]:
        if k != 3:
            (Xtr, ytr), (Xva, yva), (Xte, yte) = load_k(k)
        t0 = time.time()
        model = fit(best["cfg"], Xtr, ytr, Xva, yva, epochs=15, patience=3)
        model.save(f"models/tuned_sequence_k{k}.keras")
        res["final"][str(k)] = {"keras_val": sm.evaluate(model, Xva, yva), "keras_test": sm.evaluate(model, Xte, yte),
                                "fit_s": round(time.time() - t0, 1)}
        print(f"k={k} tuned Keras: val={res['final'][str(k)]['keras_val']['pr_auc']} "
              f"test={res['final'][str(k)]['keras_test']['pr_auc']}", flush=True)
        json.dump(res, open("data/results/tuning_keras.json", "w"), indent=2, default=float)
    print("keras tuning done", flush=True)


if __name__ == "__main__":
    main()
