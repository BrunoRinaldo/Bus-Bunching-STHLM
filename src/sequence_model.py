"""Phase 5 - Keras sequence model (task F .keras export).

Run under the Python 3.12 venv (.venv312) because TensorFlow/Keras have no
wheels yet for the Python 3.14 interpreter this project otherwise uses - see
docs/03_training_split_and_model_choice.md for that environment note.

1D-CNN over the 6-stop window (n-5..n) of (headway_ratio, dwell,
delay_growth), plus a 4th channel flagging which timesteps had a missing
observation (all three raw signals share the same missingness pattern,
since they all depend on the same observed_arrival/departure match).
Trained with class_weight (never resampling), same as the tabular models,
and evaluated with the same metrics for a fair comparison.
"""
import json
import time
import os
import duckdb
import numpy as np
import keras
from keras import layers
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, confusion_matrix, brier_score_loss
from sklearn.utils.class_weight import compute_class_weight

SEQ = "data/processed/sequence_dataset.parquet"
HORIZONS = [1, 2, 3, 5, 8]
SEQ_LEN = 6

con = duckdb.connect()
con.sql("PRAGMA threads=8")


def load(k, split):
    cols = ", ".join([f"hr_{i}, dw_{i}, dg_{i}" for i in range(SEQ_LEN)])
    q = f"SELECT {cols}, y_k{k} AS y FROM '{SEQ}' WHERE split='{split}' AND y_k{k} IS NOT NULL"
    df = con.sql(q).df()
    hr = df[[f"hr_{i}" for i in range(SEQ_LEN)]].to_numpy(dtype="float64", na_value=np.nan).astype("float32")
    dw = df[[f"dw_{i}" for i in range(SEQ_LEN)]].to_numpy(dtype="float64", na_value=np.nan).astype("float32")
    dg = df[[f"dg_{i}" for i in range(SEQ_LEN)]].to_numpy(dtype="float64", na_value=np.nan).astype("float32")
    y = df["y"].to_numpy(dtype="int32")
    return hr, dw, dg, y


def standardize(hr, dw, dg, stats=None):
    mask = (~np.isnan(hr)).astype("float32")
    if stats is None:
        stats = {
            "hr_mean": np.nanmean(hr), "hr_std": np.nanstd(hr) + 1e-6,
            "dw_mean": np.nanmean(dw), "dw_std": np.nanstd(dw) + 1e-6,
            "dg_mean": np.nanmean(dg), "dg_std": np.nanstd(dg) + 1e-6,
        }
    hr_n = np.nan_to_num((hr - stats["hr_mean"]) / stats["hr_std"], nan=0.0)
    dw_n = np.nan_to_num((dw - stats["dw_mean"]) / stats["dw_std"], nan=0.0)
    dg_n = np.nan_to_num((dg - stats["dg_mean"]) / stats["dg_std"], nan=0.0)
    X = np.stack([hr_n, dw_n, dg_n, mask], axis=-1)  # (N, 6, 4)
    return X, stats


def build_model():
    inp = keras.Input(shape=(SEQ_LEN, 4))
    x = layers.Conv1D(24, 3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(24, 3, padding="causal", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])
    return model


def evaluate(model, X, y):
    proba = model.predict(X, batch_size=8192, verbose=0).ravel()
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    precisions, recalls, _ = precision_recall_curve(y, proba)
    mask = precisions[:-1] >= 0.80
    recall_at_p80 = recalls[:-1][mask].max() if mask.any() else float("nan")
    f1 = f1_score(y, (proba >= 0.5).astype(int))
    cm = confusion_matrix(y, (proba >= 0.5).astype(int)).tolist()
    return {
        "n": len(y), "base_rate_pct": round(100 * y.mean(), 3),
        "pr_auc": round(float(pr_auc), 4), "brier": round(float(brier), 5),
        "f1_at_0.5": round(float(f1), 4),
        "recall_at_precision_0.80": round(float(recall_at_p80), 4),
        "confusion_matrix_at_0.5": cm,
    }


def main():
    os.makedirs("models", exist_ok=True)
    results = {}
    for k in HORIZONS:
        t0 = time.time()
        hr_tr, dw_tr, dg_tr, y_tr = load(k, "train")
        hr_va, dw_va, dg_va, y_va = load(k, "val")
        hr_te, dw_te, dg_te, y_te = load(k, "test")
        Xtr, stats = standardize(hr_tr, dw_tr, dg_tr)
        Xva, _ = standardize(hr_va, dw_va, dg_va, stats)
        Xte, _ = standardize(hr_te, dw_te, dg_te, stats)
        print(f"k={k} loaded {len(y_tr)}/{len(y_va)}/{len(y_te)} in {time.time()-t0:.1f}s")

        classes = np.unique(y_tr)
        cw = compute_class_weight("balanced", classes=classes, y=y_tr)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

        model = build_model()
        t0 = time.time()
        es = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=2, restore_best_weights=True)
        model.fit(Xtr, y_tr, validation_data=(Xva, y_va), epochs=12, batch_size=8192,
                  class_weight=class_weight, callbacks=[es], verbose=2)
        print(f"k={k} keras fit in {time.time()-t0:.1f}s")

        path = f"models/sequence_k{k}.keras"
        model.save(path)

        results[k] = {
            "keras_val": evaluate(model, Xva, y_va),
            "keras_test": evaluate(model, Xte, y_te),
        }
        print(f"k={k} keras test PR-AUC={results[k]['keras_test']['pr_auc']}")

    with open("data/results/phase5_keras_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
