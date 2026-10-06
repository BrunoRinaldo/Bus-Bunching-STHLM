"""Keras step for figure 22 (run under .venv312, which has Keras but no matplotlib).

Scores the per-horizon tuned sequence models (models/perk_sequence_k{k}.keras)
on the test split and adds their recall at persistence's precision to
data/results/persistence_comparison.json under "keras". Needs the
persistence precision, so run src/persistence_comparison.py first.

Usage: .venv312/bin/python src/persistence_comparison_keras.py
"""
import json

import keras
import numpy as np

import tuning_keras as tk
from bootstrap_perk import recall_at_precision

OUT_JSON = "data/results/persistence_comparison.json"
HZ = [1, 2, 3, 5, 8]


def main():
    res = json.load(open(OUT_JSON))
    for k in HZ:
        _, _, (Xte, yte) = tk.load_k(k)
        path = f"models/perk_sequence_k{k}.keras"
        p = keras.saving.load_model(path).predict(Xte, batch_size=8192, verbose=0).ravel().astype("float64")
        keras.backend.clear_session()
        hit = recall_at_precision(np.asarray(yte).astype(int), p, res[str(k)]["persistence"]["precision"])
        res[str(k)]["keras"] = hit and {"model": path, **hit}
        print(k, res[str(k)]["keras"], flush=True)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1)
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
