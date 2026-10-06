"""Figure 22 - persistence vs. the tuned models at persistence's own precision.

Persistence ("bunched now -> bunched at n+k") is a single operating point
whose precision falls with k (0.89 at k=1 to 0.67 at k=8), so it cannot be
drawn on fig 18, where every line is one fixed precision. Here each model's
threshold is set per k so its precision equals persistence's precision at
that k, and the recalls are compared directly.

  GBM    best tuned model per horizon (as fig 18)
  LR     models/perk_logreg_k{k}.joblib
  Keras  models/perk_sequence_k{k}.keras, scored by src/persistence_comparison_keras.py
A model that never reaches the target precision gets no point at that k.

Output: data/results/persistence_comparison.json, figures/fig22_persistence_vs_gbm.png.
  .venv/bin/python    src/persistence_comparison.py         persistence, GBM, LR
  .venv312/bin/python src/persistence_comparison_keras.py   Keras
  .venv/bin/python    src/persistence_comparison.py --plot  figure
"""
import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np

from style import BLUE, ORANGE, AQUA, INK2, MUTED, SURF
import models as m
import tuning as t
from bootstrap_perk import recall_at_precision
from compare_models import persistence_baseline
from false_alarms import HZ, best_model
from feature_screen import load

OUT_JSON = "data/results/persistence_comparison.json"
OUT_FIG = "figures/fig22_persistence_vs_gbm.png"
MODELS = [("gbm", "gradient boosting, tuned", BLUE),
          ("keras", "Keras sequence model, tuned", AQUA),
          ("lr", "logistic regression, tuned", ORANGE)]


def plot(res):
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    pers = [res[str(k)]["persistence"] for k in HZ]
    for key, label, col in MODELS:
        pts = [(k, res[str(k)][key]["recall"]) for k in HZ if res[str(k)].get(key)]
        missing = [k for k in HZ if not res[str(k)].get(key)]
        if missing:
            label += (" (never reaches persistence's precision)" if not pts
                      else f" (not reached at k = {', '.join(map(str, missing))})")
        xs, ys = zip(*pts) if pts else ([], [])
        ax.plot(xs, ys, color=col, linewidth=2.2, marker="o", markersize=7, label=label, zorder=4)
    for k in HZ:
        g = res[str(k)]["gbm"]["recall"]
        ax.annotate(f"{g:.2f}", (k, g), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK2)
    ax.plot(HZ, [r["recall"] for r in pers], color=MUTED, linewidth=2, linestyle="--", marker="o",
            markersize=8, markerfacecolor=SURF, markeredgecolor=MUTED, markeredgewidth=1.8,
            label="persistence (bunched now)", zorder=5)
    # persistence's precision as a row just above the x-axis, clear of the lines
    ax.text(0.55, 0.095, "persistence precision", fontsize=8, color=MUTED, ha="left", va="center")
    for k, r in zip(HZ, pers):
        ax.text(k, 0.035, f"P={r['precision']:.2f}", ha="center", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("prediction horizon k (stops ahead)")
    ax.set_xticks(HZ)
    ax.set_xlim(0.5, 8.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of bunching events caught (recall, test split)")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    fig.suptitle("Recall at persistence's precision: tuned gradient boosting leads at every horizon",
                 fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.935, "Each model's threshold set per k so its precision equals persistence's (P labels). "
             "Missing point = precision never reached.", fontsize=9, color=INK2, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)


def main():
    if "--plot" in sys.argv:
        plot(json.load(open(OUT_JSON)))
        return
    # keep the Keras entries written by persistence_comparison_keras.py
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    pers = persistence_baseline()
    for k in HZ:
        target = pers[k]["precision"]
        r = res.setdefault(str(k), {})
        r["persistence"] = {"precision": target, "recall": pers[k]["recall"]}

        clf, cols, path = best_model(k)
        te = load(k, "test")
        p = clf.predict_proba(m.prep_hgb_frame(te[cols]))[:, 1]
        hit = recall_at_precision(te["y"].to_numpy(int), p, target)
        r["gbm"] = hit and {"model": path, **hit}

        X, y = m.load(k, "test")
        path = f"models/perk_logreg_k{k}.joblib"
        lr = joblib.load(path)
        Z = t.tchunks(lr["pre"], X, lr["features"])
        p = lr["clf"].predict_proba(lr["pca"].transform(Z) if lr["pca"] is not None else Z)[:, 1]
        hit = recall_at_precision(np.asarray(y), p, target)
        r["lr"] = hit and {"model": path, **hit}
        print(k, r, flush=True)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1)
    plot(res)
    print(f"-> {OUT_JSON}, {OUT_FIG}")


if __name__ == "__main__":
    main()
