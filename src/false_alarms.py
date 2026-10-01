"""Operating points vs. horizon for the best gradient boosting model per k (test split).

The figure fixes precision (0.9, 0.8, 0.7: one alert in ten, five, about
three is false) and shows the share of bunching events still caught (recall)
at each horizon. The JSON also keeps the reverse view: for fixed recall
(0.5, 0.7, 0.8), the false alarms per test day (61 days, all 8 lines
together) and per correct alarm. Best model per k (docs/03 §6.4, §6.6):
  k=1, 2  models/perk_hgb_k{k}.joblib
  k=3     models/tuned_hgb_k3.joblib
  k=5, 8  models/long_hgb_k{k}.joblib
Output: reports/false_alarms.json, figures/fig18_recall_at_precision_vs_horizon.png.
"""
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

import models as m
from feature_screen import load

HZ = [1, 2, 3, 5, 8]
RECALLS = [0.5, 0.7, 0.8]
PRECISIONS = [0.9, 0.8, 0.7]
MIN = {k: round(k * 63 / 60, 2) for k in HZ}
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab"]  # strictest operating point light -> most permissive dark
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "text.color": INK, "axes.edgecolor": BASE,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "grid.color": GRID,
                     "axes.grid": True, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False})


def best_model(k):
    if k in (5, 8):
        o = joblib.load(f"models/long_hgb_k{k}.joblib")
        return o["clf"], o["features"], f"models/long_hgb_k{k}.joblib"
    path = "models/tuned_hgb_k3.joblib" if k == 3 else f"models/perk_hgb_k{k}.joblib"
    return joblib.load(path), list(m.ALL_COLS), path


def plot(res):
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    x = HZ
    for lvl, col in zip(PRECISIONS, BLUE_RAMP):
        q = str(lvl)
        v = [(res[str(k)]["recall_at_precision"][q] or {"recall": 0.0})["recall"] for k in HZ]
        ax.plot(x, v, color=col, linewidth=2, marker="o", markersize=6, label=f"precision {lvl:.1f}")
    ax.set_xlabel("prediction horizon k (stops ahead)")
    ax.set_xticks(HZ)
    ax.set_ylabel("share of bunching events caught (recall, test split)")
    ax.set_xlim(0.5, 8.6)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Gradient boosting: bunching events caught at a fixed precision, by horizon",
                 fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.905, "best tuned model per horizon; precision 0.8 = one alert in five is false",
             fontsize=9, color=INK2, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig("figures/fig18_recall_at_precision_vs_horizon.png", dpi=200)
    plt.close(fig)


def main():
    if "--plot" in sys.argv:
        plot(json.load(open("reports/false_alarms.json")))
        return
    res = {}
    for k in HZ:
        clf, cols, path = best_model(k)
        te = load(k, "test")
        y = te["y"].to_numpy(int)
        p = clf.predict_proba(m.prep_hgb_frame(te[cols]))[:, 1]
        n_days = int(te["date"].nunique())
        order = np.argsort(-p, kind="stable")
        tp = np.cumsum(y[order])
        fp = np.cumsum(1 - y[order])
        pos = int(y.sum())
        r = {"model": path, "pr_auc": round(float(average_precision_score(y, p)), 4), "n": int(len(y)),
             "positives": pos, "n_days": n_days, "positives_per_day": round(pos / n_days, 1), "at_recall": {}}
        for rec in RECALLS:
            i = int(np.searchsorted(tp, np.ceil(rec * pos)))  # first cut that reaches this recall
            r["at_recall"][str(rec)] = {
                "threshold": round(float(p[order][i]), 4), "true_alarms": int(tp[i]), "false_alarms": int(fp[i]),
                "false_alarms_per_day": round(float(fp[i]) / n_days, 1),
                "false_per_true_alarm": round(float(fp[i]) / float(tp[i]), 3),
                "precision": round(float(tp[i]) / float(i + 1), 4)}
        prec_curve = tp / np.arange(1, len(tp) + 1)
        r["recall_at_precision"] = {}
        for lvl in PRECISIONS:
            ok = np.nonzero(prec_curve >= lvl)[0]
            i = int(ok.max()) if len(ok) else None
            r["recall_at_precision"][str(lvl)] = None if i is None else {
                "recall": round(float(tp[i]) / pos, 4), "threshold": round(float(p[order][i]), 4),
                "false_alarms_per_day": round(float(fp[i]) / n_days, 1)}
        res[str(k)] = r
        print(k, r["pr_auc"], {q: (v["false_alarms_per_day"], v["false_per_true_alarm"]) for q, v in r["at_recall"].items()},
              flush=True)
    with open("reports/false_alarms.json", "w") as f:
        json.dump(res, f, indent=1)

    plot(res)
    print("-> reports/false_alarms.json, figures/fig18_recall_at_precision_vs_horizon.png")


if __name__ == "__main__":
    main()
