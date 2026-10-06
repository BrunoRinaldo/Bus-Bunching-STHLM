"""Figures and table for the per-horizon tuned models (run after perk_curves.py).

Reads data/results/tuning_per_k.json, data/results/tuning_keras_per_k.json and
data/results/perk_curves.json, plus the k=3-only tuning results behind fig 9
(tuning_stage1/2.json, tuning_keras.json) for comparison. Writes:
  fig11_perk_pr_auc_vs_horizon.png   per-k tuned vs k=3-tuned, PR-AUC by horizon
  fig12_perk_recall_at_p80.png       recall at precision 0.80 by horizon
  fig13_perk_tuning_spread.png       validation PR-AUC of every search trial
  fig14_perk_pr_curves.png           test PR curves at k=1, 3, 8
  fig15_perk_calibration.png         reliability curves per model and horizon
  fig16_perk_pr_auc_by_line.png      HGB test PR-AUC per line and horizon
  fig17_perk_bootstrap_diff.png      per-k minus k=3-tuned, 95% day-block bootstrap intervals
  data/results/perk_results_table.md
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from style import BLUE, ORANGE, AQUA, INK, INK2, MUTED, BASE, SURF
from compare_models import persistence_baseline

K_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue, k=1 light -> k=8 dark

HZ = ["1", "2", "3", "5", "8"]
MIN = {k: round(int(k) * 63 / 60, 2) for k in HZ}
MODELS = [("hgb", BLUE, "gradient boosting"), ("keras", AQUA, "Keras sequence model"),
          ("lr", ORANGE, "logistic regression")]

pk = json.load(open("data/results/tuning_per_k.json"))
kk = json.load(open("data/results/tuning_keras_per_k.json"))
cv = json.load(open("data/results/perk_curves.json"))
s1 = json.load(open("data/results/tuning_stage1.json"))
s2 = json.load(open("data/results/tuning_stage2.json"))
kt = json.load(open("data/results/tuning_keras.json"))
CURVE_KEY = {"hgb": "hgb", "keras": "keras", "lr": "logreg"}
bs = json.load(open("data/results/perk_bootstrap.json")) if os.path.exists("data/results/perk_bootstrap.json") else {}
lk = json.load(open("data/results/tuning_long_k.json")) if os.path.exists("data/results/tuning_long_k.json") else {}
LONG = {k[1:]: v["test"] for k, v in lk.items() if "test" in v}  # step-2 search, k=5 and 8 only


def k3_tuned(k):
    """The fig 9 'tuned' numbers: searched at k=3 only, winners reused at other k."""
    hgb = s1["hgb_final_full"] if k == "3" else s2[k]["hgb_tuned"]
    lr = s1["logreg_final_full"]["eng_pca"] if k == "3" else s2[k]["logreg_eng_pca"]
    return {"hgb": hgb["test_pr_auc"], "lr": lr["test_pr_auc"],
            "keras": kt["final"][k]["keras_test"]["pr_auc"]}


rows = []
for k in HZ:
    P, K = pk[f"k{k}"], kk[f"k{k}"]["final"]
    rows.append({
        "k": k, "min": MIN[k], "prev": k3_tuned(k),
        "auc": {"hgb": P["hgb_final"]["test_pr_auc"], "lr": P["lr_final"]["test_pr_auc"],
                "keras": K["keras_test"]["pr_auc"]},
        "r80": {"hgb": P["hgb_final"]["test_recall_at_p80"], "lr": P["lr_final"]["test_recall_at_p80"],
                "keras": K["keras_test"]["recall_at_precision_0.80"]},
        "cfg": {"hgb": P["hgb_final"]["name"], "lr": f"n_pca={P['lr_final']['n_pca']}, C={P['lr_final']['C']}",
                "keras": f"{K['name']} ({K['cfg']['arch']})"},
    })
for r in rows:  # NaN / None both mean precision 0.80 is never reached
    r["r80"] = {m: (None if v is None or v != v else v) for m, v in r["r80"].items()}
x = [int(r["k"]) for r in rows]
persist = persistence_baseline()


def title(fig, text, sub=None):
    fig.suptitle(text, fontsize=13, x=0.02, ha="left")
    if sub:
        fig.text(0.02, 0.905, sub, fontsize=9, color=INK2, ha="left")


def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=200)
    plt.close(fig)


# fig 11: PR-AUC vs horizon, per-k tuned (solid) vs k=3-tuned (dashed)
fig, ax = plt.subplots(figsize=(8.5, 5.4))
for key, col, name in MODELS:
    ax.plot(x, [r["prev"][key] for r in rows], color=col, linestyle="--", linewidth=1.3, alpha=0.7,
            marker="o", markersize=3)
    ax.plot(x, [r["auc"][key] for r in rows], color=col, linewidth=2.2, marker="o", markersize=5,
            label=f"{name} (tuned per horizon)")
    ax.annotate(f"{rows[-1]['auc'][key]:.2f}", (x[-1], rows[-1]["auc"][key]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK2)
if LONG:
    ks = [k for k in HZ if k in LONG]
    ax.scatter([int(k) for k in ks], [LONG[k]["pr_auc"] for k in ks], marker="*", s=220, color=BLUE,
               edgecolors=SURF, linewidths=1.2, zorder=6, label="gradient boosting, long-horizon search (docs/03 §6.6)")
    for k in ks:
        ax.annotate(f"{LONG[k]['pr_auc']:.2f}", (int(k), LONG[k]["pr_auc"]), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color=INK2)
ax.scatter(x, [persist[int(k)]["precision"] for k in HZ], color=MUTED, marker="x", s=50, zorder=5,
           label="persistence baseline (precision, not PR-AUC)")
ax.plot([], [], color=MUTED, linestyle="--", label="tuned at k=3 only (fig 9)")
ax.set_xlabel("prediction horizon k (stops ahead)")
ax.set_xticks([int(k) for k in HZ])
ax.set_ylabel("PR-AUC (test split, Nov-Dec)")
ax.set_ylim(0.3, 1.0)
ax.set_xlim(0.5, 9.0)
ax.legend(frameon=False, fontsize=9, loc="lower left")
title(fig, "Per-horizon tuning, PR-AUC by horizon")
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig11_perk_pr_auc_vs_horizon")

# fig 12: recall at precision 0.80
fig, ax = plt.subplots(figsize=(8.5, 5.2))
w = 0.26
xi = np.arange(len(HZ))
for j, (key, col, name) in enumerate(MODELS):
    vals = [r["r80"][key] for r in rows]
    pos = xi + (j - 1) * w
    ax.bar(pos, [v or 0 for v in vals], width=w - 0.03, color=col, label=name, zorder=3)
    for p, v in zip(pos, vals):
        if v is None:
            ax.text(p, 0.015, "not\nreached", ha="center", va="bottom", fontsize=7, color=MUTED)
        else:
            ax.text(p, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(xi, [f"k={r['k']}" for r in rows])
ax.set_ylabel("recall at precision >= 0.80 (test split)")
ax.set_ylim(0, 1.08)
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, fontsize=9, loc="upper right")
title(fig, "Share of bunching events caught when 4 in 5 alarms are correct",
      "'not reached': the model never reaches precision 0.80 at any threshold")
fig.tight_layout(rect=[0, 0, 1, 0.9])
save(fig, "fig12_perk_recall_at_p80")

# fig 13: search spread - every validation trial, baseline and winner marked
fig, ax = plt.subplots(figsize=(8.5, 5.2))
rng = np.random.RandomState(0)
search = {"hgb": lambda k: pk[f"k{k}"]["hgb_search"], "keras": lambda k: kk[f"k{k}"]["search"],
          "lr": lambda k: pk[f"k{k}"]["lr_search"]}
for j, (key, col, name) in enumerate(MODELS):
    for i, k in enumerate(HZ):
        tr = search[key](k)
        v = np.array([t["val_pr_auc"] for t in tr])
        cx = i + (j - 1) * 0.26
        ax.scatter(cx + rng.uniform(-0.07, 0.07, len(v)), v, s=14, color=col, alpha=0.45, linewidths=0, zorder=3,
                   label=name if i == 0 else None)
        ax.scatter(cx, v.max(), s=70, color=col, edgecolors=SURF, linewidths=1.5, zorder=4)
        base = [t for t in tr if t.get("name") == "baseline"]
        if base:
            ax.scatter(cx, base[0]["val_pr_auc"], s=70, marker="_", color=INK, linewidths=2, zorder=4)
ax.scatter([], [], s=70, color=MUTED, edgecolors=SURF, label="selected configuration")
ax.scatter([], [], s=70, marker="_", color=INK, linewidths=2, label="default settings (baseline)")
ax.set_xticks(range(len(HZ)), [f"k={k}" for k in HZ])
ax.set_ylabel("PR-AUC (validation split, Oct; 800k-row search subsample)")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
title(fig, "Hyperparameter search per horizon: every trial on the validation split")
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "fig13_perk_tuning_spread")

# fig 14: PR curves at k=1, 3, 8
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
for ax, k in zip(axes, ["1", "3", "8"]):
    for key, col, name in MODELS:
        c = cv[CURVE_KEY[key]][k]
        pr = c["pr_curve"]
        ax.plot(pr["recall"], [np.nan if p is None else p for p in pr["precision"]], color=col, linewidth=2,
                label=f"{name} ({c['pr_auc']:.2f})")
    br = cv["hgb"][k]["base_rate"]
    ax.axhline(br, color=MUTED, linewidth=1, linestyle=":")
    ax.text(0.02, br + 0.015, f"base rate {br:.2f}", fontsize=8, color=MUTED)
    ax.axhline(0.8, color=BASE, linewidth=1, linestyle="--")
    ax.set_title(f"k={k} stops ahead", fontsize=10, color=INK2, loc="left")
    ax.set_xlabel("recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 0.07), title="PR-AUC",
              title_fontsize=8)
axes[0].set_ylabel("precision (test split, Nov-Dec)")
title(fig, "Precision-recall curves of the per-horizon tuned models",
      "dashed line: precision 0.80; dotted: share of positives (a random classifier)")
fig.tight_layout(rect=[0, 0, 1, 0.88])
save(fig, "fig14_perk_pr_curves")

# fig 15: reliability, one panel per model, one line per horizon
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
for ax, (key, col, name) in zip(axes, MODELS):
    ax.plot([0, 1], [0, 1], color=BASE, linewidth=1, linestyle="--")
    for k, kc in zip(HZ, K_RAMP):
        rel = cv[CURVE_KEY[key]][k]["reliability"]
        ax.plot([b["mean_pred"] for b in rel], [b["obs_rate"] for b in rel], color=kc, linewidth=2,
                marker="o", markersize=4, label=f"k={k}")
    ax.set_title(name, fontsize=10, color=INK2, loc="left")
    ax.set_xlabel("mean predicted score (10 bins)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
axes[0].set_ylabel("observed bunching rate (test split)")
axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
title(fig, "Calibration: predicted score vs. observed bunching rate",
      "models were trained with balanced class weights, so scores sit above the true rate; "
      "they rank well but are not probabilities without recalibration")
fig.tight_layout(rect=[0, 0, 1, 0.88])
save(fig, "fig15_perk_calibration")

# fig 16: HGB PR-AUC per line x horizon
lines = sorted(cv["hgb"]["1"]["per_line"], key=lambda s: (len(s), s))
M = np.array([[cv["hgb"][k]["per_line"].get(ln, {}).get("pr_auc", np.nan) for k in HZ] for ln in lines])
fig, ax = plt.subplots(figsize=(7, 0.45 * len(lines) + 1.8))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
im = ax.imshow(M, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
for i in range(len(lines)):
    for j in range(len(HZ)):
        v = M[i, j]
        if v == v:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=SURF if v > 0.72 else INK)
ax.set_xticks(range(len(HZ)), [f"k={k}" for k in HZ])
ax.set_yticks(range(len(lines)), [f"line {ln}" for ln in lines])
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.outline.set_visible(False)
cb.set_label("test PR-AUC", color=INK2)
title(fig, "Gradient boosting PR-AUC per line and horizon")
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "fig16_perk_pr_auc_by_line")

# fig 17: paired bootstrap, per-k tuned minus k=3-tuned
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, (metric, ci, lab) in zip(axes, [("diff", "ci95", "PR-AUC"), ("r80_diff", "r80_ci95", "recall at precision 0.80")]):
    ax.axhline(0, color=INK2, linewidth=1)
    for j, (key, col, name) in enumerate(MODELS):
        d = bs.get(CURVE_KEY[key], {})
        ks = [k for k in HZ if k in d]
        if not ks:
            continue
        pos = np.array([HZ.index(k) for k in ks]) + (j - 1) * 0.22
        mid = np.array([d[k][metric] for k in ks])
        lo = np.array([d[k][ci][0] for k in ks])
        hi = np.array([d[k][ci][1] for k in ks])
        ax.errorbar(pos, mid, yerr=[mid - lo, hi - mid], fmt="o", color=col, markersize=6, capsize=3,
                    elinewidth=1.5, label=name, zorder=3)
    ax.set_xticks(range(len(HZ)), [f"k={k}" for k in HZ])
    ax.grid(axis="x", visible=False)
    ax.set_title(lab, fontsize=10, color=INK2, loc="left")
axes[0].set_ylabel("per-horizon tuned minus k=3-tuned (test split)")
axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
title(fig, "Gain from tuning each horizon separately, with 95% bootstrap intervals",
      "paired bootstrap over the 61 test days (B = 1000); above zero = per-horizon tuning is better")
fig.tight_layout(rect=[0, 0, 1, 0.88])
save(fig, "fig17_perk_bootstrap_diff")

# table
with open("data/results/perk_results_table.md", "w") as f:
    f.write("# Per-horizon tuned models, test results (Nov-Dec 2024)\n\n")
    f.write("Each horizon has its own search (src/tuning_per_k.py, src/tuning_keras_per_k.py): "
            "800k-row training subsample, selection on the validation split (Oct), winner refit on "
            "the full training split. 'k=3-tuned' is the fig 9 setup (searched at k=3, reused elsewhere). "
            "R@P80 = recall at precision >= 0.80; '-' = precision 0.80 never reached.\n\n")
    f.write("| k | min | HGB | HGB k=3-tuned | HGB R@P80 | Keras | Keras k=3-tuned | Keras R@P80 | "
            "LogReg | LogReg k=3-tuned | LogReg R@P80 |\n|" + "---|" * 11 + "\n")
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    for r in rows:
        f.write(f"| {r['k']} | {r['min']} | **{r['auc']['hgb']}** | {r['prev']['hgb']} | {fmt(r['r80']['hgb'])} | "
                f"{r['auc']['keras']} | {r['prev']['keras']} | {fmt(r['r80']['keras'])} | "
                f"{r['auc']['lr']} | {r['prev']['lr']} | {fmt(r['r80']['lr'])} |\n")
    f.write("\nSelected configurations:\n\n| k | HGB | Keras | LogReg |\n|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['k']} | {r['cfg']['hgb']} | {r['cfg']['keras']} | {r['cfg']['lr']} |\n")
    if LONG:
        f.write("\nLong-horizon HGB search (src/tuning_long_k.py, docs/03 §6.6), test split, vs. the per-k HGB "
                "above (paired day-block bootstrap):\n\n| k | PR-AUC | diff | 95% CI | R@P80 | diff | 95% CI |\n"
                "|---|---|---|---|---|---|---|\n")
        for k in [k for k in HZ if k in LONG]:
            b = LONG[k]["vs_perk_hgb"]
            f.write(f"| {k} | **{b['pr_auc_new']}** | {b['diff']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | "
                    f"{b['r80_new']:.3f} | {b['r80_diff']:+.4f} | [{b['r80_ci95'][0]:+.4f}, {b['r80_ci95'][1]:+.4f}] |\n")
    if bs:
        f.write("\nPaired day-block bootstrap, per-k tuned minus k=3-tuned (test split, 61 days, B = 1000, "
                "src/bootstrap_perk.py). CI = 2.5-97.5 percentile.\n\n"
                "| model | k | PR-AUC diff | 95% CI | R@P80 diff | 95% CI |\n|---|---|---|---|---|---|\n")
        for key, _, name in MODELS:
            for k in HZ:
                b = bs.get(CURVE_KEY[key], {}).get(k)
                if b:
                    f.write(f"| {name} | {k} | {b['diff']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | "
                            f"{b['r80_diff']:+.4f} | [{b['r80_ci95'][0]:+.4f}, {b['r80_ci95'][1]:+.4f}] |\n")
print(open("data/results/perk_results_table.md").read())
