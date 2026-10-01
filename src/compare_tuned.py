"""Tuned-vs-original comparison across all 5 horizons (test split, Nov-Dec).

Reads models/phase5_results.json + phase5_keras_results.json (original
models), reports/tuning_stage1.json (k=3), tuning_stage2.json (other
horizons, HGB + logreg) and tuning_keras.json. Writes
figures/fig9_tuned_pr_auc_vs_horizon.png and reports/tuned_results_table.md.
Solid lines = tuned, dashed = original, same colour per model family.
Stars = best gradient boosting at k=5 and k=8 (reports/tuning_long_k.json,
docs/03 §6.6), when that file exists.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "text.color": INK, "axes.edgecolor": BASE,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "grid.color": GRID,
                     "axes.grid": True, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False})

HZ = ["1", "2", "3", "5", "8"]
MIN = {k: round(int(k) * 63 / 60, 2) for k in HZ}

p5 = json.load(open("models/phase5_results.json"))
pk = json.load(open("models/phase5_keras_results.json"))
s1 = json.load(open("reports/tuning_stage1.json"))
s2 = json.load(open("reports/tuning_stage2.json"))
kt = json.load(open("reports/tuning_keras.json"))


def hgb_tuned(k):
    return s1["hgb_final_full"] if k == "3" else s2[k]["hgb_tuned"]


def lr_tuned(k):
    return s1["logreg_final_full"]["eng_pca"] if k == "3" else s2[k]["logreg_eng_pca"]


rows = []
for k in HZ:
    rows.append({
        "k": k, "min": MIN[k],
        "hgb_orig": p5[k]["hgb_test"]["pr_auc"], "hgb_tuned": hgb_tuned(k)["test_pr_auc"],
        "keras_orig": pk[k]["keras_test"]["pr_auc"], "keras_tuned": kt["final"][k]["keras_test"]["pr_auc"],
        "lr_orig": p5[k]["logreg_test"]["pr_auc"], "lr_tuned": lr_tuned(k)["test_pr_auc"],
    })

fig, ax = plt.subplots(figsize=(8.5, 5.2))
x = [int(r["k"]) for r in rows]
for key, col, name in [("hgb", BLUE, "gradient boosting"), ("keras", AQUA, "Keras CNN"), ("lr", ORANGE, "logistic regression")]:
    ax.plot(x, [r[f"{key}_orig"] for r in rows], color=col, linestyle="--", linewidth=1.3, alpha=0.7, marker="o", markersize=3)
    ax.plot(x, [r[f"{key}_tuned"] for r in rows], color=col, linewidth=2.2, marker="o", markersize=5, label=f"{name} (tuned)")
if os.path.exists("reports/tuning_long_k.json"):
    lk = json.load(open("reports/tuning_long_k.json"))
    best = {k[1:]: v["test"]["pr_auc"] for k, v in lk.items() if "test" in v}
    ks = [k for k in HZ if k in best]
    ax.scatter([int(k) for k in ks], [best[k] for k in ks], marker="*", s=240, color=BLUE, edgecolors=SURF,
               linewidths=1.2, zorder=6, label="gradient boosting, best (long-horizon search)")
    for k in ks:
        ax.annotate(f"{best[k]:.2f}", (int(k), best[k]), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color=INK2)
ax.plot([], [], color=MUTED, linestyle="--", label="same model, original settings")
ax.set_xlabel("prediction horizon k (stops ahead)")
ax.set_xticks([int(k) for k in HZ])
ax.set_ylabel("PR-AUC (test split, Nov-Dec)")
ax.set_ylim(0.3, 1.0)
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Tuned vs. original models, PR-AUC by horizon", fontsize=13, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("figures/fig9_tuned_pr_auc_vs_horizon.png", dpi=200)

with open("reports/tuned_results_table.md", "w") as f:
    f.write("# Tuned vs. original, test PR-AUC (Nov-Dec 2024)\n\n")
    f.write("Logistic regression 'tuned' = |headway_ratio|-style features + PCA (45 components at k=3, "
            "same config reused at other horizons) - see reports/tuning_results.md.\n\n")
    f.write("| k | min | HGB orig | HGB tuned | Keras orig | Keras tuned | LogReg orig | LogReg tuned |\n|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['k']} | {r['min']} | {r['hgb_orig']} | **{r['hgb_tuned']}** | {r['keras_orig']} | {r['keras_tuned']} | "
                f"{r['lr_orig']} | {r['lr_tuned']} |\n")
print(open("reports/tuned_results_table.md").read())
