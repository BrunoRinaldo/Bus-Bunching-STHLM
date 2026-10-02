"""Figure 20 - permutation importance, top 5 and bottom 3 features per horizon.

Reads the aggregates written by src/diagnostics_tuned.py (tuned HGB, 60,000
test rows, drop in PR-AUC, 5 repeats). One panel per horizon (k=3, k=5),
since the ranking differs between them. headway_ratio is ~8x the next
feature, so its bar is cut at the axis edge and labelled with its value;
otherwise the other bars would be unreadable.

Usage: .venv/bin/python src/fig20_permutation_importance.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eda_figures import BLUE, INK, INK_MUTED, INK_SECONDARY, SURFACE

SRC = "reports/diagnostics_tuned.json"
OUT = "figures/fig20_permutation_importance.png"
N_TOP, N_BOTTOM = 5, 3
X_MAX = 0.125


def panel(ax, imp, k):
    top, bottom = imp[:N_TOP], imp[-N_BOTTOM:]
    # rows from the top: 5 best, one empty row, 3 worst
    rows = top + [None] + bottom
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        if r is None:
            continue
        is_top = r in top
        color = BLUE if is_top else INK_MUTED
        width = min(r["mean"], X_MAX)
        ax.barh(yi, width, height=0.62, color=color, zorder=2)
        if r["std"] > 0 and r["mean"] < X_MAX:
            ax.errorbar(r["mean"], yi, xerr=r["std"], color=INK, linewidth=1, capsize=2.5, zorder=3)
        if r["mean"] > X_MAX:
            # cut bar: two surface-coloured slashes, value at the end
            for dx in (-0.006, -0.0035):
                ax.plot([X_MAX + dx - 0.0015, X_MAX + dx + 0.0015], [yi - 0.4, yi + 0.4],
                        color=SURFACE, linewidth=2.2, zorder=4)
            ax.text(X_MAX - 0.009, yi, f"{r['mean']:.3f}", ha="right", va="center",
                    fontsize=9, color=SURFACE, fontweight="bold", zorder=5)
        else:
            x_lab = max(r["mean"], 0) + r["std"] + 0.002
            ax.text(x_lab, yi, (f"{r['mean']:+.4f}" if abs(r["mean"]) >= 5e-5 else "0.0000") if not is_top else f"{r['mean']:.3f}",
                    ha="left", va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks([yi for yi, r in zip(y, rows) if r is not None])
    ax.set_yticklabels([r["feature"] for r in rows if r is not None], fontsize=9, color=INK)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8, zorder=1)
    ax.set_xlim(-0.004, X_MAX)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("drop in PR-AUC when the feature is shuffled", fontsize=8.5)
    ax.set_title(f"k = {k} stops ahead", fontsize=11, color=INK, loc="left")
    ax.text(X_MAX, y[0] + 0.55, f"best {N_TOP}", ha="right", fontsize=8, color=INK_SECONDARY)
    ax.text(X_MAX, y[N_TOP + 1] + 0.55, f"worst {N_BOTTOM} of {len(imp)}", ha="right",
            fontsize=8, color=INK_SECONDARY)


def main():
    res = json.load(open(SRC))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, k in zip(axes, ["3", "5"]):
        panel(ax, res[k]["permutation_importance"], k)
    fig.suptitle("Permutation importance: the current headway dominates, weather adds nothing",
                 fontsize=13, color=INK, x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.915, "Tuned gradient boosting, 60,000 test rows, mean of 5 shuffles (whiskers = ±1 sd). "
             "headway_ratio bar cut at the axis edge.", fontsize=9, color=INK_SECONDARY, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
