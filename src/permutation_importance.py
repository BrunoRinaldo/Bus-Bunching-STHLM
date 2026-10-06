"""Permutation importance of the tuned HGB, k=3 and k=5: tables and fig 21.

Reads the aggregates written by src/diagnostics_tuned.py (60,000 test rows,
drop in PR-AUC, 5 repeats). Split in two because the values live on scales
three orders of magnitude apart:

- tables (data/results/permutation_importance.md): the 5 most important
  features and the lowest 3 at either horizon, mean +- sd and overall rank.
- fig 21: the 3 least important per horizon, as dots with +-1 sd whiskers on
  a zoomed axis around zero.

Ranking is by the signed mean: a negative value means shuffling did not hurt.

Usage: .venv/bin/python src/permutation_importance.py
"""
import json

import matplotlib.pyplot as plt

from style import BLUE, ORANGE, INK, INK2, MUTED, GRID, BASE, SURF

SRC = "data/results/diagnostics_tuned.json"
OUT_TABLE = "data/results/permutation_importance.md"
OUT_BOTTOM = "figures/fig21_permutation_importance_bottom3.png"
COLORS = {"3": BLUE, "5": ORANGE}


def load():
    res = json.load(open(SRC))
    return {k: res[k]["permutation_importance"] for k in ["3", "5"]}


def style(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=INK)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)


def tables(imp):
    """Top 5 and lowest 3 as Markdown tables, both horizons side by side.

    The top five are the same at k=3 and k=5. The lowest three differ, so the
    second table lists every feature in the lowest three at either horizon,
    ordered by its mean rank.
    """
    by_k = {k: {r["feature"]: r for r in imp[k]} for k in imp}
    rank = {k: {r["feature"]: i + 1 for i, r in enumerate(imp[k])} for k in imp}
    n = len(imp["3"])

    def rows(feats, dec):
        out = ["| Feature | k = 3 | k = 5 |", "|---|---|---|"]
        for f in feats:
            cells = []
            for k in ["3", "5"]:
                r = by_k[k][f]
                m = 0.0 if abs(r["mean"]) < 0.5 * 10 ** -dec else r["mean"]
                cells.append(f"{m:.{dec}f} ± {r['std']:.{dec}f} ({rank[k][f]})")
            out.append(f"| `{f}` | " + " | ".join(cells) + " |")
        return out

    top = rows([r["feature"] for r in imp["3"][:5]], 3)
    low_feats = {r["feature"] for k in ["3", "5"] for r in imp[k][-3:]}
    low = rows(sorted(low_feats, key=lambda f: rank["3"][f] + rank["5"][f]), 4)
    with open(OUT_TABLE, "w") as fh:
        fh.write("# Permutation importance\n\n"
                 "Tuned gradient boosting, 60,000 test rows, drop in PR-AUC when the feature is shuffled, "
                 f"mean ± sd over 5 shuffles. Rank among all {n} features in brackets.\n\n"
                 "## Five most important\n\n" + "\n".join(top) + "\n\n"
                 "## Lowest three at either horizon\n\n"
                 "Negative = shuffling slightly improved the score, i.e. noise. All are below 0.0013 "
                 "in absolute value, against 0.011 for the fifth most important feature, and their "
                 "ranks jump between horizons (e.g. 23rd to 36th), so the order among them is noise.\n\n"
                 + "\n".join(low) + "\n")
    print(open(OUT_TABLE).read())


def fig_bottom(imp):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharex=True)
    lim = 0.0025
    for ax, k in zip(axes, ["3", "5"]):
        rows = imp[k][-3:]
        ax.axvspan(-lim, 0, color=GRID, alpha=0.3, zorder=0, linewidth=0)
        ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
        for i, r in enumerate(rows):
            y = 2 - i
            ax.errorbar(r["mean"], y, xerr=r["std"], fmt="none", ecolor=COLORS[k],
                        elinewidth=2, capsize=0, alpha=0.5, zorder=2)
            ax.plot(r["mean"], y, "o", markersize=9, color=COLORS[k],
                    markeredgecolor=SURF, markeredgewidth=2, zorder=3)
            v = r["mean"] if abs(r["mean"]) >= 5e-5 else 0.0
            ax.text(r["mean"], y + 0.3, f"{v:+.4f}".replace("+0.0000", "0.0000"), ha="center",
                    va="bottom", fontsize=8.5, color=INK2)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([r["feature"] for r in rows])
        ax.set_ylim(-0.5, 3.2)
        ax.set_xlim(-lim, lim)
        ax.set_title(f"k = {k} stops ahead", fontsize=11, color=INK, loc="left")
        style(ax)
        ax.set_xlabel("drop in PR-AUC", fontsize=9, color=INK2)
        ax.text(-lim * 0.95, 3.15, "← shuffling helped (noise)", fontsize=7.5, color=MUTED,
                ha="left", va="top")
    fig.suptitle("The three least important features: indistinguishable from zero",
                 fontsize=13, color=INK, x=0.02, ha="left", y=0.98)
    fig.text(0.02, 0.885, "Same model and test rows as the top-5 table. Axis zoomed to ±0.0025 "
             "(compare 0.791 for headway_ratio). Whiskers ±1 sd.",
             fontsize=9, color=INK2, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT_BOTTOM, dpi=200)
    plt.close(fig)


def main():
    imp = load()
    tables(imp)
    fig_bottom(imp)
    print(f"-> {OUT_TABLE}\n-> {OUT_BOTTOM}")


if __name__ == "__main__":
    main()
