"""Phase 5 headline output: PR-AUC vs horizon, all models + persistence
baseline, k converted to minutes via the median inter-stop running time
(63s, computed from headways.parquet). Reads data/results/phase5_results.json
(logreg + hgb, from src/models.py) and data/results/phase5_keras_results.json
(from src/sequence_model.py, run under .venv312). The persistence baseline
numbers are hardcoded from the real run reported in
docs/03_training_split_and_model_choice.md - recomputed here from the
dataset directly instead, so this script has no silent dependency on a
number copied by hand into a doc.
"""
import json
import duckdb
import matplotlib.pyplot as plt

from style import BLUE, ORANGE, AQUA, INK, MUTED


HORIZONS = [1, 2, 3, 5, 8]
MEDIAN_RUN_TIME_S = 63.0


def persistence_baseline():
    con = duckdb.connect()
    out = {}
    for k in HORIZONS:
        r = con.sql(f"""
            SELECT
                avg(y_k{k}) AS base_rate,
                sum(CASE WHEN is_bunched_now AND y_k{k}=1 THEN 1 ELSE 0 END) tp,
                sum(CASE WHEN is_bunched_now AND y_k{k}=0 THEN 1 ELSE 0 END) fp,
                sum(CASE WHEN NOT is_bunched_now AND y_k{k}=1 THEN 1 ELSE 0 END) fn,
                sum(CASE WHEN NOT is_bunched_now AND y_k{k}=0 THEN 1 ELSE 0 END) tn
            FROM 'data/processed/model_dataset.parquet'
            WHERE split='test' AND y_k{k} IS NOT NULL AND is_bunched_now IS NOT NULL
        """).fetchone()
        base_rate, tp, fp, fn, tn = r
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        # PR-AUC is not defined for a single-point (binary) classifier; report
        # the operating point instead and use precision as the PR-AUC proxy
        # is misleading - the headline figure marks this model as a point,
        # not a curve, and the table carries precision/recall directly.
        out[k] = {"base_rate_pct": round(100 * base_rate, 3),
                   "precision": round(precision, 4), "recall": round(recall, 4)}
    return out


def main():
    with open("data/results/phase5_results.json") as f:
        tab = json.load(f)
    with open("data/results/phase5_keras_results.json") as f:
        seq = json.load(f)
    persist = persistence_baseline()

    rows = []
    for k in HORIZONS:
        ks = str(k)
        rows.append({
            "k": k, "minutes": round(k * MEDIAN_RUN_TIME_S / 60, 2),
            "persistence_precision": persist[k]["precision"],
            "persistence_recall": persist[k]["recall"],
            "logreg_pr_auc": tab[ks]["logreg_test"]["pr_auc"],
            "hgb_pr_auc": tab[ks]["hgb_test"]["pr_auc"],
            "keras_pr_auc": seq[ks]["keras_test"]["pr_auc"],
            "logreg_recall_at_p80": tab[ks]["logreg_test"]["recall_at_precision_0.80"],
            "hgb_recall_at_p80": tab[ks]["hgb_test"]["recall_at_precision_0.80"],
            "keras_recall_at_p80": seq[ks]["keras_test"]["recall_at_precision_0.80"],
        })

    fig, ax = plt.subplots(figsize=(8, 5))
    minutes = HORIZONS  # x-axis in stops ahead
    ax.plot(minutes, [r["logreg_pr_auc"] for r in rows], color=ORANGE, marker="o", label="logistic regression", linewidth=2)
    ax.plot(minutes, [r["hgb_pr_auc"] for r in rows], color=BLUE, marker="o", label="gradient boosting", linewidth=2)
    ax.plot(minutes, [r["keras_pr_auc"] for r in rows], color=AQUA, marker="o", label="Keras 1D-CNN", linewidth=2)
    ax.scatter(minutes, [r["persistence_precision"] for r in rows], color=MUTED, marker="x", s=60,
               label="persistence baseline (precision, not PR-AUC)", zorder=5)
    ax.set_xlabel("prediction horizon k (stops ahead)")
    ax.set_xticks(HORIZONS)
    ax.set_ylabel("PR-AUC (test split, Nov-Dec)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Bunching prediction: PR-AUC vs. horizon, all models", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/fig8_pr_auc_vs_horizon.png", dpi=200)
    plt.close(fig)

    with open("data/results/results_tables.md", "w") as f:
        f.write("# Phase 5 results table\n\n")
        f.write("Test split (Nov-Dec 2024), all 8 lines pooled. k in stops, minutes via median\n")
        f.write("inter-stop running time (63s). Persistence baseline has no probability\n")
        f.write("score so PR-AUC is not defined for it; its precision/recall operating\n")
        f.write("point is reported instead and marked separately on the figure.\n\n")
        f.write("| k | minutes | persistence P/R | logreg PR-AUC | HGB PR-AUC | Keras PR-AUC | logreg R@P80 | HGB R@P80 | Keras R@P80 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        def fmt(v):
            return "n/a" if v != v else v  # v != v is the NaN check

        for r in rows:
            f.write(f"| {r['k']} | {r['minutes']} | {r['persistence_precision']:.2f}/{r['persistence_recall']:.2f} | "
                     f"{r['logreg_pr_auc']} | {r['hgb_pr_auc']} | {r['keras_pr_auc']} | "
                     f"{fmt(r['logreg_recall_at_p80'])} | {fmt(r['hgb_recall_at_p80'])} | {fmt(r['keras_recall_at_p80'])} |\n")

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
