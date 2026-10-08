"""Share of bunching chains the best model warns about before the bus is bunched (test split).

A chain is a run of consecutive bunched stops (is_bunched_now) on one trip; its
onset is the first bunched stop after a stop that was observed and not bunched.
Chains that start at a trip's first observed stop have no unbunched stop before
them and are left out. Chain onsets are the hard events: persistence ("bunched
now -> bunched later") can never flag one, while most positives in the overall
recall are stops of chains that are already running.

Horizon k catches an onset at stop j if the best model for k (false_alarms.best_model)
scores the stop j-k of the same trip at or above its threshold for precision
0.9 / 0.8 / 0.7 (data/results/false_alarms.json, thresholds set on all test
positives). The earliest warning of an onset is the largest such k.
Output: data/results/chain_onsets.json, figures/fig24_chain_onsets_caught.png.
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import INK2, MUTED
import models as m
import feature_screen as fs
from false_alarms import best_model, HZ, PRECISIONS, BLUE_RAMP

KEYS = ["date", "trip_id", "stop_sequence"]


def test_rows():
    """All test rows with their position in the trip and the chain-onset flag."""
    d = fs.con.sql(f"""
        SELECT date, trip_id, stop_sequence, is_bunched_now,
               row_number() OVER w AS rn,
               lag(is_bunched_now) OVER w AS prev_bunched
        FROM '{m.DS}' WHERE split = 'test'
        WINDOW w AS (PARTITION BY date, trip_id ORDER BY stop_sequence)
        ORDER BY date, trip_id, stop_sequence
    """).df()
    d["onset"] = (d.is_bunched_now == True) & (d.prev_bunched == False)  # noqa: E712 (NULL-aware)
    return d


def scores(k):
    """Best-model score per test row for horizon k, keyed like test_rows()."""
    clf, cols, _ = best_model(k)
    te = fs.load(k, "test")
    keys = fs.con.sql(f"""
        SELECT {", ".join("d." + c for c in KEYS)}
        FROM '{m.DS}' d JOIN '{fs.EXTRA}' e USING (date, trip_id, stop_sequence)
        WHERE d.split = 'test' AND d.y_k{k} IS NOT NULL
        ORDER BY d.date, d.trip_id, d.stop_sequence
    """).df()  # same filter and order as fs.load
    assert len(keys) == len(te)
    keys["p"] = clf.predict_proba(m.prep_hgb_frame(te[cols]))[:, 1]
    keys["y"] = te["y"].to_numpy(int)
    return keys


def compute():
    fa = json.load(open("data/results/false_alarms.json"))
    rows = test_rows()
    on = rows.loc[rows.onset, ["date", "trip_id", "rn"]].copy()
    lengths = chain_lengths(rows)
    res = {"n_onsets": int(len(on)), "n_bunched_stops": int(rows.is_bunched_now.fillna(False).sum()),
           "chain_length_median": float(np.median(lengths)), "chain_length_mean": round(float(lengths.mean()), 2),
           "per_k": {}}
    for k in HZ:
        s = scores(k).merge(rows[KEYS + ["rn"]], on=KEYS)
        s["rn"] += k  # the stop this score is about
        hit = on.merge(s[["date", "trip_id", "rn", "p", "y"]], on=["date", "trip_id", "rn"], how="left")
        assert (hit.y.dropna() == 1).all()  # the target k stops ahead is the onset itself
        r = {"onsets_with_prediction": int(hit.p.notna().sum())}
        for lvl in PRECISIONS:
            thr = fa[str(k)]["recall_at_precision"][str(lvl)]["threshold"]
            flag = (hit.p >= thr).to_numpy()
            on[f"k{k}_p{lvl}"] = flag
            r[str(lvl)] = {"threshold": thr, "caught": int(flag.sum()),
                           "share_of_onsets": round(float(flag.mean()), 4),
                           "share_of_predictable_onsets": round(float(flag.sum() / hit.p.notna().sum()), 4),
                           "recall_all_positives": fa[str(k)]["recall_at_precision"][str(lvl)]["recall"]}
        res["per_k"][str(k)] = r
        print(k, {q: r[q]["share_of_onsets"] for q in map(str, PRECISIONS)}, flush=True)
    res["earliest_warning"] = {}
    for lvl in PRECISIONS:
        earliest = np.zeros(len(on), dtype=int)
        for k in HZ:  # ascending, so the last hit is the earliest warning
            earliest[on[f"k{k}_p{lvl}"].to_numpy()] = k
        res["earliest_warning"][str(lvl)] = {str(k): round(float((earliest == k).mean()), 4) for k in [0] + HZ}
    return res


def chain_lengths(rows):
    b = rows.is_bunched_now.fillna(False).to_numpy()
    new_trip = rows.rn.to_numpy() == 1
    start = b & (~np.r_[False, b[:-1]] | new_trip)
    cid = np.cumsum(start)[b]
    return np.bincount(cid)[1:]


def plot(res):
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for lvl, col in zip(PRECISIONS, BLUE_RAMP):
        q = str(lvl)
        ax.plot(HZ, [res["per_k"][str(k)][q]["share_of_onsets"] for k in HZ], color=col, linewidth=2,
                marker="o", markersize=6, label=f"precision {lvl:.1f}")
    lvl = "0.8"
    ax.plot(HZ, [res["per_k"][str(k)][lvl]["recall_all_positives"] for k in HZ], color=MUTED, linewidth=1.3,
            linestyle="--", marker="o", markersize=4, label="all bunched stops, precision 0.8 (fig 18)")
    for k in HZ:
        v = res["per_k"][str(k)][lvl]["share_of_onsets"]
        ax.annotate(f"{v:.0%}", (k, v), xytext=(7, -13 if k == 1 else 5), textcoords="offset points", ha="left",
                    fontsize=8.5, color=INK2)
    ax.set_xticks(HZ)
    ax.set_xlim(0.5, 8.6)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("warning given k stops before the first bunched stop")
    ax.set_ylabel("share of bunching chains caught")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Bunching chains warned about before the bus was bunched (test split, Nov-Dec)",
                 fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("figures/fig24_chain_onsets_caught.png", dpi=200)
    plt.close(fig)


def main():
    if "--plot" in sys.argv:
        plot(json.load(open("data/results/chain_onsets.json")))
        return
    res = compute()
    with open("data/results/chain_onsets.json", "w") as f:
        json.dump(res, f, indent=1)
    plot(res)
    print("-> data/results/chain_onsets.json, figures/fig24_chain_onsets_caught.png")


if __name__ == "__main__":
    main()
