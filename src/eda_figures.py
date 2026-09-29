"""Phase 3 - descriptive analysis figures (report task A).

All heavy aggregation is done in DuckDB (never pull millions of raw rows into
pandas); matplotlib only receives already-aggregated tables. Palette follows
the dataviz skill's reference instance (references/palette.md): categorical
slot 1 = blue #2a78d6, slot 2 = orange #eb6834; sequential = the blue ramp;
chart chrome (ink, gridlines, baseline) from the same reference. Figure 1 uses
small multiples (one panel per line) rather than 8 overlaid colored lines,
per the skill's guidance that categorical color only guarantees pairwise
distinguishability up to 3 series - past that, fold to facets.
"""
import duckdb
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- palette (references/palette.md, light mode) ---
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DS = "data/processed/model_dataset.parquet"
HW = "data/processed/headways.parquet"

LINE_ORDER = ["4", "116", "117", "179", "401", "474", "541", "607"]

con = duckdb.connect()
con.sql("PRAGMA threads=8")


def df(sql):
    return con.sql(sql).df()


def fig1_cv_vs_stop_sequence():
    d = df(f"""
        SELECT LineNumber, stop_sequence,
            stddev(headway_obs)/avg(headway_obs) AS cv, count(*) n
        FROM '{HW}'
        WHERE headway_obs IS NOT NULL AND headway_obs > 0 AND headway_obs < 3600
        GROUP BY 1,2 HAVING count(*) >= 30
    """)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    for ax, line in zip(axes.flat, LINE_ORDER):
        sub = d[d.LineNumber == line].sort_values("stop_sequence")
        ax.plot(sub.stop_sequence, sub.cv, color=BLUE, linewidth=1.6)
        # light local trend to cut through per-stop noise
        if len(sub) > 5:
            roll = sub.set_index("stop_sequence")["cv"].rolling(5, center=True, min_periods=1).mean()
            ax.plot(roll.index, roll.values, color="#0d366b", linewidth=1.2, alpha=0.6)
        ax.set_title(f"Line {line}", fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, d.cv.quantile(0.98))
    for ax in axes[-1, :]:
        ax.set_xlabel("stop_sequence", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("headway CV", fontsize=9)
    fig.suptitle("Headway coefficient of variation along the route, by line", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("figures/fig1_cv_vs_stop_sequence.png", dpi=200)
    plt.close(fig)
    return d


def fig2_bunching_heatmap():
    d = df(f"""
        SELECT LineNumber, hour_of_day,
            100.0*avg(is_bunched_now::INT) AS pct_bunched, count(*) n
        FROM '{DS}' WHERE headway_ratio IS NOT NULL AND hour_of_day BETWEEN 4 AND 26
        GROUP BY 1,2
    """)
    hours = sorted(d.hour_of_day.unique())
    mat = d.pivot(index="LineNumber", columns="hour_of_day", values="pct_bunched").reindex(LINE_ORDER)[hours]
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(mat.values, aspect="auto", cmap=_seq_cmap(), vmin=0, vmax=mat.values[~pd.isna(mat.values)].max())
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([str(h % 24) for h in hours], fontsize=8)
    ax.set_yticks(range(len(LINE_ORDER)))
    ax.set_yticklabels(LINE_ORDER, fontsize=9)
    ax.set_xlabel("hour of day (service time, may exceed 23 for post-midnight)")
    ax.set_ylabel("line")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("% bunched (|headway ratio| < 0.25)", fontsize=9, color=INK_SECONDARY)
    fig.suptitle("Bunching rate by hour of day and line", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/fig2_bunching_heatmap_hour_line.png", dpi=200)
    plt.close(fig)
    return d


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)


def fig3_bunching_by_month():
    d = df(f"""
        SELECT month_of_year, 100.0*avg(is_bunched_now::INT) AS pct_bunched, count(*) n
        FROM '{DS}' WHERE headway_ratio IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvspan(0.5, 2.5, color=GRID, alpha=0.8, zorder=0)
    ax.axvspan(11.5, 12.5, color=GRID, alpha=0.8, zorder=0)
    ax.axvspan(6.5, 8.5, color="#fbeadd", alpha=0.8, zorder=0)
    ax.plot(d.month_of_year, d.pct_bunched, color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("month (2024)")
    ax.set_ylabel("% bunched")
    ax.set_xlim(0.5, 12.5)
    ax.text(1.5, ax.get_ylim()[1]*0.96, "winter", fontsize=8, color=INK_SECONDARY, ha="center")
    ax.text(7.5, ax.get_ylim()[1]*0.96, "summer", fontsize=8, color=INK_SECONDARY, ha="center")
    fig.suptitle("Bunching rate by month, all 8 lines pooled", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/fig3_bunching_by_month.png", dpi=200)
    plt.close(fig)
    return d


def fig4_spatial_hotspots():
    d = df(f"""
        WITH s AS (
            SELECT observed_stop_id, 100.0*avg(is_bunched_now::INT) AS pct_bunched, count(*) n
            FROM '{DS}' WHERE headway_ratio IS NOT NULL
            GROUP BY 1 HAVING count(*) >= 500
        )
        SELECT s.*, st.stop_lat, st.stop_lon, st.stop_name
        FROM s LEFT JOIN read_csv_auto('stops.csv') st ON s.observed_stop_id = st.stop_id
        WHERE st.stop_lat IS NOT NULL
    """)
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(d.stop_lon, d.stop_lat, c=d.pct_bunched, cmap=_seq_cmap(),
                     s=18 + d.n / d.n.max() * 60, edgecolors="white", linewidths=0.3, vmin=0)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect(1.8)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("% bunched at this stop", fontsize=9, color=INK_SECONDARY)
    top = d[d.n >= 2000].sort_values("pct_bunched", ascending=False).head(5)
    for rank, (_, r) in enumerate(top.iterrows()):
        ax.annotate(f"{rank+1}", (r.stop_lon, r.stop_lat), fontsize=8, color=INK, fontweight="bold",
                    xytext=(6, 6), textcoords="offset points",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=INK_MUTED, lw=0.6))
    fig.suptitle("Bunching hotspots - stop-level bunching rate, 8 selected lines", fontsize=12, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.01, "Marker size ~ observation count. Plain lat/lon axes, no basemap.", fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig("figures/fig4_spatial_hotspots.png", dpi=200)
    plt.close(fig)
    return d


def fig5_headway_ratio_distribution():
    d = df(f"""
        SELECT round(GREATEST(LEAST(headway_ratio,3.0),-1.0)*20)/20 AS bucket,
            CASE WHEN hour_of_day IN (7,8,16,17) THEN 'peak' ELSE 'off-peak' END AS period,
            count(*) n
        FROM '{DS}' WHERE headway_ratio IS NOT NULL
        GROUP BY 1,2
    """)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for period, color in [("off-peak", BLUE), ("peak", ORANGE)]:
        sub = d[d.period == period].sort_values("bucket")
        density = sub.n / sub.n.sum() / 0.05
        ax.plot(sub.bucket, density, color=color, linewidth=1.8, label=period)
        ax.fill_between(sub.bucket, density, color=color, alpha=0.12)
    ax.axvline(0.25, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.axvline(-0.25, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.axvline(1.75, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.axvspan(-0.25, 0.25, color=BLUE, alpha=0.06, zorder=0)
    ax.text(0.25, ax.get_ylim()[1]*0.9, " bunched zone", fontsize=7, color=INK_SECONDARY)
    ax.text(1.75, ax.get_ylim()[1]*0.9, " gap >", fontsize=7, color=INK_SECONDARY)
    ax.set_xlabel("headway ratio (observed / scheduled) - negative means the follower overtook its nominal leader")
    ax.set_ylabel("density")
    ax.set_xlim(-1, 3)
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Headway ratio distribution, peak vs off-peak", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/fig5_headway_ratio_distribution.png", dpi=200)
    plt.close(fig)
    return d


def fig6_dwell_by_stop_type():
    d = df(f"""
        SELECT round(LEAST(dwell,120)/2)*2 AS bucket, is_interchange, count(*) n
        FROM '{DS}' WHERE dwell IS NOT NULL AND dwell >= 0
        GROUP BY 1,2
    """)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for flag, label, color in [(False, "ordinary stop", BLUE), (True, "interchange", ORANGE)]:
        sub = d[d.is_interchange == flag].sort_values("bucket")
        density = sub.n / sub.n.sum() / 2.0
        ax.plot(sub.bucket, density, color=color, linewidth=1.8, label=label)
        ax.fill_between(sub.bucket, density, color=color, alpha=0.12)
    ax.set_xlabel("dwell time (s)")
    ax.set_ylabel("density")
    ax.set_xlim(0, 120)
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Dwell time distribution: interchange vs ordinary stops", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/fig6_dwell_by_stop_type.png", dpi=200)
    plt.close(fig)
    means = df(f"""
        SELECT is_interchange, avg(dwell) mean_dwell, median(dwell) median_dwell, count(*) n
        FROM '{DS}' WHERE dwell IS NOT NULL AND dwell >= 0 GROUP BY 1
    """)
    return d, means


def fig7_delay_growth_vs_headway_ratio():
    d = df(f"""
        SELECT round(LEAST(headway_ratio,3.0)*20)/20 AS hr_bucket,
               round(GREATEST(LEAST(delay_growth,120),-60)/5)*5 AS dg_bucket,
               count(*) n
        FROM '{HW}'
        WHERE headway_ratio BETWEEN -1 AND 3 AND delay_growth IS NOT NULL
        GROUP BY 1,2
    """)
    piv = d.pivot(index="dg_bucket", columns="hr_bucket", values="n").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap=_seq_cmap(),
                    extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()],
                    norm=matplotlib.colors.LogNorm(vmin=1, vmax=piv.values.max()))
    ax.axvline(0.25, color=INK, linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(-0.25, color=INK, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("headway ratio at stop n (bunched zone: -0.25 to 0.25)")
    ax.set_ylabel("delay growth at stop n (s, dwell-driven)")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("row count (log scale)", fontsize=9, color=INK_SECONDARY)
    corr = con.sql(f"""
        SELECT corr(headway_ratio, delay_growth) FROM '{HW}'
        WHERE headway_ratio BETWEEN -1 AND 3 AND delay_growth IS NOT NULL
    """).fetchone()[0]
    fig.suptitle(f"Delay growth vs. headway ratio (feedback loop), Pearson r = {corr:.3f}", fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figures/fig7_delay_growth_vs_headway_ratio.png", dpi=200)
    plt.close(fig)
    return d, corr


def fig10_bunching_along_route():
    # relative position along each trip, 0 = first observed stop, 1 = last,
    # binned into fifths so lines of different lengths are comparable
    seg = "least(4, floor(5.0*(stop_sequence-first_seq)/nullif(last_seq-first_seq,0)))::INT"
    where = "WHERE headway_ratio IS NOT NULL AND last_seq > first_seq"
    per_line = df(f"""
        SELECT LineNumber, {seg} AS seg, 100.0*avg(is_bunched::INT) AS pct_bunched, count(*) n
        FROM '{HW}' {where} GROUP BY 1,2 ORDER BY 1,2
    """)
    overall = df(f"""
        SELECT {seg} AS seg, 100.0*avg(is_bunched::INT) AS pct_bunched, count(*) n
        FROM '{HW}' {where} GROUP BY 1 ORDER BY 1
    """)
    x_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # individual lines as recessive context, the aggregate is the one series
    ends = []
    for line in LINE_ORDER:
        sub = per_line[per_line.LineNumber == line].sort_values("seg")
        ax.plot(sub.seg, sub.pct_bunched, color=BASELINE, linewidth=1.2, zorder=2)
        ends.append([sub.pct_bunched.iloc[-1], line])
    # nudge route-end labels apart so lines ending close together stay legible
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.2)
    for y, line in ends:
        ax.text(4.1, y, line, va="center", fontsize=8, color=INK_MUTED)
    ax.plot(overall.seg, overall.pct_bunched, color=BLUE, linewidth=2.5, zorder=3,
            marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for _, r in overall.iterrows():
        ax.annotate(f"{r.pct_bunched:.1f}%", (r.seg, r.pct_bunched),
                    xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=9, color=INK)
    ax.annotate("All lines", (overall.seg.iloc[0], overall.pct_bunched.iloc[0]),
                xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
                fontsize=9, color=INK_SECONDARY)

    ax.set_xticks(range(5))
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(0, per_line.pct_bunched.max() * 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xlabel("position along the trip (share of stops travelled)", fontsize=9)
    ax.set_ylabel("share of arrivals bunched (|headway ratio| < 0.25)", fontsize=9)
    ax.grid(axis="x", visible=False)
    first, last = overall.pct_bunched.iloc[0], overall.pct_bunched.iloc[-1]
    fig.suptitle(f"Bunching risk grows along the route: {first:.1f}% → {last:.1f}% ({last/first:.1f}×)",
                 fontsize=13, color=INK, x=0.02, y=0.98, ha="left")
    ax.set_title("All lines pooled (blue); individual lines in grey, labelled at route end",
                 fontsize=9, color=INK_SECONDARY, loc="left")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig("figures/fig10_bunching_along_route.png", dpi=200)
    plt.close(fig)
    return per_line, overall


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    print("fig1"); d1 = fig1_cv_vs_stop_sequence()
    print("fig2"); d2 = fig2_bunching_heatmap()
    print("fig3"); d3 = fig3_bunching_by_month()
    print("fig4"); d4 = fig4_spatial_hotspots()
    print("fig5"); d5 = fig5_headway_ratio_distribution()
    print("fig6"); d6, means6 = fig6_dwell_by_stop_type()
    print("fig7"); d7, corr7 = fig7_delay_growth_vs_headway_ratio()
    print("fig10"); d10 = fig10_bunching_along_route()
    print("done")
    print("fig1 cv range:", d1.cv.min(), d1.cv.max())
    print("fig3 by month:\n", d3)
    print("fig4 top hotspots:\n", d4.sort_values("pct_bunched", ascending=False).head(8)[["stop_name","pct_bunched","n"]])
    print("fig6 means:\n", means6)
    print("fig7 corr:", corr7)
