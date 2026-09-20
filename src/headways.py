"""Phase 2 - derived variables: event timestamps, dwell, delay growth, headway, labels.

Deviation from the plan (documented): headway grouping uses LineNumber rather than
route_id. The raw route_id carries a per-pattern suffix (e.g. "...-185V") that
fragments a single physical line into many groups and would break leader/follower
continuity across route variants at a shared stop. LineNumber is what a waiting
passenger actually experiences as "the line", so it is the correct grouping key for
a headway that is meant to measure bunching as riders see it. observed_stop_id
still restricts each group to trips that actually call at that stop, so route
variants that skip a stop are handled correctly.

Bug fixed in this version, in two steps: headway_ratio was originally computed
as headway_obs / headway_sched with no bound on headway_obs, so the ~0.8% of
rows with an absurd (>3h) headway_obs and a small tail of very negative ones
produced wildly out-of-range ratios (some below -40), and because
is_bunched = headway_ratio < 0.25, those extreme negative ratios were being
counted as "bunched" regardless of magnitude.

The first fix nulled out every negative headway_obs, which overcorrected:
checking the actual distribution of the 1.07% of rows with headway_obs < 0
shows 93% of them fall within -1800s (30 minutes) of zero - a follower trip
that physically overtook its scheduled leader and arrived first, which is
itself a genuine, strong bunching signal, not corrupted data. Only the
extreme tail (beyond -1800s, ~0.08% of all rows) is implausible as a single
overtake between two consecutively-scheduled trips and is almost certainly a
broken leader/follower pairing (a skipped trip in between - plan trap #2).

headway_ratio is therefore null (and headway_obs_invalid = true) only when
headway_obs falls outside [-1800, 10800] seconds; within that range,
including the negative part, it is treated as real and contributes to
is_bunched/is_gap and every downstream aggregate normally.
"""
import time
import duckdb

SRC = "data/processed/bus_subset.parquet"
OUT = "data/processed/headways.parquet"

SQL = f"""
WITH raw AS (
    SELECT *,
        strptime(CAST(date AS VARCHAR), '%Y%m%d') AS service_midnight
    FROM '{SRC}'
),
ts AS (
    SELECT *,
        service_midnight + to_seconds(observed_arrival_time_sfm)   AS event_ts_arr,
        service_midnight + to_seconds(observed_departure_time_sfm) AS event_ts_dep,
        service_midnight + to_seconds(scheduled_arrival_time_sfm)  AS sched_event_ts_arr
    FROM raw
),
per_trip AS (
    SELECT *,
        min(stop_sequence) OVER (PARTITION BY date, trip_id) AS first_seq,
        max(stop_sequence) OVER (PARTITION BY date, trip_id) AS last_seq,
        (observed_departure_time_sfm - observed_arrival_time_sfm) AS dwell_raw,
        (observed_departure_delay - observed_arrival_delay) AS delay_growth,
        observed_arrival_time_sfm
            - lag(observed_departure_time_sfm) OVER (
                PARTITION BY date, trip_id ORDER BY stop_sequence) AS run_time
    FROM ts
),
derived AS (
    SELECT *,
        CASE
            WHEN dwell_raw IS NULL THEN NULL
            WHEN stop_sequence = first_seq OR stop_sequence = last_seq THEN NULL
            WHEN dwell_raw < 0 THEN NULL
            ELSE dwell_raw
        END AS dwell,
        (dwell_raw < 0) AS dwell_negative_flag,
        CASE WHEN run_time IS NOT NULL AND run_time > 0
             THEN dist_traveled / run_time ELSE NULL END AS speed_mps
    FROM per_trip
),
headway_sched_pair AS (
    SELECT *,
        lag(trip_id) OVER (
            PARTITION BY date, LineNumber, direction_id, observed_stop_id
            ORDER BY scheduled_arrival_time_sfm) AS leader_trip_id,
        scheduled_arrival_time_sfm
            - lag(scheduled_arrival_time_sfm) OVER (
                PARTITION BY date, LineNumber, direction_id, observed_stop_id
                ORDER BY scheduled_arrival_time_sfm) AS headway_sched,
        observed_arrival_time_sfm
            - lag(observed_arrival_time_sfm) OVER (
                PARTITION BY date, LineNumber, direction_id, observed_stop_id
                ORDER BY scheduled_arrival_time_sfm) AS headway_obs,
        lag(observed_arrival_delay) OVER (
                PARTITION BY date, LineNumber, direction_id, observed_stop_id
                ORDER BY scheduled_arrival_time_sfm) AS leader_arrival_delay,
        lag(dwell) OVER (
                PARTITION BY date, LineNumber, direction_id, observed_stop_id
                ORDER BY scheduled_arrival_time_sfm) AS leader_dwell,
        lag(delay_growth) OVER (
                PARTITION BY date, LineNumber, direction_id, observed_stop_id
                ORDER BY scheduled_arrival_time_sfm) AS leader_delay_growth
    FROM derived
),
headway_obs_order AS (
    SELECT date, LineNumber, direction_id, observed_stop_id, trip_id,
        observed_arrival_time_sfm
            - lag(observed_arrival_time_sfm) OVER (
                PARTITION BY date, LineNumber, direction_id, observed_stop_id
                ORDER BY observed_arrival_time_sfm) AS headway_obs_orderdef
    FROM derived
),
joined AS (
    SELECT h.*, o.headway_obs_orderdef
    FROM headway_sched_pair h
    LEFT JOIN headway_obs_order o
      ON h.date = o.date AND h.LineNumber = o.LineNumber
     AND h.direction_id = o.direction_id AND h.observed_stop_id = o.observed_stop_id
     AND h.trip_id = o.trip_id
),
with_ratio AS (
    SELECT *,
        (headway_obs < -1800 OR headway_obs > 10800) AS headway_obs_invalid,
        CASE WHEN headway_sched > 0 AND headway_obs BETWEEN -1800 AND 10800
             THEN headway_obs::DOUBLE / headway_sched ELSE NULL END AS headway_ratio,
        CASE WHEN headway_sched > 0 AND headway_obs_orderdef BETWEEN -1800 AND 10800
             THEN headway_obs_orderdef::DOUBLE / headway_sched ELSE NULL END AS headway_ratio_obsorder,
        hour(sched_event_ts_arr) AS hour_of_day,
        dayofweek(sched_event_ts_arr) AS day_of_week,
        month(sched_event_ts_arr) AS month_of_year
    FROM joined
)
SELECT *,
    -- Bunching is physical proximity between the two vehicles, not the sign of
    -- who is nominally ahead: a follower that just overtook its scheduled
    -- leader by a few seconds (headway_ratio slightly negative) is exactly as
    -- bunched as one trailing by a few seconds (headway_ratio slightly
    -- positive). A large-MAGNITUDE negative ratio (follower arrived e.g. 20
    -- minutes before its nominal leader) is NOT bunching - the two vehicles
    -- are far apart in absolute time, just reordered - so the threshold must
    -- apply to abs(headway_ratio), not the signed value. is_gap has no
    -- negative-side analogue (a large negative ratio is reordering, not a
    -- long wait) so it is left as the plan defines it, positive side only.
    (abs(headway_ratio) < 0.25) AS is_bunched,
    (headway_ratio > 1.75) AS is_gap
FROM with_ratio
"""


def main():
    con = duckdb.connect()
    con.sql("PRAGMA threads=8")
    t0 = time.time()
    con.sql(f"COPY ({SQL}) TO '{OUT}' (FORMAT PARQUET)")
    print(f"headways written in {time.time()-t0:.1f}s")

    n = con.sql(f"SELECT count(*) FROM '{OUT}'").fetchone()[0]
    print("rows:", n)

    print("--- headway_sched distribution (mode check) ---")
    print(con.sql(f"""
        SELECT headway_sched, count(*) n FROM '{OUT}'
        WHERE headway_sched IS NOT NULL
        GROUP BY 1 ORDER BY n DESC LIMIT 10
    """).fetchall())

    print("--- null / negative / absurd headway_obs ---")
    print(con.sql(f"""
        SELECT
            round(100.0*sum(CASE WHEN headway_obs IS NULL THEN 1 ELSE 0 END)/count(*),2) AS pct_null,
            round(100.0*sum(CASE WHEN headway_obs<0 THEN 1 ELSE 0 END)/count(*),3) AS pct_negative,
            round(100.0*sum(CASE WHEN headway_obs>10800 THEN 1 ELSE 0 END)/count(*),3) AS pct_over_3h,
            count(*) total
        FROM '{OUT}'
    """).fetchall())

    print("--- base rate of is_bunched per line (abs(headway_ratio) < 0.25) ---")
    print(con.sql(f"""
        SELECT LineNumber,
            round(100.0*sum(is_bunched::INT) / sum(CASE WHEN headway_ratio IS NOT NULL THEN 1 ELSE 0 END), 2) AS pct_bunched,
            round(100.0*sum(is_gap::INT) / sum(CASE WHEN headway_ratio IS NOT NULL THEN 1 ELSE 0 END), 2) AS pct_gap,
            sum(CASE WHEN headway_ratio IS NOT NULL THEN 1 ELSE 0 END) n_valid
        FROM '{OUT}' GROUP BY 1 ORDER BY 1
    """).fetchall())

    print("--- sensitivity of bunching rate to threshold (on abs(headway_ratio)) ---")
    print(con.sql(f"""
        SELECT
            round(100.0*sum(CASE WHEN abs(headway_ratio) < 0.20 THEN 1 ELSE 0 END)/count(*),2) AS thr_020,
            round(100.0*sum(CASE WHEN abs(headway_ratio) < 0.25 THEN 1 ELSE 0 END)/count(*),2) AS thr_025,
            round(100.0*sum(CASE WHEN abs(headway_ratio) < 0.33 THEN 1 ELSE 0 END)/count(*),2) AS thr_033,
            round(100.0*sum(CASE WHEN abs(headway_ratio) < 0.50 THEN 1 ELSE 0 END)/count(*),2) AS thr_050
        FROM '{OUT}' WHERE headway_ratio IS NOT NULL
    """).fetchall())

    print("--- definition (a) vs (b) disagreement, overall and within bunched(b) ---")
    print(con.sql(f"""
        SELECT
            round(100.0*sum(CASE WHEN (abs(headway_ratio) < 0.25) != (abs(headway_ratio_obsorder) < 0.25) THEN 1 ELSE 0 END)
                  / count(*), 2) AS pct_disagree_overall,
            round(100.0*sum(CASE WHEN is_bunched
                  AND (headway_ratio_obsorder IS NULL OR abs(headway_ratio_obsorder) >= 0.25) THEN 1 ELSE 0 END)
                  / sum(is_bunched::INT), 2) AS pct_bunched_b_missed_by_a
        FROM '{OUT}' WHERE headway_ratio IS NOT NULL AND headway_ratio_obsorder IS NOT NULL
    """).fetchall())

    print("--- dwell / delay_growth sanity ---")
    print(con.sql(f"""
        SELECT
            round(100.0*sum(CASE WHEN dwell IS NULL THEN 1 ELSE 0 END)/count(*),2) AS pct_dwell_null,
            round(avg(dwell),1) AS mean_dwell, round(median(dwell),1) AS median_dwell,
            round(100.0*sum(CASE WHEN dwell_negative_flag THEN 1 ELSE 0 END)/count(*),3) AS pct_dwell_negative
        FROM '{OUT}'
    """).fetchall())


if __name__ == "__main__":
    main()
