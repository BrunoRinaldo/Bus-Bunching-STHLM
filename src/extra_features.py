"""Candidate features for the long horizons (k=5, 8), kept outside model_dataset.

Writes data/processed/extra_features.parquet, one row per model_dataset row,
keyed by (date, trip_id, stop_sequence). model_dataset.parquet is left
untouched, so every earlier result stays reproducible.

Groups (each column uses only stop n and earlier, the timetable, or
per-stop statistics computed on the training months Jan-Sep):
  long_lags  headway_ratio lag 5 and 8, closing speed over 8 stops,
             delay growth summed over 8 stops, dwell summed over 3 and 5
             stops (the rolling dwell feature missing for H2, doc 3 §5.2)
  schedule   scheduled headway (s) and scheduled running time from stop n
             to stop n+5 and n+8 of the same trip
  leader     the leader's own headway ratio and speed at stop n, kept only
             when the leader reached stop n first (headway_obs > 0)
  ahead      over the next 5 / 8 stops of the trip: summed training-month
             median dwell and number of interchange stops
"""
import time

import duckdb

HW = "data/processed/headways.parquet"
DS = "data/processed/model_dataset.parquet"
OUT = "data/processed/extra_features.parquet"

SQL = f"""
WITH stop_med AS (
    SELECT observed_stop_id, median(dwell) AS med_dwell_train
    FROM '{HW}' WHERE month_of_year <= 9 GROUP BY 1
),
hw AS (
    SELECT h.date, h.trip_id, h.stop_sequence, h.LineNumber, h.direction_id, h.observed_stop_id,
        h.headway_ratio, h.headway_sched, h.headway_obs, h.dwell, h.delay_growth, h.speed_mps,
        h.leader_trip_id, h.scheduled_arrival_time_sfm, sm.med_dwell_train
    FROM '{HW}' h LEFT JOIN stop_med sm USING (observed_stop_id)
),
ic AS (
    SELECT date, trip_id, stop_sequence, is_interchange::INT AS ic FROM '{DS}'
),
per_trip AS (
    SELECT hw.*, ic.ic,
        lag(headway_ratio, 5) OVER w AS headway_ratio_lag5,
        lag(headway_ratio, 8) OVER w AS headway_ratio_lag8,
        (headway_ratio - lag(headway_ratio, 8) OVER w) / 8.0 AS closing_speed_8,
        sum(delay_growth) OVER (w ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS cum_delay_growth_8,
        sum(dwell) OVER (w ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS dwell_sum_3,
        sum(dwell) OVER (w ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS dwell_sum_5,
        lead(scheduled_arrival_time_sfm, 5) OVER w - scheduled_arrival_time_sfm AS sched_run_5,
        lead(scheduled_arrival_time_sfm, 8) OVER w - scheduled_arrival_time_sfm AS sched_run_8,
        sum(med_dwell_train) OVER (w ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING) AS ahead_med_dwell_5,
        sum(med_dwell_train) OVER (w ROWS BETWEEN 1 FOLLOWING AND 8 FOLLOWING) AS ahead_med_dwell_8,
        sum(ic.ic) OVER (w ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING) AS ahead_interchanges_5,
        sum(ic.ic) OVER (w ROWS BETWEEN 1 FOLLOWING AND 8 FOLLOWING) AS ahead_interchanges_8
    FROM hw LEFT JOIN ic USING (date, trip_id, stop_sequence)
    WINDOW w AS (PARTITION BY date, trip_id ORDER BY stop_sequence)
)
SELECT p.date, p.trip_id, p.stop_sequence,
    p.headway_ratio_lag5, p.headway_ratio_lag8, p.closing_speed_8, p.cum_delay_growth_8,
    p.dwell_sum_3, p.dwell_sum_5,
    p.headway_sched::DOUBLE AS headway_sched_s, p.sched_run_5::DOUBLE AS sched_run_5,
    p.sched_run_8::DOUBLE AS sched_run_8,
    CASE WHEN p.headway_obs > 0 THEN l.headway_ratio END AS leader_headway_ratio,
    CASE WHEN p.headway_obs > 0 THEN l.speed_mps END AS leader_speed,
    p.ahead_med_dwell_5, p.ahead_med_dwell_8, p.ahead_interchanges_5, p.ahead_interchanges_8
FROM per_trip p
LEFT JOIN hw l
  ON l.date = p.date AND l.trip_id = p.leader_trip_id AND l.LineNumber = p.LineNumber
 AND l.direction_id = p.direction_id AND l.observed_stop_id = p.observed_stop_id
"""

GROUPS = {
    "long_lags": ["headway_ratio_lag5", "headway_ratio_lag8", "closing_speed_8", "cum_delay_growth_8",
                  "dwell_sum_3", "dwell_sum_5"],
    "schedule": ["headway_sched_s", "sched_run_5", "sched_run_8"],
    "leader": ["leader_headway_ratio", "leader_speed"],
    "ahead": ["ahead_med_dwell_5", "ahead_med_dwell_8", "ahead_interchanges_5", "ahead_interchanges_8"],
}


def main():
    con = duckdb.connect()
    con.sql("PRAGMA threads=8")
    t0 = time.time()
    con.sql(f"COPY ({SQL}) TO '{OUT}' (FORMAT PARQUET)")
    n, nd = con.sql(f"SELECT count(*), count(DISTINCT (date, trip_id, stop_sequence)) FROM '{OUT}'").fetchone()
    n_ds = con.sql(f"SELECT count(*) FROM '{DS}'").fetchone()[0]
    print(f"written in {time.time()-t0:.0f}s: {n} rows, {nd} distinct keys, model_dataset {n_ds} rows")
    cols = [c for g in GROUPS.values() for c in g]
    print(con.sql("SELECT " + ", ".join(f"round(100.0*avg(({c} IS NULL)::INT),1) AS {c}" for c in cols)
                  + f" FROM '{OUT}'").df().T.rename(columns={0: "pct_null"}))


if __name__ == "__main__":
    main()
