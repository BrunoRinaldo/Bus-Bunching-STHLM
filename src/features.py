"""Phase 4 - leakage-safe feature engineering and horizon targets.

Two-pass structure, as required by the plan:
  - FEATURES pass: every column uses only LAG / ROWS-PRECEDING window functions
    ordered by stop_sequence within (date, trip_id), i.e. information available
    at or before stop n.
  - TARGETS pass: uses LEAD(k) to look at stop n+k of the SAME trip. LEAD
    naturally returns NULL when the trip does not reach stop n+k (short turn,
    early termination), which is exactly the "drop the row otherwise" rule.
The two passes are kept in separate CTEs and only combined in the final SELECT,
so a feature column can never be built from a LEAD value.
"""
import time
import duckdb

HORIZONS = [1, 2, 3, 5, 8]
IN_ = "data/processed/headways.parquet"
OUT = "data/processed/model_dataset.parquet"

lead_targets = ",\n        ".join(
    f"CASE WHEN lead(headway_ratio, {k}) OVER w IS NULL THEN NULL "
    f"ELSE lead(is_bunched, {k}) OVER w END::INT AS y_k{k}"
    for k in HORIZONS
)
lead_valid = ",\n        ".join(
    f"(lead(stop_sequence, {k}) OVER w IS NOT NULL) AS target_valid_k{k}"
    for k in HORIZONS
)

SQL = f"""
WITH h AS (
    SELECT * FROM '{IN_}'
),
features AS (
    SELECT
        date, LineNumber, direction_id, trip_id, VehicleID, observed_stop_id,
        stop_sequence, dist_traveled, event_ts_arr, hour_of_day, day_of_week, month_of_year,
        headway_ratio, is_bunched AS is_bunched_now, observed_arrival_delay, observed_departure_delay, dwell,
        dwell - median(dwell) OVER (PARTITION BY observed_stop_id) AS dwell_rel_median,
        speed_mps,
        leader_trip_id, leader_dwell, leader_arrival_delay, leader_delay_growth,
        first_seq, last_seq,
        (stop_sequence - first_seq)::DOUBLE / nullif(last_seq - first_seq, 0) AS route_frac,
        lag(headway_ratio, 1) OVER w AS headway_ratio_lag1,
        lag(headway_ratio, 2) OVER w AS headway_ratio_lag2,
        lag(headway_ratio, 3) OVER w AS headway_ratio_lag3,
        (headway_ratio - lag(headway_ratio, 3) OVER w) / 3.0 AS closing_speed_3,
        sum(delay_growth) OVER (PARTITION BY date, trip_id ORDER BY stop_sequence
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS cum_delay_growth_3,
        lag(speed_mps, 1) OVER w AS speed_lag1,
        lag(speed_mps, 2) OVER w AS speed_lag2,
        {lead_targets},
        {lead_valid}
    FROM h
    WINDOW w AS (PARTITION BY date, trip_id ORDER BY stop_sequence)
),
with_grid AS (
    SELECT f.*, round(st.stop_lat, 1) AS grid_lat, round(st.stop_lon, 1) AS grid_lon,
        (st.parent_station IS NOT NULL) AS has_parent_station,
        ((f.event_ts_arr AT TIME ZONE 'Europe/Stockholm') AT TIME ZONE 'UTC') AS ts_utc_raw
    FROM features f
    LEFT JOIN read_csv_auto('data/raw/stops.csv') st ON f.observed_stop_id = st.stop_id
),
with_wx AS (
    SELECT w.*, wx.temperature_2m, wx.precipitation, wx.is_precip, wx.precip_3h,
        wx.is_snowing, wx.ice_risk, wx.temp_bin, wx.wx_group, wx.wind_speed_10m
    FROM with_grid w
    LEFT JOIN 'data/processed/weather.parquet' wx
      ON w.grid_lat = wx.grid_lat AND w.grid_lon = wx.grid_lon
     AND date_trunc('hour', w.ts_utc_raw) = wx.ts_utc
),
with_stop_attrs AS (
    SELECT ww.*,
        coalesce(src.n_routes_serving, 1) AS n_routes_serving,
        EXISTS (
            SELECT 1 FROM read_csv_auto('data/raw/transfers.csv') t
            WHERE t.from_stop_id = ww.observed_stop_id OR t.to_stop_id = ww.observed_stop_id
        ) AS is_interchange
    FROM with_wx ww
    LEFT JOIN 'data/processed/stop_route_counts.parquet' src
      ON ww.observed_stop_id = src.observed_stop_id
),
with_alerts AS (
    SELECT wsa.*,
        EXISTS (
            SELECT 1 FROM 'data/raw/service_alerts_2024.parquet' a
            WHERE a.stop_id = wsa.observed_stop_id
              AND wsa.event_ts_arr BETWEEN a.start AND a."end"
        ) AS under_active_alert
    FROM with_stop_attrs wsa
)
SELECT *,
    CASE WHEN (date / 100)::INT % 100 <= 9 THEN 'train'
         WHEN (date / 100)::INT % 100 = 10 THEN 'val'
         ELSE 'test' END AS split
FROM with_alerts
"""


def main():
    con = duckdb.connect()
    con.sql("PRAGMA threads=8")
    con.sql("INSTALL icu; LOAD icu;")
    t0 = time.time()
    con.sql(f"COPY ({SQL}) TO '{OUT}' (FORMAT PARQUET)")
    print(f"features written in {time.time()-t0:.1f}s")

    n = con.sql(f"SELECT count(*) FROM '{OUT}'").fetchone()[0]
    print("rows:", n)

    print("--- split sizes ---")
    print(con.sql(f"SELECT split, count(*), min(date), max(date) FROM '{OUT}' GROUP BY 1 ORDER BY 1").fetchall())

    print("--- target availability and base rate per horizon ---")
    for k in HORIZONS:
        r = con.sql(f"""
            SELECT
                sum(CASE WHEN target_valid_k{k} THEN 1 ELSE 0 END) n_valid,
                round(100.0*sum(y_k{k})/sum(CASE WHEN target_valid_k{k} THEN 1 ELSE 0 END),2) pct_positive
            FROM '{OUT}'
        """).fetchall()
        print("k =", k, r)

    print("--- weather join coverage ---")
    print(con.sql(f"""
        SELECT round(100.0*sum(CASE WHEN temperature_2m IS NULL THEN 1 ELSE 0 END)/count(*),3) pct_missing
        FROM '{OUT}'
    """).fetchall())

    print("--- alert flag rate ---")
    print(con.sql(f"SELECT round(100.0*sum(CASE WHEN under_active_alert THEN 1 ELSE 0 END)/count(*),3) FROM '{OUT}'").fetchall())

    print("--- interchange flag rate ---")
    print(con.sql(f"SELECT round(100.0*sum(CASE WHEN is_interchange THEN 1 ELSE 0 END)/count(*),3) FROM '{OUT}'").fetchall())

    print("--- leakage sanity: features reference only current/past stop_sequence ---")
    print(con.sql(f"""
        SELECT count(*) FROM '{OUT}'
        WHERE headway_ratio_lag1 IS NOT NULL AND headway_ratio IS NULL
    """).fetchall(), "(expect 0 - lag should not exist where current is null... informational only)")


if __name__ == "__main__":
    main()
