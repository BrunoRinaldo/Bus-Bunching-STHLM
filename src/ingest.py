"""Phase 1 - DuckDB scan of the raw GTFS-RT CSV.gz files into a filtered parquet subset.

Scope: BUS mode only, 8 selected high-frequency lines, all 12 months of 2024.

Line selection (Phase 0, two passes):
1. From the January 2024 inventory, candidates were ranked by trip volume among
   lines with median scheduled peak headway <= 600 s.
2. A full-year completeness scan (fraction of rows with a non-null
   observed_arrival_time_sfm) then showed two of the original eight candidates,
   lines 1 and 3, had 44% and 64% missing real-time arrivals respectively -
   far worse than every other candidate (1-28%). This is a genuine data-quality
   finding, not something to route around: lines 1 and 3 were dropped and
   replaced with the next-best candidates by completeness, keeping the
   <= 600 s peak-headway criterion. See docs/01_data_description.md for the
   full completeness table.
"""
import time
import duckdb

SELECTED_LINES = ["4", "117", "179", "401", "474", "541", "607", "116"]

COLUMNS = """
    date, LineNumber, route_id, TransportMode, direction_id, trip_id, VehicleID,
    shape_id, stop_sequence, dist_traveled, previous_stop_id, observed_stop_id,
    scheduled_arrival_time_sfm, observed_arrival_time_sfm,
    observed_arrival_delay, observed_arrival_uncertainty,
    scheduled_departure_time_sfm, observed_departure_time_sfm,
    observed_departure_delay, observed_departure_uncertainty,
    tripupdates_and_gtfs_not_merged_within_service_id AS merge_flag_1,
    tripupdates_not_merged_with_stoptimes_or_trip     AS merge_flag_2
"""


def main():
    con = duckdb.connect()
    con.sql("PRAGMA threads=8")
    lines_sql = ", ".join(f"'{l}'" for l in SELECTED_LINES)
    src = (
        "read_csv('gtfs_rt_2024*.csv.gz', "
        "types={'LineNumber':'VARCHAR','route_id':'VARCHAR'}, union_by_name=true)"
    )
    t0 = time.time()
    con.sql(f"""
        COPY (
            SELECT {COLUMNS}
            FROM {src}
            WHERE TransportMode = 'BUS'
              AND LineNumber IN ({lines_sql})
        ) TO 'data/processed/bus_subset.parquet' (FORMAT PARQUET)
    """)
    print(f"ingest done in {time.time()-t0:.1f}s")

    n = con.sql("SELECT count(*) FROM 'data/processed/bus_subset.parquet'").fetchone()[0]
    print("rows written:", n)
    per_line = con.sql("""
        SELECT LineNumber, count(*) n_rows, count(DISTINCT trip_id) n_trips,
               count(DISTINCT date) n_days
        FROM 'data/processed/bus_subset.parquet' GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()
    for row in per_line:
        print(row)


if __name__ == "__main__":
    main()
