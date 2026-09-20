"""Build the sequence table for the Keras model: (headway_ratio, dwell,
delay_growth) at stops n-5..n, one row per (date, trip_id, stop_sequence),
joined to the same y_k targets and split as the tabular dataset. Sequence
construction uses only lag/backward window frames (same leakage-safety
argument as the tabular features in features.py).
"""
import time
import duckdb

HW = "data/processed/headways.parquet"
DS = "data/processed/model_dataset.parquet"
OUT = "data/processed/sequence_dataset.parquet"

LAGS = [5, 4, 3, 2, 1, 0]

lag_cols = ",\n    ".join(
    f"lag(headway_ratio, {L}) OVER w AS hr_{5-L}, "
    f"lag(dwell, {L}) OVER w AS dw_{5-L}, "
    f"lag(delay_growth, {L}) OVER w AS dg_{5-L}"
    for L in LAGS
)

SQL = f"""
WITH seq AS (
    SELECT date, trip_id, stop_sequence,
        {lag_cols}
    FROM '{HW}'
    WINDOW w AS (PARTITION BY date, trip_id ORDER BY stop_sequence)
)
SELECT s.*, m.split, m.y_k1, m.y_k2, m.y_k3, m.y_k5, m.y_k8
FROM seq s
JOIN (SELECT date, trip_id, stop_sequence, split, y_k1, y_k2, y_k3, y_k5, y_k8
      FROM '{DS}') m
  ON s.date = m.date AND s.trip_id = m.trip_id AND s.stop_sequence = m.stop_sequence
"""


def main():
    con = duckdb.connect()
    con.sql("PRAGMA threads=8")
    t0 = time.time()
    con.sql(f"COPY ({SQL}) TO '{OUT}' (FORMAT PARQUET)")
    print(f"sequence dataset written in {time.time()-t0:.1f}s")
    n = con.sql(f"SELECT count(*) FROM '{OUT}'").fetchone()[0]
    print("rows:", n)
    print(con.sql(f"SELECT split, count(*) FROM '{OUT}' GROUP BY 1").fetchall())


if __name__ == "__main__":
    main()
