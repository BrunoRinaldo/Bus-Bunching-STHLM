"""Export a sample of the fully encoded feature matrix - the actual numeric
array the logistic regression model trains on, after imputation, missing-
indicators, scaling, and one-hot encoding - not the raw model_dataset.parquet
columns. Fit on a large train sample for stable statistics, then transform a
small, class-balanced, cross-split sample for readability.
"""
import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

K = 3  # core RQ3 horizon
NUMERIC = [
    "headway_ratio", "observed_arrival_delay", "observed_departure_delay",
    "dwell", "dwell_rel_median", "speed_mps",
    "leader_dwell", "leader_arrival_delay", "leader_delay_growth",
    "stop_sequence", "route_frac", "dist_traveled",
    "hour_of_day", "day_of_week", "month_of_year", "n_routes_serving",
    "headway_ratio_lag1", "headway_ratio_lag2", "headway_ratio_lag3",
    "closing_speed_3", "cum_delay_growth_3", "speed_lag1", "speed_lag2",
    "temperature_2m", "temperature_2m_sq", "precipitation", "precip_3h", "wind_speed_10m",
]
BOOLEAN = ["has_parent_station", "is_interchange", "under_active_alert",
           "is_precip", "is_snowing", "ice_risk"]
CATEGORICAL = ["LineNumber", "direction_id", "wx_group"]
ALL_COLS = NUMERIC + BOOLEAN + CATEGORICAL
_FETCH_COLS = [c for c in ALL_COLS if c != "temperature_2m_sq"]

con = duckdb.connect()
con.sql("PRAGMA threads=4")

print("loading a 500k-row train sample to fit the encoders on...")
fit_df = con.sql(f"""
    SELECT * FROM (
        SELECT {", ".join(_FETCH_COLS)}, y_k{K} AS y
        FROM 'data/processed/model_dataset.parquet'
        WHERE split='train' AND y_k{K} IS NOT NULL
    ) USING SAMPLE 500000 ROWS (reservoir)
""").df()
fit_df["temperature_2m_sq"] = fit_df["temperature_2m"] ** 2
for b in BOOLEAN:
    fit_df[b] = fit_df[b].astype("float64")
for c in CATEGORICAL:
    fit_df[c] = fit_df[c].astype(str)

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])
cat_pipe = OneHotEncoder(handle_unknown="ignore")
pre = ColumnTransformer([
    ("num", num_pipe, NUMERIC + BOOLEAN),
    ("cat", cat_pipe, CATEGORICAL),
])
pre.fit(fit_df[ALL_COLS])
print("fitted on", len(fit_df), "rows")

# small, readable, cross-split, class-balanced sample for the export
print("building the display sample (400 rows: 200 train, 100 val, 100 test; half bunched)...")
parts = []
for split, n in [("train", 200), ("val", 100), ("test", 100)]:
    for y_val, half in [(1, n // 2), (0, n - n // 2)]:
        # USING SAMPLE in the same SELECT as WHERE samples the raw table scan
        # BEFORE filtering (a DuckDB quirk, not standard SQL sample semantics)
        # - wrap the filtered query in a subquery so the sample is taken from
        # the filtered result instead, or a rare class returns ~0 rows.
        part = con.sql(f"""
            SELECT * FROM (
                SELECT date, LineNumber AS LineNumber_raw, direction_id AS direction_id_raw,
                       trip_id, stop_sequence, '{split}' AS split_label,
                       {", ".join(_FETCH_COLS)}, y_k{K} AS y
                FROM 'data/processed/model_dataset.parquet'
                WHERE split='{split}' AND y_k{K} = {y_val}
            ) USING SAMPLE {half} ROWS (reservoir)
        """).df()
        parts.append(part)
sample = pd.concat(parts, ignore_index=True)
sample["temperature_2m_sq"] = sample["temperature_2m"] ** 2

X = sample[ALL_COLS].copy()
for b in BOOLEAN:
    X[b] = X[b].astype("float64")
for c in CATEGORICAL:
    X[c] = X[c].astype(str)

encoded = pre.transform(X)
if hasattr(encoded, "toarray"):
    encoded = encoded.toarray()

num_feature_names = list(NUMERIC) + list(BOOLEAN)
indicator = pre.named_transformers_["num"].named_steps["impute"].indicator_
if indicator is not None and len(indicator.features_) > 0:
    src_names = NUMERIC + BOOLEAN
    num_feature_names += [f"{src_names[i]}__was_missing" for i in indicator.features_]
cat_feature_names = list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
encoded_names = num_feature_names + cat_feature_names

encoded_df = pd.DataFrame(encoded, columns=[f"enc__{n}" for n in encoded_names])

id_cols = sample[["date", "LineNumber_raw", "direction_id_raw", "trip_id", "stop_sequence", "split_label", "y"]].reset_index(drop=True)
out = pd.concat([id_cols, encoded_df], axis=1)
out.to_csv("data/processed/encoded_feature_sample_k3.csv", index=False)
print("wrote data/processed/encoded_feature_sample_k3.csv", out.shape)
print(out.head(3).T)
