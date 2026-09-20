# Implementation plan: bus bunching prediction from SL GTFS-RT data

**Course:** AH2179 Applied AI in Transportation (KTH), final project, Option 1
**Audience:** implementing agent
**Owner:** Bruno

---

## 0. Read this first

You are implementing an end-to-end ML project on Stockholm public transport data. The project is graded on innovation (20 %), quality of analysis (30 %), explanation and discussion of results (30 %), and report structure and writing (20 %).

Three rules that override convenience:

1. **Do not fabricate or paper over results.** If a validation check fails, stop and report it. A negative finding is a valid outcome for this project; a fabricated positive one is not.
2. **Two phases are hard gates** (Phase 0 and Phase 2). Do not proceed past them without reporting the numbers back to Bruno for a go/no-go.
3. **No em dashes in any written output.** Use en dashes (–).

---

## 1. Research question

> **How early can bus bunching be predicted from real-time operational data alone, and is that lead time long enough to be operationally actionable?**

Sub-questions:

- **RQ1 (descriptive):** Where and when does headway irregularity form, and does it grow systematically along a route?
- **RQ2 (mechanism):** Does dwell-time growth predict downstream bunching better than arrival delay does?
- **RQ3 (core):** How does predictive performance decay as the prediction horizon *k* increases?
- **RQ4 (generalisation):** Does a model trained on one line or one season transfer to another?

Hypotheses, to be stated in the report and explicitly tested:

- **H1.** Headway coefficient of variation increases monotonically with `stop_sequence`.
- **H2.** Dwell-time growth features outperform delay-only features for predicting bunching at horizon k ≥ 3.
- **H3.** Predictive performance decays with k, with usable performance (PR-AUC meaningfully above the persistence baseline) up to roughly k = 3–5 stops.
- **H4.** Adverse weather raises the base rate of bunching but does not materially change which features are predictive.

### Background mechanism (for the report introduction)

Bunching is a self-reinforcing instability, not random failure. A vehicle that falls slightly behind schedule encounters more accumulated passengers at the next stop, which lengthens its dwell time, which puts it further behind. The vehicle following it then faces recently-cleared stops, dwells less, and closes the gap. Headway variance amplifies along the route. See Newell & Potts (1964) and Daganzo (2009, *Transportation Research Part B*, headway-based holding control). The practical consequence for modelling: **dwell time is the causal variable, not merely lateness.**

---

## 2. Data

Location: downloaded from `https://datarepository.kth.se/projectoption1`. Confirm with Bruno where the files are on disk before starting.

### 2.1 `gtfs_rt_YYYY_MM.csv.gz` – 12 files, one per month, 2024

One row per (trip, stop) observation: scheduled stop time from static GTFS merged with the observed arrival/departure from GTFS-RT TripUpdates.

Columns you will use:

| Column | Type | Use in this project |
|---|---|---|
| `date` | date | Service date. Grouping key. |
| `LineNumber` | string | Public line number ("4", "172"). Use for plot labels. |
| `route_id` | string | Grouping key, join to `routes.txt`. |
| `TransportMode` | string | Filter to `BUS`. |
| `direction_id` | int (0/1) | Grouping key. |
| `trip_id` | int | Identifies the vehicle run. Leader/follower pairing. |
| `VehicleID` | int | Confirms physically distinct vehicles; detects short-turns. |
| `stop_sequence` | int | Position along route. Ordering within a trip. |
| `dist_traveled` | m | Segment length from previous stop. Enables speed. |
| `previous_stop_id` | int | Segment identification. |
| `observed_stop_id` | int | Grouping key, join to `stops.txt`. |
| `scheduled_arrival_time_sfm` | s | Scheduled headway, benchmark. |
| `observed_arrival_time_sfm` | s | **Core variable.** Observed headway. |
| `observed_arrival_delay` | s | Feature (+ late / − early). |
| `observed_arrival_uncertainty` | int | Data-quality weighting / filtering. |
| `scheduled_departure_time_sfm` | s | Scheduled dwell. |
| `observed_departure_time_sfm` | s | **Core variable.** Dwell time. |
| `observed_departure_delay` | s | Feature; with arrival delay gives delay growth. |
| `observed_departure_uncertainty` | int | Data-quality. |
| `tripupdates_and_gtfs_not_merged_within_service_id` | bool | 0 = clean merge. Quality filter. |
| `tripupdates_not_merged_with_stoptimes_or_trip` | bool | 0 = clean merge. Quality filter. |

**`_sfm` = seconds from midnight of the service date and can exceed 86400 for post-midnight trips.** This is trap #1. Verify before any arithmetic.

### 2.2 Static GTFS (4 CSVs)

- `stops.txt` – `stop_id`, `stop_name`, `stop_lat`, `stop_lon`, `location_type`, `parent_station`, `platform_code`. Used for mapping, weather joins, and collapsing platforms to stations.
- `routes.txt` – `route_id`, `route_short_name`, `route_long_name`, `route_type`, `route_desc`. Bus route types are **700** (Bus Service) and **714** (Rail Replacement Bus). Use for mode filtering and readable labels.
- `trips.txt` – `route_id`, `trip_id`, `trip_headsign`, `direction_id`, `shape_id`. **`shape_id` identifies route variants and short-turn patterns**, which otherwise corrupt headway calculations.
- `transfers.txt` – `from_stop_id`, `to_stop_id`, `transfer_type`, `min_transfer_time`. Used only to derive an `is_interchange` binary feature.

Note: `shapes.txt` is **not** included. No route geometry is available. Do not plan features that require it.

### 2.3 `service_alerts_2024.parquet`

1,972,779 rows, one per (alert × informed entity). Fields include `id`, `start`, `end`, `cause` (GTFS-RT Alert.Cause enum), `description_text2`, `header_text2` (Swedish), `route_id`, `trip_id`, `stop_id`, `schedule_relationship`, `date`.

**Scope limit: this project does not model the alerts.** They are used only to construct a binary `under_active_alert` flag on trip-stop observations, so that endogenous bunching (the feedback loop) can be separated from exogenous shocks. No text processing. No NLP.

### 2.4 Weather – Open-Meteo Historical Archive API

**Source:** `https://archive-api.open-meteo.com/v1/archive`. Free, no API key, ERA5-based reanalysis, full 2024 coverage at hourly resolution. **This replaces the SLB source named in the assignment brief. Do not use SLB.**

```
pip install openmeteo-requests requests-cache retry-requests numpy pandas
```

Weather is a core feature group, not an optional extra, and belongs in Phase 4 alongside everything else. The mechanism matters: precipitation slows boarding, low temperature means heavier clothing and icy pavements, and both inflate dwell time, which is the variable driving the bunching feedback loop. Weather should therefore raise the base rate of bunching (H4).

#### Do not query per stop

The archive is gridded reanalysis at roughly 10–25 km resolution, so every stop inside one grid cell returns identical values. Thousands of per-stop calls would return duplicate data and burn the rate limit for nothing.

1. Snap stops to a coarse grid: `grid_lat = round(stop_lat, 1)`, `grid_lon = round(stop_lon, 1)`. 0.1° is roughly 11 km, which matches the underlying resolution.
2. Take the unique grid points. For 5–10 inner-city lines expect fewer than 10; for the whole SL region, on the order of 10–30.
3. One call per grid point covers the entire year.
4. Persist to `data/processed/weather.parquet` and never call the API again. `requests_cache` with `expire_after=-1` caches permanently, but the parquet is what makes the pipeline reproducible offline.

#### Reference implementation

```python
import openmeteo_requests, requests_cache, pandas as pd
from retry_requests import retry

session = retry(requests_cache.CachedSession(".cache", expire_after=-1),
                retries=5, backoff_factor=0.2)
om = openmeteo_requests.Client(session=session)

# Order matters: Variables(i) is indexed by position in this list.
HOURLY = ["temperature_2m", "precipitation", "rain", "snowfall",
          "snow_depth", "weather_code", "wind_speed_10m", "cloud_cover"]

def fetch_grid_point(lat, lon):
    r = om.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params={"latitude": lat, "longitude": lon,
                "start_date": "2024-01-01", "end_date": "2024-12-31",
                "hourly": HOURLY, "wind_speed_unit": "ms"},
    )[0]
    h = r.Hourly()
    out = {"ts_utc": pd.date_range(
        start=pd.to_datetime(h.Time(), unit="s", utc=True),
        end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=h.Interval()), inclusive="left")}
    for i, name in enumerate(HOURLY):
        out[name] = h.Variables(i).ValuesAsNumpy()
    df = pd.DataFrame(out)
    df["grid_lat"], df["grid_lon"] = lat, lon
    return df
```

Build the weather table once by concatenating `fetch_grid_point` over the unique grid points, then join to trip-stop observations on `(grid_lat, grid_lon, ts_utc)`.

#### Timezone, and this is trap #9

The API returns **UTC**. The SL data is in local service dates with seconds-from-midnight. Getting this join wrong shifts weather by one or two hours, which will quietly destroy any weather signal and make H4 look false.

Adding a timedelta to a timezone-aware timestamp does absolute-time arithmetic, so on daylight-saving days it lands on the wrong local hour. Do the arithmetic in naive local time first, then localise:

```python
naive_local = (pd.to_datetime(df["date"])
               + pd.to_timedelta(df["observed_arrival_time_sfm"], unit="s"))
local = naive_local.dt.tz_localize("Europe/Stockholm",
                                   ambiguous="NaT", nonexistent="NaT")
df["ts_utc"] = local.dt.tz_convert("UTC").dt.floor("h")
```

DST transitions in 2024 were **31 March** and **27 October**. The October one falls inside the validation split, so handle it rather than dropping it silently. Report how many rows come back as NaT; it should be a handful, not thousands.

Assert after joining: the weather join must not drop rows, and `ts_utc` must have full hourly coverage for every service date.

#### Derived features, which matter more than the raw values

| Feature | Definition | Rationale |
|---|---|---|
| `is_precip` | `precipitation > 0.1` mm | Binary is more robust than a heavily skewed continuous variable |
| `precip_3h` | Rolling 3 h sum of `precipitation` | Road conditions depend on accumulation, not the current hour |
| `is_snowing` | `snowfall > 0` | Snowfall rate disrupts running speed; `snow_depth` is standing snow and a different signal |
| `ice_risk` | `temperature_2m < 1` and `precipitation > 0` | Freezing rain is the worst case for boarding times |
| `temp_bin` | Bins at −5, 0, 5, 15, 25 °C | Non-linear effect; boarding slows at both extremes |
| `wx_group` | Grouped `weather_code` | See below |

`weather_code` is a WMO code with roughly 30 distinct values. Do not one-hot encode it raw. Group it: 0 clear; 1–3 cloudy; 45, 48 fog; 51–57 drizzle; 61–67 rain; 71–77 snow; 80–82 rain showers; 85–86 snow showers; 95–99 thunderstorm.

#### No leakage

Weather is exogenous and, in a real deployment, available as a forecast at prediction time. Using it introduces no leakage and is operationally legitimate. State this explicitly in the report, because an examiner will ask.

---

## 3. Deliverables

```
repo/
├── README.md                      # setup, how to reproduce, results summary
├── requirements.txt
├── config.yaml                    # paths, selected lines, thresholds, seeds
├── data/                          # gitignored
│   ├── raw/
│   └── processed/                 # parquet subsets + weather.parquet
├── src/
│   ├── ingest.py                  # DuckDB scan → filtered parquet
│   ├── weather.py                 # Open-Meteo fetch, grid snap, UTC join
│   ├── headways.py                # headway/dwell/label construction
│   ├── features.py                # feature engineering, leakage-safe
│   ├── models.py                  # baseline, logreg, GBM, Keras
│   └── evaluate.py                # metrics, diagnostics, figures
├── notebooks/
│   ├── 01_inventory.ipynb         # Phase 0 gate checks
│   ├── 02_headway_validation.ipynb# Phase 2 gate checks
│   ├── 03_eda.ipynb               # descriptive analysis (report task A)
│   ├── 04_features.ipynb
│   ├── 05_models.ipynb
│   ├── 06_diagnostics.ipynb
│   └── 07_report_assets.ipynb
├── models/
│   └── best_model.keras           # REQUIRED by the assignment
├── figures/
└── reports/
    └── results_tables.md
```

Plus: a results summary that maps directly onto report sections A–F of the assignment brief.

---

## 4. Phase plan

### Phase 0 – Inventory and feasibility gate ⛔ HARD GATE

Load **one** month only. Do not touch all 12 yet.

Produce answers to exactly these five questions:

1. `max()` and the distribution above 86400 of all four `*_time_sfm` columns. How many rows wrap past midnight?
2. Distribution of scheduled headway after the grouping in Phase 2. Is it clean (300/600/900 s) or noisy?
3. What fraction of bus routes have a median peak scheduled headway ≤ 600 s? List the top 20 candidate lines by trip volume.
4. Share of rows where either merge-quality flag is 1.
5. Observed trips versus scheduled trips per route-day. What is the completeness rate?

Also report: row count per month, file size, memory footprint, null rates on every column in §2.1.

**Stop here and report. Do not proceed without a go.** These numbers determine whether the project is viable and which lines are in scope.

### Phase 1 – Ingestion

Use DuckDB to scan the gzipped CSVs directly rather than loading them into pandas. Select only needed columns and filter early.

```sql
COPY (
  SELECT date, LineNumber, route_id, TransportMode, direction_id,
         trip_id, VehicleID, stop_sequence, dist_traveled,
         previous_stop_id, observed_stop_id,
         scheduled_arrival_time_sfm, observed_arrival_time_sfm,
         observed_arrival_delay, observed_arrival_uncertainty,
         scheduled_departure_time_sfm, observed_departure_time_sfm,
         observed_departure_delay, observed_departure_uncertainty,
         tripupdates_and_gtfs_not_merged_within_service_id AS merge_flag_1,
         tripupdates_not_merged_with_stoptimes_or_trip     AS merge_flag_2
  FROM read_csv_auto('data/raw/gtfs_rt_2024_*.csv.gz', union_by_name=true)
  WHERE TransportMode = 'BUS'
    AND route_id IN (<selected routes from Phase 0>)
) TO 'data/processed/bus_subset.parquet' (FORMAT PARQUET);
```

Scope: 5–10 high-frequency bus lines, **all 12 months**. Seasonal coverage matters more than breadth, because RQ4 depends on it.

Build the pipeline on one month first, then re-run across all 12.

**Acceptance:** parquet written, row count reported, round-trip check that a sample of rows matches the source CSV.

### Phase 2 – Derived variables and validation gate ⛔ HARD GATE

All definitions below are normative. Implement them exactly.

**Event timestamp.** Construct `event_ts = date_at_midnight_local + observed_arrival_time_sfm seconds`. This handles the post-midnight wrap correctly *provided* sfm exceeds 86400 rather than resetting. Confirm this in Phase 0 before relying on it.

**Dwell time.**
```python
dwell = observed_departure_time_sfm - observed_arrival_time_sfm
```
Clip negatives to null and flag them. Exclude the first and last `stop_sequence` of every trip: dwell there is driver layover and schedule recovery, not boarding.

**Delay growth.**
```python
delay_growth = observed_departure_delay - observed_arrival_delay
```
Time lost while stationary. This is the feedback loop made observable.

**Segment running time and speed.** Within a trip, ordered by `stop_sequence`:
```python
run_time = observed_arrival_time_sfm - observed_departure_time_sfm.shift(1)
speed    = dist_traveled / run_time          # m/s; guard run_time <= 0
```

**Headway – use the scheduled-pair definition as primary.**

Two definitions exist and they diverge exactly when bunching occurs, because vehicles overtake:

- *(a) observed-order:* sort by `observed_arrival_time_sfm` within `(date, route_id, direction_id, observed_stop_id)` and diff.
- *(b) scheduled-pair:* the leader is the trip that was **scheduled** to precede this one at this stop; the headway is the observed gap between that pair.

Use **(b) as primary**, because it is well defined under overtaking and gives a stable leader/follower pair identity across stops, which the horizon target requires. Compute (a) as a cross-check.

```python
g = ["date", "route_id", "direction_id", "observed_stop_id"]
df = df.sort_values(g + ["scheduled_arrival_time_sfm"])

df["leader_trip_id"]  = df.groupby(g)["trip_id"].shift(1)
df["headway_sched"]   = df.groupby(g)["scheduled_arrival_time_sfm"].diff()
df["headway_obs"]     = df.groupby(g)["observed_arrival_time_sfm"].diff()
df["headway_ratio"]   = df["headway_obs"] / df["headway_sched"]
```

**Bunching label.**
```python
is_bunched = headway_ratio < 0.25      # primary threshold
is_gap     = headway_ratio > 1.75      # the paired hole behind every bunch
```
Run a sensitivity analysis over thresholds {0.20, 0.25, 0.33, 0.50} and report it. Do not treat 0.25 as sacred.

**Headway coefficient of variation.** Per `(route_id, direction_id, observed_stop_id, hour, month)`:
```python
cv = headway_obs.std() / headway_obs.mean()
```

**Excess waiting time.** For random passenger arrivals:

E[W] = (E[h] / 2) × (1 + CV²)

Report EWT observed versus EWT implied by the timetable (CV = 0). This converts the technical result into passenger-minutes and is what section E needs.

**Validation checks to report before proceeding:**

- `headway_sched` distribution: are the modes at clean timetable values?
- Fraction of null `headway_obs` (first trip of each group is legitimately null).
- Fraction of negative or absurd headways (> 3 h, < 0). These indicate missing trips or the midnight bug.
- Base rate of `is_bunched` per line. Expect roughly 3–8 %. If it is 0.1 % or 40 %, something is wrong.
- Comparison of definition (a) versus (b): how often do they disagree, and is the disagreement concentrated in bunched observations as expected?

**Stop here and report. Do not proceed without a go.**

### Phase 3 – Descriptive analysis (report task A)

Deliver these figures:

1. **CV against `stop_sequence`**, one line per route. This is the headline figure and the test of H1.
2. Bunching rate heatmap: hour of day × line.
3. Bunching rate by month, with a winter/summer annotation.
4. Spatial map of bunching hotspots using `stop_lat`/`stop_lon`.
5. Headway ratio distribution, overall and split by peak/off-peak.
6. Dwell time distribution by stop type (interchange vs ordinary, from `transfers.txt`).
7. Delay growth against headway ratio, to show the feedback loop directly.

Load the `dataviz` skill before writing any plotting code.

Each figure needs a one-paragraph interpretation. Figures without interpretation earn nothing under the 30 % "results explanations" criterion.

### Phase 4 – Target and features

**Unit of observation:** one row per (date, route, direction, follower trip, stop *n*), for each horizon *k*.

**Target:** `y = 1` if the same leader/follower pair has `headway_ratio < 0.25` at stop *n + k*. Require both trips to actually serve stop *n + k*; drop the row otherwise. Build targets for k ∈ {1, 2, 3, 5, 8}.

**Leakage rule, non-negotiable:** every feature must be computable from information available at stop *n* or earlier. Build features and targets in two separate passes and add an assertion that no feature references a `stop_sequence` greater than *n*. Groupby-shift operations make this easy to get wrong silently.

Feature set:

| Group | Features |
|---|---|
| Current state | `headway_ratio` at n; `observed_arrival_delay`; `observed_departure_delay`; `dwell`; `dwell` relative to that stop's median dwell |
| Trajectory (the strongest group) | `headway_ratio` at n−1, n−2, n−3; **rate of change of `headway_ratio` over the last 3 stops** (closing speed); cumulative `delay_growth` over last 3 stops; recent segment speeds |
| Leader | leader's `dwell`; leader's `observed_arrival_delay`; leader's `delay_growth` |
| Position | `stop_sequence`; `stop_sequence` as fraction of route length; `dist_traveled` |
| Context | hour of day; day of week; month; `direction_id`; `route_id` (categorical) |
| Stop attributes | `is_interchange` (from `transfers.txt`); number of routes serving the stop; `parent_station` flag |
| Disruption | `under_active_alert` flag (from service alerts join on route/stop × time window) |
| Weather (see §2.4) | `is_precip`; `precip_3h`; `is_snowing`; `ice_risk`; `temp_bin`; `wx_group`; `wind_speed_10m`; raw `temperature_2m` and `precipitation` |

Expect the closing-speed feature to dominate. If it does not, investigate before accepting the result.

### Phase 5 – Models

Four models. All four appear in the comparison table.

1. **Persistence baseline.** Assume `headway_ratio` at n+k equals `headway_ratio` at n. **Non-negotiable and implemented first.** If the learned models do not beat this, that is the finding, and it needs to surface in week two rather than week six.
2. **Logistic regression** on the engineered features. Interpretable coefficients carry the mechanism story for RQ2.
3. **Gradient boosting.** `HistGradientBoostingClassifier` from sklearn (no extra install) or LightGBM. Usually the strongest on tabular data. Feature importances feed the discussion.
4. **Keras sequence model.** 1D-CNN or LSTM over the sequence of (headway_ratio, dwell, delay_growth) across stops n−5 … n. **This is the `.keras` export required by assignment task F.** Train and export it regardless of whether it wins, and report the comparison honestly.

**Splits.**

- *Temporal (primary):* train Jan–Sep, validate Oct, test Nov–Dec. Never split randomly. Leakage through near-identical adjacent observations is otherwise guaranteed, and a winter test set makes generalisation honest.
- *Route holdout (for RQ4):* train on a subset of lines, test on held-out lines.

**Class imbalance.** Bunching is rare. Use class weights, not resampling, and never resample the test set.

**Metrics.** PR-AUC as primary (accuracy is meaningless at a 5 % base rate), plus F1, recall at fixed precision, confusion matrix, and a calibration plot. Report the operational framing explicitly: *at 80 % recall, what is the false alarm rate?* A controller ignores a system that cries wolf, so the precision-recall tradeoff **is** the deployment discussion.

**The headline output of the whole project:** PR-AUC against horizon *k*, all models plus baseline on one axis. Convert *k* into minutes using median inter-stop running time, so the x-axis reads as lead time.

### Phase 6 – Diagnostics (report task D)

- Performance broken out by line, hour of day, season, weather.
- **Ablation for H2:** delay-only features versus dwell-only features versus both. This is the mechanism test and the most interesting single result.
- **Robustness:** drop the leader's features entirely, simulating a real-time feed that lost a vehicle. Report the degradation.
- **Transfer:** train on line 4, test on line 1, and the route-holdout split generally.
- **Scalability:** inference latency per prediction, and whether the feature pipeline could run in real time.
- Where does the model fail? Characterise the false negatives.

### Phase 7 – Report assets

Produce tables and figures mapped to assignment sections A–F, and a `results_tables.md` with every number the report needs. Do not write the report itself unless Bruno asks.

For deployment (section E): frame the recommendation as a **holding trigger at designated control points**, with the EWT calculation giving an upper bound on the benefit. Do not overclaim. Passenger response cannot be simulated without a demand model, so this is decision support, not a validated control policy.

---

## 5. Known traps

| # | Trap | Mitigation |
|---|---|---|
| 1 | `_sfm` exceeding 86400 for post-midnight service; naive diff gives headways of −23 h | Check `max()` in Phase 0; build proper timestamps |
| 2 | Missing trips create phantom headways between vehicle *i* and *i−2* | Completeness check against the static timetable; report the rate; use merge flags |
| 3 | Terminal dwell is layover, not boarding, and will dominate the dwell distribution | Exclude first and last `stop_sequence` of each trip |
| 4 | Route variants and short-turns: not every trip serves every stop | Use `shape_id` to identify patterns; grouping by `observed_stop_id` handles most of it |
| 5 | Leakage via groupby-shift when building horizon targets | Separate passes for features and targets, plus an explicit assertion |
| 6 | File size: 12 months of the region will not fit comfortably in pandas | DuckDB scan with column selection and early filtering, write parquet once |
| 7 | Class imbalance making accuracy look excellent and meaningless | PR-AUC primary; never report bare accuracy |
| 8 | Overtaking making observed-order headways ill-defined exactly where it matters | Scheduled-pair definition (b) as primary |
| 9 | Weather joined on the wrong hour: API returns UTC, SL data is local, and tz-aware timedelta arithmetic breaks on DST days | Naive-local arithmetic then `tz_localize` with `ambiguous`/`nonexistent` handling; see §2.4 |

---

## 6. Non-goals

Do not:

- Do any text processing or NLP on the service alerts. They are a binary flag only.
- Make causal claims. Associations, clearly framed.
- Build a simulation of holding control. Out of scope and not defensible without demand data.
- Use route geometry. `shapes.txt` is not in the dataset.
- Use SLB weather data. Open-Meteo replaces it (§2.4).
- Query the weather API per stop. Snap to a grid first.
- Expand beyond the selected lines without asking. Scope creep will sink the timeline.

---

## 7. Reporting back

At each gate and at the end of each phase, report:

1. What was run and on what subset of the data.
2. The numbers, in a table.
3. Anything that contradicted an expectation stated in this plan.
4. A recommendation: proceed, adjust, or stop.

Flag immediately, without working around it, if:

- Any Phase 0 or Phase 2 validation check fails.
- The bunching base rate falls outside 1–15 %.
- No model beats the persistence baseline at any horizon.
- Data completeness falls below 90 %.

Any of these changes the project rather than being a bug to route around.
