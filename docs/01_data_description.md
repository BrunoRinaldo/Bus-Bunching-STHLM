# Data description

Course: AH2179 Applied AI in Transportation (KTH), final project, Option 1.
Scope: bus bunching prediction on 8 high-frequency SL bus lines, all of 2024.

This document describes the raw data, the inventory checks run against it (Phase
0 of the implementation plan), the line-selection decision that followed from
those checks, and the schema of the processed dataset that came out of the
pipeline.

---

## 1. Raw sources

| File | Rows | Size on disk | Grain |
|---|---|---|---|
| `gtfs_rt_2024{01..12}.csv.gz` | ~18.7M per month (all modes) | 300-350 MB each, gzipped | one row per (trip, stop) real-time observation |
| `stops.csv` | 21,661 | 1.3 MB | static GTFS stop |
| `routes.csv` | 653 | 21 KB | static GTFS route |
| `trips.csv` | 308,318 | 17 MB | static GTFS trip |
| `transfers.csv` | 40,614 | 2.8 MB | static GTFS transfer pair |
| `service_alerts_2024.parquet` | 1,972,779 | 28 MB | one row per (alert, informed entity) |
| `weather.parquet` (derived) | 79,056 | 0.5 MB | one row per (grid point, hour) |

The `gtfs_rt` files were read with DuckDB directly against the gzipped CSV
(`read_csv`), never fully materialised in pandas, per the plan's trap #6.

### 1.1 Schema notes that differ from the plan's assumption

The plan's column table was a close but not exact match to the real file. Two
corrections mattered enough to change the pipeline:

- **`LineNumber` is not purely numeric.** Values like `"185H"` and `"21L"`
  exist (branch/express suffixes), so DuckDB's type auto-detection guesses
  `BIGINT` from a small sample and then throws a cast error partway through
  the file. `LineNumber` must be read as `VARCHAR` explicitly.
- **`route_id` is not a stable per-line identifier.** ~2.6% of BUS rows carry
  a route-variant suffix, e.g. `9011001018500000-185V`, that does not appear
  in `routes.csv`. Grouping by raw `route_id` silently fragments a single
  physical line into several disconnected groups. See §3 for how this was
  handled.
- **The real file has more columns than documented**: `LineID`, `shape_id`
  (already present per-row, not only in `trips.csv`), `planned_departure_sfm`,
  `planned_departure_tripupdates_sfm`, `previous_StopAreaID`,
  `observed_StopAreaID`, `stop_headsign`. None of these were needed for this
  project's scope, but `shape_id` being present per-row is useful (it avoids
  a join to `trips.csv` for pattern identification, although this project
  ended up not needing pattern-level grouping at all - see §3).
- `transfers.csv` also carries `from_trip_id`/`to_trip_id` columns beyond the
  four documented in the plan; unused here.

### 1.2 `_sfm` overflow (plan trap #1) - confirmed present, handled

Checked on the January 2024 file: `scheduled_arrival_time_sfm` and
`observed_arrival_time_sfm` both exceed 86,400 (post-midnight service) in
roughly 0.6% of rows (103,406 and 113,026 rows out of 18,659,047 respectively;
departure columns similar). Event timestamps were reconstructed as
`service_midnight + INTERVAL sfm SECONDS`, which is correct for any positive
`sfm` value including values past 86,400 - this only breaks if code instead
takes `sfm % 86400`, which this pipeline never does.

### 1.3 Null rates, BUS rows, January 2024 (all ~536 lines, before line selection)

| Column | Null % | Note |
|---|---|---|
| `VehicleID` | 4.75 | |
| `observed_arrival_time_sfm` / `observed_arrival_delay` | 5.18 | no TripUpdate matched this stop |
| `observed_arrival_uncertainty` | 28.54 | |
| `observed_departure_time_sfm` / `observed_departure_delay` | 5.19 | |
| `observed_departure_uncertainty` | 19.22 | |
| everything else in the column set used | 0.0 - 0.07 | |

The two merge-quality flags (`merge_flag_1`, `merge_flag_2`) were **both false
for 100% of rows** in the final 8-line dataset. They flag TripUpdate-to-static
merge failures, not "no real-time update arrived for this stop" - the two are
different failure modes, and the flags are not a usable proxy for the
completeness problem described next. This contradicted the plan's assumption
that the merge flags could serve as a data-quality filter.

---

## 2. Phase 0 gate: line selection

### 2.1 First pass - frequency

From the January sample, scheduled peak-hour (07:00-09:00, 16:00-18:00)
headway was computed per line (median of the scheduled-pair diff). Candidate
lines with median peak headway <= 600 s, ranked by trip volume, included: 1,
4, 541, 3, 474, 607, 2, 117, 165, 179, 807, 172, 401, 61, 116, 54, 611, 176,
540 (partial list). The top 8 by trip volume were provisionally selected: 1,
4, 541, 3, 474, 607, 2, 117.

### 2.2 Second pass - completeness (the finding that changed the selection)

A full-year scan of the provisional 8 lines showed the null rate of
`observed_arrival_time_sfm` was **not uniform**:

| Line | Rows (year) | % missing observed arrival |
|---|---|---|
| 607 | 1,549,468 | 1.03 |
| 117 | 1,215,550 | 7.93 |
| 474 | 1,742,843 | 8.49 |
| 541 | 2,250,558 | 8.77 |
| 4 | 2,331,151 | 9.71 |
| 2 | 1,673,508 | 12.29 |
| **1** | 2,466,780 | **44.15** |
| **3** | 1,892,588 | **63.90** |

Lines 1 and 3 are unusable at this rate - roughly half to two-thirds of
their stop events have no matched real-time arrival, which would make any
headway built on them mostly fabricated interpolation. This is a genuine
data-quality finding, not a bug to route around (plan rule #1). Lines 1 and
3 were dropped and replaced with the next-best candidates by completeness
among lines still meeting the <= 600 s peak-headway bar:

| Line | Rows (year) | % missing observed arrival | Median peak headway |
|---|---|---|---|
| 401 | 1,176,786 | 9.37 | 480 s |
| 116 | 1,023,632 | 10.04 | 480 s |
| 179 | 1,393,874 | 7.52 | 480 s |

**Final 8 selected lines: 4, 116, 117, 179, 401, 474, 541, 607.** All have
median scheduled peak headway <= 600 s and full-year missing-observation
rates between 1% and 10%. Route context (from `routes.csv`): all are
`route_type` 700 (ordinary bus service), several tagged `blåbuss` (SL's
high-frequency trunk bus brand).

Also worth noting: the null rate is not stable over the year either - it
swings from under 1% in a data-quality-good stretch of September to spikes
above 50% on isolated days (e.g. a full outage on 2024-02-08, and several
other single-day spikes through autumn/winter). This volatility should be
kept in mind when interpreting month-to-month or season-to-season model
comparisons (RQ4): part of any seasonal effect could be feed reliability
rather than a real operational difference, and this should be flagged
explicitly in the report rather than absorbed silently into the weather
story.

### 2.3 Other Phase 0 numbers

- January BUS-mode rows: 16,443,319 of 18,659,047 total (88.1%); remaining
  modes are METRO, TRAM, TRAIN, SHIP.
- `scheduled_arrival_time_sfm` diffs, once grouped as in Phase 2, land on
  clean timetable values: the ten most common `headway_sched` values in the
  final dataset are 900, 600, 420, 480, 1800, 300, 360, 1200, 540, 840
  seconds - all round numbers, confirming the schedule data is clean.
- Distinct `route_id` values in January BUS data: 538 (close to, but not
  equal to, 536 distinct `LineNumber` values - consistent with the
  route-variant-suffix behaviour described in §1.1).

---

## 3. Deviation from the plan: grouping key for headway

The plan's normative headway definition groups leader/follower pairs by
`(date, route_id, direction_id, observed_stop_id)`. Given the `route_id`
variant-suffix fragmentation found in §1.1, this was changed to
`(date, LineNumber, direction_id, observed_stop_id)`.

Rationale: a waiting passenger experiences "the 4 bus", not "the 4 bus on
shape variant 9011001000400000-4B". Grouping by `route_id` would silently
split one physical line's vehicle stream into several disconnected
leader/follower chains at the exact stops where different route variants
converge - which is precisely where overtaking and bunching are most
interesting. `observed_stop_id` still restricts each group to the trips
that actually call at that stop, so variants that skip a stop are still
handled correctly. This is documented here because it is a deviation from
literal plan instructions, made for a stated, checkable reason.

---

## 4. Processed dataset: what was built

```
data/processed/
├── bus_subset.parquet          240 MB   12,683,862 rows  Phase 1 output
├── headways.parquet            928 MB   12,683,862 rows  Phase 2 output
├── weather.parquet             0.5 MB      79,056 rows   hourly, 9 grid points, full 2024
├── stop_route_counts.parquet   4 KB          377 rows    routes-per-stop (Jan sample, BUS only)
├── model_dataset.parquet       704 MB   12,683,862 rows  Phase 4 output - THE MODELING TABLE
└── model_dataset_sample.csv     83 MB      200,000 rows  random sample of the above, for quick loading
```

`model_dataset.parquet` is the file to use for independent modeling. It has
one row per (date, line, direction, trip, stop) observation, with five target
columns (one per prediction horizon) and the full feature set described in
`docs/02_data_usage_and_feature_engineering.md`.

### 4.1 Column reference

| Column | Type | Meaning |
|---|---|---|
| `date` | int (YYYYMMDD) | Service date |
| `LineNumber` | string | Public line number |
| `direction_id` | int | 0/1 |
| `trip_id`, `VehicleID` | int | Identifiers |
| `observed_stop_id` | int | Join key to `stops.csv` |
| `stop_sequence`, `first_seq`, `last_seq` | int | Position within the trip |
| `dist_traveled` | m | Segment length |
| `event_ts_arr` | timestamp | Local naive observed arrival time |
| `hour_of_day`, `day_of_week`, `month_of_year` | int | From the **scheduled** arrival time (always populated, unlike observed) |
| `headway_ratio` | double | observed / scheduled headway at stop n, scheduled-pair definition. Can be negative (the follower overtook its nominal leader); valid range is headway_obs in [-1800, 10800] seconds, see doc 2 |
| `is_bunched_now` | bool | `abs(headway_ratio) < 0.25` at stop n - the corrected, sign-agnostic bunching label, see doc 2 |
| `observed_arrival_delay`, `observed_departure_delay` | s | |
| `dwell`, `dwell_rel_median` | s | Boarding time at n, and relative to that stop's median |
| `speed_mps`, `speed_lag1`, `speed_lag2` | m/s | Segment running speed, current and previous 2 |
| `leader_trip_id`, `leader_dwell`, `leader_arrival_delay`, `leader_delay_growth` | | The scheduled-predecessor trip's own values at stop n |
| `route_frac` | double | `stop_sequence` as a fraction of the trip's stop range |
| `headway_ratio_lag1/2/3` | double | Trajectory: headway ratio at n-1, n-2, n-3 |
| `closing_speed_3` | double | `(headway_ratio - headway_ratio_lag3) / 3` |
| `cum_delay_growth_3` | double | Sum of delay growth over the last 3 stops |
| `y_k1, y_k2, y_k3, y_k5, y_k8` | 0/1 | Target: is the pair bunched (`abs(headway_ratio) < 0.25`, see doc 2) at stop n+k |
| `target_valid_k*` | bool | Whether the trip actually reaches stop n+k - filter on this before scoring, do not treat missing `y_k*` as a negative |
| `grid_lat`, `grid_lon` | double | Weather grid cell for this stop |
| `temperature_2m`, `precipitation`, `is_precip`, `precip_3h`, `is_snowing`, `ice_risk`, `temp_bin`, `wx_group`, `wind_speed_10m` | | Weather at the hour of `event_ts_arr`, converted UTC->local correctly across DST (see doc 2) |
| `n_routes_serving` | int | Distinct bus lines seen at this stop in the January sample (system-wide, not just the 8 selected lines) - a static proxy, see limitation below |
| `is_interchange`, `has_parent_station` | bool | From `transfers.csv` / `stops.csv` |
| `under_active_alert` | bool | A service alert with this `stop_id` was active at `event_ts_arr` |
| `split` | string | `train` (Jan-Sep), `val` (Oct), `test` (Nov-Dec) - assigned from `date`, never from a derived timestamp |

### 4.2 Known limitations of this table, stated plainly

- **13.1% of rows have a null `headway_obs`** (and therefore null
  `headway_ratio`), inherited directly from the ~8-10% per-row missing
  real-time match described in §2.2 (a null propagates to both the row
  itself and the following row's diff). Any model trained on `headway_ratio`
  as a feature must handle this null rather than silently dropping to zero.
- **Weather is null for 7.9% of rows** - exactly the rows where
  `event_ts_arr` itself is null, since the weather join key is derived from
  it. This is not a join bug; it is the same missing-observation rate
  propagating downstream, confirmed by the fact that the two percentages
  match to within rounding.
- **`n_routes_serving` is a January-only, system-wide static approximation.**
  No `stop_times.txt` was provided (only the RT trip-stop files), so there is
  no authoritative year-round routes-per-stop count. Treat this column as a
  coarse "how major is this stop" signal, not a precise count.
- **`under_active_alert` assumes alert `start`/`end` timestamps are in the
  same local-naive convention as the RT feed.** This was not independently
  verified against a documented timezone for the alerts file (unlike the
  weather feed, whose UTC convention is documented by Open-Meteo). If this
  flag looks wrong in EDA, check this assumption first.
- Only 8 of ~536 bus lines are covered. This was a deliberate scope decision
  (Phase 1 of the plan), not a limitation discovered late - but it means
  RQ4 (generalisation) can only be tested by holding out lines *within* this
  set of 8, which are all trunk/high-frequency lines by construction. Held-out
  lines will not represent SL's lower-frequency network.
