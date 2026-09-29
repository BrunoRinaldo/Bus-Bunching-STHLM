# Engineered features

This document lists every variable this project builds itself, as opposed to
columns that come straight from the SL GTFS-RT feed (`observed_arrival_delay`,
`stop_sequence`, `dist_traveled`, ...). For each one it gives the definition,
the reasoning behind it, and the script that produces it.

The features are built in three steps:

1. **`src/headways.py`** turns raw stop events into base variables: dwell,
   delay growth, speed, headway and the bunching label.
2. **`src/weather.py`** fetches hourly weather and derives the weather flags.
3. **`src/features.py`** builds the model features on top of those, plus the
   prediction targets.

Every model feature is computed from stop *n* or earlier. The targets are the
only columns that look ahead. See [Leakage safety](#leakage-safety) at the
end.

---

## 1. Base variables (`src/headways.py`)

### Dwell, delay growth and speed

| Feature | Definition | Why |
|---|---|---|
| `dwell` | `observed_departure_time - observed_arrival_time` at the stop, in seconds | Time spent boarding and alighting, the core of the bunching mechanism: a bus that dwells longer falls further behind. Set to null at the first and last stop of a trip (terminal layover, not boarding) and when negative (bad timestamps). |
| `dwell_negative_flag` | `true` when the raw dwell is negative | Kept so data-quality problems can be counted without polluting `dwell`. |
| `delay_growth` | `observed_departure_delay - observed_arrival_delay` | How much delay the bus *added* at this stop, as opposed to delay carried in from earlier. |
| `run_time` | arrival at this stop minus departure from the previous stop of the same trip | Travel time on the link into this stop. |
| `speed_mps` | `dist_traveled / run_time`, null if `run_time <= 0` | Link speed in m/s; captures congestion between stops. |

### Headway

Headways are computed per `(date, LineNumber, direction_id, stop)`, with
trips ordered by **scheduled** arrival. Each trip is compared with the trip
that was scheduled just before it (its *leader*).

| Feature | Definition | Why |
|---|---|---|
| `leader_trip_id` | the scheduled predecessor trip at this stop | Fixes the leader/follower pair, so the same two buses can be followed from stop to stop. |
| `headway_sched` | follower's scheduled arrival minus leader's scheduled arrival | The planned gap. |
| `headway_obs` | follower's observed arrival minus leader's observed arrival | The actual gap. Negative when the follower overtook its leader. |
| `headway_obs_invalid` | `headway_obs` outside **[-1800, 10800] s** | Values beyond this range are almost certainly a broken pairing (a skipped trip in between), not a real gap. |
| `headway_ratio` | `headway_obs / headway_sched`, null when invalid or `headway_sched <= 0` | Normalises the gap across lines and times of day with different frequencies. 1 = on plan, 0 = buses together, negative = overtaken. |
| `headway_obs_orderdef`, `headway_ratio_obsorder` | same, but the leader is whoever *actually* arrived just before | An alternative definition, kept for comparison. It can never be negative. |

**Why group by `LineNumber` and not `route_id`:** `route_id` has a suffix per
route pattern, which splits one physical line into several groups and breaks
the leader/follower chain. Passengers experience the line, so the headway is
measured per line.

**Why the leader is the *scheduled* predecessor:** if the leader were the bus
that actually arrived just before, the pair would change every time buses
overtake each other. A fixed scheduled pair allows the multi-stop trajectory
features below.

### Labels

| Feature | Definition | Why |
|---|---|---|
| `is_bunched` | `abs(headway_ratio) < 0.25` | Buses are within a quarter of the planned gap of each other. `abs()` because a follower a few seconds *ahead* of its leader is just as bunched as one a few seconds behind. |
| `is_gap` | `headway_ratio > 1.75` | The opposite failure: a long wait. |
| `hour_of_day`, `day_of_week`, `month_of_year` | taken from the **scheduled** arrival time | The scheduled time always exists; the observed time is missing for ~8% of rows. |

---

## 2. Model features (`src/features.py`)

All windows are `PARTITION BY date, trip_id ORDER BY stop_sequence`, that is,
along one trip.

### Current state at stop n

| Feature | Definition | Why |
|---|---|---|
| `is_bunched_now` | `is_bunched` at stop n | A bus that is already bunched usually stays bunched. |
| `dwell_rel_median` | `dwell - median(dwell)` at this stop over the whole year | Separates "this stop is always busy" from "this bus is dwelling unusually long here". |

### Trajectory (history of the same trip)

| Feature | Definition | Why |
|---|---|---|
| `headway_ratio_lag1/2/3` | `headway_ratio` at stops n-1, n-2, n-3 | Recent gap history. |
| `closing_speed_3` | `(headway_ratio - headway_ratio_lag3) / 3` | How fast the gap has been shrinking per stop over the last three stops. A pair that is closing quickly is heading for bunching even if it is not bunched yet. |
| `cum_delay_growth_3` | sum of `delay_growth` over stops n-2..n | Delay the bus has been adding recently, smoothed over three stops. |
| `speed_lag1/2` | `speed_mps` on the two previous links | Recent congestion. |

### Leader

| Feature | Definition | Why |
|---|---|---|
| `leader_dwell` | the leader's `dwell` at stop n | A slow leader lets the follower catch up. |
| `leader_arrival_delay` | the leader's arrival delay at stop n | Same, from the delay side. |
| `leader_delay_growth` | the leader's `delay_growth` at stop n | Whether the leader is currently losing time. |

These are the leader's values at the *same* stop n, which it has already
passed, so they are known when predicting for the follower.

### Position along the route

| Feature | Definition | Why |
|---|---|---|
| `first_seq`, `last_seq` | first and last observed `stop_sequence` of the trip | Used to normalise position. |
| `route_frac` | `(stop_sequence - first_seq) / (last_seq - first_seq)` | Position as a share of the trip (0 = start, 1 = end), comparable across lines of different lengths. Bunching builds up along the route: pooled across all lines, it rises from 1.2% in the first fifth to 3.6% in the last (see `figures/fig10_bunching_along_route.png`). |

### Stop attributes

| Feature | Definition | Why |
|---|---|---|
| `is_interchange` | stop appears in `transfers.csv` | Transfer stops have heavier, burstier boarding. |
| `has_parent_station` | stop has a `parent_station` in `stops.csv` | Marks larger stop complexes. |
| `n_routes_serving` | number of distinct bus lines seen at the stop (January sample, all SL bus lines) | Stop busyness and competing traffic at the curb. A static approximation. |

### Disruption

| Feature | Definition | Why |
|---|---|---|
| `under_active_alert` | a service alert is active for this stop at the arrival time | Diversions and incidents disturb regularity. Only a yes/no flag; alert text is not used. |

### Weather (`src/weather.py`, joined in `features.py`)

Stops are snapped to a 0.1° grid (9 grid points cover all stops), hourly
weather is fetched once from the Open-Meteo archive, and each row is joined
on grid point and UTC hour (converted from Stockholm local time, so daylight
saving is handled).

| Feature | Definition | Why |
|---|---|---|
| `temperature_2m` | air temperature, °C | Cold and heat slow boarding. |
| `temperature_2m_sq` | `temperature_2m ** 2` (added in `src/models.py`) | Lets logistic regression fit a U-shaped temperature effect with one extra term. Replaces the earlier `temp_bin` bins. |
| `temp_bin` | temperature in bins (<-5, -5–0, 0–5, 5–15, 15–25, >25) | Still in the dataset, but no longer used by the models. |
| `precipitation` | mm in the hour | |
| `is_precip` | `precipitation > 0.1` | Rain on/off. |
| `precip_3h` | rolling 3-hour precipitation sum | Wet roads and crowding last beyond the hour it rains. |
| `is_snowing` | `snowfall > 0` | |
| `ice_risk` | temperature < 1 °C **and** precipitation > 0 | Slippery conditions: slower driving and boarding. |
| `wx_group` | WMO weather code collapsed to clear / cloudy / fog / drizzle / rain / snow / rain showers / snow showers / thunderstorm | A single categorical summary of conditions. |
| `wind_speed_10m` | m/s | |

Weather counts as forecastable at prediction time, so using it is not
leakage.

### Split

| Column | Definition |
|---|---|
| `split` | from `date`: January–September = `train`, October = `val`, November–December = `test` |

The split is taken from `date`, not from a timestamp, because the observed
timestamp is sometimes null and would silently put those rows in the wrong
split.

---

## 3. Targets

For horizon k ∈ {1, 2, 3, 5, 8} stops ahead:

| Column | Definition |
|---|---|
| `y_k{k}` | `is_bunched` at stop n+k of the same trip; null if that stop has no valid `headway_ratio` |
| `target_valid_k{k}` | trip actually reaches stop n+k |

`lead()` returns null when the trip ends before n+k (short turn, end of route),
so those rows drop out automatically.

## 4. Sequence input for the Keras model (`src/sequence_data.py`)

The sequence model gets `headway_ratio`, `dwell` and `delay_growth` at stops
n-5..n (`hr_0..hr_5`, `dw_0..dw_5`, `dg_0..dg_5`), with the same targets and
split as the tabular dataset.

---

## Leakage safety

- Every feature is built with `lag()` or backward window frames (`ROWS BETWEEN
  2 PRECEDING AND CURRENT ROW`). No feature expression calls `lead()`.
- `lead()` appears only in the `y_k*` and `target_valid_k*` columns, and no
  feature is built from them.
- Leader features are taken at stop n, which the leader has already passed.
- Weather is external and forecastable.

## Known gaps

- There is a rolling delay feature (`cum_delay_growth_3`) but no rolling
  *dwell* feature. In the diagnostics, delay-based features outrank
  dwell-based ones, and a `cum_dwell_3` would test whether that is just
  because dwell is only measured at a single stop.
- `n_routes_serving` is a static January snapshot.
- No clustering is used. All groupings (weather grid, `temp_bin`, `wx_group`)
  are fixed, domain-based bins.
