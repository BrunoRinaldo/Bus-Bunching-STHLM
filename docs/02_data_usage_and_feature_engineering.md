# Data usage, derived variables, and feature engineering

This document explains how the raw trip-stop observations become the modeling
table (`data/processed/model_dataset.parquet`): the mechanics of headway and
bunching-label construction, the feature groups built on top of it, the
leakage safeguard, and where clustering fits (and does not fit) in this
pipeline. Code: `src/headways.py` (Phase 2) and `src/features.py` (Phase 4).

---

## 1. From raw rows to headway and bunching labels

### 1.1 Event timestamps

`event_ts_arr = service_midnight(date) + observed_arrival_time_sfm seconds`,
built with interval addition rather than modulo arithmetic, so post-midnight
trips (`sfm` > 86,400) land on the correct following calendar day rather than
wrapping to a negative headway (plan trap #1). A second, always-populated
timestamp, `sched_event_ts_arr`, is built the same way from
`scheduled_arrival_time_sfm` and is used for the context features
(hour/day/month) specifically because the observed timestamp is null for
~8-10% of rows and the scheduled one never is - see the note on this bug fix
below.

### 1.2 Dwell time, delay growth, running speed

```
dwell        = observed_departure_time_sfm - observed_arrival_time_sfm
delay_growth = observed_departure_delay - observed_arrival_delay
run_time     = observed_arrival_time_sfm - previous stop's observed_departure_time_sfm
speed_mps    = dist_traveled / run_time   (guarded: run_time must be > 0)
```

`dwell` is set to null (not zero) when negative, or when the row is the first
or last `stop_sequence` of its trip (terminal layover/schedule recovery, not
passenger boarding - plan trap #3). Result: dwell is null for 16.0% of rows
(negative-dwell rows are only 0.003 percentage points of that; the rest is
terminal exclusion plus missing-observation propagation), median dwell 12 s,
mean 16.7 s - a right-skewed distribution as expected for boarding time.

### 1.3 Headway - scheduled-pair definition, and why

Two ways to define "the vehicle ahead" exist, and they diverge exactly when
bunching happens (a bus that has fallen behind can be overtaken by the one
behind it, which reorders arrivals):

- **(a) observed-order**: sort by `observed_arrival_time_sfm` at the stop,
  diff.
- **(b) scheduled-pair** (primary, per the plan): the leader is whichever
  trip was *scheduled* to arrive immediately before this one at this stop;
  `headway_obs` is the actual observed gap between that fixed pair.

(b) is used as the primary definition because it gives a stable
leader/follower identity across the whole trip, which the horizon target (§2)
depends on - under definition (a), "the vehicle ahead" can change from stop to
stop as vehicles overtake, which would make a multi-stop trajectory feature
meaningless.

Both were computed and compared. They disagree on the bunched/not-bunched
call for 1.36% of all rows overall, and that disagreement concentrates where
it should: **25.7% of rows that (b) calls bunched are *not* flagged by (a)**,
i.e. overtaking is meaningfully correlated with bunching, consistent with the
expected mechanism (Newell & Potts) and a validation that (b) is doing its
job of catching bunching that (a) would miss or mis-time - though this is a
smaller, more defensible gap than an earlier version of this analysis
reported (41.3%), for the reason explained in the two corrections below.

`headway_ratio = headway_obs / headway_sched`.
**`is_bunched = abs(headway_ratio) < 0.25`; `is_gap = headway_ratio > 1.75`**
(the paired hole behind every bunch - no negative-side analogue, see
correction 2 below). Both are materialised as columns on `headways.parquet`
(`is_bunched`, `is_gap`) so every downstream file references one definition
instead of re-deriving the threshold inline - which is exactly how the two
bugs below happened.

**Two corrections to the bunching label, found and fixed while building this
delivery (worth reading - both change reported numbers materially):**

1. **`headway_obs` needs a validity bound, and it needs to be asymmetric.**
   `headway_ratio` was first computed as `headway_obs / headway_sched` with
   no bound on `headway_obs` at all. ~1.07% of rows have a negative
   `headway_obs` and ~0.8% an absurd (>3h) one (plan trap #2 - missing-trip
   pairing, or the midnight bug). Checking the distribution of the negative
   ones showed 93% fall within 30 minutes of zero - a follower that
   genuinely overtook its scheduled leader, a real and physically plausible
   event - while a small tail (0.08% of all rows) reaches as far as -47
   *scheduled headways*, which is not a plausible single overtake between
   two consecutively-scheduled trips and is almost certainly a broken
   pairing. The fix is not to drop all negative values (that would delete
   real variability with no evidence it's wrong) and not to keep them
   unbounded either (that lets pairing corruption distort every aggregate) -
   it is to bound `headway_obs` at **[-1800, 10800] seconds** and null
   `headway_ratio` only outside that range. Rows inside it, including the
   negative part, are kept as real data.
2. **The bunching test must use `abs(headway_ratio)`, not the signed value.**
   Even after fix 1, `headway_ratio < 0.25` is wrong for the negative side:
   it calls a follower that overtook its leader by 25 minutes "bunched" just
   because -25 min / headway_sched happens to be less than 0.25, when a
   25-minute separation is the opposite of bunched - the two vehicles are far
   apart, just reordered. Bunching is about *physical proximity* between the
   two vehicles, which is symmetric around zero: a follower trailing its
   leader by 10 seconds and a follower that has just overtaken its leader by
   10 seconds are equally bunched. This was a 0.56%-of-all-rows mislabeling
   (61,631 rows out of 10.9M valid), found by comparing `headway_ratio < 0.25`
   against `abs(headway_ratio) < 0.25` directly.

**Threshold sensitivity** (share of valid rows with `abs(headway_ratio)`
under the threshold, full 8-line dataset):

| Threshold | 0.20 | 0.25 (primary) | 0.33 | 0.50 |
|---|---|---|---|---|
| % bunched | 1.82 | 2.29 | 3.09 | 5.35 |

**Base rate by line** (`abs(headway_ratio) < 0.25`):

| Line | % bunched | % gap |
|---|---|---|
| 4 | 4.10 | 4.70 |
| 179 | 3.06 | 3.47 |
| 117 | 2.30 | 3.25 |
| 607 | 1.96 | 1.92 |
| 474 | 1.88 | 2.64 |
| 541 | 1.53 | 1.58 |
| 401 | 1.34 | 1.58 |
| 116 | 0.87 | 0.98 |

Overall base rate 2.29% - inside the plan's 1-15% sanity band, but below the
"expect roughly 3-8%" comment for every individual line except 4 and 179.
Not a stop condition on its own, but worth naming plainly in the report: this
is a genuinely low-bunching set of lines at the strict abs-ratio definition,
not a labeling artifact (both corrections above only ever *reduced* the
reported rate from an inflated 2.91%, they did not inflate it).

### 1.4 Excess waiting time (EWT)

For a random passenger arrival, `E[W] = (E[h]/2) x (1 + CV^2)`. Computed per
line from observed headway mean/CV against the CV = 0 timetable benchmark:

| Line | Mean headway (s) | CV | EWT observed (min) | EWT timetable (min) | Excess (min) |
|---|---|---|---|---|---|
| 117 | 887 | 0.59 | 9.96 | 7.33 | 2.63 |
| 179 | 674 | 0.58 | 7.52 | 5.54 | 1.97 |
| 607 | 861 | 0.53 | 9.17 | 7.14 | 2.03 |
| 474 | 706 | 0.52 | 7.46 | 5.83 | 1.64 |
| 116 | 1061 | 0.51 | 11.16 | 8.82 | 2.34 |
| 4 | 573 | 0.49 | 5.91 | 4.67 | 1.24 |
| 401 | 914 | 0.43 | 9.04 | 7.56 | 1.48 |
| 541 | 844 | 0.38 | 8.04 | 6.99 | 1.04 |

This is the number that converts a technical headway-variance result into
passenger-minutes for the deployment section of the report: irregularity
alone costs riders on these 8 lines roughly one to two and a half minutes of
extra average wait, on top of what the timetable already implies.

### 1.5 A preliminary look at H1 (CV grows with position along the route)

Binning all 8 lines together by `route_frac` (position along the trip as a
fraction of its stop range) and recomputing headway CV per bin shows a mild
upward drift: CV ~0.52 near the start of trips versus ~0.56-0.57 near the
end. The effect is present but small and noisy at this pooled, cross-line
granularity - a real per-route test (one line at a time, plotted against
absolute `stop_sequence`) is Phase 3 (descriptive analysis) work and is out
of scope for this delivery, but this pooled check gives no reason to doubt
H1 going in.

---

## 2. Horizon targets - two-pass, leakage-safe construction

Unit of observation: one row per (date, route, direction, follower trip, stop
n). Target for horizon k: `y_k = 1` if the same leader/follower pair (fixed
by the scheduled-pair headway definition) has `is_bunched` (`abs(headway_ratio)
< 0.25`, §1.3) at stop n+k of the *same trip*.

Implementation: `lead(is_bunched, k) OVER (PARTITION BY date, trip_id
ORDER BY stop_sequence)`, gated by `lead(headway_ratio, k)` being non-null so
a row with an out-of-range/invalid headway at n+k is excluded rather than
silently scored. `LEAD` returns null when the trip does not have a
k-th following stop (short turn, early termination, end of route), which is
exactly the plan's "require both trips to actually serve stop n+k, drop the
row otherwise" rule, enforced structurally rather than by an extra filter
step that could be forgotten. A companion `target_valid_k{k}` boolean records
whether that k-th stop exists, so a null `y_k` can be told apart from "not
bunched."

**Leakage rule** (non-negotiable per the plan): every feature must be
computable from stop n or earlier. This was enforced by construction, not by
a post-hoc check: `src/features.py` builds a `features` CTE that uses only
`lag(...)` and `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` window frames
(strictly backward-looking), and keeps the `lead(...)` calls that build the
targets in a separate set of expressions in the same CTE that are clearly
named `y_k*`/`target_valid_k*` and never referenced by any other feature
column. A feature column literally cannot be built from a `lead()` value
because none of the feature expressions call `lead()`.

**Target availability and base rate by horizon** (full 8-line dataset):

| k | valid rows | % positive |
|---|---|---|
| 1 | 12,124,681 | 2.00 |
| 2 | 11,565,500 | 2.06 |
| 3 | 11,006,319 | 2.11 |
| 5 | 9,887,965 | 2.20 |
| 8 | 8,212,813 | 2.33 |

Valid-row count shrinks with k (fewer trips have a k-th following stop) and
the positive rate rises slightly with k - consistent with headway variance
accumulating along the route (§1.5) rather than a labeling artifact.

### 2.1 A third bug caught and fixed during this build

In addition to the two headway-label corrections in §1.3, the first version
of the split assignment (§ below, and see doc 3) used
`month(event_ts_arr)` to decide train/val/test. Because `event_ts_arr` is
null for the same ~8% of rows that have no observed arrival, those rows fell
through to the `ELSE` branch of the split `CASE` regardless of their actual
date, which put January rows in the test split. This is exactly the kind of
silent leakage-adjacent bug the plan warns about (trap #5, extended to
splitting): fixed by deriving `hour_of_day`/`day_of_week`/`month_of_year`
from the always-populated *scheduled* timestamp, and by assigning `split`
directly from the `date` column (never null, never derived) rather than from
any timestamp at all. Documented here so the same mistake is not repeated
when this pipeline is extended.

---

## 3. Feature groups

| Group | Features | Leakage-safe because |
|---|---|---|
| Current state | `headway_ratio`, `is_bunched_now`, `observed_arrival_delay`, `observed_departure_delay`, `dwell`, `dwell_rel_median` | all measured at stop n itself |
| Trajectory (expected strongest group) | `headway_ratio_lag1/2/3`, `closing_speed_3 = (headway_ratio - headway_ratio_lag3)/3`, `cum_delay_growth_3` (3-stop rolling sum), `speed_lag1/2` | built with `lag`/backward window frames only |
| Leader | `leader_dwell`, `leader_arrival_delay`, `leader_delay_growth` | the scheduled-predecessor trip's own values *at stop n*, not ahead of it |
| Position | `stop_sequence`, `route_frac` (fraction of the trip's stop range), `dist_traveled` | static per row |
| Context | `hour_of_day`, `day_of_week`, `month_of_year` (from scheduled time), `direction_id`, `LineNumber` | static per row |
| Stop attributes | `is_interchange` (from `transfers.csv`), `n_routes_serving` (static, see limitation in doc 1), `has_parent_station` | static per stop |
| Disruption | `under_active_alert` (service alert active for this `stop_id` at `event_ts_arr`) | binary flag only, no alert text used (scope limit, no NLP) |
| Weather | `temperature_2m`, `precipitation`, `is_precip`, `precip_3h`, `is_snowing`, `ice_risk`, `wx_group`, `wind_speed_10m` | exogenous / forecastable at prediction time, so using it is not leakage even though it is "outside" information - see doc 1 §1.3 in the plan and the join mechanics below |

`temp_bin` is still computed and kept as a column in `model_dataset.parquet`
(weather.py §2.4 of the plan), but §3.2 below explains why the models trained
in doc 3 use `temperature_2m` plus a `temperature_2m_sq` quadratic term
instead of `temp_bin` as an input feature. Use `temp_bin` directly if you
want the simpler categorical version for your own model.

### 3.1 Temperature: continuous-plus-quadratic instead of binning (a revision)

An earlier version of this pipeline used `temp_bin`, a 6-category binning of
`temperature_2m` (breakpoints at -5, 0, 5, 15, 25 degrees C - see §4 below for
the original values), specifically because the plan hypothesises a U-shaped
effect (boarding slows at both cold and hot extremes) that a single linear
`temperature_2m` coefficient cannot represent in logistic regression. That
reasoning for *not* using raw `temperature_2m` alone in a linear model still
holds, but binning was not the best way to solve it: bin edges are an
arbitrary, hand-picked discretisation that throws away information near
every boundary, and `HistGradientBoostingClassifier` never needed the bins in
the first place since it can already split on the continuous value at any
threshold. The fix used in `src/models.py` and `src/export_encoded_sample.py`:
drop `temp_bin` for both models, and for logistic regression specifically add
one `temperature_2m_sq = temperature_2m ** 2` column (scaled the same way as
every other numeric feature) - the standard way to let a linear/GLM model fit
a U-shape with a single extra term instead of six categorical ones. Gradient
boosting keeps only plain `temperature_2m` (the squared term would be
redundant there, though harmless if included).

### 3.2 Weather join mechanics (plan trap #9)

Stops were snapped to a 0.1-degree grid (`round(lat,1)`, `round(lon,1)`),
yielding **9 grid points** covering all 380 stops used by the 8 selected
lines - one Open-Meteo Archive API call per grid point for the full year
(79,056 rows total, exactly 9 x 366 x 24, confirming no gaps in the fetched
series). Cached to `data/processed/weather.parquet` and `.wxcache/`; the API
is never called again.

The timezone conversion follows the plan's method to avoid the DST trap:
local-naive `event_ts_arr` is first attached to `Europe/Stockholm` (DuckDB's
ICU `AT TIME ZONE`, which shifts nonexistent spring-forward times forward by
the gap rather than erroring, and resolves the one ambiguous
fall-back hour to standard time rather than pandas' `NaT` - a difference from
the plan's exact recipe that affects at most one clock hour per year and was
accepted rather than reimplemented in pandas), then converted onward to UTC,
then floored to the hour and joined to `weather.parquet` on
`(grid_lat, grid_lon, hour)`. Weather is null only for the rows that already
have a null `event_ts_arr` (7.9% - see doc 1's limitations section); there is
no additional join loss.

---

## 4. On "clustering"

No unsupervised clustering algorithm (k-means, hierarchical, DBSCAN, etc.) is
used in this pipeline, and this is a deliberate choice worth stating rather
than leaving implicit. The grouping operations that exist are all
domain-driven bins, not learned clusters:

- **Weather grid snapping** (0.1 degrees) is a spatial *discretisation*
  driven by the resolution of the underlying reanalysis data, not a
  clustering of stops by similarity.
- **`temp_bin`** (breakpoints at -5, 0, 5, 15, 25 degrees C) and **`wx_group`**
  (WMO weather codes collapsed to clear/cloudy/fog/drizzle/rain/snow/rain
  showers/snow showers/thunderstorm) encode known non-linear thresholds
  (freezing, precipitation type) rather than data-driven boundaries.
- **`is_interchange`**, **`has_parent_station`**, **`n_routes_serving`** are
  stop-attribute flags/counts, again not a clustering of stops.

This was a reasonable choice for this project: the feature groups above are
already interpretable and tied to a stated mechanism (dwell time inflates,
which compresses the gap ahead), which matters directly for RQ2 and for the
logistic-regression coefficients that carry the mechanism story in the
report. An unsupervised stop clustering (e.g. k-means on each stop's
dwell-time distribution, or on its `(is_interchange, n_routes_serving,
mean_dwell, mean_delay)` profile, to get a small number of "stop archetypes"
as a categorical feature instead of many near-collinear raw attributes) would
be a legitimate extension if the model comparison in doc 3 shows the raw stop
attributes underperforming or the categorical `LineNumber`/`observed_stop_id`
cardinality becomes a problem for the logistic regression or the Keras model.
It was not built here because nothing in the Phase 0/2 validation results
pointed to a need for it, and adding it without that justification would be
exactly the kind of complexity the plan's "no half-finished implementations"
guidance warns against. If it is wanted, the natural place is between
`with_stop_attrs` and the final `SELECT` in `src/features.py`.
