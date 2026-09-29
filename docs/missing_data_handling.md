# Missing data handling

This document explains where values are missing in the modelling data, and
how each of the three models (gradient boosting, logistic regression, and the
Keras sequence model) deals with them. Code references are to `src/`.

In short: rows are only removed when the *target* is missing. Missing
*features* are never a reason to drop a row; each model has its own way of
handling them.

---

## 1. Where missing values come from

| Source | Effect |
|---|---|
| The real-time feed has no observed arrival for about 8-10% of scheduled stop events (no GPS/AVL match) | Everything computed from the observed time is missing on those rows: `headway_obs`, `headway_ratio`, `dwell`, `speed_mps`, and weather (the weather join key is the observed timestamp) |
| Start of a trip | Lag features have no history: `headway_ratio_lag3` and `closing_speed_3` cannot exist for the first three stops, the sequence model's n-5 step not for the first five |
| Deliberate blanking | `dwell` is set to null at the first and last stop of a trip (layover, not boarding) and when negative (bad timestamps); `headway_ratio` is set to null when `headway_obs` is outside [-1800, 10800] s (broken leader/follower pairing) |
| Leader not observed | `leader_dwell`, `leader_arrival_delay`, `leader_delay_growth` are missing when the leader bus has no observation at that stop |

Over the whole dataset, 13.1% of rows have no `headway_ratio` and 7.9% have
no weather (see `docs/01_data_description.md`).

### Missing shares in what the models actually train on

Measured on the k=3 modelling rows (rows with a valid target, all splits):

| Feature | Missing |
|---|---|
| `headway_ratio_lag3`, `closing_speed_3` | 15.6% |
| `headway_ratio_lag1` | 5.4% |
| `leader_dwell` | 5.4% |
| `speed_mps` | 5.3% |
| `dwell`, `dwell_rel_median` | 5.1% |
| `headway_ratio` | 0.3% |
| `leader_arrival_delay` | 0.2% |
| `observed_arrival_delay`, `cum_delay_growth_3`, weather | 0.1% |
| stop attributes, `under_active_alert` | 0.0% |

For the sequence model: 25.8% of rows have no value at step n-5 (mostly
early stops of a trip), 0.3% none at step n.

These shares are much lower than the dataset-wide ones because a row with no
observation at stop n usually also has no valid target, and those rows are
removed (next section). What remains is mostly *history* that is missing,
not the current state.

---

## 2. Rows without a target: removed, for all models

`y_k{k}` is null when the trip does not reach stop n+k (short turn, end of
route) or when `headway_ratio` at n+k is invalid. About 25% of rows have no
valid target at k=3.

These rows are dropped when loading (`WHERE y_k{k} IS NOT NULL` in
`models.load` and `sequence_model.load`), from training and from evaluation
alike. They are **not** counted as "not bunched": a missing target means the
outcome is unknown, and treating it as a negative would bias the base rate
downwards and teach the model that the end of a route is safe.

---

## 3. Gradient boosting (`HistGradientBoostingClassifier`)

**No imputation.** Missing values are passed to the model as-is.

- `HistGradientBoostingClassifier` supports missing values natively. At every
  split it learns which branch missing values should go to, using the
  training data. So "this value was not observed" can itself be used as
  information, for example if missing leader data tends to go with irregular
  service.
- Categorical columns (`LineNumber`, `direction_id`, `wx_group`) are converted
  to strings, so a missing category becomes its own level.
- Yes/no columns are stored as 0/1 floats, so a missing flag stays missing.

Code: `models.make_hgb`, `models.prep_hgb_frame`, `tuning.fit_hgb`.

---

## 4. Logistic regression

**Median imputation with missing indicators, then standardisation.**

```python
SimpleImputer(strategy="median", add_indicator=True) -> StandardScaler()
```

- Every numeric and yes/no feature with missing values is filled with its
  **median from the training split**. The medians are fitted on the training
  data only, and the same values are then applied to validation and test, so
  no information leaks from the later months.
- `add_indicator=True` adds a 0/1 column `<feature>_was_missing` for every
  feature that had missing values in training. Without it, a filled-in
  median would look exactly like a real median-sized value. With it, the
  model gets a separate coefficient for "not observed".
- Categorical columns are one-hot encoded; a missing category gets its own
  column, and categories not seen in training are ignored
  (`handle_unknown="ignore"`).
- The engineered features added during tuning (`abs_headway_ratio*`,
  `near0_headway_ratio*`, `abs_closing_speed_3`) keep the gap: they are
  missing when the underlying ratio is missing, and then go through the same
  imputer and indicator.

Code: `models.make_logreg_pipeline`, `tuning.make_pre`, `tuning.add_eng`.

---

## 5. Keras sequence model

**Standardise, fill with zero, and add a mask channel.**

The input is `headway_ratio`, `dwell` and `delay_growth` at stops n-5 to n, a
6 x 3 sequence per row.

1. Each channel is standardised with the mean and standard deviation of the
   training split, computed while ignoring missing values (`nanmean`,
   `nanstd`).
2. Missing values are then set to **0**, which after standardisation is the
   training mean.
3. A **fourth channel** is added: for each of the 6 steps, 1 if
   `headway_ratio` was observed and 0 if it was missing. This lets the
   network tell "average value" apart from "no data", and it covers the
   padding at the start of a trip, where earlier stops do not exist.

Code: `sequence_model.standardize`.

---

## 6. Evidence on the cost of missing data

- **Leader information.** Retraining gradient boosting without the three
  leader features lowers PR-AUC at k=3 from 0.814 to 0.779, a 4.3% relative
  drop (`reports/phase6_diagnostics.md` §3). This is an upper bound on what
  losing the leader's data costs: the model gets weaker but does not fail.
- **Missing values as signal.** Because gradient boosting routes missing
  values explicitly and logistic regression has the `_was_missing`
  indicators, both can use a missing observation as information in its own
  right instead of just losing it.

---

## 7. Limitations

- **The Keras mask covers `headway_ratio` only.** A missing `dwell` or
  `delay_growth` becomes 0 without its own flag, so the network cannot tell
  it apart from a genuinely average value. Separate mask channels for dwell
  and delay growth would fix this cheaply.
- **Median imputation ignores context.** A missing `dwell` is filled with the
  overall training median, not the median for that stop or hour. Gradient
  boosting, the best model, does not impute and is not affected.
- **Missing observations are probably not random.** Feed gaps vary by line
  and by month (`docs/01_data_description.md`), so they may coincide with
  disrupted or unusual service. The indicators and native handling let the
  models use that pattern, but it also means the performance on fully
  observed rows may differ from the average.
- **Rows without a target are excluded from evaluation.** The reported
  scores therefore apply to stops where the outcome is known; predictions
  near the end of a route, where k stops ahead do not exist, are not scored.
