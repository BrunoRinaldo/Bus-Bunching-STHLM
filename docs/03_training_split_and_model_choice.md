# Training split and model choice

This document covers how `data/processed/model_dataset.parquet` is split for
training and evaluation, why that split is structured the way it is, and the
four models called for by the plan - all four have now been trained and
evaluated; §5 has the full results, the headline figure, and two findings
that contradict what the plan expected going in. §6 covers hyperparameter tuning:
a search at k=3 reused at every horizon, then a separate search per
horizon, with a paired bootstrap comparing the two, and a final round of
features and search for k=5 and k=8.

---

## 1. Split strategy

### 1.1 Temporal split (primary)

| Split | Months | Rows | Share |
|---|---|---|---|
| `train` | Jan-Sep 2024 | 9,543,333 | 75.3% |
| `val` | Oct 2024 | 1,088,445 | 8.6% |
| `test` | Nov-Dec 2024 | 2,052,084 | 16.2% |

Assigned directly from the `date` column (`SELECT ... CASE WHEN
(date/100)%100 <= 9 THEN 'train' WHEN ... = 10 THEN 'val' ELSE 'test'`), not
from any derived timestamp - see doc 2 §2.1 for the split-assignment bug this
avoided (there were two further corrections to the bunching label itself,
documented in doc 2 §1.3, that changed the numbers in §2 below but not the
split). **Never
split randomly.** Adjacent stops on the same trip, and the same trip observed
at consecutive dates, are near-duplicates in feature space; a random split
would put near-identical rows on both sides of the train/test boundary and
report a leakage-inflated score. A temporal split also makes the evaluation
honest about season: the test set (Nov-Dec) is Stockholm winter, structurally
different from the mostly train months, which is the right stress test for
H4 (does weather change the base rate without changing which features
matter).

The October DST transition (27 Oct 2024) falls inside the validation month.
This was handled explicitly in the weather join (doc 2 §3.1), not dropped.

**Known limitation: the validation set has no cold-weather examples at
all.** October is mild - validation min temperature is +1.1 degC, mean 9.5
degC, and 0.00% of validation rows have `is_snowing` or `ice_risk` set,
against 4.2% and 3.9% respectively in the Nov-Dec test set and January
train-set lows down to -18.9 degC. Train (Jan-Sep) and test (Nov-Dec) both
contain real cold; only validation is blind to it. This matters specifically
where validation is used to make a decision: the Keras model's early
stopping watches `val_pr_auc`, so its stopping point was chosen against a
validation set that cannot see cold-weather behaviour. Logistic regression
and gradient boosting did not use the validation split for any decision
(HGB's own early stopping carves its validation fraction out of the training
data by default, not this split) - it is reported for reference only there.
**This changed with the later tuning work:** `reports/tuning_results.md`
selected hyperparameters for all three learned models (and the PCA / feature
choices for logistic regression) on this validation split, so those choices
were also made without seeing cold weather. The test split is the check.
`reports/split_comparison.md` tests two alternative splits against this
concern directly (random trip-level and week-based) and finds that both fix
the validation weather gap but neither is a better primary split than this
one - see that report for the full comparison and why the temporal split
was kept.

### 1.2 Route holdout (for RQ4)

Run in Phase 6: trained on lines 4, 541, 474, 607 and scored on the
held-out 116, 117, 179, 401 - essentially no degradation (-0.39%), see
`reports/phase6_diagnostics.md` §4. This is a single 4-vs-4 split rather
than the leave-one-line-out design originally considered - given the small
line count (8), repeating across several different splits and reporting the
spread, not just one number, would be the stronger design and is not
done here. The rest of this section describes that original plan. Recall the
limitation from doc 1 §4.2: all 8 lines are high-frequency trunk routes by
construction, so this tests generalisation *within* the trunk-bus regime, not
generalisation to SL's lower-frequency network.

### 1.3 Class imbalance

Base rate is 2.0-2.3% depending on horizon (§2 below). Per the plan: use
class weights during training, never resample, and never touch the test set's
class balance. `target_valid_k{k}` must be applied as a filter *before*
computing any metric - the null rows are "trip didn't reach stop n+k," not
negatives, and silently coercing them to 0 would bias every metric.

---

## 2. Persistence baseline - run now, as the plan requires ("implemented
first, non-negotiable")

Prediction: assume the pair's state at n+k equals its state at n (i.e.
predict bunched at n+k if `is_bunched_now` is true at n - `abs(headway_ratio)
< 0.25`, doc 2 §1.3). Evaluated on the **test split only** (Nov-Dec), which
is the split that matters:

| k | n (valid) | base rate | precision | recall |
|---|---|---|---|---|
| 1 | 1,464,975 | 2.29% | 88.68% | 84.06% |
| 2 | 1,396,246 | 2.35% | 84.04% | 74.91% |
| 3 | 1,327,525 | 2.40% | 80.21% | 67.32% |
| 5 | 1,190,116 | 2.50% | 73.99% | 55.04% |
| 8 | 984,445 | 2.63% | 66.03% | 41.44% |

(These numbers replace an earlier run of this same baseline that used a
signed `headway_ratio < 0.25` test; that version mislabeled large-magnitude
negative ratios - a follower far ahead of its nominal leader, not bunched
with it - as bunched, which inflated every number in this table. Doc 2 §1.3
has the full explanation. The pattern described below is unchanged, only the
precise figures are.)

This is exactly the decay pattern the plan's H3 predicts, and it is a real,
non-fabricated result from the actual pipeline, not an illustration: recall
drops from 84% one stop ahead to well below a coin flip eight stops ahead,
while precision degrades more gently (66% even at k=8). Two things follow
from this:

1. **The task is well-posed.** The baseline is strong at k=1 (as it should
   be - bunching does not appear from nothing between adjacent stops) and
   degrades smoothly rather than collapsing immediately, which is the
   signature of a real, learnable, decaying-predictability process rather
   than a noise-dominated label.
2. **The bar the learned models have to clear is high, especially at low
   k.** At k=1-2, persistence is already precision >84%; a logistic
   regression or GBM needs the trajectory/leader/weather features to add
   value on top of "is it already bunched," not just replicate that signal.
   Where the learned models should earn their keep is extending usable
   performance further out in k, and in recall - persistence's recall falls
   off faster than its precision, which is exactly where trajectory features
   (closing speed, cumulative delay growth) ought to help, since persistence
   by definition cannot see a pair that is *approaching* bunching but has not
   arrived yet.

This baseline number is also the reference point the report's headline
figure (PR-AUC vs. horizon, all models plus baseline) should be plotted
against - not zero, not a naive majority-class predictor.

---

## 3. Model choice

Four models, per the plan, all four in the final comparison table:

### 3.1 Persistence baseline - done (§2 above)

### 3.2 Logistic regression

On the engineered features (current state, trajectory, leader, position,
context, stop attributes, disruption, weather), with class weights for the
2.0-2.3% base rate. Chosen specifically because its coefficients are
interpretable and are what carries the mechanism story for RQ2 (does dwell
growth predict bunching better than delay alone) in the report - a
coefficient table split by feature group is a natural report artifact this
model produces almost for free. It was expected to underperform the GBM on
raw predictive power but not be included as a strawman - it is the model that
answers "why," while the GBM answers "how well." §5 has the actual result:
the underperformance is real and larger than the "somewhat weaker" framing
here implies (PR-AUC 0.43-0.59 against 0.62-0.90 for the other two models),
which does not undermine its role in the report but is worth stating plainly
rather than softening.

### 3.3 Gradient boosting

`HistGradientBoostingClassifier` (scikit-learn, no extra dependency beyond
what Python's ecosystem already needs). Confirmed the strongest model
overall (§5) - narrowly at short horizons, clearly ahead by k=5-8.
`HistGradientBoostingClassifier` natively handles the null values documented
in doc 1 (13% null `headway_ratio`-family features, 7.9% null weather)
without imputation, which is a practical reason to prefer it over plain
`GradientBoostingClassifier` for this specific dataset. Feature importances
from this model (via permutation importance, since
`HistGradientBoostingClassifier` does not expose the impurity-based
`feature_importances_` attribute that plain `GradientBoostingClassifier`
or `RandomForestClassifier` do) feed the discussion in §5.2 directly, and the
full ablation in Phase 6 (delay-only vs. dwell-only vs. both features, for
H2) should be run on this model since it is the strongest and the one most
likely to expose a real gap between the two feature groups.

### 3.4 Keras sequence model

A 1D-CNN over the sequence of `(headway_ratio, dwell, delay_growth)` across
stops n-5...n, plus a 4th channel marking which timesteps had a missing
observation (§5.3 has the architecture detail). This is the `.keras` export
required by the assignment's task F, trained and exported regardless of
whether it wins, per the plan. It did not need that protection: it is
essentially tied with the GBM at short horizons (actually the single best
model at k=1) and only falls behind at k=5-8 (§5). The lag features already
present in the tabular dataset (`headway_ratio_lag1/2/3`) give the
non-sequence models partial access to the same trajectory information in
flattened form, so the fact that the full 6-step sequence still matches or
beats the GBM at short horizons, despite the GBM also having those lags, is
itself a real report-worthy result, not just a formality being satisfied.

---

## 4. Metrics

PR-AUC as the primary metric (not accuracy - meaningless at a 2.0-2.3% base
rate, per the plan and confirmed by the actual base rate measured above),
plus F1, recall at a fixed precision operating point, a confusion matrix, and
a calibration plot. Report the operational framing explicitly: at 80%
recall, what is the false-alarm rate? A controller that gets cried-wolf
alerts will start ignoring the system, so this precision-recall tradeoff is
the deployment discussion, not a side note to it.

**The headline output of the whole project**: PR-AUC against horizon k, all
four models plus the persistence baseline on one axis, with k converted to
minutes using the median inter-stop running time so the x-axis reads as
operationally meaningful lead time rather than an abstract stop count. That
figure now exists - `figures/fig8_pr_auc_vs_horizon.png` - and the results
behind it are in §5.

---

## 5. Results

All four models trained on `train` (Jan-Sep), threshold and early-stopping
decisions made on `val` (Oct), scored once on `test` (Nov-Dec) - never
touched before this table. Code: `src/models.py` (logreg + HGB),
`src/sequence_model.py` (Keras, run under the separate `.venv312`
environment - see §5.3), `src/compare_models.py` (the figure and this
table). Full numbers: `data/results/results_tables.md`.

**All numbers in this section use the original default settings** (and, for
logistic regression, the original feature set). Hyperparameter tuning and a
PCA / engineered-feature study were run afterwards - see §6,
`reports/tuning_results.md` and `figures/fig9_tuned_pr_auc_vs_horizon.png`.
After tuning, gradient boosting leads at every horizon (test PR-AUC 0.958 /
0.905 / 0.856 / 0.762 / 0.652 at k = 1 / 2 / 3 / 5 / 8), including k=1 where
the untuned Keras model had been ahead, so the "essentially tied" reading
below is a statement about untuned defaults.

| k | minutes | persistence P/R | logreg PR-AUC | HGB PR-AUC | Keras PR-AUC |
|---|---|---|---|---|---|
| 1 | 1.05 | 0.89 / 0.84 | 0.586 | 0.898 | **0.917** |
| 2 | 2.10 | 0.84 / 0.75 | 0.563 | **0.860** | 0.855 |
| 3 | 3.15 | 0.81 / 0.67 | 0.539 | **0.814** | 0.786 |
| 5 | 5.25 | 0.75 / 0.55 | 0.493 | **0.696** | 0.682 |
| 8 | 8.40 | 0.67 / 0.41 | 0.425 | **0.623** | 0.562 |

(Persistence has no probability score, so PR-AUC is undefined for it; its
precision/recall operating point is shown instead, and marked with an X
rather than a line on the figure so it is not visually compared on an axis
it cannot occupy.)

### 5.1 The headline result

Both learned tabular/sequence models clear the persistence baseline's
implicit standard by a wide margin at every horizon, and the gap widens with
k: gradient boosting's PR-AUC is 0.623 at k=8 versus persistence's 0.67
precision / 0.41 recall operating point at the same horizon - not a like-for-
like comparison (a PR-AUC integrates over every threshold, a P/R pair is one
point), but the practical reading is that GBM/Keras keep meaningfully more
of their predictive power at long horizons than naive persistence does,
which is the entire premise for building a learned model at all (RQ3). All
three learned-model curves decay smoothly rather than collapsing at any
single k, which is the same "real, not noise-dominated" signature noted for
the persistence baseline itself.

### 5.2 Two findings that contradict what the plan expected going in

**Logistic regression underperforms much more than "somewhat weaker."**
PR-AUC of 0.43-0.59 against 0.62-0.92 for the other two models is a large
gap, not the mild trailing the plan's framing in §3.2 implied. This does not
disqualify it from the report - its coefficients are still the clearest
mechanism story available (below) - but the report should not describe it as
a close third model; it should say plainly that a linear model captures
much less of this signal than the two nonlinear ones do.

**Correction, added after the tuning work (`reports/tuning_results.md`
§2):** an earlier version of this paragraph said the gap "is itself
informative about how nonlinear the true relationship is." That reading
was mostly wrong. Most of the gap was one missing representation: bunching
means `abs(headway_ratio) < 0.25`, a U-shaped function of the signed ratio
that no single linear coefficient can express, and the feature set did not
include the absolute value. Adding `abs(headway_ratio)`, its lags and
near-zero flags lifted logistic regression at k=3 from 0.539 to 0.707 test
PR-AUC (0.721 with PCA on top); PCA alone and regularisation tuning changed
nothing. The logistic-regression numbers in the table above and the
coefficient discussion below describe the **original, pre-engineering
model**; they remain a correct record of that model but understate what a
linear model can do here. Even the improved version still trails gradient
boosting at every horizon.

**`closing_speed_3` does not dominate, in either model.** The plan's Phase 4
said explicitly: "Expect the closing-speed feature to dominate. If it does
not, investigate before accepting the result." It does not, and this was
checked two ways at k=3: the logistic regression coefficient on
`closing_speed_3` (-0.018) is smaller than the coefficients on
`observed_arrival_delay` (2.18), `leader_delay_growth` (0.56), and even
`leader_dwell` (0.20); and permutation importance on the (stronger) GBM
model - `sklearn.inspection.permutation_importance`, scored on PR-AUC, 60,000
held-out rows, `models/hgb_permutation_importance_k3.csv` - puts
`closing_speed_3` at 0.033, far behind `headway_ratio` itself (0.73,
overwhelmingly the single strongest feature in either model),
`observed_arrival_delay` (0.127), `speed_mps` (0.117), and
`leader_arrival_delay` (0.091). The most plausible explanation is
collinearity rather than a broken feature: `closing_speed_3` is
algebraically derived from `headway_ratio` and its own lags
(`(headway_ratio - headway_ratio_lag3) / 3`), which are already in the
model, so its independent marginal contribution is diluted once those are
present - permutation importance in particular measures marginal
contribution given everything else, so correlated features structurally
undercut each other's individual scores even when the underlying
relationship they jointly capture is strong. This is a plausible, benign
explanation, not a confirmed one; if it matters for the report, the clean
way to settle it is to compare a model with `headway_ratio`'s own lags
removed against one with them present, isolating what `closing_speed_3`
adds on its own - not done here, and a reasonable Phase 6 addition.

**A related, tentative signal on H2** (dwell growth vs. delay predicting
bunching): in both the logreg coefficients and the GBM permutation
importance above, delay-based features (`observed_arrival_delay`,
`leader_arrival_delay`, `leader_delay_growth`) outrank dwell-based ones
(`dwell`, `dwell_rel_median`, `leader_dwell`) by a wide margin at k=3. Taken
at face value this cuts against H2's specific prediction that dwell growth
would be the stronger predictor. This was suggestive, not a substitute for
the planned Phase 6 ablation (delay-only vs. dwell-only vs. both features,
compared by the resulting PR-AUC of separately trained models) - importance
scores from one full model are not the same experiment as training the
restricted models the plan actually specifies.

**The Phase 6 ablation has now been run and confirms it** (`reports/phase6_diagnostics.md`
§2): delay-only features score PR-AUC 0.8188, beating both dwell-only
(0.8034) and the combined delay+dwell feature set (0.8132). H2 does **not**
hold at this feature resolution - delay predicts bunching better than dwell
does, and combining them does not even beat delay alone. This is reported
as a genuine negative finding, not softened, and it is one of the more
interesting results in the project precisely because it contradicts the
plan's stated causal mechanism (dwell inflation as the driver). See
`reports/phase6_diagnostics.md` §2 for the fuller discussion, including a
plausible reason (a rolling multi-stop delay feature exists in this feature
set - `cum_delay_growth_3` - but no equivalent rolling *dwell* feature does,
so the comparison may be structurally unfair to dwell rather than dwell
being truly less predictive; a fair rematch would add a rolling dwell
feature before concluding H2 is false rather than untested fairly).

### 5.3 Keras architecture actually used

Two causal `Conv1D` layers (24 filters, kernel 3) over the 6-step window,
`GlobalAveragePooling1D`, a 16-unit dense layer with dropout 0.2, sigmoid
output. Trained with `class_weight` (Adam, 1e-3, batch size 8192, up to 12
epochs with early stopping on validation PR-AUC). Run under a separate
Python 3.12 virtual environment (`.venv312`) because TensorFlow/Keras had no
installable wheel for the Python 3.14 interpreter used everywhere else in
this project at the time of this work - a real environment constraint, not
a design choice; `pip install tensorflow` / `jax` / `torch` all failed with
"no matching distribution" under 3.14. `src/sequence_data.py` builds the
shared sequence table (`data/processed/sequence_dataset.parquet`) once;
`src/sequence_model.py` trains all 5 horizons from it.

---

## 6. Hyperparameter tuning

The models in §5 use default settings. Tuning was done in two rounds:
first a search at k=3 whose winners were reused at every horizon (§6.2),
then a separate search at each horizon (§6.3). A paired bootstrap on the
test split measures what the per-horizon search adds over the k=3 search
(§6.4).

### 6.1 Protocol (both rounds)

- **Search** on an 800,000-row random subsample of `train` (Jan-Sep),
  class weights as in §1.3, no resampling.
- **Selection** on the full `val` split (Oct) by PR-AUC.
- **Refit** of the selected configuration on the full `train` split; `val`
  and `test` (Nov-Dec) are scored once, for the refit model only. `test` is
  never used for a decision.
- HGB uses early stopping on an internal 10% of the training data
  (`n_iter_no_change=15`), `random_state=0`, so HGB refits are
  deterministic. Keras uses early stopping on `val` PR-AUC and is seeded per
  horizon and trial.

### 6.2 Round 1: tuned at k=3, reused at k=1, 2, 5, 8

Code `src/tuning.py`, `src/tuning_keras.py`, `src/compare_tuned.py`;
full write-up `reports/tuning_results.md`; figure
`figures/fig9_tuned_pr_auc_vs_horizon.png`.

| Model | Search at k=3 | Selected configuration |
|---|---|---|
| HGB | default + 12 random configurations. Space: learning rate {0.03, 0.05, 0.1, 0.2}, max leaf nodes {15, 31, 63, 127, 255}, min samples per leaf {20, 50, 100, 300, 1000}, L2 {0, 0.1, 1, 10}, max depth {none, 6, 10} | learning rate 0.2, 255 leaves, min leaf 20, L2 10, no depth limit |
| Logistic regression | C in {0.01, 0.1, 1, 10}, PCA 5-60 components, original vs. engineered features (`abs(headway_ratio)`, its lags, near-zero flags, `abs(closing_speed_3)`) | engineered features, PCA 45 components, C = 0.1 |
| Keras | default + 9 random configurations. Space: CNN/LSTM, 1-3 conv layers, 16/32/64 filters, kernel 2/3, pooling {average, flatten, last step}, dense 16/32/64, dropout 0.1/0.2/0.4, learning rate {3e-4, 1e-3, 3e-3}, batch {2048, 4096, 8192} | 3 causal conv layers, 64 filters, kernel 3, last-step pooling, dense 32, dropout 0.4, lr 1e-3, batch 2048 |

Test PR-AUC, default settings (§5) -> round 1:

| k | min | HGB | Keras | LogReg |
|---|---|---|---|---|
| 1 | 1.05 | 0.898 -> **0.958** | 0.917 -> 0.932 | 0.586 -> 0.859 |
| 2 | 2.10 | 0.860 -> **0.905** | 0.855 -> 0.862 | 0.563 -> 0.785 |
| 3 | 3.15 | 0.814 -> **0.856** | 0.786 -> 0.815 | 0.539 -> 0.721 |
| 5 | 5.25 | 0.696 -> **0.762** | 0.682 -> 0.709 | 0.493 -> 0.612 |
| 8 | 8.40 | 0.623 -> **0.652** | 0.562 -> 0.579 | 0.425 -> 0.488 |

Tuned HGB is the best model at every horizon, including k=1 where the
default Keras model had been ahead. For logistic regression, C and PCA
changed almost nothing; the engineered `abs(...)` features gave the gain
(0.539 -> 0.707 at k=3, 0.721 with PCA 45 on top), see §5.2.

### 6.3 Round 2: a separate search at each horizon

Code `src/tuning_per_k.py` (HGB, logistic regression),
`src/tuning_keras_per_k.py` (Keras, `.venv312`), `run_tuning_per_k.sh`
(runs both, resumable). Raw logs `data/results/tuning_per_k.json`,
`data/results/tuning_keras_per_k.json`. Models `models/perk_hgb_k{k}.joblib`,
`models/perk_logreg_k{k}.joblib`, `models/perk_sequence_k{k}.keras`.
Figures, table and curves `src/perk_curves.py`, `src/compare_perk.py`,
`data/results/perk_results_table.md`. Total run time 9.1 h (HGB + logistic
regression 6.2 h, Keras 2.9 h).

Search per horizon, same spaces as §6.2:

- **HGB**, 20-23 trials: default, the round-1 winner, 14 random
  configurations, then up to 8 neighbours of the best random-round result
  (one parameter moved one grid step). `max_iter` 800.
- **Logistic regression**, 12 trials on the engineered features: C in
  {0.01, 0.1, 1, 10} without PCA, plus PCA {20, 30, 45, 60} x C {0.1, 1}.
- **Keras**, 14 trials: default, the round-1 winner, 12 random
  configurations. Search 10 epochs (patience 2), refit 15 epochs
  (patience 3).

Selected configurations:

| k | HGB | Keras | LogReg |
|---|---|---|---|
| 1 | lr 0.2, 255 leaves, min leaf 50, L2 10, no depth limit (60 iter) | CNN 3x64, kernel 3, flatten, dense 32, dropout 0.1, lr 3e-4, batch 2048 | no PCA, C 0.01 |
| 2 | = round-1 winner (62 iter) | CNN 3x64, kernel 2, last step, dense 16, dropout 0.2, lr 1e-3, batch 4096 | no PCA, C 0.01 |
| 3 | lr 0.2, 255 leaves, min leaf 300, L2 0.1, depth 10 (70 iter) | = round-1 winner | = round-1 winner (PCA 45, C 0.1) |
| 5 | **lr 0.03**, 255 leaves, min leaf 100, L2 1, **depth 10 (653 iter)** | CNN 1x64, kernel 2, flatten, dense 64, dropout 0.4, lr 1e-3, batch 2048 | no PCA, C 0.01 |
| 8 | **lr 0.03**, 255 leaves, min leaf 100, L2 0, **depth 10 (661 iter)** | = round-1 winner | no PCA, C 1 |

At k=5 and k=8 the HGB search selects a different kind of model from the
short horizons: learning rate 0.03 instead of 0.2, depth limited to 10,
about 650 boosting iterations instead of about 60. Refitting these two
models on the full training split took 1.8 h and 2.4 h.

Test results (`figures/fig11_perk_pr_auc_vs_horizon.png`,
`figures/fig12_perk_recall_at_p80.png`):

| k | min | HGB PR-AUC | HGB R@P80 | Keras PR-AUC | Keras R@P80 | LogReg PR-AUC | LogReg R@P80 |
|---|---|---|---|---|---|---|---|
| 1 | 1.05 | **0.958** | 0.963 | 0.922 | 0.926 | 0.846 | 0.896 |
| 2 | 2.10 | **0.905** | 0.879 | 0.870 | 0.847 | 0.774 | 0.777 |
| 3 | 3.15 | **0.853** | 0.786 | 0.812 | 0.736 | 0.721 | 0.626 |
| 5 | 5.25 | **0.767** | 0.602 | 0.677 | 0.350 | 0.608 | not reached |
| 8 | 8.40 | **0.659** | 0.359 | 0.579 | 0.138 | 0.489 | not reached |

R@P80 = recall at precision >= 0.80. "Not reached" means no threshold gives
precision 0.80. HGB remains the best model at every horizon.
`figures/fig13_perk_tuning_spread.png` shows every search trial on the
validation split. Tuning gains over the default settings are largest for
Keras (+0.11 to +0.17 validation PR-AUC on the search subsample) and
+0.014 to +0.037 for HGB.

### 6.4 Per-horizon vs. k=3 search: paired bootstrap

Code `src/bootstrap_perk.py`; results `data/results/perk_bootstrap.json`,
`data/results/perk_results_table.md`; figure
`figures/fig17_perk_bootstrap_diff.png`.

Both models score the same test rows. The round-1 comparison models are
refit exactly as in `tuning.py` stage 2 (asserted: same test PR-AUC as
reported in §6.2). The 61 test days are resampled with replacement
(B = 1000), because rows from the same day share weather, disruptions and
traffic. The interval is the 2.5-97.5 percentile of the difference.

Difference, per-horizon tuned minus round 1 (test split):

| Model | k | PR-AUC diff | 95% CI | R@P80 diff | 95% CI |
|---|---|---|---|---|---|
| HGB | 1 | +0.0004 | [0.0000, +0.0009] | +0.0008 | [-0.0002, +0.0017] |
| HGB | 2 | 0 (same model) | - | 0 | - |
| HGB | 3 | -0.0028 | [-0.0042, -0.0014] | -0.0029 | [-0.0067, +0.0009] |
| HGB | **5** | **+0.0056** | **[+0.0032, +0.0079]** | **+0.0157** | **[+0.0029, +0.0232]** |
| HGB | **8** | **+0.0071** | **[+0.0039, +0.0102]** | **+0.0164** | **[+0.0031, +0.0318]** |
| Keras | 1 | -0.0097 | [-0.0131, -0.0067] | -0.0059 | [-0.0085, -0.0035] |
| Keras | 2 | +0.0081 | [+0.0050, +0.0111] | +0.0052 | [+0.0012, +0.0089] |
| Keras | 3 | -0.0031 | [-0.0050, -0.0011] | -0.0041 | [-0.0099, +0.0012] |
| Keras | 5 | -0.0322 | [-0.0377, -0.0269] | -0.1120 | [-0.4266, -0.0635] |
| Keras | 8 | +0.0004 | [-0.0031, +0.0037] | +0.0458 | [-0.0343, +0.0799] |
| LogReg | 1 | -0.0136 | [-0.0178, -0.0103] | +0.0154 | [+0.0119, +0.0197] |
| LogReg | 2 | -0.0114 | [-0.0168, -0.0068] | +0.0108 | [+0.0038, +0.0163] |
| LogReg | 3 | 0 (same model) | - | 0 | - |
| LogReg | 5 | -0.0048 | [-0.0106, +0.0003] | 0 (P 0.80 not reached) | - |
| LogReg | 8 | +0.0009 | [-0.0042, +0.0060] | 0 (P 0.80 not reached) | - |

Findings:

- **HGB at k=5 and k=8: per-horizon tuning is better.** PR-AUC +0.006 and
  +0.007, recall at 80% precision +1.6 percentage points at both horizons.
  All four intervals exclude zero, and the validation split moved in the
  same direction (+0.007 and +0.009). These are the horizons with the
  longest lead time for a controller to act.
- **HGB at k=1-3: no gain.** k=2 selected the round-1 configuration. k=1 is
  +0.0004. At k=3 the per-horizon winner is 0.003 lower on test; it was
  0.002 higher on the validation subsample (0.833 vs. 0.831), so this is a
  selection that did not carry over to Nov-Dec.
- **Keras: mixed.** Better at k=2, worse at k=1, 3 and 5, no difference at
  k=8. At k=3 and k=8 both searches selected the same configuration, so
  those two differences come only from the training seed (-0.003 and
  +0.0004 PR-AUC). At k=5 the
  per-horizon winner was 0.006 ahead of the round-1 configuration on the
  validation subsample but is 0.028 behind it on the full validation split
  after refit (0.648 vs. 0.675, -0.027) and 0.032 behind on test. Selecting Keras
  configurations on a 10-epoch subsample run did not predict the full
  refit at this horizon.
- **Logistic regression: the per-horizon search picked no PCA and C 0.01 at
  k=1-2.** That trades PR-AUC (-0.014 / -0.011) for recall at 80% precision
  (+0.015 / +0.011).
- The bootstrap intervals cover variation across test days for fixed
  models. They do not cover training randomness. For HGB this is zero
  (deterministic refit). For Keras it is of the order of 0.003 PR-AUC (k=3 and k=8 above).

**Best HGB model per horizon on test (of rounds 1-2):** per-horizon models
at k=5 and k=8 (`models/perk_hgb_k5.joblib`, `perk_hgb_k8.joblib`); the
round-1 configuration at k=3 (`models/tuned_hgb_k3.joblib`); the two are
identical at k=2 and within 0.0004 at k=1. k=5 and k=8 are improved
further in §6.6.

### 6.5 Further test-split diagnostics of the per-horizon models

- **Precision-recall curves**
  (`figures/fig14_perk_pr_curves.png`, k=1, 3, 8). At k=8 HGB holds
  precision 0.80 up to recall 0.36; Keras drops below 0.80 at recall 0.14.
  Logistic regression's precision never exceeds 0.92 at k=1, 0.84 at k=3
  and 0.63 at k=8, even at the lowest recall: its highest-scored rows are
  not more precise than the next ones.
- **Calibration** (`figures/fig15_perk_calibration.png`, 10 bins). All
  three models were trained with balanced class weights, and their scores
  sit well above the observed rate: in the 0.8-0.9 score bin the observed
  bunching rate is 0.08-0.17 (Keras, HGB) and 0.27-0.31 (logistic
  regression). Scores rank rows correctly (PR-AUC above) but
  are not probabilities. Use in operation needs recalibration (e.g.
  isotonic on `val`) or a threshold chosen on precision/recall directly.
- **Per line** (`figures/fig16_perk_pr_auc_by_line.png`, HGB). Line 116 is
  best at every horizon (0.99 at k=1, 0.81 at k=8). Line 474 is weakest,
  and its gap grows with horizon: 0.91 at k=1, 0.55 at k=5, 0.38 at k=8.
  Line 179 is second weakest at k=5-8 (0.69, 0.57). This matches the Phase 6
  finding for line 474 with the default model
  (`reports/phase6_diagnostics.md`).

### 6.6 Long horizons (k=5, 8): features and a second HGB search

The per-horizon search (§6.3-6.4) was followed by a second round for the
two longest horizons only. Code `src/extra_features.py`,
`src/feature_screen.py`, `src/tuning_long_k.py`; results
`data/results/feature_screen.json`, `data/results/tuning_long_k.json`; models
`models/long_hgb_k5.joblib`, `models/long_hgb_k8.joblib` (dict with the
classifier and its feature list). Run time 1.8 h for both horizons.

**New candidate features** (`data/processed/extra_features.parquet`,
joined on `(date, trip_id, stop_sequence)`; `model_dataset.parquet` is
unchanged). Each uses only stop n and earlier, the timetable, or per-stop
statistics from the training months:

| Group | Columns |
|---|---|
| long_lags | `headway_ratio` lag 5 and 8, closing speed over 8 stops, delay growth summed over 8 stops, dwell summed over 3 and 5 stops |
| schedule | scheduled headway (s), scheduled running time from stop n to n+5 and n+8 |
| leader | the leader's headway ratio and speed at stop n, when the leader reached stop n first |
| ahead | summed training-month median dwell and number of interchanges over the next 5 / 8 stops |

Every target row has consecutive stop sequences up to n+k (0.00% gaps), so
the scheduled running time to n+k is pure timetable information.

**Feature screen** (fixed HGB configuration per k, same 800k subsample,
early stopping on `val`, validation PR-AUC):

| Feature set | k=5 | k=8 |
|---|---|---|
| current features | 0.7354 | 0.6104 |
| without `month_of_year` | 0.7357 | 0.6112 |
| + long_lags | 0.7335 | 0.6107 |
| + schedule | 0.7381 | 0.6235 |
| + leader | 0.7348 | 0.6106 |
| + ahead | 0.7370 | 0.6130 |
| + all four groups | 0.7390 | **0.6262** |

At k=8 the new features add +0.016, most of it from the schedule group
(+0.013). At k=5 the best set adds +0.004; the rule for this round was a
minimum gain of 0.005, so k=5 keeps the current features.

**Second HGB search**, per horizon: the §6.3 winner plus 16 random
configurations on a 2M-row subsample, space extended past the edge of the
§6.3 winners (learning rate {0.02, 0.03, 0.05}, leaves {127, 255, 511,
1023}, min leaf {50, 100, 200, 500}, L2 {0, 1, 10}, depth {8, 10, 12,
none}, `max_features` {0.5, 0.8, 1.0}). Early stopping on `val` (Oct)
instead of a random 10% of training rows. The top 3 are refit on the full
training split and the choice is made on the refits. Early stopping and
selection both use `val`, so validation scores are optimistic; test is
scored once.

| k | Features | Selected | Iterations | Val PR-AUC |
|---|---|---|---|---|
| 5 | current (37) | lr 0.02, 1023 leaves, min leaf 100, L2 0, depth 12, all features per split | 326 | 0.759 |
| 8 | current - month + all groups (51) | lr 0.03, 511 leaves, min leaf 50, L2 1, depth 10, 80% features per split | 268 | 0.650 |

Test split, against the per-horizon HGB of §6.3 (paired day-block
bootstrap, 61 days, B = 1000):

| k | Test PR-AUC | Diff | 95% CI | Recall at P 0.80 | Diff | 95% CI |
|---|---|---|---|---|---|---|
| 5 | 0.767 -> **0.776** | +0.0087 | [+0.0066, +0.0107] | 0.602 -> 0.608 | +0.0064 | [-0.0014, +0.0165] |
| 8 | 0.659 -> **0.682** | +0.0236 | [+0.0193, +0.0277] | 0.359 -> **0.387** | +0.0272 | [+0.0088, +0.0422] |

Against the default model of §5 this is +0.080 at k=5 (0.696 -> 0.776) and
+0.059 at k=8 (0.623 -> 0.682). At k=8, the new features gave +0.016
validation PR-AUC in the screen at a fixed configuration; at k=5 the
features are unchanged, so the gain there comes from the search itself
(early stopping on `val`, larger subsample, wider space, choice on full
refits).

Tested, not worth it: a weighted average of the HGB and Keras scores
(+0.001 test PR-AUC at k=8, weight 1.0 = HGB alone at k=5;
`src/ensemble_perk.py`); the new features at k=5 (+0.004 validation).

### 6.7 Limitations

- Validation (Oct) contains no cold-weather rows (§1.1); all selections
  were made without cold weather.
- Searches are 12-23 configurations per model and horizon, on an 800k-row
  subsample.
- The Phase 6 diagnostics (`reports/phase6_diagnostics.md`) use the
  default HGB model and have not been re-run with the tuned models.
