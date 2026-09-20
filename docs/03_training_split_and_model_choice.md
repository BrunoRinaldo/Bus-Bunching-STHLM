# Training split and model choice

This document covers how `data/processed/model_dataset.parquet` is split for
training and evaluation, why that split is structured the way it is, and the
four models called for by the plan - all four have now been trained and
evaluated; §5 has the full results, the headline figure, and two findings
that contradict what the plan expected going in.

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
table). Full numbers: `reports/results_tables.md`.

**All numbers in this section use the original default settings** (and, for
logistic regression, the original feature set). Hyperparameter tuning and a
PCA / engineered-feature study were run afterwards - see
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
