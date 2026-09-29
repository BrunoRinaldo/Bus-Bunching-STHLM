# Split strategy comparison: temporal vs. random vs. weekly

Tests whether the plan's "never split randomly" rule and the cold-weather
validation gap (docs/03 §1.1) actually matter in practice, not just in
principle. Same model (`HistGradientBoostingClassifier`), same horizon
(k=3, the core RQ3 horizon), same features, same class-weight handling -
only the train/val/test assignment differs. Code: `src/split_variants.py`.
Raw numbers: `reports/split_comparison.json`. Fitted models saved as
`models/hgb_k3_split_random.joblib` and `models/hgb_k3_split_weekly.joblib`.

## The three splits

| | Assignment rule | Test set covers |
|---|---|---|
| **Temporal** (primary, used everywhere else in this project) | Jan-Sep train, Oct val, Nov-Dec test, by calendar month | Nov-Dec only, one contiguous block |
| **Random (trip-level)** | `hash(date, trip_id) % 100`: <75 train, <85 val, else test - whole trips stay together, scattered across all of 2024 | ~15% of trips sampled from all 12 months |
| **Weekly** | ISO week number round-robin: 7 of every 10 weeks train, 1 val, 2 test | ~20% of weeks, scattered across all 12 months, whole weeks together |

The random variant is the *trip-level* version, not row-level: assigning
individual stops of the same trip to different splits would be a much worse
form of leakage than what is tested here (adjacent stops of one trip are
near-duplicates in feature space), so this is deliberately the more
defensible of the two ways to split randomly, to avoid making the random
split look worse than it needs to.

## Headline result

| Split | Test PR-AUC (whole test set) | Test PR-AUC (Nov-Dec rows only) |
|---|---|---|
| Temporal | **0.8136** | 0.8136 (its test set *is* Nov-Dec) |
| Random | 0.7915 | **0.8207** |
| Weekly | 0.7876 | 0.7995 |

**The two columns answer different questions and must not be conflated.**

The left column (whole test set) looks like temporal wins by ~0.02-0.03
PR-AUC - the opposite of the naive expectation that a leakier split
inflates scores. But that comparison is confounded: temporal's test set is
Nov-Dec only, while random and weekly's test sets are scattered across the
whole year, including summer months whose looser timetables produce very
different bunching dynamics (Phase 3 Figure 3: July's rate is 0.65% against
a 2-4.5% range elsewhere). A whole-year test set is simply a different,
broader population to be scored on - not necessarily a harder or easier one
in any way that says anything about leakage.

The right column controls for that: it re-scores each model on **only the
Nov-Dec rows of its own test set**, the same calendar window temporal is
scored on. This is the cleaner leakage test, and it shows:

- **Random: 0.8207 vs. temporal 0.8136 - a small, ~0.007 (0.9% relative)
  advantage for the leakier split.** This is the direction leakage should
  push, and it is real, but it is modest. The plan's "never split randomly"
  warning is right in principle - a random split lets the model train on
  trips from the same days it is tested on, which a deployed model can never
  do - but in this dataset, with this model, the practical inflation from
  that leakage is small enough not to change any conclusion drawn from the
  primary temporal-split results.
- **Weekly: 0.7995 vs. temporal 0.8136 - lower, not higher.** This is *not*
  evidence that the weekly split is "more honest" or "stricter" than
  temporal. The Nov-Dec subset of the weekly test set is only 285,970 rows
  drawn from a handful of whole weeks (test weeks are 2 of every 10), so
  its score depends heavily on which specific weeks happened to land in
  test - e.g. it may include a data-outage or unusual-disruption week (Phase
  3's null-rate table shows several such single-day spikes in November).
  Random, by contrast, samples ~15% of trips *uniformly across every day*, so
  its Nov-Dec estimate averages over the whole period and is much more
  stable. The 0.014 gap between weekly and temporal is within what a
  handful-of-weeks sampling effect can plausibly produce and should not be
  read as a real difference in split quality without more repeats.

**The most important takeaway is not any one number but that all three land
in a tight 0.79-0.82 band**: the project's headline conclusion (gradient
boosting at k=3 reaches roughly 0.8 PR-AUC) does not depend meaningfully on
which of these three reasonable split strategies is used. The temporal
split remains the right primary choice for the reason it was chosen - it is
the only one of the three that tests genuine forward prediction, i.e. what a
deployed system would actually face - but its results are not an artifact of
that choice.

## The weather-coverage question (the original motivation)

The concern that motivated this comparison: October, the temporal split's
validation month, has almost no cold weather (docs/03 §1.1). Confirmed and
quantified below across all three splits' validation sets:

| Split | Validation min temp | Validation % snowing | Validation % ice risk |
|---|---|---|---|
| Temporal (Oct) | +1.1 degC | **0.00%** | **0.00%** |
| Random | -18.9 degC | 3.88% | 3.84% |
| Weekly | -7.4 degC | 2.81% | 3.14% |

The temporal validation set has **literally zero** snowing or ice-risk rows
- it is completely blind to cold-weather conditions, not just
under-representative. Both alternative splits fix this: random restores
essentially full-year weather coverage (even the -18.9 degC January extreme),
and weekly gets most of the way there (snow/ice rates close to the
temporal *test* set's own 4.2%/3.9%, though not quite the full -18.9 degC
extreme, since only a handful of validation weeks are sampled).

**But the weekly split has a mirror-image problem on its test set:**

| Split | Test min temp | Test % snowing | Test % ice risk |
|---|---|---|---|
| Temporal (Nov-Dec) | -8.9 degC | 4.20% | 3.90% |
| Random | -18.9 degC | 3.95% | 3.89% |
| Weekly | -8.9 degC | **0.93%** | **0.95%** |

The weekly split's test set has only ~0.9% snowing rows against the
temporal split's 4.2% - the weekly split fixes the *validation* weather gap
but ends up with a test set far less winter-heavy than temporal's, making it
a weaker test of H4 (does weather change what matters for prediction) than
the primary split despite being better on validation coverage. Which
weeks land in test is a matter of the round-robin arithmetic, not a design
choice about season, and this particular arithmetic happens to place most
test weeks outside winter.

## What this means for the rest of the project

- **Keep the temporal split as primary**, with the same justification as
  before: it is the only one of the three that tests real forward
  prediction. This comparison strengthens rather than weakens that choice -
  it shows the temporal results are not an artifact of the split, and that
  the leakage the plan warned about, while real, is small here.
- **State the validation-set weather blindness as a documented limitation**
  (now with hard numbers - 0.00% snow/ice in validation - not just a
  qualitative concern), specifically for the Keras model whose early
  stopping was chosen against that validation set (docs/03 §1.1). The other
  models did not use validation for any decision, so this affects them less.
- **Do not adopt the weekly split as a replacement.** It fixes the
  validation problem but creates an equivalent test-set problem of its own,
  and its small-number-of-weeks structure makes its estimates noisier.
- If a second validation-coverage fix is wanted, the cleanest option is not
  a different split of all the data but a **targeted supplementary check**:
  score the already-trained temporal-split models on a small, deliberately
  cold-weather subset carved out of the train period's coldest weeks
  (January-February) - not for tuning, only to confirm the temporal split's
  decisions did not quietly under-serve cold conditions. Not run here.

## Caveats stated plainly

- One horizon (k=3), one model (HGB), one random seed per split. The 0.007
  random-vs-temporal gap and especially the 0.014 weekly-vs-temporal gap
  are small enough that a different random seed or a different set of test
  weeks could plausibly move them - these are single-run estimates, not
  averaged over repeats, and should be read as indicative, not exact.
- The whole-year PR-AUC column is not a like-for-like comparison and is
  included for transparency about what a naive comparison would have
  suggested, not as evidence.
- Only HGB was retrained. The other three models (logistic regression,
  Keras, and the persistence baseline) were not re-evaluated under these
  alternative splits.
