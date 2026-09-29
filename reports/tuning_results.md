# Hyperparameter tuning and PCA

What was tuned, what was tried on the logistic regression (including PCA),
and what it bought. All headline numbers are **test PR-AUC (Nov-Dec 2024)**;
every choice was made on the **validation split (Oct)**, and the test split
was scored only for the final chosen configurations. Code:
`src/tuning.py` (gradient boosting and logistic regression),
`src/tuning_keras.py` (Keras, run under `.venv312`), `src/compare_tuned.py`
(figure and table). Raw numbers: `reports/tuning_stage1.json` (k=3, full
search log), `tuning_stage2.json` (other horizons), `tuning_keras.json`.
Tuned models: `models/tuned_hgb_k3.joblib`,
`models/tuned_logreg_k3_{base_C,base_pca,eng_C,eng_pca}.joblib`,
`models/tuned_sequence_k{1,2,3,5,8}.keras`. Figure:
`figures/fig9_tuned_pr_auc_vs_horizon.png`.

## Headline

| k | min | HGB orig | HGB tuned | Keras orig | Keras tuned | LogReg orig | LogReg tuned |
|---|---|---|---|---|---|---|---|
| 1 | 1.05 | 0.898 | **0.958** | 0.917 | 0.932 | 0.586 | 0.859 |
| 2 | 2.10 | 0.860 | **0.905** | 0.855 | 0.862 | 0.563 | 0.785 |
| 3 | 3.15 | 0.814 | **0.856** | 0.786 | 0.815 | 0.539 | 0.721 |
| 5 | 5.25 | 0.696 | **0.762** | 0.682 | 0.709 | 0.493 | 0.612 |
| 8 | 8.40 | 0.623 | **0.652** | 0.562 | 0.579 | 0.425 | 0.488 |

- **Tuned gradient boosting is best at every horizon**, including k=1, where
  the original Keras model had been ahead (0.917 vs 0.898). The earlier
  "Keras and GBM are essentially tied" reading (docs/03 §5) was partly a
  statement about untuned defaults.
- **Gradient boosting gained the most** (+0.03 to +0.07), **Keras the least**
  (+0.01 to +0.03), and **logistic regression the most in relative terms**
  (+0.06 to +0.27), for a reason that has nothing to do with tuning (below).
- **Gains shrink with horizon** for every model. At k=8 tuning buys little
  (+0.03 GBM), which fits the picture that the far horizon is limited by
  how much signal exists, not by model settings.
- Recall at 80% precision, tuned GBM: 0.96 (k=1), 0.88 (k=2), 0.79 (k=3),
  0.59 (k=5), **0.34 (k=8)**. The original GBM could not sustain 80%
  precision much beyond k=5 (0.20 at k=8, docs/03 §5). Tuned Keras reaches
  0.93 / 0.84 / 0.74 / 0.46 / only 0.09 at the same horizons.

## 1. Gradient boosting

Random search, 12 configurations plus the original settings as a reference,
on an 800,000-row random subsample of the training split (class ratio
preserved, class weights as before, no resampling), each scored on the full
validation split. Space: learning rate {0.03, 0.05, 0.1, 0.2}, max leaf
nodes {15, 31, 63, 127, 255}, min samples per leaf {20, 50, 100, 300, 1000},
L2 {0, 0.1, 1, 10}, max depth {none, 6, 10}; up to 500 iterations with early
stopping.

Winner: **learning rate 0.2, 255 leaves, min leaf 20, L2 = 10, no depth
limit** (validation 0.831 on the subsample vs. 0.800 for the original
settings). Refit on the full training split: validation 0.846, test 0.856.
The same configuration was then reused, without re-searching, at the other
four horizons.

Caveats: 12 configurations is a small search. The winner sits at the
**edge of the space** (largest learning rate and largest leaf count), so more
capacity may well help further; that was not explored. The configuration was
chosen at k=3 only; k=8 gained least, and a horizon-specific search might
move it. Single seed.

## 2. Logistic regression: C, PCA, and what actually mattered

Four arms, each swept on the 800k subsample and scored on validation, then
refit on the full training split (k=3):

| Arm | Validation | Test |
|---|---|---|
| original features, C tuned | 0.519 | 0.539 |
| original features + PCA | 0.519 | 0.539 |
| + \|x\| features, C tuned | 0.707 | 0.707 |
| + \|x\| features + PCA (45 comps) | 0.713 | **0.721** |
| (original logreg, for reference) | 0.519 | 0.539 |

**Regularisation strength did nothing.** C from 0.01 to 10 changed
validation PR-AUC in the fourth decimal (0.5193-0.5194). With ~7 million
rows and ~75 features there is nothing for a penalty to prevent.

**PCA never helped on its own.** On the original features validation
PR-AUC was 0.471 with 5 components, 0.468 with 10, 0.506 with 20, 0.508 with
30, 0.515 with 45, and only reached the no-PCA score (0.519) at 60
components, where 100% of variance is kept. That is what a linear
projection should do to a linear classifier: it cannot add modelling
capacity, it can only throw some away.

**The real bottleneck was a missing feature, not the estimator.** "Bunched"
means `abs(headway_ratio) < 0.25`, which is a *U-shaped* function of the
signed ratio: a straight-line coefficient on `headway_ratio` cannot say
"close to zero on either side". Adding `abs(headway_ratio)`, its three lags,
near-zero indicator flags for each (`abs(x) < 0.25`), and
`abs(closing_speed_3)` lifted test PR-AUC from 0.539 to 0.707 at k=3
(+0.17), and recall at 80% precision from "never reaches it" to 0.53. The
tree model and the neural network can learn a U-shape themselves, which is
why they never had this handicap. This also corrects an earlier claim:
docs/03 §5.2 said logistic regression's weakness was "itself informative
about how nonlinear the true relationship is". Most of the gap was this one
representation problem, which I should have engineered from the start - it
says little about the data's overall nonlinearity.

**PCA on top of the engineered features gave a small extra gain**
(0.707 to 0.721 test; 0.707 to 0.713 validation), in the same direction on
both splits, but it is modest, single-run, and non-monotone: with 30
components or fewer the score fell back to ~0.60, and it recovered only at
45+ (99% of variance). The plausible reading is that the near-zero flags
carry low-variance directions that a coarse PCA discards, and that trimming
the last tiny-variance directions at 45 slightly improves conditioning.
That is an interpretation, not something tested. It should be reported as
"small and not robust", not as PCA fixing logistic regression.

The "tuned" logistic regression at other horizons reuses the k=3
configuration (engineered features, PCA 45, C = 0.1) without re-tuning. Even
tuned it trails gradient boosting at every horizon and cannot reach 80%
precision at all at k=5 and k=8.

## 3. Keras

Random search, 9 configurations plus the original as reference, on the same
800k-row subsample, up to 10 epochs with early stopping on validation
PR-AUC. Space: CNN or LSTM, 1-3 conv layers, 16/32/64 filters, kernel 2/3,
**pooling {global average, flatten, last step}**, dense 16/32/64, dropout
0.1/0.2/0.4, learning rate {3e-4, 1e-3, 3e-3}, batch {2048, 4096, 8192}.

Validation PR-AUC on the subsample: original config 0.573; best config
**0.782** - three causal conv layers, 64 filters, kernel 3, **last-step
pooling**, dense 32, dropout 0.4, lr 1e-3, batch 2048. The
average-pooling configurations were the weakest of the search (0.46-0.67).
That matches the reason pooling was added to the space (averaging over the
window discards which step is the current one), but a random search does not
attribute the gain to any single choice - depth, width, dropout and batch size
changed at the same time, so this is a "the whole configuration is better"
result, not "pooling explains it". LSTMs landed in the middle (0.71-0.77).
Refit on full data for all 5 horizons with 15 epochs and early stopping. As
with gradient boosting, the winner is at the edge of the space (deepest,
widest) and the same configuration is reused across horizons.

## Caveats, all in one place

- **Validation has no cold-weather rows** (docs/03 §1.1: 0.00% snowing / ice
  in October). Every configuration was selected on it, so none was selected
  against cold conditions. The test split (4.2% snowing) is the check, and
  the tuned models hold up there: test is at or above validation at every
  horizon for gradient boosting and Keras, as it was for the original
  models; the one exception is logistic regression at k=1 (test 0.859 vs.
  validation 0.869). But this is a real gap in how the choices were made.
- **Small searches, subsampled, single seed.** 12 / 9 configurations on 800k
  rows; both winners at the edge of their spaces. These are improvements,
  not optima.
- **Configurations chosen at k=3 and reused at k=1, 2, 5, 8.**
- **The Phase 6 diagnostics (`reports/phase6_diagnostics.md`) were computed
  with the original, untuned gradient boosting model** and were not re-run
  on the tuned one. Their conclusions (line 474 as the weak spot, delay
  beating dwell in the ablation, graceful degradation without leader
  features, clean line transfer) are qualitative enough that a moderately
  better model is unlikely to reverse them, but that has not been checked.
- **The original persistence baseline is unchanged** (no tunable
  parameters); it is not in the tuned table. Tuned models are compared with
  their own originals above, not re-plotted against persistence.
- The engineered `|x|` features were added for logistic regression only.
  Adding them to the tree and neural models is unnecessary (they can form
  the U-shape themselves) and was not tested.
