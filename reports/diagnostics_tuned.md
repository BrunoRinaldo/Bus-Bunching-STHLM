# Diagnostics on the tuned gradient boosting models (k=3, k=5)

Re-run of the Phase 6 diagnostics (`reports/phase6_diagnostics.md`, default
model at k=3) on the best tuned gradient boosting model per horizon:

| k | Model | Test PR-AUC (95% CI) |
|---|---|---|
| 3 | `models/tuned_hgb_k3.joblib` (docs/03 §6.2) | 0.856 [0.848, 0.865] |
| 5 | `models/long_hgb_k5.joblib` (docs/03 §6.6) | 0.776 [0.765, 0.788] |

Every diagnostic that trains a new model (ablation, no-leader, transfer)
uses that horizon's tuned configuration and training procedure. All numbers
are on the test split (Nov-Dec 2024). Intervals are 95% day-block bootstrap
intervals (61 test days, B = 1000). Code `src/diagnostics_tuned.py`; raw
numbers `reports/diagnostics_tuned.json`.

## 1. Per line

| Line | n (k=3) | base rate % | k=3 PR-AUC | 95% CI | k=5 PR-AUC | 95% CI | default model k=3 |
|---|---|---|---|---|---|---|---|
| 116 | 113,797 | 1.49 | **0.933** | [0.910, 0.952] | **0.885** | [0.848, 0.916] | 0.922 |
| 607 | 224,707 | 2.00 | 0.890 | [0.873, 0.907] | 0.818 | [0.795, 0.842] | 0.877 |
| 541 | 179,168 | 1.95 | 0.885 | [0.863, 0.906] | 0.827 | [0.789, 0.859] | 0.797 |
| 4 | 272,890 | 2.88 | 0.861 | [0.847, 0.876] | 0.786 | [0.767, 0.806] | 0.846 |
| 117 | 163,692 | 3.44 | 0.855 | [0.838, 0.871] | 0.774 | [0.753, 0.796] | 0.804 |
| 401 | 105,475 | 1.41 | 0.849 | [0.812, 0.878] | 0.795 | [0.749, 0.831] | 0.809 |
| 179 | 156,873 | 3.10 | 0.817 | [0.800, 0.836] | 0.701 | [0.673, 0.729] | 0.777 |
| 474 | 112,923 | 2.12 | **0.708** | [0.671, 0.747] | **0.567** | [0.523, 0.615] | 0.628 |

Line 474 is the weakest line at both horizons, and its interval does not
overlap any other line's. Tuning raised every line; 474 remains 0.11 below
the next line at k=3 and 0.13 below at k=5. Lines 117 and 179 have the
highest base rates and score above 474.

## 2. Peak, month, weather

| Group | k=3 PR-AUC | 95% CI | k=5 PR-AUC | 95% CI |
|---|---|---|---|---|
| off-peak (base rate 1.5%) | 0.868 | [0.858, 0.880] | 0.794 | [0.779, 0.811] |
| peak 7-8, 16-17 h (base rate 4.8-5.0%) | 0.844 | [0.834, 0.854] | 0.759 | [0.745, 0.773] |
| November | 0.857 | [0.846, 0.867] | 0.776 | [0.759, 0.792] |
| December | 0.854 | [0.842, 0.868] | 0.776 | [0.760, 0.793] |
| clear | 0.854 | [0.837, 0.872] | 0.774 | [0.751, 0.798] |
| cloudy | 0.858 | [0.847, 0.868] | 0.779 | [0.764, 0.793] |
| drizzle | 0.858 | [0.844, 0.878] | 0.785 | [0.764, 0.811] |
| snow (n = 55,197 at k=3) | 0.845 | [0.798, 0.902] | 0.755 | [0.697, 0.814] |
| rain (n = 3,620) | 0.791 | too few days | 0.791 | too few days |

- **Peak is harder than off-peak for the tuned models**: -0.024 at k=3 and
  -0.035 at k=5, with non-overlapping intervals. The default model showed
  no difference (0.811 vs. 0.817).
- No difference between November and December.
- Snow is 0.009 (k=3) and 0.019 (k=5) below clear; the snow interval is
  wide (few snow days) and overlaps clear at both horizons.

## 3. Ablation (H2): delay vs. dwell

Same tuned configuration, retrained with the base features plus only the
named group:

| Feature set | k=3 | k=5 | default model k=3 |
|---|---|---|---|
| delay + dwell (full tuned model) | **0.856** | **0.776** | 0.813 |
| delay only | 0.851 | 0.772 | 0.819 |
| dwell only | 0.843 | 0.759 | 0.803 |

Delay features beat dwell features at both horizons (+0.008 at k=3, +0.013
at k=5). With the tuned model, adding dwell to delay improves PR-AUC
(+0.005 at k=3, +0.004 at k=5); with the default model it did not.

## 4. Robustness: no leader features

| | k=3 | k=5 |
|---|---|---|
| full tuned model | 0.856 | 0.776 |
| without `leader_dwell`, `leader_arrival_delay`, `leader_delay_growth` | 0.837 | 0.753 |
| relative loss | 2.2% | 3.0% |

The default model lost 4.3% at k=3.

## 5. Transfer to unseen lines

Trained on lines 4, 541, 474, 607 only; scored on the test rows of the same
four lines and of the four held-out lines (116, 117, 179, 401):

| | k=3 | k=5 |
|---|---|---|
| transfer model, own lines | 0.859 [0.847, 0.871] | 0.775 [0.758, 0.793] |
| transfer model, held-out lines | 0.826 [0.814, 0.838] | 0.719 [0.699, 0.738] |
| full 8-line model, same held-out lines | 0.854 | 0.773 |

- **The tuned model loses accuracy on unseen lines**: held-out lines score
  0.033 (k=3) and 0.056 (k=5) below the transfer model's own lines, with
  non-overlapping intervals. On the held-out lines, a model that trained on
  them scores 0.028 (k=3) and 0.054 (k=5) higher than one that did not.
- The default model showed no transfer loss (0.770 own vs. 0.773 held-out).
- Single 4-vs-4 split; all 8 lines are high-frequency trunk routes.

## 6. Where the model fails: false negatives at threshold 0.5

| | k=3 | k=5 |
|---|---|---|
| recall at 0.5 | 96.0% | 93.0% |
| false negatives / true positives | 1,263 / 30,641 | 2,093 / 27,780 |
| line 474: share of false negatives vs. true positives | 20.7% vs. 7.0% | 21.4% vs. 6.2% |
| peak share of false negatives vs. true positives | 51.3% vs. 55.5% | 51.1% vs. 55.5% |
| mean score of false negatives | 0.18 | 0.21 |

Line 474 supplies three times its share of misses at both horizons.

## 7. Permutation importance (60,000 test rows, PR-AUC, 5 repeats)

| Rank | k=3 | importance | k=5 | importance |
|---|---|---|---|---|
| 1 | `headway_ratio` | 0.791 | `headway_ratio` | 0.693 |
| 2 | `leader_arrival_delay` | 0.045 | `leader_arrival_delay` | 0.105 |
| 3 | `route_frac` | 0.019 | `LineNumber` | 0.026 |
| 4 | `leader_delay_growth` | 0.014 | `route_frac` | 0.019 |
| 5 | `LineNumber` | 0.011 | `leader_delay_growth` | 0.017 |
| 6 | `observed_departure_delay` | 0.009 | `hour_of_day` | 0.016 |

- Every weather variable (temperature, precipitation, wind, snow, ice,
  `wx_group`) scores between -0.001 and +0.0003 at both horizons: no
  measurable contribution.
- `closing_speed_3`: 0.004 (k=3), 0.006 (k=5). `month_of_year`: 0.000.
- From k=3 to k=5 the current headway loses weight (0.79 to 0.69) and the
  leader's delay gains (0.05 to 0.11).

## 8. Latency

| | k=3 | k=5 |
|---|---|---|
| single-row prediction | 6.3 ms | 8.6 ms |
| batched, per row | 0.005 ms | 0.022 ms |
