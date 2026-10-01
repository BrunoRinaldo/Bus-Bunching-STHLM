# Phase 6 - diagnostics (report task D)

Primary diagnostic model: `HistGradientBoostingClassifier` at k=3 (the core
RQ3 horizon) - confirmed the strongest tabular model in doc 3 §5, and
running every diagnostic across all 5 horizons would be excessive scope for
this phase. All numbers below are on the **test split** (Nov-Dec 2024),
scored once, using the same model unless a section says otherwise. Code:
`src/diagnostics.py`. Raw numbers: `reports/phase6_diagnostics.json`.

**Note added after the tuning work**: this whole report was computed with the
original, untuned gradient boosting model. Tuning (`reports/tuning_results.md`)
raised k=3 test PR-AUC from 0.814 to 0.856; these diagnostics were not re-run
on the tuned model. **They have since been re-run on the tuned models at k=3
and k=5: see `reports/diagnostics_tuned.md`.** Three conclusions below change
there: the tuned model shows a transfer loss to unseen lines (§4), peak hours
score lower than off-peak (§1), and delay + dwell beats delay alone (§2).

As a sanity check, the full model re-evaluated here scores PR-AUC 0.8136 -
matches doc 3 §5's reported 0.814 for HGB at k=3 to within rounding, so the
diagnostics below are being computed on the same model, not a drifted copy.

---

## 1. Performance broken out by line, hour, month, weather

**By line** - the widest split of any breakdown in this report:

| Line | n | base rate % | PR-AUC |
|---|---|---|---|
| 116 | 113,797 | 1.49 | **0.922** |
| 607 | 224,707 | 2.00 | 0.877 |
| 4 | 272,890 | 2.88 | 0.846 |
| 401 | 105,475 | 1.41 | 0.809 |
| 117 | 163,692 | 3.44 | 0.804 |
| 541 | 179,168 | 1.95 | 0.797 |
| 179 | 156,873 | 3.10 | 0.777 |
| 474 | 112,923 | 2.12 | **0.628** |

Line 474 is a genuine outlier - 0.628 PR-AUC against 0.78-0.92 for every
other line, a gap far larger than sampling noise given 112,923 rows. It is
not simply "the hardest line because it has the most bunching" (117 and 179
have higher base rates and score much better); something about line 474
specifically is harder for this feature set to predict. Line 116, by
contrast, is both the lowest-bunching line and the easiest to predict -
consistent with a line where bunching happens for a narrower, more
learnable set of reasons rather than being pervasively noisy.

**By peak/off-peak:**

| | n | base rate % | PR-AUC |
|---|---|---|---|
| off-peak | 963,092 | 1.48 | 0.817 |
| peak | 366,433 | 4.82 | 0.811 |

Peak hours have 3.3x the bunching rate of off-peak (consistent with Phase 3
Figure 2/5), but are not meaningfully easier or harder to predict - PR-AUC is
essentially flat across the two. More bunching does not mean more
*predictable* bunching.

**By month** (test split only covers Nov-Dec, so this is a narrow check, not
a full seasonal comparison):

| Month | n | base rate % | PR-AUC |
|---|---|---|---|
| Nov | 693,895 | 2.65 | 0.815 |
| Dec | 635,552 | 2.13 | 0.811 |

No meaningful difference between the two available test months.

**By weather:**

| wx_group | n | base rate % | PR-AUC |
|---|---|---|---|
| clear | 284,991 | 2.62 | 0.817 |
| snow | 55,197 | 3.17 | 0.814 |
| cloudy | 903,475 | 2.28 | 0.814 |
| drizzle | 81,095 | 2.50 | 0.812 |
| rain | 3,620 | 0.83 | 0.774 |

Performance is essentially weather-invariant across the four conditions with
real sample size - snow (n=55,197) scores within 0.003 of clear, which is
reassuring given the winter-generalisation question (H4) and the validation-
set weather-coverage gap flagged separately (doc 3 §1.1's val-set caveat).
`rain`'s lower score (0.774) should not be over-read: n=3,620 is two orders
of magnitude smaller than every other category and also has the lowest base
rate (0.83%) of any group in this table, so this could easily be a small-
sample effect rather than a real rain-specific weakness.

---

## 2. Ablation for H2 (delay-only vs. dwell-only vs. both)

| Feature set | n features | PR-AUC |
|---|---|---|
| delay-only | 34 | **0.8188** |
| both | 37 | 0.8132 |
| dwell-only | 32 | 0.8034 |
| (full model, for reference) | 37 | 0.8136 |

All three arms share the same base feature set (current state minus
delay/dwell, trajectory, leader position minus delay/dwell, route position,
context, stop attributes, disruption, weather - i.e. everything except the
5 delay features and 3 dwell features being ablated) plus only the named
group added back.

**This does not support H2.** Delay-only features alone (0.8188) slightly
*outperform* the combined delay+dwell feature set (0.8132), and dwell-only
features are clearly the weakest of the three (0.8034). Adding dwell
features on top of delay features does not help, and mildly hurts - a
plausible reason is `dwell`'s ~16% missingness (doc 1 §4.2) adding noise
without adding enough signal to offset it, though that is offered as a
plausible explanation, not a confirmed one. Combined with the doc 3 §5.2
permutation-importance finding (delay-based features outrank dwell-based
ones at k=3), this is now two independent pieces of evidence pointing the
same direction: **at this feature resolution, delay predicts bunching better
than dwell does, which is the opposite of what H2 predicts.** This should be
reported as a negative finding for H2, not softened - it is one of the more
interesting results in the whole project precisely because it contradicts
the plan's stated mechanism story, and the report should engage with why
(dwell may matter more than a single-stop dwell reading shows - the
project's `cum_delay_growth_3` rolling window exists for exactly this
reason, but there is no equivalent rolling *dwell* feature to test the same
idea on the dwell side - a gap worth naming as a follow-up rather than
papering over the asymmetry).

---

## 3. Robustness: dropping the leader's features

Simulates a real-time feed that has lost the leading vehicle (a real
failure mode - a single dropped GPS/AVL unit removes one trip's data, but
that trip is *someone else's* leader at every downstream stop it would have
reported).

| | PR-AUC |
|---|---|
| full model | 0.8136 |
| without `leader_dwell`, `leader_arrival_delay`, `leader_delay_growth` | 0.7786 |
| degradation | 4.31% relative |

A real but moderate cost, not a collapse. The model degrades gracefully when
leader information disappears rather than failing catastrophically, which
matters operationally: a deployed system should keep producing usable
(if weaker) predictions during a partial feed outage rather than going
blind.

---

## 4. Transfer: line holdout (RQ4)

Trained on the 4 highest-volume lines (4, 541, 474, 607), scored on the
other 4 (116, 117, 179, 401), never seen during this training run:

| | PR-AUC |
|---|---|
| scored on its own (training) lines' test rows | 0.7696 |
| scored on the 4 held-out lines | 0.7726 |
| degradation | **-0.39%** (i.e. no degradation) |

The model performs marginally *better* on lines it never trained on than on
its own training lines - well within noise, but the headline result is
unambiguous: **no meaningful transfer penalty** between these 8 lines. This
is a genuinely positive RQ4 result, with the caveat already noted in doc 3
§1.2: all 8 lines are high-frequency trunk routes selected the same way
(Phase 0 §2.2), so this shows generalisation *within* that regime, not to
SL's lower-frequency network generally. Both PR-AUC values here (~0.77) are
lower than the full 8-line model's 0.814, simply because this transfer model
trained on half the data (4 lines instead of 8) - the relevant comparison is
own-lines vs. held-out-lines for the *same* transfer model, not either
number against the full model.

---

## 5. Scalability

| | value |
|---|---|
| single-row `predict_proba` call | 6.10 ms |
| batched (20,000 rows in one call) | 0.006 ms/row |

Single-row latency of ~6ms is comfortably within real-time budget for a
per-arrival prediction (most of this is Python/DataFrame call overhead, not
raw model inference - the batched figure shows the model itself is
essentially free at ~1000x higher throughput per row). A deployed system
that scores all active vehicles on each GTFS-RT update (batched, not one
`predict_proba` call per vehicle) would not be latency-constrained by this
model at any realistic SL fleet size.

---

## 6. Where the model fails: false negatives

At the default 0.5 probability threshold, full 8-line test set, k=3:
1,145 false negatives against 30,759 true positives (96.4% recall at this
threshold - class weighting makes the model liberal about flagging bunching,
consistent with doc 3 §4's framing that recall/precision tradeoff is a
deliberate operating-point choice, not fixed).

| Line | share of false negatives | share of true positives |
|---|---|---|
| 117 | 20.1% | 17.5% |
| **474** | **19.6%** | **7.1%** |
| 4 | 16.2% | 24.9% |
| 179 | 12.2% | 15.4% |
| 401 | 10.8% | 4.4% |
| 607 | 10.5% | 14.2% |
| 541 | 7.9% | 11.1% |
| 116 | 2.8% | 5.4% |

Line 474 is heavily overrepresented among misses relative to correct
catches (19.6% of false negatives vs. only 7.1% of true positives) -
convergent with §1's finding that line 474 has by far the worst per-line
PR-AUC. This is the clearest single "where does the model fail" answer in
this project: **line 474 specifically**, not a time-of-day or weather
pattern. Line 4, conversely, is under-represented among misses (16.2%
of false negatives vs. 24.9% of true positives) - the model's single best
line for actually catching true bunching events, plausibly helped by having
the most training volume and the clearest bunching signal (Phase 3 Figure
2/4).

False negatives are *not* concentrated at peak hours (51.4% of misses vs.
55.5% of correct catches happen at peak) - if anything slightly less than
proportionally represented, contradicting a naive assumption that rush-hour
chaos is where the model struggles most.

The mean predicted probability among false negatives is 0.197 - not near
zero. The model is not being blindsided by these cases (a true blind spot
would show predicted probabilities clustered near 0); it is assigning them
real but sub-threshold probability, consistent with genuinely ambiguous
or borderline cases rather than a systematic failure to see the signal at
all. That distinction matters for the deployment discussion: these are
cases a lower decision threshold (accepting more false alarms for more
recall - see doc 3 §4's precision/recall operating-point framing) would
likely catch, not cases the feature set is blind to.

---

## Summary for the report

- Line 474 is the model's clear weak point (§1 and §6), not any time-of-day
  or weather condition.
- H2 does not hold at this feature resolution: delay features beat dwell
  features, and combining them does not beat delay alone (§2) - a genuine
  negative finding, consistent with doc 3 §5.2's independent
  permutation-importance evidence.
- The model degrades gracefully, not catastrophically, when the leader's
  data is missing (§3) - a real but moderate ~4% PR-AUC cost.
- The model transfers cleanly across held-out lines within this 8-line,
  high-frequency-trunk-route regime (§4) - a genuinely positive RQ4 result,
  bounded by the caveat that all 8 lines share that regime.
- Inference is not a deployment bottleneck at any credible scale (§5).
