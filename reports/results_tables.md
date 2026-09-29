# Phase 5 results table

Test split (Nov-Dec 2024), all 8 lines pooled. k in stops, minutes via median
inter-stop running time (63s). Persistence baseline has no probability
score so PR-AUC is not defined for it; its precision/recall operating
point is reported instead and marked separately on the figure.

| k | minutes | persistence P/R | logreg PR-AUC | HGB PR-AUC | Keras PR-AUC | logreg R@P80 | HGB R@P80 | Keras R@P80 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.05 | 0.89/0.84 | 0.5855 | 0.8984 | 0.917 | n/a | 0.9524 | 0.9274 |
| 2 | 2.1 | 0.84/0.75 | 0.5625 | 0.8603 | 0.8546 | n/a | 0.8649 | 0.8322 |
| 3 | 3.15 | 0.81/0.67 | 0.5387 | 0.8136 | 0.786 | n/a | 0.7517 | 0.6904 |
| 5 | 5.25 | 0.75/0.55 | 0.4926 | 0.6959 | 0.6815 | n/a | 0.4701 | 0.3581 |
| 8 | 8.4 | 0.67/0.41 | 0.4254 | 0.6231 | 0.5617 | n/a | 0.1998 | n/a |
