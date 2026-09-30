# Per-horizon tuned models, test results (Nov-Dec 2024)

Each horizon has its own search (src/tuning_per_k.py, src/tuning_keras_per_k.py): 800k-row training subsample, selection on the validation split (Oct), winner refit on the full training split. 'k=3-tuned' is the fig 9 setup (searched at k=3, reused elsewhere). R@P80 = recall at precision >= 0.80; '-' = precision 0.80 never reached.

| k | min | HGB | HGB k=3-tuned | HGB R@P80 | Keras | Keras k=3-tuned | Keras R@P80 | LogReg | LogReg k=3-tuned | LogReg R@P80 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.05 | **0.9582** | 0.9578 | 0.963 | 0.9218 | 0.9315 | 0.926 | 0.8457 | 0.8593 | 0.896 |
| 2 | 2.1 | **0.9047** | 0.9047 | 0.879 | 0.8697 | 0.8616 | 0.847 | 0.7735 | 0.7849 | 0.777 |
| 3 | 3.15 | **0.853** | 0.8558 | 0.786 | 0.812 | 0.8151 | 0.736 | 0.7211 | 0.7211 | 0.626 |
| 5 | 5.25 | **0.7673** | 0.7617 | 0.602 | 0.677 | 0.7093 | 0.350 | 0.6076 | 0.6124 | - |
| 8 | 8.4 | **0.6586** | 0.6515 | 0.359 | 0.5794 | 0.579 | 0.138 | 0.4889 | 0.488 | - |

Selected configurations:

| k | HGB | Keras | LogReg |
|---|---|---|---|
| 1 | nb2 | rs11 (cnn) | n_pca=None, C=0.01 |
| 2 | k3_winner | rs3 (cnn) | n_pca=None, C=0.01 |
| 3 | rs4 | k3_winner (cnn) | n_pca=45, C=0.1 |
| 5 | nb3 | rs4 (cnn) | n_pca=None, C=0.01 |
| 8 | nb2 | k3_winner (cnn) | n_pca=None, C=1.0 |

Paired day-block bootstrap, per-k tuned minus k=3-tuned (test split, 61 days, B = 1000, src/bootstrap_perk.py). CI = 2.5-97.5 percentile.

| model | k | PR-AUC diff | 95% CI | R@P80 diff | 95% CI |
|---|---|---|---|---|---|
| gradient boosting | 1 | +0.0004 | [-0.0000, +0.0009] | +0.0008 | [-0.0002, +0.0017] |
| gradient boosting | 2 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | [+0.0000, +0.0000] |
| gradient boosting | 3 | -0.0028 | [-0.0042, -0.0014] | -0.0029 | [-0.0067, +0.0009] |
| gradient boosting | 5 | +0.0056 | [+0.0032, +0.0079] | +0.0157 | [+0.0029, +0.0232] |
| gradient boosting | 8 | +0.0071 | [+0.0039, +0.0102] | +0.0164 | [+0.0031, +0.0318] |
| Keras sequence model | 1 | -0.0097 | [-0.0131, -0.0067] | -0.0059 | [-0.0085, -0.0035] |
| Keras sequence model | 2 | +0.0081 | [+0.0050, +0.0111] | +0.0052 | [+0.0012, +0.0089] |
| Keras sequence model | 3 | -0.0031 | [-0.0050, -0.0011] | -0.0041 | [-0.0099, +0.0012] |
| Keras sequence model | 5 | -0.0322 | [-0.0377, -0.0269] | -0.1120 | [-0.4266, -0.0635] |
| Keras sequence model | 8 | +0.0004 | [-0.0031, +0.0037] | +0.0458 | [-0.0343, +0.0799] |
| logistic regression | 1 | -0.0136 | [-0.0178, -0.0103] | +0.0154 | [+0.0119, +0.0197] |
| logistic regression | 2 | -0.0114 | [-0.0168, -0.0068] | +0.0108 | [+0.0038, +0.0163] |
| logistic regression | 3 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | [+0.0000, +0.0000] |
| logistic regression | 5 | -0.0048 | [-0.0106, +0.0003] | +0.0000 | [+0.0000, +0.0000] |
| logistic regression | 8 | +0.0009 | [-0.0042, +0.0060] | +0.0000 | [+0.0000, +0.0000] |
