# Tuned vs. original, test PR-AUC (Nov-Dec 2024)

Logistic regression 'tuned' = |headway_ratio|-style features + PCA (45 components at k=3, same config reused at other horizons) - see reports/tuning_results.md.

| k | min | HGB orig | HGB tuned | Keras orig | Keras tuned | LogReg orig | LogReg tuned |
|---|---|---|---|---|---|---|---|
| 1 | 1.05 | 0.8984 | **0.9578** | 0.917 | 0.9315 | 0.5855 | 0.8593 |
| 2 | 2.1 | 0.8603 | **0.9047** | 0.8546 | 0.8616 | 0.5625 | 0.7849 |
| 3 | 3.15 | 0.8136 | **0.8558** | 0.786 | 0.8151 | 0.5387 | 0.7211 |
| 5 | 5.25 | 0.6959 | **0.7617** | 0.6815 | 0.7093 | 0.4926 | 0.6124 |
| 8 | 8.4 | 0.6231 | **0.6515** | 0.5617 | 0.579 | 0.4254 | 0.488 |
