"""Phase 5 - logistic regression and gradient boosting, all 5 horizons.

Trained with sample_weight from class_weight.compute_sample_weight('balanced', y)
(class weights, never resampling, per the plan). Train/val/test come from the
'split' column assigned in features.py (temporal, from the date column).
Evaluated on val (for threshold selection) and test (final numbers, never
touched during training or thresholding).
"""
import json
import time
import duckdb
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    confusion_matrix, brier_score_loss,
)
from sklearn.utils.class_weight import compute_sample_weight

DS = "data/processed/model_dataset.parquet"
HORIZONS = [1, 2, 3, 5, 8]

NUMERIC = [
    "headway_ratio", "observed_arrival_delay", "observed_departure_delay",
    "dwell", "dwell_rel_median", "speed_mps",
    "leader_dwell", "leader_arrival_delay", "leader_delay_growth",
    "stop_sequence", "route_frac", "dist_traveled",
    "hour_of_day", "day_of_week", "month_of_year", "n_routes_serving",
    "headway_ratio_lag1", "headway_ratio_lag2", "headway_ratio_lag3",
    "closing_speed_3", "cum_delay_growth_3", "speed_lag1", "speed_lag2",
    "temperature_2m", "temperature_2m_sq", "precipitation", "precip_3h", "wind_speed_10m",
]
BOOLEAN = ["has_parent_station", "is_interchange", "under_active_alert",
           "is_precip", "is_snowing", "ice_risk"]
CATEGORICAL = ["LineNumber", "direction_id", "wx_group"]
ALL_COLS = NUMERIC + BOOLEAN + CATEGORICAL

con = duckdb.connect()
con.sql("PRAGMA threads=8")


def load(k, split):
    # temp_bin (6-category binning) was replaced by scaled temperature_2m plus
    # a temperature_2m_sq quadratic term: GBM never needed the bins (it splits
    # on the continuous value natively), and for logistic regression a squared
    # term is the standard way to let a linear model fit a U-shaped effect
    # (boarding slowing at both cold and hot extremes) without the arbitrary
    # bin edges. See doc 2 §3 for the reasoning and doc 3 for the before/after.
    base_cols = [c for c in ALL_COLS if c != "temperature_2m_sq"]
    cols_sql = ", ".join(base_cols + [f"y_k{k} AS y"])
    q = f"""
        SELECT {cols_sql} FROM '{DS}'
        WHERE split = '{split}' AND y_k{k} IS NOT NULL
    """
    df = con.sql(q).df()
    df["temperature_2m_sq"] = df["temperature_2m"] ** 2
    for b in BOOLEAN:
        df[b] = df[b].astype("float64")
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)
    X = df[ALL_COLS]
    y = df["y"].astype(int).values
    return X, y


def make_logreg_pipeline():
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer([
        ("num", num_pipe, NUMERIC + BOOLEAN),
        ("cat", cat_pipe, CATEGORICAL),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=200, solver="lbfgs")),
    ])


def make_hgb():
    return HistGradientBoostingClassifier(
        categorical_features=[c in CATEGORICAL for c in ALL_COLS],
        max_iter=200, random_state=0,
    )


def prep_hgb_frame(X):
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def evaluate(model, X, y, is_hgb):
    Xp = prep_hgb_frame(X) if is_hgb else X
    proba = model.predict_proba(Xp)[:, 1]
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    precisions, recalls, thresholds = precision_recall_curve(y, proba)
    # recall at precision >= 0.80 (operational framing)
    mask = precisions[:-1] >= 0.80
    recall_at_p80 = recalls[:-1][mask].max() if mask.any() else float("nan")
    f1 = f1_score(y, (proba >= 0.5).astype(int))
    cm = confusion_matrix(y, (proba >= 0.5).astype(int)).tolist()
    return {
        "n": len(y), "base_rate_pct": round(100 * y.mean(), 3),
        "pr_auc": round(pr_auc, 4), "brier": round(brier, 5),
        "f1_at_0.5": round(f1, 4), "recall_at_precision_0.80": round(float(recall_at_p80), 4),
        "confusion_matrix_at_0.5": cm,
    }


def main():
    results = {}
    for k in HORIZONS:
        t0 = time.time()
        Xtr, ytr = load(k, "train")
        Xval, yval = load(k, "val")
        Xtest, ytest = load(k, "test")
        print(f"k={k} train={len(ytr)} val={len(yval)} test={len(ytest)} loaded in {time.time()-t0:.1f}s")

        sw = compute_sample_weight("balanced", ytr)

        t0 = time.time()
        logreg = make_logreg_pipeline()
        logreg.fit(Xtr, ytr, clf__sample_weight=sw)
        print(f"k={k} logreg fit in {time.time()-t0:.1f}s")
        joblib.dump(logreg, f"models/logreg_k{k}.joblib")

        t0 = time.time()
        hgb = make_hgb()
        hgb.fit(prep_hgb_frame(Xtr), ytr, sample_weight=sw)
        print(f"k={k} hgb fit in {time.time()-t0:.1f}s")
        joblib.dump(hgb, f"models/hgb_k{k}.joblib")

        results[k] = {
            "logreg_val": evaluate(logreg, Xval, yval, False),
            "logreg_test": evaluate(logreg, Xtest, ytest, False),
            "hgb_val": evaluate(hgb, Xval, yval, True),
            "hgb_test": evaluate(hgb, Xtest, ytest, True),
        }
        print(f"k={k} logreg test PR-AUC={results[k]['logreg_test']['pr_auc']} "
              f"hgb test PR-AUC={results[k]['hgb_test']['pr_auc']}")

        # feature importance / coefficients for the mechanism story (RQ2)
        cat_names = list(logreg.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
        num_names = NUMERIC + BOOLEAN
        ind = logreg.named_steps["pre"].named_transformers_["num"].named_steps["impute"].indicator_
        if ind is not None:
            num_names = num_names + [f"{NUMERIC[i] if i < len(NUMERIC) else BOOLEAN[i-len(NUMERIC)]}_was_missing" for i in ind.features_]
        coef = logreg.named_steps["clf"].coef_[0]
        coef_names = num_names + cat_names
        coef_df = pd.DataFrame({"feature": coef_names[:len(coef)], "coef": coef})
        coef_df.to_csv(f"models/logreg_coefs_k{k}.csv", index=False)

        importances = pd.Series(hgb.feature_importances_ if hasattr(hgb, "feature_importances_") else [], dtype=float)

    with open("models/phase5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("done")


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    main()
