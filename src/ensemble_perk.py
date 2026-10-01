"""Step 1 for the long horizons: weighted average of HGB and Keras scores (k=5, 8).

  .venv312/bin/python src/ensemble_perk.py --keras   Keras val/test scores -> data/processed/
  .venv/bin/python    src/ensemble_perk.py           HGB scores, weight choice, test, bootstrap

Row-level scores are written only to data/processed/ (gitignored, never
shared). The two datasets are joined on (date, trip_id, stop_sequence). The
weight w (score = w * HGB + (1 - w) * Keras) is chosen on the validation
split (Oct); the test split is scored once for that weight and compared with
HGB alone by the same paired day-block bootstrap as src/bootstrap_perk.py.
Aggregates go to reports/ensemble_perk.json.
"""
import json
import sys

import duckdb
import numpy as np

HORIZONS = [5, 8]
WEIGHTS = np.round(np.linspace(0, 1, 21), 2)
OUT = "reports/ensemble_perk.json"
con = duckdb.connect()


def pred_path(model, k, split):
    return f"data/processed/preds_{model}_k{k}_{split}.parquet"


def keys(path, k, split):
    return con.sql(f"SELECT date, trip_id, stop_sequence, y_k{k} AS y FROM '{path}' "
                   f"WHERE split='{split}' AND y_k{k} IS NOT NULL").df()


def write(df, path):
    con.register("df_out", df)
    con.sql(f"COPY df_out TO '{path}' (FORMAT PARQUET)")
    con.unregister("df_out")


def keras_scores():
    import keras
    import sequence_model as sm
    import tuning_keras as tk

    for k in HORIZONS:
        _, (Xva, yva), (Xte, yte) = tk.load_k(k)
        model = keras.saving.load_model(f"models/perk_sequence_k{k}.keras")
        for split, X, y in [("val", Xva, yva), ("test", Xte, yte)]:
            d = keys(sm.SEQ, k, split)
            assert (d["y"].astype(int).values == y).all(), "row order mismatch"
            d["p"] = model.predict(X, batch_size=8192, verbose=0).ravel().astype("float64")
            write(d, pred_path("keras", k, split))
        keras.backend.clear_session()
        print(f"k={k}: keras scores written", flush=True)


def hgb_scores(k):
    import joblib
    import models as m
    import tuning as t

    clf = joblib.load(f"models/perk_hgb_k{k}.joblib")
    for split in ["val", "test"]:
        X, y = m.load(k, split)
        d = keys(m.DS, k, split)
        assert (d["y"].astype(int).values == y).all(), "row order mismatch"
        d["p"] = t.hgb_proba(clf, X)
        write(d, pred_path("hgb", k, split))


def joined(k, split):
    return con.sql(f"""
        SELECT h.date, h.y, h.p AS p_hgb, s.p AS p_keras
        FROM '{pred_path("hgb", k, split)}' h
        LEFT JOIN '{pred_path("keras", k, split)}' s USING (date, trip_id, stop_sequence)
    """).df()


def main():
    if "--keras" in sys.argv:
        keras_scores()
        return
    from sklearn.metrics import average_precision_score
    from bootstrap_perk import paired_bootstrap

    res = {}
    for k in HORIZONS:
        hgb_scores(k)
        va, te = joined(k, "val"), joined(k, "test")
        cover = {s: round(float(d["p_keras"].notna().mean()), 4) for s, d in [("val", va), ("test", te)]}
        # rows without a Keras score (sequence table has no row for them) keep the HGB score
        for d in (va, te):
            d["p_keras"] = d["p_keras"].fillna(d["p_hgb"])
        yv = va["y"].to_numpy(int)
        grid = {float(w): round(float(average_precision_score(yv, w * va["p_hgb"] + (1 - w) * va["p_keras"])), 4)
                for w in WEIGHTS}
        w = max(grid, key=grid.get)
        yt = te["y"].to_numpy(int)
        p_ens = (w * te["p_hgb"] + (1 - w) * te["p_keras"]).to_numpy()
        bs = paired_bootstrap(yt, p_ens, te["p_hgb"].to_numpy(), te["date"].to_numpy())
        res[str(k)] = {"keras_coverage": cover, "val_pr_auc_by_weight": grid, "w_hgb": w,
                       "val_pr_auc_hgb": grid[1.0], "val_pr_auc_ens": grid[w], "test_vs_hgb": bs}
        print(f"k={k}: w_hgb={w} val {grid[1.0]} -> {grid[w]}; test {bs['pr_auc_old']} -> {bs['pr_auc_new']} "
              f"diff {bs['diff']} CI {bs['ci95']}; R@P80 {bs['r80_old']} -> {bs['r80_new']} CI {bs['r80_ci95']}",
              flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
