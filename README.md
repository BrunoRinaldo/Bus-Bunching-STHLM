# Bus bunching prediction - SL GTFS-RT, 2024

Predicts bus bunching k = 1, 2, 3, 5 and 8 stops ahead on 8 high-frequency
Stockholm (SL) bus lines from 2024 GTFS-Realtime data, with gradient boosting
(HGB), logistic regression and a Keras sequence model, tuned per horizon and
compared with a persistence baseline.

See `bunching-implementation-plan.md` for the project plan and `docs/` for the
methodology (data, headway and label construction, features, split, model
choice, tuning).

## Layout

| Folder | Content | In git |
|---|---|---|
| `src/` | all code | yes |
| `data/raw/` | input files you provide (below) | no |
| `data/processed/` | derived tables (parquet) | no |
| `data/results/` | metrics, search results and tables (JSON / Markdown) | no |
| `models/` | trained models | yes |
| `figures/` | report figures | yes |
| `docs/` | methodology | yes |
| `logs/` | run logs of the long searches | no |

## Input data

Place in `data/raw/`:

| File | Used by |
|---|---|
| `gtfs_rt_202401.csv.gz` ... `gtfs_rt_202412.csv.gz` | `ingest.py` |
| `stops.csv` | `features.py`, `weather.py`, `eda_figures.py` |
| `transfers.csv` | `features.py` |
| `service_alerts_2024.parquet` | `features.py` |

`routes.csv` and `trips.csv` (static GTFS) are reference only. `weather.py`
needs internet access once (Open-Meteo). The raw input is several GB and the
processed tables tens of GB.

## Setup

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3.12 -m venv .venv312 && .venv312/bin/pip install -r requirements-keras.txt
```

TensorFlow has no wheel for Python 3.14, so every Keras step runs in
`.venv312`. Run everything from the repository root.

## Pipeline

```
# 1. Data
.venv/bin/python src/ingest.py                 # raw csv.gz -> data/processed/bus_subset.parquet
.venv/bin/python src/headways.py               # -> headways.parquet (dwell, headway, labels)
.venv/bin/python src/weather.py                # -> weather.parquet (network, run once)
.venv/bin/python src/features.py               # -> model_dataset.parquet (features, targets, split)
.venv/bin/python src/sequence_data.py          # -> sequence_dataset.parquet (Keras input)
.venv/bin/python src/extra_features.py         # -> extra_features.parquet (candidates for k=5, 8)
.venv/bin/python src/eda_figures.py            # figs 1-7, 10, 19

# 2. Untuned models, all horizons
.venv/bin/python src/models.py                 # models/logreg_k*, hgb_k*
.venv312/bin/python src/sequence_model.py      # models/sequence_k*
.venv/bin/python src/compare_models.py         # fig 8

# 3. Tuning at k=3, reused at the other horizons
.venv/bin/python src/tuning.py stage1          # ~30 min
.venv/bin/python src/tuning.py stage2          # ~25 min
.venv312/bin/python src/tuning_keras.py        # ~1 h
.venv/bin/python src/compare_tuned.py          # fig 9

# 4. Separate search per horizon (~7-8 h, resumable; --smoke checks it in ~2 min)
./run_tuning_per_k.sh                          # models/perk_*

# 5. Long horizons k=5, 8: feature screen + wider HGB search
.venv/bin/python src/feature_screen.py
.venv/bin/python src/tuning_long_k.py          # models/long_hgb_k5, k8 (resumable, --smoke)

# 6. Evaluation and figures
.venv/bin/python src/perk_curves.py
.venv312/bin/python src/perk_curves.py --keras
.venv/bin/python src/bootstrap_perk.py
.venv312/bin/python src/bootstrap_perk.py --keras
.venv/bin/python src/compare_perk.py           # figs 11-17
.venv/bin/python src/false_alarms.py           # fig 18
.venv/bin/python src/diagnostics_tuned.py      # by line / month / weather, ablations, permutation importance
.venv/bin/python src/permutation_importance.py # fig 21 + permutation_importance.md
.venv/bin/python src/persistence_comparison.py
.venv312/bin/python src/persistence_comparison_keras.py
.venv/bin/python src/persistence_comparison.py --plot   # fig 22
```

Best model per horizon: `perk_hgb_k1`, `perk_hgb_k2`, `tuned_hgb_k3`,
`long_hgb_k5`, `long_hgb_k8` (see `best_model()` in `src/false_alarms.py`).
Figure scripts with saved results can be redrawn on their own:
`false_alarms.py --plot`, `persistence_comparison.py --plot`,
`compare_tuned.py`, `compare_perk.py`, `permutation_importance.py`.
