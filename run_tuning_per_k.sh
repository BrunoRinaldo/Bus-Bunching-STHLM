#!/bin/zsh
# Overnight per-horizon tuning: HGB + logistic regression (.venv), then Keras (.venv312).
# Keeps the Mac awake while running (caffeinate), logs to logs/. Safe to re-run:
# both scripts resume from their reports/*.json and skip finished work.
#
#   ./run_tuning_per_k.sh           full run (~7-8 h)
#   ./run_tuning_per_k.sh --smoke   quick end-to-end check (a few minutes)

cd "$(dirname "$0")"
mkdir -p logs
suffix=""; [[ "$1" == "--smoke" ]] && suffix="_smoke"
log="logs/tuning_per_k${suffix}.log"

{
  echo "=== start $(date) ==="
  echo "--- HGB + logistic regression ---"
  .venv/bin/python src/tuning_per_k.py "$@"; rc1=$?
  echo "--- Keras ---"
  .venv312/bin/python src/tuning_keras_per_k.py "$@"; rc2=$?
  echo "=== end $(date): sklearn exit $rc1, keras exit $rc2 ==="
} 2>&1 | caffeinate -i -s tee -a "$log"
