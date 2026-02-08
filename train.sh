#!/usr/bin/env bash
# Train models. Run from Relay/: bash train.sh [MODEL_NAME]
# Models: logreg riemann_svm knn lightgbm xgboost mlp 1dcnn tcn lstm transformer mamba

cd "$(dirname "$0")"
# MODELS="logreg riemann_svm knn lightgbm xgboost mlp 1dcnn tcn lstm transformer mamba"
MODELS="logreg riemann_svm knn lightgbm xgboost mlp 1dcnn mamba"


if [ -n "$1" ]; then
  python scripts/train.py --model "$1"
  exit $?
fi

for m in $MODELS; do
  echo "=== Training $m ==="
  python scripts/train.py --model "$m" || echo "FAILED: $m"
done
