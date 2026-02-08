#!/usr/bin/env bash
# Run inference and evaluation for a model. Usage: ./run_inference.sh [MODEL_NAME]
# With no arg: runs random baseline first, then all trained models.
# Or: python scripts/run_inference.py --model MODEL_NAME

MODELS="random logreg riemann_svm knn lightgbm xgboost mlp 1dcnn tcn lstm transformer mamba"

# MODELS="baseline_logreg csp_lda logreg rf"

cd "$(dirname "$0")"

if [ -n "$1" ]; then
    python scripts/run_inference.py --model "$1"
    exit $?
fi

for m in $MODELS; do
    echo "=== Running inference for $m ==="
    python scripts/run_inference.py --model "$m" --n-inference 5000
done
