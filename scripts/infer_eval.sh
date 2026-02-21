#!/bin/bash

EXPERIMENT_DIR=$1
INFER_DIR="${EXPERIMENT_DIR}/inferred"
ANALYSIS_DIR="${EXPERIMENT_DIR}/analysis"

echo "Running inference into: $EXPERIMENT_DIR"
python scripts/infer.py \
    --experiment-dir "$EXPERIMENT_DIR" \
    --checkpoint best \
    --years 2017-2020 \
    --output-dir "$INFER_DIR"

echo "Running analysis into: $ANALYSIS_DIR"
python scripts/analyze_inference_outputs.py \
    --input-dir "$INFER_DIR" \
    --output-dir "$ANALYSIS_DIR"