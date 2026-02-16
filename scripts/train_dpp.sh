#!/bin/bash

set -euo pipefail

# Usage:
#   ./scripts/train_dpp.sh <config_path> [num_gpus]
# Examples:
#   ./scripts/train_dpp.sh configs/default.yml
#   ./scripts/train_dpp.sh configs/default.yml 2

CONFIG_PATH="${1:-}"
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Usage: $0 <config_path> [num_gpus]"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Query GPU count visible to THIS Python/Torch runtime.
TORCH_VISIBLE_GPUS=$(python - <<'PY'
import sys
try:
        import torch
        print(torch.cuda.device_count())
except Exception as e:
        print(f"ERROR:{e}")
        sys.exit(0)
PY
)

if [[ "$TORCH_VISIBLE_GPUS" == ERROR:* ]]; then
    echo "ERROR: Failed to query torch-visible GPUs."
    echo "  $TORCH_VISIBLE_GPUS"
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    exit 1
fi

# Priority for requested GPU count:
# 1) explicit CLI arg
# 2) NUM_GPUS env var
# 3) all torch-visible GPUs
if [[ -n "${2:-}" ]]; then
    NUM_GPUS="$2"
elif [[ -n "${NUM_GPUS:-}" ]]; then
    NUM_GPUS="$NUM_GPUS"
else
    NUM_GPUS="$TORCH_VISIBLE_GPUS"
fi

if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "ERROR: NUM_GPUS must be >= 1"
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Torch-visible GPUs=$TORCH_VISIBLE_GPUS"

if [[ "$TORCH_VISIBLE_GPUS" -lt "$NUM_GPUS" ]]; then
    echo "ERROR: Requested NUM_GPUS=$NUM_GPUS, but torch sees only $TORCH_VISIBLE_GPUS GPU(s)."
    echo "Fix: reduce GPU count in command or request/expose more GPUs in scheduler allocation."
    exit 1
fi

echo "Launching distributed training with NUM_GPUS=$NUM_GPUS"

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    scripts/train.py \
    --config "$CONFIG_PATH"