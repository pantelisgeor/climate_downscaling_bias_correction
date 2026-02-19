#!/bin/bash
#SBATCH --job-name=climate-train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -A aiml
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH -p a100

set -euo pipefail

# Usage:
#   sbatch scripts/resume_train.sbatch <experiment_dir> [config_path]
#
# Examples:
#   sbatch scripts/resume_train.sbatch /path/to/experiments/climate_net_vit_20260215_120000
#   sbatch scripts/resume_train.sbatch /path/to/new_experiment_dir configs/default_ViT.yml
#
# Behavior:
# - If checkpoints exist in <experiment_dir>/checkpoints, resume from latest epoch checkpoint.
# - Else if best_model.pt exists, resume from it.
# - Else start a fresh run using provided config_path (or <experiment_dir>/config.yaml if present).

PROJECT_DIR="/nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code"
EXPERIMENT_DIR="${1:-}"
CONFIG_PATH="${2:-}"

if [[ -z "$EXPERIMENT_DIR" ]]; then
  echo "ERROR: Missing experiment_dir argument."
  echo "Usage: sbatch scripts/resume_train.sbatch <experiment_dir> [config_path]"
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# Optional: activate your environment here if your cluster requires it.
# source /path/to/conda.sh
# conda activate <your_env>

# Derive config path if not explicitly provided.
if [[ -z "$CONFIG_PATH" ]]; then
  if [[ -f "$EXPERIMENT_DIR/config.yaml" ]]; then
    CONFIG_PATH="$EXPERIMENT_DIR/config.yaml"
  fi
fi

if [[ -z "$CONFIG_PATH" ]]; then
  echo "ERROR: No config found. Provide [config_path] or ensure $EXPERIMENT_DIR/config.yaml exists."
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found: $CONFIG_PATH"
  exit 1
fi

# Read test years from config for inference.
TEST_YEARS_CSV=$(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

test_years = cfg.get("data", {}).get("test_years", [])
print(",".join(str(y) for y in test_years))
PY
)

if [[ -z "$TEST_YEARS_CSV" ]]; then
  echo "ERROR: Could not resolve data.test_years from config: $CONFIG_PATH"
  exit 1
fi

CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"
RESUME_PATH=""

if [[ -d "$CHECKPOINT_DIR" ]]; then
  shopt -s nullglob
  CKPTS=("$CHECKPOINT_DIR"/checkpoint_epoch_*.pt)
  shopt -u nullglob

  if (( ${#CKPTS[@]} > 0 )); then
    RESUME_PATH=$(printf "%s\n" "${CKPTS[@]}" | sort -V | tail -n 1)
  elif [[ -f "$CHECKPOINT_DIR/best_model.pt" ]]; then
    RESUME_PATH="$CHECKPOINT_DIR/best_model.pt"
  fi
fi

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
MASTER_PORT="${MASTER_PORT:-29500}"

CMD=(
  torchrun
  --nproc_per_node="$NUM_GPUS"
  --nnodes=1
  --node_rank=0
  --master_addr=localhost
  --master_port="$MASTER_PORT"
  scripts/train.py
  --config "$CONFIG_PATH"
)

if [[ -n "$RESUME_PATH" ]]; then
  echo "Resuming from checkpoint: $RESUME_PATH"
  CMD+=(--resume "$RESUME_PATH")
else
  echo "No checkpoint found. Starting fresh training run with config: $CONFIG_PATH"
fi

echo "Running command: ${CMD[*]}"
"${CMD[@]}"

INFER_DIR="$EXPERIMENT_DIR/inferred"
ANALYSIS_DIR="$EXPERIMENT_DIR/analysis"

echo "Running inference into: $INFER_DIR"
python scripts/infer.py \
  --experiment-dir "$EXPERIMENT_DIR" \
  --checkpoint best \
  --years "$TEST_YEARS_CSV" \
  --output-dir "$INFER_DIR"

echo "Running analysis into: $ANALYSIS_DIR"
python scripts/analyze_inference_outputs.py \
  --input-dir "$INFER_DIR" \
  --output-dir "$ANALYSIS_DIR"
