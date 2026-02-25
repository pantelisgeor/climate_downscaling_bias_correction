#!/bin/bash
#SBATCH --job-name=dec-dwscl-train
#SBATCH --output=/nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code/logs/%x_%j.out
#SBATCH --error=/nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code/logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -A p185
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH -p a100
#SBATCH --chdir=/nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code

mkdir -p /nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code/logs

source /nvme/h/pgeorgiades/miniconda3/etc/profile.d/conda.sh
conda activate /nvme/h/pgeorgiades/data_p185/AI_downscale/conda_env

set -euo pipefail

# Usage:
#   sbatch scripts_dec_downscale/train_dec_downscale_dpp.sh <config_path> [num_gpus]
# Example:
#   sbatch scripts_dec_downscale/train_dec_downscale_dpp.sh \
#          configs_dec_downscale/exp_d01_cnn_tweedie.yml

CONFIG_PATH="${1:-}"
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Usage: $0 <config_path> [num_gpus]"
    exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config not found: $CONFIG_PATH"
    exit 1
fi

# Parse experiment metadata + test years from the config
readarray -t CONFIG_META < <(python - "$CONFIG_PATH" <<'PY'
import sys, yaml
from pathlib import Path
cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
base_dir   = cfg["experiment"]["base_dir"]
name       = cfg["experiment"]["name"]
output_dir = cfg.get("experiment", {}).get("output_dir", "")
test_years = cfg.get("data", {}).get("test_years", [])
years_csv  = ",".join(str(y) for y in test_years)
print(base_dir)
print(name)
print(str(Path(output_dir)) if output_dir else "")
print(years_csv)
PY
)

EXPERIMENT_BASE_DIR="${CONFIG_META[0]}"
EXPERIMENT_NAME="${CONFIG_META[1]}"
CONFIG_EXPERIMENT_DIR="${CONFIG_META[2]}"
TEST_YEARS_CSV="${CONFIG_META[3]}"

if [[ -z "$TEST_YEARS_CSV" ]]; then
    echo "ERROR: Could not read data.test_years from config."
    exit 1
fi

# Resolve GPU count
TORCH_VISIBLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ -n "${2:-}" ]]; then
    NUM_GPUS="$2"
elif [[ -n "${NUM_GPUS:-}" ]]; then
    NUM_GPUS="$NUM_GPUS"
else
    NUM_GPUS="$TORCH_VISIBLE_GPUS"
fi

if [[ "$TORCH_VISIBLE_GPUS" -lt "$NUM_GPUS" ]]; then
    echo "ERROR: Requested $NUM_GPUS GPU(s) but torch sees only $TORCH_VISIBLE_GPUS."
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Launching DDP training on $NUM_GPUS GPU(s)"

MASTER_PORT=$(( 29500 + (${SLURM_JOB_ID:-0} % 1000) ))

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port="$MASTER_PORT" \
    scripts_dec_downscale/train_dec_downscale.py \
    --config "$CONFIG_PATH"

# Resolve experiment directory
if [[ -n "$CONFIG_EXPERIMENT_DIR" ]]; then
    LATEST_EXPERIMENT_DIR="$CONFIG_EXPERIMENT_DIR"
else
    LATEST_EXPERIMENT_DIR=$(ls -dt "$EXPERIMENT_BASE_DIR"/"${EXPERIMENT_NAME}"_* 2>/dev/null | head -n 1 || true)
fi

if [[ -z "$LATEST_EXPERIMENT_DIR" || ! -d "$LATEST_EXPERIMENT_DIR" ]]; then
    echo "ERROR: Could not resolve experiment directory."
    exit 1
fi

INFER_DIR="$LATEST_EXPERIMENT_DIR/inferred"
ANALYSIS_DIR="$LATEST_EXPERIMENT_DIR/analysis"

echo "Running inference → $INFER_DIR"
python scripts_dec_downscale/infer_dec_downscale.py \
    --experiment-dir "$LATEST_EXPERIMENT_DIR" \
    --checkpoint best \
    --years "$TEST_YEARS_CSV" \
    --output-dir "$INFER_DIR"

echo "Running analysis → $ANALYSIS_DIR"
python scripts_dec_downscale/analyze_dec_downscale_outputs.py \
    --input-dir "$INFER_DIR" \
    --output-dir "$ANALYSIS_DIR"

echo "Done — results in $LATEST_EXPERIMENT_DIR"
