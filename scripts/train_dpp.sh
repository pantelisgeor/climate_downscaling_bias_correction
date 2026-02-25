#!/bin/bash
#SBATCH --job-name=climate-train
#SBATCH --output=/nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code/logs/%x_%j.out
#SBATCH --error=/nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -A climate
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:4
#SBATCH --mem=185G
#SBATCH -p gpu
#SBATCH --chdir=/nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code

# Ensure log directory exists before SLURM tries to open the output files
mkdir -p /nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code/logs

# Activate the project conda environment.
# source the conda init script first so 'conda activate' works in batch shells.
source /nvme/h/pgeorgiades/miniconda3/etc/profile.d/conda.sh
conda activate /nvme/h/pgeorgiades/data_p185/AI_downscale/conda_env

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

# Read experiment metadata and test years from config.
readarray -t CONFIG_META < <(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml
from pathlib import Path

cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

base_dir = cfg["experiment"]["base_dir"]
name = cfg["experiment"]["name"]
output_dir = cfg.get("experiment", {}).get("output_dir", "")
test_years = cfg.get("data", {}).get("test_years", [])
years_csv = ",".join(str(y) for y in test_years)

if output_dir:
    resolved_exp_dir = str(Path(output_dir))
else:
    resolved_exp_dir = ""

print(base_dir)
print(name)
print(resolved_exp_dir)
print(years_csv)
PY
)

EXPERIMENT_BASE_DIR="${CONFIG_META[0]}"
EXPERIMENT_NAME="${CONFIG_META[1]}"
CONFIG_EXPERIMENT_DIR="${CONFIG_META[2]}"
TEST_YEARS_CSV="${CONFIG_META[3]}"

if [[ -z "$TEST_YEARS_CSV" ]]; then
    echo "ERROR: Could not resolve data.test_years from config: $CONFIG_PATH"
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

# Derive a unique master port from the SLURM job ID to avoid port collisions
# when multiple jobs run concurrently on the same node.
MASTER_PORT=$(( 29500 + (${SLURM_JOB_ID:-0} % 1000) ))

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port="$MASTER_PORT" \
    scripts/train.py \
    --config "$CONFIG_PATH"

# Resolve experiment directory for this run.
if [[ -n "$CONFIG_EXPERIMENT_DIR" ]]; then
    LATEST_EXPERIMENT_DIR="$CONFIG_EXPERIMENT_DIR"
else
    LATEST_EXPERIMENT_DIR=$(ls -dt "$EXPERIMENT_BASE_DIR"/"${EXPERIMENT_NAME}"_* 2>/dev/null | head -n 1 || true)
fi

if [[ -z "$LATEST_EXPERIMENT_DIR" || ! -d "$LATEST_EXPERIMENT_DIR" ]]; then
    echo "ERROR: Could not resolve experiment directory."
    if [[ -n "$CONFIG_EXPERIMENT_DIR" ]]; then
        echo "  Configured experiment.output_dir: $CONFIG_EXPERIMENT_DIR"
    else
        echo "  Searched under: $EXPERIMENT_BASE_DIR for prefix ${EXPERIMENT_NAME}_*"
    fi
    exit 1
fi

INFER_DIR="$LATEST_EXPERIMENT_DIR/inferred"
ANALYSIS_DIR="$LATEST_EXPERIMENT_DIR/analysis"

echo "Running inference into: $INFER_DIR"
python scripts/infer.py \
    --experiment-dir "$LATEST_EXPERIMENT_DIR" \
    --checkpoint best \
    --years "$TEST_YEARS_CSV" \
    --output-dir "$INFER_DIR"

echo "Running analysis into: $ANALYSIS_DIR"
python scripts/analyze_inference_outputs.py \
    --input-dir "$INFER_DIR" \
    --output-dir "$ANALYSIS_DIR"