#!/bin/bash
#SBATCH --job-name=seasonal-train
#SBATCH --output=/nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code/logs/%x_%j.out
#SBATCH --error=/nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -A p236
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -p a100
#SBATCH --chdir=/nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code

# Single-GPU version of train_seasonal_dpp.sh.
#
# Why single GPU for the seasonal pipeline?
#   The SeasonalDataLoader loads the ~29 GB NetCDF into numpy arrays in-process
#   (rank 0).  With 2 GPUs and rank0_only_in_memory_load=true, rank 1 reads
#   every batch from disk via xarray — much slower than rank 0's RAM access.
#   DDP synchronises after every batch, so rank 0 idles waiting for rank 1 on
#   every step.  A single GPU eliminates this mismatch entirely:
#     - One rank, one load, dataset lives in RAM (~50 GB as float32).
#     - 12 DataLoader workers read from numpy → no disk overhead.
#     - 72 GB available system RAM comfortably fits the dataset.
#
# Ensure log directory exists before SLURM tries to open the output files
mkdir -p /nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/code/logs

# Activate the project conda environment
source /nvme/h/pgeorgiades/miniconda3/etc/profile.d/conda.sh
conda activate /nvme/h/pgeorgiades/data_p185/AI_downscale/conda_env

set -euo pipefail

# Usage:
#   sbatch scripts_seasonal/train_seasonal_single.sh <config_path>
# Interactive:
#   bash scripts_seasonal/train_seasonal_single.sh configs/exp_s01_cnn_tweedie_seasonal.yml

CONFIG_PATH="${1:-}"
if [[ -z "$CONFIG_PATH" ]]; then
    echo "Usage: $0 <config_path>"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Read experiment metadata and test years from config
readarray -t CONFIG_META < <(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml
from pathlib import Path

cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

base_dir    = cfg["experiment"]["base_dir"]
name        = cfg["experiment"]["name"]
output_dir  = cfg.get("experiment", {}).get("output_dir", "")
test_years  = cfg.get("data", {}).get("test_years", [])
years_csv   = ",".join(str(y) for y in test_years)

resolved = str(Path(output_dir)) if output_dir else ""

print(base_dir)
print(name)
print(resolved)
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

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Launching single-GPU seasonal training"

# Use a fixed master port (no DDP, but torchrun still initialises NCCL with
# world_size=1 — that is fine and avoids any code-path branching in the trainer)
MASTER_PORT=$(( 29500 + (${SLURM_JOB_ID:-0} % 1000) ))

torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port="$MASTER_PORT" \
    scripts_seasonal/train_seasonal.py \
    --config "$CONFIG_PATH"

# Resolve experiment directory
if [[ -n "$CONFIG_EXPERIMENT_DIR" ]]; then
    LATEST_EXPERIMENT_DIR="$CONFIG_EXPERIMENT_DIR"
else
    LATEST_EXPERIMENT_DIR=$(ls -dt "$EXPERIMENT_BASE_DIR"/"${EXPERIMENT_NAME}"_* \
        2>/dev/null | head -n 1 || true)
fi

if [[ -z "$LATEST_EXPERIMENT_DIR" || ! -d "$LATEST_EXPERIMENT_DIR" ]]; then
    echo "ERROR: Could not resolve experiment directory."
    exit 1
fi

INFER_DIR="$LATEST_EXPERIMENT_DIR/inferred"
ANALYSIS_DIR="$LATEST_EXPERIMENT_DIR/analysis"

echo "Running inference → $INFER_DIR"
python scripts_seasonal/infer_seasonal.py \
    --experiment-dir "$LATEST_EXPERIMENT_DIR" \
    --checkpoint best \
    --years "$TEST_YEARS_CSV" \
    --output-dir "$INFER_DIR"

echo "Running analysis → $ANALYSIS_DIR"
python scripts_seasonal/analyze_seasonal_outputs.py \
    --input-dir "$INFER_DIR" \
    --output-dir "$ANALYSIS_DIR"

echo "Done.  Results in $LATEST_EXPERIMENT_DIR"
