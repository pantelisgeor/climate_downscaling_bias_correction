"""
Inference script for ClimateNet.

Loads a trained model checkpoint (best/last/custom), runs inference on user-defined
years, and saves one NetCDF file per valid (run, lead) combination.

Each output NetCDF contains:
- Input climate variables (static + dynamic)
- Predicted target variables
- True target variables

This enables direct side-by-side comparison for each run/lead trajectory.
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from torch.utils.data import DataLoader

# Add project root to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DecadalDataLoader
from src.models.climate_net import ClimateNet
from src.training.climate_dataset import ClimateDataset


INPUT_VAR_NAMES = ["pr", "tas", "tasmax", "hurs"]
INPUT_CHANNEL_INDEX = {
    "pr": 0,
    "tas": 1,
    "tasmax": 2,
    "hurs": 3,
}

TARGET_VAR_NAMES = ["tasERA", "tasmaxERA", "tpERA", "rhERA"]


def parse_years(year_text: str) -> List[int]:
    """Parse year string like '2017,2018,2020-2022' into sorted unique years."""
    years = set()
    for token in year_text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            start_year = int(start)
            end_year = int(end)
            lo, hi = min(start_year, end_year), max(start_year, end_year)
            for year in range(lo, hi + 1):
                years.add(year)
        else:
            years.add(int(token))

    if not years:
        raise ValueError("No valid years were parsed from --years")

    return sorted(years)


def get_year_from_time_index(data_loader: DecadalDataLoader, time_idx: int) -> int:
    """Extract calendar year from data_loader time index."""
    time_value = data_loader.ds.time.values[time_idx]
    return pd.Timestamp(time_value).year


def get_indices_for_years(
    data_loader: DecadalDataLoader, selected_years: List[int]
) -> List[int]:
    """Return valid combination indices whose time index belongs to selected years."""
    selected_years_set = set(selected_years)
    valid_combos = data_loader.valid_combinations

    indices = []
    for idx in range(len(valid_combos)):
        time_idx = int(valid_combos.iloc[idx]["time_idx"])
        year = get_year_from_time_index(data_loader, time_idx)
        if year in selected_years_set:
            indices.append(idx)

    return indices


def sanitize_filename(text: str) -> str:
    """Sanitize a string for safe filename usage."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def resolve_checkpoint(experiment_dir: Path, checkpoint_spec: str) -> Path:
    """
    Resolve checkpoint path from spec:
    - 'best' -> checkpoints/best_model.pt
    - 'last' -> highest checkpoint_epoch_*.pt
    - otherwise treated as explicit file path
    """
    checkpoints_dir = experiment_dir / "checkpoints"

    if checkpoint_spec == "best":
        path = checkpoints_dir / "best_model.pt"
        if not path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {path}")
        return path

    if checkpoint_spec == "last":
        candidates = sorted(checkpoints_dir.glob("checkpoint_epoch_*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No epoch checkpoints found in: {checkpoints_dir}"
            )

        def epoch_num(path: Path) -> int:
            match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
            return int(match.group(1)) if match else -1

        candidates.sort(key=epoch_num)
        return candidates[-1]

    explicit = Path(checkpoint_spec)
    if not explicit.is_absolute():
        explicit = (experiment_dir / explicit).resolve()
    if not explicit.exists():
        raise FileNotFoundError(f"Checkpoint not found: {explicit}")
    return explicit


def strip_module_prefix_if_needed(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip leading 'module.' if checkpoint was saved from DDP model wrapper."""
    if not state_dict:
        return state_dict

    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def build_model_from_config(config: dict, device: str) -> ClimateNet:
    """Instantiate ClimateNet from config and move to device."""
    model = ClimateNet(
        static_channels=config["model"]["static_channels"],
        dynamic_channels=config["model"]["dynamic_channels"],
        image_size=tuple(config["model"]["image_size"]),
        encoder_type=config["model"]["encoder_type"],
        encoder_dim=config["model"]["encoder_dim"],
        encoder_blocks=config["model"]["encoder_blocks"],
        vit_patch_size=config["model"].get("vit_patch_size", 7),
        vit_num_heads=config["model"].get("vit_num_heads", 8),
        vit_mlp_ratio=config["model"].get("vit_mlp_ratio", 4.0),
        vit_dropout=config["model"].get("vit_dropout", 0.1),
        vit_attention_dropout=config["model"].get("vit_attention_dropout", 0.1),
        decoder_type=config["model"].get("decoder_type", "multi"),
        decoder_hidden_dims=config["model"]["decoder_hidden_dims"],
        target_vars=config["model"]["target_vars"],
        output_activations=config["model"].get("output_activations", None),
        use_film=config["model"]["use_film"],
        num_leads=config["model"]["num_leads"],
        lead_embed_dim=config["model"]["lead_embed_dim"],
    )
    return model.to(device)


def load_scalers_if_available(
    data_loader: DecadalDataLoader, norm_params_path: Path
) -> None:
    """Load saved normalization parameters into data_loader if file exists."""
    if norm_params_path.exists():
        with open(norm_params_path, "r") as f:
            data_loader.scalers = json.load(f)
        print(f"Loaded normalization parameters: {norm_params_path}")
    else:
        print(
            f"Normalization params not found at {norm_params_path}. "
            "Proceeding without external scaler file."
        )


def denormalize_prediction_if_needed(
    pred: np.ndarray,
    var_name: str,
    normalize_enabled: bool,
    data_loader: DecadalDataLoader,
) -> np.ndarray:
    """Denormalize prediction back to original units if normalization is enabled.

    Precipitation (tpERA) is additionally clamped to >= 0 after denormalization
    as a physical safeguard against any residual negative values that might
    survive if the model's output activation was not applied correctly.
    """
    if normalize_enabled and var_name in data_loader.scalers:
        out = data_loader.denormalize(pred.reshape(-1), var_name).reshape(pred.shape)
    else:
        out = pred
    # Physical constraint: precipitation cannot be negative
    if var_name == "tpERA":
        out = np.maximum(out, 0.0)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run inference from a trained ClimateNet model and save per-run/lead NetCDF outputs."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Experiment directory containing config.yaml and checkpoints/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint spec: 'best', 'last', or explicit checkpoint path",
    )
    parser.add_argument(
        "--years",
        type=str,
        required=True,
        help="Years to infer (e.g. '2017,2018,2020-2022')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for NetCDF files (default: <experiment-dir>/inference_outputs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. 'cpu', 'cuda', 'cuda:0')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of samples per model forward pass (default: 256)",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    checkpoint_path = resolve_checkpoint(experiment_dir, args.checkpoint)
    selected_years = parse_years(args.years)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (experiment_dir / "inference_outputs")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = config.get("training", {}).get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

    print("=" * 80)
    print("CLIMATENET INFERENCE")
    print("=" * 80)
    print(f"Experiment: {experiment_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Years: {selected_years}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    # Data loader + optional scaler params
    data_loader = DecadalDataLoader(
        nc_path=config["data"]["nc_path"],
        normalize_method=config["data"]["normalize_method"],
        cache_dir=config["data"]["cache_dir"],
        load_in_memory=config["data"].get("load_in_memory", True),
    )

    norm_params_path = experiment_dir / "normalization_params.json"
    load_scalers_if_available(data_loader, norm_params_path)

    # Dataset wrapper for model inputs
    full_dataset = ClimateDataset(
        data_loader=data_loader,
        normalize=config["data"].get("normalize", True),
        image_size=tuple(config["model"]["image_size"]),
        target_vars=config["model"]["target_vars"],
    )

    # Build model and load checkpoint
    model = build_model_from_config(config, device=device)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        # Backward compatibility with older torch versions that don't support
        # the weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError as error:
        # Some checkpoints include objects not allowlisted by PyTorch's strict
        # weights-only unpickler (e.g. numpy scalar metadata).
        print(
            "WARNING: weights_only=True failed while loading checkpoint. "
            "Falling back to weights_only=False for trusted checkpoint files only."
        )
        print(f"  Details: {error}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state_dict = strip_module_prefix_if_needed(state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    # Identify indices for requested years
    infer_indices = get_indices_for_years(data_loader, selected_years)
    if len(infer_indices) == 0:
        raise ValueError(
            f"No valid samples found for selected years: {selected_years}. "
            "Check available years in dataset."
        )

    print(f"Found {len(infer_indices):,} valid samples for inference")

    # -------------------------------------------------------------------------
    # Batched inference via DataLoader
    # -------------------------------------------------------------------------
    # Wrapper that also returns the original dataset index alongside each sample
    # so we can look up raw (un-normalised) data from data_loader after the
    # GPU forward pass.
    class _IndexedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            orig_idx = self.indices[i]
            return self.dataset[orig_idx], orig_idx

    indexed = _IndexedSubset(full_dataset, infer_indices)
    loader = DataLoader(
        indexed,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,           # data is already in RAM; workers add overhead
        pin_memory=("cuda" in device),
    )

    target_vars = config["model"]["target_vars"]
    H, W = tuple(config["model"]["image_size"])
    processed = 0
    grouped = {}

    for (batch_inputs, batch_targets, batch_meta), batch_orig_indices in loader:
        # batch_inputs  : (static [B,Cs,H,W], dynamic [B,Cd,H,W])
        # batch_targets : dict of tensors (not needed for inference output)
        # batch_meta    : dict, batch_meta["lead"] is a tensor of shape [B]
        # batch_orig_indices : LongTensor [B]

        static_b  = batch_inputs[0].to(device)
        dynamic_b = batch_inputs[1].to(device)
        lead_b    = batch_meta["lead"].to(device)

        with torch.no_grad():
            pred_dict = model(static=static_b, dynamic=dynamic_b, lead_indices=lead_b)

        B = static_b.shape[0]
        for b in range(B):
            orig_idx = int(batch_orig_indices[b].item())

            # Denormalize per-variable predictions
            pred_np = {}
            for var in target_vars:
                arr = pred_dict[var][b, 0].detach().cpu().numpy()  # (H, W)
                arr = denormalize_prediction_if_needed(
                    arr,
                    var_name=var,
                    normalize_enabled=config["data"].get("normalize", True),
                    data_loader=data_loader,
                )
                pred_np[var] = arr.astype(np.float32)

            # Raw inputs/targets are fetched per-sample from heap (already RAM)
            raw_inputs, raw_targets = data_loader[orig_idx]
            raw_inputs  = raw_inputs.reshape(19, H, W).astype(np.float32)
            raw_targets = raw_targets.reshape(4, H, W).astype(np.float32)

            combo_info = data_loader.get_combination_info(orig_idx)
            run        = combo_info["run"]
            lead       = int(combo_info["lead"])
            time_value = pd.Timestamp(combo_info["time"]).to_datetime64()

            key = (run, lead)
            if key not in grouped:
                grouped[key] = {
                    "times":   [],
                    "inputs":  {name: [] for name in INPUT_VAR_NAMES},
                    "targets": {name: [] for name in TARGET_VAR_NAMES},
                    "preds":   {name: [] for name in target_vars},
                }

            grouped[key]["times"].append(time_value)

            for in_name in INPUT_VAR_NAMES:
                grouped[key]["inputs"][in_name].append(
                    raw_inputs[INPUT_CHANNEL_INDEX[in_name]]
                )

            for i, t_name in enumerate(TARGET_VAR_NAMES):
                grouped[key]["targets"][t_name].append(raw_targets[i])

            for t_name in target_vars:
                grouped[key]["preds"][t_name].append(pred_np[t_name])

        processed += B
        if processed % 500 < args.batch_size or processed == len(infer_indices):
            print(f"Processed {processed:,}/{len(infer_indices):,} samples")

    # Shared spatial coordinates
    lat_vals = data_loader.ds.latitude.values.astype(np.float32)
    lon_vals = data_loader.ds.longitude.values.astype(np.float32)

    # Save one NetCDF per valid (run, lead)
    output_files = []
    for (run, lead), payload in grouped.items():
        times = np.array(payload["times"])
        order = np.argsort(times)
        times_sorted = times[order]

        data_vars = {}

        for in_name, arrays in payload["inputs"].items():
            stacked = np.stack(arrays, axis=0)[order]  # (T, H, W)
            data_vars[in_name] = (("time", "latitude", "longitude"), stacked)

        for t_name, arrays in payload["targets"].items():
            stacked = np.stack(arrays, axis=0)[order]
            data_vars[f"{t_name}_target"] = (
                ("time", "latitude", "longitude"),
                stacked,
            )

        for t_name, arrays in payload["preds"].items():
            stacked = np.stack(arrays, axis=0)[order]
            data_vars[f"{t_name}_pred"] = (("time", "latitude", "longitude"), stacked)

        ds_out = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": times_sorted,
                "latitude": lat_vals,
                "longitude": lon_vals,
            },
            attrs={
                "run": str(run),
                "lead": int(lead),
                "years_requested": ",".join(str(y) for y in selected_years),
                "checkpoint": str(checkpoint_path),
                "experiment_dir": str(experiment_dir),
            },
        )

        # NetCDF encoding: compress all data variables and store as float32.
        encoding = {
            name: {"zlib": True, "complevel": 5, "dtype": "float32"}
            for name in ds_out.data_vars
        }
        # Keep spatial coordinates as float32 in output file as well.
        encoding["latitude"] = {"dtype": "float32"}
        encoding["longitude"] = {"dtype": "float32"}

        safe_run = sanitize_filename(run)
        out_path = output_dir / f"inference_run_{safe_run}_lead_{lead}.nc"
        ds_out.to_netcdf(out_path, encoding=encoding)
        output_files.append(out_path)

    print("=" * 80)
    print(f"Saved {len(output_files)} NetCDF files")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    data_loader.close()


if __name__ == "__main__":
    main()
