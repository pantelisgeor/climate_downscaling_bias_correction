"""
Inference script for decadal high-resolution downscaling.

Adapted from scripts/infer.py for the DecadalDownscaleDataLoader.

Key differences from infer.py:
  • Uses DecadalDownscaleDataLoader / ClimateDatasetDownscale.
  • Spatial coordinates are read from a representative NC file
    (not from data_loader.ds, which doesn't exist in the disk-based loader).
  • Time values are looked up from the source NC files via the loader's
    LRU handle cache.
  • get_combination_info returns {file_path, time_idx, year, run, lead} —
    the actual datetime is decoded from the source file.
  • Output NetCDFs are named  inference_run_{run}_lead_{lead}.nc
    (same convention as scripts/infer.py so analyze_dec_downscale_outputs.py
    can be used without modification).
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List

import netCDF4 as nc4
import numpy as np
import torch
import xarray as xr
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader_decadal_downscale import DecadalDownscaleDataLoader, _get_handle
from src.models.climate_net import ClimateNet
from src.training_downscale.climate_dataset_downscale import ClimateDatasetDownscale


# ── constants ─────────────────────────────────────────────────────────────────
INPUT_VAR_NAMES      = ["pr", "tas", "tasmax", "hurs"]
INPUT_CHANNEL_INDEX  = {"pr": 0, "tas": 1, "tasmax": 2, "hurs": 3}
TARGET_VAR_NAMES     = ["tasERA", "tasmaxERA", "tpERA", "rhERA"]


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_years(year_text: str) -> List[int]:
    years: set = set()
    for token in year_text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            lo, hi = min(int(a), int(b)), max(int(a), int(b))
            years.update(range(lo, hi + 1))
        else:
            years.add(int(token))
    if not years:
        raise ValueError("No valid years parsed from --years")
    return sorted(years)


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def resolve_checkpoint(experiment_dir: Path, spec: str) -> Path:
    ckpt_dir = experiment_dir / "checkpoints"
    if spec == "best":
        p = ckpt_dir / "best_model.pt"
        if not p.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {p}")
        return p
    if spec == "last":
        cands = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        if not cands:
            raise FileNotFoundError(f"No epoch checkpoints in: {ckpt_dir}")
        def epoch_num(p: Path) -> int:
            m = re.search(r"checkpoint_epoch_(\d+)\.pt$", p.name)
            return int(m.group(1)) if m else -1
        return sorted(cands, key=epoch_num)[-1]
    explicit = Path(spec)
    if not explicit.is_absolute():
        explicit = (experiment_dir / explicit).resolve()
    if not explicit.exists():
        raise FileNotFoundError(f"Checkpoint not found: {explicit}")
    return explicit


def strip_module_prefix(sd: Dict) -> Dict:
    if sd and next(iter(sd)).startswith("module."):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def build_model(config: dict, device: str) -> ClimateNet:
    return ClimateNet(
        static_channels      = config["model"]["static_channels"],
        dynamic_channels     = config["model"]["dynamic_channels"],
        image_size           = tuple(config["model"]["image_size"]),
        encoder_type         = config["model"]["encoder_type"],
        encoder_dim          = config["model"]["encoder_dim"],
        encoder_blocks       = config["model"]["encoder_blocks"],
        vit_patch_size       = config["model"].get("vit_patch_size", 7),
        vit_num_heads        = config["model"].get("vit_num_heads", 8),
        vit_mlp_ratio        = config["model"].get("vit_mlp_ratio", 4.0),
        vit_dropout          = config["model"].get("vit_dropout", 0.1),
        vit_attention_dropout= config["model"].get("vit_attention_dropout", 0.1),
        decoder_type         = config["model"].get("decoder_type", "multi"),
        decoder_hidden_dims  = config["model"]["decoder_hidden_dims"],
        target_vars          = config["model"]["target_vars"],
        output_activations   = config["model"].get("output_activations", None),
        use_film             = config["model"]["use_film"],
        num_leads            = config["model"]["num_leads"],
        lead_embed_dim       = config["model"]["lead_embed_dim"],
        dilations            = config["model"].get("dilations", None),
        padding_mode         = config["model"].get("padding_mode", "reflect"),
    ).to(device)


def get_time_value(filepath: str, time_idx: int) -> np.datetime64:
    """
    Read the decoded datetime64 for time_idx from the NC file.

    Reuses the LRU handle from the data loader to avoid extra open() calls.
    Falls back to a synthetic date if decoding fails.
    """
    try:
        ds = _get_handle(filepath)
        raw = int(ds.variables["time"][time_idx])
        units = getattr(ds.variables["time"], "units", "days since 1900-01-01")
        cal   = getattr(ds.variables["time"], "calendar", "standard")
        import cftime
        dt = cftime.num2date(raw, units=units, calendar=cal)
        return np.datetime64(
            f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}", "D"
        ).astype("datetime64[ns]")
    except Exception:
        # Fallback: build a date from the file's year and time_idx as day offset
        stem  = Path(filepath).stem   # e.g. 1981_r1i1p1f1_0_
        parts = stem.rstrip("_").split("_")
        year  = int(parts[0]) if parts else 2000
        return np.datetime64(f"{year:04d}-01-01", "D").astype("datetime64[ns]") + \
               np.timedelta64(time_idx, "D")


def get_spatial_coords(data_loader: DecadalDownscaleDataLoader,
                       H: int, W: int) -> tuple:
    """Return (lat_vals, lon_vals) as float32 arrays from the first NC file."""
    fp = str(data_loader._index.iloc[0]["file_path"])
    ds = _get_handle(fp)
    lat = ds.variables["latitude"][:].astype(np.float32)
    lon = ds.variables["longitude"][:].astype(np.float32)
    return lat, lon


def denormalize_pred(pred: np.ndarray, var_name: str,
                     data_loader: DecadalDownscaleDataLoader) -> np.ndarray:
    out = data_loader.denormalize(pred.reshape(-1), var_name).reshape(pred.shape)
    if var_name == "tpERA":
        out = np.maximum(out, 0.0)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Decadal downscale inference — saves per-run/lead NetCDF files."
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--checkpoint",     default="best")
    parser.add_argument("--years",          required=True,
                        help="Years to run inference for, e.g. '2016,2017-2020'")
    parser.add_argument("--output-dir",     default=None)
    parser.add_argument("--device",         default=None)
    parser.add_argument("--batch-size",     type=int, default=16)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {experiment_dir}")

    config_path = experiment_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint_path = resolve_checkpoint(experiment_dir, args.checkpoint)
    selected_years  = parse_years(args.years)

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else (experiment_dir / "inference_outputs")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or config.get("training", {}).get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 80)
    print("DECADAL DOWNSCALE INFERENCE")
    print("=" * 80)
    print(f"Experiment: {experiment_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Years:      {selected_years}")
    print(f"Output:     {output_dir}")
    print(f"Device:     {device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    # ── data loader ───────────────────────────────────────────────────────────
    data_loader = DecadalDownscaleDataLoader(
        data_dir    = config["data"]["data_dir"],
        cache_dir   = config["data"]["cache_dir"],
        train_years = config["data"]["train_years"],
    )

    full_dataset = ClimateDatasetDownscale(
        data_loader = data_loader,
        normalize   = False,
        image_size  = tuple(config["model"]["image_size"]),
        target_vars = config["model"]["target_vars"],
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(config, device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(strip_module_prefix(sd))
    model.eval()

    # ── find indices for requested years ──────────────────────────────────────
    infer_indices = data_loader.get_split_indices(selected_years)
    if not infer_indices:
        raise ValueError(f"No valid samples for years: {selected_years}")
    print(f"Found {len(infer_indices):,} samples for inference")

    # ── spatial coordinates ───────────────────────────────────────────────────
    H, W = tuple(config["model"]["image_size"])
    lat_vals, lon_vals = get_spatial_coords(data_loader, H, W)

    # ── run inference batched ─────────────────────────────────────────────────

    class _IndexedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            orig = self.indices[i]
            return self.dataset[orig], orig

    loader = DataLoader(
        _IndexedSubset(full_dataset, infer_indices),
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = ("cuda" in device),
    )

    target_vars = config["model"]["target_vars"]
    grouped: Dict = {}
    processed = 0

    for (batch_inputs, batch_targets, batch_meta), batch_orig_indices in loader:
        static_b  = batch_inputs[0].to(device)
        dynamic_b = batch_inputs[1].to(device)
        lead_b    = batch_meta["lead"].to(device)

        with torch.no_grad():
            pred_dict = model(static=static_b, dynamic=dynamic_b, lead_indices=lead_b)

        B = static_b.shape[0]
        for b in range(B):
            orig_idx = int(batch_orig_indices[b].item())

            # Denormalize per-variable predictions
            pred_np: Dict[str, np.ndarray] = {}
            for var in target_vars:
                arr = pred_dict[var][b, 0].detach().cpu().numpy()  # (H, W)
                pred_np[var] = denormalize_pred(arr, var, data_loader).astype(np.float32)

            # Raw (un-normalized) inputs + targets from loader
            raw_inputs, raw_tgts = data_loader[orig_idx]
            raw_inputs = raw_inputs.reshape(19, H, W).astype(np.float32)
            raw_tgts   = raw_tgts.reshape(4, H, W).astype(np.float32)

            combo   = data_loader.get_combination_info(orig_idx)
            run     = str(combo["run"])
            lead    = int(combo["lead"])
            time_val = get_time_value(combo["file_path"], combo["time_idx"])

            key = (run, lead)
            if key not in grouped:
                grouped[key] = {
                    "times":   [],
                    "inputs":  {n: [] for n in INPUT_VAR_NAMES},
                    "targets": {n: [] for n in TARGET_VAR_NAMES},
                    "preds":   {n: [] for n in target_vars},
                }

            g = grouped[key]
            g["times"].append(time_val)
            for in_name in INPUT_VAR_NAMES:
                g["inputs"][in_name].append(raw_inputs[INPUT_CHANNEL_INDEX[in_name]])
            for i, t_name in enumerate(TARGET_VAR_NAMES):
                g["targets"][t_name].append(raw_tgts[i])
            for t_name in target_vars:
                g["preds"][t_name].append(pred_np[t_name])

        processed += B
        if processed % 500 < args.batch_size or processed == len(infer_indices):
            print(f"  Processed {processed:,}/{len(infer_indices):,}")

    # ── save NetCDF files ─────────────────────────────────────────────────────
    output_files = []
    for (run, lead), payload in grouped.items():
        times  = np.array(payload["times"])
        order  = np.argsort(times)
        ts_out = times[order]

        data_vars: Dict = {}

        for in_name, arrays in payload["inputs"].items():
            data_vars[in_name] = (
                ("time", "latitude", "longitude"),
                np.stack(arrays, axis=0)[order].astype(np.float32),
            )
        for t_name, arrays in payload["targets"].items():
            data_vars[f"{t_name}_target"] = (
                ("time", "latitude", "longitude"),
                np.stack(arrays, axis=0)[order].astype(np.float32),
            )
        for t_name, arrays in payload["preds"].items():
            data_vars[f"{t_name}_pred"] = (
                ("time", "latitude", "longitude"),
                np.stack(arrays, axis=0)[order].astype(np.float32),
            )

        ds_out = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time":      ts_out,
                "latitude":  lat_vals,
                "longitude": lon_vals,
            },
            attrs={
                "run":             str(run),
                "lead":            int(lead),
                "years_requested": ",".join(str(y) for y in selected_years),
                "checkpoint":      str(checkpoint_path),
                "experiment_dir":  str(experiment_dir),
            },
        )

        encoding = {
            name: {"zlib": True, "complevel": 5, "dtype": "float32"}
            for name in ds_out.data_vars
        }
        encoding["latitude"]  = {"dtype": "float32"}
        encoding["longitude"] = {"dtype": "float32"}

        out_path = output_dir / f"inference_run_{sanitize_filename(run)}_lead_{lead}.nc"
        ds_out.to_netcdf(out_path, encoding=encoding)
        output_files.append(out_path)

    print("=" * 80)
    print(f"Saved {len(output_files)} NetCDF files → {output_dir}")
    print("=" * 80)

    data_loader.close()


if __name__ == "__main__":
    main()
