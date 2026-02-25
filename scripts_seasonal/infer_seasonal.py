"""
Inference script for seasonal ClimateNet.

Adapted from scripts/infer.py.  Key differences:
  • Uses SeasonalDataLoader / ClimateDatasetSeasonal
  • Forecast variable names are  tp, t2m, tmax, hurs  (not pr, tas, tasmax, hurs)
  • Outputs are grouped by (lead_month, year) with all 25 ensemble members
    stacked along a 'member' dimension
  • Output NetCDF files are named  inference_lead_{L}_year_{Y}.nc
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader_seasonal import SeasonalDataLoader
from src.models.climate_net import ClimateNet
from src.training_seasonal.climate_dataset_seasonal import ClimateDatasetSeasonal


# ── constants ─────────────────────────────────────────────────────────────────
INPUT_VAR_NAMES = ["tp", "t2m", "tmax", "hurs"]   # channels 0-3
INPUT_CHANNEL_INDEX = {"tp": 0, "t2m": 1, "tmax": 2, "hurs": 3}
TARGET_VAR_NAMES = ["tasERA", "tasmaxERA", "tpERA", "rhERA"]


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


def get_indices_for_years(data_loader: SeasonalDataLoader,
                          selected_years: List[int]) -> List[int]:
    """Return positional indices into valid_combinations whose time year is in selected_years.

    Uses data_loader._np_times (always in memory, even when load_in_memory=True
    closes the xarray Dataset) and vectorised numpy operations for speed.
    """
    wanted = set(selected_years)
    # _np_times is a (n_time,) datetime64 array set during __init__
    time_years = pd.DatetimeIndex(data_loader._np_times).year.to_numpy()
    # _vc_time_idx is a (N,) int32 array of time indices per valid combination
    sample_years = time_years[data_loader._vc_time_idx]
    mask = np.isin(sample_years, list(wanted))
    return list(np.where(mask)[0])


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def resolve_checkpoint(experiment_dir: Path, checkpoint_spec: str) -> Path:
    ckpt_dir = experiment_dir / "checkpoints"
    if checkpoint_spec == "best":
        p = ckpt_dir / "best_model.pt"
        if not p.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {p}")
        return p
    if checkpoint_spec == "last":
        cands = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        if not cands:
            raise FileNotFoundError(f"No epoch checkpoints in: {ckpt_dir}")
        def epoch_num(p: Path) -> int:
            m = re.search(r"checkpoint_epoch_(\d+)\.pt$", p.name)
            return int(m.group(1)) if m else -1
        return sorted(cands, key=epoch_num)[-1]
    explicit = Path(checkpoint_spec)
    if not explicit.is_absolute():
        explicit = (experiment_dir / explicit).resolve()
    if not explicit.exists():
        raise FileNotFoundError(f"Checkpoint not found: {explicit}")
    return explicit


def strip_module_prefix(sd: Dict) -> Dict:
    if not sd:
        return sd
    if next(iter(sd)).startswith("module."):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def build_model(config: dict, device: str) -> ClimateNet:
    return ClimateNet(
        static_channels    = config["model"]["static_channels"],
        dynamic_channels   = config["model"]["dynamic_channels"],
        image_size         = tuple(config["model"]["image_size"]),
        encoder_type       = config["model"]["encoder_type"],
        encoder_dim        = config["model"]["encoder_dim"],
        encoder_blocks     = config["model"]["encoder_blocks"],
        vit_patch_size     = config["model"].get("vit_patch_size", 7),
        vit_num_heads      = config["model"].get("vit_num_heads", 8),
        vit_mlp_ratio      = config["model"].get("vit_mlp_ratio", 4.0),
        vit_dropout        = config["model"].get("vit_dropout", 0.1),
        vit_attention_dropout = config["model"].get("vit_attention_dropout", 0.1),
        decoder_type       = config["model"].get("decoder_type", "multi"),
        decoder_hidden_dims= config["model"]["decoder_hidden_dims"],
        target_vars        = config["model"]["target_vars"],
        output_activations = config["model"].get("output_activations", None),
        use_film           = config["model"]["use_film"],
        num_leads          = config["model"]["num_leads"],
        lead_embed_dim     = config["model"]["lead_embed_dim"],
        dilations          = config["model"].get("dilations", None),
        padding_mode       = config["model"].get("padding_mode", "zeros"),
    ).to(device)


def load_scalers(data_loader: SeasonalDataLoader, norm_params_path: Path) -> None:
    if norm_params_path.exists():
        with open(norm_params_path) as f:
            data_loader.scalers = json.load(f)
        print(f"Loaded normalization params: {norm_params_path}")
    else:
        print(f"Normalization params not found at {norm_params_path} – "
              "proceeding without.")


def denormalize_pred(pred: np.ndarray, var_name: str,
                     normalize_enabled: bool,
                     data_loader: SeasonalDataLoader) -> np.ndarray:
    if normalize_enabled and var_name in data_loader.scalers:
        out = data_loader.denormalize(pred.reshape(-1), var_name).reshape(pred.shape)
    else:
        out = pred
    if var_name == "tpERA":
        out = np.maximum(out, 0.0)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Seasonal ClimateNet inference – saves per-member/lead NetCDF files."
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--checkpoint",     default="best")
    parser.add_argument("--years",          required=True,
                        help="Years to infer, e.g. '2010,2011,2012-2015'")
    parser.add_argument("--output-dir",     default=None)
    parser.add_argument("--device",         default=None)
    parser.add_argument("--batch-size",     type=int, default=256)
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
    print("SEASONAL CLIMATENET INFERENCE")
    print("=" * 80)
    print(f"Experiment: {experiment_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Years:      {selected_years}")
    print(f"Output:     {output_dir}")
    print(f"Device:     {device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    # ── data loader ───────────────────────────────────────────────────────────
    data_loader = SeasonalDataLoader(
        nc_path          = config["data"]["nc_path"],
        normalize_method = config["data"]["normalize_method"],
        cache_dir        = config["data"]["cache_dir"],
        load_in_memory   = config["data"].get("load_in_memory", True),
    )
    load_scalers(data_loader, experiment_dir / "normalization_params.json")

    full_dataset = ClimateDatasetSeasonal(
        data_loader  = data_loader,
        normalize    = config["data"].get("normalize", True),
        image_size   = tuple(config["model"]["image_size"]),
        target_vars  = config["model"]["target_vars"],
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

    # ── find indices ──────────────────────────────────────────────────────────
    infer_indices = get_indices_for_years(data_loader, selected_years)
    if not infer_indices:
        raise ValueError(f"No valid samples for years: {selected_years}")
    print(f"Found {len(infer_indices):,} samples")

    # ── DataLoader over indexed subset ────────────────────────────────────────
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=("cuda" in device),
    )

    target_vars = config["model"]["target_vars"]
    H, W = tuple(config["model"]["image_size"])
    # grouped[(lead_month, year)][member_number] = {times, inputs, targets, preds}
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

            # Per-variable predictions (denormalized)
            pred_np: Dict[str, np.ndarray] = {}
            for var in target_vars:
                arr = pred_dict[var][b, 0].detach().cpu().numpy()
                arr = denormalize_pred(
                    arr, var_name=var,
                    normalize_enabled=config["data"].get("normalize", True),
                    data_loader=data_loader,
                )
                pred_np[var] = arr.astype(np.float32)

            # Raw (un-normalized) inputs + targets for archival
            raw_inputs, raw_tgts = data_loader[orig_idx]
            raw_inputs = raw_inputs.reshape(19, H, W).astype(np.float32)
            raw_tgts   = raw_tgts.reshape(4, H, W).astype(np.float32)

            combo_info = data_loader.get_combination_info(orig_idx)
            number     = int(combo_info["number"])
            lead_month = int(combo_info["lead_month"])
            time_val   = pd.Timestamp(combo_info["time"]).to_datetime64()
            year       = pd.Timestamp(combo_info["time"]).year

            file_key   = (lead_month, year)
            if file_key not in grouped:
                grouped[file_key] = {}
            if number not in grouped[file_key]:
                grouped[file_key][number] = {
                    "times":   [],
                    "inputs":  {n: [] for n in INPUT_VAR_NAMES},
                    "targets": {n: [] for n in TARGET_VAR_NAMES},
                    "preds":   {n: [] for n in target_vars},
                }

            g = grouped[file_key][number]
            g["times"].append(time_val)
            for in_name in INPUT_VAR_NAMES:
                g["inputs"][in_name].append(raw_inputs[INPUT_CHANNEL_INDEX[in_name]])
            for i, t_name in enumerate(TARGET_VAR_NAMES):
                g["targets"][t_name].append(raw_tgts[i])
            for t_name in target_vars:
                g["preds"][t_name].append(pred_np[t_name])

        processed += B
        if processed % 500 < args.batch_size or processed == len(infer_indices):
            print(f"Processed {processed:,}/{len(infer_indices):,}")

    # ── save NetCDF files ─────────────────────────────────────────────────────
    lat_vals = data_loader.ds.latitude.values.astype(np.float32)
    lon_vals = data_loader.ds.longitude.values.astype(np.float32)

    output_files = []
    for (lead_month, year), member_dict in grouped.items():
        members_sorted = sorted(member_dict.keys())

        # Build shared time axis from first member (all members share the same times)
        ref = member_dict[members_sorted[0]]
        times_sorted = np.sort(np.array(ref["times"]))

        data_vars: Dict = {}

        for in_name in INPUT_VAR_NAMES:
            arrs = []
            for m in members_sorted:
                g = member_dict[m]
                order = np.argsort(np.array(g["times"]))
                arrs.append(np.stack(g["inputs"][in_name], axis=0)[order])
            data_vars[in_name] = (
                ("member", "time", "latitude", "longitude"),
                np.stack(arrs, axis=0).astype(np.float32),
            )

        for t_name in TARGET_VAR_NAMES:
            arrs = []
            for m in members_sorted:
                g = member_dict[m]
                order = np.argsort(np.array(g["times"]))
                arrs.append(np.stack(g["targets"][t_name], axis=0)[order])
            data_vars[f"{t_name}_target"] = (
                ("member", "time", "latitude", "longitude"),
                np.stack(arrs, axis=0).astype(np.float32),
            )

        for t_name in target_vars:
            arrs = []
            for m in members_sorted:
                g = member_dict[m]
                order = np.argsort(np.array(g["times"]))
                arrs.append(np.stack(g["preds"][t_name], axis=0)[order])
            data_vars[f"{t_name}_pred"] = (
                ("member", "time", "latitude", "longitude"),
                np.stack(arrs, axis=0).astype(np.float32),
            )

        ds_out = xr.Dataset(
            data_vars=data_vars,
            coords={
                "member":    members_sorted,
                "time":      times_sorted,
                "latitude":  lat_vals,
                "longitude": lon_vals,
            },
            attrs={
                "lead_month":      int(lead_month),
                "year":            int(year),
                "n_members":       len(members_sorted),
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

        out_path = output_dir / f"inference_lead_{lead_month}_year_{year}.nc"
        ds_out.to_netcdf(out_path, encoding=encoding)
        output_files.append(out_path)

    print("=" * 80)
    print(f"Saved {len(output_files)} NetCDF files → {output_dir}")
    print("  Naming: inference_lead_{{lead}}_year_{{year}}.nc  (member dim inside)")
    print("=" * 80)

    data_loader.close()


if __name__ == "__main__":
    main()
