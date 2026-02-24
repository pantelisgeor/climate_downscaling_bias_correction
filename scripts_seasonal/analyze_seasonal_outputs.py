"""
Post-process seasonal ClimateNet inference NetCDF outputs.

Direct analogue of scripts/analyze_inference_outputs.py adapted for the
seasonal forecast pipeline:

  • Input NetCDF files are named  inference_member_*_lead_*.nc
  • NetCDF global attributes use  member_number / lead_month  (not run / lead)
  • Input variable names in the NetCDFs: tp, t2m, tmax, hurs
    (decadal used: pr, tas, tasmax, hurs)

All analysis functionality – monthly aggregation, spatial maps, scatter plots,
bias plots, Taylor diagrams – is preserved unchanged from the decadal script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


# ── variable configuration ────────────────────────────────────────────────────
# 'input' key = the variable name stored in the inference NetCDF for the
#  raw forecast (seasonal names: tp, t2m, tmax, hurs).

VAR_CONFIG = {
    "tas": {
        "input":  "t2m",
        "target": "tasERA_target",
        "pred":   "tasERA_pred",
        "agg":    "mean",
        "label":  "Temperature",
        "units":  "°C",
    },
    "tasmax": {
        "input":  "tmax",
        "target": "tasmaxERA_target",
        "pred":   "tasmaxERA_pred",
        "agg":    "mean",
        "label":  "Max Temperature",
        "units":  "°C",
    },
    "tp": {
        "input":  "tp",
        "target": "tpERA_target",
        "pred":   "tpERA_pred",
        "agg":    "sum",
        "label":  "Total Precipitation",
        "units":  "mm",
    },
    "rh": {
        "input":  "hurs",
        "target": "rhERA_target",
        "pred":   "rhERA_pred",
        "agg":    "mean",
        "label":  "Relative Humidity",
        "units":  "%",
    },
}


# ── unit conversion ───────────────────────────────────────────────────────────

def _convert_units(arr: xr.DataArray, key: str) -> xr.DataArray:
    """Convert to standard plotting/output units."""
    out = arr.astype(np.float32)

    if key in ("tas", "tasmax"):
        # Kelvin → Celsius
        out = (out - 273.15).astype(np.float32)
    elif key == "tp":
        # m/day (ERA5 convention) → mm;  multiply monthly sums × 1000
        out = (out * 1000.0).astype(np.float32)
    elif key == "rh":
        finite = out.values[np.isfinite(out.values)]
        if finite.size > 0 and np.nanpercentile(finite, 99) <= 1.5:
            # fraction → percent
            out = (out * 100.0).astype(np.float32)

    return out


# ── file opening ──────────────────────────────────────────────────────────────

def open_and_tag_file(path: Path) -> xr.Dataset:
    """
    Open an inference NetCDF and attach member_number / lead_month coordinates.
    """
    ds = xr.open_dataset(path)
    member = str(ds.attrs.get("member_number", ds.attrs.get("run", "unknown")))
    lead   = int(ds.attrs.get("lead_month",    ds.attrs.get("lead", -1)))
    combo  = f"member_{member}_lead_{lead}"
    ds = ds.expand_dims(combo=[combo])
    ds = ds.assign_coords(
        member_number=("combo", [member]),
        lead_month=("combo", [lead]),
    )
    return ds


# ── monthly aggregation ───────────────────────────────────────────────────────

def monthly_aggregate(ds: xr.Dataset) -> xr.Dataset:
    out = xr.Dataset()

    for key, cfg in VAR_CONFIG.items():
        input_name  = cfg["input"]
        target_name = cfg["target"]
        pred_name   = cfg["pred"]
        agg         = cfg["agg"]

        if agg == "sum":
            in_m  = ds[input_name].resample(time="MS").sum()
            tgt_m = ds[target_name].resample(time="MS").sum()
            prd_m = ds[pred_name].resample(time="MS").sum()
        else:
            in_m  = ds[input_name].resample(time="MS").mean()
            tgt_m = ds[target_name].resample(time="MS").mean()
            prd_m = ds[pred_name].resample(time="MS").mean()

        out[f"{key}_input_monthly"]  = _convert_units(in_m,  key).astype(np.float32)
        out[f"{key}_target_monthly"] = _convert_units(tgt_m, key).astype(np.float32)
        out[f"{key}_pred_monthly"]   = _convert_units(prd_m, key).astype(np.float32)

    out = out.assign_coords(
        member_number=ds.member_number,
        lead_month=ds.lead_month,
    )
    return out


def collate_monthly(files: List[Path]) -> xr.Dataset:
    monthly_list = []
    for path in files:
        ds_file  = open_and_tag_file(path)
        ds_month = monthly_aggregate(ds_file)
        monthly_list.append(ds_month)
    collated = xr.concat(monthly_list, dim="combo")
    return collated.sortby("combo")


def save_collated(collated: xr.Dataset, output_path: Path) -> None:
    enc = {
        name: {"zlib": True, "complevel": 5, "dtype": "float32"}
        for name in collated.data_vars
    }
    enc["latitude"]  = {"dtype": "float32"}
    enc["longitude"] = {"dtype": "float32"}
    collated.to_netcdf(output_path, encoding=enc)


# ── utility ───────────────────────────────────────────────────────────────────

def robust_limits(a: np.ndarray, q_low=2.5, q_high=97.5) -> Tuple[float, float]:
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = float(np.percentile(finite, q_low)), float(np.percentile(finite, q_high))
    if np.isclose(lo, hi):
        lo -= 1.0; hi += 1.0
    return lo, hi


# ── map plots ─────────────────────────────────────────────────────────────────

def _add_map_axes(fig, pos, title: str, data: np.ndarray,
                  lat: np.ndarray, lon: np.ndarray,
                  cmap="viridis", vmin=None, vmax=None, dpi=200):
    if HAS_CARTOPY:
        ax = fig.add_subplot(pos, projection=ccrs.PlateCarree())
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3)
    else:
        ax = fig.add_subplot(pos)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                       extent=[lon.min(), lon.max(), lat.min(), lat.max()])
    ax.set_title(title, fontsize=9)
    return ax, im


def plot_spatial_panels(collated: xr.Dataset, output_dir: Path, dpi: int = 200):
    """6-panel maps (input mean, target mean, pred mean, input bias, pred bias, improvement)."""
    lat = collated.latitude.values
    lon = collated.longitude.values

    for key, cfg in VAR_CONFIG.items():
        in_key  = f"{key}_input_monthly"
        tgt_key = f"{key}_target_monthly"
        prd_key = f"{key}_pred_monthly"

        if not all(k in collated for k in (in_key, tgt_key, prd_key)):
            continue

        in_mean  = float(collated[in_key].mean("combo").mean("time").values)  # scalar check
        # Actually we want (lat, lon) arrays:
        in_spatial  = collated[in_key].mean(["combo","time"]).values
        tgt_spatial = collated[tgt_key].mean(["combo","time"]).values
        prd_spatial = collated[prd_key].mean(["combo","time"]).values

        in_bias  = in_spatial  - tgt_spatial
        prd_bias = prd_spatial - tgt_spatial
        improve  = np.abs(in_bias) - np.abs(prd_bias)

        vmin_val, vmax_val = robust_limits(np.concatenate([
            in_spatial.flatten(), tgt_spatial.flatten(), prd_spatial.flatten()
        ]))
        bias_lim = max(abs(float(np.nanpercentile(np.abs(in_bias), 97))),
                       abs(float(np.nanpercentile(np.abs(prd_bias), 97))))

        fig = plt.figure(figsize=(18, 10))
        for i, (data, title, cmap, vl, vh) in enumerate([
            (in_spatial,  f"{cfg['label']} – Input mean",   "viridis", vmin_val, vmax_val),
            (tgt_spatial, f"{cfg['label']} – Target mean",  "viridis", vmin_val, vmax_val),
            (prd_spatial, f"{cfg['label']} – Pred mean",    "viridis", vmin_val, vmax_val),
            (in_bias,     f"{cfg['label']} – Input bias",   "RdBu_r",  -bias_lim, bias_lim),
            (prd_bias,    f"{cfg['label']} – Pred bias",    "RdBu_r",  -bias_lim, bias_lim),
            (improve,     f"{cfg['label']} – Improvement",  "RdYlGn",  -bias_lim, bias_lim),
        ]):
            ax, im = _add_map_axes(fig, (2, 3, i + 1), title, data,
                                   lat, lon, cmap, vl, vh)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label=cfg["units"])

        plt.suptitle(f"{cfg['label']} – Seasonal seasonal (mean over all members/leads)", y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / f"maps_{key}.png", dpi=dpi, bbox_inches="tight")
        plt.close()


# ── scatter plots ─────────────────────────────────────────────────────────────

def plot_scatter(collated: xr.Dataset, output_dir: Path, dpi: int = 200):
    """2-panel scatter: input vs target (left) and pred vs target (right)."""
    for key, cfg in VAR_CONFIG.items():
        in_key  = f"{key}_input_monthly"
        tgt_key = f"{key}_target_monthly"
        prd_key = f"{key}_pred_monthly"
        if not all(k in collated for k in (in_key, tgt_key, prd_key)):
            continue

        in_v   = collated[in_key].values.flatten()
        tgt_v  = collated[tgt_key].values.flatten()
        prd_v  = collated[prd_key].values.flatten()

        mask = np.isfinite(in_v) & np.isfinite(tgt_v) & np.isfinite(prd_v)
        in_v, tgt_v, prd_v = in_v[mask], tgt_v[mask], prd_v[mask]

        if len(tgt_v) > 15_000:
            s = np.random.choice(len(tgt_v), 15_000, replace=False)
            in_v, tgt_v, prd_v = in_v[s], tgt_v[s], prd_v[s]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for ax, x, label in [(ax1, in_v, "Input"), (ax2, prd_v, "Pred")]:
            ax.scatter(tgt_v, x, alpha=0.3, s=1, c="steelblue")
            lo = min(tgt_v.min(), x.min())
            hi = max(tgt_v.max(), x.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
            r2 = 1 - np.sum((tgt_v - x)**2) / (np.sum((tgt_v - tgt_v.mean())**2) + 1e-8)
            bias = float(np.mean(x - tgt_v))
            rmse = float(np.sqrt(np.mean((x - tgt_v)**2)))
            ax.text(0.05, 0.95,
                    f"R²={r2:.3f}\nRMSE={rmse:.3f}\nBias={bias:.3f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            ax.set_xlabel(f"Target ({cfg['units']})")
            ax.set_ylabel(f"{label} ({cfg['units']})")
            ax.set_title(f"{cfg['label']} – {label} vs Target")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"scatter_{key}.png", dpi=dpi, bbox_inches="tight")
        plt.close()


# ── bias reliability ──────────────────────────────────────────────────────────

def plot_reliability_bias(collated: xr.Dataset, output_dir: Path,
                          n_bins: int = 20, dpi: int = 200):
    """2-row bias plot: top = input/pred bias by intensity; bottom = improvement."""
    for key, cfg in VAR_CONFIG.items():
        in_key  = f"{key}_input_monthly"
        tgt_key = f"{key}_target_monthly"
        prd_key = f"{key}_pred_monthly"
        if not all(k in collated for k in (in_key, tgt_key, prd_key)):
            continue

        in_v  = collated[in_key].values.flatten()
        tgt_v = collated[tgt_key].values.flatten()
        prd_v = collated[prd_key].values.flatten()
        mask  = np.isfinite(in_v) & np.isfinite(tgt_v) & np.isfinite(prd_v)
        in_v, tgt_v, prd_v = in_v[mask], tgt_v[mask], prd_v[mask]

        bins    = np.percentile(tgt_v, np.linspace(0, 100, n_bins + 1))
        centres = 0.5 * (bins[:-1] + bins[1:])
        in_bias_bins  = np.full(n_bins, np.nan)
        prd_bias_bins = np.full(n_bins, np.nan)
        improve_bins  = np.full(n_bins, np.nan)

        for b in range(n_bins):
            sel = (tgt_v >= bins[b]) & (tgt_v < bins[b + 1])
            if sel.sum() < 3:
                continue
            in_bias_bins[b]  = float(np.mean(in_v[sel]  - tgt_v[sel]))
            prd_bias_bins[b] = float(np.mean(prd_v[sel] - tgt_v[sel]))
            improve_bins[b]  = float(np.mean(np.abs(in_v[sel] - tgt_v[sel])
                                            - np.abs(prd_v[sel] - tgt_v[sel])))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        ax1.plot(centres, in_bias_bins,  "b-o", ms=4, label="Input bias")
        ax1.plot(centres, prd_bias_bins, "r-s", ms=4, label="Pred bias")
        ax1.axhline(0, color="k", lw=0.8, ls="--")
        ax1.set_ylabel(f"Bias ({cfg['units']})")
        ax1.set_title(f"{cfg['label']} – Reliability bias")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.bar(centres, improve_bins, width=np.diff(bins).mean() * 0.8,
                color=np.where(improve_bins >= 0, "green", "tomato"), alpha=0.8)
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_xlabel(f"Target intensity ({cfg['units']})")
        ax2.set_ylabel("MAE improvement")
        ax2.set_title("Improvement (positive = model better)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"bias_reliability_{key}.png",
                    dpi=dpi, bbox_inches="tight")
        plt.close()


# ── Taylor diagram ────────────────────────────────────────────────────────────

def plot_taylor_diagram(collated: xr.Dataset, output_dir: Path, dpi: int = 200):
    """Rudimentary Taylor diagram (normalised std vs correlation) for each variable."""
    import matplotlib.patches as mpatches

    for key, cfg in VAR_CONFIG.items():
        in_key  = f"{key}_input_monthly"
        tgt_key = f"{key}_target_monthly"
        prd_key = f"{key}_pred_monthly"
        if not all(k in collated for k in (in_key, tgt_key, prd_key)):
            continue

        tgt_v = collated[tgt_key].values.flatten()
        in_v  = collated[in_key].values.flatten()
        prd_v = collated[prd_key].values.flatten()
        mask  = np.isfinite(tgt_v) & np.isfinite(in_v) & np.isfinite(prd_v)
        tgt_v, in_v, prd_v = tgt_v[mask], in_v[mask], prd_v[mask]

        std_ref = float(np.std(tgt_v)) + 1e-8
        ref_std = 1.0

        def point(x):
            r   = float(np.corrcoef(tgt_v, x)[0, 1])
            std = float(np.std(x)) / std_ref
            return std, r

        in_std,  in_r  = point(in_v)
        prd_std, prd_r = point(prd_v)

        angles = np.linspace(0, np.pi / 2, 181)
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
        ax.set_thetamax(90)

        # Concentric std circles
        for s in [0.5, 1.0, 1.5, 2.0]:
            ax.plot(angles, np.full_like(angles, s), "gray", lw=0.5, ls="--")
        # Reference point
        ax.plot(0, ref_std, "k*", ms=12, label="Reference (ERA5)")
        # Input
        ax.plot(np.arccos(in_r), in_std, "b^", ms=10, label=f"Input  r={in_r:.2f}")
        # Prediction
        ax.plot(np.arccos(prd_r), prd_std, "rs", ms=10, label=f"Pred   r={prd_r:.2f}")

        ax.set_title(f"{cfg['label']} – Taylor diagram", pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"taylor_{key}.png", dpi=dpi, bbox_inches="tight")
        plt.close()


# ── per-lead analysis ─────────────────────────────────────────────────────────

def plot_per_lead_metrics(collated: xr.Dataset, output_dir: Path, dpi: int = 200):
    """RMSE / bias as a function of lead month."""
    if "lead_month" not in collated.coords:
        return

    unique_leads = sorted(int(x) for x in np.unique(collated.lead_month.values))
    if len(unique_leads) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_i, (key, cfg) in enumerate(VAR_CONFIG.items()):
        ax     = axes[ax_i]
        tgt_k  = f"{key}_target_monthly"
        prd_k  = f"{key}_pred_monthly"
        if not all(k in collated for k in (tgt_k, prd_k)):
            continue

        rmse_per_lead = []
        bias_per_lead = []
        for lm in unique_leads:
            sel = collated.sel(combo=(collated.lead_month == lm))
            tgt = sel[tgt_k].values.flatten()
            prd = sel[prd_k].values.flatten()
            mask = np.isfinite(tgt) & np.isfinite(prd)
            if mask.sum() < 2:
                rmse_per_lead.append(np.nan)
                bias_per_lead.append(np.nan)
                continue
            rmse_per_lead.append(float(np.sqrt(np.mean((prd[mask] - tgt[mask])**2))))
            bias_per_lead.append(float(np.mean(prd[mask] - tgt[mask])))

        ax.plot(unique_leads, rmse_per_lead, "r-o", label="RMSE", ms=6)
        ax.plot(unique_leads, bias_per_lead, "b-s", label="Bias", ms=6)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Lead month"); ax.set_ylabel(cfg["units"])
        ax.set_title(f"{cfg['label']} – metrics by lead month")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_by_lead_month.png", dpi=dpi, bbox_inches="tight")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and visualise seasonal inference NetCDF outputs"
    )
    parser.add_argument("--input-dir",        required=True)
    parser.add_argument("--output-dir",        default=None)
    parser.add_argument("--pattern",           default="inference_member_*_lead_*.nc")
    parser.add_argument("--reliability-bins",  type=int, default=20)
    parser.add_argument("--dpi",               type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir \
                 else (input_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{args.pattern}' in {input_dir}"
        )
    print(f"Found {len(files)} inference files")

    # ── collate monthly aggregates ────────────────────────────────────────────
    print("Computing monthly aggregates …")
    collated = collate_monthly(files)

    collated_path = output_dir / "collated_monthly.nc"
    save_collated(collated, collated_path)
    print(f"Collated monthly data → {collated_path}")

    # ── plots ─────────────────────────────────────────────────────────────────
    print("Generating spatial maps …")
    plot_spatial_panels(collated, output_dir, dpi=args.dpi)

    print("Generating scatter plots …")
    plot_scatter(collated, output_dir, dpi=args.dpi)

    print("Generating reliability bias plots …")
    plot_reliability_bias(collated, output_dir,
                          n_bins=args.reliability_bins, dpi=args.dpi)

    print("Generating Taylor diagrams …")
    plot_taylor_diagram(collated, output_dir, dpi=args.dpi)

    print("Generating per-lead-month metrics …")
    plot_per_lead_metrics(collated, output_dir, dpi=args.dpi)

    print(f"All outputs saved → {output_dir}")


if __name__ == "__main__":
    main()
