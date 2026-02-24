"""
Post-process ClimateNet inference NetCDF outputs.

This script:
1) Reads all inference NetCDF files from a directory, computes monthly aggregates
   (means for temperature/humidity, sums for precipitation) for input/target/pred,
   and saves a single collated NetCDF.
2) Creates 6-panel (3x2) maps for each variable.
3) Creates reliability-style bias plots (top: input/pred bias, bottom: improvement).
4) Creates 1x2 scatter plots (input vs target, pred vs target).
5) Creates Taylor diagrams (input and pred vs target).
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


VAR_CONFIG = {
    "tas": {
        "input": "tas",
        "target": "tasERA_target",
        "pred": "tasERA_pred",
        "agg": "mean",
        "label": "Temperature",
        "units": "°C",
    },
    "tasmax": {
        "input": "tasmax",
        "target": "tasmaxERA_target",
        "pred": "tasmaxERA_pred",
        "agg": "mean",
        "label": "Max Temperature",
        "units": "°C",
    },
    "tp": {
        "input": "pr",
        "target": "tpERA_target",
        "pred": "tpERA_pred",
        "agg": "sum",
        "label": "Total Precipitation",
        "units": "mm",
    },
    "rh": {
        "input": "hurs",
        "target": "rhERA_target",
        "pred": "rhERA_pred",
        "agg": "mean",
        "label": "Relative Humidity",
        "units": "%",
    },
}


def _convert_units(arr: xr.DataArray, key: str) -> xr.DataArray:
    """Convert to requested plotting/output units.

    - tas/tasmax and corresponding targets/preds: Kelvin -> Celsius
    - tp and pr monthly totals: meters -> millimeters
    - rh/hurs and corresponding targets/preds: fraction -> percent (if needed)
    """
    out = arr.astype(np.float32)

    if key in {"tas", "tasmax"}:
        out = (out - 273.15).astype(np.float32)
    elif key == "tp":
        out = (out * 1000.0).astype(np.float32)
    elif key == "rh":
        finite = out.values[np.isfinite(out.values)]
        if finite.size > 0 and np.nanpercentile(finite, 99) <= 1.5:
            out = (out * 100.0).astype(np.float32)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize inference NetCDF outputs"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing inference_run_*_lead_*.nc files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <input-dir>/analysis)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="inference_run_*_lead_*.nc",
        help="Glob pattern for inference files",
    )
    parser.add_argument(
        "--reliability-bins",
        type=int,
        default=20,
        help="Number of bins for reliability plots",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI",
    )
    return parser.parse_args()


def open_and_tag_file(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)
    run = str(ds.attrs.get("run", "unknown"))
    lead = int(ds.attrs.get("lead", -1))
    combo = f"run_{run}_lead_{lead}"
    ds = ds.expand_dims(combo=[combo])
    ds = ds.assign_coords(run=("combo", [run]), lead=("combo", [lead]))
    return ds


def monthly_aggregate(ds: xr.Dataset) -> xr.Dataset:
    out = xr.Dataset()

    for key, cfg in VAR_CONFIG.items():
        input_name = cfg["input"]
        target_name = cfg["target"]
        pred_name = cfg["pred"]
        agg = cfg["agg"]

        if agg == "sum":
            input_month = ds[input_name].resample(time="MS").sum()
            target_month = ds[target_name].resample(time="MS").sum()
            pred_month = ds[pred_name].resample(time="MS").sum()
        else:
            input_month = ds[input_name].resample(time="MS").mean()
            target_month = ds[target_name].resample(time="MS").mean()
            pred_month = ds[pred_name].resample(time="MS").mean()

        input_month = _convert_units(input_month, key)
        target_month = _convert_units(target_month, key)
        pred_month = _convert_units(pred_month, key)

        out[f"{key}_input_monthly"] = input_month.astype(np.float32)
        out[f"{key}_target_monthly"] = target_month.astype(np.float32)
        out[f"{key}_pred_monthly"] = pred_month.astype(np.float32)

    out = out.assign_coords(run=ds.run, lead=ds.lead)
    return out


def collate_monthly(files: List[Path]) -> xr.Dataset:
    monthly_list = []
    for path in files:
        ds_file = open_and_tag_file(path)
        ds_month = monthly_aggregate(ds_file)
        monthly_list.append(ds_month)

    collated = xr.concat(monthly_list, dim="combo")
    # Ensure deterministic order
    collated = collated.sortby("combo")
    return collated


def save_collated(collated: xr.Dataset, output_path: Path) -> None:
    encoding = {
        name: {"zlib": True, "complevel": 5, "dtype": "float32"}
        for name in collated.data_vars
    }
    encoding["latitude"] = {"dtype": "float32"}
    encoding["longitude"] = {"dtype": "float32"}
    collated.to_netcdf(output_path, encoding=encoding)


def robust_limits(a: np.ndarray, q_low: float = 2.5, q_high: float = 97.5) -> Tuple[float, float]:
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.percentile(finite, q_low))
    hi = float(np.percentile(finite, q_high))
    if np.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    return lo, hi


def sym_limits(a: np.ndarray, q: float = 97.5) -> Tuple[float, float]:
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = float(np.percentile(np.abs(finite), q))
    if np.isclose(vmax, 0.0):
        vmax = 1.0
    return -vmax, vmax


def create_six_panel_maps(collated: xr.Dataset, output_dir: Path, dpi: int = 200) -> None:
    map_dir = output_dir / "maps_6panel"
    map_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_CARTOPY:
        print(
            "WARNING: cartopy is not installed. Maps will be generated without coastlines/country borders."
        )

    for key, cfg in VAR_CONFIG.items():
        input_name = f"{key}_input_monthly"
        target_name = f"{key}_target_monthly"
        pred_name = f"{key}_pred_monthly"

        inp = collated[input_name].mean(dim=["combo", "time"]).values
        tgt = collated[target_name].mean(dim=["combo", "time"]).values
        pred = collated[pred_name].mean(dim=["combo", "time"]).values

        d_in_tgt = inp - tgt
        d_pred_tgt = pred - tgt
        # Negative values = improvement, positive values = worse
        improvement = np.abs(pred - tgt) - np.abs(inp - tgt)

        vmin, vmax = robust_limits(np.concatenate([inp.ravel(), tgt.ravel(), pred.ravel()]))
        dvmin, dvmax = sym_limits(np.concatenate([d_in_tgt.ravel(), d_pred_tgt.ravel()]))
        ivmin, ivmax = sym_limits(improvement.ravel())

        lat = collated.latitude.values
        lon = collated.longitude.values

        lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
        lat_min, lat_max = float(np.min(lat)), float(np.max(lat))

        if HAS_CARTOPY:
            fig, axes = plt.subplots(
                2,
                3,
                figsize=(20, 8.5),
                constrained_layout=True,
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
        else:
            fig, axes = plt.subplots(2, 3, figsize=(20, 8.5), constrained_layout=True)

        if HAS_CARTOPY:
            p0 = axes[0, 0].pcolormesh(
                lon,
                lat,
                inp,
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
        else:
            p0 = axes[0, 0].pcolormesh(lon, lat, inp, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f"{cfg['label']} Input ({cfg['input']})")

        if HAS_CARTOPY:
            p1 = axes[0, 1].pcolormesh(
                lon,
                lat,
                tgt,
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
        else:
            p1 = axes[0, 1].pcolormesh(lon, lat, tgt, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f"{cfg['label']} Target")

        if HAS_CARTOPY:
            p2 = axes[0, 2].pcolormesh(
                lon,
                lat,
                pred,
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
        else:
            p2 = axes[0, 2].pcolormesh(lon, lat, pred, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, 2].set_title(f"{cfg['label']} Prediction")

        if HAS_CARTOPY:
            p3 = axes[1, 0].pcolormesh(
                lon,
                lat,
                d_in_tgt,
                shading="auto",
                cmap="RdBu_r",
                vmin=dvmin,
                vmax=dvmax,
                transform=ccrs.PlateCarree(),
            )
        else:
            p3 = axes[1, 0].pcolormesh(lon, lat, d_in_tgt, shading="auto", cmap="RdBu_r", vmin=dvmin, vmax=dvmax)
        axes[1, 0].set_title("Input - Target")

        if HAS_CARTOPY:
            p4 = axes[1, 1].pcolormesh(
                lon,
                lat,
                d_pred_tgt,
                shading="auto",
                cmap="RdBu_r",
                vmin=dvmin,
                vmax=dvmax,
                transform=ccrs.PlateCarree(),
            )
        else:
            p4 = axes[1, 1].pcolormesh(lon, lat, d_pred_tgt, shading="auto", cmap="RdBu_r", vmin=dvmin, vmax=dvmax)
        axes[1, 1].set_title("Prediction - Target")

        if HAS_CARTOPY:
            p5 = axes[1, 2].pcolormesh(
                lon,
                lat,
                improvement,
                shading="auto",
                cmap="RdBu_r",
                vmin=ivmin,
                vmax=ivmax,
                transform=ccrs.PlateCarree(),
            )
        else:
            p5 = axes[1, 2].pcolormesh(lon, lat, improvement, shading="auto", cmap="RdBu_r", vmin=ivmin, vmax=ivmax)
        axes[1, 2].set_title("Improvement (blue better, red worse)")

        for ax in axes.ravel():
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            if HAS_CARTOPY:
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                ax.coastlines(resolution="50m", linewidth=0.7)
                ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            else:
                ax.set_aspect("auto")

        cbar1 = fig.colorbar(p2, ax=axes[0, :], shrink=0.85, location="right")
        cbar1.set_label(cfg["units"])

        cbar2 = fig.colorbar(p4, ax=axes[1, :2], shrink=0.85, location="right")
        cbar2.set_label(cfg["units"])

        cbar3 = fig.colorbar(p5, ax=axes[1, 2], shrink=0.85, location="right")
        cbar3.set_label(cfg["units"])

        out_path = map_dir / f"{key}_6panel_map.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def binned_bias_curves(
    target: np.ndarray,
    input_vals: np.ndarray,
    pred_vals: np.ndarray,
    nbins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(target) & np.isfinite(input_vals) & np.isfinite(pred_vals)
    target = target[finite_mask]
    input_vals = input_vals[finite_mask]
    pred_vals = pred_vals[finite_mask]

    if target.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    lo = np.percentile(target, 2.5)
    hi = np.percentile(target, 97.5)
    if np.isclose(lo, hi):
        hi = lo + 1e-6

    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    input_bias = np.full(nbins, np.nan, dtype=np.float64)
    pred_bias = np.full(nbins, np.nan, dtype=np.float64)

    input_err = input_vals - target
    pred_err = pred_vals - target

    for i in range(nbins):
        in_bin = (target >= edges[i]) & (target < edges[i + 1])
        if i == nbins - 1:
            in_bin = (target >= edges[i]) & (target <= edges[i + 1])

        if np.any(in_bin):
            input_bias[i] = np.mean(input_err[in_bin])
            pred_bias[i] = np.mean(pred_err[in_bin])

    improvement = np.abs(input_bias) - np.abs(pred_bias)
    return centers, input_bias, pred_bias, improvement


def create_reliability_plots(
    collated: xr.Dataset,
    output_dir: Path,
    nbins: int = 20,
    dpi: int = 200,
) -> None:
    rel_dir = output_dir / "reliability"
    rel_dir.mkdir(parents=True, exist_ok=True)

    for key, cfg in VAR_CONFIG.items():
        input_name = f"{key}_input_monthly"
        target_name = f"{key}_target_monthly"
        pred_name = f"{key}_pred_monthly"

        target = collated[target_name].values.ravel()
        inp = collated[input_name].values.ravel()
        pred = collated[pred_name].values.ravel()

        x, b_in, b_pred, b_imp = binned_bias_curves(target, inp, pred, nbins)
        if x.size == 0:
            continue

        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1])
        ax_top = fig.add_subplot(gs[:2, 0])
        ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top)

        ax_top.plot(x, b_in, marker="o", label="Input bias", color="tab:orange")
        ax_top.plot(x, b_pred, marker="o", label="Prediction bias", color="tab:blue")
        ax_top.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax_top.set_ylabel("Bias (value - target)")
        ax_top.set_title(f"Reliability-style Bias Curves: {cfg['label']} (95% range)")
        ax_top.grid(alpha=0.3)
        ax_top.legend()

        ax_bot.plot(x, b_imp, marker="o", color="tab:green")
        ax_bot.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax_bot.set_xlabel(f"Target value bins (2.5%-97.5%)")
        ax_bot.set_ylabel("|input| - |pred|")
        ax_bot.set_title("Bias improvement (>0 better)")
        ax_bot.grid(alpha=0.3)

        out_path = rel_dir / f"{key}_reliability_bias.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def create_scatter_plots(collated: xr.Dataset, output_dir: Path, dpi: int = 200) -> None:
    scat_dir = output_dir / "scatter"
    scat_dir.mkdir(parents=True, exist_ok=True)

    for key, cfg in VAR_CONFIG.items():
        input_name = f"{key}_input_monthly"
        target_name = f"{key}_target_monthly"
        pred_name = f"{key}_pred_monthly"

        target = collated[target_name].values.ravel()
        inp = collated[input_name].values.ravel()
        pred = collated[pred_name].values.ravel()

        mask1 = np.isfinite(target) & np.isfinite(inp)
        mask2 = np.isfinite(target) & np.isfinite(pred)

        x1, y1 = target[mask1], inp[mask1]
        x2, y2 = target[mask2], pred[mask2]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        axes[0].scatter(x1, y1, s=3, alpha=0.2, color="tab:orange")
        axes[0].set_title(f"Input vs Target ({cfg['label']})")
        axes[0].set_xlabel("Target")
        axes[0].set_ylabel("Input")
        axes[0].grid(alpha=0.3)

        axes[1].scatter(x2, y2, s=3, alpha=0.2, color="tab:blue")
        axes[1].set_title(f"Prediction vs Target ({cfg['label']})")
        axes[1].set_xlabel("Target")
        axes[1].set_ylabel("Prediction")
        axes[1].grid(alpha=0.3)

        # Precipitation axis limit at 90th percentile to reduce outlier impact
        if key == "tp":
            pooled = np.concatenate([x1, y1, x2, y2])
            pooled = pooled[np.isfinite(pooled)]
            if pooled.size > 0:
                q90 = float(np.percentile(pooled, 90.0))
                lim_min = 0.0
                lim_max = max(q90, 1e-6)
                for ax in axes:
                    ax.set_xlim(lim_min, lim_max)
                    ax.set_ylim(lim_min, lim_max)

        # 1:1 line
        for ax in axes:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            lo = min(xmin, ymin)
            hi = max(xmax, ymax)
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)

        out_path = scat_dir / f"{key}_scatter.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def create_qq_plots(collated: xr.Dataset, output_dir: Path, n_quantiles: int = 500, dpi: int = 200) -> None:
    """Q-Q plots of input-vs-target and prediction-vs-target on one shared axis.

    For each variable a single figure is produced with one panel per variable.
    Both curves share the same x-axis (target quantiles); the input quantiles are
    drawn in orange and the prediction quantiles in blue.  A black dashed 1:1
    reference line marks perfect agreement.

    For precipitation the zero-inflated distribution is handled by computing
    quantiles over *all* values (including zeros) so that the departure of the
    model from the target is visible across the full CDF.
    """
    qq_dir = output_dir / "qq_plots"
    qq_dir.mkdir(parents=True, exist_ok=True)

    probs = np.linspace(0.0, 100.0, n_quantiles)

    for key, cfg in VAR_CONFIG.items():
        input_name = f"{key}_input_monthly"
        target_name = f"{key}_target_monthly"
        pred_name = f"{key}_pred_monthly"

        target = collated[target_name].values.ravel()
        inp = collated[input_name].values.ravel()
        pred = collated[pred_name].values.ravel()

        # Build a common finite mask across all three fields
        mask = np.isfinite(target) & np.isfinite(inp) & np.isfinite(pred)
        if mask.sum() < 10:
            continue

        tgt_q = np.percentile(target[mask], probs)
        inp_q = np.percentile(inp[mask], probs)
        pred_q = np.percentile(pred[mask], probs)

        # Axis limits: span all three distributions
        all_vals = np.concatenate([tgt_q, inp_q, pred_q])
        lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        pad = (hi - lo) * 0.03
        lo -= pad
        hi += pad

        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

        ax.plot(
            tgt_q, inp_q,
            color="tab:orange", linewidth=1.5, label="Input vs Target",
        )
        ax.plot(
            tgt_q, pred_q,
            color="tab:blue", linewidth=1.5, label="Prediction vs Target",
        )
        # 1:1 reference line
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="1:1 (perfect)")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Target quantiles ({cfg['units']})")
        ax.set_ylabel(f"Field quantiles ({cfg['units']})")
        ax.set_title(f"Q-Q plot: {cfg['label']}")
        ax.legend(framealpha=0.85)
        ax.grid(alpha=0.3)

        out_path = qq_dir / f"{key}_qq.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def _stats_for_taylor(target: np.ndarray, field: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(target) & np.isfinite(field)
    t = target[mask]
    f = field[mask]
    if t.size < 2:
        return np.nan, np.nan

    t_std = np.std(t)
    f_std = np.std(f)
    corr = np.corrcoef(t, f)[0, 1]

    if np.isclose(t_std, 0.0):
        return np.nan, np.nan

    std_ratio = f_std / t_std
    corr = np.clip(corr, -1.0, 1.0)
    return std_ratio, corr


def create_taylor_diagram_for_variable(
    collated: xr.Dataset,
    key: str,
    output_path: Path,
    dpi: int = 200,
) -> None:
    cfg = VAR_CONFIG[key]
    input_name = f"{key}_input_monthly"
    target_name = f"{key}_target_monthly"
    pred_name = f"{key}_pred_monthly"

    target = collated[target_name].values.ravel()
    inp = collated[input_name].values.ravel()
    pred = collated[pred_name].values.ravel()

    in_std_ratio, in_corr = _stats_for_taylor(target, inp)
    pr_std_ratio, pr_corr = _stats_for_taylor(target, pred)

    if not (np.isfinite(in_std_ratio) and np.isfinite(in_corr) and np.isfinite(pr_std_ratio) and np.isfinite(pr_corr)):
        return

    theta_input = np.arccos(in_corr)
    theta_pred = np.arccos(pr_corr)

    r_max = max(1.6, in_std_ratio, pr_std_ratio) * 1.15

    fig = plt.figure(figsize=(7, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")

    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rlim(0, r_max)

    # Correlation grid labels
    corr_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    theta_ticks = np.arccos(np.clip(corr_ticks, -1, 1))
    ax.set_thetagrids(np.degrees(theta_ticks), labels=[f"{c:.2f}" for c in corr_ticks])
    ax.set_title(f"Taylor Diagram: {cfg['label']}\n(angle=correlation, radius=std ratio)")

    # Reference point: target => std ratio=1 at corr=1 (theta=0)
    ax.plot(0, 1.0, "k*", markersize=12, label="Target (ref)")

    # Input and prediction points
    ax.plot(theta_input, in_std_ratio, "o", color="tab:orange", markersize=9, label="Input")
    ax.plot(theta_pred, pr_std_ratio, "o", color="tab:blue", markersize=9, label="Prediction")

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def create_taylor_diagrams(collated: xr.Dataset, output_dir: Path, dpi: int = 200) -> None:
    taylor_dir = output_dir / "taylor"
    taylor_dir.mkdir(parents=True, exist_ok=True)

    for key in VAR_CONFIG:
        out_path = taylor_dir / f"{key}_taylor.png"
        create_taylor_diagram_for_variable(collated, key, out_path, dpi=dpi)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (input_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No inference NetCDF files found in {input_dir} with pattern '{args.pattern}'"
        )

    print(f"Found {len(files)} files")

    collated = collate_monthly(files)
    collated_path = output_dir / "monthly_collated.nc"
    save_collated(collated, collated_path)
    print(f"Saved collated monthly NetCDF: {collated_path}")

    create_six_panel_maps(collated, output_dir=output_dir, dpi=args.dpi)
    print(f"Saved six-panel maps in: {output_dir / 'maps_6panel'}")

    create_reliability_plots(
        collated,
        output_dir=output_dir,
        nbins=args.reliability_bins,
        dpi=args.dpi,
    )
    print(f"Saved reliability plots in: {output_dir / 'reliability'}")

    create_scatter_plots(collated, output_dir=output_dir, dpi=args.dpi)
    print(f"Saved scatter plots in: {output_dir / 'scatter'}")

    create_taylor_diagrams(collated, output_dir=output_dir, dpi=args.dpi)
    print(f"Saved Taylor diagrams in: {output_dir / 'taylor'}")

    create_qq_plots(collated, output_dir=output_dir, dpi=args.dpi)
    print(f"Saved Q-Q plots in: {output_dir / 'qq_plots'}")

if __name__ == "__main__":
    main()
