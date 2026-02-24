"""
Data loader for seasonal climate prediction bias correction.

Mirrors DecadalDataLoader but adapted for the seasonal NetCDF structure:

  Dimensions : time, number (ensemble member), latitude, longitude, cci_class
  Forecast variables (time, number, lat, lon) :
      t2m, hurs, tmax, tp*
  Static / time-only variables (time, lat, lon) :
      tasERA, tasmaxERA, tpERA, rhERA (targets)
      dem, rho, phi, sin_time, cos_time
  Ancillary (time, cci_class, lat, lon) :
      cci_agg
  Lead variable (time, number, lat, lon) :
      lead_time  [timedelta64[ns]] – converted to integer months 0-6

  * tp is stored with the WRONG dimension order (number, lat, lon, time) in the
    source NetCDF.  It is transposed to (time, number, lat, lon) at open time.

Valid combinations are (number_idx, time_idx) pairs – analogous to the
(run_idx, lead_idx, time_idx) triplets in DecadalDataLoader.

The public interface (scalers, normalize, denormalize, __len__, __getitem__,
get_combination_info, close, context-manager) is identical to DecadalDataLoader
so the rest of the pipeline (ClimateDatasetSeasonal, Trainer, Evaluator …)
can reuse as much existing code as possible.

Output channel layout (matches decadal):
  0  : tp    (precipitation forecast)
  1  : t2m   (2-m temperature)
  2  : tmax  (max temperature)
  3  : hurs  (relative humidity)
  4  : dem   (static)
  5  : rho   (static)
  6  : phi   (static)
  7  : sin_time
  8  : cos_time
  9-18 : cci_agg (10 classes)
Total: 19 channels  →  3 static + 16 dynamic (same split as decadal)
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Average days per month used for timedelta → integer month conversion.
_DAYS_PER_MONTH = 30.4375  # 365.25 / 12


def _td_to_months(td_ns: int) -> int:
    """Convert a timedelta64[ns] integer to the nearest integer month (0-6)."""
    days = td_ns / 1e9 / 86400
    return int(round(days / _DAYS_PER_MONTH))


class SeasonalDataLoader:
    """
    Data loader for seasonal climate forecasting datasets.

    Handles loading and preprocessing of climate data with ensemble members
    (number) and time dimensions, automatically detecting and filtering out
    invalid (number, time) combinations that contain NaN values.

    Supports full in-memory loading for maximum performance.
    """

    def __init__(
        self,
        nc_path: str,
        normalize_method: str = "minmax",
        cache_dir: str = ".",
        force_recompute: bool = False,
        load_in_memory: bool = True,
    ):
        """
        Initialise SeasonalDataLoader.

        Parameters
        ----------
        nc_path : str
            Path to the seasonal NetCDF file.
        normalize_method : str
            'minmax' or 'zscore'.
        cache_dir : str
            Directory where the valid-combinations CSV cache is stored.
        force_recompute : bool
            Re-scan for valid combinations even if a cache exists.
        load_in_memory : bool
            Load the entire dataset into RAM for maximum throughput.
        """
        self._closed = True
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("INITIALIZING SEASONAL DATA LOADER")
        logger.info("=" * 70)

        # ── validate inputs ───────────────────────────────────────────────────
        if normalize_method not in ("minmax", "zscore"):
            raise ValueError(
                f"normalize_method must be 'minmax' or 'zscore', got '{normalize_method}'"
            )

        nc_path = Path(nc_path)
        if not nc_path.exists():
            raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

        file_size_mb = nc_path.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ File: {nc_path}  ({file_size_mb:.1f} MB)")

        self.nc_path = str(nc_path)
        self.normalize_method = normalize_method
        self.scalers: Dict = {}
        self.cache_dir = Path(cache_dir)
        self.force_recompute = force_recompute
        self.load_in_memory = load_in_memory
        self._closed = False

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ── variable structure ────────────────────────────────────────────────
        # Channel order presented to the model (must match decadal layout).
        # Internally we store the raw NC variable names.
        self.fc_vars = ["tp", "t2m", "tmax", "hurs"]   # (time, number, lat, lon)
        self.static_vars = ["dem", "rho", "phi"]        # (lat, lon) or (time, lat, lon)
        self.time_only_vars = ["sin_time", "cos_time"]  # (time, lat, lon)
        self.cci_var = "cci_agg"                        # (time, cci_class, lat, lon)
        self.target_vars = ["tasERA", "tasmaxERA", "tpERA", "rhERA"]

        # All channel names in order (used externally / by the dataset wrapper)
        self.input_vars = self.fc_vars + self.static_vars + self.time_only_vars + ["cci_agg"]

        logger.info(f"  ✓ Forecast vars : {self.fc_vars}")
        logger.info(f"  ✓ Static vars   : {self.static_vars}")
        logger.info(f"  ✓ Target vars   : {self.target_vars}")

        # ── load dataset ──────────────────────────────────────────────────────
        logger.info("Loading NetCDF …")
        try:
            self.ds = xr.open_dataset(self.nc_path)
        except Exception as e:
            raise IOError(f"Failed to open {self.nc_path}: {e}")

        # Fix tp dimension order: (number, lat, lon, time) → (time, number, lat, lon)
        if "tp" in self.ds.data_vars:
            tp_dims = self.ds["tp"].dims
            if tp_dims != ("time", "number", "latitude", "longitude"):
                logger.info(
                    f"  Transposing 'tp' from {tp_dims} "
                    f"→ ('time','number','latitude','longitude')"
                )
                self.ds["tp"] = self.ds["tp"].transpose(
                    "time", "number", "latitude", "longitude"
                )

        self._validate_dataset()

        # ── optional in-memory load ───────────────────────────────────────────
        if load_in_memory:
            logger.info("Loading dataset into RAM …  (this may take a while)")
            mem_start = time.time()
            try:
                self.ds = self.ds.load()
                total_gb = sum(
                    v.nbytes for v in self.ds.data_vars.values() if hasattr(v, "nbytes")
                ) / 1024 ** 3
                logger.info(
                    f"  ✓ Loaded in {time.time()-mem_start:.1f}s  "
                    f"({total_gb:.2f} GB)"
                )
            except MemoryError:
                raise MemoryError(
                    "Insufficient RAM to load dataset.  "
                    "Try load_in_memory=False or use a larger machine."
                )
        else:
            logger.info("  Using lazy (disk-based) loading.")

        # tp and tpERA are already on the same scale in this dataset
        # (both ~0.0012, daily values in m/day).  No unit conversion needed.
        logger.info("  'tp' and 'tpERA' share the same unit scale — no conversion applied.")

        # ── precompute lead months ────────────────────────────────────────────
        # lead_time has shape (time, number, lat, lon); average over lat/lon first.
        logger.info("Pre-computing lead months (mean over spatial dims) …")
        lead_mean = self.ds["lead_time"].mean(dim=["latitude", "longitude"])
        # lead_mean: (time, number)  [timedelta64[ns] → float64 ns]
        lead_mean_ns = lead_mean.values.astype("timedelta64[ns]").view(np.int64)
        # Vectorised conversion to integer months
        self._lead_months = np.round(
            lead_mean_ns / 1e9 / 86400 / _DAYS_PER_MONTH
        ).astype(int)  # shape (n_time, n_number)
        logger.info(
            f"  Lead month range: {self._lead_months.min()} – {self._lead_months.max()}"
        )

        # ── valid combinations ────────────────────────────────────────────────
        logger.info("Loading / computing valid combinations …")
        valid_start = time.time()
        self.valid_combinations = self._load_or_compute_valid_combinations()
        logger.info(
            f"  ✓ {len(self.valid_combinations):,} valid samples  "
            f"({time.time()-valid_start:.1f}s)"
        )

        logger.info("=" * 70)
        logger.info(
            f"SEASONAL LOADER READY  –  {len(self.valid_combinations):,} samples  "
            f"(total {time.time()-start_time:.1f}s)"
        )
        logger.info("=" * 70)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _validate_dataset(self):
        required_dims = ["time", "number", "latitude", "longitude", "cci_class"]
        missing = [d for d in required_dims if d not in self.ds.dims]
        if missing:
            raise KeyError(f"Missing dimensions: {missing}")

        all_vars = self.fc_vars + self.static_vars + self.time_only_vars + \
                   [self.cci_var, "lead_time"] + self.target_vars
        missing_vars = [v for v in all_vars if v not in self.ds.variables]
        if missing_vars:
            raise KeyError(f"Missing variables: {missing_vars}")

        for d in required_dims:
            logger.info(f"    • {d}: {len(self.ds[d])}")
        logger.info(f"  ✓ All required variables present.")

    def _get_cache_filename(self) -> Path:
        try:
            file_size = Path(self.nc_path).stat().st_size
            h = hashlib.md5(f"{self.nc_path}_{file_size}".encode()).hexdigest()[:8]
            return self.cache_dir / f"{Path(self.nc_path).stem}_valid_combinations_{h}.csv"
        except Exception:
            return self.cache_dir / f"{Path(self.nc_path).stem}_valid_combinations.csv"

    def _load_or_compute_valid_combinations(self) -> pd.DataFrame:
        cache_file = self._get_cache_filename()
        required_cols = ["number_idx", "time_idx", "number", "lead_month"]

        if cache_file.exists() and not self.force_recompute:
            try:
                df = pd.read_csv(cache_file)
                if all(c in df.columns for c in required_cols) and len(df) > 0:
                    logger.info(f"  Loaded {len(df):,} combinations from cache.")
                    return df
            except Exception as e:
                logger.warning(f"  Cache load failed ({e}); recomputing …")

        n_number = len(self.ds["number"])
        n_time = len(self.ds["time"])
        logger.info(
            f"  Computing valid combinations  "
            f"({n_number} members × {n_time} times = {n_number*n_time:,} total) …"
        )

        valid_mask = np.ones((n_number, n_time), dtype=bool)

        # ---- forecast variables (time, number, lat, lon) --------------------
        for var in self.fc_vars:
            if var not in self.ds.data_vars:
                continue
            v_start = time.time()
            has_nan = (
                self.ds[var]
                .isnull()
                .any(dim=["latitude", "longitude"])
            )
            # Ensure shape is (time, number); transpose to (number, time)
            if has_nan.dims == ("time", "number"):
                has_nan_np = has_nan.values.T          # (number, time)
            elif has_nan.dims == ("number", "time"):
                has_nan_np = has_nan.values
            else:
                has_nan = has_nan.transpose("number", "time")
                has_nan_np = has_nan.values
            n_inv = has_nan_np.sum()
            valid_mask &= ~has_nan_np
            logger.info(
                f"    {var}: {n_inv:,} invalid  ({time.time()-v_start:.1f}s)"
            )

        # ---- static / time-only variables -----------------------------------
        for var in self.static_vars + self.time_only_vars:
            if var not in self.ds.data_vars:
                continue
            if "time" in self.ds[var].dims:
                has_nan = (
                    self.ds[var]
                    .isnull()
                    .any(dim=["latitude", "longitude"])
                    .values
                )   # (n_time,)
                # Broadcast to (n_number, n_time)
                has_nan_bc = np.broadcast_to(
                    has_nan[np.newaxis, :], (n_number, n_time)
                )
                valid_mask &= ~has_nan_bc

        # ---- cci_agg --------------------------------------------------------
        if self.cci_var in self.ds.data_vars:
            has_nan = (
                self.ds[self.cci_var]
                .isnull()
                .any(dim=["cci_class", "latitude", "longitude"])
                .values
            )   # (n_time,)
            has_nan_bc = np.broadcast_to(
                has_nan[np.newaxis, :], (n_number, n_time)
            )
            valid_mask &= ~has_nan_bc

        # ---- target variables -----------------------------------------------
        for var in self.target_vars:
            if var not in self.ds.data_vars:
                continue
            has_nan = (
                self.ds[var]
                .isnull()
                .any(dim=["latitude", "longitude"])
                .values
            )   # (n_time,)
            has_nan_bc = np.broadcast_to(
                has_nan[np.newaxis, :], (n_number, n_time)
            )
            valid_mask &= ~has_nan_bc

        n_valid = valid_mask.sum()
        if n_valid == 0:
            raise ValueError("No valid (number, time) combinations found.")

        logger.info(
            f"  {n_valid:,} valid  ({100*n_valid/(n_number*n_time):.1f}%)"
        )

        number_indices, time_indices = np.where(valid_mask)

        # Retrieve lead months from the precomputed matrix
        lead_months = self._lead_months[time_indices, number_indices]
        number_values = [int(self.ds.number.values[i]) for i in number_indices]

        df = pd.DataFrame(
            {
                "number_idx": number_indices.astype(int),
                "time_idx": time_indices.astype(int),
                "number": number_values,
                "lead_month": lead_months.astype(int),
            }
        )

        try:
            df.to_csv(cache_file, index=False)
            logger.info(f"  Cache saved: {cache_file.name}")
        except Exception as e:
            logger.warning(f"  Could not save cache: {e}")

        return df

    # ── normalization interface ───────────────────────────────────────────────

    def normalize(
        self, data: np.ndarray, var_name: str, fit: bool = True
    ) -> np.ndarray:
        """
        Normalize *data* using the stored scaler for *var_name*.

        log1p transform is applied first to precipitation variables
        (tpERA and tp) before the min-max / z-score step.
        """
        if not fit and var_name not in self.scalers:
            raise ValueError(
                f"No scaler for '{var_name}'.  Call with fit=True first."
            )

        if var_name in ("tpERA", "tp"):
            data = np.log1p(np.maximum(data, 0))

        if self.normalize_method == "minmax":
            if fit:
                vmin = float(np.nanmin(data))
                vmax = float(np.nanmax(data))
                if abs(vmax - vmin) < 1e-8:
                    vmax = vmin + 1.0
                self.scalers[var_name] = {"min": vmin, "max": vmax}
            else:
                s = self.scalers[var_name]
                vmin = float(s["min"] if isinstance(s, dict) else s[0])
                vmax = float(s["max"] if isinstance(s, dict) else s[1])
            return (data - vmin) / (vmax - vmin + 1e-8)

        else:  # zscore
            if fit:
                vmean = float(np.nanmean(data))
                vstd = float(np.nanstd(data))
                if vstd < 1e-8:
                    vstd = 1.0
                self.scalers[var_name] = {"mean": vmean, "std": vstd}
            else:
                s = self.scalers[var_name]
                vmean = float(s["mean"] if isinstance(s, dict) else s[0])
                vstd = float(s["std"] if isinstance(s, dict) else s[1])
            return (data - vmean) / (vstd + 1e-8)

    def denormalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Reverse normalization (and log1p for precipitation)."""
        if var_name not in self.scalers:
            raise ValueError(f"No scaler for '{var_name}'.")

        s = self.scalers[var_name]
        if self.normalize_method == "minmax":
            vmin = float(s["min"] if isinstance(s, dict) else s[0])
            vmax = float(s["max"] if isinstance(s, dict) else s[1])
            out = data * (vmax - vmin) + vmin
        else:
            vmean = float(s["mean"] if isinstance(s, dict) else s[0])
            vstd = float(s["std"] if isinstance(s, dict) else s[1])
            out = data * vstd + vmean

        if var_name == "tpERA":
            out = np.expm1(out)

        return out

    # ── dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.valid_combinations)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (input_data, target_data) for sample *idx*.

        input_data  : (19, H*W) float32
        target_data : (4,  H*W) float32

        Channel layout:
          0  tp,  1  t2m,  2  tmax,  3  hurs,
          4  dem, 5  rho,  6  phi,
          7  sin_time, 8  cos_time,
          9-18  cci_agg (10 classes)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        combo = self.valid_combinations.iloc[idx]
        number_idx = int(combo["number_idx"])
        time_idx = int(combo["time_idx"])

        inputs = []

        # ── forecast channels 0-3 ────────────────────────────────────────────
        for var in self.fc_vars:   # tp, t2m, tmax, hurs
            data = (
                self.ds[var]
                .isel(time=time_idx, number=number_idx)
                .values
                .flatten()
            )
            inputs.append(data)

        # ── static channels 4-6 ──────────────────────────────────────────────
        for var in self.static_vars:   # dem, rho, phi
            var_data = self.ds[var]
            if "time" in var_data.dims:
                data = var_data.isel(time=time_idx).values.flatten()
            else:
                data = var_data.values.flatten()
            inputs.append(data)

        # ── time-only channels 7-8 ───────────────────────────────────────────
        for var in self.time_only_vars:   # sin_time, cos_time
            data = self.ds[var].isel(time=time_idx).values.flatten()
            inputs.append(data)

        # ── cci_agg channels 9-18 ────────────────────────────────────────────
        cci_data = (
            self.ds[self.cci_var]
            .isel(time=time_idx)
            .values   # (10, H, W)
        )
        n_cls, H, W = cci_data.shape
        cci_flat = cci_data.reshape(n_cls, H * W)
        for c in range(n_cls):
            inputs.append(cci_flat[c])

        inputs_np = np.stack(inputs, axis=0).astype(np.float32)  # (19, H*W)

        # ── target variables ─────────────────────────────────────────────────
        targets = []
        for var in self.target_vars:
            data = self.ds[var].isel(time=time_idx).values.flatten()
            targets.append(data)

        targets_np = np.stack(targets, axis=0).astype(np.float32)  # (4, H*W)

        return inputs_np, targets_np

    def get_combination_info(self, idx: int) -> Dict:
        """
        Return metadata for sample *idx*.

        Keys:
            number      : ensemble member integer (0-24)
            number_idx  : positional index of member
            time_idx    : positional index in time dim
            lead_month  : integer forecast lead time in months (0-6)
            time        : numpy datetime64 value
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        combo = self.valid_combinations.iloc[idx]
        return {
            "number":     int(combo["number"]),
            "number_idx": int(combo["number_idx"]),
            "time_idx":   int(combo["time_idx"]),
            "lead_month": int(combo["lead_month"]),
            "time":       self.ds.time.values[int(combo["time_idx"])],
        }

    def get_indices_by_member_lead(
        self,
        number: Optional[int] = None,
        lead_month: Optional[int] = None,
    ) -> List[int]:
        """Return sample indices matching optional *number* and/or *lead_month*."""
        mask = pd.Series([True] * len(self.valid_combinations))
        if number is not None:
            mask &= self.valid_combinations["number"] == number
        if lead_month is not None:
            mask &= self.valid_combinations["lead_month"] == lead_month
        return self.valid_combinations[mask].index.tolist()

    # ── memory / lifecycle ────────────────────────────────────────────────────

    def get_summary_stats(self) -> Dict:
        return {
            "total_valid_samples": len(self),
            "n_members":    len(self.ds["number"]),
            "n_times":      len(self.ds["time"]),
            "members":      list(self.ds["number"].values),
            "lead_months":  sorted(self.valid_combinations["lead_month"].unique().tolist()),
            "samples_per_member": (
                self.valid_combinations.groupby("number").size().to_dict()
            ),
            "samples_per_lead_month": (
                self.valid_combinations.groupby("lead_month").size().to_dict()
            ),
            "in_memory": self.load_in_memory,
        }

    def get_memory_usage(self) -> Dict:
        """Return estimated memory usage of the in-memory dataset."""
        if not self.load_in_memory:
            return {"status": "not_in_memory", "usage_gb": 0.0}

        total_bytes = 0
        var_usage = {}
        for var_name in list(self.ds.data_vars):
            var = self.ds[var_name]
            if hasattr(var, "nbytes"):
                var_bytes = var.nbytes
                total_bytes += var_bytes
                var_usage[var_name] = var_bytes / (1024 ** 3)

        return {
            "status": "in_memory",
            "total_gb": total_bytes / (1024 ** 3),
            "variables": var_usage,
        }

    def close(self):
        if not self._closed and hasattr(self, "ds"):
            try:
                self.ds.close()
                self._closed = True
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False

    def __del__(self):
        self.close()
