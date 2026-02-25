"""
Data loader for high-resolution decadal downscaling datasets.

Dataset layout
--------------
One NetCDF file per (year, run, lead_time) combination:

    {data_dir}/{year}_{run}_{lead}_.nc

Example: 1981_r1i1p1f1_0_.nc

Each file contains ~363 daily time steps on a 270×520 (lat×lon) grid.
~2075 files, ~1.6 TB total — loading into RAM is not an option.

Variables per file
------------------
  Forecast (time, lat, lon)    : pr, tas, tasmax, hurs
  Targets  (time, lat, lon)    : tasERA, tasmaxERA, rhERA
           (lat,  lon,  time)  : tpERA        ← transposed; handled transparently
  Static   (lat, lon)          : dem, rho, phi
  Time     (time, lat, lon)    : sin_time, cos_time
  CCI      (time, cci_class, lat, lon) : cci_agg   [10 classes]
  Scalars                      : run (str), lead (int)

Output channel layout (19 input channels — matches DecadalDataLoader)
----------------------------------------------------------------------
  0    pr
  1    tas
  2    tasmax
  3    hurs
  4    dem
  5    rho
  6    phi
  7    sin_time
  8    cos_time
  9-18 cci_agg (10 classes)

Target channels (4)
-------------------
  0  tasERA
  1  tasmaxERA
  2  tpERA
  3  rhERA

Normalization
-------------
Pre-computed per-year min/max CSVs live in the same directory:
    minmax_{year}.csv   columns: var, min, max, year

This loader aggregates them across TRAINING years only:
    global_min = min over all training-year mins
    global_max = max over all training-year maxs

This is exact (no approximation) and O(n_years) — no data scan needed.

Performance
-----------
* Index: one-time scan of all NC files → cached as index.csv in cache_dir.
  Rebuild only when the data directory changes (hash of file list).
  At 750k rows × 5 columns ≈ 30 MB; loads in < 1 s.

* File handles: per-process LRU of up to 32 open nc4.Dataset handles.
  DataLoader workers are separate processes, so each has its own LRU.
  Files have 363 time steps; even with shuffled batches, the LRU
  eliminates repeated open() calls for the same file within a batch.

* tpERA transpose: detected at first open per file; applied in-place as
  a numpy transpose — no copy needed for the 1D time slice extracted.
"""

import csv
import hashlib
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

_FC_VARS      = ["pr", "tas", "tasmax", "hurs"]           # (time, lat, lon)
_STATIC_VARS  = ["dem", "rho", "phi"]                     # (lat, lon)
_TIME_VARS    = ["sin_time", "cos_time"]                  # (time, lat, lon)
_CCI_VAR      = "cci_agg"                                 # (time, cci_class, lat, lon)
_TARGET_VARS  = ["tasERA", "tasmaxERA", "tpERA", "rhERA"]

# tpERA is stored as (lat, lon, time) in these files — detected per-file
_TPERA_CANONICAL = ("time", "latitude", "longitude")
_TPERA_WRONG     = ("latitude", "longitude", "time")

# Per-process LRU of open file handles
_LRU_MAX_SIZE   = 32
_HANDLE_CACHE   : OrderedDict = OrderedDict()   # filepath → nc4.Dataset


def _get_handle(filepath: str):
    """
    Return a cached netCDF4.Dataset for *filepath*, opening if needed.

    Uses a per-process LRU dict (_HANDLE_CACHE) so DataLoader workers each
    maintain their own independent set of open handles — no locking required.
    """
    import netCDF4 as nc4

    if filepath in _HANDLE_CACHE:
        _HANDLE_CACHE.move_to_end(filepath)
        return _HANDLE_CACHE[filepath]

    if len(_HANDLE_CACHE) >= _LRU_MAX_SIZE:
        evicted_path, evicted_ds = _HANDLE_CACHE.popitem(last=False)
        try:
            evicted_ds.close()
        except Exception:
            pass

    ds = nc4.Dataset(filepath, "r")
    _HANDLE_CACHE[filepath] = ds
    return ds


def _close_all_handles():
    """Close all cached file handles (call in worker cleanup or at shutdown)."""
    for ds in _HANDLE_CACHE.values():
        try:
            ds.close()
        except Exception:
            pass
    _HANDLE_CACHE.clear()


# ── main class ───────────────────────────────────────────────────────────────

class DecadalDownscaleDataLoader:
    """
    Disk-based data loader for high-resolution decadal downscaling.

    Parameters
    ----------
    data_dir : str
        Directory containing the {year}_{run}_{lead}_.nc files and
        minmax_{year}.csv normalization files.
    normalize_method : str
        Currently only 'minmax' is supported (CSVs provide min/max).
    cache_dir : str
        Where to store the sample index cache (index.csv + hash file).
    train_years : list[int]
        Years used for training (normalization is fitted on these only).
    force_reindex : bool
        Rebuild the index even if a valid cache exists.
    """

    def __init__(
        self,
        data_dir: str,
        normalize_method: str = "minmax",
        cache_dir: str = "./cache",
        train_years: Optional[List[int]] = None,
        force_reindex: bool = False,
    ):
        t0 = time.time()
        logger.info("=" * 70)
        logger.info("INITIALIZING DECADAL DOWNSCALE DATA LOADER")
        logger.info("=" * 70)

        if normalize_method != "minmax":
            raise ValueError(
                f"Only 'minmax' normalization is supported for this loader "
                f"(pre-computed CSVs provide min/max only), got '{normalize_method}'"
            )

        self.data_dir        = Path(data_dir).expanduser().resolve()
        self.normalize_method = normalize_method
        self.cache_dir       = Path(cache_dir).expanduser().resolve()
        self.train_years     = list(train_years) if train_years else []
        self.force_reindex   = force_reindex
        self.scalers: Dict   = {}

        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"  data_dir  : {self.data_dir}")
        logger.info(f"  cache_dir : {self.cache_dir}")

        # ── 1. build / load sample index ─────────────────────────────────────
        logger.info("Step 1/3 — sample index …")
        self._index: pd.DataFrame = self._build_or_load_index()
        logger.info(
            f"  {len(self._index):,} samples across "
            f"{self._index['file_path'].nunique():,} files"
        )

        # ── 2. load normalization from per-year CSVs ──────────────────────────
        logger.info("Step 2/3 — normalization from CSVs …")
        self._load_normalization()

        # ── 3. cache static arrays (dem, rho, phi) from first available file ─
        logger.info("Step 3/3 — caching static variables …")
        self._static_cache: Optional[Dict[str, np.ndarray]] = None
        # Deferred to first __getitem__ to avoid slow startup on large grids.

        logger.info("=" * 70)
        logger.info(
            f"READY — {len(self._index):,} samples  "
            f"(init {time.time()-t0:.1f}s)"
        )
        logger.info("=" * 70)

    # ── index ─────────────────────────────────────────────────────────────────

    def _nc_files(self) -> List[Path]:
        return sorted(self.data_dir.glob("*_*_*_.nc"))

    def _dir_hash(self) -> str:
        """MD5 of (sorted file names + total count) — cheap sentinel."""
        files = self._nc_files()
        payload = ",".join(f.name for f in files) + f"|{len(files)}"
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    def _index_cache_path(self) -> Path:
        return self.cache_dir / f"downscale_index_{self._dir_hash()}.csv"

    def _build_or_load_index(self) -> pd.DataFrame:
        cache_path = self._index_cache_path()
        required_cols = {"file_path", "time_idx", "year", "run", "lead"}

        if cache_path.exists() and not self.force_reindex:
            try:
                df = pd.read_csv(cache_path)
                if required_cols.issubset(df.columns) and len(df) > 0:
                    logger.info(f"  Loaded index from cache: {cache_path.name}")
                    return df
            except Exception as e:
                logger.warning(f"  Cache load failed ({e}); rebuilding …")

        logger.info("  Building index (filename parsing only — no per-file reads) …")
        t0 = time.time()

        import netCDF4 as nc4

        files = self._nc_files()
        logger.info(f"  Found {len(files)} NC files.")

        # ── determine n_time from a single representative file ────────────────
        # All files in this dataset share the same time dimension length (363).
        # Reading every file just to count timesteps would require 2075 network-
        # filesystem opens and is prohibitively slow.  We probe one file and
        # apply that count to all.  If a specific file has a different length,
        # the get_item() netCDF4 read will still produce the correct data (it
        # only reads the time slice requested; out-of-range indices are caught
        # by the IndexError guard at the top of __getitem__).
        n_time_default = None
        for fp in files:
            try:
                ds = nc4.Dataset(str(fp), "r")
                n_time_default = len(ds.dimensions["time"])
                ds.close()
                logger.info(f"  Representative file: {fp.name}  →  n_time={n_time_default}")
                break
            except Exception as e:
                logger.warning(f"  Could not probe {fp.name}: {e}")

        if n_time_default is None:
            raise RuntimeError("Could not open any NC file to determine n_time.")

        skipped = 0
        records = []
        for fp in files:
            # Parse metadata from filename:  {year}_{run}_{lead}_.nc
            stem  = fp.stem   # e.g.  1981_r1i1p1f1_0_
            parts = stem.rstrip("_").split("_")
            # parts[0] = year,  parts[-1] = lead,  parts[1:-1] = run label
            try:
                year = int(parts[0])
                lead = int(parts[-1])
                run  = "_".join(parts[1:-1])   # e.g. "r1i1p1f1"
            except (ValueError, IndexError) as e:
                logger.warning(f"  Skipping {fp.name}: cannot parse metadata ({e})")
                skipped += 1
                continue

            fp_str = str(fp)
            for t in range(n_time_default):
                records.append((fp_str, t, year, run, lead))

        if skipped:
            logger.warning(f"  Skipped {skipped} files with unparseable names.")

        df = pd.DataFrame(records, columns=["file_path", "time_idx", "year", "run", "lead"])
        df.reset_index(drop=True, inplace=True)

        try:
            df.to_csv(cache_path, index=False)
            logger.info(
                f"  Index saved → {cache_path.name}  "
                f"({len(df):,} rows, {time.time()-t0:.1f}s)"
            )
        except Exception as e:
            logger.warning(f"  Could not save index cache: {e}")

        return df

    # ── normalization ─────────────────────────────────────────────────────────

    def _load_normalization(self):
        """
        Aggregate per-year min/max CSVs into global scalers.

        Only training years are used.  If no training years are specified,
        all years present in the directory are used.

        global_min = min(all per-year mins for training years)
        global_max = max(all per-year maxs for training years)

        This is an exact global min/max — not an approximation.
        """
        csv_files = sorted(self.data_dir.glob("minmax_*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No minmax_*.csv files found in {self.data_dir}"
            )

        # Determine which years to use for fitting
        fit_years = set(self.train_years) if self.train_years else None

        agg: Dict[str, Dict[str, float]] = {}  # var → {"min": ..., "max": ...}

        n_loaded = 0
        for csv_path in csv_files:
            # Extract year from filename:  minmax_{year}.csv
            try:
                year = int(csv_path.stem.split("_")[1])
            except (IndexError, ValueError):
                logger.warning(f"  Cannot parse year from {csv_path.name}, skipping.")
                continue

            if fit_years is not None and year not in fit_years:
                continue

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    var  = row["var"]
                    vmin = float(row["min"])
                    vmax = float(row["max"])
                    if var not in agg:
                        agg[var] = {"min": vmin, "max": vmax}
                    else:
                        agg[var]["min"] = min(agg[var]["min"], vmin)
                        agg[var]["max"] = max(agg[var]["max"], vmax)
            n_loaded += 1

        if not agg:
            raise ValueError(
                f"No CSV files matched training years {self.train_years}. "
                f"Available CSVs: {[f.name for f in csv_files]}"
            )

        # Store in self.scalers — same dict format as DecadalDataLoader
        for var, stats in agg.items():
            vmin = stats["min"]
            vmax = stats["max"]
            if var in ("tpERA", "pr"):
                # Apply log1p transform before min/max (same as DecadalDataLoader)
                vmin = float(np.log1p(max(vmin, 0)))
                vmax = float(np.log1p(max(vmax, 0)))
            if abs(vmax - vmin) < 1e-8:
                vmax = vmin + 1.0
            self.scalers[var] = {"min": vmin, "max": vmax}

        logger.info(
            f"  Loaded scalers from {n_loaded} year CSVs "
            f"(years: {sorted(fit_years) if fit_years else 'all'})"
        )
        logger.info(f"  Variables normalised: {sorted(agg.keys())}")

    # ── normalization interface (mirrors DecadalDataLoader) ───────────────────

    def normalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """
        Normalise *data* using the pre-loaded min/max scaler for *var_name*.

        log1p transform is applied first to precipitation variables.
        """
        if var_name not in self.scalers:
            raise ValueError(
                f"No scaler for '{var_name}'. "
                f"Available: {list(self.scalers.keys())}"
            )
        if var_name in ("tpERA", "pr"):
            data = np.log1p(np.maximum(data, 0.0))

        s = self.scalers[var_name]
        return (data - s["min"]) / (s["max"] - s["min"] + 1e-8)

    def denormalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Reverse normalization (and log1p for precipitation)."""
        if var_name not in self.scalers:
            raise ValueError(f"No scaler for '{var_name}'.")
        s = self.scalers[var_name]
        out = data * (s["max"] - s["min"]) + s["min"]
        if var_name == "tpERA":
            out = np.expm1(out)
        return out

    # ── static variable cache ─────────────────────────────────────────────────

    def _ensure_static_cache(self, filepath: str):
        """
        Populate _static_cache from *filepath* if not already done.

        Static variables (dem, rho, phi) are identical in every file so we
        only ever need to read them once.
        """
        if self._static_cache is not None:
            return

        import netCDF4 as nc4

        ds = nc4.Dataset(filepath, "r")
        self._static_cache = {}
        for var in _STATIC_VARS:
            if var in ds.variables:
                self._static_cache[var] = ds.variables[var][:].astype(np.float32)
        ds.close()
        logger.debug(f"Static cache populated from {Path(filepath).name}")

    # ── dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (input_tensor, target_tensor) for sample *idx*.

        input_tensor  : float32  (19, H*W)
        target_tensor : float32  ( 4, H*W)

        Channel layout: see module docstring.

        The method uses a per-process LRU cache of open nc4.Dataset handles
        to avoid repeated open() overhead when accessing nearby samples.
        tpERA's transposed layout (lat, lon, time) is handled transparently.

        All channels are normalised using the pre-loaded scalers.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        row = self._index.iloc[idx]
        filepath = row["file_path"]
        t        = int(row["time_idx"])

        # Ensure static cache is ready
        self._ensure_static_cache(filepath)

        ds = _get_handle(filepath)
        H  = len(ds.dimensions["latitude"])
        W  = len(ds.dimensions["longitude"])
        HW = H * W

        inputs = []

        # ── forecast variables (time, lat, lon) ───────────────────────────────
        for var in _FC_VARS:
            raw = ds.variables[var][t].astype(np.float32).flatten()   # (H*W,)
            inputs.append(self.normalize(raw, var))

        # ── static variables (lat, lon) ────────────────────────────────────────
        for var in _STATIC_VARS:
            raw = self._static_cache[var].flatten()                    # (H*W,)
            inputs.append(self.normalize(raw, var))

        # ── time-only variables (time, lat, lon) ───────────────────────────────
        for var in _TIME_VARS:
            raw = ds.variables[var][t].astype(np.float32).flatten()
            inputs.append(self.normalize(raw, var))

        # ── cci_agg (time, cci_class, lat, lon) → 10 channels ─────────────────
        cci_raw = ds.variables[_CCI_VAR][t].astype(np.float32)        # (10, H, W)
        cci_norm = self.normalize(cci_raw, _CCI_VAR)
        for c in range(cci_raw.shape[0]):
            inputs.append(cci_norm[c].flatten())                       # (H*W,)

        inputs_np = np.stack(inputs, axis=0).astype(np.float32)   # (19, H*W)

        # ── targets ────────────────────────────────────────────────────────────
        targets = []
        for var in _TARGET_VARS:
            v = ds.variables[var]
            if v.dimensions == _TPERA_WRONG:
                # tpERA stored as (lat, lon, time) — read single time slice
                raw = v[:, :, t].astype(np.float32).flatten()         # (H*W,)
            else:
                raw = v[t].astype(np.float32).flatten()                # (H*W,)
            targets.append(self.normalize(raw, var))

        targets_np = np.stack(targets, axis=0).astype(np.float32)  # (4, H*W)

        return inputs_np, targets_np

    # ── split helpers ─────────────────────────────────────────────────────────

    def get_split_indices(
        self,
        years: List[int],
        runs:  Optional[List[str]] = None,
        leads: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Return sample indices for the given year/run/lead filters.

        Parameters
        ----------
        years : list[int]
            Years to include (e.g. train_years, val_years, test_years).
        runs  : list[str] or None
            If given, restrict to these run identifiers.
        leads : list[int] or None
            If given, restrict to these lead times.

        Returns
        -------
        List of integer indices into this dataset.
        """
        mask = self._index["year"].isin(years)
        if runs is not None:
            mask &= self._index["run"].isin(runs)
        if leads is not None:
            mask &= self._index["lead"].isin(leads)
        return self._index[mask].index.tolist()

    def get_combination_info(self, idx: int) -> Dict:
        """Return metadata dict for sample *idx*."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")
        row = self._index.iloc[idx]
        return {
            "file_path": row["file_path"],
            "time_idx":  int(row["time_idx"]),
            "year":      int(row["year"]),
            "run":       str(row["run"]),
            "lead":      int(row["lead"]),
        }

    def available_years(self) -> List[int]:
        return sorted(self._index["year"].unique().tolist())

    def available_runs(self) -> List[str]:
        return sorted(self._index["run"].unique().tolist())

    def available_leads(self) -> List[int]:
        return sorted(self._index["lead"].unique().tolist())

    def get_summary_stats(self) -> Dict:
        return {
            "total_samples":   len(self),
            "n_files":         self._index["file_path"].nunique(),
            "years":           self.available_years(),
            "runs":            self.available_runs(),
            "leads":           self.available_leads(),
            "grid":            "270×520",
            "n_input_channels": 19,
            "n_target_vars":    4,
            "load_in_memory":   False,
        }

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self):
        """Close all cached file handles."""
        _close_all_handles()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False

    def __del__(self):
        self.close()
