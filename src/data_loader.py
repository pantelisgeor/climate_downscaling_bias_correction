"""
Data loader for decadal climate prediction bias correction.

This module provides a data loader for handling multi-dimensional climate datasets
with run and lead time dimensions, filtering out invalid combinations with NaN values.
Supports in-memory loading for maximum performance on systems with sufficient RAM.
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging
import hashlib
import time

logger = logging.getLogger(__name__)


class DecadalDataLoader:
    """
    Data loader for decadal climate prediction datasets.

    Handles loading and preprocessing of climate data with multiple runs and lead times,
    automatically detecting and filtering out invalid (run, lead, time) combinations
    that contain NaN values.

    Supports full in-memory loading for maximum performance.

    Attributes:
        nc_path (str): Path to the netCDF file
        ds (xr.Dataset): Loaded xarray Dataset
        normalize_method (str): Normalization method ('minmax' or 'zscore')
        scalers (dict): Dictionary storing normalization parameters
        valid_combinations (pd.DataFrame): DataFrame of valid (run, lead, time) indices
        input_vars (list): List of input variable names
        target_vars (list): List of target variable names
        load_in_memory (bool): Whether dataset is loaded into RAM
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
        Initialize the DecadalDataLoader.

        Args:
            nc_path: Path to netCDF file containing the climate data
            normalize_method: Normalization method, either "minmax" or "zscore"
            cache_dir: Directory to store the valid combinations cache file
            force_recompute: If True, recompute valid combinations even if cache exists
            load_in_memory: If True, load entire dataset into RAM for maximum speed
                           (requires ~50-100GB RAM for large datasets)

        Raises:
            FileNotFoundError: If the netCDF file does not exist
            ValueError: If normalize_method is not 'minmax' or 'zscore'
            KeyError: If required variables or dimensions are missing from the dataset
        """
        start_time = time.time()
        logger.info("=" * 70)
        logger.info("INITIALIZING DECADAL DATA LOADER")
        logger.info("=" * 70)

        # Validate inputs
        logger.info("Step 1/7: Validating input parameters...")
        if normalize_method not in ["minmax", "zscore"]:
            raise ValueError(
                f"normalize_method must be 'minmax' or 'zscore', got '{normalize_method}'"
            )
        logger.info(f"  ✓ Normalization method: {normalize_method}")

        nc_path = Path(nc_path)
        if not nc_path.exists():
            raise FileNotFoundError(f"NetCDF file not found: {nc_path}")
        logger.info(f"  ✓ NetCDF file exists: {nc_path}")

        file_size_mb = nc_path.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ File size: {file_size_mb:.1f} MB")

        self.nc_path = str(nc_path)
        self.normalize_method = normalize_method
        self.scalers = {}
        self.cache_dir = Path(cache_dir)
        self.force_recompute = force_recompute
        self.load_in_memory = load_in_memory
        self._closed = False

        # Create cache directory if it doesn't exist
        logger.info("Step 2/7: Setting up cache directory...")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✓ Cache directory: {self.cache_dir.absolute()}")
        except Exception as e:
            raise IOError(f"Failed to create cache directory {cache_dir}: {e}")

        # Define variable names
        logger.info("Step 3/7: Defining variable structure...")
        self.input_vars = [
            "pr",
            "tas",
            "tasmax",
            "hurs",
            "dem",
            "rho",
            "phi",
            "sin_time",
            "cos_time",
            "cci_agg",
        ]
        self.target_vars = ["tasERA", "tasmaxERA", "tpERA", "rhERA"]
        logger.info(
            f"  ✓ Input variables ({len(self.input_vars)}): {', '.join(self.input_vars)}"
        )
        logger.info(
            f"  ✓ Target variables ({len(self.target_vars)}): {', '.join(self.target_vars)}"
        )

        # Load dataset
        logger.info("Step 4/7: Loading netCDF dataset...")
        load_start = time.time()
        try:
            self.ds = xr.open_dataset(self.nc_path)
            load_time = time.time() - load_start
            logger.info(f"  ✓ Dataset opened in {load_time:.2f}s")
        except Exception as e:
            raise IOError(f"Failed to open netCDF file {self.nc_path}: {e}")

        # Validate dataset structure
        logger.info("Step 5/7: Validating dataset structure...")
        self._validate_dataset()

        # Load into memory if requested
        if self.load_in_memory:
            logger.info("Step 6/7: Loading dataset into memory...")
            logger.info(
                "  ⚠ This will use significant RAM (~50-100GB for large datasets)"
            )
            mem_start = time.time()
            try:
                # Load all data into memory
                self.ds = self.ds.load()
                mem_time = time.time() - mem_start

                # Estimate memory usage
                total_bytes = sum(
                    var.nbytes
                    for var in self.ds.data_vars.values()
                    if hasattr(var, "nbytes")
                )
                total_gb = total_bytes / (1024**3)

                logger.info(f"  ✓ Dataset loaded into RAM in {mem_time:.2f}s")
                logger.info(f"  ✓ Estimated memory usage: {total_gb:.2f} GB")
                logger.info(
                    "  ✓ All subsequent data access will be from RAM (very fast!)"
                )
            except MemoryError as e:
                logger.error(f"  ✗ Failed to load into memory: {e}")
                logger.error("  ✗ Dataset is too large for available RAM")
                raise MemoryError(
                    f"Insufficient memory to load dataset. "
                    f"Try setting load_in_memory=False or use a machine with more RAM."
                )
            except Exception as e:
                logger.error(f"  ✗ Failed to load into memory: {e}")
                raise
        else:
            logger.info("Step 6/7: Using lazy loading (disk-based access)")
            logger.info("  Data will be read from disk as needed")

        # Load or compute valid combinations
        logger.info("Step 7/7: Loading/computing valid combinations...")
        valid_start = time.time()
        self.valid_combinations = self._load_or_compute_valid_combinations()
        valid_time = time.time() - valid_start
        logger.info(f"  ✓ Valid combinations ready in {valid_time:.2f}s")

        total_time = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"INITIALIZATION COMPLETE in {total_time:.2f}s")
        logger.info(f"Ready with {len(self.valid_combinations):,} valid samples")
        if self.load_in_memory:
            logger.info("🚀 Dataset in RAM - Maximum performance mode!")
        logger.info("=" * 70)

    def _validate_dataset(self):
        """
        Validate that the dataset contains all required variables and dimensions.

        Raises:
            KeyError: If required variables or dimensions are missing
        """
        logger.info("  Checking dimensions...")

        # Check required dimensions
        required_dims = ["run", "lead", "time", "latitude", "longitude"]
        missing_dims = [dim for dim in required_dims if dim not in self.ds.dims]
        if missing_dims:
            raise KeyError(f"Missing required dimensions in dataset: {missing_dims}")

        # Log dimension sizes
        for dim in required_dims:
            logger.info(f"    • {dim}: {len(self.ds[dim])}")

        logger.info("  Checking variables...")

        # Check required variables
        all_vars = self.input_vars + self.target_vars
        missing_vars = [var for var in all_vars if var not in self.ds.variables]
        if missing_vars:
            raise KeyError(f"Missing required variables in dataset: {missing_vars}")

        logger.info(f"    ✓ All {len(all_vars)} required variables present")

        # Log additional info about variables
        logger.info("  Variable dimensions:")
        for var in self.input_vars[:4]:  # Log first 4 to avoid clutter
            dims = self.ds[var].dims
            shape = self.ds[var].shape
            logger.info(f"    • {var}: {dims} -> {shape}")

        logger.info(f"  ✓ Dataset validation passed")

    def _get_cache_filename(self) -> Path:
        """
        Generate cache filename based on the netCDF file.

        Uses MD5 hash of file path and size to ensure cache invalidation
        if the source file changes.

        Returns:
            Path object for the cache file
        """
        try:
            file_size = Path(self.nc_path).stat().st_size
            hash_input = f"{self.nc_path}_{file_size}".encode()
            file_hash = hashlib.md5(hash_input).hexdigest()[:8]

            nc_name = Path(self.nc_path).stem
            cache_file = (
                self.cache_dir / f"{nc_name}_valid_combinations_{file_hash}.csv"
            )
            logger.debug(f"Cache filename: {cache_file.name}")
            return cache_file
        except Exception as e:
            logger.warning(f"Failed to generate hash for cache file: {e}")
            # Fallback to simple naming
            nc_name = Path(self.nc_path).stem
            return self.cache_dir / f"{nc_name}_valid_combinations.csv"

    def _load_or_compute_valid_combinations(self) -> pd.DataFrame:
        """
        Load valid (run, lead, time) combinations from cache or compute them.

        Uses vectorized operations to efficiently find all combinations without NaN values.
        When data is in memory, this is extremely fast.

        Returns:
            DataFrame with columns: run_idx, lead_idx, time_idx, run, lead

        Raises:
            IOError: If cache file cannot be read or written
            ValueError: If no valid combinations are found
        """
        cache_file = self._get_cache_filename()

        # Try to load from cache
        if cache_file.exists() and not self.force_recompute:
            logger.info(f"  Cache file found: {cache_file.name}")
            logger.info("  Attempting to load from cache...")
            try:
                df = pd.read_csv(cache_file)

                # Validate cache file structure
                required_cols = ["run_idx", "lead_idx", "time_idx", "run", "lead"]
                if all(col in df.columns for col in required_cols) and len(df) > 0:
                    logger.info(
                        f"  ✓ Successfully loaded {len(df):,} valid combinations from cache"
                    )
                    return df
                else:
                    logger.warning(
                        "  ✗ Cache file has invalid structure, will recompute..."
                    )
            except Exception as e:
                logger.warning(
                    f"  ✗ Failed to load cache file ({e}), will recompute..."
                )
        else:
            if self.force_recompute:
                logger.info("  Force recompute enabled, ignoring cache")
            else:
                logger.info("  No cache file found")

        # Compute valid combinations using vectorized operations
        logger.info("  " + "=" * 66)
        if self.load_in_memory:
            logger.info("  COMPUTING VALID COMBINATIONS (fast - data in RAM)")
        else:
            logger.info("  COMPUTING VALID COMBINATIONS (this may take a few minutes)")
        logger.info("  " + "=" * 66)

        compute_start = time.time()

        nrun = len(self.ds["run"])
        nlead = len(self.ds["lead"])
        ntime = len(self.ds["time"])

        total_combinations = nrun * nlead * ntime
        logger.info(f"  Dataset dimensions:")
        logger.info(f"    • Runs: {nrun}")
        logger.info(f"    • Leads: {nlead}")
        logger.info(f"    • Time steps: {ntime}")
        logger.info(f"    • Total combinations: {total_combinations:,}")

        # Start with all True
        logger.info(f"  Initializing validity mask ({nrun} × {nlead} × {ntime})...")
        valid_mask = np.ones((nrun, nlead, ntime), dtype=bool)
        logger.info(f"    ✓ Mask initialized (all combinations initially valid)")

        # Check input variables with (run, lead, time) dimensions
        input_vars_to_check = ["pr", "tas", "tasmax", "hurs"]

        logger.info(f"  Checking input variables with (run, lead, time) dimensions...")
        for i, var in enumerate(input_vars_to_check, 1):
            if var not in self.ds.data_vars:
                logger.warning(f"    ✗ Variable '{var}' not found in dataset, skipping")
                continue

            var_start = time.time()
            logger.info(f"    [{i}/{len(input_vars_to_check)}] Processing {var}...")

            # Compute NaN mask over latitude, longitude dims
            # This is MUCH faster when data is in memory
            has_nan = self.ds[var].isnull().any(dim=["latitude", "longitude"])

            # Ensure dims order is (run, lead, time)
            if has_nan.dims != ("run", "lead", "time"):
                logger.debug(
                    f"      Transposing from {has_nan.dims} to (run, lead, time)"
                )
                has_nan = has_nan.transpose("run", "lead", "time")

            # Update mask
            n_invalid = has_nan.values.sum()
            valid_mask &= ~has_nan.values
            n_remaining = valid_mask.sum()

            var_time = time.time() - var_start
            logger.info(f"      • Found {n_invalid:,} invalid combinations")
            logger.info(f"      • Remaining valid: {n_remaining:,}")
            logger.info(f"      • Time: {var_time:.2f}s")

        # Check static and time-varying input variables
        logger.info(f"  Checking static and time-varying input variables...")
        static_and_time_vars = ["dem", "rho", "phi", "sin_time", "cos_time"]

        for var in static_and_time_vars:
            if var not in self.ds.data_vars:
                logger.warning(f"    ✗ Variable '{var}' not found, skipping")
                continue

            logger.info(f"    Checking {var}...")

            if "time" in self.ds[var].dims:
                # Time-varying: shape (time, lat, lon)
                has_nan = self.ds[var].isnull().any(dim=["latitude", "longitude"])
                # Broadcast to (run, lead, time)
                has_nan_broadcast = np.broadcast_to(
                    has_nan.values[np.newaxis, np.newaxis, :], (nrun, nlead, ntime)
                )
                n_invalid = has_nan_broadcast.sum()
                valid_mask &= ~has_nan_broadcast
                logger.info(f"      • Time-varying: {n_invalid:,} invalid combinations")
            else:
                # Static: shape (lat, lon)
                if self.ds[var].isnull().any():
                    logger.warning(f"      ⚠ Static variable {var} contains NaN values")
                else:
                    logger.info(f"      ✓ No NaN values")

        # Check cci_agg separately (has cci_class dimension)
        if "cci_agg" in self.ds.data_vars:
            logger.info(f"    Checking cci_agg...")
            cci_start = time.time()
            # Shape: (time, cci_class, lat, lon)
            has_nan = (
                self.ds["cci_agg"]
                .isnull()
                .any(dim=["cci_class", "latitude", "longitude"])
            )
            # Broadcast to (run, lead, time)
            has_nan_broadcast = np.broadcast_to(
                has_nan.values[np.newaxis, np.newaxis, :], (nrun, nlead, ntime)
            )
            n_invalid = has_nan_broadcast.sum()
            valid_mask &= ~has_nan_broadcast
            cci_time = time.time() - cci_start
            logger.info(
                f"      • Found {n_invalid:,} invalid combinations in {cci_time:.2f}s"
            )

        # Check target variables (they don't have run/lead dimensions)
        logger.info(f"  Checking target variables...")
        for i, var in enumerate(self.target_vars, 1):
            if var not in self.ds.data_vars:
                logger.warning(f"    ✗ Target variable '{var}' not found, skipping")
                continue

            var_start = time.time()
            logger.info(f"    [{i}/{len(self.target_vars)}] Processing {var}...")

            # has_nan shape is (time,)
            has_nan = self.ds[var].isnull().any(dim=["latitude", "longitude"])

            # Broadcast to (run, lead, time) shape
            has_nan_broadcast = np.broadcast_to(
                has_nan.values[np.newaxis, np.newaxis, :], (nrun, nlead, ntime)
            )

            n_invalid = has_nan_broadcast.sum()
            valid_mask &= ~has_nan_broadcast
            n_remaining = valid_mask.sum()

            var_time = time.time() - var_start
            logger.info(f"      • Found {n_invalid:,} invalid combinations")
            logger.info(f"      • Remaining valid: {n_remaining:,}")
            logger.info(f"      • Time: {var_time:.2f}s")

        # Count valid combinations
        n_valid = valid_mask.sum()
        pct_valid = 100 * n_valid / total_combinations if total_combinations > 0 else 0

        logger.info("  " + "-" * 66)
        logger.info(f"  VALIDATION COMPLETE:")
        logger.info(f"    • Valid combinations: {n_valid:,} ({pct_valid:.1f}%)")
        logger.info(f"    • Invalid combinations: {total_combinations - n_valid:,}")
        logger.info("  " + "-" * 66)

        if n_valid == 0:
            raise ValueError(
                "No valid (run, lead, time) combinations found in dataset. "
                "All combinations contain NaN values."
            )

        # Convert mask to list of indices
        logger.info("  Converting validity mask to index list...")
        index_start = time.time()

        # Get indices where mask is True
        run_indices, lead_indices, time_indices = np.where(valid_mask)

        logger.info(f"    ✓ Extracted {len(run_indices):,} valid indices")

        # Build dataframe efficiently
        logger.info("  Building DataFrame...")
        df = pd.DataFrame(
            {
                "run_idx": run_indices.astype(int),
                "lead_idx": lead_indices.astype(int),
                "time_idx": time_indices.astype(int),
                "run": [str(self.ds.run.values[i]) for i in run_indices],
                "lead": [int(self.ds.lead.values[i]) for i in lead_indices],
            }
        )

        index_time = time.time() - index_start
        logger.info(f"    ✓ DataFrame created in {index_time:.2f}s")

        # Save to cache
        logger.info("  Saving to cache...")
        cache_start = time.time()
        try:
            df.to_csv(cache_file, index=False)
            cache_size_kb = cache_file.stat().st_size / 1024
            cache_time = time.time() - cache_start
            logger.info(f"    ✓ Cache saved: {cache_file.name}")
            logger.info(f"    ✓ Cache size: {cache_size_kb:.1f} KB")
            logger.info(f"    ✓ Save time: {cache_time:.2f}s")
        except Exception as e:
            logger.warning(f"    ✗ Failed to save cache file: {e}")

        total_compute_time = time.time() - compute_start
        logger.info("  " + "=" * 66)
        logger.info(f"  COMPUTATION COMPLETE in {total_compute_time:.2f}s")
        if self.load_in_memory:
            logger.info("  (Lightning fast thanks to in-memory data!)")
        logger.info("  " + "=" * 66)

        return df

    def normalize(
        self, data: np.ndarray, var_name: str, fit: bool = True
    ) -> np.ndarray:
        """
        Normalize data using the specified method.

        Applies log1p transform to precipitation (tpERA) before normalization.

        Args:
            data: Input data array to normalize
            var_name: Variable name (used as key for storing/retrieving scaler params)
            fit: If True, compute and store normalization parameters;
                 if False, use existing parameters

        Returns:
            Normalized data array

        Raises:
            ValueError: If fit=False but no scaler exists for var_name
        """
        if not fit and var_name not in self.scalers:
            raise ValueError(
                f"No scaler found for variable '{var_name}'. "
                f"Must call normalize with fit=True first."
            )

        # Apply log transform to precipitation
        if var_name == "tpERA":
            data = np.log1p(np.maximum(data, 0))

        if self.normalize_method == "minmax":
            if fit:
                min_val = float(np.nanmin(data))
                max_val = float(np.nanmax(data))
                if abs(max_val - min_val) < 1e-8:
                    logger.warning(
                        f"Variable {var_name} has constant value {min_val}, "
                        f"using range [0, 1] to avoid division by zero"
                    )
                    max_val = min_val + 1.0
                self.scalers[var_name] = {"min": min_val, "max": max_val}
                logger.debug(
                    f"Fitted minmax scaler for {var_name}: [{min_val:.4f}, {max_val:.4f}]"
                )
            else:
                # Retrieve and convert to float
                scaler = self.scalers[var_name]
                if isinstance(scaler, dict):
                    min_val = float(scaler["min"])
                    max_val = float(scaler["max"])
                else:
                    min_val, max_val = float(scaler[0]), float(scaler[1])

            return (data - min_val) / (max_val - min_val + 1e-8)

        elif self.normalize_method == "zscore":
            if fit:
                mean_val = float(np.nanmean(data))
                std_val = float(np.nanstd(data))
                if std_val < 1e-8:
                    logger.warning(
                        f"Variable {var_name} has zero standard deviation, "
                        f"using std=1.0 to avoid division by zero"
                    )
                    std_val = 1.0
                self.scalers[var_name] = {"mean": mean_val, "std": std_val}
                logger.debug(
                    f"Fitted zscore scaler for {var_name}: mean={mean_val:.4f}, std={std_val:.4f}"
                )
            else:
                # Retrieve and convert to float
                scaler = self.scalers[var_name]
                if isinstance(scaler, dict):
                    mean_val = float(scaler["mean"])
                    std_val = float(scaler["std"])
                else:
                    mean_val, std_val = float(scaler[0]), float(scaler[1])

            return (data - mean_val) / (std_val + 1e-8)

    def denormalize(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """
        Denormalize data using stored normalization parameters.

        Applies inverse log1p transform to precipitation (tpERA) after denormalization.

        Args:
            data: Normalized data array
            var_name: Variable name (used to retrieve scaler parameters)

        Returns:
            Denormalized data array

        Raises:
            ValueError: If no scaler exists for var_name
        """
        if var_name not in self.scalers:
            raise ValueError(
                f"No scaler found for variable '{var_name}'. "
                f"Cannot denormalize without first normalizing."
            )

        scaler = self.scalers[var_name]

        if self.normalize_method == "minmax":
            if isinstance(scaler, dict):
                min_val = float(scaler["min"])
                max_val = float(scaler["max"])
            else:
                min_val, max_val = float(scaler[0]), float(scaler[1])
            denorm_data = data * (max_val - min_val) + min_val

        elif self.normalize_method == "zscore":
            if isinstance(scaler, dict):
                mean_val = float(scaler["mean"])
                std_val = float(scaler["std"])
            else:
                mean_val, std_val = float(scaler[0]), float(scaler[1])
            denorm_data = data * std_val + mean_val

        # Apply inverse log transform for precipitation
        if var_name == "tpERA":
            denorm_data = np.expm1(denorm_data)

        return denorm_data

    def __len__(self) -> int:
        """
        Return the number of valid samples.

        Returns:
            Number of valid (run, lead, time) combinations
        """
        return len(self.valid_combinations)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a sample at the given index.

        When data is loaded in memory, this is extremely fast (pure numpy operations).

        Args:
            idx: Index into the valid combinations (must be < len(self))

        Returns:
            Tuple of (input_data, target_data) as numpy arrays:
            - input_data: shape (n_input_channels, H*W) where n_input_channels includes 10 for cci_agg
            - target_data: shape (n_target_vars, H*W)

        Raises:
            IndexError: If idx is out of range
            KeyError: If expected variables are missing from dataset
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Get the valid combination for this index
        combo = self.valid_combinations.iloc[idx]
        run_idx = int(combo["run_idx"])
        lead_idx = int(combo["lead_idx"])
        time_idx = int(combo["time_idx"])

        logger.debug(
            f"Accessing sample {idx}: run={run_idx}, lead={lead_idx}, time={time_idx}"
        )

        # Extract input variables
        # We need to handle cci_agg specially because it has 10 classes
        inputs = []

        for var in self.input_vars:
            try:
                var_data = self.ds[var]

                # Handle different variable dimensions
                if var == "cci_agg":
                    # cci_agg has shape (time, cci_class=10, lat, lon)
                    # Extract for this time and flatten all 10 classes
                    data = var_data.isel(time=time_idx).values  # Shape: (10, lat, lon)
                    # Flatten to (10, lat*lon)
                    n_classes = data.shape[0]
                    H, W = data.shape[1], data.shape[2]
                    data = data.reshape(n_classes, H * W)  # Shape: (10, H*W)
                    # Add each class as a separate "variable"
                    for c in range(n_classes):
                        inputs.append(data[c])  # Each is (H*W,)
                elif "run" in var_data.dims and "lead" in var_data.dims:
                    # Variables with (run, lead, time, lat, lon)
                    data = var_data.isel(
                        run=run_idx, lead=lead_idx, time=time_idx
                    ).values.flatten()
                    inputs.append(data)
                elif "time" in var_data.dims:
                    # Time-varying variables without run/lead (e.g., sin_time, cos_time)
                    data = var_data.isel(time=time_idx).values.flatten()
                    inputs.append(data)
                else:
                    # Static variables (dem, rho, phi) - no time dimension
                    data = var_data.values.flatten()
                    inputs.append(data)

            except Exception as e:
                raise KeyError(
                    f"Failed to extract variable '{var}' at index {idx}: {e}"
                )

        # Now inputs is a list where most entries have shape (H*W,)
        # but cci_agg contributed 10 entries
        # Total: 9 regular vars + 10 cci_agg channels = 19 channels
        inputs = np.stack(inputs, axis=0)  # Shape: (19, H*W)

        # Extract target variables (they don't have run/lead dimensions)
        targets = []
        for var in self.target_vars:
            try:
                data = self.ds[var].isel(time=time_idx).values.flatten()
                targets.append(data)
            except Exception as e:
                raise KeyError(
                    f"Failed to extract target variable '{var}' at index {idx}: {e}"
                )

        targets = np.stack(targets, axis=0)  # Shape: (4, H*W)

        logger.debug(
            f"Sample {idx}: inputs shape={inputs.shape}, targets shape={targets.shape}"
        )

        return inputs, targets

    def get_combination_info(self, idx: int) -> Dict[str, any]:
        """
        Get metadata about the run, lead, and time for a given index.

        Args:
            idx: Index into the valid combinations

        Returns:
            Dictionary containing:
            - run: Run identifier string
            - lead: Lead time value
            - run_idx: Run index in dataset
            - lead_idx: Lead index in dataset
            - time_idx: Time index in dataset
            - time: Datetime value

        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        combo = self.valid_combinations.iloc[idx]
        time_idx = int(combo["time_idx"])

        return {
            "run": combo["run"],
            "lead": int(combo["lead"]),
            "run_idx": int(combo["run_idx"]),
            "lead_idx": int(combo["lead_idx"]),
            "time_idx": time_idx,
            "time": self.ds.time.values[time_idx],
        }

    def get_indices_by_run_lead(
        self, run: Optional[str] = None, lead: Optional[int] = None
    ) -> List[int]:
        """
        Get indices for specific run and/or lead values.

        Args:
            run: Run identifier (e.g., 'r1i1p1f1'). If None, all runs are included.
            lead: Lead time (0-10). If None, all leads are included.

        Returns:
            List of indices matching the criteria

        Examples:
            >>> loader.get_indices_by_run_lead(run='r1i1p1f1')  # All leads for run r1i1p1f1
            >>> loader.get_indices_by_run_lead(lead=0)  # All runs at lead time 0
            >>> loader.get_indices_by_run_lead(run='r1i1p1f1', lead=5)  # Specific combination
        """
        mask = pd.Series([True] * len(self.valid_combinations))

        if run is not None:
            mask &= self.valid_combinations["run"] == run
        if lead is not None:
            mask &= self.valid_combinations["lead"] == lead

        indices = self.valid_combinations[mask].index.tolist()
        logger.debug(f"Found {len(indices)} indices for run={run}, lead={lead}")

        return indices

    def get_summary_stats(self) -> Dict[str, any]:
        """
        Get summary statistics about the dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        logger.info("Computing dataset summary statistics...")

        stats = {
            "total_valid_samples": len(self),
            "n_runs": len(self.ds.run),
            "n_leads": len(self.ds.lead),
            "n_times": len(self.ds.time),
            "runs": list(self.ds.run.values),
            "leads": list(self.ds.lead.values),
            "samples_per_run": self.valid_combinations.groupby("run").size().to_dict(),
            "samples_per_lead": self.valid_combinations.groupby("lead")
            .size()
            .to_dict(),
            "in_memory": self.load_in_memory,
        }

        logger.debug(
            f"Summary stats: {stats['total_valid_samples']:,} samples across "
            f"{stats['n_runs']} runs and {stats['n_leads']} leads"
        )

        return stats

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get estimated memory usage of the dataset.

        Returns:
            Dictionary with memory usage in GB for different components
        """
        if not self.load_in_memory:
            return {"status": "not_in_memory", "usage_gb": 0.0}

        total_bytes = 0
        var_usage = {}

        for var_name in list(self.ds.data_vars):
            var = self.ds[var_name]
            if hasattr(var, "nbytes"):
                var_bytes = var.nbytes
                total_bytes += var_bytes
                var_usage[var_name] = var_bytes / (1024**3)  # Convert to GB

        return {
            "status": "in_memory",
            "total_gb": total_bytes / (1024**3),
            "variables": var_usage,
        }

    def close(self):
        """Explicitly close the dataset and free memory."""
        if not self._closed and hasattr(self, "ds"):
            try:
                self.ds.close()
                self._closed = True
                if self.load_in_memory:
                    logger.info("Dataset closed - memory freed")
                else:
                    logger.debug("Dataset closed successfully")
            except Exception as e:
                logger.warning(f"Failed to close dataset: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close dataset."""
        self.close()
        return False

    def __del__(self):
        """Close the dataset when the object is destroyed."""
        self.close()
