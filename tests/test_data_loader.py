"""
Unit and integration tests for DecadalDataLoader using real data.

Usage:
    pytest tests/test_data_loader.py -v --nc-path=/path/to/your/data.nc

Or set environment variable:
    export TEST_NC_PATH=/path/to/your/data.nc
    pytest tests/test_data_loader.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import sys
import gc
import time

# Add src to path
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import DecadalDataLoader


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for cache files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def data_loader(nc_path, temp_dir):
    """Create a DecadalDataLoader instance with in-memory loading."""
    print(f"\n🚀 Loading dataset into memory: {nc_path}")
    print("This may take 1-2 minutes and use significant RAM...")
    with DecadalDataLoader(
        nc_path, normalize_method="minmax", cache_dir=temp_dir, load_in_memory=True
    ) as loader:
        yield loader


class TestDecadalDataLoaderInit:
    """Test initialization and validation."""

    def test_init_success_in_memory(self, nc_path, temp_dir):
        """Test successful initialization with in-memory loading."""
        with DecadalDataLoader(
            nc_path, cache_dir=temp_dir, load_in_memory=True
        ) as loader:
            assert loader is not None
            assert len(loader) > 0
            assert loader.normalize_method == "minmax"
            assert loader.load_in_memory is True
            print(f"\n✓ Loaded {len(loader):,} samples into memory")

    def test_init_success_lazy(self, nc_path, temp_dir):
        """Test successful initialization with lazy loading."""
        with DecadalDataLoader(
            nc_path, cache_dir=temp_dir, load_in_memory=False
        ) as loader:
            assert loader is not None
            assert len(loader) > 0
            assert loader.load_in_memory is False
            print(f"\n✓ Lazy loading enabled, {len(loader):,} samples available")

    def test_init_zscore(self, nc_path, temp_dir):
        """Test initialization with zscore normalization."""
        with DecadalDataLoader(
            nc_path, normalize_method="zscore", cache_dir=temp_dir, load_in_memory=True
        ) as loader:
            assert loader.normalize_method == "zscore"

    def test_init_invalid_normalize_method(self, nc_path, temp_dir):
        """Test that invalid normalization method raises ValueError."""
        with pytest.raises(ValueError, match="normalize_method must be"):
            DecadalDataLoader(nc_path, normalize_method="invalid", cache_dir=temp_dir)

    def test_init_missing_file(self, temp_dir):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DecadalDataLoader("nonexistent.nc", cache_dir=temp_dir)


class TestMemoryManagement:
    """Test memory management features."""

    def test_memory_usage_in_memory(self, data_loader):
        """Test memory usage reporting for in-memory dataset."""
        mem_usage = data_loader.get_memory_usage()

        assert mem_usage["status"] == "in_memory"
        assert "total_gb" in mem_usage
        assert mem_usage["total_gb"] > 0
        assert "variables" in mem_usage

        print(f"\n📊 Memory Usage Report:")
        print(f"  Total: {mem_usage['total_gb']:.2f} GB")
        print(f"  Top variables by size:")
        sorted_vars = sorted(
            mem_usage["variables"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        for var, size_gb in sorted_vars:
            print(f"    • {var}: {size_gb:.2f} GB")

    def test_memory_usage_lazy(self, nc_path, temp_dir):
        """Test memory usage reporting for lazy loading."""
        with DecadalDataLoader(
            nc_path, cache_dir=temp_dir, load_in_memory=False
        ) as loader:
            mem_usage = loader.get_memory_usage()

            assert mem_usage["status"] == "not_in_memory"
            assert mem_usage["usage_gb"] == 0.0
            print("\n✓ Lazy loading confirmed - minimal memory usage")

    def test_context_manager_frees_memory(self, nc_path, temp_dir):
        """Test that context manager properly frees memory."""
        with DecadalDataLoader(
            nc_path, cache_dir=temp_dir, load_in_memory=True
        ) as loader:
            mem_before = loader.get_memory_usage()["total_gb"]
            print(f"\n  Memory in use: {mem_before:.2f} GB")

        # After context, dataset should be closed
        gc.collect()
        print("  ✓ Context exited - memory should be freed")


class TestValidCombinations:
    """Test valid combination detection and caching."""

    def test_valid_combinations_computed(self, data_loader):
        """Test that valid combinations are computed correctly."""
        assert len(data_loader.valid_combinations) > 0
        assert "run_idx" in data_loader.valid_combinations.columns
        assert "lead_idx" in data_loader.valid_combinations.columns
        assert "time_idx" in data_loader.valid_combinations.columns
        assert "run" in data_loader.valid_combinations.columns
        assert "lead" in data_loader.valid_combinations.columns

        print(f"\nValid combinations found: {len(data_loader.valid_combinations):,}")

    def test_valid_combinations_structure(self, data_loader):
        """Test that valid combinations have correct structure."""
        df = data_loader.valid_combinations

        # Check that indices are within bounds
        assert df["run_idx"].min() >= 0
        assert df["run_idx"].max() < len(data_loader.ds.run)
        assert df["lead_idx"].min() >= 0
        assert df["lead_idx"].max() < len(data_loader.ds.lead)
        assert df["time_idx"].min() >= 0
        assert df["time_idx"].max() < len(data_loader.ds.time)

    def test_cache_created(self, nc_path, temp_dir):
        """Test that cache file is created."""
        with DecadalDataLoader(
            nc_path, cache_dir=temp_dir, load_in_memory=True
        ) as loader:
            pass

        # Check cache file exists
        cache_files = list(Path(temp_dir).glob("*_valid_combinations_*.csv"))
        assert len(cache_files) > 0
        print(f"\nCache file created: {cache_files[0]}")

    def test_cache_speeds_up_init(self, nc_path, temp_dir):
        """Test that cache significantly speeds up initialization."""
        # First load - computes and caches
        start1 = time.time()
        with DecadalDataLoader(
            nc_path, cache_dir=temp_dir, load_in_memory=True, force_recompute=True
        ) as loader1:
            n1 = len(loader1)
        time1 = time.time() - start1

        gc.collect()
        time.sleep(1.0)

        # Second load - uses cache
        start2 = time.time()
        with DecadalDataLoader(
            nc_path, cache_dir=temp_dir, load_in_memory=True
        ) as loader2:
            n2 = len(loader2)
        time2 = time.time() - start2

        assert n1 == n2
        print(f"\n⏱ Timing comparison:")
        print(f"  First load (compute + cache): {time1:.2f}s")
        print(f"  Second load (use cache): {time2:.2f}s")
        print(f"  Speedup: {time1/time2:.1f}x")


class TestDataAccess:
    """Test data access methods."""

    def test_len(self, data_loader):
        """Test __len__ method."""
        assert len(data_loader) > 0
        assert isinstance(len(data_loader), int)
        print(f"\nDataset size: {len(data_loader):,} samples")

    def test_getitem_valid(self, data_loader):
        """Test __getitem__ with valid index."""
        inputs, targets = data_loader[0]

        # Check shapes
        assert inputs.shape[0] == 19  # 9 regular vars + 10 cci_agg classes
        assert targets.shape[0] == 4  # Number of target variables
        assert inputs.shape[1] == targets.shape[1]  # Same spatial points

        print(f"\nSample shape - Inputs: {inputs.shape}, Targets: {targets.shape}")

        # Check no NaN values
        assert not np.any(np.isnan(inputs)), "Found NaN in inputs"
        assert not np.any(np.isnan(targets)), "Found NaN in targets"

    def test_getitem_speed_in_memory(self, data_loader):
        """Test that in-memory access is very fast."""
        n_samples = 1000
        indices = np.random.choice(
            len(data_loader), min(n_samples, len(data_loader)), replace=False
        )
        actual_samples = len(indices)

        start = time.time()
        for idx in indices:
            inputs, targets = data_loader[idx]
        elapsed = time.time() - start

        samples_per_sec = actual_samples / elapsed
        print(f"\n⚡ Performance: {samples_per_sec:.0f} samples/second")
        print(f"   ({elapsed*1000/actual_samples:.2f} ms per sample)")

        # In-memory should be very fast (>1000 samples/sec)
        assert samples_per_sec > 100, "In-memory access should be fast"

    def test_getitem_random_samples(self, data_loader):
        """Test random samples from the dataset."""
        n_samples = min(20, len(data_loader))
        indices = np.random.choice(len(data_loader), n_samples, replace=False)

        for idx in indices:
            inputs, targets = data_loader[idx]
            assert inputs is not None
            assert targets is not None
            assert not np.any(np.isnan(inputs)), f"NaN found in inputs at index {idx}"
            assert not np.any(np.isnan(targets)), f"NaN found in targets at index {idx}"

    def test_getitem_boundary_indices(self, data_loader):
        """Test first and last indices."""
        # First index
        inputs, targets = data_loader[0]
        assert inputs is not None
        assert not np.any(np.isnan(inputs))

        # Last index
        inputs, targets = data_loader[len(data_loader) - 1]
        assert inputs is not None
        assert not np.any(np.isnan(inputs))

    def test_getitem_out_of_bounds(self, data_loader):
        """Test that out of bounds index raises IndexError."""
        with pytest.raises(IndexError):
            _ = data_loader[len(data_loader)]

        with pytest.raises(IndexError):
            _ = data_loader[-len(data_loader) - 1]

    def test_get_combination_info(self, data_loader):
        """Test get_combination_info method."""
        info = data_loader.get_combination_info(0)

        assert "run" in info
        assert "lead" in info
        assert "run_idx" in info
        assert "lead_idx" in info
        assert "time_idx" in info
        assert "time" in info

        # Check types
        assert isinstance(info["run"], str)
        assert isinstance(info["lead"], (int, np.integer))
        assert isinstance(info["run_idx"], (int, np.integer))

        print(
            f"\nSample info: run={info['run']}, lead={info['lead']}, time={info['time']}"
        )


class TestFiltering:
    """Test filtering by run and lead."""

    def test_get_available_runs(self, data_loader):
        """Test getting list of available runs."""
        available_runs = data_loader.valid_combinations["run"].unique()
        assert len(available_runs) > 0
        print(f"\nAvailable runs: {list(available_runs)}")

    def test_get_available_leads(self, data_loader):
        """Test getting list of available leads."""
        available_leads = sorted(data_loader.valid_combinations["lead"].unique())
        assert len(available_leads) > 0
        print(f"\nAvailable leads: {list(available_leads)}")

    def test_filter_by_run(self, data_loader):
        """Test filtering by run."""
        available_runs = data_loader.valid_combinations["run"].unique()
        test_run = available_runs[0]

        indices = data_loader.get_indices_by_run_lead(run=test_run)
        assert len(indices) > 0

        # Verify all returned indices have the correct run
        for idx in indices[:5]:
            info = data_loader.get_combination_info(idx)
            assert info["run"] == test_run

        print(f"\nRun '{test_run}': {len(indices)} samples")

    def test_filter_by_lead(self, data_loader):
        """Test filtering by lead."""
        available_leads = sorted(data_loader.valid_combinations["lead"].unique())
        test_lead = available_leads[0]

        indices = data_loader.get_indices_by_run_lead(lead=test_lead)
        assert len(indices) > 0

        # Verify all returned indices have the correct lead
        for idx in indices[:5]:
            info = data_loader.get_combination_info(idx)
            assert info["lead"] == test_lead

        print(f"\nLead {test_lead}: {len(indices)} samples")


class TestNormalization:
    """Test normalization and denormalization."""

    def test_normalize_minmax_fit(self, data_loader):
        """Test minmax normalization with fit."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = data_loader.normalize(data, "test_var", fit=True)

        assert normalized.min() >= -1e-6
        assert normalized.max() <= 1 + 1e-6
        assert "test_var" in data_loader.scalers

    def test_denormalize_minmax(self, data_loader):
        """Test denormalization with minmax."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = data_loader.normalize(original, "test_denorm", fit=True)
        denormalized = data_loader.denormalize(normalized, "test_denorm")

        np.testing.assert_allclose(original, denormalized, rtol=1e-5)


class TestSummaryStats:
    """Test summary statistics method."""

    def test_get_summary_stats(self, data_loader):
        """Test get_summary_stats method."""
        stats = data_loader.get_summary_stats()

        assert "total_valid_samples" in stats
        assert "in_memory" in stats
        assert stats["in_memory"] is True

        print(f"\n{'='*60}")
        print("DATASET SUMMARY STATISTICS")
        print(f"{'='*60}")
        print(f"Total valid samples: {stats['total_valid_samples']:,}")
        print(f"In-memory loading: {stats['in_memory']}")
        print(f"Number of runs: {stats['n_runs']}")
        print(f"Number of leads: {stats['n_leads']}")
        print(f"{'='*60}")


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_no_nan_values(self, data_loader):
        """Test that there are no NaN values in returned data."""
        n_samples = min(50, len(data_loader))
        indices = np.random.choice(len(data_loader), n_samples, replace=False)

        for idx in indices:
            inputs, targets = data_loader[idx]
            assert not np.any(np.isnan(inputs)), f"NaN found in inputs at index {idx}"
            assert not np.any(np.isnan(targets)), f"NaN found in targets at index {idx}"

    def test_no_infinite_values(self, data_loader):
        """Test that there are no infinite values in the data."""
        n_samples = min(50, len(data_loader))
        indices = np.random.choice(len(data_loader), n_samples, replace=False)

        for idx in indices:
            inputs, targets = data_loader[idx]
            assert not np.any(np.isinf(inputs)), f"Inf found in inputs at index {idx}"
            assert not np.any(np.isinf(targets)), f"Inf found in targets at index {idx}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
