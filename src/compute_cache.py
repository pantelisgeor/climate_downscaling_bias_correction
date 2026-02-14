# compute_cache.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import DecadalDataLoader

nc_path = "/nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models/preprocessed_data/decadal_for_training/full_dataset.nc"
cache_dir = "."  # or specify your preferred cache location

print("Computing valid combinations (this will take a while on first run)...")
loader = DecadalDataLoader(nc_path, cache_dir=cache_dir)
print(f"Done! Found {len(loader):,} valid samples")
print(f"Cache saved for future use")
