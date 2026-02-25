"""
PyTorch Dataset wrapper for DecadalDownscaleDataLoader.

Mirrors ClimateDataset (src/training/climate_dataset.py) but adapted for the
1.6 TB disk-based downscaling dataset.

Key differences from ClimateDataset:
  • No second normalization pass — DecadalDownscaleDataLoader.__getitem__
    already returns normalized float32 arrays (normalization from pre-computed
    per-year minmax CSVs).  normalize flag is accepted but does nothing.
  • Image size is 270×520 (not 35×77).
  • get_combination_info returns {file_path, time_idx, year, run, lead} —
    no run_idx.  run_idx is computed here via a sorted unique-runs mapping.
  • metadata tensor contains 'lead', 'run_idx', 'lead_idx', 'time_idx'.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Channel layout produced by DecadalDownscaleDataLoader.__getitem__:
#   0  pr              4  dem            7  sin_time
#   1  tas             5  rho            8  cos_time
#   2  tasmax          6  phi            9-18  cci_agg×10
#   3  hurs
_STATIC_INDICES  = [4, 5, 6]                        # dem, rho, phi  → 3 ch
_DYNAMIC_INDICES = [0, 1, 2, 3, 7, 8] + list(range(9, 19))  # 16 ch


class ClimateDatasetDownscale(Dataset):
    """
    PyTorch Dataset that wraps DecadalDownscaleDataLoader for ClimateNet
    training/inference.

    Parameters
    ----------
    data_loader : DecadalDownscaleDataLoader
        Configured downscale data loader.
    normalize : bool
        Accepted for API compatibility but ignored — the loader already
        normalizes every sample.
    image_size : tuple[int, int]
        (H, W) grid size.  Must match the loaded data.
    target_vars : list[str]
        Target variable names in output order.
    """

    def __init__(
        self,
        data_loader,
        normalize: bool = True,
        image_size: Tuple[int, int] = (270, 520),
        target_vars: Optional[List[str]] = None,
    ):
        self.data_loader = data_loader
        self.image_size  = tuple(image_size)
        self.target_vars = target_vars or ["tasERA", "tasmaxERA", "tpERA", "rhERA"]

        # Build run→index mapping for metadata tensors
        self._run_to_idx: Dict[str, int] = {
            r: i for i, r in enumerate(data_loader.available_runs())
        }

        logger.info(
            f"ClimateDatasetDownscale: {len(self)} samples  "
            f"grid={self.image_size}  "
            f"static={len(_STATIC_INDICES)}ch  dynamic={len(_DYNAMIC_INDICES)}ch  "
            f"targets={self.target_vars}"
        )

    # ─────────────────────────────────────── dataset interface

    def __len__(self) -> int:
        return len(self.data_loader)

    def __getitem__(self, idx: int) -> Tuple[Tuple, Dict, Dict]:
        """
        Return one sample as ((static, dynamic), targets_dict, metadata_dict).

        static   : float32 tensor  (3,  H, W)  — dem, rho, phi
        dynamic  : float32 tensor  (16, H, W)  — pr, tas, tasmax, hurs,
                                                  sin_time, cos_time, cci×10
        targets  : {var: float32 tensor (1, H, W)}
        metadata : {'lead': int64, 'run_idx': int64, 'lead_idx': int64,
                    'time_idx': int64}
        """
        H, W = self.image_size

        # (19, H*W)  and  (4, H*W)  — already normalized float32
        inputs_flat, targets_flat = self.data_loader[idx]

        # --- reshape and split inputs ---
        inputs = inputs_flat.reshape(19, H, W)          # (19, H, W)

        static  = torch.from_numpy(inputs[_STATIC_INDICES])   # (3,  H, W)
        dynamic = torch.from_numpy(inputs[_DYNAMIC_INDICES])  # (16, H, W)

        # --- targets ---
        targets_arr = targets_flat.reshape(len(self.target_vars), H, W)  # (4, H, W)
        targets_dict = {
            var: torch.from_numpy(targets_arr[i]).unsqueeze(0)  # (1, H, W)
            for i, var in enumerate(self.target_vars)
        }

        # --- metadata ---
        combo = self.data_loader.get_combination_info(idx)
        lead     = int(combo["lead"])
        run_name = str(combo["run"])
        run_idx  = self._run_to_idx.get(run_name, 0)
        time_idx = int(combo["time_idx"])

        metadata = {
            "lead":      torch.tensor(lead,     dtype=torch.long),
            "run_idx":   torch.tensor(run_idx,  dtype=torch.long),
            "lead_idx":  torch.tensor(lead,     dtype=torch.long),  # == lead (0-10)
            "time_idx":  torch.tensor(time_idx, dtype=torch.long),
        }

        return (static, dynamic), targets_dict, metadata
