"""
PyTorch Dataset wrapper for DecadalDataLoader.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ClimateDataset(Dataset):
    """
    PyTorch Dataset wrapper for DecadalDataLoader.

    Handles data preprocessing and batching for ClimateNet training.
    """

    def __init__(
        self,
        data_loader,  # DecadalDataLoader instance
        normalize: bool = True,
        image_size: Tuple[int, int] = (35, 77),
        target_vars: list = ["tasERA", "tasmaxERA", "tpERA", "rhERA"],
    ):
        """
        Initialize ClimateDataset.

        Args:
            data_loader: DecadalDataLoader instance
            normalize: Whether to normalize data
            target_vars: List of target variable names
        """
        self.data_loader = data_loader
        self.normalize = normalize
        self.image_size = tuple(image_size)
        self.target_vars = target_vars

        # Input structure from data_loader:
        # 0: pr, 1: tas, 2: tasmax, 3: hurs, 4: dem, 5: rho, 6: phi,
        # 7: sin_time, 8: cos_time, 9-18: cci_agg (10 classes)
        # Total: 19 channels

        # Static channels: dem, rho, phi (indices 4, 5, 6)
        self.static_indices = [4, 5, 6]

        # Dynamic channels: pr, tas, tasmax, hurs, sin_time, cos_time, cci_agg (10 classes)
        # Indices: 0, 1, 2, 3, 7, 8, 9-18
        self.dynamic_indices = [0, 1, 2, 3, 7, 8] + list(
            range(9, 19)
        )  # Total: 6 + 10 = 16 channels

        logger.info(
            f"Dataset initialized with {len(self.static_indices)} static channels "
            f"and {len(self.dynamic_indices)} dynamic channels"
        )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_loader)

    def __getitem__(self, idx: int) -> Tuple[Tuple, Dict, Dict]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of ((static, dynamic), targets, metadata)
            - static: [3, H, W] - static features (dem, rho, phi)
            - dynamic: [16, H, W] - dynamic features (pr, tas, tasmax, hurs, sin_time, cos_time, cci_agg[10])
            - targets: {var_name: [1, H, W]} - target variables
            - metadata: {'lead': int, 'run_idx': int, ...}
        """
        # Get raw data
        inputs, targets = self.data_loader[idx]

        # Get metadata
        metadata = self.data_loader.get_combination_info(idx)

        # Reshape from [19, H*W] to [19, H, W]
        H, W = self.image_size
        inputs = inputs.reshape(inputs.shape[0], H, W)  # [19, H, W]

        # Split static and dynamic features
        static = inputs[self.static_indices]  # [3, H, W]
        dynamic = inputs[self.dynamic_indices]  # [16, H, W]

        # Normalize if requested
        if self.normalize:
            # Normalize static channels
            static_var_names = ["dem", "rho", "phi"]
            for i, var_name in enumerate(static_var_names):
                if var_name in self.data_loader.scalers:
                    static[i] = self.data_loader.normalize(
                        static[i], var_name, fit=False
                    )

            # Normalize dynamic channels
            # First 6 are regular variables
            dynamic_var_names = ["pr", "tas", "tasmax", "hurs", "sin_time", "cos_time"]
            for i, var_name in enumerate(dynamic_var_names):
                if var_name in self.data_loader.scalers:
                    dynamic[i] = self.data_loader.normalize(
                        dynamic[i], var_name, fit=False
                    )

            # Next 10 are cci_agg classes - normalize together
            if "cci_agg" in self.data_loader.scalers:
                for i in range(6, 16):  # indices 6-15 in dynamic (cci_agg classes)
                    dynamic[i] = self.data_loader.normalize(
                        dynamic[i], "cci_agg", fit=False
                    )

        # Process targets
        targets_dict = {}
        for i, var in enumerate(self.target_vars):
            target = targets[i].reshape(H, W)  # [H, W]

            if self.normalize and var in self.data_loader.scalers:
                target = self.data_loader.normalize(target, var, fit=False)

            targets_dict[var] = (
                torch.from_numpy(target).unsqueeze(0).float()
            )  # [1, H, W]

        # Convert to tensors
        static = torch.from_numpy(static).float()  # [3, H, W]
        dynamic = torch.from_numpy(dynamic).float()  # [16, H, W]

        # Prepare metadata tensor
        metadata_tensor = {
            "lead": torch.tensor(metadata["lead"], dtype=torch.long),
            "run_idx": torch.tensor(metadata["run_idx"], dtype=torch.long),
            "lead_idx": torch.tensor(metadata["lead_idx"], dtype=torch.long),
            "time_idx": torch.tensor(metadata["time_idx"], dtype=torch.long),
        }

        return (static, dynamic), targets_dict, metadata_tensor
