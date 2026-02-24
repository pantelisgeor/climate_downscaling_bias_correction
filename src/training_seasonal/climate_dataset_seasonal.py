"""
PyTorch Dataset wrapper for SeasonalDataLoader.

Mirrors src/training/climate_dataset.py but adapted for the seasonal forecast
dataset where:

  • "pr"  channel is stored as "tp"      (precipitation ensemble forecast)
  • "tas" channel is stored as "t2m"     (2-m temperature)
  • "tasmax" channel is stored as "tmax" (max temperature)
  • There is no separate run/lead dimension pair.  Instead there is a single
    "number" dimension (ensemble member) and a scalar lead-month per sample.

The public interface is identical to ClimateDataset so that Trainer, Evaluator,
and the training scripts can reuse them without modification – the only
difference visible downstream is the metadata dict, which exposes:
  metadata["lead"]       → lead_month integer (0-6)  [same key as decadal]
  metadata["number_idx"] → positional index of ensemble member
  metadata["time_idx"]   → positional index in time dimension
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ClimateDatasetSeasonal(Dataset):
    """
    PyTorch Dataset wrapper for SeasonalDataLoader.

    Input channel layout (19 channels, matches decadal):
      0  : tp          (precipitation forecast)
      1  : t2m         (2-m temperature)
      2  : tmax        (max temperature)
      3  : hurs        (relative humidity)
      4  : dem         (static)
      5  : rho         (static)
      6  : phi         (static)
      7  : sin_time
      8  : cos_time
      9-18 : cci_agg   (10 land-cover classes)

    Splits:
      static  = channels [4, 5, 6]   →  shape (3,  H, W)
      dynamic = channels [0,1,2,3,7,8,9-18]  →  shape (16, H, W)
    """

    def __init__(
        self,
        data_loader,  # SeasonalDataLoader instance
        normalize: bool = True,
        image_size: Tuple[int, int] = (28, 53),
        target_vars: List[str] = ["tasERA", "tasmaxERA", "tpERA", "rhERA"],
    ):
        """
        Parameters
        ----------
        data_loader : SeasonalDataLoader
            Fully initialised seasonal data loader.
        normalize : bool
            Whether to apply normalisation using data_loader.scalers.
        image_size : (H, W)
            Spatial grid of the seasonal dataset.
        target_vars : list[str]
            Names of target variables in output order.
        """
        self.data_loader = data_loader
        self.normalize = normalize
        self.image_size = tuple(image_size)
        self.target_vars = target_vars

        if self.normalize:
            if not hasattr(self.data_loader, "scalers") or not isinstance(
                getattr(self.data_loader, "scalers", None), dict
            ):
                raise ValueError(
                    "Normalization requested but `data_loader.scalers` is missing or"
                    " not a dict.  Ensure scalers are fitted before creating"
                    " ClimateDatasetSeasonal."
                )

        # ── channel index mapping (identical to decadal) ───────────────────
        # Channels 4, 5, 6  →  dem, rho, phi
        self.static_indices = [4, 5, 6]

        # Channels 0,1,2,3 (fc vars) + 7,8 (time) + 9-18 (cci) = 16 dynamic
        self.dynamic_indices = [0, 1, 2, 3, 7, 8] + list(range(9, 19))

        # Scaler keys used by each dynamic channel (for normalisation)
        # The seasonal NC uses 'tp', 't2m', 'tmax' instead of 'pr', 'tas', 'tasmax'
        self._dynamic_var_names = ["tp", "t2m", "tmax", "hurs", "sin_time", "cos_time"]
        self._static_var_names = ["dem", "rho", "phi"]

        logger.info(
            f"ClimateDatasetSeasonal initialised  |  "
            f"{len(self.static_indices)} static  |  "
            f"{len(self.dynamic_indices)} dynamic  |  "
            f"grid {image_size[0]}×{image_size[1]}"
        )

    # ── dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data_loader)

    def __getitem__(self, idx: int) -> Tuple[Tuple, Dict, Dict]:
        """
        Return ``((static, dynamic), targets_dict, metadata_tensor)`` for *idx*.

        Shapes
        ------
        static   : (3,  H, W) float tensor
        dynamic  : (16, H, W) float tensor
        targets  : {var_name: (1, H, W) float tensor}
        metadata : {'lead': LongTensor, 'number_idx': LongTensor,
                    'time_idx': LongTensor}
        """
        # ── raw data from loader ──────────────────────────────────────────────
        inputs, targets = self.data_loader[idx]
        combo_info = self.data_loader.get_combination_info(idx)

        # ── reshape (19, H*W) → (19, H, W) ───────────────────────────────────
        H, W = self.image_size
        inputs = inputs.reshape(inputs.shape[0], H, W).copy()  # (19, H, W)

        # ── split static / dynamic ────────────────────────────────────────────
        static  = inputs[self.static_indices]   # (3,  H, W)
        dynamic = inputs[self.dynamic_indices]  # (16, H, W)

        # ── normalisation ─────────────────────────────────────────────────────
        if self.normalize:
            # Static channels: dem, rho, phi
            for i, var_name in enumerate(self._static_var_names):
                if var_name in self.data_loader.scalers:
                    static[i] = self.data_loader.normalize(
                        static[i], var_name, fit=False
                    )

            # Dynamic channels 0-5: tp, t2m, tmax, hurs, sin_time, cos_time
            for i, var_name in enumerate(self._dynamic_var_names):
                if var_name in self.data_loader.scalers:
                    dynamic[i] = self.data_loader.normalize(
                        dynamic[i], var_name, fit=False
                    )

            # Dynamic channels 6-15: cci_agg (10 classes, shared scaler)
            if "cci_agg" in self.data_loader.scalers:
                for i in range(6, 16):
                    dynamic[i] = self.data_loader.normalize(
                        dynamic[i], "cci_agg", fit=False
                    )

        # ── targets ───────────────────────────────────────────────────────────
        targets_dict: Dict[str, torch.Tensor] = {}
        for i, var in enumerate(self.target_vars):
            target = targets[i].reshape(H, W)   # (H, W)

            if self.normalize and var in self.data_loader.scalers:
                target = self.data_loader.normalize(target, var, fit=False)

            targets_dict[var] = torch.from_numpy(target).unsqueeze(0).float()  # (1, H, W)

        # ── tensors ───────────────────────────────────────────────────────────
        static  = torch.from_numpy(static).float()   # (3,  H, W)
        dynamic = torch.from_numpy(dynamic).float()  # (16, H, W)

        # ── metadata ──────────────────────────────────────────────────────────
        # Use 'lead' as the key so the model / trainer code works unchanged.
        metadata_tensor = {
            "lead":       torch.tensor(combo_info["lead_month"], dtype=torch.long),
            "number_idx": torch.tensor(combo_info["number_idx"], dtype=torch.long),
            "time_idx":   torch.tensor(combo_info["time_idx"], dtype=torch.long),
        }

        return (static, dynamic), targets_dict, metadata_tensor
