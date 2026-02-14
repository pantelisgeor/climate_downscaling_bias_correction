"""
Data-driven loss functions for climate predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MSELoss(nn.Module):
    """Mean Squared Error loss."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MSE loss.

        Args:
            pred: Predictions [batch, 1, H, W]
            target: Targets [batch, 1, H, W]
            mask: Optional mask [batch, 1, H, W] (1 = valid, 0 = ignore)

        Returns:
            Loss value (scalar)
        """
        loss = (pred - target) ** 2

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        else:
            return loss.mean()


class MAELoss(nn.Module):
    """Mean Absolute Error loss."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MAE loss.

        Args:
            pred: Predictions [batch, 1, H, W]
            target: Targets [batch, 1, H, W]
            mask: Optional mask [batch, 1, H, W]

        Returns:
            Loss value (scalar)
        """
        loss = torch.abs(pred - target)

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        else:
            return loss.mean()


class HybridLoss(nn.Module):
    """
    Hybrid MSE-MAE loss.

    Useful for precipitation which has outliers.
    """

    def __init__(self, alpha: float = 0.7):
        """
        Initialize hybrid loss.

        Args:
            alpha: Weight for MSE (1-alpha for MAE)
        """
        super().__init__()
        self.alpha = alpha
        self.mse = MSELoss()
        self.mae = MAELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute hybrid loss."""
        mse_loss = self.mse(pred, target, mask)
        mae_loss = self.mae(pred, target, mask)

        return self.alpha * mse_loss + (1 - self.alpha) * mae_loss


class QuantileLoss(nn.Module):
    """
    Quantile loss for distributional predictions.
    """

    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        """
        Initialize quantile loss.

        Args:
            quantiles: List of quantile values to predict
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, quantile: float
    ) -> torch.Tensor:
        """
        Compute quantile loss for a specific quantile.

        Args:
            pred: Predictions [batch, 1, H, W]
            target: Targets [batch, 1, H, W]
            quantile: Quantile value (0-1)

        Returns:
            Loss value (scalar)
        """
        error = target - pred
        loss = torch.max(quantile * error, (quantile - 1) * error)
        return loss.mean()


class MultiVariableDataLoss(nn.Module):
    """
    Combined data loss for all target variables.
    """

    def __init__(
        self,
        target_vars: list = ["tasERA", "tasmaxERA", "tpERA", "rhERA"],
        loss_types: Dict[str, str] = None,
    ):
        """
        Initialize multi-variable data loss.

        Args:
            target_vars: List of target variable names
            loss_types: Dictionary mapping variable names to loss types
                       Options: 'mse', 'mae', 'hybrid'
        """
        super().__init__()

        self.target_vars = target_vars

        # Default loss types
        if loss_types is None:
            loss_types = {
                "tasERA": "mse",
                "tasmaxERA": "mse",
                "tpERA": "hybrid",  # Hybrid for precipitation
                "rhERA": "mse",
            }

        # Initialize loss functions for each variable
        self.loss_functions = nn.ModuleDict()
        for var in target_vars:
            loss_type = loss_types.get(var, "mse")

            if loss_type == "mse":
                self.loss_functions[var] = MSELoss()
            elif loss_type == "mae":
                self.loss_functions[var] = MAELoss()
            elif loss_type == "hybrid":
                self.loss_functions[var] = HybridLoss(alpha=0.7)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute data losses for all variables.

        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            masks: Optional dictionary of masks

        Returns:
            Dictionary of individual losses
        """
        losses = {}

        for var in self.target_vars:
            if var in predictions and var in targets:
                mask = masks.get(var) if masks else None

                loss = self.loss_functions[var](
                    pred=predictions[var], target=targets[var], mask=mask
                )

                losses[var] = loss

        return losses
