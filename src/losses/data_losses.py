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


class TweedieDevianceLoss(nn.Module):
    """
    Tweedie deviance loss for compound Poisson-Gamma targets (1 < p < 2).

    Suitable for precipitation-like variables with a point mass at zero and
    continuous positive tail.
    """

    def __init__(self, power: float = 1.5, eps: float = 1e-6):
        super().__init__()
        if not (1.0 < power < 2.0):
            raise ValueError(f"Tweedie power must satisfy 1 < p < 2, got {power}")
        self.power = float(power)
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Tweedie deviance:
            D(y, mu) = 2 * [ y^(2-p)/((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p) ]

        Args:
            pred: Mean predictions mu [batch, 1, H, W]
            target: Targets y [batch, 1, H, W]
            mask: Optional mask [batch, 1, H, W]

        Returns:
            Scalar loss
        """
        p = self.power

        # mu must be strictly positive because exponent 1-p is negative.
        mu = torch.clamp(pred, min=self.eps)
        y = torch.clamp(target, min=0.0)

        term1 = torch.pow(y, 2.0 - p) / ((1.0 - p) * (2.0 - p))
        term2 = y * torch.pow(mu, 1.0 - p) / (1.0 - p)
        term3 = torch.pow(mu, 2.0 - p) / (2.0 - p)
        loss = 2.0 * (term1 - term2 + term3)

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)

        return loss.mean()


class WetDayHybridLoss(nn.Module):
    """
    Hybrid MSE-MAE loss with separate weighting for wet and dry days.

    Precipitation targets are highly zero-inflated (dry days dominate).
    Plain MSE/MAE therefore receives a gradient signal dominated by pulling
    predictions toward zero.  This loss separates the two regimes:

    - **Dry pixels** (target == 0): penalised with a small weight so the
      network still learns to predict zero for no-rain situations.
    - **Wet pixels** (target > wet_threshold): penalised with a higher weight
      so the network sees meaningful gradient on the interesting cases.

    Args:
        alpha: MSE fraction (1-alpha = MAE fraction), same as HybridLoss.
        wet_weight: Multiplier applied to the loss on wet pixels (default 5).
        dry_weight: Multiplier applied to the loss on dry pixels (default 1).
        wet_threshold: Minimum normalised target value to be considered "wet"
                       (default 0.0, i.e. any positive value after log1p normalization).
    """

    def __init__(
        self,
        alpha: float = 0.7,
        wet_weight: float = 5.0,
        dry_weight: float = 1.0,
        wet_threshold: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.wet_weight = wet_weight
        self.dry_weight = dry_weight
        self.wet_threshold = wet_threshold
        self.mse = MSELoss()
        self.mae = MAELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute wet-day-weighted hybrid loss."""
        # Pixel-wise combined loss before weighting
        per_pixel_mse = (pred - target) ** 2
        per_pixel_mae = torch.abs(pred - target)
        per_pixel = self.alpha * per_pixel_mse + (1.0 - self.alpha) * per_pixel_mae

        # Build weight map: wet pixels get higher weight
        wet_mask = (target > self.wet_threshold).float()
        dry_mask = 1.0 - wet_mask
        weight_map = self.wet_weight * wet_mask + self.dry_weight * dry_mask

        per_pixel = per_pixel * weight_map

        if mask is not None:
            per_pixel = per_pixel * mask
            return per_pixel.sum() / (mask.sum() + 1e-8)

        return per_pixel.mean()


class MultiVariableDataLoss(nn.Module):
    """
    Combined data loss for all target variables.
    """

    def __init__(
        self,
        target_vars: list = ["tasERA", "tasmaxERA", "tpERA", "rhERA"],
        loss_types: Dict[str, str] = None,
        tweedie_power: float = 1.5,
        tweedie_eps: float = 1e-6,
        wet_weight: float = 5.0,
        dry_weight: float = 1.0,
    ):
        """
        Initialize multi-variable data loss.

        Args:
            target_vars: List of target variable names
            loss_types: Dictionary mapping variable names to loss types
                       Options: 'mse', 'mae', 'hybrid', 'wethybrid', 'tweedie'
                       Use 'wethybrid' for precipitation to handle zero-inflation.
            tweedie_power: Tweedie power parameter p for tweedie loss (1<p<2)
            tweedie_eps: Minimum clamp for prediction mean mu in tweedie loss
            wet_weight: Up-weight multiplier for wet (non-zero target) pixels
                        when using 'wethybrid' loss (default 5).
            dry_weight: Weight for dry (zero target) pixels in 'wethybrid' (default 1).
        """
        super().__init__()

        self.target_vars = target_vars

        # Default loss types
        if loss_types is None:
            loss_types = {
                "tasERA": "mse",
                "tasmaxERA": "mse",
                "tpERA": "wethybrid",  # Wet-day-weighted hybrid for precipitation
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
            elif loss_type == "wethybrid":
                self.loss_functions[var] = WetDayHybridLoss(
                    alpha=0.7,
                    wet_weight=wet_weight,
                    dry_weight=dry_weight,
                )
            elif loss_type == "tweedie":
                self.loss_functions[var] = TweedieDevianceLoss(
                    power=tweedie_power, eps=tweedie_eps
                )
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
