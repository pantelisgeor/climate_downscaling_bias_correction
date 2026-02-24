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
        """Compute wet-day-weighted hybrid loss.

        The key design choice: compute the MEAN loss *within* each regime
        (wet / dry) independently, then combine with the regime weights.
        This ensures wet_weight truly scales the average wet-pixel loss
        relative to the average dry-pixel loss, regardless of the wet/dry
        pixel ratio (which is ~15/85 for Cyprus — .mean() over all pixels
        would otherwise dilute the wet signal to near-insignificance).
        """
        # Pixel-wise hybrid loss (before any regime weighting)
        per_pixel_mse = (pred - target) ** 2
        per_pixel_mae = torch.abs(pred - target)
        per_pixel = self.alpha * per_pixel_mse + (1.0 - self.alpha) * per_pixel_mae

        # Apply spatial mask if provided
        if mask is not None:
            per_pixel = per_pixel * mask

        # Regime masks
        wet_mask = (target > self.wet_threshold).float()
        dry_mask = 1.0 - wet_mask

        if mask is not None:
            wet_mask = wet_mask * mask
            dry_mask = dry_mask * mask

        # Per-regime means (clamp denominator to avoid div-by-zero on all-dry batches)
        n_wet = wet_mask.sum().clamp(min=1.0)
        n_dry = dry_mask.sum().clamp(min=1.0)

        wet_loss = (per_pixel * wet_mask).sum() / n_wet
        dry_loss = (per_pixel * dry_mask).sum() / n_dry

        return self.wet_weight * wet_loss + self.dry_weight * dry_loss


class CompositeExtremeDryLoss(nn.Module):
    """
    Focal MAE with an asymmetric dry-day penalty for precipitation.

    Addresses two failure modes of WetDayHybridLoss:
      1. MSE-dominated losses minimise E[Y|X] (conditional mean) which
         systematically under-predicts heavy-tail events.
      2. Symmetric penalties give no special treatment to extreme under-
         predictions, where the gradient signal matters most.

    This loss:
      • Uses MAE as the base (avoids mean-seeking MSE bias).
      • Applies a focal weight  err^gamma_precip  to amplify gradient on
        large errors (hardest cases dominate the learning signal).
      • Adds an asymmetric dry penalty: over-predictions on actually-dry
        pixels are penalised more heavily, while under-predictions on dry
        pixels are ignored (they are already near zero).

    Integration note: designed to be called on a single-channel tpERA slice
    [B, 1, H, W] via MultiVariableDataLoss, so set precip_channel=0.
    """

    def __init__(
        self,
        gamma_precip: float = 1.0,
        overestimate_weight: float = 2.5,
        dry_threshold: float = 0.002,
        lambda_extreme: float = 1.0,
        lambda_dry: float = 0.7,
        precip_channel: int = 0,
        precip_min: float = 0.0,
        precip_max: float = 1.0,
    ):
        """
        Args:
            gamma_precip: Focal exponent for precipitation — amplifies gradient
                          on hard (large-error) examples. 0 = plain MAE,
                          1 = error^2 (quadratic amplification), 2 = cubic.
            overestimate_weight: Extra multiplier applied when the model
                                 over-predicts a dry pixel (false drizzle).
            dry_threshold: Threshold in denormalised space below which a pixel
                           is considered dry (default 0.002 ≈ 0 mm/day in
                           log1p-minmax-normalised [0, 1] space).
            lambda_extreme: Weight of the focal MAE term.
            lambda_dry: Weight of the dry asymmetric penalty term.
            precip_channel: Channel index for precipitation inside the input
                            tensor (0 when called per-variable).
            precip_min / precip_max: Normalisation bounds used to reverse the
                                     minmax step before applying dry_threshold.
        """
        super().__init__()
        self.gamma_precip = gamma_precip
        self.overestimate_weight = overestimate_weight
        self.dry_threshold = dry_threshold
        self.lambda_extreme = lambda_extreme
        self.lambda_dry = lambda_dry
        self.precip_channel = precip_channel
        self.precip_min = precip_min
        self.precip_max = precip_max

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.precip_max - self.precip_min) + self.precip_min

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_error = torch.abs(pred - target)          # [B, 1, H, W]

        # ── Focal term ────────────────────────────────────────────────────────
        # focal_weight is detached so it only rescales the gradient magnitude,
        # not introduce second-order effects into the backward graph.
        focal_weight = torch.pow(base_error.detach() + 1e-8, self.gamma_precip)
        focal_loss = focal_weight * base_error         # [B, 1, H, W]

        # ── Dry asymmetric penalty ────────────────────────────────────────────
        p = pred[:, self.precip_channel, :, :]         # [B, H, W]
        t = target[:, self.precip_channel, :, :]
        e = base_error[:, self.precip_channel, :, :]

        dry_mask = self._denorm(t) < self.dry_threshold     # over_mask = pred > target on dry day
        over_mask = p > t
        penalty_mask = (dry_mask & over_mask).float()

        dry_penalty = penalty_mask * self.overestimate_weight * e  # [B, H, W]

        # Embed back into a [B, 1, H, W] container so shapes stay consistent
        dry_loss = torch.zeros_like(base_error)
        dry_loss[:, self.precip_channel, :, :] = dry_penalty

        total_loss = self.lambda_extreme * focal_loss + self.lambda_dry * dry_loss

        if mask is not None:
            total_loss = total_loss * mask
            return total_loss.sum() / (mask.sum().clamp(min=1.0))

        return total_loss.mean()


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
        # compositedry params
        compositedry_gamma: float = 1.0,
        compositedry_overestimate_weight: float = 2.5,
        compositedry_dry_threshold: float = 0.002,
        compositedry_lambda_extreme: float = 1.0,
        compositedry_lambda_dry: float = 0.7,
    ):
        """
        Initialize multi-variable data loss.

        Args:
            target_vars: List of target variable names
            loss_types: Dictionary mapping variable names to loss types
                       Options: 'mse', 'mae', 'hybrid', 'wethybrid', 'tweedie',
                                'compositedry'
                       Use 'wethybrid' or 'compositedry' for precipitation.
            tweedie_power: Tweedie power parameter p for tweedie loss (1<p<2)
            tweedie_eps: Minimum clamp for prediction mean mu in tweedie loss
            wet_weight: Up-weight multiplier for wet (non-zero target) pixels
                        when using 'wethybrid' loss (default 5).
            dry_weight: Weight for dry (zero target) pixels in 'wethybrid' (default 1).
            compositedry_gamma: Focal exponent for 'compositedry'. 0=plain MAE,
                                1=quadratic amplification, 2=cubic.
            compositedry_overestimate_weight: Multiplier on dry-day over-predictions.
            compositedry_dry_threshold: Denormalised threshold below which a pixel
                                        is dry (default 0.002 ≈ ~0 mm in log1p space).
            compositedry_lambda_extreme: Weight of focal MAE term.
            compositedry_lambda_dry: Weight of dry asymmetric penalty term.
        """
        super().__init__()

        self.target_vars = target_vars
        # Store compositedry params for use in the loop below
        self._compositedry_kwargs = dict(
            gamma_precip=compositedry_gamma,
            overestimate_weight=compositedry_overestimate_weight,
            dry_threshold=compositedry_dry_threshold,
            lambda_extreme=compositedry_lambda_extreme,
            lambda_dry=compositedry_lambda_dry,
        )

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
            elif loss_type == "compositedry":
                self.loss_functions[var] = CompositeExtremeDryLoss(
                    **self._compositedry_kwargs,
                    precip_channel=0,   # always 0: each var gets [B,1,H,W]
                    precip_min=0.0,
                    precip_max=1.0,
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
