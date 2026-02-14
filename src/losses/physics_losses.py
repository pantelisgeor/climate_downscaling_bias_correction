"""
Physics-informed loss functions for climate model bias correction.

Implements constraints based on thermodynamic relationships and physical bounds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ClauisiusClapeyronLoss(nn.Module):
    """
    Clausius-Clapeyron constraint for temperature-humidity relationship.

    Simplified version that checks if the relationship holds without computing gradients.
    """

    def __init__(
        self, L_v: float = 2.5e6, R_v: float = 461.0, T_offset: float = 273.15
    ):
        super().__init__()

        self.L_v = L_v
        self.R_v = R_v
        self.T_offset = T_offset

    def compute_saturation_vapor_pressure(
        self, T_celsius: torch.Tensor
    ) -> torch.Tensor:
        """Compute saturation vapor pressure using Magnus formula."""
        numerator = 17.67 * T_celsius
        denominator = T_celsius + 243.5
        e_s = 6.112 * torch.exp(numerator / denominator)
        return e_s

    def forward(
        self,
        T_pred: torch.Tensor,
        RH_pred: torch.Tensor,
        T_target: Optional[torch.Tensor] = None,
        RH_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Clausius-Clapeyron constraint using relative humidity consistency.

        Instead of computing gradients, we check if the predicted temperature
        and humidity are consistent with the CC relationship.
        """
        # Compute saturation vapor pressure at predicted temperature
        e_s_pred = self.compute_saturation_vapor_pressure(T_pred)

        # Compute actual vapor pressure from predicted RH
        e_pred = (RH_pred / 100.0) * e_s_pred

        # If we have targets, compute consistency
        if T_target is not None and RH_target is not None:
            # Compute what RH should be given the predicted temperature
            # and target vapor pressure
            e_s_at_pred_T = self.compute_saturation_vapor_pressure(T_pred)
            e_s_at_target_T = self.compute_saturation_vapor_pressure(T_target)
            e_target = (RH_target / 100.0) * e_s_at_target_T

            # Expected RH at predicted temperature
            RH_expected = 100.0 * e_target / e_s_at_pred_T

            # Loss: difference between predicted and expected RH
            loss = F.mse_loss(RH_pred, RH_expected.clamp(0, 100))
        else:
            # Simple constraint: check temperature-RH relationship
            # Warmer air should generally have lower RH for same absolute humidity
            # This is a soft constraint

            # Compute the consistency metric
            # Higher temperature should correlate with higher saturation capacity
            T_norm = (T_pred - T_pred.mean()) / (T_pred.std() + 1e-6)
            e_s_norm = (e_s_pred - e_s_pred.mean()) / (e_s_pred.std() + 1e-6)

            # They should be highly correlated
            correlation_loss = 1.0 - F.cosine_similarity(
                T_norm.flatten(), e_s_norm.flatten(), dim=0
            )

            loss = correlation_loss.abs()

        return loss


class TemperatureConsistencyLoss(nn.Module):
    """
    Ensure maximum temperature >= mean temperature.

    Physical constraint: T_max >= T_mean
    """

    def __init__(self, margin: float = 0.0):
        """
        Initialize temperature consistency loss.

        Args:
            margin: Minimum difference T_max - T_mean (default 0)
        """
        super().__init__()
        self.margin = margin

    def forward(self, T_mean: torch.Tensor, T_max: torch.Tensor) -> torch.Tensor:
        """
        Compute temperature consistency loss.

        Args:
            T_mean: Mean temperature [batch, 1, H, W]
            T_max: Maximum temperature [batch, 1, H, W]

        Returns:
            Loss value (scalar)
        """
        # Penalize cases where T_mean > T_max - margin
        violation = F.relu(T_mean - T_max + self.margin)
        loss = (violation**2).mean()

        return loss


class HumidityBoundsLoss(nn.Module):
    """
    Ensure relative humidity stays within physical bounds [0, 100]%.
    """

    def __init__(self, soft_margin: float = 5.0):
        """
        Initialize humidity bounds loss.

        Args:
            soft_margin: Soft margin beyond bounds (allows small violations)
        """
        super().__init__()
        self.soft_margin = soft_margin

    def forward(self, RH_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute humidity bounds violation loss.

        Args:
            RH_pred: Predicted relative humidity in % [batch, 1, H, W]

        Returns:
            Loss value (scalar)
        """
        # Penalize values below 0 - margin
        lower_violation = F.relu(-self.soft_margin - RH_pred)

        # Penalize values above 100 + margin
        upper_violation = F.relu(RH_pred - (100.0 + self.soft_margin))

        loss = (lower_violation**2).mean() + (upper_violation**2).mean()

        return loss


class PrecipitationNonNegativityLoss(nn.Module):
    """
    Ensure precipitation is non-negative.
    """

    def __init__(self):
        super().__init__()

    def forward(self, precip_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute precipitation non-negativity loss.

        Args:
            precip_pred: Predicted precipitation [batch, 1, H, W]

        Returns:
            Loss value (scalar)
        """
        # Penalize negative values
        violation = F.relu(-precip_pred)
        loss = (violation**2).mean()

        return loss


class SpatialSmoothnessLoss(nn.Module):
    """
    Encourage spatial smoothness in predictions.

    Useful for preventing unrealistic spatial discontinuities.
    """

    def __init__(self, order: int = 1):
        """
        Initialize spatial smoothness loss.

        Args:
            order: Order of spatial derivative (1 or 2)
        """
        super().__init__()
        self.order = order

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial smoothness loss.

        Args:
            pred: Predictions [batch, 1, H, W]

        Returns:
            Loss value (scalar)
        """
        # Compute gradients
        grad_h = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        grad_w = pred[:, :, :, 1:] - pred[:, :, :, :-1]

        if self.order == 1:
            loss = (grad_h**2).mean() + (grad_w**2).mean()
        else:  # Second order
            grad_hh = grad_h[:, :, 1:, :] - grad_h[:, :, :-1, :]
            grad_ww = grad_w[:, :, :, 1:] - grad_w[:, :, :, :-1]
            loss = (grad_hh**2).mean() + (grad_ww**2).mean()

        return loss


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss with configurable components.
    """

    def __init__(
        self,
        use_clausius_clapeyron: bool = True,
        use_temp_consistency: bool = True,
        use_humidity_bounds: bool = True,
        use_precip_nonnegativity: bool = True,
        use_spatial_smoothness: bool = False,
        # Loss weights
        cc_weight: float = 0.1,
        temp_consistency_weight: float = 0.5,
        humidity_bounds_weight: float = 0.3,
        precip_nonneg_weight: float = 0.2,
        spatial_smooth_weight: float = 0.01,
    ):
        """
        Initialize combined physics loss.

        Args:
            use_*: Flags to enable/disable specific constraints
            *_weight: Weights for each constraint
        """
        super().__init__()

        self.use_clausius_clapeyron = use_clausius_clapeyron
        self.use_temp_consistency = use_temp_consistency
        self.use_humidity_bounds = use_humidity_bounds
        self.use_precip_nonnegativity = use_precip_nonnegativity
        self.use_spatial_smoothness = use_spatial_smoothness

        # Initialize loss components
        if use_clausius_clapeyron:
            self.cc_loss = ClauisiusClapeyronLoss()
            self.cc_weight = cc_weight

        if use_temp_consistency:
            self.temp_consistency_loss = TemperatureConsistencyLoss()
            self.temp_consistency_weight = temp_consistency_weight

        if use_humidity_bounds:
            self.humidity_bounds_loss = HumidityBoundsLoss()
            self.humidity_bounds_weight = humidity_bounds_weight

        if use_precip_nonnegativity:
            self.precip_nonneg_loss = PrecipitationNonNegativityLoss()
            self.precip_nonneg_weight = precip_nonneg_weight

        if use_spatial_smoothness:
            self.spatial_smooth_loss = SpatialSmoothnessLoss()
            self.spatial_smooth_weight = spatial_smooth_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all physics-informed losses.

        Args:
            predictions: Dictionary of predictions for each variable
                - 'tasERA': mean temperature [batch, 1, H, W]
                - 'tasmaxERA': maximum temperature [batch, 1, H, W]
                - 'tpERA': total precipitation [batch, 1, H, W]
                - 'rhERA': relative humidity [batch, 1, H, W]
            targets: Dictionary of target values (optional)

        Returns:
            Dictionary of individual losses and total loss
        """
        losses = {}
        total_loss = 0.0

        # Clausius-Clapeyron constraint
        if (
            self.use_clausius_clapeyron
            and "tasERA" in predictions
            and "rhERA" in predictions
        ):
            cc_loss = self.cc_loss(
                T_pred=predictions["tasERA"],
                RH_pred=predictions["rhERA"],
                T_target=targets.get("tasERA") if targets else None,
                RH_target=targets.get("rhERA") if targets else None,
            )
            losses["clausius_clapeyron"] = cc_loss
            total_loss += self.cc_weight * cc_loss

        # Temperature consistency
        if (
            self.use_temp_consistency
            and "tasERA" in predictions
            and "tasmaxERA" in predictions
        ):
            temp_consistency = self.temp_consistency_loss(
                T_mean=predictions["tasERA"], T_max=predictions["tasmaxERA"]
            )
            losses["temp_consistency"] = temp_consistency
            total_loss += self.temp_consistency_weight * temp_consistency

        # Humidity bounds
        if self.use_humidity_bounds and "rhERA" in predictions:
            humidity_bounds = self.humidity_bounds_loss(predictions["rhERA"])
            losses["humidity_bounds"] = humidity_bounds
            total_loss += self.humidity_bounds_weight * humidity_bounds

        # Precipitation non-negativity
        if self.use_precip_nonnegativity and "tpERA" in predictions:
            precip_nonneg = self.precip_nonneg_loss(predictions["tpERA"])
            losses["precip_nonnegativity"] = precip_nonneg
            total_loss += self.precip_nonneg_weight * precip_nonneg

        # Spatial smoothness (applied to all variables)
        if self.use_spatial_smoothness:
            smooth_loss = 0.0
            for var_name, pred in predictions.items():
                smooth_loss += self.spatial_smooth_loss(pred)
            smooth_loss /= len(predictions)
            losses["spatial_smoothness"] = smooth_loss
            total_loss += self.spatial_smooth_weight * smooth_loss

        losses["total_physics"] = total_loss

        return losses
