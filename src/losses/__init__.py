"""
Loss functions module.
"""

from .data_losses import MSELoss, MAELoss, HybridLoss, MultiVariableDataLoss
from .physics_losses import (
    ClauisiusClapeyronLoss,
    TemperatureConsistencyLoss,
    HumidityBoundsLoss,
    PrecipitationNonNegativityLoss,
    SpatialSmoothnessLoss,
    PhysicsInformedLoss
)
from .task_weighting import (
    UncertaintyWeighting,
    DynamicWeightAverage,
    GradientNormalization,
    CombinedLoss
)

__all__ = [
    'MSELoss',
    'MAELoss',
    'HybridLoss',
    'MultiVariableDataLoss',
    'ClauisiusClapeyronLoss',
    'TemperatureConsistencyLoss',
    'HumidityBoundsLoss',
    'PrecipitationNonNegativityLoss',
    'SpatialSmoothnessLoss',
    'PhysicsInformedLoss',
    'UncertaintyWeighting',
    'DynamicWeightAverage',
    'GradientNormalization',
    'CombinedLoss'
]
