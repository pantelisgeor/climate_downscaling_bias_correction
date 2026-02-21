"""
Loss functions module.
"""

from .data_losses import MSELoss, MAELoss, HybridLoss, WetDayHybridLoss, MultiVariableDataLoss
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
    'WetDayHybridLoss',
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
