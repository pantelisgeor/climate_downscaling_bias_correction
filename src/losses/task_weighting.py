"""
Task weighting strategies for multi-task learning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np


class UncertaintyWeighting(nn.Module):
    """
    Uncertainty-based task weighting (Kendall et al., 2018).

    Automatically learns task weights based on homoscedastic uncertainty.

    Loss = sum_k [1/(2*sigma_k^2) * L_k + log(sigma_k)]

    where sigma_k is a learnable parameter representing task uncertainty.
    """

    def __init__(
        self, task_names: List[str], init_log_vars: Optional[Dict[str, float]] = None
    ):
        """
        Initialize uncertainty weighting.

        Args:
            task_names: List of task names
            init_log_vars: Initial log variance for each task (optional)
        """
        super().__init__()

        self.task_names = task_names
        self.num_tasks = len(task_names)

        # Learnable log variance parameters (one per task)
        # Using log variance for numerical stability
        log_vars = []
        for task in task_names:
            if init_log_vars and task in init_log_vars:
                init_val = init_log_vars[task]
            else:
                init_val = 0.0  # corresponds to sigma = 1

            log_vars.append(init_val)

        self.log_vars = nn.Parameter(torch.tensor(log_vars, dtype=torch.float32))

    def forward(self, task_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute weighted losses.

        Args:
            task_losses: Dictionary mapping task names to loss values

        Returns:
            Dictionary with weighted losses and total loss
        """
        weighted_losses = {}
        total_loss = 0.0

        for i, task in enumerate(self.task_names):
            if task in task_losses:
                loss = task_losses[task]
                log_var = self.log_vars[i]

                # Weighted loss: 1/(2*sigma^2) * L + log(sigma)
                # = 1/2 * exp(-log_var) * L + 0.5 * log_var
                precision = torch.exp(-log_var)
                weighted_loss = 0.5 * precision * loss + 0.5 * log_var

                weighted_losses[f"{task}_weighted"] = weighted_loss
                total_loss += weighted_loss

        weighted_losses["total_weighted"] = total_loss

        # Also store the learned uncertainties (sigma = exp(0.5 * log_var))
        uncertainties = {}
        for i, task in enumerate(self.task_names):
            sigma = torch.exp(0.5 * self.log_vars[i])
            uncertainties[task] = sigma.item()

        return weighted_losses, uncertainties

    def get_weights(self) -> Dict[str, float]:
        """
        Get current task weights (1 / sigma^2).

        Returns:
            Dictionary mapping task names to weights
        """
        weights = {}
        for i, task in enumerate(self.task_names):
            # Weight = 1 / sigma^2 = exp(-log_var)
            weight = torch.exp(-self.log_vars[i]).item()
            weights[task] = weight

        return weights


class DynamicWeightAverage(nn.Module):
    """
    Dynamic Weight Average (DWA) for multi-task learning.

    Adjusts task weights based on the rate of change of task losses.
    """

    def __init__(
        self, task_names: List[str], temperature: float = 2.0, window_size: int = 2
    ):
        """
        Initialize DWA.

        Args:
            task_names: List of task names
            temperature: Temperature parameter for softmax
            window_size: Window size for computing loss rate
        """
        super().__init__()

        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.temperature = temperature
        self.window_size = window_size

        # Store loss history
        self.loss_history = {task: [] for task in task_names}

    def update_history(self, task_losses: Dict[str, torch.Tensor]):
        """Update loss history."""
        for task in self.task_names:
            if task in task_losses:
                loss_val = task_losses[task].item()
                self.loss_history[task].append(loss_val)

                # Keep only recent history
                if len(self.loss_history[task]) > self.window_size + 1:
                    self.loss_history[task].pop(0)

    def compute_weights(self) -> Dict[str, float]:
        """
        Compute task weights based on loss rate.

        Returns:
            Dictionary mapping task names to weights
        """
        if any(
            len(history) < self.window_size + 1
            for history in self.loss_history.values()
        ):
            # Not enough history, use uniform weights
            return {task: 1.0 / self.num_tasks for task in self.task_names}

        # Compute loss ratios (current / previous window average)
        loss_ratios = []
        for task in self.task_names:
            history = self.loss_history[task]
            current_loss = history[-1]
            prev_avg = np.mean(history[-self.window_size - 1 : -1])

            ratio = current_loss / (prev_avg + 1e-8)
            loss_ratios.append(ratio)

        # Apply temperature scaling and softmax
        loss_ratios = torch.tensor(loss_ratios, dtype=torch.float32)
        weights_unnorm = torch.exp(loss_ratios / self.temperature)
        weights = self.num_tasks * weights_unnorm / weights_unnorm.sum()

        return {task: weights[i].item() for i, task in enumerate(self.task_names)}

    def forward(self, task_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute weighted losses using DWA.

        Args:
            task_losses: Dictionary mapping task names to loss values

        Returns:
            Dictionary with weighted losses and total loss
        """
        # Update history
        self.update_history(task_losses)

        # Get current weights
        weights = self.compute_weights()

        # Apply weights
        weighted_losses = {}
        total_loss = 0.0

        for task in self.task_names:
            if task in task_losses:
                weight = weights[task]
                weighted_loss = weight * task_losses[task]

                weighted_losses[f"{task}_weighted"] = weighted_loss
                total_loss += weighted_loss

        weighted_losses["total_weighted"] = total_loss

        return weighted_losses, weights


class GradientNormalization(nn.Module):
    """
    GradNorm: Gradient Normalization for multi-task learning.

    Balances task training by normalizing gradient magnitudes.
    """

    def __init__(
        self,
        task_names: List[str],
        alpha: float = 1.5,
        init_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize GradNorm.

        Args:
            task_names: List of task names
            alpha: Restoring force hyperparameter
            init_weights: Initial task weights (optional)
        """
        super().__init__()

        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.alpha = alpha

        # Initialize learnable weights
        if init_weights:
            weights = [init_weights.get(task, 1.0) for task in task_names]
        else:
            weights = [1.0] * self.num_tasks

        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

        # Store initial task losses for normalization
        self.initial_losses = None

    def set_initial_losses(self, task_losses: Dict[str, torch.Tensor]):
        """Set initial task losses for normalization."""
        if self.initial_losses is None:
            self.initial_losses = {
                task: loss.item() for task, loss in task_losses.items()
            }

    def forward(self, task_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute weighted losses using current weights.

        Args:
            task_losses: Dictionary mapping task names to loss values

        Returns:
            Dictionary with weighted losses and total loss
        """
        weighted_losses = {}
        total_loss = 0.0

        for i, task in enumerate(self.task_names):
            if task in task_losses:
                weight = self.weights[i]
                weighted_loss = weight * task_losses[task]

                weighted_losses[f"{task}_weighted"] = weighted_loss
                total_loss += weighted_loss

        weighted_losses["total_weighted"] = total_loss

        return weighted_losses, self.get_weights()

    def get_weights(self) -> Dict[str, float]:
        """Get current task weights (normalized)."""
        # Normalize weights to sum to num_tasks
        weights_norm = self.num_tasks * self.weights / self.weights.sum()
        return {task: weights_norm[i].item() for i, task in enumerate(self.task_names)}


class CombinedLoss(nn.Module):
    """
    Combined loss with data losses, physics losses, and task weighting.
    """

    def __init__(
        self,
        target_vars: List[str] = ["tasERA", "tasmaxERA", "tpERA", "rhERA"],
        weighting_strategy: str = "uncertainty",  # 'uncertainty', 'dwa', 'gradnorm', 'equal'
        physics_weight: float = 0.1,
        use_physics: bool = True,
        **kwargs,
    ):
        """
        Initialize combined loss.

        Args:
            target_vars: List of target variable names
            weighting_strategy: Task weighting strategy
            physics_weight: Weight for physics loss relative to data loss
            use_physics: Whether to use physics-informed losses
            **kwargs: Additional arguments for weighting strategy
        """
        super().__init__()

        from .data_losses import MultiVariableDataLoss
        from .physics_losses import PhysicsInformedLoss

        self.target_vars = target_vars
        self.physics_weight = physics_weight
        self.use_physics = use_physics

        # Data loss
        self.data_loss = MultiVariableDataLoss(target_vars=target_vars)

        # Physics loss
        if use_physics:
            self.physics_loss = PhysicsInformedLoss(**kwargs)

        # Task weighting
        if weighting_strategy == "uncertainty":
            self.task_weighting = UncertaintyWeighting(task_names=target_vars)
        elif weighting_strategy == "dwa":
            self.task_weighting = DynamicWeightAverage(task_names=target_vars)
        elif weighting_strategy == "gradnorm":
            self.task_weighting = GradientNormalization(task_names=target_vars)
        elif weighting_strategy == "equal":
            self.task_weighting = None
        else:
            raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")

        self.weighting_strategy = weighting_strategy

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute total loss.

        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            masks: Optional dictionary of masks
            return_components: If True, return dict with loss components

        Returns:
            Total loss (or dictionary if return_components=True)
        """
        # Compute data losses
        data_losses = self.data_loss(predictions, targets, masks)

        # Apply task weighting
        if self.task_weighting is not None:
            if self.weighting_strategy == "gradnorm":
                # Set initial losses on first call
                self.task_weighting.set_initial_losses(data_losses)

            weighted_data_losses, weights = self.task_weighting(data_losses)
            total_data_loss = weighted_data_losses["total_weighted"]
        else:
            # Equal weighting
            total_data_loss = sum(data_losses.values()) / len(data_losses)
            weights = {task: 1.0 for task in self.target_vars}

        # Compute physics losses
        if self.use_physics:
            physics_losses = self.physics_loss(predictions, targets)
            total_physics_loss = physics_losses["total_physics"]
        else:
            physics_losses = {}
            total_physics_loss = 0.0

        # Total loss
        total_loss = total_data_loss + self.physics_weight * total_physics_loss

        if return_components:
            return {
                "total": total_loss,
                "data_losses": data_losses,
                "physics_losses": physics_losses,
                "total_data": total_data_loss,
                "total_physics": total_physics_loss,
                "task_weights": weights,
            }
        else:
            return total_loss
