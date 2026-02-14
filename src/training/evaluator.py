"""
Evaluator for ClimateNet with comprehensive metrics and visualizations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluator for ClimateNet.

    Computes metrics and generates visualizations for model predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda",
        target_vars: List[str] = ["tasERA", "tasmaxERA", "tpERA", "rhERA"],
        results_dir: Path = Path("./results"),
        figures_dir: Path = Path("./figures"),
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained ClimateNet model
            test_loader: Test data loader
            device: Device for evaluation
            target_vars: List of target variable names
            results_dir: Directory to save results
            figures_dir: Directory to save figures
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.target_vars = target_vars
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)

        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Evaluator initialized")
        logger.info(f"  Results directory: {self.results_dir}")
        logger.info(f"  Figures directory: {self.figures_dir}")

    @torch.no_grad()
    def evaluate(self, save_predictions: bool = True) -> Dict:
        """
        Run full evaluation on test set.

        Args:
            save_predictions: Whether to save all predictions

        Returns:
            Dictionary with metrics and optionally predictions
        """
        self.model.eval()

        # Storage for predictions and targets
        all_predictions = {var: [] for var in self.target_vars}
        all_targets = {var: [] for var in self.target_vars}
        all_metadata = []

        logger.info("Running inference on test set...")
        pbar = tqdm(self.test_loader, desc="Evaluating")

        for batch in pbar:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = self.model(inputs)

            # Move to CPU and store
            predictions = predictions.cpu().numpy()  # (batch, n_vars, H, W)
            targets = targets.cpu().numpy()  # (batch, n_vars, H, W)

            # Store predictions and targets for each variable
            for i, var in enumerate(self.target_vars):
                all_predictions[var].append(
                    predictions[:, i : i + 1, :, :]
                )  # Keep channel dim
                all_targets[var].append(targets[:, i : i + 1, :, :])

        # Concatenate all batches
        logger.info("Computing metrics...")
        for var in self.target_vars:
            all_predictions[var] = np.concatenate(
                all_predictions[var], axis=0
            )  # (N, 1, H, W)
            all_targets[var] = np.concatenate(all_targets[var], axis=0)  # (N, 1, H, W)

        # Denormalize predictions and targets
        logger.info("Denormalizing predictions and targets...")
        all_predictions_denorm = {}
        all_targets_denorm = {}

        for var in self.target_vars:
            # Get the dataset's data_loader for denormalization
            dataset = self.test_loader.dataset
            # Handle Subset wrapper
            if hasattr(dataset, "dataset"):
                base_dataset = dataset.dataset
            else:
                base_dataset = dataset

            data_loader = base_dataset.data_loader

            # Denormalize
            pred = all_predictions[var]  # (N, 1, H, W)
            target = all_targets[var]  # (N, 1, H, W)

            # Flatten for denormalization
            pred_flat = pred.reshape(-1)
            target_flat = target.reshape(-1)

            # Denormalize
            pred_denorm = data_loader.denormalize(pred_flat, var)
            target_denorm = data_loader.denormalize(target_flat, var)

            # Note: denormalize() in data_loader.py now handles inverse log transform for tpERA
            # So pred_denorm and target_denorm are already in original scale

            # Reshape back
            pred_denorm = pred_denorm.reshape(pred.shape)
            target_denorm = target_denorm.reshape(target.shape)

            all_predictions_denorm[var] = pred_denorm
            all_targets_denorm[var] = target_denorm

            logger.info(f"  {var}: denormalized")
            logger.info(
                f"    Prediction range: [{pred_denorm.min():.4f}, {pred_denorm.max():.4f}]"
            )
            logger.info(
                f"    Target range: [{target_denorm.min():.4f}, {target_denorm.max():.4f}]"
            )

        # Compute metrics on denormalized data
        metrics = self.compute_metrics(all_predictions_denorm, all_targets_denorm)

        # Compute per-lead metrics if metadata is available
        per_lead_metrics = {}

        # Save metrics
        metrics_path = self.results_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"overall": metrics, "per_lead": per_lead_metrics}, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")

        # Generate visualizations
        logger.info("Generating visualizations...")
        self.create_visualizations(
            all_predictions_denorm, all_targets_denorm, all_metadata, metrics
        )

        return {
            "metrics": metrics,
            "per_lead_metrics": per_lead_metrics,
            "predictions": all_predictions_denorm if save_predictions else None,
            "targets": all_targets_denorm if save_predictions else None,
        }

    def compute_metrics(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Compute evaluation metrics for all variables.

        Args:
            predictions: Dictionary of predictions (N, 1, H, W)
            targets: Dictionary of targets (N, 1, H, W)

        Returns:
            Dictionary of metrics for each variable
        """
        metrics = {}

        for var in self.target_vars:
            pred = predictions[var].flatten()
            target = targets[var].flatten()

            # Remove NaN values
            mask = ~(np.isnan(pred) | np.isnan(target))
            pred = pred[mask]
            target = target[mask]

            if len(pred) == 0:
                logger.warning(f"No valid predictions for {var}")
                metrics[var] = {
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "bias": float("nan"),
                    "r2": float("nan"),
                }
                continue

            # Compute metrics
            rmse = np.sqrt(np.mean((pred - target) ** 2))
            mae = np.mean(np.abs(pred - target))
            bias = np.mean(pred - target)

            # R² score
            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))

            metrics[var] = {
                "rmse": float(rmse),
                "mae": float(mae),
                "bias": float(bias),
                "r2": float(r2),
            }

            logger.info(f"Metrics for {var}:")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE:  {mae:.4f}")
            logger.info(f"  Bias: {bias:.4f}")
            logger.info(f"  R²:   {r2:.4f}")

        return metrics

    def compute_per_lead_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        metadata: Dict[str, np.ndarray],
    ) -> Dict:
        """
        Compute metrics separately for each lead time.

        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            metadata: Dictionary with 'lead' array

        Returns:
            Dictionary of metrics per lead time
        """
        if "lead" not in metadata:
            logger.warning("No lead information in metadata, skipping per-lead metrics")
            return {}

        leads = metadata["lead"]
        unique_leads = np.unique(leads)

        per_lead_metrics = {}

        for lead in unique_leads:
            lead_key = f"lead_{int(lead)}"
            per_lead_metrics[lead_key] = {}

            # Get indices for this lead
            lead_mask = leads == lead

            for var in self.target_vars:
                pred = predictions[var][lead_mask].flatten()
                target = targets[var][lead_mask].flatten()

                # Remove NaN
                mask = ~(np.isnan(pred) | np.isnan(target))
                pred = pred[mask]
                target = target[mask]

                if len(pred) == 0:
                    continue

                # Compute metrics
                rmse = np.sqrt(np.mean((pred - target) ** 2))
                mae = np.mean(np.abs(pred - target))
                bias = np.mean(pred - target)

                per_lead_metrics[lead_key][var] = {
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "bias": float(bias),
                }

        return per_lead_metrics

    def create_visualizations(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        metadata: Dict[str, np.ndarray],
        metrics: Dict,
    ):
        """
        Create visualization plots.

        Args:
            predictions: Predictions dictionary
            targets: Targets dictionary
            metadata: Metadata dictionary
            metrics: Computed metrics
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style("whitegrid")
        except ImportError:
            logger.warning("Matplotlib/Seaborn not installed, skipping visualizations")
            return

        logger.info("Generating visualizations...")

        # 1. Target vs Prediction scatter plots
        logger.info("Creating scatter plots...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, var in enumerate(self.target_vars):
            ax = axes[idx]

            pred = predictions[var].flatten()
            target = targets[var].flatten()

            # Remove NaN
            mask = ~(np.isnan(pred) | np.isnan(target))
            pred = pred[mask]
            target = target[mask]

            if len(pred) == 0:
                continue

            # Downsample for plotting if too many points
            if len(pred) > 10000:
                sample_idx = np.random.choice(len(pred), 10000, replace=False)
                pred = pred[sample_idx]
                target = target[sample_idx]

            # Scatter plot
            ax.scatter(target, pred, alpha=0.3, s=1, c="steelblue")

            # Perfect prediction line
            min_val = min(target.min(), pred.min())
            max_val = max(target.max(), pred.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=2,
                label="Perfect prediction",
            )

            # Add metrics text
            r2 = metrics[var]["r2"]
            rmse = metrics[var]["rmse"]
            bias = metrics[var]["bias"]

            text_str = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:.3f}"
            ax.text(
                0.05,
                0.95,
                text_str,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            ax.set_xlabel("Target", fontsize=12)
            ax.set_ylabel("Prediction", fontsize=12)
            ax.set_title(f"{var} - Target vs Prediction", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "scatter_plots.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Error distributions
        logger.info("Creating error distribution plots...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, var in enumerate(self.target_vars):
            ax = axes[idx]

            pred = predictions[var].flatten()
            target = targets[var].flatten()

            # Remove NaN
            mask = ~(np.isnan(pred) | np.isnan(target))
            pred = pred[mask]
            target = target[mask]

            errors = pred - target

            ax.hist(errors, bins=50, alpha=0.7, edgecolor="black")
            ax.axvline(0, color="r", linestyle="--", linewidth=2, label="Zero error")
            ax.axvline(
                metrics[var]["bias"],
                color="g",
                linestyle="--",
                linewidth=2,
                label=f"Mean bias={metrics[var]['bias']:.3f}",
            )

            ax.set_xlabel("Prediction Error", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(f"{var} - Error Distribution", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "error_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 3. Metrics comparison bar plot
        logger.info("Creating metrics comparison plot...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metric_names = ["rmse", "mae", "bias", "r2"]
        metric_titles = ["RMSE", "MAE", "Bias", "R²"]

        for idx, (metric_name, metric_title) in enumerate(
            zip(metric_names, metric_titles)
        ):
            ax = axes[idx // 2, idx % 2]

            values = [metrics[var][metric_name] for var in self.target_vars]

            bars = ax.bar(self.target_vars, values, alpha=0.7, edgecolor="black")

            # Color bars
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_ylabel(metric_title, fontsize=12)
            ax.set_title(f"{metric_title} by Variable", fontsize=12)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 4. Spatial predictions (sample maps)
        logger.info("Creating spatial prediction maps...")

        # Pick a random sample
        sample_idx = np.random.randint(0, predictions[self.target_vars[0]].shape[0])

        fig, axes = plt.subplots(
            len(self.target_vars), 3, figsize=(15, 4 * len(self.target_vars))
        )

        for var_idx, var in enumerate(self.target_vars):
            # Get sample
            target_map = targets[var][sample_idx, 0, :, :]  # (H, W)
            pred_map = predictions[var][sample_idx, 0, :, :]  # (H, W)
            error_map = pred_map - target_map

            # Determine colormap limits
            vmin = min(target_map.min(), pred_map.min())
            vmax = max(target_map.max(), pred_map.max())

            # Target
            im1 = axes[var_idx, 0].imshow(
                target_map, cmap="viridis", vmin=vmin, vmax=vmax
            )
            axes[var_idx, 0].set_title(f"{var} - Target")
            axes[var_idx, 0].axis("off")
            plt.colorbar(im1, ax=axes[var_idx, 0], fraction=0.046, pad=0.04)

            # Prediction
            im2 = axes[var_idx, 1].imshow(
                pred_map, cmap="viridis", vmin=vmin, vmax=vmax
            )
            axes[var_idx, 1].set_title(f"{var} - Prediction")
            axes[var_idx, 1].axis("off")
            plt.colorbar(im2, ax=axes[var_idx, 1], fraction=0.046, pad=0.04)

            # Error
            error_vmax = max(abs(error_map.min()), abs(error_map.max()))
            im3 = axes[var_idx, 2].imshow(
                error_map, cmap="RdBu_r", vmin=-error_vmax, vmax=error_vmax
            )
            axes[var_idx, 2].set_title(f"{var} - Error")
            axes[var_idx, 2].axis("off")
            plt.colorbar(im3, ax=axes[var_idx, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "spatial_predictions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(f"Visualizations saved to {self.figures_dir}")
