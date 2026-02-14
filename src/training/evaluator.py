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
            # Prepare batch
            inputs, targets, metadata = batch
            static, dynamic = inputs

            static = static.to(self.device)
            dynamic = dynamic.to(self.device)
            lead_indices = metadata["lead"].to(self.device)

            # Forward pass
            predictions = self.model(
                static=static, dynamic=dynamic, lead_indices=lead_indices
            )

            # Store predictions and targets
            for var in self.target_vars:
                all_predictions[var].append(predictions[var].cpu().numpy())
                all_targets[var].append(targets[var].cpu().numpy())

            # Store metadata
            all_metadata.append(
                {
                    "lead": metadata["lead"].cpu().numpy(),
                    "run_idx": metadata["run_idx"].cpu().numpy(),
                    "lead_idx": metadata["lead_idx"].cpu().numpy(),
                    "time_idx": metadata["time_idx"].cpu().numpy(),
                }
            )

        # Concatenate all batches
        logger.info("Computing metrics...")
        for var in self.target_vars:
            all_predictions[var] = np.concatenate(all_predictions[var], axis=0)
            all_targets[var] = np.concatenate(all_targets[var], axis=0)

        all_metadata = {
            key: np.concatenate([m[key] for m in all_metadata])
            for key in all_metadata[0].keys()
        }

        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_targets)

        # Compute per-lead metrics
        logger.info("Computing per-lead metrics...")
        per_lead_metrics = self.compute_per_lead_metrics(
            all_predictions, all_targets, all_metadata
        )

        # Save metrics
        metrics_path = self.results_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"overall": metrics, "per_lead": per_lead_metrics}, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")

        # Save predictions if requested
        if save_predictions:
            logger.info("Saving predictions...")
            pred_path = self.results_dir / "predictions.npz"
            np.savez_compressed(
                pred_path,
                **{f"pred_{var}": all_predictions[var] for var in self.target_vars},
                **{f"target_{var}": all_targets[var] for var in self.target_vars},
                **all_metadata,
            )
            logger.info(f"Predictions saved to: {pred_path}")

        # Generate visualizations
        logger.info("Generating visualizations...")
        self.create_visualizations(all_predictions, all_targets, all_metadata, metrics)

        return {
            "metrics": metrics,
            "per_lead_metrics": per_lead_metrics,
            "predictions": all_predictions if save_predictions else None,
            "targets": all_targets if save_predictions else None,
        }

    def compute_metrics(
        self, predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Compute evaluation metrics for all variables.

        Args:
            predictions: Dictionary of predictions [n_samples, 1, H, W]
            targets: Dictionary of targets [n_samples, 1, H, W]

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

            # Compute metrics
            mse = np.mean((pred - target) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred - target))
            bias = np.mean(pred - target)

            # R² score
            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            # Pearson correlation
            corr = np.corrcoef(pred, target)[0, 1]

            # Percentile metrics
            percentile_errors = np.abs(pred - target)
            p50 = np.percentile(percentile_errors, 50)
            p90 = np.percentile(percentile_errors, 90)
            p95 = np.percentile(percentile_errors, 95)

            metrics[var] = {
                "rmse": float(rmse),
                "mae": float(mae),
                "bias": float(bias),
                "r2": float(r2),
                "correlation": float(corr),
                "p50_error": float(p50),
                "p90_error": float(p90),
                "p95_error": float(p95),
                "n_samples": int(len(pred)),
            }

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
            metadata: Dictionary with lead time info

        Returns:
            Dictionary of metrics per lead time
        """
        leads = np.unique(metadata["lead"])
        per_lead_metrics = {}

        for lead in leads:
            lead_mask = metadata["lead"] == lead
            lead_key = f"lead_{int(lead)}"
            per_lead_metrics[lead_key] = {}

            for var in self.target_vars:
                pred = predictions[var][lead_mask].flatten()
                target = targets[var][lead_mask].flatten()

                # Remove NaN
                mask = ~(np.isnan(pred) | np.isnan(target))
                pred = pred[mask]
                target = target[mask]

                if len(pred) > 0:
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

        # 1. Scatter plots: predictions vs targets
        logger.info("Creating scatter plots...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for idx, var in enumerate(self.target_vars):
            ax = axes[idx]

            pred = predictions[var].flatten()
            target = targets[var].flatten()

            # Sample for plotting (max 10000 points)
            if len(pred) > 10000:
                sample_idx = np.random.choice(len(pred), 10000, replace=False)
                pred = pred[sample_idx]
                target = target[sample_idx]

            ax.scatter(target, pred, alpha=0.3, s=1)
            ax.plot(
                [target.min(), target.max()],
                [target.min(), target.max()],
                "r--",
                lw=2,
                label="Perfect prediction",
            )

            ax.set_xlabel("Target", fontsize=12)
            ax.set_ylabel("Prediction", fontsize=12)
            ax.set_title(
                f'{var}\nRMSE={metrics[var]["rmse"]:.3f}, R²={metrics[var]["r2"]:.3f}',
                fontsize=12,
            )
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
            errors = pred - target

            ax.hist(errors, bins=50, alpha=0.7, edgecolor="black")
            ax.axvline(0, color="r", linestyle="--", linewidth=2, label="Zero error")
            ax.axvline(
                metrics[var]["bias"],
                color="g",
                linestyle="--",
                linewidth=2,
                label=f'Mean bias={metrics[var]["bias"]:.3f}',
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

        logger.info(f"Visualizations saved to: {self.figures_dir}")
