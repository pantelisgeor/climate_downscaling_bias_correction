"""
Evaluator for seasonal ClimateNet – mirrors src/training/evaluator.py.

Key differences from the decadal Evaluator:
  • "lead" metadata key contains the integer lead-month (0-6) rather than the
    lead-time index used in the decadal pipeline.
  • Per-lead metrics are labelled lead_month_0 … lead_month_6.
  • Per-member metrics are optionally computed (member 0-24).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EvaluatorSeasonal:
    """
    Comprehensive evaluator for seasonal ClimateNet.

    Computes metrics and generates visualizations for model predictions.
    The interface is identical to the decadal Evaluator so it can be
    dropped-in inside train_seasonal.py.
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
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.target_vars = target_vars
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info("EvaluatorSeasonal initialized")
        logger.info(f"  Results: {self.results_dir}")
        logger.info(f"  Figures: {self.figures_dir}")

    # ── main entry point ──────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, save_predictions: bool = True) -> Dict:
        """
        Run full evaluation on the test DataLoader.

        Returns a dict with keys: 'metrics', 'per_lead_metrics',
        'predictions', 'targets'.
        """
        self.model.eval()

        all_predictions = {var: [] for var in self.target_vars}
        all_targets     = {var: [] for var in self.target_vars}
        all_lead_months: List[np.ndarray] = []

        logger.info("Running inference on test set …")
        for batch in tqdm(self.test_loader, desc="Evaluating"):
            inputs, targets, metadata = batch
            static, dynamic = inputs

            static       = static.to(self.device, non_blocking=True)
            dynamic      = dynamic.to(self.device, non_blocking=True)
            lead_indices = metadata["lead"].to(self.device, non_blocking=True)

            targets_dev = {
                var: t.to(self.device, non_blocking=True)
                for var, t in targets.items()
            }

            predictions = self.model(
                static=static, dynamic=dynamic, lead_indices=lead_indices
            )

            for var in self.target_vars:
                all_predictions[var].append(
                    predictions[var].detach().cpu().numpy()
                )
                all_targets[var].append(targets_dev[var].detach().cpu().numpy())

            all_lead_months.append(lead_indices.detach().cpu().numpy())

        # Concatenate batches
        logger.info("Concatenating batches …")
        for var in self.target_vars:
            all_predictions[var] = np.concatenate(all_predictions[var], axis=0)
            all_targets[var]     = np.concatenate(all_targets[var],     axis=0)
        lead_array = np.concatenate(all_lead_months, axis=0)

        # Denormalisation
        dataset = self.test_loader.dataset
        base_dataset = getattr(dataset, "dataset", dataset)
        data_loader  = getattr(base_dataset, "data_loader", None)

        use_denorm = (
            getattr(base_dataset, "normalize", False)
            and data_loader is not None
            and hasattr(data_loader, "scalers")
            and all(var in data_loader.scalers for var in self.target_vars)
        )

        logger.info(
            "Denormalizing …" if use_denorm
            else "Skipping denormalization (no scalers available) …"
        )

        preds_denorm  = {}
        tgts_denorm   = {}
        for var in self.target_vars:
            pred   = all_predictions[var]
            target = all_targets[var]
            if use_denorm:
                pred_d   = data_loader.denormalize(pred.reshape(-1),   var).reshape(pred.shape)
                target_d = data_loader.denormalize(target.reshape(-1), var).reshape(target.shape)
            else:
                pred_d, target_d = pred, target
            preds_denorm[var]  = pred_d
            tgts_denorm[var]   = target_d

            logger.info(
                f"  {var}  pred=[{pred_d.min():.3f}, {pred_d.max():.3f}]  "
                f"target=[{target_d.min():.3f}, {target_d.max():.3f}]"
            )

        # Overall metrics
        metrics          = self.compute_metrics(preds_denorm, tgts_denorm)
        per_lead_metrics = self.compute_per_lead_metrics(
            preds_denorm, tgts_denorm, lead_array
        )

        # Save
        metrics_path = self.results_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {"overall": metrics, "per_lead_month": per_lead_metrics},
                f, indent=2
            )
        logger.info(f"Metrics saved → {metrics_path}")

        # Visualisations
        self.create_visualizations(preds_denorm, tgts_denorm, lead_array, metrics)

        return {
            "metrics":          metrics,
            "per_lead_metrics": per_lead_metrics,
            "predictions":      preds_denorm if save_predictions else None,
            "targets":          tgts_denorm  if save_predictions else None,
        }

    # ── metric helpers ────────────────────────────────────────────────────────

    def compute_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        targets:     Dict[str, np.ndarray],
    ) -> Dict:
        metrics = {}
        for var in self.target_vars:
            pred   = predictions[var].flatten()
            target = targets[var].flatten()
            mask   = ~(np.isnan(pred) | np.isnan(target))
            pred, target = pred[mask], target[mask]

            if len(pred) == 0:
                metrics[var] = {k: float("nan") for k in ("rmse","mae","bias","r2")}
                continue

            rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
            mae  = float(np.mean(np.abs(pred - target)))
            bias = float(np.mean(pred - target))
            ss_res = np.sum((target - pred)   ** 2)
            ss_tot = np.sum((target - target.mean()) ** 2)
            r2   = float(1 - ss_res / (ss_tot + 1e-8))

            metrics[var] = {"rmse": rmse, "mae": mae, "bias": bias, "r2": r2}
            logger.info(
                f"  {var}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
                f"Bias={bias:.4f}  R²={r2:.4f}"
            )
        return metrics

    def compute_per_lead_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        targets:     Dict[str, np.ndarray],
        lead_array:  np.ndarray,
    ) -> Dict:
        """Compute metrics broken down by integer lead-month."""
        per_lead = {}
        for lead_month in np.unique(lead_array):
            key  = f"lead_month_{int(lead_month)}"
            mask = lead_array == lead_month
            per_lead[key] = {}
            for var in self.target_vars:
                pred   = predictions[var][mask].flatten()
                target = targets[var][mask].flatten()
                valid  = ~(np.isnan(pred) | np.isnan(target))
                pred, target = pred[valid], target[valid]
                if len(pred) == 0:
                    continue
                per_lead[key][var] = {
                    "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
                    "mae":  float(np.mean(np.abs(pred - target))),
                    "bias": float(np.mean(pred - target)),
                    "n":    int(len(pred)),
                }
        return per_lead

    # ── visualisations ────────────────────────────────────────────────────────

    def create_visualizations(
        self,
        predictions: Dict[str, np.ndarray],
        targets:     Dict[str, np.ndarray],
        lead_array:  np.ndarray,
        metrics:     Dict,
    ):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            logger.warning("matplotlib/seaborn not installed – skipping plots")
            return

        # 1  Scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        for idx, var in enumerate(self.target_vars):
            ax   = axes[idx]
            pred = predictions[var].flatten()
            tgt  = targets[var].flatten()
            m    = ~(np.isnan(pred) | np.isnan(tgt))
            pred, tgt = pred[m], tgt[m]
            if len(pred) == 0:
                continue
            if len(pred) > 10_000:
                s = np.random.choice(len(pred), 10_000, replace=False)
                pred, tgt = pred[s], tgt[s]
            ax.scatter(tgt, pred, alpha=0.3, s=1, c="steelblue")
            lo = min(tgt.min(), pred.min())
            hi = max(tgt.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect")
            r2 = metrics[var]["r2"]
            rmse = metrics[var]["rmse"]
            bias = metrics[var]["bias"]
            ax.text(0.05, 0.95, f"R²={r2:.3f}\nRMSE={rmse:.3f}\nBias={bias:.3f}",
                    transform=ax.transAxes, fontsize=10, va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            ax.set_xlabel("Target"); ax.set_ylabel("Prediction")
            ax.set_title(f"{var}"); ax.legend()
        plt.tight_layout()
        plt.savefig(self.figures_dir / "scatter_plots.png", dpi=200, bbox_inches="tight")
        plt.close()

        # 2  Error histograms
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for idx, var in enumerate(self.target_vars):
            ax   = axes[idx]
            pred = predictions[var].flatten()
            tgt  = targets[var].flatten()
            m    = ~(np.isnan(pred) | np.isnan(tgt))
            errors = pred[m] - tgt[m]
            ax.hist(errors, bins=50, alpha=0.7, edgecolor="black")
            ax.axvline(0, color="r", ls="--", lw=2)
            ax.axvline(metrics[var]["bias"], color="g", ls="--", lw=2,
                       label=f"bias={metrics[var]['bias']:.3f}")
            ax.set_title(f"{var} – error distribution"); ax.legend()
        plt.tight_layout()
        plt.savefig(self.figures_dir / "error_distributions.png", dpi=200, bbox_inches="tight")
        plt.close()

        # 3  Per-lead-month RMSE
        unique_leads = sorted(np.unique(lead_array).tolist())
        if len(unique_leads) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            for ax_idx, var in enumerate(self.target_vars):
                ax    = axes[ax_idx]
                rmses = []
                for lm in unique_leads:
                    mask = lead_array == lm
                    p = predictions[var][mask].flatten()
                    t = targets[var][mask].flatten()
                    valid = ~(np.isnan(p) | np.isnan(t))
                    p, t = p[valid], t[valid]
                    rmses.append(float(np.sqrt(np.mean((p - t)**2))) if len(p) else float("nan"))
                ax.plot(unique_leads, rmses, "o-", lw=2)
                ax.set_xlabel("Lead month"); ax.set_ylabel("RMSE")
                ax.set_title(f"{var} – RMSE by lead month")
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.figures_dir / "rmse_by_lead_month.png", dpi=200,
                        bbox_inches="tight")
            plt.close()

        # 4  Spatial sample maps
        logger.info("Creating spatial maps …")
        sample_idx = np.random.randint(0, predictions[self.target_vars[0]].shape[0])
        fig, axes  = plt.subplots(len(self.target_vars), 3,
                                  figsize=(15, 4 * len(self.target_vars)))
        for vi, var in enumerate(self.target_vars):
            tgt_map  = targets[var][sample_idx, 0]
            pred_map = predictions[var][sample_idx, 0]
            err_map  = pred_map - tgt_map
            vmin = min(tgt_map.min(), pred_map.min())
            vmax = max(tgt_map.max(), pred_map.max())
            for col, (data, title) in enumerate([
                (tgt_map, f"{var} – Target"),
                (pred_map, f"{var} – Pred"),
                (err_map, f"{var} – Error"),
            ]):
                cmap = "viridis" if col < 2 else "RdBu_r"
                kw = {} if col < 2 else {"vmin": -abs(err_map).max(), "vmax": abs(err_map).max()}
                if col < 2:
                    kw = {"vmin": vmin, "vmax": vmax}
                im = axes[vi, col].imshow(data, cmap=cmap, **kw)
                axes[vi, col].set_title(title)
                axes[vi, col].axis("off")
                plt.colorbar(im, ax=axes[vi, col], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "spatial_predictions.png", dpi=200,
                    bbox_inches="tight")
        plt.close()

        logger.info(f"Figures saved → {self.figures_dir}")
