"""
Trainer for ClimateNet with two-stage training strategy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import time
import logging
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import numpy as np
import json

logger = logging.getLogger(__name__)


class Trainer:
    """
    Two-stage trainer for ClimateNet.

    Stage 1: Train with data loss only (physics losses disabled)
    Stage 2: Fine-tune with full loss (data + physics)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        # Training configuration
        max_epochs: int = 100,
        stage1_epochs: int = 50,
        gradient_clip_val: float = 1.0,
        early_stopping_patience: int = 15,
        # Checkpointing
        checkpoint_dir: str = "./checkpoints",
        save_every_n_epochs: int = 5,
        # Performance
        use_amp: bool = True,
        # Logging
        log_interval: int = 50,
        val_interval: int = 1,
        # Distributed training
        rank: int = 0,
        world_size: int = 1,
        is_distributed: bool = False,
        use_rank0_batch_broadcast: bool = False,
    ):
        """
        Initialize trainer.

        Args:
            model: ClimateNet model
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Combined loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device for training
            max_epochs: Maximum number of epochs
            stage1_epochs: Number of epochs for stage 1 (data loss only)
            gradient_clip_val: Gradient clipping value
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            save_every_n_epochs: Save checkpoint every N epochs
            use_amp: Use automatic mixed precision
            log_interval: Log every N batches
            val_interval: Validate every N epochs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Training config
        self.max_epochs = max_epochs
        self.stage1_epochs = stage1_epochs
        self.gradient_clip_val = gradient_clip_val
        self.early_stopping_patience = early_stopping_patience

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs

        # Mixed precision
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Logging
        self.log_interval = log_interval
        self.val_interval = val_interval

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "task_weights": [],
        }

        self.rank = rank
        self.world_size = world_size
        self.is_distributed = is_distributed
        self.use_rank0_batch_broadcast = use_rank0_batch_broadcast

        # Move model to device
        self.model = self.model.to(self.device)

        logger.info("=" * 70)
        logger.info("TRAINER INITIALIZED")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.use_amp}")
        logger.info(f"Max Epochs: {self.max_epochs}")
        logger.info(f"Stage 1 Epochs: {self.stage1_epochs}")
        if self.train_loader is not None and hasattr(self.train_loader, "dataset"):
            logger.info(f"Training samples: {len(train_loader.dataset):,}")
        else:
            logger.info("Training samples: provided by rank-0 batch broadcast")
        if self.val_loader is not None and hasattr(self.val_loader, "dataset"):
            logger.info(f"Validation samples: {len(val_loader.dataset):,}")
        else:
            logger.info("Validation samples: provided by rank-0 validation broadcast")
        if self.train_loader is not None and hasattr(self.train_loader, "batch_size"):
            logger.info(f"Batch size: {train_loader.batch_size}")
        logger.info("=" * 70)

    def _broadcast_batch_payload(
        self, inputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], lead_indices: torch.Tensor
    ):
        payload = {
            "stop": False,
            "static": inputs["static"].detach().cpu(),
            "dynamic": inputs["dynamic"].detach().cpu(),
            "lead": lead_indices.detach().cpu(),
            "targets": {k: v.detach().cpu() for k, v in targets.items()},
        }
        obj = [payload]
        dist.broadcast_object_list(obj, src=0)

    def _broadcast_stop_payload(self):
        obj = [{"stop": True}]
        dist.broadcast_object_list(obj, src=0)

    def _receive_batch_payload(self):
        obj = [None]
        dist.broadcast_object_list(obj, src=0)
        payload = obj[0]
        if payload.get("stop", False):
            return None, None, None, True

        inputs = {
            "static": payload["static"].to(self.device, non_blocking=True),
            "dynamic": payload["dynamic"].to(self.device, non_blocking=True),
        }
        lead_indices = payload["lead"].to(self.device, non_blocking=True)
        targets = {
            var: tensor.to(self.device, non_blocking=True)
            for var, tensor in payload["targets"].items()
        }
        return inputs, targets, lead_indices, False

    def _prepare_batch(self, batch: Tuple) -> Tuple[Dict, Dict, torch.Tensor]:
        """
        Prepare batch data.

        Args:
            batch: Batch from DataLoader

        Returns:
            Tuple of (inputs, targets, lead_indices)
        """
        # Assuming batch structure from custom dataset
        # batch = (inputs, targets, metadata)
        # inputs = (static, dynamic)
        # targets = {var_name: tensor}
        # metadata = {'lead': tensor, 'run_idx': tensor, ...}

        inputs, targets, metadata = batch
        static, dynamic = inputs

        # Move to device
        static = static.to(self.device, non_blocking=True)
        dynamic = dynamic.to(self.device, non_blocking=True)
        lead_indices = metadata["lead"].to(self.device, non_blocking=True)

        targets = {
            var: target.to(self.device, non_blocking=True)
            for var, target in targets.items()
        }

        inputs_dict = {"static": static, "dynamic": dynamic}

        return inputs_dict, targets, lead_indices

    def _distributed_mean(self, values: List[float]) -> float:
        """Compute mean value across all ranks using sum/count reduction."""
        local_sum = float(np.sum(values)) if values else 0.0
        local_count = float(len(values))

        if self.is_distributed and dist.is_available() and dist.is_initialized():
            tensor = torch.tensor([local_sum, local_count], device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            global_sum = tensor[0].item()
            global_count = tensor[1].item()
            return global_sum / max(global_count, 1.0)

        return local_sum / max(local_count, 1.0)

    def train_epoch(self, epoch: int, stage: int = 1) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
            stage: Training stage (1 or 2)

        Returns:
            Dictionary of average losses
        """
        self.model.train()

        # Set epoch for distributed sampler (only when local train_loader exists).
        if (
            self.is_distributed
            and self.train_loader is not None
            and hasattr(self.train_loader, "sampler")
            and hasattr(self.train_loader.sampler, "set_epoch")
        ):
            self.train_loader.sampler.set_epoch(epoch)

        # Disable physics losses in stage 1
        if stage == 1 and hasattr(self.loss_fn, "use_physics"):
            original_use_physics = self.loss_fn.use_physics
            self.loss_fn.use_physics = False

        epoch_losses = {
            "total": [],
            "data": [],
            "physics": [],
        }

        use_server_mode = (
            self.is_distributed
            and self.use_rank0_batch_broadcast
            and dist.is_available()
            and dist.is_initialized()
        )

        if use_server_mode and self.rank != 0:
            pbar = None
            batch_idx = 0
            while True:
                inputs, targets, lead_indices, should_stop = self._receive_batch_payload()
                if should_stop:
                    break

                # Forward pass with mixed precision
                with autocast("cuda", enabled=self.use_amp):
                    predictions = self.model(
                        static=inputs["static"],
                        dynamic=inputs["dynamic"],
                        lead_indices=lead_indices,
                    )

                    loss_dict = self.loss_fn(
                        predictions=predictions, targets=targets, return_components=True
                    )

                    total_loss = loss_dict["total"]

                self.optimizer.zero_grad()

                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_val
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_val
                        )
                    self.optimizer.step()

                self.global_step += 1
                epoch_losses["total"].append(total_loss.item())
                epoch_losses["data"].append(loss_dict["total_data"].item())
                if isinstance(loss_dict["total_physics"], torch.Tensor):
                    epoch_losses["physics"].append(loss_dict["total_physics"].item())
                else:
                    epoch_losses["physics"].append(loss_dict["total_physics"])
                batch_idx += 1
        else:
            # Progress bar
            pbar = tqdm(
                self.train_loader,
                desc=f"Stage {stage} - Epoch {epoch}/{self.max_epochs}",
                leave=False,
            )

            for batch_idx, batch in enumerate(pbar):
                # Prepare batch
                inputs, targets, lead_indices = self._prepare_batch(batch)

                if use_server_mode and self.rank == 0:
                    self._broadcast_batch_payload(inputs, targets, lead_indices)

                # Forward pass with mixed precision
                with autocast("cuda", enabled=self.use_amp):
                    # Model forward
                    predictions = self.model(
                        static=inputs["static"],
                        dynamic=inputs["dynamic"],
                        lead_indices=lead_indices,
                    )

                    # Compute loss
                    loss_dict = self.loss_fn(
                        predictions=predictions, targets=targets, return_components=True
                    )

                    total_loss = loss_dict["total"]

                # Backward pass
                self.optimizer.zero_grad()

                if self.use_amp:
                    self.scaler.scale(total_loss).backward()

                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_val
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()

                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_val
                        )

                    self.optimizer.step()

                # Update global step
                self.global_step += 1

                # Log losses
                epoch_losses["total"].append(total_loss.item())
                epoch_losses["data"].append(loss_dict["total_data"].item())

                if isinstance(loss_dict["total_physics"], torch.Tensor):
                    epoch_losses["physics"].append(loss_dict["total_physics"].item())
                else:
                    epoch_losses["physics"].append(loss_dict["total_physics"])

                # Update progress bar
                if batch_idx % self.log_interval == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{total_loss.item():.4f}",
                            "data": f"{loss_dict['total_data'].item():.4f}",
                            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        }
                    )

            if use_server_mode and self.rank == 0:
                self._broadcast_stop_payload()

        # Restore physics losses if in stage 1
        if stage == 1 and hasattr(self.loss_fn, "use_physics"):
            self.loss_fn.use_physics = original_use_physics

        # Compute average losses
        avg_losses = {key: self._distributed_mean(values) for key, values in epoch_losses.items()}

        return avg_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of average validation losses
        """
        self.model.eval()

        use_server_mode = (
            self.is_distributed
            and self.use_rank0_batch_broadcast
            and dist.is_available()
            and dist.is_initialized()
        )

        if use_server_mode and self.rank != 0:
            obj = [None]
            dist.broadcast_object_list(obj, src=0)
            return obj[0]

        val_losses = {
            "total": [],
            "data": [],
            "physics": [],
        }

        # Per-variable losses
        var_losses = {var: [] for var in self.loss_fn.target_vars}

        pbar = tqdm(self.val_loader, desc=f"Validation - Epoch {epoch}", leave=False)

        for batch in pbar:
            # Prepare batch
            inputs, targets, lead_indices = self._prepare_batch(batch)

            # Forward pass
            with autocast("cuda", enabled=self.use_amp):
                predictions = self.model(
                    static=inputs["static"],
                    dynamic=inputs["dynamic"],
                    lead_indices=lead_indices,
                )

                loss_dict = self.loss_fn(
                    predictions=predictions, targets=targets, return_components=True
                )

            # Log losses
            val_losses["total"].append(loss_dict["total"].item())
            val_losses["data"].append(loss_dict["total_data"].item())

            if isinstance(loss_dict["total_physics"], torch.Tensor):
                val_losses["physics"].append(loss_dict["total_physics"].item())
            else:
                val_losses["physics"].append(loss_dict["total_physics"])

            # Per-variable losses
            for var in self.loss_fn.target_vars:
                if var in loss_dict["data_losses"]:
                    var_losses[var].append(loss_dict["data_losses"][var].item())

        # Compute averages
        avg_losses = {key: self._distributed_mean(values) for key, values in val_losses.items()}

        # Add per-variable losses
        for var in self.loss_fn.target_vars:
            if var_losses[var]:
                avg_losses[f"{var}_loss"] = self._distributed_mean(var_losses[var])

        if use_server_mode and self.rank == 0:
            obj = [avg_losses]
            dist.broadcast_object_list(obj, src=0)

        return avg_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False, filename: str = None):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
            filename: Custom filename (optional)
        """
        """Save checkpoint (only on rank 0)."""
        if self.rank != 0:
            return
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        # Prepare checkpoint
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save task weights if using uncertainty weighting
        if hasattr(self.loss_fn, "task_weighting") and hasattr(
            self.loss_fn.task_weighting, "log_vars"
        ):
            checkpoint["task_log_vars"] = self.loss_fn.task_weighting.log_vars.data

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save as best model if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state (handle DDP/non-DDP key prefix mismatch)
        checkpoint_state_dict = checkpoint["model_state_dict"]
        model_state_keys = list(self.model.state_dict().keys())
        checkpoint_keys = list(checkpoint_state_dict.keys())

        if model_state_keys and checkpoint_keys:
            model_uses_module_prefix = model_state_keys[0].startswith("module.")
            checkpoint_uses_module_prefix = checkpoint_keys[0].startswith("module.")

            if checkpoint_uses_module_prefix and not model_uses_module_prefix:
                checkpoint_state_dict = {
                    key.replace("module.", "", 1): value
                    for key, value in checkpoint_state_dict.items()
                }
            elif not checkpoint_uses_module_prefix and model_uses_module_prefix:
                checkpoint_state_dict = {
                    f"module.{key}": value
                    for key, value in checkpoint_state_dict.items()
                }

        self.model.load_state_dict(checkpoint_state_dict)

        # Load optimizer state
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state
        if self.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Load task weights
        if (
            "task_log_vars" in checkpoint
            and hasattr(self.loss_fn, "task_weighting")
            and hasattr(self.loss_fn.task_weighting, "log_vars")
        ):
            self.loss_fn.task_weighting.log_vars.data = checkpoint["task_log_vars"]

        # Load training state
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.training_history = checkpoint.get(
            "training_history",
            {
                "train_loss": [],
                "val_loss": [],
                "learning_rates": [],
                "task_weights": [],
            },
        )

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def train(self, resume_from: Optional[str] = None):
        """
        Main training loop with two-stage strategy.

        Args:
            resume_from: Path to checkpoint to resume from (optional)
        """
        # Resume from checkpoint if provided
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 1

        logger.info("\n" + "=" * 70)
        logger.info("STARTING TRAINING")
        logger.info("=" * 70)

        try:
            for epoch in range(start_epoch, self.max_epochs + 1):
                epoch_start_time = time.time()

                # Determine training stage
                if epoch <= self.stage1_epochs:
                    stage = 1
                    stage_name = "Stage 1 (Data Loss Only)"
                else:
                    stage = 2
                    stage_name = "Stage 2 (Data + Physics)"

                logger.info(f"\n{'='*70}")
                logger.info(f"Epoch {epoch}/{self.max_epochs} - {stage_name}")
                logger.info(f"{'='*70}")

                # Train
                train_losses = self.train_epoch(epoch, stage=stage)

                # Update DWA weights using the epoch-averaged losses just collected
                if (
                    hasattr(self.loss_fn, "task_weighting")
                    and self.loss_fn.task_weighting is not None
                    and hasattr(self.loss_fn.task_weighting, "end_of_epoch")
                ):
                    self.loss_fn.task_weighting.end_of_epoch()

                # Validate
                if epoch % self.val_interval == 0:
                    val_losses = self.validate(epoch)
                else:
                    val_losses = {"total": 0.0}

                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_losses["total"])
                    else:
                        self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]["lr"]

                # Log epoch results
                epoch_time = time.time() - epoch_start_time

                logger.info(f"\nEpoch {epoch} Summary:")
                logger.info(f"  Time: {epoch_time:.2f}s")
                logger.info(f"  Learning Rate: {current_lr:.2e}")
                logger.info(f"  Train Loss: {train_losses['total']:.4f}")
                logger.info(f"    - Data: {train_losses['data']:.4f}")
                logger.info(f"    - Physics: {train_losses['physics']:.4f}")

                if epoch % self.val_interval == 0:
                    logger.info(f"  Val Loss: {val_losses['total']:.4f}")
                    logger.info(f"    - Data: {val_losses['data']:.4f}")
                    logger.info(f"    - Physics: {val_losses['physics']:.4f}")

                    # Log per-variable losses
                    for var in self.loss_fn.target_vars:
                        var_key = f"{var}_loss"
                        if var_key in val_losses:
                            logger.info(f"    - {var}: {val_losses[var_key]:.4f}")

                # Get task weights
                task_weights = {}
                if (
                    hasattr(self.loss_fn, "task_weighting")
                    and self.loss_fn.task_weighting is not None
                ):
                    task_weights = self.loss_fn.task_weighting.get_weights()
                    logger.info(f"  Task Weights:")
                    for var, weight in task_weights.items():
                        logger.info(f"    - {var}: {weight:.4f}")

                # Update history
                self.training_history["train_loss"].append(train_losses["total"])
                self.training_history["val_loss"].append(val_losses["total"])
                self.training_history["learning_rates"].append(current_lr)
                self.training_history["task_weights"].append(task_weights)

                # Check for improvement
                is_best = False
                if val_losses["total"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total"]
                    self.epochs_without_improvement = 0
                    is_best = True
                    logger.info(
                        f"  *** New best validation loss: {self.best_val_loss:.4f} ***"
                    )
                else:
                    self.epochs_without_improvement += 1

                # Save checkpoint
                if epoch % self.save_every_n_epochs == 0 or is_best:
                    self.save_checkpoint(epoch, is_best=is_best)

                # Early stopping
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                    logger.info(
                        f"No improvement for {self.early_stopping_patience} epochs"
                    )
                    break

                self.current_epoch = epoch

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
            logger.info("Saving checkpoint...")
            self.save_checkpoint(self.current_epoch, filename="interrupted.pt")

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Total epochs: {self.current_epoch}")
        logger.info(f"Checkpoints saved to: {self.checkpoint_dir}")

        # Save final training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            # Convert numpy types to Python types
            history_serializable = {}
            for key, value in self.training_history.items():
                if isinstance(value, list):
                    history_serializable[key] = [
                        float(v) if isinstance(v, (np.floating, float)) else v
                        for v in value
                    ]
                else:
                    history_serializable[key] = value

            json.dump(history_serializable, f, indent=2)

        logger.info(f"Training history saved to: {history_path}")


class EarlyStopping:
    """
    Early stopping helper class.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop
