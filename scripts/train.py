"""
Main training script for ClimateNet with multi-GPU support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import argparse
import yaml
from pathlib import Path
import sys
import logging
from datetime import datetime
import shutil
import json
import numpy as np
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DecadalDataLoader
from src.training.climate_dataset import ClimateDataset
from src.models.climate_net import ClimateNet
from src.losses.task_weighting import CombinedLoss
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


def setup_logger(
    log_dir: Path, rank: int = 0, log_level: str = "INFO"
) -> logging.Logger:
    """
    Set up logger with file and console handlers.

    Only rank 0 logs to console and file.
    Other ranks are silenced.
    """
    if rank != 0:
        # Disable logging for non-zero ranks
        logging.basicConfig(level=logging.ERROR)
        return logging.getLogger(__name__)

    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, save_path: Path):
    """Save configuration to YAML file."""
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_experiment_directory(config: dict, rank: int = 0) -> Path:
    """
    Create experiment directory (only on rank 0).
    """
    if rank == 0:
        base_dir = Path(config["experiment"]["base_dir"])
        experiment_name = config["experiment"]["name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        exp_dir = base_dir / f"{experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)
        (exp_dir / "figures").mkdir(exist_ok=True)

        return exp_dir
    else:
        # Non-zero ranks will receive the directory path later
        return None


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialize distributed training.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def main(rank: int, world_size: int, config_path: str, resume_path: str = None):
    """
    Main training function.

    Args:
        rank: Process rank for distributed training
        world_size: Total number of processes
        config_path: Path to configuration file
        resume_path: Path to checkpoint to resume from (optional)
    """
    # Load configuration
    config = load_config(config_path)

    # Setup distributed training if world_size > 1
    if world_size > 1:
        setup_distributed(rank, world_size)

    # Create experiment directory (only rank 0)
    exp_dir = create_experiment_directory(config, rank)

    # Broadcast experiment directory path to all ranks
    if world_size > 1:
        exp_dir_list = [exp_dir]
        dist.broadcast_object_list(exp_dir_list, src=0)
        exp_dir = exp_dir_list[0]

    # Setup logger
    logger = setup_logger(
        exp_dir / "logs",
        rank=rank,
        log_level=config.get("logging", {}).get("level", "INFO"),
    )

    # Save configuration (rank 0 only)
    if rank == 0:
        save_config(config, exp_dir / "config.yaml")
        logger.info("=" * 70)
        logger.info("CLIMATENET TRAINING")
        logger.info("=" * 70)
        logger.info(f"Experiment directory: {exp_dir}")
        logger.info(f"Configuration: {config_path}")
        if resume_path:
            logger.info(f"Resuming from: {resume_path}")
        logger.info("=" * 70)

    # Set device
    # Set device
    if world_size > 1:
        device = f"cuda:{rank}"
    else:
        device = config["training"].get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    logger.info("=" * 70)
    logger.info("DATA LOADING")
    logger.info("=" * 70)

    # Create data loader
    data_loader = DecadalDataLoader(
        nc_path=config["data"]["nc_path"],
        normalize_method=config["data"]["normalize_method"],
        cache_dir=config["data"]["cache_dir"],
        load_in_memory=config["data"]["load_in_memory"],
    )

    # Variables to normalize
    static_vars = ["dem", "rho", "phi"]
    dynamic_vars = ["pr", "tas", "tasmax", "hurs", "sin_time", "cos_time"]
    cci_vars = ["cci_agg"]  # CCI aggregate (10 classes, normalized together)
    target_vars = config["model"]["target_vars"]

    # Fit normalizer on training set (only rank 0 needs to do this)
    if config["data"]["normalize"] and rank == 0:
        logger.info("=" * 70)
        logger.info("FITTING NORMALIZER ON TRAINING DATA")
        logger.info("=" * 70)

        # Determine sample size for fitting
        total_samples = len(data_loader)
        sample_size = min(config["data"].get("normalize_samples", 1000), total_samples)

        logger.info(f"Total available samples: {total_samples:,}")
        logger.info(f"Using {sample_size:,} samples for normalization fitting")

        # Sample uniformly across the dataset
        sample_indices = np.linspace(0, total_samples - 1, sample_size, dtype=int)

        # Collect values for each variable
        var_values = {
            var: [] for var in static_vars + dynamic_vars + cci_vars + target_vars
        }

        logger.info("Collecting data from samples...")
        for i, idx in enumerate(sample_indices):
            if (i + 1) % 200 == 0:
                logger.info(f"  Processed {i + 1}/{sample_size} samples...")

            try:
                inputs, targets = data_loader[idx]

                # Static vars (indices 0-2: dem, rho, phi)
                for i, var in enumerate(static_vars):
                    var_values[var].append(inputs[i].flatten())

                # Dynamic vars (indices 3-8: pr, tas, tasmax, hurs, sin_time, cos_time)
                for i, var in enumerate(dynamic_vars):
                    var_values[var].append(inputs[i + 3].flatten())

                # CCI vars (indices 9-18: 10 classes)
                # Collect all 10 classes together
                cci_data = inputs[9:19].flatten()  # All 10 classes
                var_values["cci_agg"].append(cci_data)

                # Target variables
                for i, var in enumerate(target_vars):
                    var_values[var].append(targets[i].flatten())

            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue

        # Compute and store normalization parameters
        logger.info("\nComputing normalization parameters:")
        logger.info("-" * 70)

        for var_name, values_list in var_values.items():
            if len(values_list) == 0:
                logger.warning(f"  {var_name:15s}: No data collected, skipping")
                continue

            all_values = np.concatenate(values_list)

            # Apply log transform to precipitation before computing normalization params
            if var_name == "tpERA":
                logger.info(f"  Applying log1p transform to {var_name}")
                all_values = np.log1p(np.maximum(all_values, 0))

            if config["data"]["normalize_method"] == "minmax":
                vmin = float(np.nanmin(all_values))
                vmax = float(np.nanmax(all_values))

                # Avoid division by zero
                if vmax - vmin < 1e-8:
                    logger.warning(
                        f"  {var_name:15s}: Constant values detected, using [0, 1] range"
                    )
                    vmax = vmin + 1.0

                data_loader.scalers[var_name] = {"min": vmin, "max": vmax}
                logger.info(f"  {var_name:15s}: min={vmin:12.4f}, max={vmax:12.4f}")

            elif config["data"]["normalize_method"] == "zscore":
                vmean = float(np.nanmean(all_values))
                vstd = float(np.nanstd(all_values))

                # Avoid division by zero
                if vstd < 1e-8:
                    logger.warning(
                        f"  {var_name:15s}: Zero std detected, using std=1.0"
                    )
                    vstd = 1.0

                data_loader.scalers[var_name] = {"mean": vmean, "std": vstd}
                logger.info(f"  {var_name:15s}: mean={vmean:12.4f}, std={vstd:12.4f}")

        logger.info("-" * 70)
        logger.info(
            f"Successfully fitted normalizer for {len(data_loader.scalers)} variables"
        )
        logger.info("=" * 70)

        # Save normalization parameters
        norm_params_path = exp_dir / "normalization_params.json"
        with open(norm_params_path, "w") as f:
            json.dump(data_loader.scalers, f, indent=2)
        logger.info(f"Normalization parameters saved to: {norm_params_path}")

    # Broadcast normalization parameters to all ranks
    if world_size > 1:
        # Rank 0 broadcasts scalers
        scalers_list = [data_loader.scalers if rank == 0 else None]
        dist.broadcast_object_list(scalers_list, src=0)
        data_loader.scalers = scalers_list[0]

        if rank != 0:
            logger.info("Received normalization parameters from rank 0")

    # Create full dataset
    if rank == 0:
        logger.info("Creating datasets...")

    full_dataset = ClimateDataset(
        data_loader=data_loader,
        normalize=config["data"]["normalize"],
        fit_normalizer=False,  # Already fitted above
        target_vars=config["model"]["target_vars"],
    )

    # Split into train/val/test
    total_size = len(full_dataset)
    train_ratio = config["data"]["train_ratio"]
    val_ratio = config["data"]["val_ratio"]

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Use fixed indices for reproducibility
    indices = np.arange(total_size)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    if rank == 0:
        logger.info(f"Dataset split:")
        logger.info(f"  Train: {len(train_dataset):,} samples ({train_ratio*100:.1f}%)")
        logger.info(f"  Val:   {len(val_dataset):,} samples ({val_ratio*100:.1f}%)")
        logger.info(
            f"  Test:  {len(test_dataset):,} samples ({(1-train_ratio-val_ratio)*100:.1f}%)"
        )

    # Create data loaders
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )

    if rank == 0:
        logger.info(f"Created data loaders:")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches:   {len(val_loader)}")
        logger.info(f"  Test batches:  {len(test_loader)}")

    # =========================================================================
    # MODEL CREATION
    # =========================================================================
    if rank == 0:
        logger.info("=" * 70)
        logger.info("MODEL CREATION")
        logger.info("=" * 70)

    model = ClimateNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        embed_dim=config["model"]["embed_dim"],
        depth=config["model"]["depth"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        dropout=config["model"]["dropout"],
        target_vars=config["model"]["target_vars"],
    )

    model = model.to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters:")
        logger.info(f"  Total:     {total_params:,}")
        logger.info(f"  Trainable: {trainable_params:,}")

    # Wrap model with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
        if rank == 0:
            logger.info(f"Model wrapped with DistributedDataParallel")

    # =========================================================================
    # LOSS FUNCTION
    # =========================================================================
    if rank == 0:
        logger.info("=" * 70)
        logger.info("LOSS FUNCTION")
        logger.info("=" * 70)

    criterion = CombinedLoss(  # <-- Use CombinedLoss instead
        task_names=config["model"]["target_vars"],
        loss_types=config["loss"]["loss_types"],
        weighting_strategy=config["loss"]["weighting_strategy"],
        initial_weights=config["loss"].get("initial_weights"),
        device=device,
    )

    if rank == 0:
        logger.info(f"Loss configuration:")
        logger.info(f"  Strategy: {config['loss']['weighting_strategy']}")
        logger.info(f"  Loss types: {config['loss']['loss_types']}")

    # =========================================================================
    # OPTIMIZER & SCHEDULER
    # =========================================================================
    if rank == 0:
        logger.info("=" * 70)
        logger.info("OPTIMIZER & SCHEDULER")
        logger.info("=" * 70)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    if config["training"].get("use_scheduler", True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
            eta_min=config["training"].get("min_lr", 1e-6),
        )
    else:
        scheduler = None

    if rank == 0:
        logger.info(f"Optimizer: AdamW")
        logger.info(f"  Learning rate: {config['training']['learning_rate']}")
        logger.info(f"  Weight decay: {config['training']['weight_decay']}")
        if scheduler:
            logger.info(f"Scheduler: CosineAnnealingLR")

    # =========================================================================
    # TRAINER
    # =========================================================================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        target_vars=config["model"]["target_vars"],
        checkpoint_dir=exp_dir / "checkpoints",
        log_dir=exp_dir / "logs",
        rank=rank,
        world_size=world_size,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_path:
        start_epoch = trainer.load_checkpoint(resume_path)
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}")

    # =========================================================================
    # TRAINING
    # =========================================================================
    if rank == 0:
        logger.info("=" * 70)
        logger.info("STARTING TRAINING")
        logger.info("=" * 70)

    trainer.train(
        epochs=config["training"]["epochs"],
        start_epoch=start_epoch,
        save_every=config["training"].get("save_every", 10),
        eval_every=config["training"].get("eval_every", 1),
    )

    # =========================================================================
    # EVALUATION
    # =========================================================================
    if rank == 0:
        logger.info("=" * 70)
        logger.info("FINAL EVALUATION ON TEST SET")
        logger.info("=" * 70)

        # Load best model
        best_model_path = exp_dir / "checkpoints" / "best_model.pt"
        if best_model_path.exists():
            logger.info(f"Loading best model from: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            if world_size > 1:
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])

        evaluator = Evaluator(
            model=model.module if world_size > 1 else model,
            test_loader=test_loader,
            device=device,
            target_vars=config["model"]["target_vars"],
            results_dir=exp_dir / "results",
            figures_dir=exp_dir / "figures",
        )

        # Run evaluation
        results = evaluator.evaluate(save_predictions=True)

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Results saved to: {exp_dir}")
        logger.info("=" * 70)

    # Clean up distributed training
    if world_size > 1:
        cleanup_distributed()

    # Close data loader
    data_loader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ClimateNet")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs to use for training"
    )

    args = parser.parse_args()

    # Check if config file exists
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Single GPU or CPU training
    if args.gpus <= 1:
        main(rank=0, world_size=1, config_path=args.config, resume_path=args.resume)
    # Multi-GPU training
    else:
        import torch.multiprocessing as mp

        mp.spawn(
            main,
            args=(args.gpus, args.config, args.resume),
            nprocs=args.gpus,
            join=True,
        )
