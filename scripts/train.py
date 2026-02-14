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
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import DecadalDataLoader
from models.climate_net import ClimateNet
from losses.task_weighting import CombinedLoss
from training.climate_dataset import ClimateDataset
from training.trainer import Trainer
from training.evaluator import Evaluator


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    dist.barrier()

    return True, rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_logging(log_dir: Path, log_level: str = "INFO", rank: int = 0):
    """Setup logging configuration (only on rank 0)."""
    if rank != 0:
        # Disable logging for non-zero ranks
        logging.basicConfig(level=logging.ERROR)
        return logging.getLogger(__name__)

    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

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
    """Create experiment directory (only on rank 0)."""
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


def get_year_from_time_index(data_loader, time_idx: int) -> int:
    """Extract year from time index."""
    import pandas as pd

    time_value = data_loader.ds.time.values[time_idx]
    timestamp = pd.Timestamp(time_value)
    return timestamp.year


def create_year_based_splits(
    data_loader: DecadalDataLoader, train_years: list, val_years: list, test_years: list
) -> dict:
    """Create train/val/test splits based on years."""
    logger = logging.getLogger(__name__)

    logger.info("Creating year-based data splits...")
    logger.info(
        f"  Train years: {min(train_years)}-{max(train_years)} ({len(train_years)} years)"
    )
    logger.info(
        f"  Val years: {min(val_years)}-{max(val_years)} ({len(val_years)} years)"
    )
    logger.info(
        f"  Test years: {min(test_years)}-{max(test_years)} ({len(test_years)} years)"
    )

    valid_combos = data_loader.valid_combinations

    train_indices = []
    val_indices = []
    test_indices = []

    for idx in range(len(valid_combos)):
        time_idx = valid_combos.iloc[idx]["time_idx"]
        year = get_year_from_time_index(data_loader, time_idx)

        if year in train_years:
            train_indices.append(idx)
        elif year in val_years:
            val_indices.append(idx)
        elif year in test_years:
            test_indices.append(idx)

    logger.info(f"\nSplit sizes:")
    logger.info(f"  Train: {len(train_indices):,} samples")
    logger.info(f"  Val: {len(val_indices):,} samples")
    logger.info(f"  Test: {len(test_indices):,} samples")

    return {"train": train_indices, "val": val_indices, "test": test_indices}


def create_data_loaders(
    config: dict, exp_dir: Path, rank: int, world_size: int, is_distributed: bool
):
    """Create training, validation, and test data loaders with distributed support."""
    logger = logging.getLogger(__name__)

    if rank == 0:
        logger.info("Loading dataset...")

    data_loader = DecadalDataLoader(
        nc_path=config["data"]["nc_path"],
        normalize_method=config["data"]["normalize_method"],
        cache_dir=config["data"]["cache_dir"],
        load_in_memory=config["data"]["load_in_memory"],
    )

    # Create year-based splits (same on all ranks)
    splits = create_year_based_splits(
        data_loader=data_loader,
        train_years=config["data"]["train_years"],
        val_years=config["data"]["val_years"],
        test_years=config["data"]["test_years"],
    )

    # Save split indices (only rank 0)
    if rank == 0:
        split_path = exp_dir / "data_splits.json"
        with open(split_path, "w") as f:
            json.dump(
                {
                    "train_indices": splits["train"],
                    "val_indices": splits["val"],
                    "test_indices": splits["test"],
                    "train_years": config["data"]["train_years"],
                    "val_years": config["data"]["val_years"],
                    "test_years": config["data"]["test_years"],
                },
                f,
                indent=2,
            )
        logger.info(f"Data splits saved to: {split_path}")

    # Create full dataset
    if rank == 0:
        logger.info("Creating datasets...")

    full_dataset = ClimateDataset(
        data_loader=data_loader,
        normalize=config["data"]["normalize"],
        fit_normalizer=False,
        target_vars=config["model"]["target_vars"],
    )

    # Create subset datasets
    train_dataset = Subset(full_dataset, splits["train"])
    val_dataset = Subset(full_dataset, splits["val"])
    test_dataset = Subset(full_dataset, splits["test"])

    # Fit normalizer on training set (only rank 0 needs to do this)
    if config["data"]["normalize"] and rank == 0:
        logger.info("=" * 70)
        logger.info("FITTING NORMALIZER ON TRAINING DATA")
        logger.info("=" * 70)

        # Variables to normalize
        static_vars = ["dem", "rho", "phi"]
        dynamic_vars = ["pr", "tas", "tasmax", "hurs", "sin_time", "cos_time"]
        cci_vars = ["cci_agg"]  # CCI aggregate (10 classes, normalized together)
        target_vars = config["model"]["target_vars"]

        # Sample indices for statistics computation
        sample_size = min(1000, len(splits["train"]))
        logger.info(
            f"Sampling {sample_size} training samples for normalization statistics..."
        )
        sample_indices = np.random.choice(splits["train"], sample_size, replace=False)

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

                # CCI aggregate (indices 9-18: 10 land cover classes)
                cci_data = inputs[9:19].flatten()
                var_values["cci_agg"].append(cci_data)

                # Target variables
                for i, var in enumerate(target_vars):
                    var_values[var].append(targets[i].flatten())

            except Exception as e:
                logger.warning(f"  Error processing sample {idx}: {e}")
                continue

        # Compute and store normalization parameters
        logger.info("\nComputing normalization parameters:")
        logger.info("-" * 70)

        for var_name, values_list in var_values.items():
            if len(values_list) == 0:
                logger.warning(f"  {var_name:15s}: No data collected, skipping")
                continue

            all_values = np.concatenate(values_list)

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
            f"✓ Successfully fitted normalizer for {len(data_loader.scalers)} variables"
        )
        logger.info("=" * 70)

        # Save normalization parameters for reproducibility
        scaler_path = exp_dir / "normalization_params.json"
        with open(scaler_path, "w") as f:
            json.dump(data_loader.scalers, f, indent=2)
        logger.info(f"Normalization parameters saved to: {scaler_path}\n")

    # Synchronize all processes (wait for rank 0 to finish fitting)
    if is_distributed:
        dist.barrier()

        # Broadcast scalers from rank 0 to all other ranks
        if rank == 0:
            scaler_list = [data_loader.scalers]
        else:
            scaler_list = [None]

        dist.broadcast_object_list(scaler_list, src=0)

        if rank != 0:
            data_loader.scalers = scaler_list[0]
            logger.info(f"Rank {rank}: Received normalization parameters from rank 0")

    # Create distributed samplers if using DDP
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        shuffle = False  # Sampler handles shuffling
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle = True

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        sampler=test_sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, data_loader


def create_model(config: dict) -> ClimateNet:
    """Create ClimateNet model."""
    logger = logging.getLogger(__name__)

    model = ClimateNet(
        static_channels=config["model"]["static_channels"],
        dynamic_channels=config["model"]["dynamic_channels"],
        image_size=tuple(config["model"]["image_size"]),
        encoder_type=config["model"]["encoder_type"],
        encoder_dim=config["model"]["encoder_dim"],
        encoder_blocks=config["model"]["encoder_blocks"],
        decoder_hidden_dims=config["model"]["decoder_hidden_dims"],
        target_vars=config["model"]["target_vars"],
        use_film=config["model"]["use_film"],
        num_leads=config["model"]["num_leads"],
        lead_embed_dim=config["model"]["lead_embed_dim"],
    )

    # Add ViT-specific parameters if using ViT encoder
    if config["model"]["encoder_type"] == "vit":
        # These will be passed to VisionTransformerEncoder
        pass

    param_counts = model.count_parameters()
    logger.info(f"Model created: {param_counts['total_millions']:.2f}M parameters")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train ClimateNet with multi-GPU support"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires --resume)",
    )

    args = parser.parse_args()

    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()

    # Load config
    config = load_config(args.config)

    # Adjust batch size for distributed training
    if is_distributed:
        # Each GPU gets batch_size samples, so effective batch = batch_size * world_size
        print(f"Rank {rank}: Distributed training with {world_size} GPUs")
        print(f"Per-GPU batch size: {config['training']['batch_size']}")
        print(f"Effective batch size: {config['training']['batch_size'] * world_size}")

    # Create experiment directory (only rank 0)
    if rank == 0:
        exp_dir = create_experiment_directory(config, rank)
    else:
        exp_dir = None

    # Broadcast experiment directory to all ranks
    if is_distributed:
        if rank == 0:
            exp_dir_str = str(exp_dir)
        else:
            exp_dir_str = None

        # Create list to store broadcasted value
        exp_dir_list = [exp_dir_str]
        dist.broadcast_object_list(exp_dir_list, src=0)

        if rank != 0:
            exp_dir = Path(exp_dir_list[0])

    # Setup logging
    logger = setup_logging(exp_dir / "logs", config["logging"]["log_level"], rank)

    if rank == 0:
        logger.info("=" * 70)
        logger.info("CLIMATENET TRAINING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Experiment: {config['experiment']['name']}")
        logger.info(f"Experiment directory: {exp_dir}")
        logger.info(f"Configuration: {args.config}")
        if is_distributed:
            logger.info(f"Distributed training: {world_size} GPUs")
        logger.info("=" * 70)

    # Save configuration (only rank 0)
    if rank == 0:
        config_save_path = exp_dir / "config.yaml"
        save_config(config, config_save_path)
        logger.info(f"Configuration saved to: {config_save_path}")

    # Set random seeds
    seed = config["seed"] + rank  # Different seed per rank for data augmentation
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create data loaders
    if rank == 0:
        logger.info("\n" + "=" * 70)
        logger.info("DATA LOADING")
        logger.info("=" * 70)

    train_loader, val_loader, test_loader, data_loader = create_data_loaders(
        config, exp_dir, rank, world_size, is_distributed
    )

    # Create model
    if rank == 0:
        logger.info("\n" + "=" * 70)
        logger.info("MODEL CREATION")
        logger.info("=" * 70)

    model = create_model(config)

    # Move model to GPU
    device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
    model = model.to(device)

    # Wrap model with DDP
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # Create loss function
    loss_fn = CombinedLoss(
        target_vars=config["model"]["target_vars"],
        weighting_strategy=config["loss"]["weighting_strategy"],
        physics_weight=config["loss"]["physics_weight"],
        use_physics=config["loss"]["use_physics"],
        use_clausius_clapeyron=config["loss"]["use_clausius_clapeyron"],
        use_temp_consistency=config["loss"]["use_temp_consistency"],
        use_humidity_bounds=config["loss"]["use_humidity_bounds"],
        use_precip_nonnegativity=config["loss"]["use_precip_nonnegativity"],
        use_spatial_smoothness=config["loss"].get("use_spatial_smoothness", False),
    ).to(device)

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=tuple(config["optimizer"]["betas"]),
    )

    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["max_epochs"],
        eta_min=config["scheduler"]["min_lr"],
    )

    if not args.eval_only:
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_epochs=config["training"]["max_epochs"],
            stage1_epochs=config["training"]["stage1_epochs"],
            gradient_clip_val=config["training"]["gradient_clip_val"],
            early_stopping_patience=config["training"]["early_stopping_patience"],
            checkpoint_dir=str(exp_dir / "checkpoints"),
            save_every_n_epochs=config["training"]["save_every_n_epochs"],
            use_amp=config["training"]["use_amp"],
            log_interval=config["logging"]["log_interval"],
            val_interval=config["training"]["val_interval"],
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )

        # Train
        if rank == 0:
            logger.info("\n" + "=" * 70)
            logger.info("TRAINING")
            logger.info("=" * 70)

        trainer.train(resume_from=args.resume)

        best_model_path = exp_dir / "checkpoints" / "best_model.pt"
    else:
        if args.resume is None:
            raise ValueError("--resume must be specified when using --eval-only")
        best_model_path = Path(args.resume)

    # Evaluation (only on rank 0)
    if rank == 0:
        logger.info("\n" + "=" * 70)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 70)

        # Unwrap DDP model for evaluation
        eval_model = model.module if is_distributed else model

        evaluator = Evaluator(
            model=eval_model,
            test_loader=test_loader,
            device=device,
            target_vars=config["model"]["target_vars"],
            results_dir=exp_dir / "results",
            figures_dir=exp_dir / "figures",
        )

        logger.info(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)

        # Handle DDP checkpoint
        if is_distributed:
            eval_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            eval_model.load_state_dict(checkpoint["model_state_dict"])

        logger.info("Running evaluation on test set...")
        test_results = evaluator.evaluate(
            save_predictions=config["evaluation"]["save_predictions"]
        )

        logger.info("\n" + "=" * 70)
        logger.info("TEST SET RESULTS")
        logger.info("=" * 70)
        for var in config["model"]["target_vars"]:
            logger.info(f"\n{var}:")
            logger.info(f"  RMSE: {test_results['metrics'][var]['rmse']:.4f}")
            logger.info(f"  MAE: {test_results['metrics'][var]['mae']:.4f}")
            logger.info(f"  R²: {test_results['metrics'][var]['r2']:.4f}")
            logger.info(f"  Bias: {test_results['metrics'][var]['bias']:.4f}")

        logger.info(f"\nAll results saved to: {exp_dir}")

    # Cleanup
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
