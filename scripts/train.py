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


def load_state_dict_flexible(model: nn.Module, state_dict: dict) -> None:
    """
    Load a checkpoint state_dict while handling DDP/non-DDP key prefix differences.

    Supports both key styles:
    - with "module." prefix (DDP-wrapped state dict)
    - without "module." prefix (plain module state dict)
    """
    if not state_dict:
        raise ValueError("Empty state_dict provided")

    model_state_keys = list(model.state_dict().keys())
    checkpoint_keys = list(state_dict.keys())

    model_uses_module_prefix = model_state_keys[0].startswith("module.")
    checkpoint_uses_module_prefix = checkpoint_keys[0].startswith("module.")

    adjusted_state_dict = state_dict
    if checkpoint_uses_module_prefix and not model_uses_module_prefix:
        adjusted_state_dict = {
            key.replace("module.", "", 1): value for key, value in state_dict.items()
        }
    elif not checkpoint_uses_module_prefix and model_uses_module_prefix:
        adjusted_state_dict = {f"module.{key}": value for key, value in state_dict.items()}

    model.load_state_dict(adjusted_state_dict)


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


def create_experiment_directory(
    config: dict, rank: int = 0, resume_path: str = None
) -> Path:
    """
    Create experiment directory (only on rank 0).
    """
    if rank == 0:
        # When resuming, keep using the SAME experiment directory.
        if resume_path is not None:
            checkpoint_path = Path(resume_path).resolve()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

            # Expected layout: <exp_dir>/checkpoints/<checkpoint>.pt
            if checkpoint_path.parent.name == "checkpoints":
                exp_dir = checkpoint_path.parent.parent
            else:
                # Fallback: treat checkpoint parent as experiment directory
                exp_dir = checkpoint_path.parent

            # Ensure expected subdirs exist (idempotent for existing runs)
            (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (exp_dir / "logs").mkdir(exist_ok=True)
            (exp_dir / "results").mkdir(exist_ok=True)
            (exp_dir / "figures").mkdir(exist_ok=True)

            return exp_dir

        experiment_cfg = config["experiment"]
        configured_output_dir = experiment_cfg.get("output_dir")

        if configured_output_dir:
            exp_dir = Path(configured_output_dir)
        else:
            base_dir = Path(experiment_cfg["base_dir"])
            experiment_name = experiment_cfg["name"]
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
    data_loader,
    train_years: list,
    val_years: list,
    test_years: list,
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

    logger.info(f"Split sizes:")
    logger.info(f"  Train: {len(train_indices):,} samples")
    logger.info(f"  Val: {len(val_indices):,} samples")
    logger.info(f"  Test: {len(test_indices):,} samples")

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def setup_distributed(
    rank: int, world_size: int, backend: str = "nccl", local_rank: int = None
):
    """
    Initialize distributed training.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        device_idx = rank if local_rank is None else local_rank
        torch.cuda.set_device(device_idx)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def main(
    rank: int,
    world_size: int,
    config_path: str,
    resume_path: str = None,
    local_rank: int = None,
):
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
        setup_distributed(rank, world_size, local_rank=local_rank)

    # Create experiment directory (only rank 0)
    exp_dir = create_experiment_directory(config, rank, resume_path=resume_path)

    # Broadcast experiment directory path to all ranks
    if world_size > 1:
        exp_dir_list = [exp_dir]
        dist.broadcast_object_list(exp_dir_list, src=0)
        exp_dir = exp_dir_list[0]

    # Setup logger
    logger = setup_logger(
        exp_dir / "logs",
        rank=rank,
        log_level=config.get("logging", {}).get(
            "level", config.get("logging", {}).get("log_level", "INFO")
        ),
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
    if world_size > 1:
        device_idx = rank if local_rank is None else local_rank
        device = f"cuda:{device_idx}"
    else:
        device = config["training"].get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    import random
    random.seed(seed + rank)
    # If CUDA is available, set CUDA seeds and deterministic flags
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)

    # Control cudnn determinism (configurable)
    cudnn_cfg = config.get("training", {}).get("cudnn", {})
    # Expected keys: 'deterministic' (bool), 'benchmark' (bool)
    deterministic = cudnn_cfg.get("deterministic", True)
    benchmark = cudnn_cfg.get("benchmark", False)
    try:
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(benchmark)
    except Exception:
        # Older/newer PyTorch may not allow changes; ignore safely
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    logger.info("=" * 70)
    logger.info("DATA LOADING")
    logger.info("=" * 70)

    # Optional mode: rank 0 serves batches to all other ranks via broadcast.
    rank0_batch_broadcast = config["data"].get("rank0_batch_broadcast", False)

    # Create data loader
    # Optional mode: only rank 0 performs in-memory loading while other ranks
    # use lazy loading from disk. Default behavior remains unchanged.
    rank0_only_in_memory = config["data"].get("rank0_only_in_memory_load", False)

    if world_size > 1 and rank0_only_in_memory:
        if rank == 0:
            logger.info(
                "rank0_only_in_memory_load enabled: rank 0 loads in memory, other ranks use lazy loading"
            )
            data_loader = DecadalDataLoader(
                nc_path=config["data"]["nc_path"],
                normalize_method=config["data"]["normalize_method"],
                cache_dir=config["data"]["cache_dir"],
                load_in_memory=config["data"]["load_in_memory"],
            )

        # Ensure rank 0 finishes potential cache creation first.
        dist.barrier()

        if rank != 0:
            data_loader = DecadalDataLoader(
                nc_path=config["data"]["nc_path"],
                normalize_method=config["data"]["normalize_method"],
                cache_dir=config["data"]["cache_dir"],
                load_in_memory=False,
            )
    else:
        data_loader = DecadalDataLoader(
            nc_path=config["data"]["nc_path"],
            normalize_method=config["data"]["normalize_method"],
            cache_dir=config["data"]["cache_dir"],
            load_in_memory=config["data"]["load_in_memory"],
        )

    train_loader = None
    val_loader = None
    test_loader = None

    # Variables to normalize
    static_vars = ["dem", "rho", "phi"]
    dynamic_vars = ["pr", "tas", "tasmax", "hurs", "sin_time", "cos_time"]
    cci_vars = ["cci_agg"]  # CCI aggregate (10 classes, normalized together)
    target_vars = config["model"]["target_vars"]

    if world_size > 1 and rank0_batch_broadcast and rank != 0:
        # Non-zero ranks skip dataset iteration and receive batches from rank 0.
        train_indices = []
        val_indices = []
        test_indices = []
    else:
        # Create splits before normalization so scalers are fitted on TRAIN years only.
        splits = create_year_based_splits(
            data_loader=data_loader,
            train_years=config["data"]["train_years"],
            val_years=config["data"]["val_years"],
            test_years=config["data"]["test_years"],
        )

        train_indices = splits["train"]
        val_indices = splits["val"]
        test_indices = splits["test"]

    # Fit normalizer on training set (only rank 0 needs to do this)
    if config["data"]["normalize"] and rank == 0 and data_loader is not None:
        logger.info("=" * 70)
        logger.info("FITTING NORMALIZER ON TRAINING DATA")
        logger.info("=" * 70)

        # Determine sample size for fitting
        total_samples = len(train_indices)
        if total_samples == 0:
            raise ValueError(
                "No training samples found for configured train_years; cannot fit normalizer"
            )
        sample_size = min(config["data"].get("normalize_samples", 40000), total_samples)

        logger.info(f"Total available samples: {total_samples:,}")
        logger.info(f"Using {sample_size:,} samples for normalization fitting")

        # Sample uniformly across the training set only
        sample_positions = np.linspace(0, total_samples - 1, sample_size, dtype=int)
        sample_indices = [train_indices[pos] for pos in sample_positions]

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

                # Static vars (indices 4-6: dem, rho, phi)
                for i, var in enumerate(static_vars):
                    var_values[var].append(inputs[i + 4].flatten())

                # Dynamic vars (indices 0-3, 7-8: pr, tas, tasmax, hurs, sin_time, cos_time)
                for i, var in enumerate(dynamic_vars):
                    if i < 4:
                        var_values[var].append(inputs[i].flatten())
                    else:
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
        if data_loader is not None:
            data_loader.scalers = scalers_list[0]

        if rank != 0:
            logger.info("Received normalization parameters from rank 0")

    if not (world_size > 1 and rank0_batch_broadcast and rank != 0):
        # Create full dataset
        if rank == 0:
            logger.info("Creating datasets...")

        full_dataset = ClimateDataset(
            data_loader=data_loader,
            normalize=config["data"]["normalize"],
            image_size=config["model"]["image_size"],
            target_vars=config["model"]["target_vars"],
        )

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

        if rank == 0:
            logger.info(f"Dataset split:")
            logger.info(f"  Train: {len(train_dataset):,} samples")
            logger.info(f"  Val:   {len(val_dataset):,} samples")
            logger.info(f"  Test:  {len(test_dataset):,} samples")

            # Save split indices
            split_path = exp_dir / "data_splits.json"
            with open(split_path, "w") as f:
                json.dump(
                    {
                        "train_indices": train_indices,
                        "val_indices": val_indices,
                        "test_indices": test_indices,
                        "train_years": config["data"]["train_years"],
                        "val_years": config["data"]["val_years"],
                        "test_years": config["data"]["test_years"],
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Data splits saved to: {split_path}")

        # Create data loaders
        if world_size > 1 and not rank0_batch_broadcast:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
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
    else:
        if rank == 0:
            logger.info("Using rank0_batch_broadcast mode for distributed training")

    # =========================================================================
    # MODEL CREATION
    # =========================================================================
    if rank == 0:
        logger.info("=" * 70)
        logger.info("MODEL CREATION")
        logger.info("=" * 70)

    model = ClimateNet(
        static_channels=config["model"]["static_channels"],
        dynamic_channels=config["model"]["dynamic_channels"],
        image_size=config["model"]["image_size"],
        encoder_type=config["model"]["encoder_type"],
        encoder_dim=config["model"]["encoder_dim"],
        encoder_blocks=config["model"]["encoder_blocks"],
        vit_patch_size=config["model"].get("vit_patch_size", 7),
        vit_num_heads=config["model"].get("vit_num_heads", 8),
        vit_mlp_ratio=config["model"].get("vit_mlp_ratio", 4.0),
        vit_dropout=config["model"].get("vit_dropout", 0.1),
        vit_attention_dropout=config["model"].get("vit_attention_dropout", 0.1),
        decoder_hidden_dims=config["model"]["decoder_hidden_dims"],
        target_vars=config["model"]["target_vars"],
        output_activations=config["model"].get("output_activations", None),
        use_film=config["model"]["use_film"],
        num_leads=config["model"]["num_leads"],
        lead_embed_dim=config["model"]["lead_embed_dim"],
    )

    model = model.to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters:")
        logger.info(f"  Total:     {total_params:,}")
        logger.info(f"  Trainable: {trainable_params:,}")
        logger.info(f"  Architecture: {config['model']['encoder_type']}")
        logger.info(f"  Image size: {config['model']['image_size']}")

    # Wrap model with DDP if using multiple GPUs
    if world_size > 1:
        ddp_device_idx = rank if local_rank is None else local_rank
        model = DDP(model, device_ids=[ddp_device_idx], output_device=ddp_device_idx)
        if rank == 0:
            logger.info(f"Model wrapped with DistributedDataParallel")

    # =========================================================================
    # LOSS FUNCTION
    # =========================================================================
    if rank == 0:
        logger.info("=" * 70)
        logger.info("LOSS FUNCTION")
        logger.info("=" * 70)

    criterion = CombinedLoss(
        target_vars=config["model"]["target_vars"],
        weighting_strategy=config["loss"]["weighting_strategy"],
        physics_weight=config["loss"].get("physics_weight", 0.1),
        use_physics=config["loss"].get("use_physics", False),
        data_loss_types=config["loss"].get("data_loss_types", None),
        tweedie_power=config["loss"].get("tweedie_power", 1.5),
        tweedie_eps=config["loss"].get("tweedie_eps", 1e-6),
        use_clausius_clapeyron=config["loss"].get("use_clausius_clapeyron", False),
        use_temp_consistency=config["loss"].get("use_temp_consistency", False),
        use_humidity_bounds=config["loss"].get("use_humidity_bounds", False),
        use_precip_nonnegativity=config["loss"].get(
            "use_precip_nonnegativity", False
        ),
        use_spatial_smoothness=config["loss"].get("use_spatial_smoothness", False),
    ).to(device)

    if rank == 0:
        logger.info(f"Loss configuration:")
        logger.info(f"  Strategy: {config['loss']['weighting_strategy']}")

    # =========================================================================
    # OPTIMIZER & SCHEDULER
    # =========================================================================
    if rank == 0:
        logger.info("=" * 70)
        logger.info("OPTIMIZER & SCHEDULER")
        logger.info("=" * 70)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        betas=tuple(config["optimizer"].get("betas", [0.9, 0.999])),
    )

    scheduler = None
    scheduler_type = config.get("scheduler", {}).get("type", "none")
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["max_epochs"],
            eta_min=config.get("scheduler", {}).get("min_lr", 1e-6),
        )
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.get("scheduler", {}).get("factor", 0.5),
            patience=config.get("scheduler", {}).get("patience", 5),
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("scheduler", {}).get("step_size", 30),
            gamma=config.get("scheduler", {}).get("gamma", 0.1),
        )

    if rank == 0:
        logger.info(f"Optimizer: AdamW")
        logger.info(f"  Learning rate: {config['optimizer']['lr']}")
        logger.info(f"  Weight decay: {config['optimizer']['weight_decay']}")
        if scheduler:
            logger.info(f"Scheduler: {scheduler.__class__.__name__}")

    # =========================================================================
    # TRAINER
    # =========================================================================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=config["training"]["max_epochs"],
        stage1_epochs=config["training"]["stage1_epochs"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        save_every_n_epochs=config["training"]["save_every_n_epochs"],
        use_amp=config["training"].get("use_amp", True),
        log_interval=config.get("logging", {}).get("log_interval", 50),
        val_interval=config["training"]["val_interval"],
        checkpoint_dir=exp_dir / "checkpoints",
        rank=rank,
        world_size=world_size,
        is_distributed=world_size > 1,
        use_rank0_batch_broadcast=rank0_batch_broadcast,
    )

    # =========================================================================
    # TRAINING
    # =========================================================================
    if rank == 0:
        logger.info("=" * 70)
        logger.info("STARTING TRAINING")
        logger.info("=" * 70)

    trainer.train(resume_from=resume_path)

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
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            checkpoint_state_dict = checkpoint.get("model_state_dict", checkpoint)
            target_model = model.module if world_size > 1 else model
            load_state_dict_flexible(target_model, checkpoint_state_dict)

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
    if data_loader is not None:
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

    # torchrun path (WORLD_SIZE/RANK/LOCAL_RANK are set by launcher)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world_size > 1:
        env_rank = int(os.environ["RANK"])
        env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        main(
            rank=env_rank,
            world_size=env_world_size,
            config_path=args.config,
            resume_path=args.resume,
            local_rank=env_local_rank,
        )
        sys.exit(0)

    # Single GPU or CPU training
    if args.gpus <= 1:
        main(
            rank=0,
            world_size=1,
            config_path=args.config,
            resume_path=args.resume,
            local_rank=0,
        )
    # Multi-GPU training
    else:
        import torch.multiprocessing as mp

        mp.spawn(
            main,
            args=(args.gpus, args.config, args.resume, None),
            nprocs=args.gpus,
            join=True,
        )
