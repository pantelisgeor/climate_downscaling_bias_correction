"""
Training script for decadal high-resolution downscaling.

Adapted from scripts/train.py for the DecadalDownscaleDataLoader.

Key differences from train.py:
  • Uses DecadalDownscaleDataLoader — disk-based, no in-memory load option.
    All ranks read from disk concurrently via per-process LRU file handles.
  • Uses ClimateDatasetDownscale   — thin wrapper; data already normalized.
  • Normalization is pre-computed (per-year minmax CSVs) — no fitting step.
  • Year/run/lead splits via data_loader.get_split_indices().
  • No rank0_only_in_memory_load logic needed.
  • Image size is 270×520 instead of 35×77.
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader_decadal_downscale import DecadalDownscaleDataLoader
from src.training_downscale.climate_dataset_downscale import ClimateDatasetDownscale
from src.models.climate_net import ClimateNet
from src.losses.task_weighting import CombinedLoss
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


# ── helpers ───────────────────────────────────────────────────────────────────

def load_state_dict_flexible(model: nn.Module, state_dict: dict) -> None:
    model_keys = list(model.state_dict().keys())
    ckpt_keys  = list(state_dict.keys())
    model_has_mod = model_keys[0].startswith("module.")
    ckpt_has_mod  = ckpt_keys[0].startswith("module.")
    adj = state_dict
    if ckpt_has_mod and not model_has_mod:
        adj = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not ckpt_has_mod and model_has_mod:
        adj = {f"module.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(adj)


def setup_logger(log_dir: Path, rank: int = 0,
                 log_level: str = "INFO") -> logging.Logger:
    if rank != 0:
        logging.basicConfig(level=logging.ERROR)
        return logging.getLogger(__name__)
    log_dir.mkdir(parents=True, exist_ok=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: Path):
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_experiment_directory(config: dict, rank: int = 0,
                                resume_path: str = None) -> Path:
    if rank == 0:
        if resume_path is not None:
            cp = Path(resume_path).resolve()
            if not cp.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {cp}")
            exp_dir = cp.parent.parent if cp.parent.name == "checkpoints" else cp.parent
            for d in ("checkpoints", "logs", "results", "figures"):
                (exp_dir / d).mkdir(parents=True, exist_ok=True)
            return exp_dir
        cfg = config["experiment"]
        if cfg.get("output_dir"):
            exp_dir = Path(cfg["output_dir"])
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = Path(cfg["base_dir"]) / f"{cfg['name']}_{ts}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        for d in ("checkpoints", "logs", "results", "figures"):
            (exp_dir / d).mkdir(exist_ok=True)
        return exp_dir
    return None


def setup_distributed(rank: int, world_size: int,
                      backend: str = "nccl", local_rank: int = None):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank if local_rank is None else local_rank)


def cleanup_distributed():
    dist.destroy_process_group()


# ── main ──────────────────────────────────────────────────────────────────────

def main(rank: int, world_size: int, config_path: str,
         resume_path: str = None, local_rank: int = None):

    config = load_config(config_path)

    if world_size > 1:
        setup_distributed(rank, world_size, local_rank=local_rank)

    exp_dir = create_experiment_directory(config, rank, resume_path=resume_path)

    if world_size > 1:
        exp_dir_list = [exp_dir]
        dist.broadcast_object_list(exp_dir_list, src=0)
        exp_dir = exp_dir_list[0]

    logger = setup_logger(
        exp_dir / "logs", rank=rank,
        log_level=config.get("logging", {}).get(
            "level", config.get("logging", {}).get("log_level", "INFO")
        ),
    )

    if rank == 0:
        save_config(config, exp_dir / "config.yaml")
        logger.info("=" * 70)
        logger.info("DECADAL DOWNSCALE TRAINING")
        logger.info("=" * 70)
        logger.info(f"Experiment dir: {exp_dir}")
        logger.info(f"Config:         {config_path}")
        if resume_path:
            logger.info(f"Resuming from:  {resume_path}")

    # ── device ────────────────────────────────────────────────────────────────
    if world_size > 1:
        device_idx = rank if local_rank is None else local_rank
        device = f"cuda:{device_idx}"
    else:
        device = config["training"].get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    logger.info(f"Device: {device}")

    # ── seeds ─────────────────────────────────────────────────────────────────
    seed = config.get("seed", 42)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    cu = config.get("training", {}).get("cudnn", {})
    try:
        torch.backends.cudnn.deterministic = bool(cu.get("deterministic", True))
        torch.backends.cudnn.benchmark     = bool(cu.get("benchmark", False))
    except Exception:
        pass

    # ── data loader ───────────────────────────────────────────────────────────
    # DecadalDownscaleDataLoader is disk-based; all ranks set up identically.
    # Normalization is pre-computed from per-year minmax CSVs; no fitting step.
    logger.info("=" * 70)
    logger.info("DATA LOADING")
    logger.info("=" * 70)

    data_loader = DecadalDownscaleDataLoader(
        data_dir    = config["data"]["data_dir"],
        cache_dir   = config["data"]["cache_dir"],
        train_years = config["data"]["train_years"],
        force_reindex = config["data"].get("force_reindex", False),
    )

    if rank == 0:
        logger.info(f"Loader ready: {len(data_loader):,} total samples")
        logger.info(f"  Runs:  {data_loader.available_runs()}")
        logger.info(f"  Leads: {data_loader.available_leads()}")
        logger.info(f"  Years: {data_loader.available_years()[0]}–"
                    f"{data_loader.available_years()[-1]}")

    # ── splits ────────────────────────────────────────────────────────────────
    train_indices = data_loader.get_split_indices(config["data"]["train_years"])
    val_indices   = data_loader.get_split_indices(config["data"]["val_years"])
    test_indices  = data_loader.get_split_indices(config["data"]["test_years"])

    if rank == 0:
        logger.info(
            f"Splits — train={len(train_indices):,}  "
            f"val={len(val_indices):,}  test={len(test_indices):,}"
        )
        split_path = exp_dir / "data_splits.json"
        with open(split_path, "w") as f:
            json.dump({
                "train_size":  len(train_indices),
                "val_size":    len(val_indices),
                "test_size":   len(test_indices),
                "train_years": config["data"]["train_years"],
                "val_years":   config["data"]["val_years"],
                "test_years":  config["data"]["test_years"],
            }, f, indent=2)

    # ── datasets & loaders ────────────────────────────────────────────────────
    target_vars = config["model"]["target_vars"]

    full_dataset = ClimateDatasetDownscale(
        data_loader = data_loader,
        normalize   = False,          # loader already normalizes
        image_size  = config["model"]["image_size"],
        target_vars = target_vars,
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)
    test_dataset  = Subset(full_dataset, test_indices)

    nw = config["data"].get("num_workers", 4)
    bs = config["training"]["batch_size"]

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, world_size, rank, shuffle=True, seed=seed)
        val_sampler = DistributedSampler(
            val_dataset, world_size, rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler   = None

    pw = nw > 0   # persistent_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size  = bs,
        sampler     = train_sampler,
        shuffle     = (train_sampler is None),
        num_workers = nw,
        pin_memory  = True,
        drop_last   = True,
        persistent_workers = pw,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = bs,
        sampler     = val_sampler,
        shuffle     = False,
        num_workers = nw,
        pin_memory  = True,
        persistent_workers = pw,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = bs,
        shuffle     = False,
        num_workers = nw,
        pin_memory  = True,
        persistent_workers = pw,
    )

    if rank == 0:
        logger.info(
            f"Loaders — train={len(train_loader)} batches  "
            f"val={len(val_loader)}  test={len(test_loader)}"
        )

    # ── model ─────────────────────────────────────────────────────────────────
    model = ClimateNet(
        static_channels      = config["model"]["static_channels"],
        dynamic_channels     = config["model"]["dynamic_channels"],
        image_size           = config["model"]["image_size"],
        encoder_type         = config["model"]["encoder_type"],
        encoder_dim          = config["model"]["encoder_dim"],
        encoder_blocks       = config["model"]["encoder_blocks"],
        vit_patch_size       = config["model"].get("vit_patch_size", 7),
        vit_num_heads        = config["model"].get("vit_num_heads", 8),
        vit_mlp_ratio        = config["model"].get("vit_mlp_ratio", 4.0),
        vit_dropout          = config["model"].get("vit_dropout", 0.1),
        vit_attention_dropout= config["model"].get("vit_attention_dropout", 0.1),
        decoder_type         = config["model"].get("decoder_type", "multi"),
        decoder_hidden_dims  = config["model"]["decoder_hidden_dims"],
        target_vars          = target_vars,
        output_activations   = config["model"].get("output_activations", None),
        use_film             = config["model"]["use_film"],
        num_leads            = config["model"]["num_leads"],
        lead_embed_dim       = config["model"]["lead_embed_dim"],
        dilations            = config["model"].get("dilations", None),
        padding_mode         = config["model"].get("padding_mode", "reflect"),
    ).to(device)

    if rank == 0:
        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model — {tp:,} trainable parameters  ({config['model']['encoder_type']})")

    if world_size > 1:
        di = rank if local_rank is None else local_rank
        model = DDP(model, device_ids=[di], output_device=di)

    # ── loss ──────────────────────────────────────────────────────────────────
    criterion = CombinedLoss(
        target_vars            = target_vars,
        weighting_strategy     = config["loss"]["weighting_strategy"],
        physics_weight         = config["loss"].get("physics_weight", 0.0),
        use_physics            = config["loss"].get("use_physics", False),
        data_loss_types        = config["loss"].get("data_loss_types", None),
        tweedie_power          = config["loss"].get("tweedie_power", 1.5),
        tweedie_eps            = config["loss"].get("tweedie_eps", 1e-6),
        use_clausius_clapeyron = config["loss"].get("use_clausius_clapeyron", False),
        use_temp_consistency   = config["loss"].get("use_temp_consistency", False),
        use_humidity_bounds    = config["loss"].get("use_humidity_bounds", False),
        use_precip_nonnegativity = config["loss"].get("use_precip_nonnegativity", False),
        precip_nonneg_weight   = config["loss"].get("precip_nonneg_weight", 0.2),
        use_spatial_smoothness = config["loss"].get("use_spatial_smoothness", False),
        spatial_smooth_weight  = config["loss"].get("spatial_smooth_weight", 0.01),
        wet_weight             = config["loss"].get("wet_weight", 10.0),
        dry_weight             = config["loss"].get("dry_weight", 1.0),
        compositedry_gamma     = config["loss"].get("compositedry_gamma", 1.0),
        compositedry_overestimate_weight = config["loss"].get("compositedry_overestimate_weight", 2.5),
        compositedry_dry_threshold = config["loss"].get("compositedry_dry_threshold", 0.002),
        compositedry_lambda_extreme = config["loss"].get("compositedry_lambda_extreme", 1.0),
        compositedry_lambda_dry = config["loss"].get("compositedry_lambda_dry", 0.7),
        scalers                = data_loader.scalers,
        normalize_method       = "minmax",
    ).to(device)

    # ── optimizer / scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = config["optimizer"]["lr"],
        weight_decay = config["optimizer"]["weight_decay"],
        betas        = tuple(config["optimizer"].get("betas", [0.9, 0.999])),
    )

    scheduler = None
    sched_type = config.get("scheduler", {}).get("type", "none")
    if sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["max_epochs"],
            eta_min=config.get("scheduler", {}).get("min_lr", 1e-6),
        )
    elif sched_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor   = config.get("scheduler", {}).get("factor", 0.5),
            patience = config.get("scheduler", {}).get("patience", 5),
        )
    elif sched_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.get("scheduler", {}).get("step_size", 30),
            gamma     = config.get("scheduler", {}).get("gamma", 0.1),
        )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model               = model,
        train_loader        = train_loader,
        val_loader          = val_loader,
        loss_fn             = criterion,
        optimizer           = optimizer,
        scheduler           = scheduler,
        device              = device,
        max_epochs          = config["training"]["max_epochs"],
        stage1_epochs       = config["training"]["stage1_epochs"],
        gradient_clip_val   = config["training"]["gradient_clip_val"],
        early_stopping_patience = config["training"]["early_stopping_patience"],
        save_every_n_epochs = config["training"]["save_every_n_epochs"],
        use_amp             = config["training"].get("use_amp", True),
        log_interval        = config.get("logging", {}).get("log_interval", 50),
        val_interval        = config["training"]["val_interval"],
        checkpoint_dir      = exp_dir / "checkpoints",
        rank                = rank,
        world_size          = world_size,
        is_distributed      = world_size > 1,
        use_rank0_batch_broadcast = False,
    )

    trainer.train(resume_from=resume_path)

    # ── final evaluation (rank 0 only) ────────────────────────────────────────
    if rank == 0:
        logger.info("=" * 70)
        logger.info("FINAL EVALUATION ON TEST SET")
        logger.info("=" * 70)

        best_ckpt = exp_dir / "checkpoints" / "best_model.pt"
        if best_ckpt.exists():
            ckpt  = torch.load(best_ckpt, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            target_model = model.module if world_size > 1 else model
            load_state_dict_flexible(target_model, state)

        evaluator = Evaluator(
            model       = model.module if world_size > 1 else model,
            test_loader = test_loader,
            device      = device,
            target_vars = target_vars,
            results_dir = exp_dir / "results",
            figures_dir = exp_dir / "figures",
        )
        evaluator.evaluate(save_predictions=True)

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Results → {exp_dir}")
        logger.info("=" * 70)

    if world_size > 1:
        cleanup_distributed()

    data_loader.close()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train decadal high-resolution downscaling model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gpus",   type=int, default=1)
    args = parser.parse_args()

    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world_size > 1:
        env_rank       = int(os.environ["RANK"])
        env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        main(rank=env_rank, world_size=env_world_size,
             config_path=args.config, resume_path=args.resume,
             local_rank=env_local_rank)
        sys.exit(0)

    if args.gpus <= 1:
        main(rank=0, world_size=1,
             config_path=args.config, resume_path=args.resume, local_rank=0)
    else:
        import torch.multiprocessing as mp
        mp.spawn(
            main,
            args=(args.gpus, args.config, args.resume, None),
            nprocs=args.gpus,
            join=True,
        )
