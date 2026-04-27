"""
Train the anomaly detection model on Brain MRI data.

Usage
-----
    python experiments/brain_mri/train.py
    python experiments/brain_mri/train.py --config experiments/brain_mri/config.yaml
    python experiments/brain_mri/train.py --epochs 200 --lr 5e-5
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from framework.datasets import MedicalImageDataset
from framework.models import AnomalyAutoencoder
from framework.trainers import AnomalyTrainer
from framework.utils import plot_training_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_loaders(cfg: dict):
    data_cfg = cfg["data"]
    full_dataset = MedicalImageDataset(
        root=data_cfg["root"],
        split="train",
        grayscale=data_cfg.get("grayscale", True),
        image_size=data_cfg.get("image_size", 128),
    )
    n_train = int(len(full_dataset) * data_cfg.get("train_val_split", 0.9))
    n_val = len(full_dataset) - n_train
    train_set, val_set = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["experiment"]["seed"]),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )
    return train_loader, val_loader


def main(args):
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr

    logging.basicConfig(
        level=getattr(logging, cfg["output"].get("log_level", "INFO")),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    set_seed(cfg["experiment"]["seed"])
    logger.info("Experiment: %s", cfg["experiment"]["name"])

    train_loader, val_loader = build_loaders(cfg)
    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    model_cfg = cfg["model"]
    model = AnomalyAutoencoder(
        in_channels=model_cfg.get("in_channels", 1),
        latent_dim=model_cfg.get("latent_dim", 512),
        base_channels=model_cfg.get("base_channels", 32),
        use_skip=model_cfg.get("use_skip", True),
    )
    logger.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    train_cfg = cfg["training"]
    trainer = AnomalyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=train_cfg["learning_rate"],
        num_epochs=train_cfg["num_epochs"],
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        l1_weight=train_cfg.get("l1_weight", 1.0),
        ssim_weight=train_cfg.get("ssim_weight", 1.5),
        perceptual_weight=train_cfg.get("perceptual_weight", 0.2),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    history = trainer.train()

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, save_path=str(results_dir / "training_curves.png"))
    logger.info("Training complete. Results saved to %s", results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain MRI anomaly detection model")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    main(parser.parse_args())
