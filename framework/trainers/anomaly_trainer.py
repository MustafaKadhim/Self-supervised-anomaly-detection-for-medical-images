"""
Training loop for the self-supervised anomaly detection autoencoder.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from framework.models import AnomalyAutoencoder
from framework.losses import AnomalyLoss

logger = logging.getLogger(__name__)


class AnomalyTrainer:
    """
    Trainer for the self-supervised anomaly detection autoencoder.

    Handles the full training lifecycle:
      - Gradient updates with the combined anomaly loss
      - Periodic checkpoint saving
      - Learning-rate scheduling
      - Training / validation metric logging

    Args:
        model (AnomalyAutoencoder): The model to train.
        train_loader (DataLoader): DataLoader providing healthy training images.
        val_loader (DataLoader, optional): DataLoader for validation images.
        lr (float): Initial learning rate.
        num_epochs (int): Total number of training epochs.
        checkpoint_dir (str | Path): Directory to save model checkpoints.
        device (str): Device identifier ('cuda', 'cpu', etc.).
        l1_weight (float): Weight for the L1 reconstruction loss term.
        ssim_weight (float): Weight for the SSIM loss term.
        perceptual_weight (float): Weight for the perceptual loss term.
    """

    def __init__(
        self,
        model: AnomalyAutoencoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        num_epochs: int = 100,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda",
        l1_weight: float = 1.0,
        ssim_weight: float = 1.0,
        perceptual_weight: float = 0.1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = AnomalyLoss(
            l1_weight=l1_weight,
            ssim_weight=ssim_weight,
            perceptual_weight=perceptual_weight,
        )
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=lr / 100)

        self.best_val_loss = float("inf")
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_l1": [], "train_ssim": [],
        }

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {"total": 0, "l1": 0, "ssim": 0, "perceptual": 0}
        n = 0

        for images, _ in self.train_loader:
            images = images.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(images)
            losses = self.criterion(out["reconstruction"], images)
            losses["total"].backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = images.size(0)
            for k in totals:
                totals[k] += losses[k].item() * batch_size
            n += batch_size

        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def _val_epoch(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total, n = 0.0, 0
        for images, _ in self.val_loader:
            images = images.to(self.device)
            out = self.model(images)
            loss = self.criterion(out["reconstruction"], images)["total"]
            total += loss.item() * images.size(0)
            n += images.size(0)
        return total / n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, list]:
        """
        Run the full training loop.

        Returns:
            Training history dictionary with per-epoch metrics.
        """
        logger.info("Starting training on device: %s", self.device)
        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch()
            val_loss = self._val_epoch()
            self.scheduler.step()

            self.history["train_loss"].append(train_metrics["total"])
            self.history["val_loss"].append(val_loss)
            self.history["train_l1"].append(train_metrics["l1"])
            self.history["train_ssim"].append(train_metrics["ssim"])

            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d | train=%.4f | val=%.4f | lr=%.2e | %.1fs",
                epoch, self.num_epochs, train_metrics["total"], val_loss,
                self.optimizer.param_groups[0]["lr"], elapsed,
            )

            self._save_checkpoint(epoch, val_loss)

        return self.history

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        torch.save(state, latest_path)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(state, best_path)
            logger.info("  ↳ New best model saved (val_loss=%.4f)", val_loss)

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint and return the epoch number."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.best_val_loss = state.get("val_loss", float("inf"))
        logger.info("Loaded checkpoint from epoch %d", state["epoch"])
        return state["epoch"]
