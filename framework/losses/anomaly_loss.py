"""
Loss functions for self-supervised anomaly detection training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss for perceptually faithful reconstruction.

    Args:
        window_size (int): Size of the Gaussian window used in SSIM computation.
        reduction (str): 'mean' or 'sum'.
    """

    def __init__(self, window_size: int = 11, reduction: str = "mean"):
        super().__init__()
        self.window_size = window_size
        self.reduction = reduction
        self.register_buffer("window", self._create_window(window_size))

    @staticmethod
    def _create_window(window_size: int) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)
        return window.unsqueeze(0).unsqueeze(0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.shape[1]
        window = self.window.expand(channels, 1, -1, -1).to(pred.device)

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channels)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channels) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        loss = 1.0 - ssim_map
        return loss.mean() if self.reduction == "mean" else loss.sum()


class PerceptualLoss(nn.Module):
    """
    Feature-level perceptual loss using intermediate VGG-style activations.
    For medical images we use a lightweight multi-scale feature extractor.

    Args:
        weight (float): Scaling factor for the perceptual loss term.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        for p in self.extractor.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle multi-channel by converting to grayscale
        if pred.shape[1] > 1:
            pred = pred.mean(dim=1, keepdim=True)
            target = target.mean(dim=1, keepdim=True)
        f_pred = self.extractor(pred)
        f_target = self.extractor(target)
        return self.weight * F.l1_loss(f_pred, f_target)


class AnomalyLoss(nn.Module):
    """
    Combined reconstruction loss for training the anomaly detection autoencoder.

    Combines:
      - L1 reconstruction loss  (pixel-level fidelity)
      - SSIM loss               (structural similarity)
      - Perceptual loss         (feature-level fidelity)

    Args:
        l1_weight (float): Weight of the L1 loss term.
        ssim_weight (float): Weight of the SSIM loss term.
        perceptual_weight (float): Weight of the perceptual loss term.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 1.0,
        perceptual_weight: float = 0.1,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual = PerceptualLoss(weight=perceptual_weight)
        self.ssim = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred: Reconstructed image tensor (B, C, H, W).
            target: Ground-truth image tensor (B, C, H, W).

        Returns:
            Dictionary with individual loss components and total loss.
        """
        l1 = F.l1_loss(pred, target)
        ssim = self.ssim(pred, target)
        perceptual = self.perceptual(pred, target)

        total = self.l1_weight * l1 + self.ssim_weight * ssim + perceptual

        return {
            "total": total,
            "l1": l1,
            "ssim": ssim,
            "perceptual": perceptual,
        }
