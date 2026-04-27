"""
Anomaly Autoencoder — the core model combining Encoder and Decoder.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from framework.models.encoder import Encoder
from framework.models.decoder import Decoder


class AnomalyAutoencoder(nn.Module):
    """
    Self-Supervised Anomaly Detection Autoencoder.

    The model is trained exclusively on healthy (normal) medical images using
    reconstruction loss. At inference time, anomalous regions yield high
    reconstruction errors that are used to generate pixel-level anomaly maps.

    Architecture
    ------------
    Input Image  →  Encoder  →  Latent z  →  Decoder  →  Reconstructed Image
                       ↓                           ↑
                  skip features  ─────────────────

    Args:
        in_channels (int): Number of input image channels.
        latent_dim (int): Dimensionality of the bottleneck latent space.
        base_channels (int): Base number of feature maps in encoder/decoder.
        use_skip (bool): Use U-Net skip connections for better reconstruction.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        base_channels: int = 32,
        use_skip: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            base_channels=base_channels,
            use_skip=use_skip,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Dictionary with keys:
                - 'reconstruction': Reconstructed image tensor.
                - 'latent': Latent code vector.
                - 'anomaly_map': Per-pixel L1 anomaly score map.
        """
        z, features = self.encoder(x)
        reconstruction = self.decoder(z, features)

        # Upsample reconstruction to match input if needed
        if reconstruction.shape != x.shape:
            reconstruction = nn.functional.interpolate(
                reconstruction, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        anomaly_map = torch.abs(x - reconstruction)

        return {
            "reconstruction": reconstruction,
            "latent": z,
            "anomaly_map": anomaly_map,
        }

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-image anomaly score and pixel-level anomaly map.

        Args:
            x: Input image tensor.

        Returns:
            Tuple of (anomaly_scores, anomaly_maps) where anomaly_scores has
            shape (B,) and anomaly_maps has shape (B, 1, H, W).
        """
        self.eval()
        out = self.forward(x)
        anomaly_map = out["anomaly_map"]
        score = anomaly_map.mean(dim=[1, 2, 3])
        return score, anomaly_map
