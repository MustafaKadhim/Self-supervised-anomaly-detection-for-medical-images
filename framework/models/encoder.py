"""
Encoder module for self-supervised anomaly detection.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ResidualBlock(nn.Module):
    """Residual block for the encoder network."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class Encoder(nn.Module):
    """
    Convolutional encoder that maps input images to a compact latent representation.

    The encoder uses residual blocks for stable gradient flow and extracts
    multi-scale features from the input medical image.

    Args:
        in_channels (int): Number of input image channels (1 for grayscale, 3 for RGB).
        latent_dim (int): Dimensionality of the latent space.
        base_channels (int): Number of feature channels in the first encoder block.
        num_blocks (int): Number of residual blocks per resolution level.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        base_channels: int = 32,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        channels = [base_channels * (2 ** i) for i in range(4)]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(channels[0], channels[0], num_blocks, stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], num_blocks, stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], num_blocks, stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], num_blocks, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(channels[3] * 16, latent_dim)

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Tuple of (latent vector, list of intermediate feature maps).
        """
        features = []
        x = self.stem(x)
        x = self.layer1(x); features.append(x)
        x = self.layer2(x); features.append(x)
        x = self.layer3(x); features.append(x)
        x = self.layer4(x); features.append(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z, features
