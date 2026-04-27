"""
Decoder module for self-supervised anomaly detection.
"""

import torch
import torch.nn as nn


class UpBlock(nn.Module):
    """Upsampling block with optional skip connection."""

    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    """
    Convolutional decoder that reconstructs images from a latent representation.

    Uses a U-Net-style skip connection architecture to preserve fine-grained
    spatial information, improving reconstruction quality for anomaly detection.

    Args:
        latent_dim (int): Dimensionality of the input latent vector.
        out_channels (int): Number of output image channels.
        base_channels (int): Number of feature channels in the deepest decoder block.
        use_skip (bool): Whether to use skip connections from the encoder.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        out_channels: int = 1,
        base_channels: int = 32,
        use_skip: bool = True,
    ):
        super().__init__()
        self.use_skip = use_skip

        channels = [base_channels * (2 ** i) for i in range(4)]

        self.fc = nn.Linear(latent_dim, channels[3] * 4 * 4)
        self.reshape_channels = channels[3]

        # Encoder features are reversed: deepest first.
        # up4 receives the deepest encoder feature (channels[3]),
        # up3 receives channels[2], up2 receives channels[1], up1 has no skip.
        skip_ch = [c if use_skip else 0 for c in [channels[3], channels[2], channels[1], 0]]

        self.up4 = UpBlock(channels[3], channels[2], skip_ch[0])
        self.up3 = UpBlock(channels[2], channels[1], skip_ch[1])
        self.up2 = UpBlock(channels[1], channels[0], skip_ch[2])
        self.up1 = UpBlock(channels[0], channels[0], skip_ch[3])

        self.upsample_stem = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0] // 2, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, encoder_features=None) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            z: Latent vector of shape (B, latent_dim).
            encoder_features: List of encoder feature maps for skip connections.

        Returns:
            Reconstructed image tensor.
        """
        x = self.fc(z)
        x = x.view(x.size(0), self.reshape_channels, 4, 4)

        skips = encoder_features[::-1] if (self.use_skip and encoder_features) else [None] * 4

        x = self.up4(x, skips[0] if self.use_skip else None)
        x = self.up3(x, skips[1] if self.use_skip else None)
        x = self.up2(x, skips[2] if self.use_skip else None)
        x = self.up1(x)
        x = self.upsample_stem(x)
        return self.out_conv(x)
