import math
import os
import random
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import Compose, RandAdjustContrast, RandAffine, RandFlip, RandGaussianNoise, RandScaleIntensity
from vector_quantize_pytorch import ResidualVQ

try:
    from transformers import CLIPImageProcessor, CLIPVisionModel
except Exception:  # pragma: no cover - handled with a clear error message at runtime
    CLIPImageProcessor = None
    CLIPVisionModel = None

try:
    import open_clip
except Exception:  # pragma: no cover - handled with a clear error message at runtime
    open_clip = None

# =============================================================================
# =============================================================================
# CORE — AUROC pipeline
# -----------------------------------------------------------------------------
# Everything in this section is on the path that produces the per-slice
# `Final_Binary_sum_of_anomaly_maps` field, which is the ONLY quantity consumed by the
# patient-level ROC / AUROC computation in the Plot_Bars script.
#
# Trace: model forward → ensemble_heal → LPIPS heatmap → binary mask fusion
#        (masked_score ∪ token_surprisal ∪ lpips_backflow ∪ edge erosion)
#        → Final_Binary_sum_of_anomaly_maps → patient aggregation → ROC / AUROC.
# =============================================================================
# =============================================================================


def default_init(module: nn.Module) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module



class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 256, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> tokens: [B, HW/ps^2, D]
        tokens = self.proj(x)
        tokens = tokens.flatten(2).transpose(1, 2)
        return tokens



class ViTEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, depth: int = 8, num_heads: int = 8, seq_len: int = 256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.pos_embed
        return self.encoder(tokens)



class MultiScaleEncoder(nn.Module):
    """
    Lightweight feature pyramid for RVQ-VAE encoder outputs.
    Produces stride-1/2/4 token grids and fuses them with cross-scale attention.
    """

    def __init__(self, embed_dim: int = 256, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.scale_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2 ** i, padding=1)
            for i in range(num_scales)
        ])
        self.cross_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        bsz = tokens.size(0)
        feat_2d = tokens.transpose(1, 2).view(bsz, -1, h, w)

        scale_features: List[torch.Tensor] = []
        for proj in self.scale_projs:
            scale_feat = proj(feat_2d)
            scale_features.append(scale_feat.flatten(2).transpose(1, 2))

        query = scale_features[0]
        key_value = torch.cat(scale_features, dim=1)
        fused, _ = self.cross_attn(query, key_value, key_value)

        return fused, scale_features



class PixelShuffleDecoder(nn.Module):
    def __init__(self, embed_dim: int = 512, base_channels: int = 256, num_upsample: int = 3):
        super().__init__()
        self.num_upsample = num_upsample
        
        # Start wide to process texture
        hidden_dim = base_channels * 2 

        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True), # Switched to SiLU for smoother texture gradients
        )

        # Deeper, wider residual stack
        self.res_layers = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), # Added 3rd block
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
        )

        # Build upsampling blocks dynamically based on patch_size
        # patch_size=8 -> 3 blocks (32->64->128->256)
        # patch_size=16 -> 4 blocks (16->32->64->128->256)
        self.up_blocks = nn.ModuleList()
        in_ch = hidden_dim
        for i in range(num_upsample):
            out_ch = base_channels // (2 ** i)
            self.up_blocks.append(self._up_block(in_ch, out_ch))
            in_ch = out_ch

        self.head = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

    @staticmethod
    def _up_block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(inplace=True), # Switched to SiLU
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = x + self.res_layers(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        return self.head(x)

class Stage1RVQVAE(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 256,
        patch_size: int = 8,
        embed_dim: int = 512,
        encoder_depth: int = 8,
        encoder_heads: int = 8,
        codebook_size: int = 512,
        num_quantizers: int = 2,
        commitment_cost: float = 0.25,
        lr: float = 2e-4,
        perceptual_weight: float = 0.5,
        biomedclip_model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        biomedclip_open_clip_model: Optional[str] = None,
        biomedclip_open_clip_pretrained: Optional[str] = None,
        biomedclip_feature_layer: str | int = "pooled",
        biomedclip_normalize_mode: str = "minmax",
        use_augmentations: bool = True,
        sanity_check_aug: bool = True,
        gaussian_noise_prob: float = 0.50,
        gaussian_noise_std: float = 0.30,
    ):
        super().__init__()
        self.save_hyperparameters()
        torch.set_float32_matmul_precision("medium")

        seq_len = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.encoder = ViTEncoder(embed_dim, encoder_depth, encoder_heads, seq_len)

        self.quantizer = ResidualVQ(
            dim=embed_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            commitment_weight=commitment_cost,          # CHANGED: Lowered from 0.25
            orthogonal_reg_weight=0.1,       # ADDED: Forces diversity in codebook
            orthogonal_reg_max_codes=128,    # ADDED: Limit calculation for speed
            kmeans_init=True,
            threshold_ema_dead_code=0.1,     # CHANGED: Lower threshold to catch more dead codes
            decay=0.85,
        )

        # Calculate number of upsample blocks needed: patch_size = 2^num_upsample
        num_upsample = int(math.log2(patch_size))
        self.decoder = PixelShuffleDecoder(embed_dim, base_channels=embed_dim, num_upsample=num_upsample)
        self.decode_adapter = None  # lazily-created 1x1 conv when decoder channels differ
        # --- AYNU begin: training-time augmentations + perceptual loss ---
        # AYNU: multiscale fusion encoder (used only by encode_multiscale; not on the CORE encode_tokens path)
        self.num_scales = 3
        self.multi_scale_encoder = MultiScaleEncoder(embed_dim, num_scales=self.num_scales)
        self.l1_loss = nn.L1Loss()
        self.lr = lr
        self.seq_hw = int(math.sqrt(seq_len))
        self.use_augmentations = use_augmentations
        self.sanity_check_aug = sanity_check_aug
        self._aug_viz_done = False

        # cache for validation visualization
        self._val_vis_cache = None

        # MONAI image augmentations with a kill switch
        translate_pix = 15
        rotation_rad = math.radians(15)  # ~5 degrees
        zoom_min, zoom_max = 0.8, 1.2
        self.aug_intensity = RandScaleIntensity(factors=0.10, prob=0.33)  # brightness-like scaling
        self.aug_contrast = RandAdjustContrast(gamma=(0.5, 1.5), prob=0.33)  # gamma-based contrast shift
        self.aug_noise = RandGaussianNoise(prob=gaussian_noise_prob, mean=0.0, std=gaussian_noise_std)
        self.aug_affine = RandAffine(
            prob=0.33,
            rotate_range=(-rotation_rad, rotation_rad),
            translate_range=(translate_pix, translate_pix),      # horizontal + vertical translation
            scale_range=(zoom_min - 1.0, zoom_max - 1.0),         # zoom 0.9x to 1.1x
            padding_mode="border",
        )
        self.aug_flip = RandFlip(prob=0.5, spatial_axis=2)  # horizontal (left-right) flip
        self.train_aug = Compose([
            self.aug_intensity,
            self.aug_contrast,
            self.aug_noise,
            self.aug_affine,
            self.aug_flip,
        ]) if use_augmentations else None

        # Initialize only our trainable modules (NOT perceptual loss which has pretrained weights)
        self.patch_embed.apply(default_init)
        self.encoder.apply(default_init)
        self.multi_scale_encoder.apply(default_init)
        self.decoder.apply(default_init)

        # Perceptual term uses BiomedCLIP vision tower
        self.perceptual_loss = BiomedCLIPLoss(
            weight=perceptual_weight,
            model_name=biomedclip_model_name,
            open_clip_model=biomedclip_open_clip_model,
            open_clip_pretrained=biomedclip_open_clip_pretrained,
            feature_layer=biomedclip_feature_layer,
            normalize_mode=biomedclip_normalize_mode,
        )
        # --- AYNU end ---

    def encode_tokens(self, images: torch.Tensor):
        tokens = self.patch_embed(images)
        tokens = self.encoder(tokens)
        quantized, indices, commit_loss = self.quantizer(tokens)
        if isinstance(commit_loss, (tuple, list)):
            commit_loss = sum(commit_loss)
        return tokens, quantized, indices, commit_loss

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        b, n, d = quantized.shape
        h = w = self.seq_hw
        feat = quantized.transpose(1, 2).reshape(b, d, h, w)

        expected_c = self.decoder.stem[0].in_channels
        if feat.shape[1] != expected_c:
            if (self.decode_adapter is None) or (self.decode_adapter.in_channels != feat.shape[1]) or (self.decode_adapter.out_channels != expected_c):
                self.decode_adapter = nn.Conv2d(feat.shape[1], expected_c, kernel_size=1, bias=False).to(feat.device)
            feat = self.decode_adapter(feat)

        recon = self.decoder(feat)
        return torch.clamp(recon, min=-3.0, max=3.0)

    def forward(self, images: torch.Tensor):
        tokens, quantized, indices, commit_loss = self.encode_tokens(images)
        recon = self.decode(quantized)
        q_error = (tokens.detach() - quantized).pow(2).sum(-1).reshape(-1, self.seq_hw, self.seq_hw)
        return {
            "recon": recon,
            "indices": indices,
            "commit_loss": commit_loss,
            "quant_error_map": q_error,
        }

    # =========================================================================
    # AYNU — auxiliary methods of Stage1RVQVAE (training, validation, figures)
    # =========================================================================

    def encode_multiscale(self, images: torch.Tensor):
        tokens = self.patch_embed(images)
        tokens = self.encoder(tokens)
        fused_tokens, scale_features = self.multi_scale_encoder(tokens, self.seq_hw, self.seq_hw)
        quantized, indices, commit_loss = self.quantizer(fused_tokens)
        if isinstance(commit_loss, (tuple, list)):
            commit_loss = sum(commit_loss)
        return fused_tokens, quantized, indices, commit_loss, scale_features

    def training_step(self, batch, batch_idx):
        images = batch["image"] if isinstance(batch, dict) and "image" in batch else (
            batch["MRI_image"] if isinstance(batch, dict) else (batch[0] if isinstance(batch, (list, tuple)) else batch)
        )
        if self.use_augmentations and self.train_aug is not None:
            if self.sanity_check_aug and (not self._aug_viz_done) and batch_idx == 0:
                self._save_aug_preview(images)
            images = self.train_aug(images)
        outputs = self(images)
        recon = outputs["recon"]
        rec_loss = self.l1_loss(recon, images)
        perceptual = self.perceptual_loss(recon, images)
        commit = outputs["commit_loss"]
        if torch.is_tensor(commit) and commit.ndim > 0:
            commit = commit.mean()
        loss = rec_loss + perceptual + commit
        psnr = self._compute_psnr(recon, images)
        self.log_dict({
            "train/l1": rec_loss,
            "train/perceptual": perceptual,
            "train/commit": commit,
            "train/loss": loss,
            "train/psnr": psnr,
        }, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"] if isinstance(batch, dict) and "image" in batch else (
            batch["MRI_image"] if isinstance(batch, dict) else (batch[0] if isinstance(batch, (list, tuple)) else batch)
        )
        paths = batch.get("path", None) if isinstance(batch, dict) else None
        
        outputs = self(images)
        recon = outputs["recon"]
        rec_loss = self.l1_loss(recon, images)
        perceptual = self.perceptual_loss(recon, images)
        commit = outputs["commit_loss"]
        if torch.is_tensor(commit) and commit.ndim > 0:
            commit = commit.mean()
        loss = rec_loss + perceptual + commit
        psnr = self._compute_psnr(recon, images)
        self.log_dict({
            "val/l1": rec_loss,
            "val/perceptual": perceptual,
            "val/commit": commit,
            "val/loss": loss,
            "val/psnr": psnr,
        }, prog_bar=True, on_epoch=True, batch_size=images.size(0))

        # Store for visualization (rank zero only) using first validation batch
        if batch_idx == 0 and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            with torch.no_grad():
                self._val_vis_cache = {
                    "images": images.detach().cpu(),
                    "recon": recon.detach().cpu(),
                    "indices": outputs["indices"].detach().cpu(),
                    "paths": paths,
                }

        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    @torch.no_grad()
    def _save_aug_preview(self, images: torch.Tensor, num_samples: int = 8, num_augments: int = 3):
        """Save a visualization grid of augmented samples for sanity-checking."""
        if not (self.use_augmentations and self.train_aug is not None):
            return
        self._aug_viz_done = True

        save_dir = os.path.join(os.path.dirname(__file__), "FastMRI_RQC_ValExamples")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "augmentations_preview.png")

        base = images[:num_samples].detach().cpu()

        transforms = [
            ("original", None),
            ("intensity", self.aug_intensity),
            ("contrast", self.aug_contrast),
            ("noise", self.aug_noise),
            ("affine", self.aug_affine),
            ("flip", self.aug_flip),
            ("combined", self.train_aug),
        ]

        rows = len(transforms)
        cols = min(num_samples, base.size(0))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, 0)

        def _apply_with_prob_one(transform, tensor):
            if hasattr(transform, "prob"):
                prev = transform.prob
                transform.prob = 1.0
                try:
                    return transform(tensor)
                finally:
                    transform.prob = prev
            return transform(tensor)

        for r, (label, transform) in enumerate(transforms):
            if label == "original":
                aug = base
            elif label == "combined":
                # Force all component probs to 1.0 for combined preview
                prob_backup = []
                for t in self.train_aug.transforms:
                    if hasattr(t, "prob"):
                        prob_backup.append((t, t.prob))
                        t.prob = 1.0
                try:
                    aug = self.train_aug(base.clone()).detach().cpu()
                finally:
                    for t, prob in prob_backup:
                        t.prob = prob
            else:
                aug = _apply_with_prob_one(transform, base.clone()).detach().cpu()
            for c in range(cols):
                aimg = aug[c].squeeze().numpy()
                aimg = self._scale_for_display(aimg)
                axes[r, c].imshow(aimg, cmap="gray", interpolation="nearest")
                axes[r, c].axis("off")
                axes[r, c].set_title(label)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _compute_psnr(self, recon: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR between reconstruction and target."""
        mse = F.mse_loss(recon, target).item()
        if mse < 1e-10:
            return float('inf')
        # For z-score normalized data, use data range based on min/max
        data_range = target.max().item() - target.min().item()
        if data_range < 1e-10:
            data_range = 1.0
        psnr = 10 * math.log10((data_range ** 2) / mse)
        return psnr

    @staticmethod
    def _scale_for_display(arr: np.ndarray) -> np.ndarray:
        """Percentile-based contrast scaling for visualization."""
        if arr.size == 0:
            return arr
        vmin, vmax = np.percentile(arr, [1, 99])
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0
        arr = np.clip(arr, vmin, vmax)
        return (arr - vmin) / (vmax - vmin + 1e-8)

    def visualize_reconstruction(
        self,
        images: torch.Tensor,
        recon: torch.Tensor,
        indices: torch.Tensor,
        file_paths: list = None,
        save_dir: str = None,
        max_samples: int = 4,
    ):
        """
        Visualize input, reconstruction, and quantized latents.
        
        Args:
            images: Input images (B, C, H, W)
            recon: Reconstructed images (B, C, H, W)
            indices: Quantized indices (B, seq_len, num_quantizers)
            file_paths: Optional list of file paths for filtering by slice index
            save_dir: Directory to save figures
            max_samples: Maximum number of samples to visualize
        """
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), "FastMRI_RQC_ValExamples")
        os.makedirs(save_dir, exist_ok=True)

        # Do not assume slice indices in filenames; sample from available batch
        valid_indices = list(range(images.size(0)))

        # Sample up to max_samples
        if len(valid_indices) > max_samples:
            valid_indices = random.sample(valid_indices, max_samples)

        if not valid_indices:
            return

        epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0

        for sample_idx, idx in enumerate(valid_indices):
            img = images[idx].detach().cpu().squeeze().numpy()
            rec = recon[idx].detach().cpu().squeeze().numpy()
            img_disp = self._scale_for_display(img)
            rec_disp = self._scale_for_display(rec)
            
            # Get indices for this sample - shape is (seq_len, num_quantizers)
            idx_tensor = indices[idx].detach().cpu()
            
            # Reshape to spatial grid
            h = w = self.seq_hw
            if idx_tensor.ndim == 2:  # (seq_len, num_quantizers)
                latent_q1 = idx_tensor[:, 0].reshape(h, w).numpy()
                latent_q2 = idx_tensor[:, 1].reshape(h, w).numpy() if idx_tensor.shape[1] > 1 else latent_q1
            else:
                latent_q1 = idx_tensor.reshape(h, w).numpy()
                latent_q2 = latent_q1

            # Compute PSNR
            psnr = self._compute_psnr(recon[idx:idx+1], images[idx:idx+1])

            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle(f"Epoch {epoch} | Sample {sample_idx} | PSNR: {psnr:.2f} dB", fontsize=14)

            # Input image
            ax = axes[0, 0]
            im = ax.imshow(img_disp, cmap='gray', interpolation='nearest')
            ax.set_title(f"Input\nShape: {img.shape}, Range: [{img.min():.2f}, {img.max():.2f}]")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Reconstructed image
            ax = axes[0, 1]
            im = ax.imshow(rec_disp, cmap='gray', interpolation='nearest')
            ax.set_title(f"Reconstruction\nShape: {rec.shape}, Range: [{rec.min():.2f}, {rec.max():.2f}]")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Latent Q1 (first quantizer)
            ax = axes[1, 0]
            im = ax.imshow(latent_q1, cmap='jet', interpolation='nearest')
            unique_codes_q1 = len(np.unique(latent_q1))
            ax.set_title(f"Latent Q1 (Codebook Indices)\nShape: {latent_q1.shape}, Unique: {unique_codes_q1}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Latent Q2 (second quantizer)
            ax = axes[1, 1]
            im = ax.imshow(latent_q2, cmap='jet', interpolation='nearest')
            unique_codes_q2 = len(np.unique(latent_q2))
            ax.set_title(f"Latent Q2 (Codebook Indices)\nShape: {latent_q2.shape}, Unique: {unique_codes_q2}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{sample_idx:02d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        print(f"Saved {len(valid_indices)} validation visualizations to {save_dir}")

    def on_validation_epoch_end(self):
        """Visualize a few samples at the end of every validation epoch (rank zero)."""
        if self._val_vis_cache is None:
            return
        cache = self._val_vis_cache
        self._val_vis_cache = None

        self.visualize_reconstruction(
            images=cache["images"],
            recon=cache["recon"],
            indices=cache["indices"],
            file_paths=cache.get("paths"),
            save_dir=os.path.join(os.path.dirname(__file__), "FastMRI_RQC_ValExamples"),
            max_samples=4,
        )

# =============================================================================
# =============================================================================
# AYNU — AVAILABLE YET NOT ROC-Relevant(auxiliary code, NOT on the AUROC path)
# -----------------------------------------------------------------------------
# Code below is preserved for reproducibility, training, alternate scoring,
# bounding-box evaluation tables, sanity-check figures, and per-patient bar
# charts. None of it feeds the AUROC. Skim freely; do not let it distract
# from the CORE pipeline above.
# =============================================================================
# =============================================================================


class PerceptualLossStub(nn.Module):
    """Lightweight perceptual placeholder; BiomedCLIP dependency removed."""

    def __init__(self, weight: float = 0.0):
        super().__init__()
        self.weight = weight

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0:
            return recon.new_zeros(())
        # Fallback to simple L1 to preserve a small perceptual-like regularizer without external backbones
        return self.weight * F.l1_loss(recon, target)



class BiomedCLIPLoss(nn.Module):
    """BiomedCLIP-based perceptual loss using the vision tower."""

    def __init__(
        self,
        weight: float = 1.0,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        open_clip_model: Optional[str] = None,
        open_clip_pretrained: Optional[str] = None,
        feature_layer: str | int = "pooled",
        normalize_mode: str = "minmax",
    ):
        super().__init__()
        self.weight = weight
        self.model_name = model_name
        self.open_clip_model = open_clip_model
        self.open_clip_pretrained = open_clip_pretrained
        self.feature_layer = feature_layer
        self.normalize_mode = normalize_mode
        self._device: Optional[torch.device] = None
        self._backend = "transformers"

        if CLIPVisionModel is None or CLIPImageProcessor is None:
            self._backend = "open_clip"
        else:
            try:
                self.processor = CLIPImageProcessor.from_pretrained(model_name)
            except Exception:
                self.processor = None
            try:
                self.model = CLIPVisionModel.from_pretrained(model_name)
            except Exception:
                self._backend = "open_clip"
            else:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad_(False)

                self._use_hidden_states = feature_layer != "pooled"

        if self._backend == "open_clip":
            if open_clip is None:
                raise ImportError("open_clip is required to use BiomedCLIPLoss when transformers weights are unavailable.")

            if self.open_clip_model is None or self.open_clip_pretrained is None:
                self.model = self._load_open_clip_from_hf(self.model_name)
            else:
                self.model, _, _ = open_clip.create_model_and_transforms(
                    self.open_clip_model,
                    pretrained=self.open_clip_pretrained,
                )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

            self._use_hidden_states = False

    @staticmethod
    def _pick_open_clip_checkpoint() -> Tuple[str, str]:
        candidates = []
        for model, pretrained in open_clip.list_pretrained():
            token = f"{model}:{pretrained}".lower()
            if "biomed" in token or "biomedclip" in token:
                candidates.append((model, pretrained))
        if not candidates:
            raise RuntimeError("No BiomedCLIP checkpoints found in open_clip.list_pretrained().")
        return candidates[0]

    @staticmethod
    def _load_open_clip_from_hf(model_name: str) -> nn.Module:
        try:
            if not model_name.startswith("hf-hub:"):
                model_name = f"hf-hub:{model_name}"
            model, _ = open_clip.create_model_from_pretrained(model_name)
            return model
        except Exception as exc:
            raise RuntimeError(
                "Failed to load BiomedCLIP with open_clip. "
                "Provide --biomedclip-open-clip-model and --biomedclip-open-clip-pretrained, "
                "or ensure the HF checkpoint is available locally."
            ) from exc

    def _ensure_device(self, device: torch.device) -> None:
        if self._device != device:
            self.model.to(device)
            self._device = device

    def _target_size(self) -> Tuple[int, int]:
        if self._backend == "open_clip":
            image_size = getattr(getattr(self.model, "visual", None), "image_size", 224)
            if isinstance(image_size, (tuple, list)):
                return int(image_size[0]), int(image_size[1])
            return int(image_size), int(image_size)

        if self.processor is None:
            return 224, 224
        size = getattr(self.processor, "size", 224)
        if isinstance(size, dict):
            if "shortest_edge" in size:
                return int(size["shortest_edge"]), int(size["shortest_edge"])
            return int(size.get("height", 224)), int(size.get("width", 224))
        return int(size), int(size)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_mode == "none":
            return x
        if self.normalize_mode == "minmax":
            x_min = x.amin(dim=(-2, -1), keepdim=True)
            x_max = x.amax(dim=(-2, -1), keepdim=True)
            return (x - x_min) / (x_max - x_min + 1e-6)
        raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) > 3:
            x = x[:, :3]

        x = self._normalize(x)

        target_h, target_w = self._target_size()
        if x.shape[-2:] != (target_h, target_w):
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

        if self._backend == "open_clip":
            mean = getattr(getattr(self.model, "visual", None), "image_mean", [0.48145466, 0.4578275, 0.40821073])
            std = getattr(getattr(self.model, "visual", None), "image_std", [0.26862954, 0.26130258, 0.27577711])
            mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
            std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
        elif self.processor is None:
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        else:
            mean = torch.tensor(self.processor.image_mean, device=x.device).view(1, 3, 1, 1)
            std = torch.tensor(self.processor.image_std, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self._backend == "open_clip":
            if self.feature_layer != "pooled":
                raise ValueError("feature_layer is not supported with open_clip backend. Use 'pooled'.")
            return self.model.encode_image(x)

        outputs = self.model(pixel_values=x, output_hidden_states=self._use_hidden_states)
        if self.feature_layer == "pooled":
            feats = outputs.pooler_output
        else:
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("BiomedCLIP hidden states were not returned.")
            feats = hidden_states[self.feature_layer]
            feats = feats.mean(dim=1)
        return feats

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0:
            return recon.new_zeros(())

        self._ensure_device(recon.device)
        recon_proc = self._preprocess(recon)
        target_proc = self._preprocess(target)
        combined = torch.cat([recon_proc, target_proc], dim=0)

        feats = self._encode(combined)
        recon_feats, target_feats = feats.chunk(2, dim=0)
        recon_feats = F.normalize(recon_feats, dim=-1)
        target_feats = F.normalize(target_feats, dim=-1)

        loss = 1.0 - (recon_feats * target_feats).sum(dim=-1)
        return self.weight * loss.mean()

