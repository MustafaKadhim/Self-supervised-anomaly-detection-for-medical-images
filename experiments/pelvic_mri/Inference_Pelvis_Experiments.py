"""
Inference script for pelvis experiments
===================================================================

Key Innovation:
- Calibration Phase: Run on healthy volunteers to compute per-pixel μ and σ
- Inference Phase: Convert LPIPS to Z-scores using population statistics
- Result: Regions with systematic high error (bowel, bone) → Z ≈ 0
         Real anomalies (tumors) → Z >> threshold

Z-Score Formula:
    Z = (LPIPS_test - μ_healthy) / (σ_healthy + ε)

Two Modes:
1. Calibration Mode (--calibration-mode): Process healthy volunteers, save μ/σ maps
2. Inference Mode (--calibration-map): Load calibration, apply Z-score normalization

"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import patches
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. Run: pip install lpips")


HEATMAP_CMAP = "gist_heat"

# Core execution path for anomaly detection + ROC:
#   load stage1/stage2 -> compute Binary token surprisal maps -> aggregate scores.
# Extra visualization/calibration helpers are retained and grouped as
# "AVAILABLE YET NOT USED" further below.


# =============================================================================
# AVAILABLE YET NOT USED
# =============================================================================
# Annotation overlays are optional and do not affect Binary token surprisal
# score calculations used by ROC_Curves_Calculations.py.

# =============================================================================
# Annotation Box Helpers
# =============================================================================

ANNOTATION_BASE_SIZE = 320


def _get_first(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            value = str(row[key]).strip()
            if value:
                return value
    return ""


def _to_int(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def load_annotation_boxes(annotation_csv: Path) -> dict[str, dict[int, list[dict]]]:
    if not annotation_csv.exists():
        print(f"Warning: annotation CSV not found: {annotation_csv}")
        return {}

    with annotation_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        boxes_by_file: dict[str, dict[int, list[dict]]] = {}
        for row in reader:
            file_val = _get_first(row, ("file", "File", "filename", "Filename", "path", "Path"))
            slice_val = _get_first(row, ("slice", "Slice"))
            if not file_val or slice_val == "":
                continue

            file_stem = Path(file_val).stem
            slice_idx = _to_int(slice_val)
            if slice_idx is None:
                continue

            x = _to_int(_get_first(row, ("x", "X")))
            y = _to_int(_get_first(row, ("y", "Y")))
            w = _to_int(_get_first(row, ("width", "Width", "w", "W")))
            h = _to_int(_get_first(row, ("height", "Height", "h", "H")))
            label = _get_first(row, ("label", "Label"))
            study_level = _get_first(row, ("study_level", "StudyLevel", "study", "Study"))

            box = {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "label": label,
                "study_level": study_level,
            }

            boxes_by_file.setdefault(file_stem, {}).setdefault(slice_idx, []).append(box)
        return boxes_by_file


def draw_annotation_boxes(
    ax: plt.Axes,
    boxes: list[dict],
    scale_x: float,
    scale_y: float,
    color: str = "cyan",
) -> int:
    drawn = 0
    for box in boxes:
        if str(box.get("study_level", "")).strip().lower() == "yes":
            continue
        x = box.get("x")
        y = box.get("y")
        w = box.get("width")
        h = box.get("height")
        if None in (x, y, w, h):
            continue
        rect = patches.Rectangle(
            (x * scale_x, y * scale_y),
            w * scale_x,
            h * scale_y,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        drawn += 1
    return drawn


def parse_slice_info(filename: str) -> tuple[str, int | None]:
    stem = Path(filename).stem
    match = re.search(r"_slice_(\d+)$", stem)
    if not match:
        return stem, None
    slice_idx = int(match.group(1))
    file_stem = stem[: match.start()]
    return file_stem, slice_idx


# =============================================================================
# Quantizer Codebook Access
# =============================================================================

def get_codebook_embeddings(quantizer, indices_l1: torch.Tensor, indices_l2: torch.Tensor) -> torch.Tensor:
    """Get quantized embeddings from codebook for given indices."""
    B, S = indices_l1.shape
    device = indices_l1.device
    
    codebook_l1 = None
    codebook_l2 = None
    
    if hasattr(quantizer, 'codebooks') and quantizer.codebooks is not None:
        cbs = quantizer.codebooks
        if torch.is_tensor(cbs):
            if cbs.dim() == 3 and cbs.shape[0] >= 2:
                codebook_l1 = cbs[0]
                codebook_l2 = cbs[1]
            elif cbs.dim() == 2:
                codebook_l1 = cbs
                codebook_l2 = cbs
        elif isinstance(cbs, (nn.ModuleList, list)):
            if len(cbs) >= 2:
                if hasattr(cbs[0], 'weight'):
                    codebook_l1 = cbs[0].weight
                    codebook_l2 = cbs[1].weight
                elif torch.is_tensor(cbs[0]):
                    codebook_l1 = cbs[0]
                    codebook_l2 = cbs[1]
                elif hasattr(cbs[0], 'embed'):
                    codebook_l1 = cbs[0].embed.weight if hasattr(cbs[0].embed, 'weight') else cbs[0].embed
                    codebook_l2 = cbs[1].embed.weight if hasattr(cbs[1].embed, 'weight') else cbs[1].embed
    
    if codebook_l1 is None and hasattr(quantizer, 'layers'):
        layers = quantizer.layers
        if len(layers) >= 2:
            layer0, layer1 = layers[0], layers[1]
            if hasattr(layer0, '_codebook'):
                cb0 = layer0._codebook
                cb1 = layer1._codebook
                if hasattr(cb0, 'embed'):
                    codebook_l1 = cb0.embed.weight if hasattr(cb0.embed, 'weight') else cb0.embed
                    codebook_l2 = cb1.embed.weight if hasattr(cb1.embed, 'weight') else cb1.embed
                elif hasattr(cb0, 'weight'):
                    codebook_l1 = cb0.weight
                    codebook_l2 = cb1.weight
            if codebook_l1 is None and hasattr(layer0, 'embed'):
                codebook_l1 = layer0.embed.weight if hasattr(layer0.embed, 'weight') else layer0.embed
                codebook_l2 = layer1.embed.weight if hasattr(layer1.embed, 'weight') else layer1.embed
    
    if codebook_l1 is None and hasattr(quantizer, 'get_codes_from_indices'):
        try:
            stacked_indices = torch.stack([indices_l1, indices_l2], dim=-1)
            quantized = quantizer.get_codes_from_indices(stacked_indices)
            return quantized
        except Exception:
            pass
    
    if codebook_l1 is None:
        raise RuntimeError(f"Could not access quantizer codebook. Type: {type(quantizer)}")
    
    if codebook_l2 is None:
        codebook_l2 = codebook_l1
    
    codebook_l1 = codebook_l1.to(device)
    codebook_l2 = codebook_l2.to(device)
    if codebook_l1.dim() > 2:
        codebook_l1 = codebook_l1.view(-1, codebook_l1.shape[-1])
    if codebook_l2.dim() > 2:
        codebook_l2 = codebook_l2.view(-1, codebook_l2.shape[-1])
    
    embeddings_l1 = F.embedding(indices_l1, codebook_l1)
    embeddings_l2 = F.embedding(indices_l2, codebook_l2)
    
    return embeddings_l1 + embeddings_l2


# =============================================================================
# Perceptual Loss
# =============================================================================

class PerceptualLoss(nn.Module):
    """LPIPS-based perceptual loss with spatial output."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        
        if LPIPS_AVAILABLE:
            self.loss_fn = lpips.LPIPS(net='vgg', spatial=True).to(device)
            self.loss_fn.eval()
            for param in self.loss_fn.parameters():
                param.requires_grad = False
            self.method = "lpips"
        else:
            self.method = "l1"
    
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)
        x = (x - self.mean) / self.std
        return x
    
    @torch.no_grad()
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if self.method == "lpips":
            orig_size = img1.shape[-2:]
            img1 = self._preprocess(img1)
            img2 = self._preprocess(img2)
            diff = self.loss_fn(img1, img2)
            if diff.shape[-2:] != orig_size:
                diff = F.interpolate(diff, size=orig_size, mode='bilinear', align_corners=False)
            return diff
        else:
            return (img1 - img2).abs()


# =============================================================================
# Token Surprisal Scoring (Pseudo-PLL)
# =============================================================================

@torch.no_grad()
def compute_token_surprisal_map(
    stage1,
    stage2,
    images: torch.Tensor,
    slice_pos: Optional[torch.Tensor] = None,
    num_samples: int = 50,
    mask_ratio: float = 0.15,
) -> torch.Tensor:
    """Approximate token surprisal via repeated random masking.

    Surprisal is computed as negative log-likelihood of the true token
    (not entropy/max-prob). Only masked positions receive values.

    Returns a [B, 1, Ht, Wt] map in token-grid resolution.
    """
    device = images.device
    B = images.shape[0]
    _, _, indices, _ = stage1.encode_tokens(images)
    original_l1 = indices[:, :, 0].clone()

    S = stage2.seq_len
    h = w = stage2.seq_hw

    task_emb_l1 = stage2.task_embed(torch.zeros(B, dtype=torch.long, device=device)).unsqueeze(1)
    l2_context = torch.full((B, S), stage2.mask_token_id_l2, device=device, dtype=torch.long)

    accum = torch.zeros(B, S, device=device)
    counts = torch.zeros(B, S, device=device)

    for _ in range(max(int(num_samples), 1)):
        mask = torch.rand(B, S, device=device) < float(mask_ratio)
        if mask.sum() == 0:
            continue
        masked_l1 = original_l1.clone()
        masked_l1[mask] = stage2.mask_token_id_l1

        tokens = stage2.l1_embed(masked_l1) + stage2.l2_embed(l2_context) + task_emb_l1
        hidden = stage2.transformer(tokens, slice_pos=slice_pos)
        logits = stage2.head_l1(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        nll_true = -log_probs.gather(-1, original_l1.unsqueeze(-1)).squeeze(-1)

        accum[mask] += nll_true[mask]
        counts[mask] += 1

    avg_surprisal = accum / counts.clamp(min=1)
    return avg_surprisal.view(B, 1, h, w)


# =============================================================================
# Heatmap Aggregation Utilities
# =============================================================================

def aggregate_heatmaps(
    heatmaps: List[torch.Tensor],
    method: str = "mean",
    logsumexp_temp: float = 1.0,
) -> torch.Tensor:
    """Aggregate a list of heatmaps into a single heatmap.

    Args:
        heatmaps: list of [B, 1, H, W] tensors
        method: "mean" | "max" | "logsumexp" | "geomean"
        logsumexp_temp: temperature for logsumexp (higher = softer)
    """
    if not heatmaps:
        raise ValueError("heatmaps list is empty")
    stack = torch.stack(heatmaps, dim=0)  # [N, B, 1, H, W]
    method = method.lower().strip()
    if method == "mean":
        return stack.mean(dim=0)
    if method == "max":
        return stack.max(dim=0).values
    if method == "logsumexp":
        temp = max(float(logsumexp_temp), 1e-6)
        return torch.logsumexp(stack / temp, dim=0) * temp
    if method == "geomean":
        return torch.exp(torch.log(stack.clamp(min=1e-8)).mean(dim=0))
    raise ValueError(f"Unsupported aggregation method: {method}")


def fuse_lpips_with_token_surprisal_for_display(
    final_lpips: np.ndarray,
    token_surprisal: Optional[np.ndarray],
    lpips_percentile: float = 60.0,
    display_floor: float = 0.5,
) -> tuple[np.ndarray, float, bool]:
    """Create a display map by fusing LPIPS with token surprisal hotspots."""
    lpips_cutoff = np.percentile(final_lpips, lpips_percentile)
    lpips_clamped = np.where(final_lpips >= lpips_cutoff, final_lpips, np.nan)

    if token_surprisal is None:
        display_map = np.where(lpips_clamped > display_floor, lpips_clamped, np.nan)
        return display_map, float(lpips_cutoff), False

    lpips_dense = np.nan_to_num(lpips_clamped, nan=0.0)
    lpips_norm = np.clip((lpips_dense - display_floor) / (1.0 - display_floor), 0.0, 1.0)

    token_binary = token_surprisal > 0.0
    token_soft = ndimage.gaussian_filter(token_binary.astype(np.float32), sigma=1.2)
    if token_soft.max() > 0:
        token_soft = token_soft / (token_soft.max() + 1e-8)

    fused_norm = 0.55 * lpips_norm + 0.45 * np.maximum(lpips_norm, token_soft)
    fused_norm = np.clip(fused_norm, 0.0, 1.0)
    fused_map = np.where(fused_norm > 0, display_floor + (1.0 - display_floor) * fused_norm, np.nan)
    fused_map = np.where(fused_map > display_floor, fused_map, np.nan)

    return fused_map, float(lpips_cutoff), True


def visualize_heatmap_aggregation_comparison(
    heatmaps: List[torch.Tensor],
    save_path: str,
    sample_idx: int = 0,
    title: str = "",
    logsumexp_temp: float = 1.0,
):
    """Save a comparison figure for mean/max/logsumexp aggregation."""
    agg_mean = aggregate_heatmaps(heatmaps, method="mean", logsumexp_temp=logsumexp_temp)
    agg_max = aggregate_heatmaps(heatmaps, method="max", logsumexp_temp=logsumexp_temp)
    agg_lse = aggregate_heatmaps(heatmaps, method="logsumexp", logsumexp_temp=logsumexp_temp)

    def to_np(x):
        t = x[sample_idx] if x.dim() == 4 else x
        return t[0].detach().cpu().numpy()

    mean_np = to_np(agg_mean)
    max_np = to_np(agg_max)
    lse_np = to_np(agg_lse)
    vmax = max(mean_np.max(), max_np.max(), lse_np.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, label in zip(
        axes,
        [mean_np, max_np, lse_np],
        ["Mean", "Max", f"LogSumExp (T={logsumexp_temp:.2f})"],
    ):
        im = ax.imshow(data, cmap=HEATMAP_CMAP, vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Token Masking Utilities
# =============================================================================

def build_token_mask(
    anomaly_mask: torch.Tensor,
    patch_size: int,
    mode: str = "max",
    avg_threshold: float = 0.5,
    topk_ratio: float = 0.1,
) -> torch.Tensor:
    """Build a token mask from pixel-level anomaly mask.

    Returns:
        token_mask: [B, Ht, Wt] bool tensor
    """
    mode = mode.lower().strip()
    scores_avg = F.avg_pool2d(anomaly_mask, kernel_size=patch_size, stride=patch_size)
    scores_max = F.max_pool2d(anomaly_mask, kernel_size=patch_size, stride=patch_size)

    if mode == "max":
        token_mask = scores_max > 0.5
    elif mode == "avg":
        token_mask = scores_avg > avg_threshold
    elif mode == "topk":
        B, _, Ht, Wt = scores_avg.shape
        token_mask = torch.zeros((B, Ht, Wt), device=scores_avg.device, dtype=torch.bool)
        scores_flat = scores_avg.view(B, -1)
        total = scores_flat.shape[1]
        k = int(round(float(topk_ratio) * total))
        if k > 0:
            k = min(k, total)
            for b in range(B):
                _, top_idx = torch.topk(scores_flat[b], k=k, largest=True)
                token_mask.view(B, -1)[b, top_idx] = True
    else:
        raise ValueError(f"Unsupported token mask mode: {mode}")

    if token_mask.dim() == 4:
        token_mask = token_mask[:, 0]
    return token_mask


def visualize_token_masking_comparison(
    images: torch.Tensor,
    anomaly_mask: torch.Tensor,
    patch_size: int,
    save_path: str,
    sample_idx: int = 0,
    avg_threshold: float = 0.5,
    topk_ratio: float = 0.1,
    title: str = "",
):
    """Save a comparison figure for token masking strategies."""
    modes = ["max", "avg", "topk"]
    token_masks = [
        build_token_mask(anomaly_mask, patch_size, mode=m, avg_threshold=avg_threshold, topk_ratio=topk_ratio)
        for m in modes
    ]

    def to_np_img(x):
        t = x[sample_idx]
        return t[0].detach().cpu().numpy()

    def upsample(mask):
        mask_f = mask[sample_idx].float().unsqueeze(0).unsqueeze(0)
        return F.interpolate(mask_f, size=images.shape[-2:], mode='nearest')[0, 0].cpu().numpy()

    input_np = to_np_img(images)
    vmin = np.percentile(input_np, 0.1)
    vmax = np.percentile(input_np, 99.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mode, tmask in zip(axes, modes, token_masks):
        up = upsample(tmask)
        ax.imshow(input_np, cmap='gray', vmin=vmin, vmax=vmax)
        overlay = up.astype(float, copy=True)
        overlay[overlay == 0] = np.nan
        cmap = plt.cm.get_cmap('Reds').copy()
        cmap.set_bad(alpha=0)
        ax.imshow(overlay, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
        coverage = float(tmask[sample_idx].float().mean().item())
        ax.set_title(f"{mode.upper()} mask\ncoverage={coverage:.2%}", fontsize=12, fontweight='bold')
        ax.axis('off')

    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Model Loading
# =============================================================================

def load_models(stage1_ckpt: str, stage2_ckpt: str, device: str = "cuda"):
    """Load models."""
    from model_stage1 import Stage1RVQVAE
    from model_stage2 import FactorizedMaskGIT
    from monai.utils.enums import TraceKeys
    
    print(f"Loading Stage 1 from: {stage1_ckpt}")
    try:
        torch.serialization.add_safe_globals([TraceKeys])
    except Exception:
        pass

    ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("perceptual_loss.")}
    hparams = ckpt.get("hyper_parameters", {})
    stage1 = Stage1RVQVAE(**hparams)
    stage1.load_state_dict(filtered_state, strict=False)
    stage1.eval().to(device)
    
    print(f"Loading Stage 2 from: {stage2_ckpt}")
    stage2 = FactorizedMaskGIT.load_from_checkpoint(stage2_ckpt, stage1=stage1, map_location=device, strict=False)
    stage2.eval().to(device)
    
    for param in stage1.parameters():
        param.requires_grad = False
    for param in stage2.parameters():
        param.requires_grad = False
    
    return stage1, stage2


# =============================================================================
# Sharpness Score & Mask Generation
# =============================================================================

def compute_sharpness_score(images: torch.Tensor) -> torch.Tensor:
    """Laplacian variance (Tenengrad-style) sharpness over the full image."""
    device = images.device
    if images.shape[1] > 1:
        img_gray = images.mean(dim=1, keepdim=True)
    else:
        img_gray = images
    lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device).view(1, 1, 3, 3)
    lap = F.conv2d(img_gray, lap_kernel, padding=1)
    energy = lap.pow(2)
    num = energy.sum(dim=(1, 2, 3))
    denom = torch.tensor(img_gray[0].numel(), device=device, dtype=energy.dtype).repeat(images.shape[0])
    return num / denom


def compute_sharpness_map(images: torch.Tensor) -> torch.Tensor:
    """Per-pixel Laplacian energy as a sharpness heatmap."""
    device = images.device
    if images.shape[1] > 1:
        img_gray = images.mean(dim=1, keepdim=True)
    else:
        img_gray = images
    lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device).view(1, 1, 3, 3)
    lap = F.conv2d(img_gray, lap_kernel, padding=1)
    return lap.pow(2)


# =============================================================================
# Healing Functions
# =============================================================================

def _build_checker_mask(h: int, w: int, device: torch.device, pattern: int = 0) -> torch.Tensor:
    """Return deterministic checkerboard-like masks."""
    row_idx = torch.arange(h, device=device).view(-1, 1)
    col_idx = torch.arange(w, device=device).view(1, -1)
    if pattern in (0, 1):
        return ((row_idx + col_idx + pattern) % 2 == 0)
    if pattern in (2, 3):
        return (((row_idx // 2) + (col_idx // 2) + (pattern - 2)) % 2 == 0)
    if pattern in (4, 5):
        return (((row_idx // 4) + (col_idx // 4) + (pattern - 4)) % 2 == 0)
    raise ValueError(f"Unsupported mask pattern: {pattern}")


@torch.no_grad()
def _heal_with_mask(
    stage1, stage2,
    original_l1: torch.Tensor,
    original_l2: torch.Tensor,
    slice_pos: Optional[torch.Tensor],
    mask: torch.Tensor,
    num_steps: int = 12,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single healing pass with given mask.

    Returns the healed token grids alongside per-token entropy estimates
    (MaskGIT uncertainty) for both codebook levels.
    """
    device = original_l1.device
    B = original_l1.shape[0]
    S = stage2.seq_len
    h = w = stage2.seq_hw
    
    working_l1 = original_l1.clone()
    entropy_l1_last = torch.zeros(B, S, device=device)
    working_l1_grid = working_l1.view(B, h, w)
    working_l1_grid[mask] = stage2.mask_token_id_l1
    working_l1 = working_l1_grid.view(B, S)
    
    working_l2 = original_l2.clone()
    working_l2_grid = working_l2.view(B, h, w)
    working_l2_grid[mask] = stage2.mask_token_id_l2
    working_l2 = working_l2_grid.view(B, S)
    
    task_emb_l1 = stage2.task_embed(torch.zeros(B, dtype=torch.long, device=device)).unsqueeze(1)
    l2_context = torch.full((B, S), stage2.mask_token_id_l2, device=device, dtype=torch.long)
    
    for step in range(num_steps):
        tokens = stage2.l1_embed(working_l1) + stage2.l2_embed(l2_context) + task_emb_l1
        hidden = stage2.transformer(tokens, slice_pos=slice_pos)
        logits = stage2.head_l1(hidden)
        
        if temperature != 1.0:
            logits = logits / temperature
        
        probs = F.softmax(logits, dim=-1)
        max_probs, predictions = probs.max(dim=-1)
        entropy_tokens = -(probs.clamp(min=1e-8) * probs.clamp(min=1e-8).log()).sum(dim=-1)
        entropy_l1_last = entropy_tokens
        
        is_masked = (working_l1 == stage2.mask_token_id_l1)
        num_masked = is_masked.sum(dim=1)
        
        if num_masked.sum() == 0:
            break
        
        num_to_unmask = torch.ceil(num_masked.float() / (num_steps - step)).long().clamp(min=1)
        
        for b in range(B):
            masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
            if len(masked_indices) == 0:
                continue
            conf = max_probs[b, masked_indices]
            n_unmask = min(num_to_unmask[b].item(), len(masked_indices))
            _, top_k = conf.topk(n_unmask)
            unmask_pos = masked_indices[top_k]
            working_l1[b, unmask_pos] = predictions[b, unmask_pos]
    
    healed_l1 = working_l1
    entropy_l1 = entropy_l1_last.detach()
    
    task_emb_l2 = stage2.task_embed(torch.ones(B, dtype=torch.long, device=device)).unsqueeze(1)
    entropy_l2_last = torch.zeros(B, S, device=device)
    
    for step in range(num_steps):
        tokens = stage2.l1_embed(healed_l1) + stage2.l2_embed(working_l2) + task_emb_l2
        hidden = stage2.transformer(tokens, slice_pos=slice_pos)
        logits = stage2.head_l2(hidden)
        
        if temperature != 1.0:
            logits = logits / temperature
        
        probs = F.softmax(logits, dim=-1)
        max_probs, predictions = probs.max(dim=-1)
        entropy_tokens = -(probs.clamp(min=1e-8) * probs.clamp(min=1e-8).log()).sum(dim=-1)
        entropy_l2_last = entropy_tokens
        
        is_masked = (working_l2 == stage2.mask_token_id_l2)
        num_masked = is_masked.sum(dim=1)
        
        if num_masked.sum() == 0:
            break
        
        num_to_unmask = torch.ceil(num_masked.float() / (num_steps - step)).long().clamp(min=1)
        
        for b in range(B):
            masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
            if len(masked_indices) == 0:
                continue
            conf = max_probs[b, masked_indices]
            n_unmask = min(num_to_unmask[b].item(), len(masked_indices))
            _, top_k = conf.topk(n_unmask)
            unmask_pos = masked_indices[top_k]
            working_l2[b, unmask_pos] = predictions[b, unmask_pos]
    
    healed_l2 = working_l2
    entropy_l2 = entropy_l2_last.detach()

    return healed_l1, healed_l2, entropy_l1, entropy_l2


@torch.no_grad()
def ensemble_heal(
    stage1, stage2,
    images: torch.Tensor,
    slice_pos: Optional[torch.Tensor] = None,
    num_steps: int = 12,
    temperature: float = 0.8,
    mask_patterns: List[int] = [0, 1],
    debug: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Single iteration with ensemble of mask patterns."""
    device = images.device
    B = images.shape[0]
    h = w = stage2.seq_hw
    
    _, _, indices, _ = stage1.encode_tokens(images)
    original_l1 = indices[:, :, 0].clone()
    original_l2 = indices[:, :, 1].clone()
    
    healed_images_list = []
    token_changes_l1 = []
    token_changes_l2 = []
    
    for pattern in mask_patterns:
        mask = _build_checker_mask(h, w, device, pattern=pattern)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        
        healed_l1, healed_l2, entropy_l1, entropy_l2 = _heal_with_mask(
            stage1, stage2,
            original_l1, original_l2,
            slice_pos, mask,
            num_steps=num_steps,
            temperature=temperature,
        )
        
        healed_quant = get_codebook_embeddings(stage1.quantizer, healed_l1, healed_l2)
        healed_img = stage1.decode(healed_quant)
        healed_images_list.append(healed_img)

        l1_change = (healed_l1 != original_l1).float().mean().item()
        l2_change = (healed_l2 != original_l2).float().mean().item()
        token_changes_l1.append(l1_change)
        token_changes_l2.append(l2_change)
    
    healed_ensemble = torch.stack(healed_images_list, dim=0).mean(dim=0)
    
    info = {
        "original_l1": original_l1,
        "original_l2": original_l2,
        "healed_images_list": healed_images_list,
        "heal_patterns": mask_patterns,
        "mean_l1_change": np.mean(token_changes_l1),
        "mean_l2_change": np.mean(token_changes_l2),
    }
    
    return healed_ensemble, info


# =============================================================================
# Targeted Inpainting
# =============================================================================

@torch.no_grad()
def targeted_inpaint(
    stage1, stage2,
    images: torch.Tensor,
    anomaly_mask: torch.Tensor,
    slice_pos: Optional[torch.Tensor] = None,
    num_steps: int = 12,
    temperature: float = 0.9,
    token_mask_mode: str = "max",
    token_mask_avg_threshold: float = 0.5,
    token_mask_topk_ratio: float = 0.1,
    debug: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Targeted inpainting - only regenerate masked tokens."""
    device = images.device
    B = images.shape[0]
    S = stage2.seq_len
    h = w = stage2.seq_hw
    patch_size = stage2.patch_size
    
    _, _, indices, _ = stage1.encode_tokens(images)
    original_l1 = indices[:, :, 0].clone()
    original_l2 = indices[:, :, 1].clone()
    
    token_mask = build_token_mask(
        anomaly_mask,
        patch_size=patch_size,
        mode=token_mask_mode,
        avg_threshold=token_mask_avg_threshold,
        topk_ratio=token_mask_topk_ratio,
    ).view(B, h, w)
    
    if not token_mask.any():
        if debug:
            print("  No tokens to inpaint - returning reconstruction")
        outputs = stage1(images)
        return outputs["recon"], {"no_inpaint": True, "l1_change": 0, "l2_change": 0, "locked_preserved": True}
    
    working_l1 = original_l1.clone()
    working_l2 = original_l2.clone()
    
    working_l1_grid = working_l1.view(B, h, w)
    working_l2_grid = working_l2.view(B, h, w)
    working_l1_grid[token_mask] = stage2.mask_token_id_l1
    working_l2_grid[token_mask] = stage2.mask_token_id_l2
    working_l1 = working_l1_grid.view(B, S)
    working_l2 = working_l2_grid.view(B, S)
    
    task_emb_l1 = stage2.task_embed(torch.zeros(B, dtype=torch.long, device=device)).unsqueeze(1)
    l2_context_full_mask = torch.full((B, S), stage2.mask_token_id_l2, device=device, dtype=torch.long)
    
    for step in range(num_steps):
        tokens = stage2.l1_embed(working_l1) + stage2.l2_embed(l2_context_full_mask) + task_emb_l1
        hidden = stage2.transformer(tokens, slice_pos=slice_pos)
        logits = stage2.head_l1(hidden)
        
        if temperature != 1.0:
            logits = logits / temperature
        
        probs = F.softmax(logits, dim=-1)
        max_probs, predictions = probs.max(dim=-1)
        
        is_masked = (working_l1 == stage2.mask_token_id_l1)
        num_masked = is_masked.sum(dim=1)
        
        if num_masked.sum() == 0:
            break
        
        num_to_unmask = torch.ceil(num_masked.float() / (num_steps - step)).long().clamp(min=1)
        
        for b in range(B):
            masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
            if len(masked_indices) == 0:
                continue
            conf = max_probs[b, masked_indices]
            n_unmask = min(num_to_unmask[b].item(), len(masked_indices))
            _, top_k = conf.topk(n_unmask)
            unmask_pos = masked_indices[top_k]
            working_l1[b, unmask_pos] = predictions[b, unmask_pos]
    
    inpainted_l1 = working_l1
    
    task_emb_l2 = stage2.task_embed(torch.ones(B, dtype=torch.long, device=device)).unsqueeze(1)
    working_l2 = original_l2.clone()
    working_l2_grid = working_l2.view(B, h, w)
    working_l2_grid[token_mask] = stage2.mask_token_id_l2
    working_l2 = working_l2_grid.view(B, S)
    
    for step in range(num_steps):
        tokens = stage2.l1_embed(inpainted_l1) + stage2.l2_embed(working_l2) + task_emb_l2
        hidden = stage2.transformer(tokens, slice_pos=slice_pos)
        logits = stage2.head_l2(hidden)
        
        if temperature != 1.0:
            logits = logits / temperature
        
        probs = F.softmax(logits, dim=-1)
        max_probs, predictions = probs.max(dim=-1)
        
        is_masked = (working_l2 == stage2.mask_token_id_l2)
        num_masked = is_masked.sum(dim=1)
        
        if num_masked.sum() == 0:
            break
        
        num_to_unmask = torch.ceil(num_masked.float() / (num_steps - step)).long().clamp(min=1)
        
        for b in range(B):
            masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
            if len(masked_indices) == 0:
                continue
            conf = max_probs[b, masked_indices]
            n_unmask = min(num_to_unmask[b].item(), len(masked_indices))
            _, top_k = conf.topk(n_unmask)
            unmask_pos = masked_indices[top_k]
            working_l2[b, unmask_pos] = predictions[b, unmask_pos]
    
    inpainted_l2 = working_l2
    
    locked_mask_flat = ~token_mask.view(B, S)
    l1_locked_preserved = (inpainted_l1[locked_mask_flat] == original_l1[locked_mask_flat]).all().item()
    l2_locked_preserved = (inpainted_l2[locked_mask_flat] == original_l2[locked_mask_flat]).all().item()
    
    inpainted_quant = get_codebook_embeddings(stage1.quantizer, inpainted_l1, inpainted_l2)
    inpainted_images = stage1.decode(inpainted_quant)
    
    l1_change = (inpainted_l1 != original_l1).float().mean().item()
    l2_change = (inpainted_l2 != original_l2).float().mean().item()
    
    info = {
        "token_mask": token_mask,
        "l1_change": l1_change,
        "l2_change": l2_change,
        "locked_preserved": l1_locked_preserved and l2_locked_preserved,
    }
    
    return inpainted_images, info


# =============================================================================
# Z-SCORE CALIBRATION SYSTEM
# =============================================================================

class ZScoreCalibration:
    """
    Population-Based Z-Score Calibration for Anomaly Detection.
    
    Concept:
    - Healthy tissue has CONSISTENT reconstruction error (high μ, low σ in certain regions)
    - Anomalies have UNUSUAL reconstruction error (deviation from expected)
    
    Z-Score Formula:
        Z = (LPIPS_test - μ_healthy) / (σ_healthy + ε)
    
    Result:
    - Regions with systematic high error (bowel, bone): Z ≈ 0 (expected error)
    - Real anomalies: Z >> threshold (unexpected error)
    """
    
    def __init__(self, calibration_path: Optional[str] = None):
        self.mu = None  # Per-pixel mean [H, W]
        self.sigma = None  # Per-pixel std [H, W]
        self.n_samples = 0
        self.smoothing_kernel = 15  # Default smoothing kernel
        self.slice_index_stats = {}  # Optional: per-slice-index statistics
        
        if calibration_path is not None:
            self.load(calibration_path)
    
    def load(self, path: str):
        """Load calibration from .npz file."""
        print(f"Loading Z-score calibration from: {path}")
        data = np.load(path)
        self.mu = data['mu']
        self.sigma = data['sigma']
        self.n_samples = int(data['n_samples'])
        self.smoothing_kernel = int(data['smoothing_kernel']) if 'smoothing_kernel' in data else 15
        
        # Load per-slice-index stats if available
        if 'slice_indices' in data:
            slice_indices = data['slice_indices']
            for idx in slice_indices:
                self.slice_index_stats[int(idx)] = {
                    'mu': data[f'mu_slice_{idx}'],
                    'sigma': data[f'sigma_slice_{idx}'],
                }
        
        print(f"  Loaded calibration from {self.n_samples} healthy samples")
        print(f"  Smoothing kernel: {self.smoothing_kernel}")
        print(f"  μ range: [{self.mu.min():.4f}, {self.mu.max():.4f}]")
        print(f"  σ range: [{self.sigma.min():.4f}, {self.sigma.max():.4f}]")
        if self.slice_index_stats:
            print(f"  Per-slice-index stats available for {len(self.slice_index_stats)} slice positions")
    
    def save(self, path: str, slice_index_data: Optional[Dict] = None, smoothing_kernel: int = 15):
        """Save calibration to .npz file."""
        save_dict = {
            'mu': self.mu,
            'sigma': self.sigma,
            'n_samples': self.n_samples,
            'smoothing_kernel': smoothing_kernel,  # Store smoothing used during calibration
        }
        
        # Save per-slice-index stats if available
        if slice_index_data:
            save_dict['slice_indices'] = np.array(list(slice_index_data.keys()))
            for idx, stats in slice_index_data.items():
                save_dict[f'mu_slice_{idx}'] = stats['mu']
                save_dict[f'sigma_slice_{idx}'] = stats['sigma']
        
        np.savez_compressed(path, **save_dict)
        print(f"Saved Z-score calibration to: {path}")
    
    def compute_zscore(
        self, 
        heatmap: torch.Tensor, 
        epsilon: float = 0.01,
        slice_indices: Optional[List[int]] = None,
        smoothing_kernel: int = 15,
    ) -> torch.Tensor:
        """
        Convert raw LPIPS heatmap to Z-scores.
        
        Args:
            heatmap: [B, 1, H, W] raw LPIPS values
            epsilon: Small constant to prevent division by zero
            slice_indices: Optional list of slice indices for per-slice calibration (one per batch item)
            smoothing_kernel: Kernel size for spatial smoothing (handles registration noise)
            
        Returns:
            zscore: [B, 1, H, W] Z-score values
        """
        if self.mu is None or self.sigma is None:
            raise ValueError("Calibration not loaded. Call load() first.")
        
        device = heatmap.device
        B = heatmap.shape[0]
        
        # 1. Prepare Batch of Mean/Std Maps (per-slice or global)
        batch_mu = []
        batch_sigma = []
        
        if slice_indices is not None and self.slice_index_stats:
            # Per-slice lookup for each item in batch
            for idx in slice_indices:
                if idx in self.slice_index_stats:
                    batch_mu.append(torch.from_numpy(self.slice_index_stats[idx]['mu']).float())
                    batch_sigma.append(torch.from_numpy(self.slice_index_stats[idx]['sigma']).float())
                else:
                    # Fallback to global if this slice index is missing
                    batch_mu.append(torch.from_numpy(self.mu).float())
                    batch_sigma.append(torch.from_numpy(self.sigma).float())
            
            # Stack: [B, H, W] -> [B, 1, H, W]
            mu_tensor = torch.stack(batch_mu).unsqueeze(1).to(device)
            sigma_tensor = torch.stack(batch_sigma).unsqueeze(1).to(device)
        else:
            # Global fallback - broadcast to batch size
            mu_tensor = torch.from_numpy(self.mu).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
            sigma_tensor = torch.from_numpy(self.sigma).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
        
        # 2. Apply spatial smoothing to input heatmap (MUST match calibration smoothing!)
        # This handles slight registration differences between patients
        if smoothing_kernel > 1:
            padding = smoothing_kernel // 2
            heatmap_smooth = F.avg_pool2d(heatmap, kernel_size=smoothing_kernel, stride=1, padding=padding)
        else:
            heatmap_smooth = heatmap
        
        # 3. Handle size mismatch (resize calibration maps if needed)
        if mu_tensor.shape[-2:] != heatmap_smooth.shape[-2:]:
            mu_tensor = F.interpolate(mu_tensor, size=heatmap_smooth.shape[-2:], mode='bilinear', align_corners=False)
            sigma_tensor = F.interpolate(sigma_tensor, size=heatmap_smooth.shape[-2:], mode='bilinear', align_corners=False)
        
        # 4. Compute Z-score
        zscore = (heatmap_smooth - mu_tensor) / (sigma_tensor + epsilon)
        
        return zscore
    
    def is_loaded(self) -> bool:
        """Check if calibration data is loaded."""
        return self.mu is not None and self.sigma is not None


@torch.no_grad()
def run_calibration(
    stage1, stage2,
    perceptual_loss: PerceptualLoss,
    dataloader,
    output_path: str,
    device: str = "cuda",
    heal_steps: int = 12,
    heal_temperature: float = 0.8,
    heal_patterns: List[int] = [0, 1],
    use_tta: bool = True,
    use_per_slice_stats: bool = True,
    smoothing_kernel: int = 15,
    heatmap_aggregation: str = "mean",
    logsumexp_temp: float = 1.0,
    save_aggregation_figures: bool = True,
    aggregation_figures_max_samples: int = 3,
    debug: bool = True,
    flip_upside_down: bool = False,
) -> ZScoreCalibration:
    """
    Run calibration on healthy volunteers.
    
    This processes all healthy slices and computes per-pixel statistics:
    - μ[h,w] = mean LPIPS error at each pixel across all healthy samples
    - σ[h,w] = std LPIPS error at each pixel across all healthy samples
    
    Args:
        stage1, stage2: Models
        perceptual_loss: LPIPS loss
        dataloader: DataLoader for healthy volunteers
        output_path: Path to save calibration (.npz)
        device: Device
        heal_steps, heal_temperature, heal_patterns: Healing parameters
        use_tta: Use TTA for healing
        use_per_slice_stats: Compute per-slice-index statistics
        debug: Print debug info
        
    Returns:
        ZScoreCalibration object with computed statistics
    """
    print("\n" + "="*70)
    print("Z-SCORE CALIBRATION MODE")
    print("Processing healthy volunteers to compute population statistics...")
    print("="*70)
    
    all_heatmaps = []
    slice_index_heatmaps = defaultdict(list)
    
    total_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Calibrating")):
        images = batch["image"].to(device)
        if flip_upside_down:
            images = torch.flip(images, dims=[2])
        paths = batch["path"]
        B = images.shape[0]
        
        if hasattr(stage2, "_extract_slice_indices"):
            slice_pos = stage2._extract_slice_indices(paths, device)
        else:
            slice_pos = None
        
        # ---------------------------------------------------------
        # Healing (same as inference)
        # ---------------------------------------------------------
        healed_A, heal_info = ensemble_heal(
            stage1, stage2, images, slice_pos,
            num_steps=heal_steps, temperature=heal_temperature,
            mask_patterns=heal_patterns, debug=False
        )
        
        if use_tta:
            images_flipped = torch.flip(images, dims=[-1])
            healed_flipped, heal_info_flip = ensemble_heal(
                stage1, stage2, images_flipped, slice_pos,
                num_steps=heal_steps, temperature=heal_temperature,
                mask_patterns=heal_patterns, debug=False
            )
            healed_B = torch.flip(healed_flipped, dims=[-1])
            healed_images_list_tta = [torch.flip(h, dims=[-1]) for h in heal_info_flip.get("healed_images_list", [])]
        else:
            healed_B = healed_A
            healed_images_list_tta = []
        
        # ---------------------------------------------------------
        # Compute LPIPS(Input, Healed)
        # ---------------------------------------------------------
        if heatmap_aggregation.lower().strip() == "geomean":
            # Smart-equivalent fusion:
            #   heatmap = sqrt(LPIPS(Input, healed_A) * LPIPS(Input, healed_B))
            # where healed_A and healed_B are ensemble outputs for native/TTA branches.
            lpips_map_A = perceptual_loss(images, healed_A)
            if use_tta:
                lpips_map_flip = perceptual_loss(images_flipped, healed_flipped)
                lpips_map_B = torch.flip(lpips_map_flip, dims=[-1])
            else:
                lpips_map_B = lpips_map_A
            heatmap = torch.sqrt(lpips_map_A * lpips_map_B + 1e-8)
            heatmaps = [lpips_map_A, lpips_map_B]
        else:
            heatmaps = []
            for h in heal_info.get("healed_images_list", []):
                heatmaps.append(perceptual_loss(images, h))
            if use_tta and healed_images_list_tta:
                for h in healed_images_list_tta:
                    heatmaps.append(perceptual_loss(images, h))

            if not heatmaps:
                heatmaps = [perceptual_loss(images, healed_A)]

            heatmap = aggregate_heatmaps(
                heatmaps,
                method=heatmap_aggregation,
                logsumexp_temp=logsumexp_temp,
            )
        
        # ---------------------------------------------------------
        # Apply spatial smoothing BEFORE storing
        # This handles slight registration differences between patients
        # The smoothing makes statistics represent "anatomical regions"
        # rather than exact pixel locations
        # ---------------------------------------------------------
        padding = smoothing_kernel // 2
        heatmap_smooth = F.avg_pool2d(heatmap, kernel_size=smoothing_kernel, stride=1, padding=padding)
        
        # ---------------------------------------------------------
        # Store SMOOTHED heatmaps for statistics
        # ---------------------------------------------------------
        for i in range(B):
            hmap_np = heatmap_smooth[i, 0].cpu().numpy()  # Store SMOOTHED version
            all_heatmaps.append(hmap_np)
            
            # Extract slice index from path for per-slice stats
            if use_per_slice_stats:
                path = paths[i]
                # Try to extract slice index from filename (e.g., "slice_044.npy")
                try:
                    match = re.search(r'slice_(\d+)', os.path.basename(path))
                    if match:
                        slice_idx = int(match.group(1))
                        slice_index_heatmaps[slice_idx].append(hmap_np)  # Store SMOOTHED version
                except:
                    pass
            
            total_samples += 1

        if save_aggregation_figures and batch_idx == 0:
            out_dir = os.path.dirname(output_path)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(min(B, aggregation_figures_max_samples)):
                fig_path = os.path.join(out_dir, f"calib_heatmap_aggregations_batch0_{i}.png")
                visualize_heatmap_aggregation_comparison(
                    heatmaps,
                    save_path=fig_path,
                    sample_idx=i,
                    title=f"Calibration heatmap aggregations (sample {i})",
                    logsumexp_temp=logsumexp_temp,
                )
        
        if debug and batch_idx == 0:
            print(f"\n[Batch 0] Heatmap stats:")
            print(f"  Range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
            print(f"  Mean: {heatmap.mean():.4f}")
    
    # ---------------------------------------------------------
    # Compute Global Statistics
    # ---------------------------------------------------------
    print(f"\nComputing statistics from {total_samples} healthy samples...")
    
    stacked = np.stack(all_heatmaps, axis=0)  # [N, H, W]
    global_mu = np.mean(stacked, axis=0)  # [H, W]
    global_sigma = np.std(stacked, axis=0)  # [H, W]
    
    print(f"\nGlobal Statistics:")
    print(f"  μ range: [{global_mu.min():.4f}, {global_mu.max():.4f}]")
    print(f"  μ mean:  {global_mu.mean():.4f}")
    print(f"  σ range: [{global_sigma.min():.4f}, {global_sigma.max():.4f}]")
    print(f"  σ mean:  {global_sigma.mean():.4f}")
    
    # ---------------------------------------------------------
    # Compute Per-Slice-Index Statistics (optional)
    # ---------------------------------------------------------
    slice_index_data = None
    if use_per_slice_stats and slice_index_heatmaps:
        print(f"\nComputing per-slice-index statistics...")
        slice_index_data = {}
        
        for slice_idx in sorted(slice_index_heatmaps.keys()):
            hmaps = slice_index_heatmaps[slice_idx]
            if len(hmaps) >= 3:  # Need at least 3 samples for meaningful stats
                stacked_slice = np.stack(hmaps, axis=0)
                slice_index_data[slice_idx] = {
                    'mu': np.mean(stacked_slice, axis=0),
                    'sigma': np.std(stacked_slice, axis=0),
                    'n_samples': len(hmaps),
                }
                if debug:
                    print(f"  Slice {slice_idx}: {len(hmaps)} samples")
    
    # ---------------------------------------------------------
    # Create and Save Calibration
    # ---------------------------------------------------------
    calibration = ZScoreCalibration()
    calibration.mu = global_mu
    calibration.sigma = global_sigma
    calibration.n_samples = total_samples
    calibration.smoothing_kernel = smoothing_kernel  # Store the smoothing used
    
    if slice_index_data:
        calibration.slice_index_stats = {
            idx: {'mu': data['mu'], 'sigma': data['sigma']}
            for idx, data in slice_index_data.items()
        }
    
    calibration.save(output_path, slice_index_data, smoothing_kernel=smoothing_kernel)
    
    # ---------------------------------------------------------
    # Save Visualization of Calibration Maps
    # ---------------------------------------------------------
    viz_path = output_path.replace('.npz', '_visualization.png')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # μ map
    ax = axes[0, 0]
    im = ax.imshow(global_mu, cmap=HEATMAP_CMAP)
    ax.set_title(f'μ (Mean LPIPS)\nRange: [{global_mu.min():.3f}, {global_mu.max():.3f}]', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # σ map
    ax = axes[0, 1]
    im = ax.imshow(global_sigma, cmap=HEATMAP_CMAP)
    ax.set_title(f'σ (Std LPIPS)\nRange: [{global_sigma.min():.3f}, {global_sigma.max():.3f}]', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # μ/σ ratio (inverse coefficient of variation) - shows "consistent error" regions
    cv_inv = global_mu / (global_sigma + 0.01)
    ax = axes[0, 2]
    im = ax.imshow(cv_inv, cmap=HEATMAP_CMAP, vmin=0, vmax=np.percentile(cv_inv, 99))
    ax.set_title('μ/σ Ratio\n(High = Consistent False Positive)', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Histogram of μ
    ax = axes[1, 0]
    ax.hist(global_mu.flatten(), bins=100, color='red', alpha=0.7, edgecolor='darkred')
    ax.set_xlabel('μ (Mean LPIPS)')
    ax.set_ylabel('Pixel Count')
    ax.set_title('Distribution of μ')
    ax.grid(True, alpha=0.3)
    
    # Histogram of σ
    ax = axes[1, 1]
    ax.hist(global_sigma.flatten(), bins=100, color='green', alpha=0.7, edgecolor='darkgreen')
    ax.set_xlabel('σ (Std LPIPS)')
    ax.set_ylabel('Pixel Count')
    ax.set_title('Distribution of σ')
    ax.grid(True, alpha=0.3)
    
    # Threshold regions (high μ, low σ = false positive prone)
    ax = axes[1, 2]
    # Binary map: high mean, low std → false positive regions
    fp_prone = (global_mu > np.percentile(global_mu, 75)) & (global_sigma < np.percentile(global_sigma, 50))
    ax.imshow(fp_prone.astype(float), cmap=HEATMAP_CMAP)
    ax.set_title('False Positive Prone Regions\n(High μ, Low σ)', fontsize=11)
    ax.axis('off')
    
    plt.suptitle(f'Z-Score Calibration\n{total_samples} Healthy Samples', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved calibration visualization to: {viz_path}")
    
    return calibration


# =============================================================================
# Mask Generation with Z-Score Support
# =============================================================================

def heatmap_to_binary_mask_zscore(
    heatmap: torch.Tensor,
    calibration: ZScoreCalibration,
    z_threshold: float = 2.0,
    epsilon: float = 0.01,
    dilation_kernel_size: int = 3,
    min_region_size: int = 5,
    slice_indices: Optional[List[int]] = None,
    smoothing_kernel: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert heatmap to binary mask using Z-score thresholding.
    
    Args:
        heatmap: [B, 1, H, W] raw LPIPS values
        calibration: ZScoreCalibration object with μ and σ
        z_threshold: Z-score threshold (default: 2.0 = 2 standard deviations)
        epsilon: Small constant for numerical stability
        dilation_kernel_size: Kernel size for mask dilation
        min_region_size: Minimum region size in pixels
        slice_indices: Optional list of slice indices for per-slice calibration (one per batch item)
        smoothing_kernel: Kernel size for spatial smoothing (must match calibration!)
        
    Returns:
        binary_mask: [B, 1, H, W] thresholded mask
        zscore_map: [B, 1, H, W] Z-score values for visualization
    """
    device = heatmap.device
    B = heatmap.shape[0]
    
    # Compute Z-scores (smoothing is applied inside compute_zscore)
    zscore_map = calibration.compute_zscore(
        heatmap, 
        epsilon=epsilon, 
        slice_indices=slice_indices,
        smoothing_kernel=smoothing_kernel,
    )
    
    # Threshold on Z-score
    binary_mask = (zscore_map > z_threshold).float()
    
    # Dilation
    if dilation_kernel_size > 1:
        binary_mask = F.max_pool2d(
            binary_mask, 
            kernel_size=dilation_kernel_size, 
            stride=1, 
            padding=dilation_kernel_size // 2
        )
    
    # Connected component filtering
    binary_mask_np = binary_mask.cpu().numpy()
    cleaned_mask = np.zeros_like(binary_mask_np)
    
    for b in range(B):
        mask_2d = binary_mask_np[b, 0]
        labeled, num_features = ndimage.label(mask_2d)
        for region_id in range(1, num_features + 1):
            region = (labeled == region_id)
            if region.sum() >= min_region_size:
                cleaned_mask[b, 0][region] = 1
    
    binary_mask = torch.from_numpy(cleaned_mask).float().to(device)
    
    return binary_mask, zscore_map


def heatmap_to_binary_mask(
    heatmap: torch.Tensor,
    threshold_percentile: float = 95.0,
    threshold_absolute: Optional[float] = None,
    dilation_kernel_size: int = 3,
    min_region_size: int = 5,
) -> torch.Tensor:
    """Convert a heatmap to binary mask using percentile or absolute thresholding."""
    device = heatmap.device
    B = heatmap.shape[0]
    
    binary_mask = torch.zeros_like(heatmap)
    
    for b in range(B):
        hmap = heatmap[b].flatten()
        
        if threshold_absolute is not None:
            threshold = threshold_absolute
        else:
            threshold = torch.quantile(hmap, threshold_percentile / 100.0)
        
        binary_mask[b] = (heatmap[b] > threshold).float()
    
    if dilation_kernel_size > 1:
        binary_mask = F.max_pool2d(
            binary_mask, 
            kernel_size=dilation_kernel_size, 
            stride=1, 
            padding=dilation_kernel_size // 2
        )
    
    binary_mask_np = binary_mask.cpu().numpy()
    cleaned_mask = np.zeros_like(binary_mask_np)
    
    for b in range(B):
        mask_2d = binary_mask_np[b, 0]
        labeled, num_features = ndimage.label(mask_2d)
        for region_id in range(1, num_features + 1):
            region = (labeled == region_id)
            if region.sum() >= min_region_size:
                cleaned_mask[b, 0][region] = 1
    
    binary_mask = torch.from_numpy(cleaned_mask).float().to(device)
    
    return binary_mask


# =============================================================================
# RECURSIVE AUTOMASK V4 with Z-SCORE - Main Pipeline
# =============================================================================

@torch.no_grad()
def recursive_automask_v4_zscore(
    stage1, stage2,
    perceptual_loss: PerceptualLoss,
    images: torch.Tensor,
    slice_pos: Optional[torch.Tensor] = None,
    # Z-Score calibration
    calibration: Optional[ZScoreCalibration] = None,
    z_threshold: float = 2.0,
    z_epsilon: float = 0.01,
    slice_indices: Optional[List[int]] = None,  # NEW: List of slice indices for per-slice Z-score lookup
    smoothing_kernel: int = 15,  # NEW: Must match calibration smoothing
    # Heatmap aggregation
    heatmap_aggregation: str = "mean",
    logsumexp_temp: float = 1.0,
    # Recursive parameters
    num_iterations: int = 3,
    inter_iteration_dilation: int = 5,
    # Healing parameters
    heal_steps: int = 12,
    heal_temperature: float = 0.8,
    heal_patterns: List[int] = [0, 1],
    # Mask parameters (fallback if no calibration)
    mask_threshold_percentile: float = 95.0,
    mask_dilation: int = 3,
    mask_min_region: int = 5,
    # Artifact guard (sharpness only)
    blur_threshold: float = 0.002,
    # Inpainting parameters
    inpaint_steps: int = 12,
    inpaint_temperature: float = 0.9,
    # Token masking strategy
    token_mask_mode: str = "max",
    token_mask_avg_threshold: float = 0.5,
    token_mask_topk_ratio: float = 0.1,
    # Token surprisal (pseudo-PLL)
    token_surprisal_samples: int = 50,
    token_surprisal_mask_ratio: float = 0.15,
    token_surprisal_clamp: float = 5.0,
    compute_token_surprisal: bool = True,
    # TTA
    use_tta: bool = True,
    # Debug
    debug: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Recursive-AutoMask V4 with Z-Score Normalization.
    
    Key Innovation:
    - Iteration 0: Use Z-SCORE thresholding (if calibration available)
      - Converts LPIPS to Z = (LPIPS - μ) / (σ + ε)
      - Thresholds on Z > z_threshold
      - Eliminates "consistent" false positives (high μ, low σ regions)
    - Iteration 1+: Use percentile thresholding on LPIPS(Input, Inpainted)
      - Refinement iterations work on already-cleaned initial mask
    
    Returns:
        Dictionary with final results and iteration_history for visualization
    """
    device = images.device
    B = images.shape[0]
    
    use_zscore = calibration is not None and calibration.is_loaded()
    
    if debug:
        print(f"\n{'='*60}")
        print(f"RECURSIVE AUTOMASK V4 - Z-SCORE ENABLED: {use_zscore}")
        if use_zscore:
            print(f"Z-threshold: {z_threshold}")
            print(f"Smoothing kernel: {smoothing_kernel}")
            if slice_indices is not None:
                unique_slices = set(slice_indices)
                print(f"Slice indices in batch: {sorted(unique_slices)}")
                if calibration.slice_index_stats:
                    available = sum(1 for s in unique_slices if s in calibration.slice_index_stats)
                    print(f"Per-slice stats available: {available}/{len(unique_slices)}")
        print(f"{'='*60}")
    
    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------
    # Input-only artifact cues
    sharpness_scores = compute_sharpness_score(images)
    sharpness_map = compute_sharpness_map(images)
    
    token_surprisal_map = None
    if compute_token_surprisal:
        token_surprisal_map = compute_token_surprisal_map(
            stage1,
            stage2,
            images,
            slice_pos=slice_pos,
            num_samples=token_surprisal_samples,
            mask_ratio=token_surprisal_mask_ratio,
        )
        if token_surprisal_map.shape[-2:] != images.shape[-2:]:
            token_surprisal_map = F.interpolate(
                token_surprisal_map,
                size=images.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        if token_surprisal_clamp is not None:
            token_surprisal_map = torch.where(
                token_surprisal_map > float(token_surprisal_clamp),
                token_surprisal_map,
                torch.zeros_like(token_surprisal_map),
            )

    stage1_outputs = stage1(images)
    reconstruction = stage1_outputs["recon"]

    # Precompute LPIPS between input and reconstruction for first-iteration comparison
    lpips_input_recon = perceptual_loss(images, reconstruction)
    
    # ---------------------------------------------------------
    # Healing
    # ---------------------------------------------------------
    if debug:
        print("\n[HEAL] Running TTA Healing...")
    
    healed_A, heal_info = ensemble_heal(
        stage1, stage2, images, slice_pos,
        num_steps=heal_steps, temperature=heal_temperature,
        mask_patterns=heal_patterns, debug=False
    )
    
    if use_tta:
        images_flipped = torch.flip(images, dims=[-1])
        healed_flipped, heal_info_flip = ensemble_heal(
            stage1, stage2, images_flipped, slice_pos,
            num_steps=heal_steps, temperature=heal_temperature,
            mask_patterns=heal_patterns, debug=False
        )
        healed_B = torch.flip(healed_flipped, dims=[-1])
        healed_images_list_tta = [torch.flip(h, dims=[-1]) for h in heal_info_flip.get("healed_images_list", [])]
    else:
        healed_B = healed_A
        healed_images_list_tta = []
    
    healed_avg = 0.5 * (healed_A + healed_B)
    
    # ---------------------------------------------------------
    # ITERATION LOOP
    # ---------------------------------------------------------
    iteration_history = []
    current_inpainted = None
    zscore_map_iter0 = None
    
    for iter_idx in range(num_iterations):
        if debug:
            print(f"\n{'─'*50}")
            print(f"[ITERATION {iter_idx}]")
            print(f"{'─'*50}")
        
        # ---------------------------------------------------------
        # Compute Heatmap
        # ---------------------------------------------------------
        if iter_idx == 0:
            if heatmap_aggregation.lower().strip() == "geomean":
                # Smart-equivalent first-iteration heatmap.
                lpips_map_A = perceptual_loss(images, healed_A)
                if use_tta:
                    lpips_map_flip = perceptual_loss(images_flipped, healed_flipped)
                    lpips_map_B = torch.flip(lpips_map_flip, dims=[-1])
                else:
                    lpips_map_B = lpips_map_A
                current_heatmap = torch.sqrt(lpips_map_A * lpips_map_B + 1e-8)
                heatmap_source = "LPIPS(Input, Healed) [geomean-smart]"
            else:
                heatmaps = []
                for h in heal_info.get("healed_images_list", []):
                    heatmaps.append(perceptual_loss(images, h))
                if use_tta and healed_images_list_tta:
                    for h in healed_images_list_tta:
                        heatmaps.append(perceptual_loss(images, h))

                if not heatmaps:
                    heatmaps = [perceptual_loss(images, healed_A)]

                current_heatmap = aggregate_heatmaps(
                    heatmaps,
                    method=heatmap_aggregation,
                    logsumexp_temp=logsumexp_temp,
                )
                heatmap_source = f"LPIPS(Input, Healed) [{heatmap_aggregation}]"
            
        else:
            lpips_map_inpainted = perceptual_loss(images, current_inpainted)
            
            if use_tta:
                inpainted_flipped = torch.flip(current_inpainted, dims=[-1])
                lpips_map_flip = perceptual_loss(images_flipped, inpainted_flipped)
                lpips_map_B = torch.flip(lpips_map_flip, dims=[-1])
                current_heatmap = torch.sqrt(lpips_map_inpainted * lpips_map_B + 1e-8)
            else:
                current_heatmap = lpips_map_inpainted
            
            heatmap_source = f"LPIPS(Input, Inpainted_{iter_idx-1})"
        
        if debug:
            print(f"  Heatmap: {heatmap_source}")
            print(f"  LPIPS range: [{current_heatmap.min():.4f}, {current_heatmap.max():.4f}]")
        
        # ---------------------------------------------------------
        # Generate Binary Mask
        # ---------------------------------------------------------
        zscore_map = None
        if iter_idx == 0 and use_zscore:
            # ITERATION 0 with Z-SCORE: Use calibrated thresholding
            current_mask, zscore_map = heatmap_to_binary_mask_zscore(
                current_heatmap,
                calibration=calibration,
                z_threshold=z_threshold,
                epsilon=z_epsilon,
                dilation_kernel_size=mask_dilation,
                min_region_size=mask_min_region,
                slice_indices=slice_indices,  # PASS SLICE INDICES
                smoothing_kernel=smoothing_kernel,  # PASS SMOOTHING KERNEL
            )
            zscore_map_iter0 = zscore_map.clone()
            
            if debug:
                print(f"  Z-SCORE thresholding (z > {z_threshold})")
                print(f"  Z-score range: [{zscore_map.min():.2f}, {zscore_map.max():.2f}]")
                print(f"  Pixels with Z > {z_threshold}: {(zscore_map > z_threshold).float().mean()*100:.2f}%")
        else:
            # ITERATIONS 1+ (or no calibration): Use percentile thresholding
            current_mask = heatmap_to_binary_mask(
                current_heatmap,
                threshold_percentile=mask_threshold_percentile,
                dilation_kernel_size=mask_dilation,
                min_region_size=mask_min_region,
            )
            
            if debug:
                print(f"  Percentile thresholding (p{mask_threshold_percentile})")
        
        mask_pre_dilation = current_mask.clone()
        mask_coverage_pre_dilation = mask_pre_dilation.mean().item()
        
        # Inter-iteration dilation
        if iter_idx > 0 and inter_iteration_dilation > 1:
            current_mask = F.max_pool2d(
                current_mask,
                kernel_size=inter_iteration_dilation,
                stride=1,
                padding=inter_iteration_dilation // 2
            )
        
        mask_coverage = current_mask.mean().item()
        
        if debug:
            print(f"  Mask coverage: {mask_coverage*100:.2f}%")
        
        # ---------------------------------------------------------
        # Targeted Inpainting
        # ---------------------------------------------------------
        current_inpainted, inpaint_info = targeted_inpaint(
            stage1, stage2, images, current_mask, slice_pos,
            num_steps=inpaint_steps,
            temperature=inpaint_temperature,
            token_mask_mode=token_mask_mode,
            token_mask_avg_threshold=token_mask_avg_threshold,
            token_mask_topk_ratio=token_mask_topk_ratio,
            debug=False
        )
        
        lpips_input_inpainted = perceptual_loss(images, current_inpainted)
        
        # ---------------------------------------------------------
        # Compute scalar scores
        # ---------------------------------------------------------
        denom = torch.tensor(current_heatmap[0].numel(), device=device).repeat(B)
        masked_heatmap = current_heatmap

        heatmap_score_mean = masked_heatmap.sum(dim=(1, 2, 3)) / denom
        heatmap_score_max = masked_heatmap.view(B, -1).max(dim=1).values
        
        # ---------------------------------------------------------
        # Store Iteration Results
        # ---------------------------------------------------------
        iter_result = {
            "iteration": iter_idx,
            "heatmap_source": heatmap_source,
            "heatmap": current_heatmap.clone(),
            "mask_pre_dilation": mask_pre_dilation.clone(),
            "mask": current_mask.clone(),
            "inpainted": current_inpainted.clone(),
            "lpips_input_inpainted": lpips_input_inpainted.clone(),
            "mask_coverage_pre_dilation": mask_coverage_pre_dilation,
            "mask_coverage": mask_coverage,
            "inpaint_l1_change": inpaint_info.get("l1_change", 0),
            "inpaint_l2_change": inpaint_info.get("l2_change", 0),
            "locked_preserved": inpaint_info.get("locked_preserved", True),
            "heatmap_score_mean": heatmap_score_mean.detach().clone(),
            "heatmap_score_max": heatmap_score_max.detach().clone(),
        }
        
        if iter_idx == 0:
            iter_result["lpips_input_recon"] = lpips_input_recon.clone()
            iter_result["masked_sum_input_recon"] = (lpips_input_recon * current_mask).sum(dim=(1, 2, 3)).detach().clone()
            if zscore_map is not None:
                iter_result["zscore_map"] = zscore_map.clone()
            # Sharpness-modulated LPIPS(Input, Recon) for visualization
            iter_result["lpips_over_sharpness"] = lpips_input_recon / (sharpness_map + 1e-6)
            iter_result["heatmap_aggregation"] = heatmap_aggregation
        
        iteration_history.append(iter_result)
    
    # ---------------------------------------------------------
    # Final Results
    # ---------------------------------------------------------
    final_iteration = iteration_history[-1]

    artifact_flag = (sharpness_scores < blur_threshold)
    final_mask = final_iteration["mask"].clone()
    if artifact_flag.any():
        guard_mask = torch.ones_like(final_mask)
        final_mask = final_mask.clone()
        final_mask[artifact_flag] = guard_mask[artifact_flag]
    final_masked_score = final_iteration["heatmap"] * final_mask
    
    results = {
        # Core outputs
        "input": images,
        "reconstruction": reconstruction,
        "healed": healed_A,
        "healed_tta": healed_B,
        "healed_avg": healed_avg,
        "healed_images_list": heal_info.get("healed_images_list", []),
        "healed_images_list_tta": healed_images_list_tta,
        "heal_patterns": heal_info.get("heal_patterns", []),
        
        # Final outputs
        "anomaly_mask": final_mask,
        "inpainted": final_iteration["inpainted"],
        "final_heatmap": final_iteration["heatmap"],
        "masked_score": final_masked_score,
        "lpips_input_inpainted": final_iteration["lpips_input_inpainted"],
        "perceptual_score_mean": final_iteration["heatmap_score_mean"],
        "perceptual_score_max": final_iteration["heatmap_score_max"],
        
        # Z-score specific
        "zscore_map": zscore_map_iter0,
        "z_threshold": z_threshold if use_zscore else None,
        "used_zscore": use_zscore,
        
        # Metrics
        "mask_coverage": final_iteration["mask_coverage"],
        "heal_l1_change": heal_info["mean_l1_change"],
        "heal_l2_change": heal_info["mean_l2_change"],
        "inpaint_l1_change": final_iteration["inpaint_l1_change"],
        "inpaint_l2_change": final_iteration["inpaint_l2_change"],
        "locked_preserved": final_iteration["locked_preserved"],
        "num_iterations": num_iterations,
        "heatmap_aggregation": heatmap_aggregation,
        # Artifact guards
        "artifact_flag": artifact_flag,
        "sharpness_score": sharpness_scores,
        "blur_threshold": blur_threshold,
        "sharpness_map": sharpness_map,

        # Token surprisal
        "token_surprisal_map": token_surprisal_map,
        "token_surprisal_samples": token_surprisal_samples if compute_token_surprisal else None,
        "token_surprisal_mask_ratio": token_surprisal_mask_ratio if compute_token_surprisal else None,
        "token_surprisal_clamp": token_surprisal_clamp if compute_token_surprisal else None,
        
        # Full history
        "iteration_history": iteration_history,
    }
    
    if debug:
        print(f"\n{'='*60}")
        print("CONVERGENCE SUMMARY")
        print(f"{'='*60}")
        coverages = [h["mask_coverage"] for h in iteration_history]
        for i, cov in enumerate(coverages):
            method = "Z-score" if (i == 0 and use_zscore) else "Percentile"
            change = "" if i == 0 else f" (Δ={cov - coverages[i-1]:+.4f})"
            print(f"  Iter {i} ({method}): Coverage = {cov*100:.2f}%{change}")
    
    return results


# =============================================================================
# Visualization with Z-Score Display
# =============================================================================

def visualize_v4_zscore(results: Dict, sample_idx: int = 0, title: str = "", save_path: str = None,
                        mask_threshold_percentile: float = 97.0, clamp_threshold: float = 0.60,
                        first_heatmap_sum_thresh: float = 300.0,
                        binary_mask_threshold: float = 0.10,
                        binary_mask_iteration: int = 0,
                        save_token_surprisal_overlay: bool = False):
    """Comprehensive visualization with Z-score map display."""
    iteration_history = results["iteration_history"]
    num_iters = len(iteration_history)
    use_zscore = results.get("used_zscore", False)
    artifact_flags = results.get("artifact_flag")
    sharpness_scores = results.get("sharpness_score")
    sharpness_map = results.get("sharpness_map")
    first_masked_sum = (
        iteration_history[0]["heatmap"][sample_idx] * iteration_history[0]["mask"][sample_idx]
    ).sum().item()
    
    num_rows = 3 + num_iters  # Added 1 row for First × Last masked score visualization
    fig = plt.figure(figsize=(32, 5 * num_rows))
    
    def to_np(x):
        if x is None:
            return None
        if x.dim() == 1:
            return x[sample_idx].detach().cpu().numpy()
        t = x[sample_idx] if x.dim() == 4 else x
        if t.dim() == 3:
            return t[0].cpu().numpy()
        return t.cpu().numpy()
    
    # Add extra columns for Z-score, artifact guard, token surprisal, and clamped visualization
    base_cols = 4  # input, recon, healed, healed_tta
    extra_z = 1 if use_zscore else 0
    extra_surprisal = 2 if results.get("token_surprisal_map") is not None else 0
    extra_guard = 1
    extra_binary = 1
    extra_legend = 1
    n_cols = base_cols + extra_z + extra_surprisal + extra_guard + extra_binary + extra_legend
    gs = gridspec.GridSpec(num_rows, n_cols, figure=fig, hspace=0.35, wspace=0.15)
    
    # ---------------------------------------------------------
    # ROW 0: Header
    # ---------------------------------------------------------
    row = 0
    col = 0
    
    ax = fig.add_subplot(gs[row, col]); col += 1
    ax.imshow(to_np(results["input"]), cmap='gray')
    ax.set_title("Input", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[row, col]); col += 1
    ax.imshow(to_np(results["reconstruction"]), cmap='gray')
    ax.set_title("Reconstruction", fontsize=12)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[row, col]); col += 1
    ax.imshow(to_np(results["healed"]), cmap='gray')
    ax.set_title("Healed", fontsize=12)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[row, col]); col += 1
    ax.imshow(to_np(results["healed_tta"]), cmap='gray')
    ax.set_title("Healed (TTA)", fontsize=12)
    ax.axis('off')

    # Token surprisal map panel (NLL of true token)
    if results.get("token_surprisal_map") is not None:
        ax = fig.add_subplot(gs[row, col]); col += 1
        surprisal_np = to_np(results["token_surprisal_map"])
        im = ax.imshow(surprisal_np, cmap=HEATMAP_CMAP)
        samples = results.get("token_surprisal_samples")
        ratio = results.get("token_surprisal_mask_ratio")
        clamp_val = results.get("token_surprisal_clamp")
        clamp_str = f", clamp>{clamp_val:g}" if clamp_val is not None else ""
        hot_px = int((surprisal_np > 0).sum())
        ax.set_title(
            f"Token Surprisal (NLL)\nT={samples}, mask={ratio}{clamp_str} | hot_px={hot_px}",
            fontsize=11,
            fontweight='bold',
        )
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(gs[row, col]); col += 1
        ax.axis('off')
        nll_info = (
            "TOKEN NLL (FIX)\n"
            "═══════════════\n"
            "logp = log_softmax(logits)\n"
            "nll = -logp[true_token]\n\n"
            "Write NLL only on\n"
            "masked tokens;\n"
            "aggregate over T."
        )
        ax.text(0.05, 0.5, nll_info, fontsize=10, family='monospace',
                va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lavender', edgecolor='purple', alpha=0.9))
    
    # Z-score info panel
    if use_zscore:
        ax = fig.add_subplot(gs[row, col]); col += 1
        ax.axis('off')
        zscore_info = (
            "Z-SCORE CALIBRATION\n"
            "═══════════════════\n"
            f"Z-threshold: {results['z_threshold']:.1f}\n\n"
            "Formula:\n"
            "Z = (LPIPS - μ) / (σ + ε)\n\n"
            "Interpretation:\n"
            "• Z > threshold → Anomaly\n"
            "• Z ≈ 0 → Expected error\n"
            "• Z < 0 → Lower than expected"
        )
        ax.text(0.1, 0.5, zscore_info, fontsize=10, family='monospace',
                va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='darkgreen', alpha=0.9))

    # Artifact guard panel
    ax = fig.add_subplot(gs[row, col]); col += 1
    ax.axis('off')
    sharp = float(sharpness_scores[sample_idx].item()) if sharpness_scores is not None else float('nan')
    flag = bool(artifact_flags[sample_idx].item()) if artifact_flags is not None else False
    guard_text = (
        "ARTIFACT GUARD\n"
        "═════════════\n"
        f"sharp: {sharp:.4f} (T<{results.get('blur_threshold', float('nan')):.4f})\n"
        f"decision: {'ARTIFACT' if flag else 'ok'}"
    )
    ax.text(0.05, 0.5, guard_text, fontsize=10, family='monospace', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='mistyrose' if flag else 'honeydew',
                      edgecolor='red' if flag else 'green', alpha=0.9))
    
    # Binary mask from masked score (>0.1) with count of active pixels
    ax = fig.add_subplot(gs[row, col]); col += 1
    iter_idx = int(np.clip(binary_mask_iteration, 0, num_iters - 1))
    masked_score_tensor = iteration_history[iter_idx]["heatmap"] * iteration_history[iter_idx]["mask"]
    masked_score_np = to_np(masked_score_tensor)
    if masked_score_np is not None:
        binary_mask = (masked_score_np > binary_mask_threshold).astype(np.float32)
        binary_sum = float(binary_mask.sum())
        im = ax.imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
        ax.set_title(
            f"Binary Sum (iter {iter_idx}, > {binary_mask_threshold:.2f})\nΣ={binary_sum:.0f}",
            fontsize=12,
            fontweight='bold',
        )
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No Masked Score", ha='center', va='center', fontsize=11)
        ax.axis('off')

    # Legend
    ax = fig.add_subplot(gs[row, col])
    ax.axis('off')
    final_score_mean = results["perceptual_score_mean"][sample_idx].item()
    final_score_max = results["perceptual_score_max"][sample_idx].item()
    legend_text = (
        f"RECURSIVE AUTOMASK V4\n"
        f"{'═'*22}\n"
        f"Iterations: {num_iters}\n"
        f"Z-Score: {'ENABLED' if use_zscore else 'DISABLED'}\n\n"
        f"First masked sum: {first_masked_sum:.2f}\n"
        f"Final mean: {final_score_mean:.4f}\n"
        f"Final max:  {final_score_max:.4f}"
    )
    ax.text(0.05, 0.5, legend_text, fontsize=10, family='monospace',
            va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='darkblue', alpha=0.9))
    
    # ---------------------------------------------------------
    # ITERATION ROWS
    # ---------------------------------------------------------
    vmax_heatmap = max(h["heatmap"][sample_idx].max().item() for h in iteration_history)
    vmax_lpips = max(h["lpips_input_inpainted"][sample_idx].max().item() for h in iteration_history)
    vmax_recon = None
    vmax_lpips_sharp = None
    if "lpips_input_recon" in iteration_history[0]:
        vmax_recon = iteration_history[0]["lpips_input_recon"][sample_idx].max().item()
    if "lpips_over_sharpness" in iteration_history[0]:
        vmax_lpips_sharp = iteration_history[0]["lpips_over_sharpness"][sample_idx].max().item()
    
    for iter_idx, iter_data in enumerate(iteration_history):
        row = 1 + iter_idx
        col = 0
        
        # Heatmap
        ax = fig.add_subplot(gs[row, col])
        hmap = to_np(iter_data["heatmap"])
        im = ax.imshow(hmap, cmap=HEATMAP_CMAP, vmin=0, vmax=vmax_heatmap)
        score_mean = iter_data["heatmap_score_mean"][sample_idx].item()
        score_max = iter_data["heatmap_score_max"][sample_idx].item()
        ax.set_title(f"ITER {iter_idx}: Raw LPIPS\nmax={hmap.max():.3f}", fontsize=10, fontweight='bold')
        ax.text(0.5, 0.05, f"mean={score_mean:.4f} max={score_max:.4f}",
                transform=ax.transAxes, fontsize=9, ha='center', color='yellow',
                bbox=dict(facecolor='black', alpha=0.7))
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        col += 1
        
        # Z-Score map (iter 0 only)
        if use_zscore:
            ax = fig.add_subplot(gs[row, col])
            if iter_idx == 0 and "zscore_map" in iter_data:
                zscore_np = to_np(iter_data["zscore_map"])
                # Center colormap around 0, cap at reasonable range
                vmax_z = max(abs(zscore_np.min()), abs(zscore_np.max()), 5)
                im = ax.imshow(zscore_np, cmap='RdBu_r', vmin=-vmax_z, vmax=vmax_z)
                ax.set_title(f"Z-SCORE Map\nThreshold: {results['z_threshold']}", 
                            fontsize=10, color='purple', fontweight='bold')
                # Mark threshold contour
                ax.contour(zscore_np, levels=[results['z_threshold']], colors='lime', linewidths=2)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, "N/A\n(percentile mode)", ha='center', va='center', fontsize=10)
            ax.axis('off')
            col += 1
        
        # Pre-dilation mask
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(to_np(results["input"]), cmap='gray', alpha=0.5)
        mask_pre = to_np(iter_data["mask_pre_dilation"])
        ax.imshow(mask_pre, cmap='Oranges', alpha=0.6, vmin=0, vmax=1)
        cov_pre = iter_data["mask_coverage_pre_dilation"]
        method = "Z>thr" if (iter_idx == 0 and use_zscore) else f"p{mask_threshold_percentile}"
        ax.set_title(f"Mask ({method})\nCov: {cov_pre:.2%}", fontsize=10, color='orange')
        ax.axis('off')
        col += 1
        
        # Post-dilation mask
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(to_np(results["input"]), cmap='gray', alpha=0.5)
        mask_np = to_np(iter_data["mask"])
        ax.imshow(mask_np, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        cov = iter_data["mask_coverage"]
        ax.set_title(f"Mask (dilated)\nCov: {cov:.2%}", fontsize=10, color='red')
        ax.axis('off')
        col += 1
        
        # Inpainted
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(to_np(iter_data["inpainted"]), cmap='gray')
        ax.set_title(f"Inpainted {iter_idx}", fontsize=10, color='green')
        ax.axis('off')
        col += 1
        
        # LPIPS(Input, Inpainted)
        ax = fig.add_subplot(gs[row, col])
        lpips_np = to_np(iter_data["lpips_input_inpainted"])
        im = ax.imshow(lpips_np, cmap=HEATMAP_CMAP, vmin=0, vmax=vmax_lpips)
        ax.set_title(f"LPIPS(In, Inp)\nmax={lpips_np.max():.3f}", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        col += 1

        # LPIPS(Input, Reconstruction) — first iteration only
        if iter_idx == 0 and "lpips_input_recon" in iter_data:
            ax = fig.add_subplot(gs[row, col])
            lpips_recon = to_np(iter_data["lpips_input_recon"])
            v_recon = vmax_recon if vmax_recon is not None else lpips_recon.max()
            im = ax.imshow(lpips_recon, cmap=HEATMAP_CMAP, vmin=0, vmax=v_recon)
            mask_np = to_np(iter_data["mask"])
            masked_sum_recon = float((lpips_recon * mask_np).sum()) if mask_np is not None else float(lpips_recon.sum())
            ax.set_title(
                f"LPIPS(In, Recon)\nΣmask={masked_sum_recon:.2f}",
                fontsize=10,
                color='purple',
                fontweight='bold',
            )
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            col += 1

            if "lpips_over_sharpness" in iter_data and col < n_cols:
                ax = fig.add_subplot(gs[row, col])
                lpips_sharp = to_np(iter_data["lpips_over_sharpness"])
                v_lpips_sharp = vmax_lpips_sharp if vmax_lpips_sharp is not None else lpips_sharp.max()
                im = ax.imshow(lpips_sharp, cmap=HEATMAP_CMAP, vmin=0, vmax=v_lpips_sharp)
                ax.set_title("LPIPS(Recon) / Sharpness", fontsize=10, fontweight='bold', color='darkmagenta')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                col += 1
        
        # Masked score
        if col < n_cols:
            ax = fig.add_subplot(gs[row, col])
            masked_score = hmap * mask_np
            im = ax.imshow(masked_score, cmap=HEATMAP_CMAP, vmin=0, vmax=vmax_heatmap)
            masked_sum = masked_score.sum().item()
            title = f"Masked Score\nmax={masked_score.max():.3f}"
            if iter_idx == 0:
                # Highlight the first-iteration masked heatmap sum for anomaly risk context.
                title = f"Masked Score\nΣ={masked_sum:.2f} | max={masked_score.max():.3f}"
                title_color = 'green' if masked_sum < first_heatmap_sum_thresh else 'red'
                ax.set_title(title, fontsize=10, color=title_color)
            else:
                ax.set_title(title, fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # ---------------------------------------------------------
    # FINAL ROW
    # ---------------------------------------------------------
    row = 1 + num_iters
    
    ax = fig.add_subplot(gs[row, 0])
    ax.imshow(to_np(results["input"]), cmap='gray', alpha=0.5)
    ax.imshow(to_np(iteration_history[0]["mask"]), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    ax.set_title(f"INITIAL Mask\nCov: {iteration_history[0]['mask_coverage']:.2%}", 
                fontsize=11, color='darkred', fontweight='bold')
    ax.axis('off')
    
    col = 1
    if use_zscore:
        ax = fig.add_subplot(gs[row, col])
        zscore_map = results.get("zscore_map")
        if zscore_map is not None:
            zscore_np = to_np(zscore_map)
            # Show histogram of Z-scores
            ax.hist(zscore_np.flatten(), bins=100, color='purple', alpha=0.7, edgecolor='darkviolet')
            ax.axvline(results['z_threshold'], color='red', linestyle='--', linewidth=2, label=f'Z={results["z_threshold"]}')
            ax.set_xlabel('Z-score')
            ax.set_ylabel('Pixel count')
            ax.set_title(f'Z-Score Distribution\nPixels > {results["z_threshold"]}: {(zscore_np > results["z_threshold"]).mean()*100:.1f}%')
            ax.legend()
            ax.grid(True, alpha=0.3)
        col += 1
    
    ax = fig.add_subplot(gs[row, col])
    ax.imshow(to_np(results["input"]), cmap='gray', alpha=0.5)
    mask_overlay = to_np(results["anomaly_mask"])
    mask_overlay = mask_overlay.astype(float, copy=True)
    mask_overlay[mask_overlay == 0] = np.nan
    greens_cmap = plt.cm.get_cmap('Greens').copy()
    greens_cmap.set_bad(alpha=0)
    ax.imshow(mask_overlay, cmap=greens_cmap, alpha=0.6, vmin=0, vmax=1)
    ax.set_title(f"OUTPUT Mask (post-guard)\nCov: {results['anomaly_mask'].mean().item():.2%}", 
                fontsize=11, color='darkgreen', fontweight='bold')
    ax.axis('off')
    col += 1
    
    # Mask evolution
    ax = fig.add_subplot(gs[row, col])
    ax.imshow(to_np(results["input"]), cmap='gray', alpha=0.4)
    initial_mask = to_np(iteration_history[0]["mask"])
    final_mask = to_np(iteration_history[-1]["mask"])
    rgb_overlay = np.zeros((*initial_mask.shape, 3))
    rgb_overlay[..., 0] = initial_mask
    rgb_overlay[..., 1] = final_mask
    overlay_alpha = (rgb_overlay.sum(axis=-1) > 0).astype(float) * 0.7
    ax.imshow(rgb_overlay, alpha=overlay_alpha)
    ax.set_title("Evolution\nRed=Init | Green=Final", fontsize=10)
    ax.axis('off')
    col += 1
    
    # Coverage plot
    ax = fig.add_subplot(gs[row, col])
    coverages = [h["mask_coverage"] * 100 for h in iteration_history]
    iterations = list(range(len(coverages)))
    ax.plot(iterations, coverages, 'bo-', linewidth=2, markersize=10)
    ax.fill_between(iterations, coverages, alpha=0.3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"Coverage Evolution\n{coverages[0]:.1f}% → {coverages[-1]:.1f}%", fontweight='bold')
    ax.grid(True, alpha=0.3)
    col += 1
    
    # Final inpainted
    ax = fig.add_subplot(gs[row, col])
    ax.imshow(to_np(results["inpainted"]), cmap='gray')
    ax.set_title("FINAL Inpainted", fontsize=11, color='darkgreen', fontweight='bold')
    ax.axis('off')
    col += 1

    # Guarded mask overlay (if space)
    if col < n_cols:
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(to_np(results["input"]), cmap='gray', alpha=1)
        guarded_mask_overlay = to_np(results["anomaly_mask"])
        guarded_mask_overlay = guarded_mask_overlay.astype(float, copy=True)
        guarded_mask_overlay[guarded_mask_overlay == 0] = np.nan
        viridis_cmap = plt.cm.get_cmap('viridis').copy()
        viridis_cmap.set_bad(alpha=0)
        ax.imshow(guarded_mask_overlay, cmap=viridis_cmap, alpha=0.5, vmax=1)
        flag = bool(artifact_flags[sample_idx].item()) if artifact_flags is not None else False
        ax.set_title(f"Guarded Mask Overlay\nartifact={flag}", fontsize=10, fontweight='bold')
        ax.axis('off')
        col += 1
    
    # Final LPIPS
    if col < n_cols:
        ax = fig.add_subplot(gs[row, col])
        final_lpips = to_np(results["lpips_input_inpainted"])
        token_surprisal_np = to_np(results.get("token_surprisal_map"))
        final_lpips_display, _, used_token_fusion = fuse_lpips_with_token_surprisal_for_display(
            final_lpips,
            token_surprisal_np,
            lpips_percentile=60,
            display_floor=0.5,
        )
        ax.imshow(to_np(results["input"]), cmap='gray', alpha=1)
        heatmap_cmap = plt.cm.get_cmap(HEATMAP_CMAP).copy()
        heatmap_cmap.set_bad(alpha=0)
        alpha = np.zeros_like(final_lpips_display, dtype=np.float32)
        valid = np.isfinite(final_lpips_display)
        if np.any(valid):
            alpha[valid] = 0.25 + 0.45 * np.clip((final_lpips_display[valid] - 0.5) / 0.5, 0.0, 1.0)
        im = ax.imshow(final_lpips_display, cmap=heatmap_cmap, alpha=alpha, vmin=0.5, vmax=1.0)
        max_val = float(np.nanmax(final_lpips_display)) if np.any(np.isfinite(final_lpips_display)) else 0.0
        if used_token_fusion:
            ax.set_title(
                f"FINAL LPIPS + TokenSurprisal (>0.5)\nmax={max_val:.3f}",
                fontsize=11,
                fontweight='bold',
            )
        else:
            ax.set_title(
                f"FINAL LPIPS (>= p60, >0.5)\nmax={max_val:.3f}",
                fontsize=11,
                fontweight='bold',
            )
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # ---------------------------------------------------------
    # NEW ROW: First × Last Masked Score Multiplication
    # ---------------------------------------------------------
    row = 2 + num_iters
    
    # Compute first and last masked scores
    first_hmap = to_np(iteration_history[0]["heatmap"])
    first_mask = to_np(iteration_history[0]["mask"])
    first_masked_score = first_hmap * first_mask
    
    last_hmap = to_np(iteration_history[-1]["heatmap"])
    last_mask = to_np(iteration_history[-1]["mask"])
    last_masked_score = last_hmap * last_mask
    
    # Multiplication of first and last masked scores
    multiplied_score = first_masked_score * last_masked_score
    
    # Normalize for better visualization (optional: geometric mean style)
    # sqrt to keep scale reasonable when multiplying two heatmaps
    multiplied_score_sqrt = np.sqrt(multiplied_score)
    
    # Compute sum of all pixels in sqrt map (anomaly severity metric)
    pixel_sum_sqrt = np.sum(multiplied_score_sqrt)
    
    # Clamped version: only pixels exceeding threshold contribute
    multiplied_score_sqrt_clamped = np.where(multiplied_score_sqrt > clamp_threshold, multiplied_score_sqrt, 0)
    pixel_sum_clamped = np.sum(multiplied_score_sqrt_clamped)
    num_pixels_above_thresh = np.sum(multiplied_score_sqrt > clamp_threshold)
    
    col = 0
    
    # First masked score
    ax = fig.add_subplot(gs[row, col])
    im = ax.imshow(first_masked_score, cmap=HEATMAP_CMAP, vmin=0)
    ax.set_title(f"First Masked Score\nmax={first_masked_score.max():.3f}", fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    col += 1
    
    # Multiplication symbol / info panel
    if use_zscore:
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
        ax.text(0.5, 0.5, "×", fontsize=48, ha='center', va='center', fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.2, "First × Last\nMasked Scores", fontsize=10, ha='center', va='center',
                transform=ax.transAxes)
        col += 1
    
    # Last masked score
    ax = fig.add_subplot(gs[row, col])
    im = ax.imshow(last_masked_score, cmap=HEATMAP_CMAP, vmin=0)
    ax.set_title(f"Last Masked Score\nmax={last_masked_score.max():.3f}", fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    col += 1
    
    # Equals symbol
    ax = fig.add_subplot(gs[row, col])
    ax.axis('off')
    ax.text(0.5, 0.5, "=", fontsize=48, ha='center', va='center', fontweight='bold',
            transform=ax.transAxes)
    col += 1
    
    # Multiplied result (raw)
    ax = fig.add_subplot(gs[row, col])
    im = ax.imshow(multiplied_score, cmap=HEATMAP_CMAP, vmin=0)
    ax.set_title(f"First × Last\nmax={multiplied_score.max():.4f}", fontsize=10, fontweight='bold', color='purple')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    col += 1
    
    # Multiplied result (sqrt normalized for better visualization)
    ax = fig.add_subplot(gs[row, col])
    im = ax.imshow(multiplied_score_sqrt, cmap=HEATMAP_CMAP, vmin=0)
    ax.set_title(f"√(First × Last)\nmax={multiplied_score_sqrt.max():.3f} | Σpx={pixel_sum_sqrt:.2f}", fontsize=10, fontweight='bold', color='darkmagenta')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    col += 1
    
    # Clamped √(First × Last) - only pixels > threshold
    ax = fig.add_subplot(gs[row, col])
    im = ax.imshow(multiplied_score_sqrt_clamped, cmap=HEATMAP_CMAP, vmin=0)
    ax.set_title(f"√(F×L) > {clamp_threshold}\nΣpx={pixel_sum_clamped:.2f} | #px={num_pixels_above_thresh}", fontsize=10, fontweight='bold', color='crimson')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    col += 1
    
    # Overlay on input image (using clamped version)
    if col < n_cols:
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(to_np(results["input"]), cmap='gray', alpha=0.6)
        # Normalize clamped score to [0,1] for overlay
        mult_norm = multiplied_score_sqrt_clamped / (multiplied_score_sqrt_clamped.max() + 1e-8)
        im = ax.imshow(mult_norm, cmap=HEATMAP_CMAP, alpha=0.7, vmin=0, vmax=1)
        ax.set_title(f"Consistent Anomalies\n(clamped overlay)", fontsize=10, fontweight='bold', color='red')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Recursive AutoMask V4 {'(Z-Score Calibrated)' if use_zscore else ''}\n{title}", 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if save_token_surprisal_overlay and results.get("token_surprisal_map") is not None:
            base_lpips = to_np(results["lpips_input_inpainted"])
            surprisal_np = to_np(results["token_surprisal_map"])
            if base_lpips is not None and surprisal_np is not None:
                p_lpips = 60
                p_cut = np.percentile(base_lpips, p_lpips)
                lpips_clamped = np.where(base_lpips >= p_cut, base_lpips, np.nan)

                fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(6, 6))
                im_base = ax_overlay.imshow(lpips_clamped, cmap=HEATMAP_CMAP)
                im_overlay = ax_overlay.imshow(surprisal_np, cmap=HEATMAP_CMAP, alpha=0.6)
                ax_overlay.set_title(f"Token Surprisal (NLL) on Final LPIPS\nLPIPS >= p{p_lpips}", fontsize=11, fontweight='bold')
                ax_overlay.axis('off')
                plt.colorbar(im_base, ax=ax_overlay, fraction=0.046, pad=0.04)

                overlay_path = save_path.replace("_full.png", "_TokenSurprisal_On_FinalLPIPS.png")
                plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
                plt.close(fig_overlay)
    
    return fig


# =============================================================================
# Anomaly Overlay Visualization (Simplified Independent Figure)
# =============================================================================

def visualize_anomaly_overlay(
    results: Dict, 
    sample_idx: int = 0, 
    title: str = "", 
    save_path: str = None,
    heatmap_overlay_viz_clamp: float = 0.5,
    annotation_boxes: Optional[dict[str, dict[int, list[dict]]]] = None,
    file_stem: Optional[str] = None,
    slice_idx: Optional[int] = None,
):
    """
    Create a simplified anomaly overlay figure showing:
    - Input image
    - Reconstruction image  
    - Healed image
    - Average mask score heatmap (sum of 3 iteration heatmaps / 3)
    - Heatmap overlay on input
    - Total pixel sum of the average heatmap
    
    Args:
        results: Dictionary containing pipeline results
        sample_idx: Index of sample in batch
        title: Title for the figure
        save_path: Path to save figure (should end with _Anomaly_Overlay.png)
    
    Returns:
        matplotlib figure
    """
    iteration_history = results["iteration_history"]
    num_iters = len(iteration_history)
    sharpness_map = results.get("sharpness_map")
    has_sharpness_map = False
    
    # Helper to convert tensor to numpy
    def to_np(x):
        if x is None:
            return None
        t = x[sample_idx] if x.dim() == 4 else x
        if t.dim() == 3:
            return t[0].cpu().numpy()
        return t.cpu().numpy()
    
    # Compute average heatmap across all iterations (masked scores)
    heatmaps = []
    for iter_data in iteration_history:
        hmap = iter_data["heatmap"][sample_idx, 0].cpu().numpy()
        mask = iter_data["mask"][sample_idx, 0].cpu().numpy()
        masked_hmap = hmap * mask
        heatmaps.append(masked_hmap)
    
    # Average heatmap = sum / number of iterations
    average_heatmap = np.sum(heatmaps, axis=0) / num_iters
    
    # Compute total pixel sum of average heatmap
    total_pixel_sum = np.sum(average_heatmap)
    
    # Create figure without sharpness map subplot
    n_cols = 5
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    # 1. Input Image
    input_img = to_np(results["input"])
    vmin = np.percentile(input_img, 0.1)
    vmax = np.percentile(input_img, 99.0)

    axes[0].imshow(input_img, cmap='gray', alpha=1, vmin=vmin, vmax=vmax)
    axes[0].set_title("Input", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Reconstruction Image
    healed_img = to_np(results["healed"])
    #recon_img = to_np(results["reconstruction"])
    #
    axes[1].imshow(healed_img, cmap='gray')
    axes[1].set_title("Healed", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    #
    ## 3. Healed Image
    #axes[2].imshow(healed_img, cmap='gray')
    #axes[2].set_title("Healed", fontsize=14, fontweight='bold')
    #axes[2].axis('off')
    #
    ## 4. Average Mask Score Heatmap
    #im = axes[3].imshow(average_heatmap, cmap='inferno', vmin=0)
    #axes[3].set_title(f"Avg Masked Heatmap\n(Σpx = {total_pixel_sum:.2f})", fontsize=14, fontweight='bold')
    #axes[3].axis('off')
    #plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    # 5. Heatmap Overlay on Input

    axes[2].imshow(input_img, cmap='gray', alpha=1, vmin=vmin, vmax=vmax)
    # Normalize heatmap for overlay visualization
    heatmap_norm = average_heatmap / (average_heatmap.max() + 1e-8)
    heatmap_norm_clamped = np.where(heatmap_norm >= heatmap_overlay_viz_clamp, heatmap_norm, 0)
    heatmap_overlay = heatmap_norm_clamped.astype(float, copy=True)
    heatmap_overlay[heatmap_overlay == 0] = np.nan
    heatmap_cmap = plt.cm.get_cmap(HEATMAP_CMAP).copy()
    heatmap_cmap.set_bad(alpha=0)
    im_overlay = axes[2].imshow(heatmap_overlay, cmap=heatmap_cmap, alpha=0.5)
    axes[2].set_title(
        f"Heatmap Overlay on Input (>= {heatmap_overlay_viz_clamp:.2f})",  #\n(Σpx = {total_pixel_sum:.2f})",
        fontsize=14,
        fontweight='bold',
        color='darkred',
    )
    axes[2].axis('off')
    plt.colorbar(im_overlay, ax=axes[2], fraction=0.046, pad=0.04)

    # 6. Final LPIPS heatmap (clamped at p75) overlaid on input
    final_lpips = to_np(results.get("lpips_input_inpainted"))
    if final_lpips is not None:
        token_surprisal_np = to_np(results.get("token_surprisal_map"))
        final_lpips_display, _, used_token_fusion = fuse_lpips_with_token_surprisal_for_display(
            final_lpips,
            token_surprisal_np,
            lpips_percentile=60,
            display_floor=0.5,
        )

        axes[3].imshow(input_img, cmap='gray', alpha=1, vmin=vmin, vmax=vmax)
        heatmap_cmap = plt.cm.get_cmap(HEATMAP_CMAP).copy()
        heatmap_cmap.set_bad(alpha=0)
        lpips_alpha = np.zeros_like(final_lpips_display, dtype=np.float32)
        lpips_valid = np.isfinite(final_lpips_display)
        if np.any(lpips_valid):
            lpips_alpha[lpips_valid] = 0.25 + 0.45 * np.clip((final_lpips_display[lpips_valid] - 0.5) / 0.5, 0.0, 1.0)
        im_lpips = axes[3].imshow(final_lpips_display, cmap=heatmap_cmap, alpha=lpips_alpha, vmin=0.5, vmax=1.0)

        box_count = 0
        if annotation_boxes is not None and file_stem is not None and slice_idx is not None:
            boxes = annotation_boxes.get(file_stem, {}).get(slice_idx, [])
            if boxes:
                h, w = final_lpips.shape
                scale_x = w / ANNOTATION_BASE_SIZE
                scale_y = h / ANNOTATION_BASE_SIZE
                box_count = draw_annotation_boxes(axes[3], boxes, scale_x=scale_x, scale_y=scale_y)

        title_suffix = f"\nboxes={box_count}" if box_count else ""
        if used_token_fusion:
            axes[3].set_title(
                f"Final LPIPS + TokenSurprisal (>0.5){title_suffix}",
                fontsize=13,
                fontweight='bold',
                color='darkblue',
            )
        else:
            axes[3].set_title(
                f"Final LPIPS Overlay (>= p60, >0.5){title_suffix}",
                fontsize=13,
                fontweight='bold',
                color='darkblue',
            )
        axes[3].axis('off')
        plt.colorbar(im_lpips, ax=axes[3], fraction=0.046, pad=0.04)
    else:
        axes[3].text(0.5, 0.5, "No final LPIPS", ha='center', va='center', fontsize=12)
        axes[3].axis('off')

    # Final LPIPS raw heatmap
    if final_lpips is not None:
        im_raw = axes[4].imshow(final_lpips, cmap=HEATMAP_CMAP)
        axes[4].set_title(f"Final LPIPS (raw)\nmax={np.nanmax(final_lpips):.3f}", fontsize=13, fontweight='bold')
        axes[4].axis('off')
        plt.colorbar(im_raw, ax=axes[4], fraction=0.046, pad=0.04)
    else:
        axes[4].text(0.5, 0.5, "No final LPIPS", ha='center', va='center', fontsize=12)
        axes[4].axis('off')

    # Sharpness map disabled per request
    
    # Add overall title with artifact guard decision
    flag = bool(results.get("artifact_flag", torch.zeros(1))[sample_idx].item()) if results.get("artifact_flag") is not None else False
    sharp = float(results.get("sharpness_score", torch.zeros(1))[sample_idx].item()) if results.get("sharpness_score") is not None else float('nan')
    guard_summary = f"artifact_guard: {'ARTIFACT' if flag else 'ok'} | sharp={sharp:.4f}"
    plt.suptitle(f"{title}\n{guard_summary}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


# =============================================================================
# Main Inference
# =============================================================================

def run_inference_v4_zscore(
    stage1, stage2,
    perceptual_loss: PerceptualLoss,
    dataloader,
    output_dir: str,
    calibration: Optional[ZScoreCalibration] = None,
    z_threshold: float = 2.0,
    smoothing_kernel: int = 15,
    heatmap_aggregation: str = "mean",
    logsumexp_temp: float = 1.0,
    clamp_threshold: float = 0.60,
    first_heatmap_sum_thresh: float = 300.0,
    binary_mask_threshold: float = 0.10,
    binary_mask_iteration: int = 0,
    heatmap_overlay_viz_clamp: float = 0.5,
    device: str = "cuda",
    num_iterations: int = 3,
    inter_iteration_dilation: int = 5,
    save_all_visualizations: bool = True,
    flip_upside_down: bool = False,
    enable_visualizations: bool = True,
    save_aggregation_figures: bool = True,
    aggregation_figures_max_samples: int = 3,
    save_token_mask_figures: bool = True,
    token_mask_avg_threshold: float = 0.5,
    token_mask_topk_ratio: float = 0.1,
    token_surprisal_samples: int = 50,
    token_surprisal_mask_ratio: float = 0.15,
    token_surprisal_clamp: float = 5.0,
    compute_token_surprisal: bool = True,
    annotation_csv: Optional[str] = None,
    overlay_annotation_boxes: bool = True,
    **pipeline_kwargs,
) -> Dict:
    """Run V4 inference with Z-score calibration."""
    
    os.makedirs(output_dir, exist_ok=True)
    visualizations_dir = os.path.join(output_dir, "visualizations_full")
    if enable_visualizations:
        os.makedirs(visualizations_dir, exist_ok=True)
    
    results_list = []
    debug_first = pipeline_kwargs.pop("debug_first_batch", True)
    mask_threshold_percentile = pipeline_kwargs.get("mask_threshold_percentile", 97.0)
    annotation_boxes = {}
    if overlay_annotation_boxes and annotation_csv:
        annotation_boxes = load_annotation_boxes(Path(annotation_csv))
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running V4 Pipeline")):
        images = batch["image"].to(device)
        if flip_upside_down:
            images = torch.flip(images, dims=[2])
        paths = batch["path"]
        metadata_list = batch.get("metadata", [{}] * len(paths))
        
        if hasattr(stage2, "_extract_slice_indices"):
            slice_pos = stage2._extract_slice_indices(paths, device)
        else:
            slice_pos = None
        
        # ---------------------------------------------------------
        # Extract integer slice indices for Z-score per-slice lookup
        # ---------------------------------------------------------
        slice_indices_list = []
        for p in paths:
            match = re.search(r'slice_(\d+)', os.path.basename(p))
            if match:
                slice_indices_list.append(int(match.group(1)))
            else:
                slice_indices_list.append(-1)  # -1 indicates no slice index found
        
        debug = debug_first and batch_idx == 0
        
        pipeline_results = recursive_automask_v4_zscore(
            stage1, stage2, perceptual_loss,
            images, slice_pos,
            calibration=calibration,
            z_threshold=z_threshold,
            slice_indices=slice_indices_list,  # PASS SLICE INDICES
            smoothing_kernel=smoothing_kernel,  # PASS SMOOTHING KERNEL
            heatmap_aggregation=heatmap_aggregation,
            logsumexp_temp=logsumexp_temp,
            num_iterations=num_iterations,
            inter_iteration_dilation=inter_iteration_dilation,
            debug=debug,
            token_surprisal_samples=token_surprisal_samples,
            token_surprisal_mask_ratio=token_surprisal_mask_ratio,
            token_surprisal_clamp=token_surprisal_clamp,
            compute_token_surprisal=compute_token_surprisal,
            **pipeline_kwargs,
        )
        
        for i in range(images.shape[0]):
            path = paths[i]
            if isinstance(metadata_list, dict):
                meta = {k: (v[i] if isinstance(v, (list, tuple)) else v) for k, v in metadata_list.items()}
            elif isinstance(metadata_list, list):
                meta = metadata_list[i] if i < len(metadata_list) else {}
            else:
                meta = {}
            
            filename = os.path.basename(path)
            file_stem, slice_idx = parse_slice_info(filename)
            category = meta.get('category', 'Unknown') if isinstance(meta, dict) else 'Unknown'
            case_folder = meta.get('case_folder', 'Unknown') if isinstance(meta, dict) else 'Unknown'
            
            # Binary sum of masked score heatmap for plotting later
            iter_sel = num_iterations - 1 if binary_mask_iteration < 0 else binary_mask_iteration
            iter_sel = max(0, min(iter_sel, num_iterations - 1))
            hmap_i = pipeline_results["iteration_history"][iter_sel]["heatmap"][i, 0].cpu().numpy()
            mask_i = pipeline_results["iteration_history"][iter_sel]["mask"][i, 0].cpu().numpy()
            masked_score_i = hmap_i * mask_i
            binary_sum_heatmap = float((masked_score_i > binary_mask_threshold).sum())

            iteration_metrics = []
            for iter_data in pipeline_results["iteration_history"]:
                iteration_metrics.append({
                    "iteration": iter_data["iteration"],
                    "mask_coverage_pre_dilation": iter_data["mask_coverage_pre_dilation"],
                    "mask_coverage": iter_data["mask_coverage"],
                    "inpaint_l1_change": iter_data["inpaint_l1_change"],
                    "max_lpips_inpainted": iter_data["lpips_input_inpainted"][i].max().item(),
                })
            
            # Compute clamped pixel sum for this slice (First × Last masked score)
            first_hmap = pipeline_results["iteration_history"][0]["heatmap"][i, 0].cpu().numpy()
            first_mask = pipeline_results["iteration_history"][0]["mask"][i, 0].cpu().numpy()
            last_hmap = pipeline_results["iteration_history"][-1]["heatmap"][i, 0].cpu().numpy()
            last_mask = pipeline_results["iteration_history"][-1]["mask"][i, 0].cpu().numpy()
            
            first_masked_score = first_hmap * first_mask
            first_heatmap_sum = float(np.sum(first_masked_score))
            last_masked_score = last_hmap * last_mask
            multiplied_score_sqrt = np.sqrt(first_masked_score * last_masked_score)
            
            clamped_pixel_sum = float(np.sum(np.where(multiplied_score_sqrt > clamp_threshold, multiplied_score_sqrt, 0)))
            num_pixels_above_thresh = int(np.sum(multiplied_score_sqrt > clamp_threshold))

            masked_sum_input_recon = None
            if "masked_sum_input_recon" in pipeline_results["iteration_history"][0]:
                masked_sum_input_recon = float(pipeline_results["iteration_history"][0]["masked_sum_input_recon"][i].item())
            token_surprisal_hot_px = None
            if pipeline_results.get("token_surprisal_map") is not None:
                token_surprisal_hot_px = int((pipeline_results["token_surprisal_map"][i, 0] > 0).sum().item())
            
            result = {
                "path": path,
                "filename": filename,
                "category": category,
                "case_folder": case_folder,
                "num_iterations": num_iterations,
                "used_zscore": pipeline_results["used_zscore"],
                "z_threshold": z_threshold if pipeline_results["used_zscore"] else None,
                "initial_mask_coverage": pipeline_results["iteration_history"][0]["mask_coverage"],
                "final_mask_coverage": pipeline_results["mask_coverage"],
                "coverage_reduction": pipeline_results["iteration_history"][0]["mask_coverage"] - pipeline_results["mask_coverage"],
                "final_perceptual_mean": pipeline_results['perceptual_score_mean'][i].item(),
                "final_perceptual_max": pipeline_results['perceptual_score_max'][i].item(),
                "clamped_pixel_sum": clamped_pixel_sum,
                "clamped_pixel_sum_FirstHeatmap": first_heatmap_sum,
                "num_pixels_above_clamp_thresh": num_pixels_above_thresh,
                "clamp_threshold": clamp_threshold,
                "lpips_input_recon_sum_mask": masked_sum_input_recon,
                "token_surprisal_hot_px": token_surprisal_hot_px,
                "iteration_metrics": iteration_metrics,
                "Binary_Sum_Heatmap": binary_sum_heatmap,
                "artifact_flag": bool(pipeline_results.get("artifact_flag")[i].item()),
                "sharpness_score": float(pipeline_results.get("sharpness_score")[i].item()),
            }
            
            results_list.append(result)
            
            if enable_visualizations and (save_all_visualizations or (batch_idx < 3 and i < 4)):
                base_name = f"{category}_{case_folder}_{filename.replace('.npy', '')}"
                save_path = os.path.join(visualizations_dir, f"{base_name}_full.png")
                visualize_v4_zscore(
                    pipeline_results, sample_idx=i,
                    title=f"{category}/{case_folder}/{filename}",
                    save_path=save_path,
                    mask_threshold_percentile=mask_threshold_percentile,
                    clamp_threshold=clamp_threshold,
                    first_heatmap_sum_thresh=first_heatmap_sum_thresh,
                    binary_mask_threshold=binary_mask_threshold,
                    binary_mask_iteration=binary_mask_iteration,
                )
                
                # Save simplified Anomaly Overlay figure
                anomaly_overlay_path = os.path.join(visualizations_dir, f"{base_name}_Anomaly_Overlay.png")
                visualize_anomaly_overlay(
                    pipeline_results, sample_idx=i,
                    title=f"{category}/{case_folder}/{filename}",
                    save_path=anomaly_overlay_path,
                    heatmap_overlay_viz_clamp=heatmap_overlay_viz_clamp,
                    annotation_boxes=annotation_boxes,
                    file_stem=file_stem,
                    slice_idx=slice_idx,
                )

                if save_aggregation_figures and i < aggregation_figures_max_samples:
                    healed_list = pipeline_results.get("healed_images_list", [])
                    healed_list_tta = pipeline_results.get("healed_images_list_tta", [])
                    heatmaps = [perceptual_loss(images, h) for h in healed_list]
                    heatmaps += [perceptual_loss(images, h) for h in healed_list_tta]
                    if heatmaps:
                        agg_path = os.path.join(visualizations_dir, f"{base_name}_Heatmap_Aggregations.png")
                        visualize_heatmap_aggregation_comparison(
                            heatmaps,
                            save_path=agg_path,
                            sample_idx=i,
                            title=f"Heatmap aggregation comparison\n{category}/{case_folder}/{filename}",
                            logsumexp_temp=logsumexp_temp,
                        )

                if save_token_mask_figures and i < aggregation_figures_max_samples:
                    token_mask_path = os.path.join(visualizations_dir, f"{base_name}_TokenMask_Comparison.png")
                    visualize_token_masking_comparison(
                        images=images,
                        anomaly_mask=pipeline_results["iteration_history"][0]["mask"],
                        patch_size=stage2.patch_size,
                        save_path=token_mask_path,
                        sample_idx=i,
                        avg_threshold=token_mask_avg_threshold,
                        topk_ratio=token_mask_topk_ratio,
                        title=f"Token mask modes\n{category}/{case_folder}/{filename}",
                    )
    
    return {"slices": results_list}


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Recursive-AutoMask V4 with Z-Score Calibration")
    
    # Model paths
    parser.add_argument("--stage1-ckpt", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/lightningCheckpoints_LUNDPROBE/Modified_stage1-epoch=094-val/loss=0.8587.ckpt")
    parser.add_argument("--stage2-ckpt", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/lightningCheckpoints_LUNDPROBE/Modified_stage2-epoch=093-val/loss=2.9171.ckpt")
    #parser.add_argument("--data-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Anomaly_Inference_Cases/Synth_Calibration_Data_npy")
    #parser.add_argument("--data-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Anomaly_Inference_Cases/Testdataset_with_clinical_npy")
    #parser.add_argument("--data-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Anomaly_Inference_Cases/Cervix_Brachy_Resampled_npy")
    #parser.add_argument("--data-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Anomaly_Inference_Cases/test_ChristianData2Musti_npy") #FLIPPA-upp-o-ner!!!!!
    #parser.add_argument("--data-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Anomaly_Inference_Cases/SpacerResampled_Patienter_npy") #FLIPPA-upp-o-ner!!!!!
    parser.add_argument("--data-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_SpaceOAR_full_FOV_npy") 
    #parser.add_argument("--data-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Anomaly_Inference_Cases/test_LUND_PROBE_extended_npy")
    #parser.add_argument("--output-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_LUND_PROBE_Synth_Calibration_Data")
    #parser.add_argument("--output-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_Testdataset_with_clinical_npy")
    #parser.add_argument("--output-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_test_LUND_PROBE_extended_npy")
    #parser.add_argument("--output-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_LUND_PROBE_Global_Clinical")
    #parser.add_argument("--output-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_LUND_PROBE_ESTROTestdata_CervixBrachy")
    #parser.add_argument("--output-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_LUND_PROBE_Christian_Clinical")
    #parser.add_argument("--output-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_LUND_PROBE_SpacerResampled")
    parser.add_argument("--output-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_Results_SpaceOAR_full_FOV_npy")
    
    # Mode selection
    parser.add_argument("--calibration-mode", action="store_true",
                        help="Run calibration on healthy volunteers (saves μ/σ maps)")
    parser.add_argument("--calibration-map", type=str, default="/home/mluser1/Musti_Anomaly_Detection/Complete_AnomalyDetection_LUND-PROBE/Inference_LUND_PROBE_ZscoreMapCalibration/zscore_calibration.npz", help="Path to calibration .npz file (for inference mode)")
    parser.add_argument("--annotation-csv", type=str, default=None,
                        help="Optional CSV with bounding boxes/labels for overlay")
    parser.add_argument("--no-annotation-boxes", action="store_true",
                        help="Disable annotation box overlay on Final LPIPS panel")
    
    # Z-Score parameters
    parser.add_argument("--z-threshold", type=float, default=2.0,
                        help="Z-score threshold for anomaly detection (default: 2.0 = 2 std devs)")
    parser.add_argument("--z-epsilon", type=float, default=0.01,
                        help="Epsilon for numerical stability in Z-score computation")
    parser.add_argument("--smoothing-kernel", type=int, default=15,
                        help="Kernel size for spatial smoothing (handles registration noise, default: 15)")
    parser.add_argument("--heatmap-aggregation", type=str, default="geomean",
                        choices=["mean", "max", "logsumexp", "geomean"],
                        help="How to aggregate LPIPS heatmaps across healed samples (use 'geomean' to match Smart)")
    parser.add_argument("--logsumexp-temp", type=float, default=1.0,
                        help="Temperature for logsumexp aggregation (higher = softer)")
    
    # Data parameters
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--catagory", "--category-name-contains", dest="catagory", type=str, default=None,
                        help="Substring filter on patient filenames (e.g. RandomSpike). Useful for quick subset inference.")
    parser.add_argument("--json-categories", type=str, nargs="+", default=None,
                        help="Only include these categories in saved JSON (case-insensitive exact match), e.g. --json-categories orig CTVAverage")
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--patient-filter", type=str, nargs="+", default=None,
                        help="Filter by specific patient filename(s). Can be exact filenames or substring patterns.")
    parser.add_argument("--calibration-substring", type=str, default="orig",
                        help="Substring required in filenames during --calibration-mode (default: orig)")
    parser.add_argument("--no-calibration-substring-filter", action="store_true",
                        help="Disable automatic calibration filename substring filtering")
    
    # V4 Recursive parameters
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--inter-iteration-dilation", type=int, default=5)

    # Artifact guards
    parser.add_argument("--blur-threshold", type=float, default=0.002,
                        help="Sharpness (Laplacian variance) threshold; below -> artifact blur")
    
    # Healing parameters
    parser.add_argument("--heal-steps", type=int, default=6)
    parser.add_argument("--heal-temperature", type=float, default=0.3)
    parser.add_argument("--heal-patterns", type=str, default="2,3")
    
    # Mask parameters (fallback)
    parser.add_argument("--mask-threshold", type=float, default=97.0)
    parser.add_argument("--binary-threshold", type=float, default=0.60,
                        help="Threshold on masked score for binary sum map visualization")
    parser.add_argument("--mask-dilation", type=int, default=3)
    parser.add_argument("--token-mask-mode", type=str, default="max",
                        choices=["max", "avg", "topk"],
                        help="Token mask mode: max-pool, avg-pool, or top-k tokens")
    parser.add_argument("--token-mask-avg-threshold", type=float, default=0.5,
                        help="Avg-pool threshold for token mask (mode=avg)")
    parser.add_argument("--token-mask-topk-ratio", type=float, default=0.1,
                        help="Top-k ratio for token mask (mode=topk)")

    parser.add_argument("--token-surprisal-samples", type=int, default=50,
                        help="Number of random masks for token surprisal (pseudo-PLL)")
    parser.add_argument("--token-surprisal-mask-ratio", type=float, default=0.90,
                        help="Mask ratio per sample for token surprisal")
    
    parser.add_argument("--token-surprisal-clamp", type=float, default=8.0,
                        help="Clamp threshold for token surprisal NLL map (values <= threshold set to 0)")
    parser.add_argument("--no-token-surprisal", action="store_true",
                        help="Disable token surprisal scoring (faster)")

    parser.add_argument("--clamp-threshold", type=float, default=0.80,
                        help="Threshold applied to sqrt(First × Last) for clamped sums/visualizations")
    parser.add_argument("--pixle-sum-first-heatmap-thresh", type=float, default=300.0,
                        help="Threshold for coloring the first masked heatmap sum (green below, red above)")

    parser.add_argument("--binary-mask-iteration", type=int, default=0,
                        help="Which iteration's masked score to binarize (0 = first, -1 = last)")
    parser.add_argument("--Heatmap-overlay-viz-clamp", type=float, default=0.30,
                        help="Clamp threshold for heatmap overlay visualization (values below set to 0)")
    
    # Inpainting parameters
    parser.add_argument("--inpaint-steps", type=int, default=12)
    parser.add_argument("--inpaint-temperature", type=float, default=0.3)
    
    # TTA
    parser.add_argument("--use-tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="use_tta")
    parser.add_argument("--flip-upside-down", action="store_true", default=False,
                        help="Flip input slices vertically before processing")
    
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--save-all-visualizations", action="store_true", default=True)
    parser.add_argument("--no-figures-only-json", action="store_true",
                        help="Skip all figure generation and only emit the JSON output")
    parser.add_argument("--no-aggregation-figures", action="store_true",
                        help="Disable heatmap aggregation comparison figures")
    parser.add_argument("--no-token-mask-figures", action="store_true",
                        help="Disable token mask comparison figures")
    parser.add_argument("--aggregation-figures-max-samples", type=int, default=3,
                        help="Max samples per batch for aggregation/token mask comparison figures")
    
    args = parser.parse_args()

    figures_enabled = not args.no_figures_only_json
    save_visualizations = args.save_all_visualizations and figures_enabled
    save_aggregation_figures = not args.no_aggregation_figures
    save_token_mask_figures = not args.no_token_mask_figures
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    stage1, stage2 = load_models(args.stage1_ckpt, args.stage2_ckpt, device)
    
    print("\nInitializing LPIPS...")
    perceptual_loss = PerceptualLoss(device=device)
    
    from External_dataset import create_dataloader
    
    combined_patient_filter = list(args.patient_filter) if args.patient_filter else []
    if args.catagory:
        combined_patient_filter.append(args.catagory)

    json_category_tokens = [
        c.strip() for c in (args.json_categories or []) if str(c).strip()
    ]

    if json_category_tokens:
        # Ensure category-selected runs are filtered already at dataset level for speed.
        # We apply both metadata category filtering and filename substring filtering.
        # This handles datasets where category metadata may be missing/inconsistent.
        for token in json_category_tokens:
            if token not in combined_patient_filter:
                combined_patient_filter.append(token)
        print(f"JSON categories will also filter dataset loading: {json_category_tokens}")

    if args.calibration_mode and not args.no_calibration_substring_filter:
        calib_sub = (args.calibration_substring or "").strip()
        if calib_sub:
            combined_patient_filter.append(calib_sub)
            print(f"Calibration mode: enforcing filename substring filter = '{calib_sub}'")

    effective_category_filter = [args.category] if args.category else None
    if json_category_tokens:
        if effective_category_filter is None:
            effective_category_filter = list(json_category_tokens)
        else:
            for token in json_category_tokens:
                if token not in effective_category_filter:
                    effective_category_filter.append(token)

    dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        category_filter=effective_category_filter,
        case_filter=[args.case] if args.case else None,
        patient_filter=combined_patient_filter if combined_patient_filter else None,
    )
    
    print(f"\nDataset: {len(dataloader.dataset)} slices")
    if combined_patient_filter:
        print(f"Patient filter applied: {combined_patient_filter}")
    
    heal_patterns = [int(p) for p in args.heal_patterns.split(",")]
    
    # =================================================================
    # CALIBRATION MODE
    # =================================================================
    if args.calibration_mode:
        print("\n" + "="*70)
        print("CALIBRATION MODE - Processing Healthy Volunteers")
        print("="*70)
        
        os.makedirs(args.output_dir, exist_ok=True)
        calibration_path = os.path.join(args.output_dir, "zscore_calibration.npz")
        
        calibration = run_calibration(
            stage1, stage2, perceptual_loss,
            dataloader, calibration_path,
            device=device,
            heal_steps=args.heal_steps,
            heal_temperature=args.heal_temperature,
            heal_patterns=heal_patterns,
            use_tta=args.use_tta,
            smoothing_kernel=args.smoothing_kernel,
            heatmap_aggregation=args.heatmap_aggregation,
            logsumexp_temp=args.logsumexp_temp,
            save_aggregation_figures=save_aggregation_figures,
            aggregation_figures_max_samples=args.aggregation_figures_max_samples,
            debug=True,
            flip_upside_down=args.flip_upside_down,
        )
        
        print(f"\nCalibration complete! Saved to: {calibration_path}")
        print("\nTo use in inference, run with:")
        print(f"  --calibration-map {calibration_path}")
        
        return
    
    # =================================================================
    # INFERENCE MODE
    # =================================================================
    print("\n" + "="*70)
    print("INFERENCE MODE")
    print("="*70)
    
    # Load calibration if provided
    calibration = None
    if args.calibration_map:
        calibration = ZScoreCalibration(args.calibration_map)
        print(f"\nZ-Score calibration loaded!")
        print(f"  Z-threshold: {args.z_threshold}")
    else:
        print("\nNo calibration map provided - using percentile thresholding")
    
    print(f"\n{'='*60}")
    print(f"RECURSIVE AUTOMASK V4 - Z-SCORE {'ENABLED' if calibration else 'DISABLED'}")
    print(f"{'='*60}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Z-threshold: {args.z_threshold}")
    print(f"{'='*60}\n")
    
    results = run_inference_v4_zscore(
        stage1, stage2, perceptual_loss,
        dataloader, args.output_dir,
        calibration=calibration,
        z_threshold=args.z_threshold,
        smoothing_kernel=args.smoothing_kernel,
        heatmap_aggregation=args.heatmap_aggregation,
        logsumexp_temp=args.logsumexp_temp,
        clamp_threshold=args.clamp_threshold,
        first_heatmap_sum_thresh=args.pixle_sum_first_heatmap_thresh,
        binary_mask_threshold=args.binary_threshold,
        binary_mask_iteration=args.binary_mask_iteration,
        heatmap_overlay_viz_clamp=args.Heatmap_overlay_viz_clamp,
        device=device,
        num_iterations=args.num_iterations,
        inter_iteration_dilation=args.inter_iteration_dilation,
        heal_steps=args.heal_steps,
        heal_temperature=args.heal_temperature,
        heal_patterns=heal_patterns,
        mask_threshold_percentile=args.mask_threshold,
        mask_dilation=args.mask_dilation,
        blur_threshold=args.blur_threshold,
        use_tta=args.use_tta,
        inpaint_steps=args.inpaint_steps,
        inpaint_temperature=args.inpaint_temperature,
        token_mask_mode=args.token_mask_mode,
        token_mask_avg_threshold=args.token_mask_avg_threshold,
        token_mask_topk_ratio=args.token_mask_topk_ratio,
        token_surprisal_samples=args.token_surprisal_samples,
        token_surprisal_mask_ratio=args.token_surprisal_mask_ratio,
        token_surprisal_clamp=args.token_surprisal_clamp,
        compute_token_surprisal=not args.no_token_surprisal,
        annotation_csv=args.annotation_csv,
        overlay_annotation_boxes=not args.no_annotation_boxes,
        save_all_visualizations=save_visualizations,
        enable_visualizations=figures_enabled,
        save_aggregation_figures=save_aggregation_figures,
        aggregation_figures_max_samples=args.aggregation_figures_max_samples,
        save_token_mask_figures=save_token_mask_figures,
        debug_first_batch=True,
        flip_upside_down=args.flip_upside_down,
    )
    
    # Summary
    slices = results["slices"]
    slices_for_json = slices
    if args.json_categories:
        selected_json_categories = [
            c.strip().lower() for c in args.json_categories if str(c).strip()
        ]

        def _matches_json_category(row: dict, selected: List[str]) -> bool:
            category = str(row.get("category", "") or "").strip().lower()
            case_folder = str(row.get("case_folder", "") or "").strip().lower()
            filename = str(row.get("filename", "") or "").strip().lower()
            path = str(row.get("path", "") or "").strip().lower()

            for token in selected:
                if not token:
                    continue
                if (
                    token == category
                    or token in category
                    or token in case_folder
                    or token in filename
                    or token in path
                ):
                    return True
            return False

        slices_for_json = [r for r in slices if _matches_json_category(r, selected_json_categories)]
        matched_categories = sorted({str(r.get("category", "Unknown")) for r in slices_for_json})
        print(
            f"JSON category filter enabled: {selected_json_categories} | "
            f"kept {len(slices_for_json)}/{len(slices)} slices | "
            f"matched categories: {matched_categories}"
        )

    print("\n" + "="*60)
    print("V4 SUMMARY - Z-SCORE CALIBRATED")
    print("="*60)
    print(f"Slices processed: {len(slices)}")
    print(f"Slices included in JSON: {len(slices_for_json)}")
    print(f"Z-Score used: {slices[0]['used_zscore'] if slices else 'N/A'}")
    
    initial_coverages = [r["initial_mask_coverage"] for r in slices_for_json]
    final_coverages = [r["final_mask_coverage"] for r in slices_for_json]
    
    if slices_for_json:
        print(f"\nInitial mask coverage: {np.mean(initial_coverages)*100:.2f}% ± {np.std(initial_coverages)*100:.2f}%")
        print(f"Final mask coverage:   {np.mean(final_coverages)*100:.2f}% ± {np.std(final_coverages)*100:.2f}%")
    else:
        print("\nNo slices matched --json-categories. JSON will contain empty results.")
    
    def _patient_label_from_result(r: dict) -> str:
        """Derive a stable patient label even when case_folder is unknown.

        If case_folder is non-empty and not 'unknown', use it.
        Otherwise, fall back to filename stem before '_slice_'.
        """
        case_folder = str(r.get("case_folder", "") or "").strip()
        if case_folder and case_folder.lower() != "unknown":
            return case_folder
        fname = str(r.get("filename", "") or "").strip()
        stem = os.path.splitext(fname)[0]
        if "_slice_" in stem:
            return stem.split("_slice_", 1)[0]
        return stem or case_folder or "unknown"

    # Aggregate per-patient clamped pixel sums
    patient_sums = defaultdict(lambda: {
        "total_clamped_sum": 0.0,
        "total_first_heatmap_sum": 0.0,
        "total_lpips_input_recon_sum_mask": 0.0,
        "total_pixels_above_thresh": 0,
        "total_token_surprisal_hot_px": 0,
        "num_slices": 0,
        "category": "",
        "slice_sums": []
    })
    for r in slices_for_json:
        patient_label = _patient_label_from_result(r)
        patient_key = f"{r['category']}/{patient_label}"
        patient_sums[patient_key]["total_clamped_sum"] += r.get("clamped_pixel_sum", 0.0)
        patient_sums[patient_key]["total_first_heatmap_sum"] += r.get("clamped_pixel_sum_FirstHeatmap", 0.0)
        patient_sums[patient_key]["total_lpips_input_recon_sum_mask"] += r.get("lpips_input_recon_sum_mask", 0.0) or 0.0
        patient_sums[patient_key]["total_pixels_above_thresh"] += r.get("num_pixels_above_clamp_thresh", 0)
        patient_sums[patient_key]["total_token_surprisal_hot_px"] += int(r.get("token_surprisal_hot_px") or 0)
        patient_sums[patient_key]["num_slices"] += 1
        patient_sums[patient_key]["category"] = r["category"]
        patient_sums[patient_key]["case_folder"] = r.get("case_folder", "")
        patient_sums[patient_key]["patient_label"] = patient_label
        patient_sums[patient_key]["slice_sums"].append({
            "filename": r["filename"],
            "clamped_sum": r.get("clamped_pixel_sum", 0.0),
            "clamped_sum_first_heatmap": r.get("clamped_pixel_sum_FirstHeatmap", 0.0),
            "lpips_input_recon_sum_mask": r.get("lpips_input_recon_sum_mask", 0.0),
            "pixels_above_thresh": r.get("num_pixels_above_clamp_thresh", 0),
            "token_surprisal_hot_px": r.get("token_surprisal_hot_px")
        })
    
    # Convert to list and sort by total clamped sum (descending)
    patient_summary = []
    for patient_key, data in patient_sums.items():
        patient_summary.append({
            "patient_id": patient_key,
            "category": data["category"],
            "case_folder": data["case_folder"],
            "total_clamped_pixel_sum": data["total_clamped_sum"],
            "total_clamped_pixel_sum_first_heatmap": data["total_first_heatmap_sum"],
            "total_lpips_input_recon_sum_mask": data["total_lpips_input_recon_sum_mask"],
            "total_pixels_above_thresh": data["total_pixels_above_thresh"],
            "total_token_surprisal_hot_px": data["total_token_surprisal_hot_px"],
            "num_slices": data["num_slices"],
            "mean_clamped_sum_per_slice": data["total_clamped_sum"] / data["num_slices"] if data["num_slices"] > 0 else 0,
            "mean_clamped_sum_first_heatmap_per_slice": data["total_first_heatmap_sum"] / data["num_slices"] if data["num_slices"] > 0 else 0,
            "mean_lpips_input_recon_sum_mask_per_slice": data["total_lpips_input_recon_sum_mask"] / data["num_slices"] if data["num_slices"] > 0 else 0,
            "mean_token_surprisal_hot_px_per_slice": data["total_token_surprisal_hot_px"] / data["num_slices"] if data["num_slices"] > 0 else 0,
            "slice_details": data["slice_sums"]
        })
    
    patient_summary = sorted(patient_summary, key=lambda x: x["total_clamped_pixel_sum"], reverse=True)

    # Print patient-level summary
    print(f"\n{'='*60}")
    print("PER-PATIENT CLAMPED PIXEL SUM (sorted by total sum)")
    print(f"{'='*60}")
    print(f"Clamp threshold: {args.clamp_threshold}")
    
    # Save results
    with open(os.path.join(args.output_dir, "results_v4_zscore.json"), 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "calibration_mode": False,
                "calibration_map": args.calibration_map,
                "json_categories": args.json_categories,
                "z_threshold": args.z_threshold,
                "smoothing_kernel": args.smoothing_kernel,
                "heatmap_aggregation": args.heatmap_aggregation,
                "logsumexp_temp": args.logsumexp_temp,
                "num_iterations": args.num_iterations,
                "mask_threshold": args.mask_threshold,
                "catagory": args.catagory,
                "token_mask_mode": args.token_mask_mode,
                "token_mask_avg_threshold": args.token_mask_avg_threshold,
                "token_mask_topk_ratio": args.token_mask_topk_ratio,
                "clamp_threshold": args.clamp_threshold,
                "pixle_sum_first_heatmap_thresh": args.pixle_sum_first_heatmap_thresh,
            },
            "summary": {
                "num_slices": len(slices_for_json),
                "mean_initial_coverage": float(np.mean(initial_coverages)) if slices_for_json else 0.0,
                "mean_final_coverage": float(np.mean(final_coverages)) if slices_for_json else 0.0,
                "mean_lpips_input_recon_sum_mask": float(np.mean([r.get("lpips_input_recon_sum_mask", 0.0) or 0.0 for r in slices_for_json])) if slices_for_json else 0.0,
            },
            "patient_summary": patient_summary,
            "results": slices_for_json
        }, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
