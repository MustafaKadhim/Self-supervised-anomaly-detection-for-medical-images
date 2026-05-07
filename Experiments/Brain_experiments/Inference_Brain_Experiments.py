"""
Recursive-AutoMask V4 with Population-Based Z-Score Normalization
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

Author: Recursive-AutoMask V4 with Z-Score Calibration
"""

from __future__ import annotations

import argparse
import ast
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
from torch.utils.data import DataLoader, Dataset
from scipy import ndimage
from tqdm import tqdm
from Inference_heatmaps_ideas_generator import generate_heatmap_ideas_figure

# Try to import LPIPS for perceptual loss
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# Heatmap colormap for visualizations
HEATMAP_CMAP = "hot"

# AYNU-purpose annotation constants kept before CORE for default-argument compatibility.


# =============================================================================
# Annotation Box Helpers
# =============================================================================

ANNOTATION_BASE_SIZE = 320
TP_INSIDE_RATIO_THRESHOLD = 0.10
ANNOTATION_PREPROCESS_LEGACY = "legacy"
ANNOTATION_PREPROCESS_RENDER_FASTMRI = "render_fastmri"
ANNOTATION_PREPROCESS_MASK_PIPELINE = "mask_pipeline"

# =============================================================================
# =============================================================================
# CORE — AUROC pipeline
# -----------------------------------------------------------------------------
# Everything in this section is on the path that produces the per-slice
# `Binary_Sum_Heatmap` field, which is the ONLY quantity consumed by the
# patient-level ROC / AUROC computation in the Plot_Bars script.
#
# Trace: model forward → ensemble_heal → LPIPS heatmap → binary mask fusion
#        (masked_score ∪ token_surprisal ∪ lpips_backflow ∪ edge erosion)
#        → Binary_Sum_Heatmap → patient aggregation → ROC / AUROC.
# =============================================================================
# =============================================================================
# -----------------------------------------------------------------------------
# Quantizer codebook access  (CORE)
# -----------------------------------------------------------------------------


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

# -----------------------------------------------------------------------------
# Perceptual loss (LPIPS) — heatmap engine  (CORE)
# -----------------------------------------------------------------------------


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

# -----------------------------------------------------------------------------
# Token surprisal (pseudo-PLL)  (CORE)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Heatmap aggregation  (CORE)
# -----------------------------------------------------------------------------


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

# -----------------------------------------------------------------------------
# Token mask building (for targeted inpainting)  (CORE)
# -----------------------------------------------------------------------------


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

# -----------------------------------------------------------------------------
# LPIPS-backflow union step  (CORE)
# -----------------------------------------------------------------------------


def build_lpips_backflow_mask(
    lpips_map: np.ndarray,
    selector: Union[float, Tuple[float, float], List[float]],
) -> tuple[np.ndarray, float]:
    percentile = 0.0
    fixed_threshold = 0.0

    if isinstance(selector, (tuple, list)):
        if len(selector) != 2:
            raise ValueError("LPIPS backflow selector must have 2 values: (percentile, fixed_threshold)")
        percentile = float(selector[0])
        fixed_threshold = float(selector[1])
    else:
        fixed_threshold = float(selector)

    percentile = float(np.clip(percentile, 0.0, 100.0))
    fixed_threshold = float(max(0.0, fixed_threshold))

    if percentile > 0.0 and fixed_threshold > 0.0:
        raise ValueError(
            "Invalid LPIPS backflow selector: both percentile and fixed threshold are set. "
            "Use either (percentile, 0) or (0, fixed_threshold)."
        )

    if percentile > 0.0:
        cutoff = float(np.percentile(lpips_map, percentile))
    else:
        cutoff = fixed_threshold

    return lpips_map > cutoff, cutoff



def parse_lpips_backflow_selector_arg(
    value: Union[str, float, int, Tuple[float, float], List[float]],
) -> Union[float, Tuple[float, float]]:
    """Parse LPIPS backflow selector.

    Supports:
      - Fixed threshold: 0.33 or "0.33"
      - Tuple mode: "(95, 0)" (percentile mode) or "(0, 0.33)" (fixed mode)
    """
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Backflow selector tuple must have exactly two values: (percentile, fixed_threshold)")
        p = float(value[0])
        t = float(value[1])
        if p < 0 or p > 100:
            raise ValueError("Backflow percentile must be in [0, 100]")
        if t < 0:
            raise ValueError("Backflow fixed threshold must be >= 0")
        if p > 0 and t > 0:
            raise ValueError("Use either (percentile, 0) or (0, fixed_threshold), not both positive")
        return (p, t)

    raw = str(value).strip()
    if not raw:
        raise ValueError("LPIPS backflow selector cannot be empty")

    try:
        return float(raw)
    except ValueError:
        pass

    try:
        literal = ast.literal_eval(raw)
        if isinstance(literal, (tuple, list)) and len(literal) == 2:
            p = float(literal[0])
            t = float(literal[1])
            if p < 0 or p > 100:
                raise ValueError("Backflow percentile must be in [0, 100]")
            if t < 0:
                raise ValueError("Backflow fixed threshold must be >= 0")
            if p > 0 and t > 0:
                raise ValueError("Use either (percentile, 0) or (0, fixed_threshold), not both positive")
            return (p, t)
    except (ValueError, SyntaxError):
        pass

    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) == 2:
            p = float(parts[0])
            t = float(parts[1])
            if p < 0 or p > 100:
                raise ValueError("Backflow percentile must be in [0, 100]")
            if t < 0:
                raise ValueError("Backflow fixed threshold must be >= 0")
            if p > 0 and t > 0:
                raise ValueError("Use either (percentile, 0) or (0, fixed_threshold), not both positive")
            return (p, t)

    raise ValueError(
        f"Invalid LPIPS backflow selector '{value}'. Use a fixed float (e.g. 0.33), "
        "or tuple mode (percentile, fixed_threshold), e.g. '(95, 0)' or '(0, 0.33)'."
    )

# -----------------------------------------------------------------------------
# Edge-aware binary mask cleanup  (CORE)
# -----------------------------------------------------------------------------


def apply_edge_to_center_erosion(
    binary_mask: np.ndarray,
    max_edge_erosion_iters: int = 0,
    center_protect_radius_ratio: float = 0.25,
    erosion_kernel_size: int = 3,
) -> np.ndarray:
    """Apply stronger erosion near edges and weaker erosion near image center.

    Args:
        binary_mask: [H, W] bool/0-1 mask
        max_edge_erosion_iters: max erosion iterations at image borders
        center_protect_radius_ratio: central radius (0-1) where erosion is minimal
        erosion_kernel_size: odd kernel size for binary erosion
    """
    max_edge_erosion_iters = int(max(0, max_edge_erosion_iters))
    if max_edge_erosion_iters == 0:
        return binary_mask.astype(bool)

    mask_bool = binary_mask.astype(bool)
    if not mask_bool.any():
        return mask_bool

    h, w = mask_bool.shape
    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0

    dy = (yy - cy) / (max(cy, 1.0))
    dx = (xx - cx) / (max(cx, 1.0))
    dist_norm = np.sqrt(dx**2 + dy**2)
    dist_norm = dist_norm / (dist_norm.max() + 1e-8)

    center_protect_radius_ratio = float(np.clip(center_protect_radius_ratio, 0.0, 0.95))
    if center_protect_radius_ratio > 0:
        dist_scaled = np.clip(
            (dist_norm - center_protect_radius_ratio) / (1.0 - center_protect_radius_ratio + 1e-8),
            0.0,
            1.0,
        )
    else:
        dist_scaled = dist_norm

    required_iters = np.floor(dist_scaled * max_edge_erosion_iters + 1e-8).astype(np.int32)

    erosion_kernel_size = int(max(1, erosion_kernel_size))
    if erosion_kernel_size % 2 == 0:
        erosion_kernel_size += 1
    structure = np.ones((erosion_kernel_size, erosion_kernel_size), dtype=bool)

    erosion_levels = [mask_bool]
    eroded = mask_bool
    for _ in range(max_edge_erosion_iters):
        eroded = ndimage.binary_erosion(eroded, structure=structure, border_value=0)
        erosion_levels.append(eroded)

    output = np.zeros_like(mask_bool)
    for iter_idx, mask_level in enumerate(erosion_levels):
        selector = required_iters == iter_idx
        output[selector] = mask_level[selector]

    return output

# -----------------------------------------------------------------------------
# Z-score threshold parsing / application  (CORE)
# -----------------------------------------------------------------------------


def parse_z_threshold_arg(value: Union[str, float, int, Tuple[float, float], List[float]]) -> Union[float, Tuple[float, float]]:
    """Parse z-threshold from CLI.

    Supports:
      - Single-sided: 1.25
      - Two-sided: "(-4.5, 1.2)" or "-4.5,1.2"
    """
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Two-sided z-threshold must contain exactly two values: (low, high)")
        low = float(value[0])
        high = float(value[1])
        if low > high:
            low, high = high, low
        return (low, high)

    raw = str(value).strip()
    if not raw:
        raise ValueError("--z-threshold cannot be empty")

    # Try simple float first
    try:
        return float(raw)
    except ValueError:
        pass

    # Try Python literal syntax: "(-4.5, 1.2)" or "[-4.5, 1.2]"
    try:
        literal = ast.literal_eval(raw)
        if isinstance(literal, (tuple, list)) and len(literal) == 2:
            low = float(literal[0])
            high = float(literal[1])
            if low > high:
                low, high = high, low
            return (low, high)
    except (ValueError, SyntaxError):
        pass

    # Try comma-separated fallback: "-4.5,1.2"
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) == 2:
            low = float(parts[0])
            high = float(parts[1])
            if low > high:
                low, high = high, low
            return (low, high)

    raise ValueError(
        f"Invalid --z-threshold '{value}'. Use a single float (e.g. 1.25) "
        f"or a tuple/range (e.g. '(-4.5, 1.2)')."
    )



def format_z_threshold(z_threshold: Union[float, Tuple[float, float]]) -> str:
    if isinstance(z_threshold, (tuple, list)) and len(z_threshold) == 2:
        low, high = float(z_threshold[0]), float(z_threshold[1])
        return f"({low:g}, {high:g})"
    return f"{float(z_threshold):g}"



def threshold_zscore_map(
    zscore_map: torch.Tensor,
    z_threshold: Union[float, Tuple[float, float]],
) -> torch.Tensor:
    """Apply one-sided or two-sided thresholding to a Z-score map."""
    if isinstance(z_threshold, (tuple, list)) and len(z_threshold) == 2:
        low, high = float(z_threshold[0]), float(z_threshold[1])
        return ((zscore_map < low) | (zscore_map > high)).float()
    return (zscore_map > float(z_threshold)).float()



def z_threshold_exceedance_ratio(
    zscore_map: torch.Tensor,
    z_threshold: Union[float, Tuple[float, float]],
) -> float:
    return float(threshold_zscore_map(zscore_map, z_threshold).mean().item())

# -----------------------------------------------------------------------------
# Heatmap → binary mask  (CORE)
# -----------------------------------------------------------------------------


# =============================================================================
# Mask Generation with Z-Score Support
# =============================================================================

def heatmap_to_binary_mask_zscore(
    heatmap: torch.Tensor,
    calibration: ZScoreCalibration,
    z_threshold: Union[float, Tuple[float, float]] = 2.0,
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
    binary_mask = threshold_zscore_map(zscore_map, z_threshold)
    
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



def build_final_lpips_binary_token_eval_mask(
    final_lpips: np.ndarray,
    binary_token_mask: np.ndarray,
    include_token_surprisal: bool,
    token_surprisal_map: Optional[np.ndarray] = None,
    lpips_percentile: float = 60.0,
    display_floor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact fused Final LPIPS + Binary+Token map used for evaluation.

    Returns:
        fused_display_map: float map with NaN in background (same display logic as figure)
        fused_binary_mask: bool mask of highlighted pixels in fused_display_map
    """
    p_cut = np.percentile(final_lpips, lpips_percentile)
    final_lpips_clamped = np.where(final_lpips >= p_cut, final_lpips, np.nan)
    fused_display_map = final_lpips_clamped.copy()

    if include_token_surprisal and token_surprisal_map is not None:
        final_lpips_dense = np.nan_to_num(fused_display_map, nan=0.0)
        lpips_norm = np.clip((final_lpips_dense - display_floor) / (1.0 - display_floor), 0.0, 1.0)

        binary_soft = ndimage.gaussian_filter(binary_token_mask.astype(np.float32), sigma=1.2)
        if binary_soft.max() > 0:
            binary_soft = binary_soft / (binary_soft.max() + 1e-8)

        fused_norm = 0.55 * lpips_norm + 0.45 * np.maximum(lpips_norm, binary_soft)
        fused_norm = np.clip(fused_norm, 0.0, 1.0)
        fused_display_map = np.where(fused_norm > 0, display_floor + (1.0 - display_floor) * fused_norm, np.nan)

    fused_display_map = np.where(fused_display_map > display_floor, fused_display_map, np.nan)
    fused_binary_mask = np.isfinite(fused_display_map)
    return fused_display_map, fused_binary_mask

# -----------------------------------------------------------------------------
# Z-score calibration  (CORE)
# -----------------------------------------------------------------------------


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
        
        # Reconstruction is the LPIPS reference used for calibration/inference.
        reconstruction = stage1(images)["recon"]

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
            reconstruction_flipped = torch.flip(reconstruction, dims=[-1])
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
        # Compute LPIPS(Reconstruction, Healed)
        # ---------------------------------------------------------
        if heatmap_aggregation.lower().strip() == "geomean":
            # Smart-equivalent fusion:
            #   heatmap = sqrt(LPIPS(Rec, healed_A) * LPIPS(Rec, healed_B))
            # where healed_A and healed_B are ensemble outputs for native/TTA branches.
            lpips_map_A = perceptual_loss(reconstruction, healed_A)
            if use_tta:
                lpips_map_flip = perceptual_loss(reconstruction_flipped, healed_flipped)
                lpips_map_B = torch.flip(lpips_map_flip, dims=[-1])
            else:
                lpips_map_B = lpips_map_A
            heatmap = torch.sqrt(lpips_map_A * lpips_map_B + 1e-8)
            heatmaps = [lpips_map_A, lpips_map_B]
        else:
            heatmaps = []
            for h in heal_info.get("healed_images_list", []):
                heatmaps.append(perceptual_loss(reconstruction, h))
            if use_tta and healed_images_list_tta:
                for h in healed_images_list_tta:
                    heatmaps.append(perceptual_loss(reconstruction, h))

            if not heatmaps:
                heatmaps = [perceptual_loss(reconstruction, healed_A)]

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

    if total_samples == 0 or len(all_heatmaps) == 0:
        raise RuntimeError(
            "Calibration received 0 samples. "
            "Check --data-dir and active filters (category/case/patient/calibration-substring). "
            "For FastMRI, ensure the folder contains slice files (.npz) and that filters are not over-restrictive."
        )
    
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

# -----------------------------------------------------------------------------
# Healing (mask-pattern ensemble)  (CORE)
# -----------------------------------------------------------------------------


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

# -----------------------------------------------------------------------------
# Targeted inpainting  (CORE)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Sharpness signals (artifact guard inputs)  (CORE)
# -----------------------------------------------------------------------------


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

# -----------------------------------------------------------------------------
# Recursive AutoMask V4 — main pipeline  (CORE)
# -----------------------------------------------------------------------------
def recursive_automask_v4_zscore(
    stage1, stage2,
    perceptual_loss: PerceptualLoss,
    images: torch.Tensor,
    slice_pos: Optional[torch.Tensor] = None,
    # Z-Score calibration
    calibration: Optional[ZScoreCalibration] = None,
    z_threshold: Union[float, Tuple[float, float]] = 2.0,
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
    - Thresholds on Z > z_threshold (or outside [z_low, z_high] for two-sided mode)
      - Eliminates "consistent" false positives (high μ, low σ regions)
    - Iteration 1+: Use percentile thresholding on LPIPS(Recon, Inpainted)
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
            print(f"Z-threshold: {format_z_threshold(z_threshold)}")
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
        reconstruction_flipped = torch.flip(reconstruction, dims=[-1])
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
                lpips_map_A = perceptual_loss(reconstruction, healed_A)
                if use_tta:
                    lpips_map_flip = perceptual_loss(reconstruction_flipped, healed_flipped)
                    lpips_map_B = torch.flip(lpips_map_flip, dims=[-1])
                else:
                    lpips_map_B = lpips_map_A
                current_heatmap = torch.sqrt(lpips_map_A * lpips_map_B + 1e-8)
                heatmap_source = "LPIPS(Rec, Healed) [geomean-smart]"
            else:
                heatmaps = []
                for h in heal_info.get("healed_images_list", []):
                    heatmaps.append(perceptual_loss(reconstruction, h))
                if use_tta and healed_images_list_tta:
                    for h in healed_images_list_tta:
                        heatmaps.append(perceptual_loss(reconstruction, h))

                if not heatmaps:
                    heatmaps = [perceptual_loss(reconstruction, healed_A)]

                current_heatmap = aggregate_heatmaps(
                    heatmaps,
                    method=heatmap_aggregation,
                    logsumexp_temp=logsumexp_temp,
                )
                heatmap_source = f"LPIPS(Rec, Healed) [{heatmap_aggregation}]"
            
        else:
            lpips_map_inpainted = perceptual_loss(reconstruction, current_inpainted)
            
            if use_tta:
                inpainted_flipped = torch.flip(current_inpainted, dims=[-1])
                lpips_map_flip = perceptual_loss(reconstruction_flipped, inpainted_flipped)
                lpips_map_B = torch.flip(lpips_map_flip, dims=[-1])
                current_heatmap = torch.sqrt(lpips_map_inpainted * lpips_map_B + 1e-8)
            else:
                current_heatmap = lpips_map_inpainted
            
            heatmap_source = f"LPIPS(Rec, Inpainted_{iter_idx-1})"
        
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
                if isinstance(z_threshold, (tuple, list)) and len(z_threshold) == 2:
                    print(f"  Z-SCORE thresholding (z < {z_threshold[0]} OR z > {z_threshold[1]})")
                else:
                    print(f"  Z-SCORE thresholding (z > {z_threshold})")
                print(f"  Z-score range: [{zscore_map.min():.2f}, {zscore_map.max():.2f}]")
                print(f"  Pixels flagged by Z-threshold: {z_threshold_exceedance_ratio(zscore_map, z_threshold)*100:.2f}%")
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
        
        # Keep LPIPS(In, Inp) for full-analysis visualization only.
        lpips_input_inpainted = perceptual_loss(images, current_inpainted)
        lpips_recon_inpainted = perceptual_loss(reconstruction, current_inpainted)
        
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
            "lpips_recon_inpainted": lpips_recon_inpainted.clone(),
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
        "lpips_recon_inpainted": final_iteration["lpips_recon_inpainted"],
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

# -----------------------------------------------------------------------------
# Model and dataset I/O  (CORE)
# -----------------------------------------------------------------------------


# =============================================================================
# Model Loading
# =============================================================================

def load_models(stage1_ckpt: str, stage2_ckpt: str, device: str = "cuda"):
    """Load models."""
    from Final_Clean_to_Github_Brain.Model_Stage1 import Stage1RVQVAE
    from Final_Clean_to_Github_Brain.Model_Stage_2 import FactorizedMaskGIT
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



def _build_fastrmri_metadata(path: str, category_name: str = "FastMRI") -> dict:
    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]
    return {
        "category": category_name,
        "case_folder": Path(path).parent.name if Path(path).parent.name else (stem.split("_slice_", 1)[0] if "_slice_" in stem else "Unknown"),
    }



class FastMRINpzDataset(Dataset):
    def __init__(self, files: List[str], category_name: str = "FastMRI"):
        self.files = files
        self.category_name = category_name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path) as data:
            if "arr" in data:
                arr = data["arr"].astype(np.float32)
            elif len(data.files) > 0:
                arr = data[data.files[0]].astype(np.float32)
            else:
                raise RuntimeError(f"No arrays found in npz file: {path}")

        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[-1] in (1, 3):
            arr = np.moveaxis(arr, -1, 0)
        elif arr.ndim != 3:
            raise RuntimeError(f"Unsupported array shape {arr.shape} in file: {path}")

        return {
            "image": torch.from_numpy(arr),
            "path": path,
            "metadata": _build_fastrmri_metadata(path, category_name=self.category_name),
        }



def create_inference_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    category_filter: Optional[List[str]] = None,
    case_filter: Optional[List[str]] = None,
    patient_filter: Optional[List[str]] = None,
    category_name: str = "FastMRI",
) -> DataLoader:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    npz_files = sorted(Path(data_dir).rglob("*.npz"))
    if npz_files:
        files = [str(p) for p in npz_files]

        def _match_filters(path: str) -> bool:
            name = os.path.basename(path).lower()
            category_ok = True
            case_ok = True
            patient_ok = True

            if category_filter:
                category_ok = any(str(token).lower() in name for token in category_filter)
            if case_filter:
                case_ok = any(str(token).lower() in name for token in case_filter)
            if patient_filter:
                patient_ok = any(str(token).lower() in name for token in patient_filter)

            return category_ok and case_ok and patient_ok

        files = [f for f in files if _match_filters(f)]
        print(f"FastMRINpzDataset: Loaded {len(files)} slices from {data_dir}")

        dataset = FastMRINpzDataset(files, category_name=category_name)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    npy_files = sorted(Path(data_dir).rglob("*.npy"))
    if npy_files:
        class FastMRINpyDataset(Dataset):
            def __init__(self, files: List[str], category_name_local: str = "FastMRI"):
                self.files = files
                self.category_name_local = category_name_local

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                path = self.files[idx]
                arr = np.load(path).astype(np.float32)

                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                elif arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[-1] in (1, 3):
                    arr = np.moveaxis(arr, -1, 0)
                elif arr.ndim != 3:
                    raise RuntimeError(f"Unsupported array shape {arr.shape} in file: {path}")

                return {
                    "image": torch.from_numpy(arr),
                    "path": path,
                    "metadata": _build_fastrmri_metadata(path, category_name=self.category_name_local),
                }

        files = [str(p) for p in npy_files]

        def _match_filters(path: str) -> bool:
            name = os.path.basename(path).lower()
            category_ok = True
            case_ok = True
            patient_ok = True

            if category_filter:
                category_ok = any(str(token).lower() in name for token in category_filter)
            if case_filter:
                case_ok = any(str(token).lower() in name for token in case_filter)
            if patient_filter:
                patient_ok = any(str(token).lower() in name for token in patient_filter)

            return category_ok and case_ok and patient_ok

        files = [f for f in files if _match_filters(f)]
        print(f"FastMRINpyDataset: Loaded {len(files)} slices from {data_dir}")

        dataset = FastMRINpyDataset(files, category_name_local=category_name)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    raise RuntimeError(
        f"No slice files found in {data_dir}. Expected .npz (FastMRI) or .npy (external format)."
    )



def find_anomaly_label_dirs(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Anomaly root directory not found: {root_dir}")

    label_dirs: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        has_npz = any(child.rglob("*.npz"))
        has_npy = any(child.rglob("*.npy"))
        if has_npz or has_npy:
            label_dirs.append(child)

    if not label_dirs:
        raise RuntimeError(f"No anomaly label folders with slice files found under: {root_dir}")

    return label_dirs

# -----------------------------------------------------------------------------
# Inference orchestration + JSON serialization  (CORE)
# -----------------------------------------------------------------------------


# =============================================================================
# Main Inference
# =============================================================================

def run_inference_v4_zscore(
    stage1, stage2,
    perceptual_loss: PerceptualLoss,
    dataloader,
    output_dir: str,
    calibration: Optional[ZScoreCalibration] = None,
    z_threshold: Union[float, Tuple[float, float]] = 2.0,
    smoothing_kernel: int = 15,
    heatmap_aggregation: str = "mean",
    logsumexp_temp: float = 1.0,
    clamp_threshold: float = 0.60,
    first_heatmap_sum_thresh: float = 300.0,
    binary_mask_threshold: float = 0.10,
    binary_mask_iteration: int = 0,
    binary_include_token_surprisal: bool = False,
    binary_token_surprisal_threshold: float = 0.0,
    lpips_rec_inp_threshold_back_to_binary_token_map: Union[float, Tuple[float, float]] = (0.0, 0.585),
    binary_edge_erosion_iters: int = 1,
    binary_center_protect_radius_ratio: float = 0.40,
    binary_edge_erosion_kernel: int = 13,
    heatmap_overlay_viz_clamp: float = 0.5,
    device: str = "cuda",
    num_iterations: int = 3,
    inter_iteration_dilation: int = 5,
    save_all_visualizations: bool = True,
    include_full_analysis_figure: bool = False,
    flip_upside_down: bool = False,
    enable_visualizations: bool = True,
    save_aggregation_figures: bool = True,
    aggregation_figures_max_samples: int = 3,
    save_token_mask_figures: bool = True,
    heatmap_ideas_generator: bool = False,
    token_mask_avg_threshold: float = 0.5,
    token_mask_topk_ratio: float = 0.1,
    token_surprisal_samples: int = 50,
    token_surprisal_mask_ratio: float = 0.15,
    token_surprisal_clamp: float = 5.0,
    compute_token_surprisal: bool = True,
    annotation_csv: Optional[str] = None,
    overlay_annotation_boxes: bool = True,
    annotation_flip_vertical: Optional[bool] = None,
    annotation_flip_horizontal: bool = False,
    annotation_preprocess_mode: str = ANNOTATION_PREPROCESS_LEGACY,
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
    if annotation_flip_vertical is None:
        annotation_flip_vertical = bool(flip_upside_down)
    annotation_flip_horizontal = bool(annotation_flip_horizontal)
    if annotation_preprocess_mode not in {
        ANNOTATION_PREPROCESS_LEGACY,
        ANNOTATION_PREPROCESS_RENDER_FASTMRI,
        ANNOTATION_PREPROCESS_MASK_PIPELINE,
    }:
        raise ValueError(
            f"Unsupported annotation preprocess mode: {annotation_preprocess_mode}. "
            f"Use '{ANNOTATION_PREPROCESS_LEGACY}', '{ANNOTATION_PREPROCESS_RENDER_FASTMRI}', "
            f"or '{ANNOTATION_PREPROCESS_MASK_PIPELINE}'."
        )
    annotation_boxes = {}
    if overlay_annotation_boxes and annotation_csv:
        annotation_boxes = load_annotation_boxes(Path(annotation_csv))
    heatmap_ideas_saved = False
    
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
            binary_mask_base_i = (masked_score_i > binary_mask_threshold)
            binary_sum_heatmap_base = float(binary_mask_base_i.sum())

            binary_sum_heatmap = binary_sum_heatmap_base
            binary_sum_heatmap_token = 0.0
            binary_sum_heatmap_overlap = 0.0
            binary_sum_heatmap_lpips_backflow = 0.0
            binary_sum_heatmap_overlap_lpips_backflow = 0.0
            binary_mask_eval_i = binary_mask_base_i.copy()
            token_surprisal_eval_i = None
            if binary_include_token_surprisal and pipeline_results.get("token_surprisal_map") is not None:
                surprisal_i = pipeline_results["token_surprisal_map"][i, 0].cpu().numpy()
                token_binary_i = (surprisal_i > float(binary_token_surprisal_threshold))
                binary_sum_heatmap_token = float(token_binary_i.sum())
                binary_sum_heatmap_overlap = float(np.logical_and(binary_mask_base_i, token_binary_i).sum())
                binary_sum_heatmap = float(np.logical_or(binary_mask_base_i, token_binary_i).sum())
                binary_mask_eval_i = np.logical_or(binary_mask_base_i, token_binary_i)
                token_surprisal_eval_i = surprisal_i

            final_lpips_i = pipeline_results["lpips_recon_inpainted"][i, 0].cpu().numpy()
            binary_mask_before_lpips_backflow_i = binary_mask_eval_i.copy()
            lpips_backflow_mask_i, lpips_backflow_cutoff_i = build_lpips_backflow_mask(
                final_lpips_i,
                lpips_rec_inp_threshold_back_to_binary_token_map,
            )
            binary_sum_heatmap_lpips_backflow = float(lpips_backflow_mask_i.sum())
            binary_sum_heatmap_overlap_lpips_backflow = float(np.logical_and(binary_mask_before_lpips_backflow_i, lpips_backflow_mask_i).sum())
            binary_mask_eval_i = np.logical_or(binary_mask_eval_i, lpips_backflow_mask_i)

            binary_mask_eval_i = apply_edge_to_center_erosion(
                binary_mask_eval_i,
                max_edge_erosion_iters=binary_edge_erosion_iters,
                center_protect_radius_ratio=binary_center_protect_radius_ratio,
                erosion_kernel_size=binary_edge_erosion_kernel,
            )
            binary_sum_heatmap = float(binary_mask_eval_i.sum())

            binary_map_total_pixels = int(binary_mask_eval_i.size)
            binary_sum_heatmap_ratio = float(binary_sum_heatmap) / float(binary_map_total_pixels) if binary_map_total_pixels > 0 else 0.0

            fused_eval_map_i, fused_eval_mask_i = build_final_lpips_binary_token_eval_mask(
                final_lpips=final_lpips_i,
                binary_token_mask=binary_mask_eval_i,
                include_token_surprisal=bool(binary_include_token_surprisal and token_surprisal_eval_i is not None),
                token_surprisal_map=token_surprisal_eval_i,
                lpips_percentile=60.0,
                display_floor=0.5,
            )
            fused_eval_white_pixels = int(fused_eval_mask_i.sum())
            fused_eval_white_ratio = float(fused_eval_white_pixels) / float(binary_map_total_pixels) if binary_map_total_pixels > 0 else 0.0

            boxes_for_slice = []
            if slice_idx is not None:
                boxes_for_slice = annotation_boxes.get(file_stem, {}).get(slice_idx, [])
            bbox_metrics = compute_bbox_detection_metrics(
                predicted_anomaly_mask=binary_mask_eval_i,
                boxes=boxes_for_slice,
                anomaly_score_map=binary_mask_eval_i.astype(np.float32),
                tp_inside_ratio_threshold=TP_INSIDE_RATIO_THRESHOLD,
                flip_vertical=annotation_flip_vertical,
                flip_horizontal=annotation_flip_horizontal,
                preprocess_mode=annotation_preprocess_mode,
            )

            iteration_metrics = []
            for iter_data in pipeline_results["iteration_history"]:
                iteration_metrics.append({
                    "iteration": iter_data["iteration"],
                    "mask_coverage_pre_dilation": iter_data["mask_coverage_pre_dilation"],
                    "mask_coverage": iter_data["mask_coverage"],
                    "inpaint_l1_change": iter_data["inpaint_l1_change"],
                    "max_lpips_recon_inpainted": iter_data["lpips_recon_inpainted"][i].max().item(),
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
                "Binary_Sum_Heatmap_Base": binary_sum_heatmap_base,
                "Binary_Sum_Heatmap_Token": binary_sum_heatmap_token,
                "Binary_Sum_Heatmap_Overlap": binary_sum_heatmap_overlap,
                "Binary_Sum_Heatmap_LPIPS_Backflow": binary_sum_heatmap_lpips_backflow,
                "Binary_Sum_Heatmap_Overlap_LPIPS_Backflow": binary_sum_heatmap_overlap_lpips_backflow,
                "Binary_Sum_Heatmap_WhitePixel_Ratio": binary_sum_heatmap_ratio,
                "Binary_Include_TokenSurprisal": bool(binary_include_token_surprisal),
                "Binary_TokenSurprisal_Threshold": float(binary_token_surprisal_threshold),
                "LPIPS_rec_inp_threshold_back_to_binary_token_map": lpips_rec_inp_threshold_back_to_binary_token_map,
                "LPIPS_rec_inp_threshold_cutoff_value": float(lpips_backflow_cutoff_i),
                "LPIPS_rec_inp_percentile_back_to_binary_token_map": (
                    float(lpips_rec_inp_threshold_back_to_binary_token_map[0])
                    if isinstance(lpips_rec_inp_threshold_back_to_binary_token_map, (tuple, list)) and len(lpips_rec_inp_threshold_back_to_binary_token_map) == 2
                    else 0.0
                ),
                "LPIPS_rec_inp_percentile_cutoff_value": float(lpips_backflow_cutoff_i),
                "Binary_Edge_Erosion_Iters": int(binary_edge_erosion_iters),
                "Binary_Center_Protect_Radius_Ratio": float(binary_center_protect_radius_ratio),
                "Binary_Edge_Erosion_Kernel": int(binary_edge_erosion_kernel),
                "bbox_evaluation_map_source": "Binary_Token_Map",
                "final_fused_lpips_binary_token_white_pixels": fused_eval_white_pixels,
                "final_fused_lpips_binary_token_white_pixel_ratio": fused_eval_white_ratio,
                "has_ground_truth_bbox": bool(bbox_metrics["has_ground_truth_bbox"]),
                "num_ground_truth_boxes": bbox_metrics["num_ground_truth_boxes"],
                "num_bbox_evaluations": int(bbox_metrics.get("num_bbox_evaluations", bbox_metrics["num_ground_truth_boxes"])),
                "num_true_positive_bboxes": int(bbox_metrics.get("num_true_positive_bboxes", 0)),
                "ground_truth_bbox_pixels": bbox_metrics["ground_truth_bbox_pixels"],
                "highlighted_anomaly_pixels_binary_token": int(binary_sum_heatmap),
                "highlighted_anomaly_pixels_fused_eval_map": int(fused_eval_white_pixels),
                "highlighted_anomaly_score_sum_fused_eval_map": float(np.nansum(fused_eval_map_i)),
                "predicted_anomaly_pixels_inside_bbox": bbox_metrics["predicted_anomaly_pixels_inside_bbox"],
                "predicted_anomaly_pixels_outside_bbox": bbox_metrics["predicted_anomaly_pixels_outside_bbox"],
                "predicted_anomaly_pixels_inside_other_bboxes_excluded": int(bbox_metrics.get("predicted_anomaly_pixels_inside_other_bboxes_excluded", 0)),
                "anomaly_score_sum_inside_bbox": float(bbox_metrics.get("anomaly_score_sum_inside_bbox", 0.0)),
                "anomaly_score_sum_outside_bbox": float(bbox_metrics.get("anomaly_score_sum_outside_bbox", 0.0)),
                "anomaly_score_sum_inside_other_bboxes_excluded": float(bbox_metrics.get("anomaly_score_sum_inside_other_bboxes_excluded", 0.0)),
                "inside_bbox_detection_ratio": bbox_metrics["inside_bbox_detection_ratio"],
                "true_positive": bbox_metrics["true_positive"],
                "false_positive_ratio": bbox_metrics["false_positive_ratio"],
                "precision": bbox_metrics["precision"],
                "f1_score": bbox_metrics.get("f1_score"),
                "bbox_evaluation_mode": bbox_metrics.get("bbox_evaluation_mode", "per_pathology_bbox"),
                "bbox_evaluations": bbox_metrics.get("bbox_evaluations", []),
                "tp_inside_ratio_threshold": bbox_metrics["tp_inside_ratio_threshold"],
                "detected_based_on_thresholds": bbox_metrics["true_positive"],
                "artifact_flag": bool(pipeline_results.get("artifact_flag")[i].item()),
                "sharpness_score": float(pipeline_results.get("sharpness_score")[i].item()),
            }
            
            results_list.append(result)
            
            if enable_visualizations and (save_all_visualizations or (batch_idx < 3 and i < 4)):
                base_name = f"{category}_{case_folder}_{filename.replace('.npy', '')}"
                if include_full_analysis_figure:
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
                        binary_include_token_surprisal=binary_include_token_surprisal,
                        binary_token_surprisal_threshold=binary_token_surprisal_threshold,
                        lpips_rec_inp_threshold_back_to_binary_token_map=lpips_rec_inp_threshold_back_to_binary_token_map,
                        binary_edge_erosion_iters=binary_edge_erosion_iters,
                        binary_center_protect_radius_ratio=binary_center_protect_radius_ratio,
                        binary_edge_erosion_kernel=binary_edge_erosion_kernel,
                        annotation_boxes=annotation_boxes,
                        file_stem=file_stem,
                        slice_idx=slice_idx,
                        annotation_focus_label=(
                            category if str(category).strip().lower() not in ("", "unknown", "fastmri") else case_folder
                        ),
                        annotation_flip_vertical=annotation_flip_vertical,
                        annotation_flip_horizontal=annotation_flip_horizontal,
                        annotation_preprocess_mode=annotation_preprocess_mode,
                    )
                
                # Save simplified Anomaly Overlay figure
                anomaly_overlay_path = os.path.join(visualizations_dir, f"{base_name}_Anomaly_Overlay.png")
                visualize_anomaly_overlay(
                    pipeline_results, sample_idx=i,
                    title=f"{category}/{case_folder}/{filename}",
                    save_path=anomaly_overlay_path,
                    heatmap_overlay_viz_clamp=heatmap_overlay_viz_clamp,
                    binary_mask_threshold=binary_mask_threshold,
                    binary_mask_iteration=binary_mask_iteration,
                    binary_include_token_surprisal=binary_include_token_surprisal,
                    binary_token_surprisal_threshold=binary_token_surprisal_threshold,
                    lpips_rec_inp_threshold_back_to_binary_token_map=lpips_rec_inp_threshold_back_to_binary_token_map,
                    binary_edge_erosion_iters=binary_edge_erosion_iters,
                    binary_center_protect_radius_ratio=binary_center_protect_radius_ratio,
                    binary_edge_erosion_kernel=binary_edge_erosion_kernel,
                    binary_token_boost_value=0.95,
                    annotation_boxes=annotation_boxes,
                    file_stem=file_stem,
                    slice_idx=slice_idx,
                    annotation_focus_label=(
                        category if str(category).strip().lower() not in ("", "unknown", "fastmri") else case_folder
                    ),
                    annotation_flip_vertical=annotation_flip_vertical,
                    annotation_flip_horizontal=annotation_flip_horizontal,
                    annotation_preprocess_mode=annotation_preprocess_mode,
                    detected_based_on_thresholds=result.get("detected_based_on_thresholds"),
                    binary_token_white_pixel_ratio=result.get("Binary_Sum_Heatmap_WhitePixel_Ratio"),
                )
    
    return {"slices": results_list}



def summarize_and_save_results(args, results: Dict, output_dir: str):
    """
    Serialize per-slice and per-patient results to JSON.

    AUROC-feeding fields (CORE — consumed by Plot_Bars/run_fastmri_roc_analysis):
      • per-slice: "Binary_Sum_Heatmap", "category", "case_folder",
                    "filename", "path", "patient_id" (derived).
      • patient_summary: "patient_id", "case_folder", "category",
                          "slice_details" (filename + Binary_Sum_Heatmap link).

    All other fields (clamped sums, lpips_input_recon_sum_mask, bbox precision/F1,
    detected_based_on_thresholds, fused-eval-map metrics, etc.) are AYNU.
    They populate auxiliary tables/plots but are NOT used to compute AUROC.
    """
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
        case_folder = str(r.get("case_folder", "") or "").strip()
        if case_folder and case_folder.lower() != "unknown":
            return case_folder
        fname = str(r.get("filename", "") or "").strip()
        stem = os.path.splitext(fname)[0]
        if "_slice_" in stem:
            return stem.split("_slice_", 1)[0]
        return stem or case_folder or "unknown"

    patient_sums = defaultdict(lambda: {
        "total_clamped_sum": 0.0,
        "total_first_heatmap_sum": 0.0,
        "total_lpips_input_recon_sum_mask": 0.0,
        "total_pixels_above_thresh": 0,
        "total_token_surprisal_hot_px": 0,
        "total_binary_sum_heatmap": 0.0,
        "total_binary_sum_heatmap_base": 0.0,
        "total_binary_sum_heatmap_token": 0.0,
        "total_binary_sum_heatmap_overlap": 0.0,
        "sum_gt_bbox_pixels": 0,
        "sum_predicted_inside_bbox": 0,
        "sum_predicted_outside_bbox": 0,
        "sum_anomaly_score_inside_bbox": 0.0,
        "sum_anomaly_score_outside_bbox": 0.0,
        "sum_highlighted_anomaly_pixels_binary_token": 0,
        "sum_highlighted_anomaly_pixels_no_bbox": 0,
        "num_slices_with_gt_bbox": 0,
        "num_tp_slices": 0,
        "num_bbox_evaluations": 0,
        "num_true_positive_bboxes": 0,
        "sum_predicted_inside_other_bboxes_excluded": 0,
        "num_slices": 0,
        "category": "",
        "binary_include_token_surprisal": False,
        "binary_token_surprisal_threshold": 0.0,
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
        patient_sums[patient_key]["total_binary_sum_heatmap"] += float(r.get("Binary_Sum_Heatmap", 0.0) or 0.0)
        patient_sums[patient_key]["total_binary_sum_heatmap_base"] += float(r.get("Binary_Sum_Heatmap_Base", 0.0) or 0.0)
        patient_sums[patient_key]["total_binary_sum_heatmap_token"] += float(r.get("Binary_Sum_Heatmap_Token", 0.0) or 0.0)
        patient_sums[patient_key]["total_binary_sum_heatmap_overlap"] += float(r.get("Binary_Sum_Heatmap_Overlap", 0.0) or 0.0)
        patient_sums[patient_key]["sum_highlighted_anomaly_pixels_binary_token"] += int(r.get("highlighted_anomaly_pixels_binary_token", 0) or 0)
        patient_sums[patient_key]["num_bbox_evaluations"] += int(r.get("num_bbox_evaluations", r.get("num_ground_truth_boxes", 0)) or 0)
        patient_sums[patient_key]["num_true_positive_bboxes"] += int(r.get("num_true_positive_bboxes", 0) or 0)
        has_gt = bool(r.get("has_ground_truth_bbox", False))
        if has_gt:
            patient_sums[patient_key]["sum_gt_bbox_pixels"] += int(r.get("ground_truth_bbox_pixels", 0) or 0)
            patient_sums[patient_key]["sum_predicted_inside_bbox"] += int(r.get("predicted_anomaly_pixels_inside_bbox", 0) or 0)
            patient_sums[patient_key]["sum_predicted_outside_bbox"] += int(r.get("predicted_anomaly_pixels_outside_bbox", 0) or 0)
            patient_sums[patient_key]["sum_anomaly_score_inside_bbox"] += float(r.get("anomaly_score_sum_inside_bbox", 0.0) or 0.0)
            patient_sums[patient_key]["sum_anomaly_score_outside_bbox"] += float(r.get("anomaly_score_sum_outside_bbox", 0.0) or 0.0)
            patient_sums[patient_key]["sum_predicted_inside_other_bboxes_excluded"] += int(r.get("predicted_anomaly_pixels_inside_other_bboxes_excluded", 0) or 0)
            patient_sums[patient_key]["num_slices_with_gt_bbox"] += 1
            if bool(r.get("true_positive", False)):
                patient_sums[patient_key]["num_tp_slices"] += 1
        else:
            patient_sums[patient_key]["sum_highlighted_anomaly_pixels_no_bbox"] += int(r.get("highlighted_anomaly_pixels_binary_token", 0) or 0)
        patient_sums[patient_key]["num_slices"] += 1
        patient_sums[patient_key]["category"] = r["category"]
        patient_sums[patient_key]["case_folder"] = r.get("case_folder", "")
        patient_sums[patient_key]["patient_label"] = patient_label
        patient_sums[patient_key]["binary_include_token_surprisal"] = bool(r.get("Binary_Include_TokenSurprisal", False))
        patient_sums[patient_key]["binary_token_surprisal_threshold"] = float(r.get("Binary_TokenSurprisal_Threshold", 0.0) or 0.0)
        patient_sums[patient_key]["slice_sums"].append({
            "filename": r["filename"],
            "clamped_sum": r.get("clamped_pixel_sum", 0.0),
            "clamped_sum_first_heatmap": r.get("clamped_pixel_sum_FirstHeatmap", 0.0),
            "lpips_input_recon_sum_mask": r.get("lpips_input_recon_sum_mask", 0.0),
            "pixels_above_thresh": r.get("num_pixels_above_clamp_thresh", 0),
            "token_surprisal_hot_px": r.get("token_surprisal_hot_px"),
            "Binary_Sum_Heatmap": r.get("Binary_Sum_Heatmap", 0.0),
            "Binary_Sum_Heatmap_Base": r.get("Binary_Sum_Heatmap_Base", 0.0),
            "Binary_Sum_Heatmap_Token": r.get("Binary_Sum_Heatmap_Token", 0.0),
            "Binary_Sum_Heatmap_Overlap": r.get("Binary_Sum_Heatmap_Overlap", 0.0),
            "Binary_Include_TokenSurprisal": bool(r.get("Binary_Include_TokenSurprisal", False)),
            "Binary_TokenSurprisal_Threshold": float(r.get("Binary_TokenSurprisal_Threshold", 0.0) or 0.0),
            "has_ground_truth_bbox": bool(r.get("has_ground_truth_bbox", False)),
            "num_ground_truth_boxes": int(r.get("num_ground_truth_boxes", 0) or 0),
            "num_bbox_evaluations": int(r.get("num_bbox_evaluations", r.get("num_ground_truth_boxes", 0)) or 0),
            "num_true_positive_bboxes": int(r.get("num_true_positive_bboxes", 0) or 0),
            "ground_truth_bbox_pixels": int(r.get("ground_truth_bbox_pixels", 0) or 0),
            "highlighted_anomaly_pixels_binary_token": int(r.get("highlighted_anomaly_pixels_binary_token", 0) or 0),
            "highlighted_anomaly_score_sum_fused_eval_map": float(r.get("highlighted_anomaly_score_sum_fused_eval_map", 0.0) or 0.0),
            "predicted_anomaly_pixels_inside_bbox": int(r.get("predicted_anomaly_pixels_inside_bbox", 0) or 0),
            "predicted_anomaly_pixels_outside_bbox": int(r.get("predicted_anomaly_pixels_outside_bbox", 0) or 0),
            "predicted_anomaly_pixels_inside_other_bboxes_excluded": int(r.get("predicted_anomaly_pixels_inside_other_bboxes_excluded", 0) or 0),
            "anomaly_score_sum_inside_bbox": float(r.get("anomaly_score_sum_inside_bbox", 0.0) or 0.0),
            "anomaly_score_sum_outside_bbox": float(r.get("anomaly_score_sum_outside_bbox", 0.0) or 0.0),
            "anomaly_score_sum_inside_other_bboxes_excluded": float(r.get("anomaly_score_sum_inside_other_bboxes_excluded", 0.0) or 0.0),
            "inside_bbox_detection_ratio": r.get("inside_bbox_detection_ratio"),
            "true_positive": r.get("true_positive"),
            "false_positive_ratio": r.get("false_positive_ratio"),
            "precision": r.get("precision"),
            "f1_score": r.get("f1_score"),
            "bbox_evaluation_mode": r.get("bbox_evaluation_mode", "per_pathology_bbox"),
            "bbox_evaluations": r.get("bbox_evaluations", []),
            "detected_based_on_thresholds": r.get("detected_based_on_thresholds"),
        })

    patient_summary = []
    for patient_key, data in patient_sums.items():
        case_fp_ratio = None
        if data["sum_predicted_inside_bbox"] > 0:
            case_fp_ratio = float(data["sum_predicted_outside_bbox"]) / float(data["sum_predicted_inside_bbox"])
        case_detected_tp = bool(data["num_true_positive_bboxes"] > 0) if data["num_bbox_evaluations"] > 0 else None
        case_tp_value = 1.0 if case_detected_tp else 0.0
        case_precision = None
        if case_fp_ratio is not None and case_detected_tp is not None:
            denom = case_tp_value + case_fp_ratio
            case_precision = float(case_tp_value / denom) if denom > 0 else 0.0
        case_f1 = 0.0
        if case_detected_tp is None:
            case_f1 = None
        elif case_detected_tp and case_precision is not None:
            case_f1 = float((2.0 * case_precision) / (case_precision + 1.0)) if (case_precision + 1.0) > 0 else 0.0

        patient_summary.append({
            "patient_id": patient_key,
            "category": data["category"],
            "case_folder": data["case_folder"],
            "total_clamped_pixel_sum": data["total_clamped_sum"],
            "total_clamped_pixel_sum_first_heatmap": data["total_first_heatmap_sum"],
            "total_lpips_input_recon_sum_mask": data["total_lpips_input_recon_sum_mask"],
            "total_pixels_above_thresh": data["total_pixels_above_thresh"],
            "total_token_surprisal_hot_px": data["total_token_surprisal_hot_px"],
            "total_Binary_Sum_Heatmap": data["total_binary_sum_heatmap"],
            "total_Binary_Sum_Heatmap_Base": data["total_binary_sum_heatmap_base"],
            "total_Binary_Sum_Heatmap_Token": data["total_binary_sum_heatmap_token"],
            "total_Binary_Sum_Heatmap_Overlap": data["total_binary_sum_heatmap_overlap"],
            "num_bbox_evaluations": data["num_bbox_evaluations"],
            "num_true_positive_bboxes": data["num_true_positive_bboxes"],
            "num_slices_with_ground_truth_bbox": data["num_slices_with_gt_bbox"],
            "num_true_positive_slices": data["num_tp_slices"],
            "detected_based_on_thresholds": case_detected_tp,
            "sum_ground_truth_bbox_pixels": data["sum_gt_bbox_pixels"],
            "sum_predicted_anomaly_pixels_inside_bbox": data["sum_predicted_inside_bbox"],
            "sum_predicted_anomaly_pixels_outside_bbox": data["sum_predicted_outside_bbox"],
            "sum_anomaly_score_inside_bbox": data["sum_anomaly_score_inside_bbox"],
            "sum_anomaly_score_outside_bbox": data["sum_anomaly_score_outside_bbox"],
            "sum_predicted_anomaly_pixels_inside_other_bboxes_excluded": data["sum_predicted_inside_other_bboxes_excluded"],
            "sum_highlighted_anomaly_pixels_binary_token": data["sum_highlighted_anomaly_pixels_binary_token"],
            "sum_highlighted_anomaly_pixels_no_bbox": data["sum_highlighted_anomaly_pixels_no_bbox"],
            "false_positive_ratio": case_fp_ratio,
            "precision": case_precision,
            "f1_score": case_f1,
            "tp_inside_ratio_threshold": TP_INSIDE_RATIO_THRESHOLD,
            "num_slices": data["num_slices"],
            "mean_clamped_sum_per_slice": data["total_clamped_sum"] / data["num_slices"] if data["num_slices"] > 0 else 0,
            "mean_clamped_sum_first_heatmap_per_slice": data["total_first_heatmap_sum"] / data["num_slices"] if data["num_slices"] > 0 else 0,
            "mean_lpips_input_recon_sum_mask_per_slice": data["total_lpips_input_recon_sum_mask"] / data["num_slices"] if data["num_slices"] > 0 else 0,
            "mean_token_surprisal_hot_px_per_slice": data["total_token_surprisal_hot_px"] / data["num_slices"] if data["num_slices"] > 0 else 0,
            "mean_Binary_Sum_Heatmap_per_slice": data["total_binary_sum_heatmap"] / data["num_slices"] if data["num_slices"] > 0 else 0,
            "mean_Binary_Sum_Heatmap_Base_per_slice": data["total_binary_sum_heatmap_base"] / data["num_slices"] if data["num_slices"] > 0 else 0,
            "mean_Binary_Sum_Heatmap_Token_per_slice": data["total_binary_sum_heatmap_token"] / data["num_slices"] if data["num_slices"] > 0 else 0,
            "mean_Binary_Sum_Heatmap_Overlap_per_slice": data["total_binary_sum_heatmap_overlap"] / data["num_slices"] if data["num_slices"] > 0 else 0,
            "Binary_Include_TokenSurprisal": bool(data["binary_include_token_surprisal"]),
            "Binary_TokenSurprisal_Threshold": float(data["binary_token_surprisal_threshold"]),
            "slice_details": data["slice_sums"]
        })

    patient_summary = sorted(patient_summary, key=lambda x: x["total_clamped_pixel_sum"], reverse=True)

    print(f"\n{'='*60}")
    print("PER-PATIENT CLAMPED PIXEL SUM (sorted by total sum)")
    print(f"{'='*60}")
    print(f"Clamp threshold: {args.clamp_threshold}")

    evaluated_slices = [r for r in slices_for_json if r.get("has_ground_truth_bbox")]
    evaluated_bboxes = int(sum(int(r.get("num_bbox_evaluations", r.get("num_ground_truth_boxes", 0)) or 0) for r in slices_for_json))
    tp_bboxes = int(sum(int(r.get("num_true_positive_bboxes", 0) or 0) for r in slices_for_json))
    print(
        f"BBox eval slices: {len(evaluated_slices)} | "
        f"BBox evaluations: {evaluated_bboxes} | "
        f"TP bboxes (>= {TP_INSIDE_RATIO_THRESHOLD:.0%} bbox coverage): {tp_bboxes}"
    )

    with open(os.path.join(output_dir, "results_v4_zscore.json"), 'w') as f:
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
                "binary_include_token_surprisal": args.binary_include_token_surprisal,
                "binary_token_surprisal_threshold": args.binary_token_surprisal_threshold,
                "lpips_rec_inp_threshold_back_to_binary_token_map": args.lpips_rec_inp_threshold_back_to_binary_token_map,
                "lpips_rec_inp_percentile_back_to_binary_token_map": (
                    float(args.lpips_rec_inp_threshold_back_to_binary_token_map[0])
                    if isinstance(args.lpips_rec_inp_threshold_back_to_binary_token_map, (tuple, list))
                    else 0.0
                ),
                "lpips_rec_inp_fixed_threshold_back_to_binary_token_map": (
                    float(args.lpips_rec_inp_threshold_back_to_binary_token_map[1])
                    if isinstance(args.lpips_rec_inp_threshold_back_to_binary_token_map, (tuple, list))
                    else float(args.lpips_rec_inp_threshold_back_to_binary_token_map)
                ),
                "binary_edge_erosion_iters": args.binary_edge_erosion_iters,
                "binary_center_protect_radius_ratio": args.binary_center_protect_radius_ratio,
                "binary_edge_erosion_kernel": args.binary_edge_erosion_kernel,
                "annotation_preprocess_mode": args.annotation_preprocess_mode,
                "tp_inside_ratio_threshold": TP_INSIDE_RATIO_THRESHOLD,
            },
            "summary": {
                "num_slices": len(slices_for_json),
                "mean_initial_coverage": float(np.mean(initial_coverages)) if slices_for_json else 0.0,
                "mean_final_coverage": float(np.mean(final_coverages)) if slices_for_json else 0.0,
                "mean_lpips_input_recon_sum_mask": float(np.mean([r.get("lpips_input_recon_sum_mask", 0.0) or 0.0 for r in slices_for_json])) if slices_for_json else 0.0,
                "num_slices_with_ground_truth_bbox": len(evaluated_slices),
                "num_bbox_evaluations": int(evaluated_bboxes),
                "num_true_positive_bboxes": int(tp_bboxes),
                "tp_rate_on_evaluated_bboxes": float(tp_bboxes / evaluated_bboxes) if evaluated_bboxes > 0 else 0.0,
            },
            "patient_summary": patient_summary,
            "results": slices_for_json
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")



# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Recursive-AutoMask V4 with Z-Score Calibration")
    ###python  "file_name.py" --run-all-anomaly-folders --binary-include-token-surprisal --no-figures-only-json

    # Model paths
    parser.add_argument("--stage1-ckpt", type=str, default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_IXI_Augmented_lightningCheckpoints/FastMRI_stage1-epoch=099-val/loss=0.0891.ckpt")
    parser.add_argument("--stage2-ckpt", type=str, default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_IXI_Augmented_lightningCheckpoints/FastMRI_stage2-epoch=098-val/loss=2.1141.ckpt")
    
    parser.add_argument("--data-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_Anomalies_Collection/FastMRI_ALL_Anomalies_ByLabel_BestSlice_with_Label_full_val/Validation_samples_FastMRI")
    parser.add_argument("--output-dir", type=str, default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/Inference_FastMRI_Results_SOTA_5p_figures_Rec_Heal__Automatic_ALLanomalies_full_val_with_figures")

    parser.add_argument("--anomaly-root-dir", type=str,
                        #default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_Anomalies_Collection/FastMRI_Local_Anomalies_ByLabel_BestSlice_with_Label_SOTA",
                        default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_Anomalies_Collection/FastMRI_ALL_Anomalies_ByLabel_BestSlice_with_Label_full_val",
                        help="Root directory containing anomaly label folders")
    
    parser.add_argument("--run-all-anomaly-folders", action="store_true",
                        help="Run inference for each anomaly label folder under --anomaly-root-dir and save in output_dir/<label>")
    # Mode selection
    parser.add_argument("--calibration-mode", action="store_true",help="Run calibration on healthy volunteers (saves μ/σ maps)")
    parser.add_argument("--calibration-map", type=str, default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/Inference_FastMRI_Results_SOTA_5p_figures_Rec_Heal__Automatic_ALLanomalies_full_val/zscore_calibration.npz", help="Path to calibration .npz file (for inference mode)")
    
    parser.add_argument("--annotation-csv", type=str, default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/Annotated_fastMRI_Brains_Detailed.csv",
                        help="Optional CSV with bounding boxes/labels for overlay")
    parser.add_argument("--no-annotation-boxes", action="store_true",
                        help="Disable annotation box overlay on Final LPIPS panel")
    
    # Z-Score parameters
    parser.add_argument("--z-threshold", type=str, default="(-2.5 , 6.0)", help="Z-score threshold: single value (e.g. 1.25) or two-sided range (e.g. '(-4.5, 1.2)')")

    parser.add_argument("--z-epsilon", type=float, default=0.01,
                        help="Epsilon for numerical stability in Z-score computation")
    parser.add_argument("--smoothing-kernel", type=int, default=7,
                        help="Kernel size for spatial smoothing (handles registration noise, default: 15)")
    parser.add_argument("--heatmap-aggregation", type=str, default="mean",
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
    parser.add_argument("--calibration-substring", type=str, default=None,
                        help="Substring required in filenames during --calibration-mode (default: orig)")
    parser.add_argument("--no-calibration-substring-filter", action="store_true",
                        help="Disable automatic calibration filename substring filtering")
    
    # V4 Recursive parameters
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--inter-iteration-dilation", type=int, default=1)

    # Artifact guards
    parser.add_argument("--blur-threshold", type=float, default=0.002,
                        help="Sharpness (Laplacian variance) threshold; below -> artifact blur")
#---------------------------------------------------------------------------------------------------------------------------------------------    
#---------------------------------------------------------------------------------------------------------------------------------------------    
#---------------------------------------------------------------------------------------------------------------------------------------------    
    
    # Healing parameters
    parser.add_argument("--heal-steps", type=int, default=6)
    parser.add_argument("--heal-temperature", type=float, default=0.9)
    parser.add_argument("--heal-patterns", type=str, default="4")
    
    # Mask parameters (fallback)
    parser.add_argument("--token-surprisal-clamp", type=float, default=5.0, 
                        help="Clamp threshold for token surprisal NLL map (values <= threshold set to 0)")

    parser.add_argument("--binary-token-surprisal-threshold", type=float, default=5.0,
                        help="Token Surprisal threshold for inclusion in Binary Sum (default: >0)")

    parser.add_argument(
        "--LPIPS-rec-inp-threshold-back-to-binary-token-map",
        "--LPIPS-in-inp-threshold-back-to-binary-token-map",
        dest="lpips_rec_inp_threshold_back_to_binary_token_map",
        type=str,
        #default="(0, 0.35)",
        default="(97, 0)",
        help=(
            "Backflow mode selector for LPIPS(rec, inp). "
            "Use '(percentile, 0)' for percentile mode (e.g. '(95, 0)') "
            "or '(0, threshold)' for fixed-threshold mode (e.g. '(0, 0.33)')."
        ),
    )

#---------------------------------------------------------------------------------------------------------------------------------------------    
#---------------------------------------------------------------------------------------------------------------------------------------------    
#---------------------------------------------------------------------------------------------------------------------------------------------    

    parser.add_argument("--mask-threshold", type=float, default=95.0)
    parser.add_argument("--binary-threshold", type=float, default=0.585,
                        help="Threshold on masked score for binary sum map visualization")
    
    parser.add_argument("--mask-dilation", type=int, default=1)

    parser.add_argument("--token-mask-mode", type=str, default="max",
                        choices=["max", "avg", "topk"],
                        help="Token mask mode: max-pool, avg-pool, or top-k tokens")
    parser.add_argument("--token-mask-avg-threshold", type=float, default=0.5,
                        help="Avg-pool threshold for token mask (mode=avg)")
    parser.add_argument("--token-mask-topk-ratio", type=float, default=0.1,
                        help="Top-k ratio for token mask (mode=topk)")

    parser.add_argument("--token-surprisal-samples", type=int, default=100,
                        help="Number of random masks for token surprisal (pseudo-PLL)")
    parser.add_argument("--token-surprisal-mask-ratio", type=float, default=0.820,
                        help="Mask ratio per sample for token surprisal")
    

    parser.add_argument("--no-token-surprisal", action="store_true",
                        help="Disable token surprisal scoring (faster)")

    parser.add_argument("--clamp-threshold", type=float, default=0.90,
                        help="Threshold applied to sqrt(First × Last) for clamped sums/visualizations")
    parser.add_argument("--pixle-sum-first-heatmap-thresh", type=float, default=300.0,
                        help="Threshold for coloring the first masked heatmap sum (green below, red above)")

    parser.add_argument("--binary-mask-iteration", type=int, default=0,
                        help="Which iteration's masked score to binarize (0 = first, -1 = last)")
    parser.add_argument("--binary-include-token-surprisal", action="store_true",
                        help="Add visible Token Surprisal (NLL) pixels to the Binary Sum map (union)")
    
    parser.add_argument("--binary-edge-erosion-iters", type=int, default=2,
                        help="Edge-aware erosion strength: max erosion iterations at image borders (0 disables)")
    parser.add_argument("--binary-center-protect-radius-ratio", type=float, default=0.35,
                        help="Normalized center radius (0-1) protected from edge erosion")
    parser.add_argument("--binary-edge-erosion-kernel", type=int, default=13,
                        help="Kernel size for edge-aware binary erosion")


    parser.add_argument("--Heatmap-overlay-viz-clamp", type=float, default=0.3,
                        help="Clamp threshold for heatmap overlay visualization (values below set to 0)")
    
    # Inpainting parameters
    parser.add_argument("--inpaint-steps", type=int, default=12)
    parser.add_argument("--inpaint-temperature", type=float, default=0.5)
    
    # TTA
    parser.add_argument("--use-tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="use_tta")
    parser.add_argument("--flip-upside-down", action="store_true", default=False,
                        help="Flip input slices vertically before processing")
    parser.add_argument("--annotation-flip-vertical", action="store_true", default=False,
                        help="Flip annotation boxes vertically to match image orientation")
    parser.add_argument("--annotation-flip-horizontal", action="store_true", default=False,
                        help="Flip annotation boxes horizontally to match image orientation")
    parser.add_argument(
        "--annotation-preprocess-mode",
        type=str,
        default=ANNOTATION_PREPROCESS_LEGACY,
        choices=[
            ANNOTATION_PREPROCESS_LEGACY,
            ANNOTATION_PREPROCESS_RENDER_FASTMRI,
            ANNOTATION_PREPROCESS_MASK_PIPELINE,
        ],
        help=(
            "How to preprocess annotation coordinates before overlay. "
            "'legacy' keeps historical behavior; 'render_fastmri' applies coordinate-level vertical remap; "
            "'mask_pipeline' rasterizes box masks and applies image-like preprocessing (flip/crop/resize) before projecting back to rectangles."
        ),
    )
    
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--save-all-visualizations", action="store_true", default=True)
    parser.add_argument(
        "--include-full-analysis-figure",
        action="store_true",
        default=False,
        help="Enable generation of *_full.png analysis figures (disabled by default)",
    )
    parser.add_argument("--no-figures-only-json", action="store_true",
                        help="Skip all figure generation and only emit the JSON output")
    parser.add_argument("--no-aggregation-figures", action="store_true",
                        help="Disable heatmap aggregation comparison figures")
    parser.add_argument("--no-token-mask-figures", action="store_true",
                        help="Disable token mask comparison figures")
    parser.add_argument("--heatmap-ideas-generator", action="store_true",
                        help="Generate visual colormap comparison figure (heatmap_ideas.png) from first eligible sample")
    parser.add_argument("--aggregation-figures-max-samples", type=int, default=1,
                        help="Max samples per batch for aggregation/token mask comparison figures")
    
    args = parser.parse_args()

    try:
        args.z_threshold = parse_z_threshold_arg(args.z_threshold)
    except ValueError as exc:
        raise SystemExit(f"Invalid --z-threshold: {exc}")

    try:
        args.lpips_rec_inp_threshold_back_to_binary_token_map = parse_lpips_backflow_selector_arg(
            args.lpips_rec_inp_threshold_back_to_binary_token_map
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid --LPIPS-rec-inp-threshold-back-to-binary-token-map: {exc}")

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
    
    heal_patterns = [int(p) for p in args.heal_patterns.split(",")]
    
    # =================================================================
    # CALIBRATION MODE
    # =================================================================
    if args.calibration_mode:
        if args.run_all_anomaly_folders:
            raise RuntimeError("--run-all-anomaly-folders is only supported for inference mode.")

        os.makedirs(args.output_dir, exist_ok=True)

        dataloader = create_inference_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=4,
            category_filter=effective_category_filter,
            case_filter=[args.case] if args.case else None,
            patient_filter=combined_patient_filter if combined_patient_filter else None,
            category_name="FastMRI",
        )

        selected_files = sorted(getattr(dataloader.dataset, "files", []))
        calibration_audit_path = os.path.join(args.output_dir, "calibration_input_files.txt")

        print("\n" + "="*70)
        print("CALIBRATION INPUT AUDIT")
        print("="*70)
        print(f"Calibration data-dir: {args.data_dir}")
        print(f"Resolved absolute data-dir: {os.path.abspath(args.data_dir)}")
        print(f"Category filter: {effective_category_filter if effective_category_filter else 'None'}")
        print(f"Case filter: {[args.case] if args.case else 'None'}")
        print(f"Patient filter: {combined_patient_filter if combined_patient_filter else 'None'}")
        print(f"Total calibration files selected: {len(selected_files)}")
        if selected_files:
            print("Selected files:")
            for idx, fpath in enumerate(selected_files, start=1):
                print(f"  [{idx:04d}] {fpath}")
        else:
            print("Selected files: none")

        with open(calibration_audit_path, "w") as f:
            f.write(f"timestamp: {datetime.now().isoformat()}\n")
            f.write(f"calibration_data_dir: {args.data_dir}\n")
            f.write(f"calibration_data_dir_abs: {os.path.abspath(args.data_dir)}\n")
            f.write(f"category_filter: {effective_category_filter if effective_category_filter else 'None'}\n")
            f.write(f"case_filter: {[args.case] if args.case else 'None'}\n")
            f.write(f"patient_filter: {combined_patient_filter if combined_patient_filter else 'None'}\n")
            f.write(f"num_files: {len(selected_files)}\n")
            f.write("files:\n")
            for idx, fpath in enumerate(selected_files, start=1):
                f.write(f"{idx:04d}\t{fpath}\n")

        print(f"Calibration file audit saved to: {calibration_audit_path}")

        print(f"\nDataset: {len(dataloader.dataset)} slices")
        if combined_patient_filter:
            print(f"Patient filter applied: {combined_patient_filter}")

        print("\n" + "="*70)
        print("CALIBRATION MODE - Processing Healthy Volunteers")
        print("="*70)

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
        print(f"  Z-threshold: {format_z_threshold(args.z_threshold)}")
    else:
        print("\nNo calibration map provided - using percentile thresholding")
    
    print(f"\n{'='*60}")
    print(f"RECURSIVE AUTOMASK V4 - Z-SCORE {'ENABLED' if calibration else 'DISABLED'}")
    print(f"{'='*60}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Z-threshold: {format_z_threshold(args.z_threshold)}")
    print(f"{'='*60}\n")

    def _run_single_inference(data_dir: str, output_dir: str, category_name: str):
        dataloader = create_inference_dataloader(
            data_dir=data_dir,
            batch_size=args.batch_size,
            num_workers=4,
            category_filter=effective_category_filter,
            case_filter=[args.case] if args.case else None,
            patient_filter=combined_patient_filter if combined_patient_filter else None,
            category_name=category_name,
        )

        print(f"\nDataset ({category_name}): {len(dataloader.dataset)} slices")
        if combined_patient_filter:
            print(f"Patient filter applied: {combined_patient_filter}")

        os.makedirs(output_dir, exist_ok=True)

        results = run_inference_v4_zscore(
            stage1, stage2, perceptual_loss,
            dataloader, output_dir,
            calibration=calibration,
            z_threshold=args.z_threshold,
            smoothing_kernel=args.smoothing_kernel,
            heatmap_aggregation=args.heatmap_aggregation,
            logsumexp_temp=args.logsumexp_temp,
            clamp_threshold=args.clamp_threshold,
            first_heatmap_sum_thresh=args.pixle_sum_first_heatmap_thresh,
            binary_mask_threshold=args.binary_threshold,
            binary_mask_iteration=args.binary_mask_iteration,
            binary_include_token_surprisal=args.binary_include_token_surprisal,
            binary_token_surprisal_threshold=args.binary_token_surprisal_threshold,
            lpips_rec_inp_threshold_back_to_binary_token_map=args.lpips_rec_inp_threshold_back_to_binary_token_map,
            binary_edge_erosion_iters=args.binary_edge_erosion_iters,
            binary_center_protect_radius_ratio=args.binary_center_protect_radius_ratio,
            binary_edge_erosion_kernel=args.binary_edge_erosion_kernel,
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
            annotation_flip_vertical=args.annotation_flip_vertical,
            annotation_flip_horizontal=args.annotation_flip_horizontal,
            annotation_preprocess_mode=args.annotation_preprocess_mode,
            save_all_visualizations=save_visualizations,
            include_full_analysis_figure=args.include_full_analysis_figure,
            enable_visualizations=figures_enabled,
            save_aggregation_figures=save_aggregation_figures,
            aggregation_figures_max_samples=args.aggregation_figures_max_samples,
            save_token_mask_figures=save_token_mask_figures,
            heatmap_ideas_generator=args.heatmap_ideas_generator,
            debug_first_batch=True,
            flip_upside_down=args.flip_upside_down,
        )

        summarize_and_save_results(args, results, output_dir)

    if args.run_all_anomaly_folders:
        label_dirs = find_anomaly_label_dirs(args.anomaly_root_dir)
        print(f"Running batch inference over {len(label_dirs)} anomaly labels from: {args.anomaly_root_dir}")

        for idx, label_dir in enumerate(label_dirs, start=1):
            label = label_dir.name
            run_output_dir = os.path.join(args.output_dir, label)
            print(f"\n[{idx}/{len(label_dirs)}] Processing label: {label}")
            print(f"  Data dir: {label_dir}")
            print(f"  Output dir: {run_output_dir}")
            _run_single_inference(str(label_dir), run_output_dir, category_name=label)
    else:
        _run_single_inference(args.data_dir, args.output_dir, category_name="FastMRI")

# =============================================================================
# =============================================================================
# AYNU — Available Yet Not Used (auxiliary code, NOT on the AUROC path)
# -----------------------------------------------------------------------------
# Code below is preserved for reproducibility, training, alternate scoring,
# bounding-box evaluation tables, sanity-check figures, and per-patient bar
# charts. None of it feeds the AUROC. Skim freely; do not let it distract
# from the CORE pipeline above.
# =============================================================================
# =============================================================================
# -----------------------------------------------------------------------------
# Annotation / bounding-box helpers  (AYNU)
# -----------------------------------------------------------------------------


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
            base_size = _to_int(_get_first(
                row,
                (
                    "base_size",
                    "BaseSize",
                    "base",
                    "Base",
                    "image_size",
                    "ImageSize",
                    "img_size",
                    "ImgSize",
                    "original_size",
                    "OriginalSize",
                ),
            ))
            label = _get_first(row, ("label", "Label"))
            study_level = _get_first(row, ("study_level", "StudyLevel", "study", "Study"))

            box = {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "label": label,
                "study_level": study_level,
                "base_size": int(base_size) if base_size is not None else ANNOTATION_BASE_SIZE,
            }

            boxes_by_file.setdefault(file_stem, {}).setdefault(slice_idx, []).append(box)
        return boxes_by_file



def draw_annotation_boxes(
    ax: plt.Axes,
    boxes: list[dict],
    image_height: int,
    image_width: int,
    base_size: int = ANNOTATION_BASE_SIZE,
    color: str = "yellow",
    focus_label: Optional[str] = None,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
    preprocess_mode: str = ANNOTATION_PREPROCESS_LEGACY,
    only_focus_matches: bool = True,
) -> int:
    def _normalize_text(value: str) -> str:
        value = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
        return re.sub(r"\s+", " ", value)

    focus_norm = _normalize_text(focus_label or "")

    def _is_focus_box(box: dict) -> bool:
        if not focus_norm:
            return False
        label_norm = _normalize_text(box.get("label", ""))
        if not label_norm:
            return False
        if focus_norm in label_norm or label_norm in focus_norm:
            return True
        focus_tokens = [t for t in focus_norm.split(" ") if len(t) > 2]
        return any(tok in label_norm for tok in focus_tokens)

    focus_flags = [_is_focus_box(box) for box in boxes]
    has_focus_match = any(focus_flags)

    drawn = 0
    for idx, box in enumerate(boxes):
        # Draw only the currently filtered anomaly box when a focus label exists.
        if only_focus_matches and focus_norm:
            if not has_focus_match:
                continue
            if not focus_flags[idx]:
                continue

        rect_xyxy, _ = _project_box_to_image_rect(
            box,
            image_height=image_height,
            image_width=image_width,
            base_size=base_size,
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
            preprocess_mode=preprocess_mode,
        )
        if rect_xyxy is None:
            continue
        x1, y1, x2, y2 = rect_xyxy

        edgecolor = color
        linestyle = "-"
        linewidth = 4.5

        rect = patches.Rectangle(
            (x1, y1),
            (x2 - x1),
            (y2 - y1),
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor="none",
            linestyle=linestyle,
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



def _project_box_to_image_rect(
    box: dict,
    image_height: int,
    image_width: int,
    base_size: int = ANNOTATION_BASE_SIZE,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
    preprocess_mode: str = ANNOTATION_PREPROCESS_LEGACY,
) -> tuple[Optional[tuple[int, int, int, int]], int]:
    if str(box.get("study_level", "")).strip().lower() == "yes":
        return None, 0

    x = box.get("x")
    y = box.get("y")
    w = box.get("width")
    h = box.get("height")
    if None in (x, y, w, h):
        return None, 0

    box_base_size = int(box.get("base_size", base_size) or base_size)

    def _center_crop_or_pad_binary_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        src_h, src_w = mask.shape
        out = np.zeros((target_h, target_w), dtype=bool)

        copy_h = min(src_h, target_h)
        copy_w = min(src_w, target_w)

        src_h0 = max((src_h - copy_h) // 2, 0)
        src_w0 = max((src_w - copy_w) // 2, 0)
        dst_h0 = max((target_h - copy_h) // 2, 0)
        dst_w0 = max((target_w - copy_w) // 2, 0)

        out[dst_h0:dst_h0 + copy_h, dst_w0:dst_w0 + copy_w] = mask[src_h0:src_h0 + copy_h, src_w0:src_w0 + copy_w]
        return out

    def _resize_binary_mask_nearest(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        if mask.shape == (out_h, out_w):
            return mask
        tensor = torch.from_numpy(mask.astype(np.float32, copy=False))[None, None]
        resized = F.interpolate(tensor, size=(out_h, out_w), mode="nearest")
        return resized[0, 0].cpu().numpy() > 0.5

    if preprocess_mode == ANNOTATION_PREPROCESS_MASK_PIPELINE:
        mask = np.zeros((box_base_size, box_base_size), dtype=bool)
        x1_src = int(np.floor(float(x)))
        y1_src = int(np.floor(float(y)))
        x2_src = int(np.ceil(float(x) + float(w)))
        y2_src = int(np.ceil(float(y) + float(h)))

        x1_src = int(np.clip(x1_src, 0, box_base_size))
        y1_src = int(np.clip(y1_src, 0, box_base_size))
        x2_src = int(np.clip(x2_src, 0, box_base_size))
        y2_src = int(np.clip(y2_src, 0, box_base_size))
        if x2_src <= x1_src or y2_src <= y1_src:
            return None, 0

        mask[y1_src:y2_src, x1_src:x2_src] = True

        mask = _center_crop_or_pad_binary_mask(mask, ANNOTATION_BASE_SIZE, ANNOTATION_BASE_SIZE)
        mask = _resize_binary_mask_nearest(mask, image_height, image_width)

        if flip_horizontal:
            mask = np.fliplr(mask)
        if flip_vertical:
            mask = np.flipud(mask)

        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            return None, 0

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1

        x1 = int(np.clip(x1, 0, image_width))
        x2 = int(np.clip(x2, 0, image_width))
        y1 = int(np.clip(y1, 0, image_height))
        y2 = int(np.clip(y2, 0, image_height))
        if x2 <= x1 or y2 <= y1:
            return None, 0

        return (x1, y1, x2, y2), int((x2 - x1) * (y2 - y1))

    if preprocess_mode == ANNOTATION_PREPROCESS_RENDER_FASTMRI:
        y = float(box_base_size) - (float(y) + float(h))

    scale_x = float(image_width) / float(box_base_size)
    scale_y = float(image_height) / float(box_base_size)

    x1 = int(np.floor(float(x) * scale_x))
    y1 = int(np.floor(float(y) * scale_y))
    x2 = int(np.ceil((float(x) + float(w)) * scale_x))
    y2 = int(np.ceil((float(y) + float(h)) * scale_y))

    x1 = int(np.clip(x1, 0, image_width))
    x2 = int(np.clip(x2, 0, image_width))
    y1 = int(np.clip(y1, 0, image_height))
    y2 = int(np.clip(y2, 0, image_height))

    if flip_horizontal:
        x1, x2 = int(image_width - x2), int(image_width - x1)
    if flip_vertical:
        y1, y2 = int(image_height - y2), int(image_height - y1)

    x1 = int(np.clip(x1, 0, image_width))
    x2 = int(np.clip(x2, 0, image_width))
    y1 = int(np.clip(y1, 0, image_height))
    y2 = int(np.clip(y2, 0, image_height))
    if x2 <= x1 or y2 <= y1:
        return None, 0

    return (x1, y1, x2, y2), int((x2 - x1) * (y2 - y1))



def _build_single_bbox_mask(
    box: dict,
    image_height: int,
    image_width: int,
    base_size: int = ANNOTATION_BASE_SIZE,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
    preprocess_mode: str = ANNOTATION_PREPROCESS_LEGACY,
) -> tuple[Optional[np.ndarray], int]:
    rect_xyxy, _ = _project_box_to_image_rect(
        box,
        image_height=image_height,
        image_width=image_width,
        base_size=base_size,
        flip_vertical=flip_vertical,
        flip_horizontal=flip_horizontal,
        preprocess_mode=preprocess_mode,
    )
    if rect_xyxy is None:
        return None, 0
    x1, y1, x2, y2 = rect_xyxy

    mask = np.zeros((image_height, image_width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask, int(mask.sum())



def compute_bbox_detection_metrics(
    predicted_anomaly_mask: np.ndarray,
    boxes: list[dict],
    anomaly_score_map: Optional[np.ndarray] = None,
    tp_inside_ratio_threshold: float = TP_INSIDE_RATIO_THRESHOLD,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
    preprocess_mode: str = ANNOTATION_PREPROCESS_LEGACY,
) -> dict:
    pred_mask = predicted_anomaly_mask.astype(bool)
    h, w = pred_mask.shape
    highlighted_pixels = int(pred_mask.sum())
    if anomaly_score_map is None:
        score_map = pred_mask.astype(np.float32)
    else:
        score_map = np.asarray(anomaly_score_map, dtype=np.float32)
        if score_map.shape != pred_mask.shape:
            score_map = pred_mask.astype(np.float32)
    score_map = np.nan_to_num(score_map, nan=0.0, posinf=0.0, neginf=0.0)
    highlighted_score_sum = float(score_map.sum())

    valid_box_entries = []
    for box_idx, box in enumerate(boxes):
        box_mask, box_pixels = _build_single_bbox_mask(
            box,
            image_height=h,
            image_width=w,
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
            preprocess_mode=preprocess_mode,
        )
        if box_mask is None or box_pixels <= 0:
            continue
        valid_box_entries.append({
            "box_index": int(box_idx),
            "label": str(box.get("label", "") or ""),
            "mask": box_mask,
            "pixels": int(box_pixels),
        })

    valid_boxes = len(valid_box_entries)
    if valid_boxes == 0:
        return {
            "has_ground_truth_bbox": False,
            "num_ground_truth_boxes": 0,
            "num_bbox_evaluations": 0,
            "num_true_positive_bboxes": 0,
            "ground_truth_bbox_pixels": 0,
            "highlighted_anomaly_pixels_binary_token": highlighted_pixels,
            "highlighted_anomaly_score_sum": highlighted_score_sum,
            "predicted_anomaly_pixels_inside_bbox": 0,
            "predicted_anomaly_pixels_outside_bbox": int(pred_mask.sum()),
            "predicted_anomaly_pixels_inside_other_bboxes_excluded": 0,
            "anomaly_score_sum_inside_bbox": 0.0,
            "anomaly_score_sum_outside_bbox": float(highlighted_score_sum),
            "anomaly_score_sum_inside_other_bboxes_excluded": 0.0,
            "inside_bbox_detection_ratio": None,
            "true_positive": None,
            "false_positive_ratio": None,
            "precision": None,
            "f1_score": None,
            "bbox_evaluation_mode": "per_pathology_bbox",
            "bbox_evaluations": [],
            "tp_inside_ratio_threshold": float(tp_inside_ratio_threshold),
        }

    per_bbox_metrics = []
    sum_gt_pixels = 0
    sum_pred_inside = 0
    sum_pred_outside_healthy = 0
    sum_pred_inside_other_bboxes = 0
    sum_score_inside = 0.0
    sum_score_outside_healthy = 0.0
    sum_score_inside_other_bboxes = 0.0
    tp_count = 0

    for eval_idx, target in enumerate(valid_box_entries):
        target_mask = target["mask"]
        other_boxes_mask = np.zeros((h, w), dtype=bool)
        for other_idx, other in enumerate(valid_box_entries):
            if other_idx == eval_idx:
                continue
            other_boxes_mask |= other["mask"]

        healthy_mask = np.logical_not(np.logical_or(target_mask, other_boxes_mask))
        gt_pixels = int(target["pixels"])
        pred_inside = int(np.logical_and(pred_mask, target_mask).sum())
        pred_outside_healthy = int(np.logical_and(pred_mask, healthy_mask).sum())
        pred_inside_other = int(np.logical_and(pred_mask, other_boxes_mask).sum())
        score_inside = float(score_map[target_mask].sum())
        score_outside_healthy = float(score_map[healthy_mask].sum())
        score_inside_other = float(score_map[other_boxes_mask].sum())
        inside_ratio = float(pred_inside) / float(gt_pixels) if gt_pixels > 0 else 0.0
        tp_flag = bool(inside_ratio >= float(tp_inside_ratio_threshold))

        fp_ratio = None
        if pred_inside > 0:
            fp_ratio = float(pred_outside_healthy) / float(pred_inside)

        precision = 0.0
        tp_value = 1.0 if tp_flag else 0.0
        if fp_ratio is not None:
            denom = tp_value + fp_ratio
            precision = float(tp_value / denom) if denom > 0 else 0.0
        f1_score = float((2.0 * precision) / (precision + 1.0)) if tp_flag else 0.0

        per_bbox_metrics.append({
            "bbox_eval_index": int(eval_idx),
            "box_index": int(target["box_index"]),
            "label": target["label"],
            "ground_truth_bbox_pixels": int(gt_pixels),
            "predicted_anomaly_pixels_inside_bbox": int(pred_inside),
            "predicted_anomaly_pixels_outside_bbox_healthy": int(pred_outside_healthy),
            "predicted_anomaly_pixels_inside_other_bboxes_excluded": int(pred_inside_other),
            "anomaly_score_sum_inside_bbox": float(score_inside),
            "anomaly_score_sum_outside_bbox_healthy": float(score_outside_healthy),
            "anomaly_score_sum_inside_other_bboxes_excluded": float(score_inside_other),
            "inside_bbox_detection_ratio": float(inside_ratio),
            "true_positive": bool(tp_flag),
            "false_positive_ratio": fp_ratio,
            "precision": float(precision),
            "f1_score": float(f1_score),
            "tp_inside_ratio_threshold": float(tp_inside_ratio_threshold),
        })

        sum_gt_pixels += int(gt_pixels)
        sum_pred_inside += int(pred_inside)
        sum_pred_outside_healthy += int(pred_outside_healthy)
        sum_pred_inside_other_bboxes += int(pred_inside_other)
        sum_score_inside += float(score_inside)
        sum_score_outside_healthy += float(score_outside_healthy)
        sum_score_inside_other_bboxes += float(score_inside_other)
        if tp_flag:
            tp_count += 1

    aggregate_inside_ratio = float(np.mean([m["inside_bbox_detection_ratio"] for m in per_bbox_metrics])) if per_bbox_metrics else None
    aggregate_tp = bool(tp_count > 0)
    aggregate_tp_value = 1.0 if aggregate_tp else 0.0
    aggregate_fp_ratio = None
    if sum_pred_inside > 0:
        aggregate_fp_ratio = float(sum_pred_outside_healthy) / float(sum_pred_inside)

    aggregate_precision = 0.0
    if aggregate_fp_ratio is not None:
        denom = aggregate_tp_value + aggregate_fp_ratio
        aggregate_precision = float(aggregate_tp_value / denom) if denom > 0 else 0.0
    aggregate_f1 = float((2.0 * aggregate_precision) / (aggregate_precision + 1.0)) if aggregate_tp else 0.0

    return {
        "has_ground_truth_bbox": True,
        "num_ground_truth_boxes": int(valid_boxes),
        "num_bbox_evaluations": int(valid_boxes),
        "num_true_positive_bboxes": int(tp_count),
        "ground_truth_bbox_pixels": int(sum_gt_pixels),
        "highlighted_anomaly_pixels_binary_token": highlighted_pixels,
        "highlighted_anomaly_score_sum": float(highlighted_score_sum),
        "predicted_anomaly_pixels_inside_bbox": int(sum_pred_inside),
        "predicted_anomaly_pixels_outside_bbox": int(sum_pred_outside_healthy),
        "predicted_anomaly_pixels_inside_other_bboxes_excluded": int(sum_pred_inside_other_bboxes),
        "anomaly_score_sum_inside_bbox": float(sum_score_inside),
        "anomaly_score_sum_outside_bbox": float(sum_score_outside_healthy),
        "anomaly_score_sum_inside_other_bboxes_excluded": float(sum_score_inside_other_bboxes),
        "inside_bbox_detection_ratio": aggregate_inside_ratio,
        "true_positive": aggregate_tp,
        "false_positive_ratio": aggregate_fp_ratio,
        "precision": aggregate_precision,
        "f1_score": aggregate_f1,
        "bbox_evaluation_mode": "per_pathology_bbox",
        "bbox_evaluations": per_bbox_metrics,
        "tp_inside_ratio_threshold": float(tp_inside_ratio_threshold),
    }

# -----------------------------------------------------------------------------
# Sanity-check / comparison figures  (AYNU)
# -----------------------------------------------------------------------------


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

# -----------------------------------------------------------------------------
# Per-sample analysis figures  (AYNU)
# -----------------------------------------------------------------------------


# =============================================================================
# Visualization with Z-Score Display
# =============================================================================

def visualize_v4_zscore(results: Dict, sample_idx: int = 0, title: str = "", save_path: str = None,
                        mask_threshold_percentile: float = 97.0, clamp_threshold: float = 0.60,
                        first_heatmap_sum_thresh: float = 300.0,
                        binary_mask_threshold: float = 0.10,
                        binary_mask_iteration: int = 0,
                        binary_include_token_surprisal: bool = False,
                        binary_token_surprisal_threshold: float = 0.0,
                        lpips_rec_inp_threshold_back_to_binary_token_map: Union[float, Tuple[float, float]] = (0.0, 0.585),
                        binary_edge_erosion_iters: int = 1,
                        binary_center_protect_radius_ratio: float = 0.40,
                        binary_edge_erosion_kernel: int = 13,
                        annotation_boxes: Optional[dict[str, dict[int, list[dict]]]] = None,
                        file_stem: Optional[str] = None,
                        slice_idx: Optional[int] = None,
                        annotation_focus_label: Optional[str] = None,
                        annotation_flip_vertical: bool = False,
                        annotation_flip_horizontal: bool = False,
                        annotation_preprocess_mode: str = ANNOTATION_PREPROCESS_LEGACY,
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
    extra_lpips_rec_inp = 1
    extra_legend = 1
    n_cols = base_cols + extra_z + extra_surprisal + extra_guard + extra_binary + extra_lpips_rec_inp + extra_legend
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
            f"Z-threshold: {format_z_threshold(results['z_threshold'])}\n\n"
            "Formula:\n"
            "Z = (LPIPS - μ) / (σ + ε)\n\n"
            "Interpretation:\n"
            "• Z outside threshold rule → Anomaly\n"
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
    
    # Binary mask from masked score with optional token surprisal fusion
    ax = fig.add_subplot(gs[row, col]); col += 1
    iter_idx = int(np.clip(binary_mask_iteration, 0, num_iters - 1))
    masked_score_tensor = iteration_history[iter_idx]["heatmap"] * iteration_history[iter_idx]["mask"]
    masked_score_np = to_np(masked_score_tensor)
    binary_mask_for_lpips = None
    if masked_score_np is not None:
        binary_mask_base = (masked_score_np > binary_mask_threshold)
        binary_mask_show = binary_mask_base.copy()

        token_hot_px = 0
        overlap_px = 0
        if binary_include_token_surprisal and results.get("token_surprisal_map") is not None:
            surprisal_np = to_np(results["token_surprisal_map"])
            token_binary = (surprisal_np > float(binary_token_surprisal_threshold))
            token_hot_px = int(token_binary.sum())
            overlap_px = int(np.logical_and(binary_mask_base, token_binary).sum())
            binary_mask_show = np.logical_or(binary_mask_base, token_binary)

        lpips_backflow_px = 0
        lpips_rec_inp_np = to_np(iteration_history[iter_idx]["lpips_recon_inpainted"])
        if lpips_rec_inp_np is not None:
            lpips_backflow_mask, _ = build_lpips_backflow_mask(
                lpips_rec_inp_np,
                lpips_rec_inp_threshold_back_to_binary_token_map,
            )
            lpips_backflow_px = int(lpips_backflow_mask.sum())
            binary_mask_show = np.logical_or(binary_mask_show, lpips_backflow_mask)

        binary_mask_show = apply_edge_to_center_erosion(
            binary_mask_show,
            max_edge_erosion_iters=binary_edge_erosion_iters,
            center_protect_radius_ratio=binary_center_protect_radius_ratio,
            erosion_kernel_size=binary_edge_erosion_kernel,
        )

        binary_sum = float(binary_mask_show.sum())
        binary_mask_for_lpips = binary_mask_show
        im = ax.imshow(binary_mask_show.astype(np.float32), cmap='gray', vmin=0, vmax=1)

        binary_box_count = 0
        if annotation_boxes is not None and file_stem is not None and slice_idx is not None:
            boxes = annotation_boxes.get(file_stem, {}).get(slice_idx, [])
            if boxes:
                h, w = binary_mask_show.shape
                binary_box_count = draw_annotation_boxes(
                    ax,
                    boxes,
                    image_height=h,
                    image_width=w,
                    color="yellow",
                    focus_label=annotation_focus_label,
                    flip_vertical=annotation_flip_vertical,
                    flip_horizontal=annotation_flip_horizontal,
                    preprocess_mode=annotation_preprocess_mode,
                )

        box_suffix = f" | boxes={binary_box_count}" if binary_box_count else ""
        if binary_include_token_surprisal and results.get("token_surprisal_map") is not None:
            ax.set_title(
                f"Binary+Token (iter {iter_idx})\nΣ={binary_sum:.0f} | T={token_hot_px} | ∩={overlap_px} | L={lpips_backflow_px}{box_suffix}",
                fontsize=12,
                fontweight='bold',
            )
        else:
            ax.set_title(
                f"Binary Sum (iter {iter_idx}, > {binary_mask_threshold:.2f})\nΣ={binary_sum:.0f} | L={lpips_backflow_px}{box_suffix}",
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
    vmax_lpips_rec_inp = max(h["lpips_recon_inpainted"][sample_idx].max().item() for h in iteration_history)
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
                im = ax.imshow(zscore_np, cmap=HEATMAP_CMAP, vmin=-vmax_z, vmax=vmax_z)
                ax.set_title(f"Z-SCORE Map\nThreshold: {results['z_threshold']}", 
                            fontsize=10, color='purple', fontweight='bold')
                # Mark threshold contour
                zthr = results["z_threshold"]
                if isinstance(zthr, (tuple, list)) and len(zthr) == 2:
                    ax.contour(zscore_np, levels=[float(zthr[0])], colors='cyan', linewidths=2)
                    ax.contour(zscore_np, levels=[float(zthr[1])], colors='lime', linewidths=2)
                else:
                    ax.contour(zscore_np, levels=[float(zthr)], colors='lime', linewidths=2)
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
        if iter_idx == 0 and use_zscore:
            method = f"Z:{format_z_threshold(results['z_threshold'])}"
        else:
            method = f"p{mask_threshold_percentile}"
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

        # LPIPS(Reconstruction, Inpainted)
        ax = fig.add_subplot(gs[row, col])
        lpips_rec_inp_np = to_np(iter_data["lpips_recon_inpainted"])
        im = ax.imshow(lpips_rec_inp_np, cmap=HEATMAP_CMAP, vmin=0, vmax=vmax_lpips_rec_inp)
        ax.set_title(f"LPIPS(Rec, Inp)\nmax={lpips_rec_inp_np.max():.3f}", fontsize=10)
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
            zthr = results["z_threshold"]
            if isinstance(zthr, (tuple, list)) and len(zthr) == 2:
                low_thr = float(zthr[0])
                high_thr = float(zthr[1])
                ax.axvline(low_thr, color='cyan', linestyle='--', linewidth=2, label=f'Z={low_thr:g}')
                ax.axvline(high_thr, color='red', linestyle='--', linewidth=2, label=f'Z={high_thr:g}')
                flagged_pct = ((zscore_np < low_thr) | (zscore_np > high_thr)).mean() * 100
                title_txt = f'Z-Score Distribution\nPixels outside [{low_thr:g}, {high_thr:g}]: {flagged_pct:.1f}%'
            else:
                thr = float(zthr)
                ax.axvline(thr, color='red', linestyle='--', linewidth=2, label=f'Z={thr:g}')
                flagged_pct = (zscore_np > thr).mean() * 100
                title_txt = f'Z-Score Distribution\nPixels > {thr:g}: {flagged_pct:.1f}%'
            ax.set_xlabel('Z-score')
            ax.set_ylabel('Pixel count')
            ax.set_title(title_txt)
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
        final_lpips = to_np(results["lpips_recon_inpainted"])
        p_lpips = 60
        p_cut = np.percentile(final_lpips, p_lpips)
        final_lpips_clamped = np.where(final_lpips >= p_cut, final_lpips, np.nan)

        final_lpips_for_display = final_lpips_clamped.copy()
        if (
            binary_include_token_surprisal
            and binary_mask_for_lpips is not None
            and results.get("token_surprisal_map") is not None
        ):
            final_lpips_dense = np.nan_to_num(final_lpips_for_display, nan=0.0)
            lpips_norm = np.clip((final_lpips_dense - 0.5) / 0.5, 0.0, 1.0)

            binary_soft = ndimage.gaussian_filter(binary_mask_for_lpips.astype(np.float32), sigma=1.2)
            if binary_soft.max() > 0:
                binary_soft = binary_soft / (binary_soft.max() + 1e-8)

            fused_norm = 0.55 * lpips_norm + 0.45 * np.maximum(lpips_norm, binary_soft)
            fused_norm = np.clip(fused_norm, 0.0, 1.0)
            final_lpips_for_display = np.where(fused_norm > 0, 0.5 + 0.5 * fused_norm, np.nan)

        final_lpips_for_display = np.where(final_lpips_for_display > 0.5, final_lpips_for_display, np.nan)

        ax.imshow(to_np(results["input"]), cmap='gray', alpha=1)
        heatmap_cmap = plt.cm.get_cmap(HEATMAP_CMAP).copy()
        heatmap_cmap.set_bad(alpha=0)
        lpips_alpha = np.zeros_like(final_lpips_for_display, dtype=np.float32)
        lpips_valid = np.isfinite(final_lpips_for_display)
        lpips_alpha[lpips_valid] = 0.25 + 0.45 * np.clip((final_lpips_for_display[lpips_valid] - 0.5) / 0.5, 0.0, 1.0)
        im = ax.imshow(final_lpips_for_display, cmap=heatmap_cmap, alpha=lpips_alpha, vmin=0.5, vmax=1.0)
        if binary_include_token_surprisal and binary_mask_for_lpips is not None:
            ax.set_title(
                f"FINAL LPIPS + Binary+Token (>0.5)\nmax={np.nanmax(final_lpips_for_display):.3f}",
                fontsize=11,
                fontweight='bold'
            )
        else:
            ax.set_title(
                f"FINAL LPIPS (>= p{p_lpips}, >0.5)\nmax={np.nanmax(final_lpips_for_display):.3f}",
                fontsize=11,
                fontweight='bold'
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
    
    plt.suptitle(
        f"Recursive AutoMask V4 {'(Z-Score Calibrated)' if use_zscore else ''}\n{title}",
        fontsize=14,
        fontweight='bold',
        y=0.995,
    )
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.94], pad=1.2)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if save_token_surprisal_overlay and results.get("token_surprisal_map") is not None:
            base_lpips = to_np(results["lpips_recon_inpainted"])
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
    binary_mask_threshold: float = 0.10,
    binary_mask_iteration: int = 0,
    binary_include_token_surprisal: bool = False,
    binary_token_surprisal_threshold: float = 0.0,
    lpips_rec_inp_threshold_back_to_binary_token_map: Union[float, Tuple[float, float]] = (0.0, 0.585),
    binary_edge_erosion_iters: int = 1,
    binary_center_protect_radius_ratio: float = 0.40,
    binary_edge_erosion_kernel: int = 13,
    binary_token_boost_value: float = 0.95,
    annotation_boxes: Optional[dict[str, dict[int, list[dict]]]] = None,
    file_stem: Optional[str] = None,
    slice_idx: Optional[int] = None,
    annotation_focus_label: Optional[str] = None,
    annotation_flip_vertical: bool = False,
    annotation_flip_horizontal: bool = False,
    annotation_preprocess_mode: str = ANNOTATION_PREPROCESS_LEGACY,
    detected_based_on_thresholds: Optional[bool] = None,
    binary_token_white_pixel_ratio: Optional[float] = None,
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
    
    # Binary+Token comparison mask for consistent visualization with full figure
    iter_sel = len(iteration_history) - 1 if binary_mask_iteration < 0 else binary_mask_iteration
    iter_sel = max(0, min(iter_sel, len(iteration_history) - 1))
    hmap_sel = iteration_history[iter_sel]["heatmap"][sample_idx, 0].cpu().numpy()
    mask_sel = iteration_history[iter_sel]["mask"][sample_idx, 0].cpu().numpy()
    masked_score_sel = hmap_sel * mask_sel
    binary_mask_base = masked_score_sel > binary_mask_threshold
    binary_mask_show = binary_mask_base.copy()
    token_hot_px = 0
    overlap_px = 0
    if binary_include_token_surprisal and results.get("token_surprisal_map") is not None:
        token_np = to_np(results.get("token_surprisal_map"))
        token_binary = token_np > float(binary_token_surprisal_threshold)
        token_hot_px = int(token_binary.sum())
        overlap_px = int(np.logical_and(binary_mask_base, token_binary).sum())
        binary_mask_show = np.logical_or(binary_mask_base, token_binary)

    lpips_backflow_px = 0
    lpips_sel = iteration_history[iter_sel]["lpips_recon_inpainted"][sample_idx, 0].cpu().numpy()
    lpips_backflow_mask, _ = build_lpips_backflow_mask(
        lpips_sel,
        lpips_rec_inp_threshold_back_to_binary_token_map,
    )
    lpips_backflow_px = int(lpips_backflow_mask.sum())
    binary_mask_show = np.logical_or(binary_mask_show, lpips_backflow_mask)

    binary_mask_pre_erosion = binary_mask_show.copy()

    binary_mask_show = apply_edge_to_center_erosion(
        binary_mask_show,
        max_edge_erosion_iters=binary_edge_erosion_iters,
        center_protect_radius_ratio=binary_center_protect_radius_ratio,
        erosion_kernel_size=binary_edge_erosion_kernel,
    )

    binary_removed_by_erosion = np.logical_and(binary_mask_pre_erosion, np.logical_not(binary_mask_show))
    bin_sum_pre = int(binary_mask_pre_erosion.sum())
    bin_sum_post = int(binary_mask_show.sum())
    bin_removed = int(binary_removed_by_erosion.sum())

    bin_sum = int(binary_mask_show.sum())
    bin_ratio_from_plot = float(bin_sum) / float(binary_mask_show.size) if binary_mask_show.size > 0 else 0.0
    bin_ratio_to_show = (
        float(binary_token_white_pixel_ratio)
        if binary_token_white_pixel_ratio is not None
        else float(bin_ratio_from_plot)
    )

    # Create figure without sharpness map subplot
    n_cols = 7
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    title_pad = 16
    
    # 1. Input Image
    input_img = to_np(results["input"])
    vmin = np.percentile(input_img, 0.1)
    vmax = np.percentile(input_img, 99.0)

    axes[0].imshow(input_img, cmap='gray', alpha=1, vmin=vmin, vmax=vmax)
    axes[0].set_title("Input", fontsize=14, fontweight='bold', pad=title_pad)
    axes[0].axis('off')
    
    # 2. Reconstruction Image
    healed_img = to_np(results["healed"])
    #recon_img = to_np(results["reconstruction"])
    #
    axes[1].imshow(healed_img, cmap='gray')
    axes[1].set_title("Healed", fontsize=14, fontweight='bold', pad=title_pad)
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
        pad=title_pad,
    )
    axes[2].axis('off')
    plt.colorbar(im_overlay, ax=axes[2], fraction=0.046, pad=0.04)

    # 6. Final LPIPS heatmap (clamped at p75) overlaid on input
    final_lpips = to_np(results.get("lpips_recon_inpainted"))
    if final_lpips is not None:
        display_floor = 0.3
        p = 75
        p_cut = np.percentile(final_lpips, p)
        final_lpips_clamped = np.where(final_lpips >= p_cut, final_lpips, np.nan)
        final_lpips_display = final_lpips_clamped.copy()
        lpips_contrast = np.zeros_like(final_lpips, dtype=np.float32)
        if binary_include_token_surprisal:
            final_lpips_dense = np.nan_to_num(final_lpips_display, nan=0.0)
            lpips_norm = np.clip((final_lpips_dense - display_floor) / (1.0 - display_floor), 0.0, 1.0)

            binary_soft = ndimage.gaussian_filter(binary_mask_show.astype(np.float32), sigma=1.2)
            if binary_soft.max() > 0:
                binary_soft = binary_soft / (binary_soft.max() + 1e-8)

            token_weight = float(np.clip(binary_token_boost_value - 0.5, 0.0, 1.0))
            fused_norm = (1.0 - token_weight) * lpips_norm + token_weight * np.maximum(lpips_norm, binary_soft)
            fused_norm = np.clip(fused_norm, 0.0, 1.0)

            final_lpips_display = np.where(fused_norm > 0, display_floor + (1.0 - display_floor) * fused_norm, np.nan)

        final_lpips_display = np.where(final_lpips_display > display_floor, final_lpips_display, np.nan)

        # Contrast stretch + nonlinear brightening so top anomalies stand out clearly.
        valid_mask = np.isfinite(final_lpips_display)
        valid_vals = final_lpips_display[valid_mask]
        if valid_vals.size > 0:
            lo = np.percentile(valid_vals, 60)
            hi = np.percentile(valid_vals, 99.5)
            if hi <= lo:
                hi = lo + 1e-6
            lpips_contrast = np.clip((final_lpips_display - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)
            lpips_contrast = np.power(lpips_contrast, 0.6)  # gamma < 1 boosts bright regions
            final_lpips_display = np.where(valid_mask, display_floor + (1.0 - display_floor) * lpips_contrast, np.nan)

        # Hide floor-level values (e.g. 0.3) to avoid dark stain artifacts in overlay.
        overlay_visible = np.isfinite(final_lpips_display) & (lpips_contrast > 1e-6)
        final_lpips_display = np.where(overlay_visible, final_lpips_display, np.nan)

        axes[3].imshow(input_img, cmap='gray', alpha=1, vmin=vmin, vmax=vmax)
        heatmap_cmap = plt.cm.get_cmap(HEATMAP_CMAP).copy()
        heatmap_cmap.set_bad(alpha=0)
        lpips_alpha = np.zeros_like(final_lpips_display, dtype=np.float32)
        lpips_alpha[overlay_visible] = 0.35 + 0.60 * lpips_contrast[overlay_visible]
        im_lpips = axes[3].imshow(final_lpips_display, cmap=heatmap_cmap, alpha=lpips_alpha, vmin=display_floor, vmax=1.0)

        box_count = 0
        if annotation_boxes is not None and file_stem is not None and slice_idx is not None:
            boxes = annotation_boxes.get(file_stem, {}).get(slice_idx, [])
            if boxes:
                h, w = input_img.shape
                box_count = draw_annotation_boxes(
                    axes[3],
                    boxes,
                    image_height=h,
                    image_width=w,
                    color="yellow",
                    focus_label=annotation_focus_label,
                    flip_vertical=annotation_flip_vertical,
                    flip_horizontal=annotation_flip_horizontal,
                    preprocess_mode=annotation_preprocess_mode,
                )

        title_suffix = f"\nboxes={box_count}" if box_count else ""
        if binary_include_token_surprisal:
            axes[3].set_title(
                f"Final LPIPS + Binary+Token (>{display_floor:.1f}){title_suffix}",
                fontsize=13,
                fontweight='bold',
                color='darkblue',
                pad=title_pad,
            )
        else:
            axes[3].set_title(
                f"Final LPIPS Overlay (>= p{p}, >{display_floor:.1f}){title_suffix}",
                fontsize=13,
                fontweight='bold',
                color='darkblue',
                pad=title_pad,
            )
        axes[3].axis('off')
        plt.colorbar(im_lpips, ax=axes[3], fraction=0.046, pad=0.04)
    else:
        axes[3].text(0.5, 0.5, "No final LPIPS", ha='center', va='center', fontsize=12)
        axes[3].axis('off')

    # Final LPIPS raw heatmap
    if final_lpips is not None:
        im_raw = axes[4].imshow(final_lpips, cmap=HEATMAP_CMAP)
        axes[4].set_title(
            f"Final LPIPS (raw)\nmax={np.nanmax(final_lpips):.3f}",
            fontsize=13,
            fontweight='bold',
            pad=title_pad,
        )
        axes[4].axis('off')
        plt.colorbar(im_raw, ax=axes[4], fraction=0.046, pad=0.04)
    else:
        axes[4].text(0.5, 0.5, "No final LPIPS", ha='center', va='center', fontsize=12)
        axes[4].axis('off')

    # Binary+Token comparison panel
    im_bin = axes[5].imshow(binary_mask_show.astype(np.float32), cmap='gray', vmin=0, vmax=1)
    binary_box_count = 0
    if annotation_boxes is not None and file_stem is not None and slice_idx is not None:
        boxes = annotation_boxes.get(file_stem, {}).get(slice_idx, [])
        if boxes:
            h, w = binary_mask_show.shape
            binary_box_count = draw_annotation_boxes(
                axes[5],
                boxes,
                image_height=h,
                image_width=w,
                color="yellow",
                focus_label=annotation_focus_label,
                flip_vertical=annotation_flip_vertical,
                flip_horizontal=annotation_flip_horizontal,
                preprocess_mode=annotation_preprocess_mode,
            )
    box_suffix = f" | boxes={binary_box_count}" if binary_box_count else ""
    if binary_include_token_surprisal and results.get("token_surprisal_map") is not None:
        axes[5].set_title(
            f"Binary+Token map\niter={iter_sel}, Σ={bin_sum}, T={token_hot_px}, ∩={overlap_px}, L={lpips_backflow_px}, ratio={bin_ratio_to_show * 100.0:.2f}%{box_suffix}",
            fontsize=13,
            fontweight='bold',
            color='black',
            pad=title_pad,
        )
    else:
        axes[5].set_title(
            f"Binary map\niter={iter_sel}, Σ={bin_sum}, L={lpips_backflow_px}, ratio={bin_ratio_to_show * 100.0:.2f}%{box_suffix}",
            fontsize=13,
            fontweight='bold',
            color='black',
            pad=title_pad,
        )
    axes[5].axis('off')
    plt.colorbar(im_bin, ax=axes[5], fraction=0.046, pad=0.04)

    # Binary+Token erosion effect panel
    erosion_effect_rgb = np.zeros((*binary_mask_show.shape, 3), dtype=np.float32)
    erosion_effect_rgb[..., 1] = binary_mask_show.astype(np.float32)            # kept (green)
    erosion_effect_rgb[..., 0] = binary_removed_by_erosion.astype(np.float32)   # removed (red)

    axes[6].imshow(input_img, cmap='gray', alpha=0.35, vmin=vmin, vmax=vmax)
    axes[6].imshow(erosion_effect_rgb, alpha=0.75, vmin=0, vmax=1)
    axes[6].set_title(
        f"Binary+Token Erosion Effect\npre={bin_sum_pre}, post={bin_sum_post}, removed={bin_removed}",
        fontsize=12,
        fontweight='bold',
        color='darkred' if bin_removed > 0 else 'darkgreen',
        pad=title_pad,
    )
    axes[6].axis('off')

    # Sharpness map disabled per request
    
    # Add overall title with detection summary
    if detected_based_on_thresholds is True:
        detection_text = "DETECTED"
        detection_color = "green"
    elif detected_based_on_thresholds is False:
        detection_text = "NOT DETECTED"
        detection_color = "red"
    else:
        detection_text = "N/A (no bbox)"
        detection_color = "black"
    ratio_percent = bin_ratio_to_show * 100.0

    plt.suptitle(f"{title}", fontsize=14, fontweight='bold', y=0.992)
    fig.text(
        0.5,
        0.952,
        f"detection: {detection_text} | Binary+Token white-pixel ratio={ratio_percent:.2f}%",
        ha='center',
        va='center',
        fontsize=13,
        fontweight='bold',
        color=detection_color,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.8),
    )
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.86], pad=1.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig

