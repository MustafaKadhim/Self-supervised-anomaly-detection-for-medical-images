"""
Visualization utilities for anomaly maps and results.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch


def _to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def plot_anomaly_map(
    image: Union[torch.Tensor, np.ndarray],
    reconstruction: Union[torch.Tensor, np.ndarray],
    anomaly_map: Union[torch.Tensor, np.ndarray],
    score: Optional[float] = None,
    title: str = "",
    save_path: Optional[str] = None,
    colormap: str = "hot",
) -> plt.Figure:
    """
    Plot the input image, its reconstruction, and the anomaly heatmap side by side.

    Args:
        image: Original input image tensor/array (C, H, W) or (H, W).
        reconstruction: Reconstructed image tensor/array (same shape as image).
        anomaly_map: Per-pixel anomaly score map (C, H, W) or (H, W).
        score: Optional scalar anomaly score displayed in the title.
        title: Optional super-title for the figure.
        save_path: If provided, save the figure to this path.
        colormap: Matplotlib colormap name for the anomaly heatmap.

    Returns:
        The created ``matplotlib.figure.Figure``.
    """
    image = _to_numpy(image)
    reconstruction = _to_numpy(reconstruction)
    anomaly_map = _to_numpy(anomaly_map)

    # Squeeze channel dimension for grayscale
    def _squeeze(arr):
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            return arr.squeeze(0) if arr.shape[0] == 1 else arr.transpose(1, 2, 0)
        return arr

    image = _squeeze(image)
    reconstruction = _squeeze(reconstruction)
    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    cmap_gray = "gray" if image.ndim == 2 else None

    axes[0].imshow(np.clip(image, 0, 1), cmap=cmap_gray)
    axes[0].set_title("Input Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(np.clip(reconstruction, 0, 1), cmap=cmap_gray)
    axes[1].set_title("Reconstruction", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    im = axes[2].imshow(anomaly_map, cmap=colormap, vmin=0, vmax=anomaly_map.max() + 1e-8)
    score_str = f" (score={score:.4f})" if score is not None else ""
    axes[2].set_title(f"Anomaly Map{score_str}", fontsize=13, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_roc_curve(
    fprs: Sequence[float],
    tprs: Sequence[float],
    auroc: float,
    experiment_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the ROC curve for an anomaly detection experiment."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fprs, tprs, color="#E74C3C", lw=2, label=f"ROC (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {experiment_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training and validation loss curves over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Total", color="#3498DB")
    if any(not np.isnan(v) for v in history["val_loss"]):
        axes[0].plot(epochs, history["val_loss"], label="Val Total", color="#E74C3C", linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss", fontweight="bold"); axes[0].legend(); axes[0].grid(alpha=0.3)

    if "train_l1" in history and "train_ssim" in history:
        axes[1].plot(epochs, history["train_l1"], label="L1", color="#2ECC71")
        axes[1].plot(epochs, history["train_ssim"], label="SSIM", color="#9B59B6")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
        axes[1].set_title("Loss Components", fontweight="bold")
        axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
