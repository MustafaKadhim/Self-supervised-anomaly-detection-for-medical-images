import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COLORMAPS: tuple[str, ...] = (
    "inferno", "magma", "plasma", "viridis", "cividis",
    "turbo", "hot", "afmhot", "YlOrRd", "OrRd",
    "Spectral", "coolwarm", "RdYlBu_r", "RdPu", "cubehelix",
    "gnuplot2", "CMRmap", "gist_heat", "nipy_spectral", "twilight_shifted", #or any others you find interesting to add
)


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    h = np.nan_to_num(np.asarray(heatmap, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if h.size == 0:
        return h
    p_low = np.percentile(h, 5)
    p_high = np.percentile(h, 99)
    if p_high <= p_low:
        p_low = float(h.min())
        p_high = float(h.max())
    if p_high <= p_low:
        return np.zeros_like(h, dtype=np.float32)
    h = np.clip((h - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
    return h


def generate_heatmap_ideas_figure(
    input_img: np.ndarray,
    heatmap: np.ndarray,
    save_path: str | Path,
    colormaps: Sequence[str] = DEFAULT_COLORMAPS,
    title: str = "Heatmap Overlay Ideas (20 colormaps)",
) -> Path:
    input_img = np.asarray(input_img, dtype=np.float32)
    heatmap_norm = _normalize_heatmap(heatmap)

    overlay = np.where(heatmap_norm > 0.03, heatmap_norm, np.nan)
    alpha_map = np.zeros_like(heatmap_norm, dtype=np.float32)
    valid = np.isfinite(overlay)
    alpha_map[valid] = 0.20 + 0.55 * heatmap_norm[valid]

    vmin = float(np.percentile(input_img, 0.1))
    vmax = float(np.percentile(input_img, 99.0))

    n_maps = len(colormaps)
    n_cols = 5
    n_rows = int(np.ceil(n_maps / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for idx, cmap_name in enumerate(colormaps):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        ax.imshow(input_img, cmap="gray", vmin=vmin, vmax=vmax)
        cmap = plt.cm.get_cmap(cmap_name).copy()
        cmap.set_bad(alpha=0)
        ax.imshow(overlay, cmap=cmap, alpha=alpha_map, vmin=0.0, vmax=1.0)
        ax.set_title(cmap_name, fontsize=10, fontweight="bold")
        ax.axis("off")

    for idx in range(n_maps, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate heatmap_ideas.png from input and heatmap arrays")
    parser.add_argument("--input-npy", type=str, required=True, help="Path to input image .npy")
    parser.add_argument("--heatmap-npy", type=str, required=True, help="Path to heatmap .npy")
    parser.add_argument("--output", type=str, default="heatmap_ideas.png", help="Output png path")
    args = parser.parse_args()

    input_img = np.load(args.input_npy)
    heatmap = np.load(args.heatmap_npy)
    out_path = generate_heatmap_ideas_figure(input_img=input_img, heatmap=heatmap, save_path=args.output)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    _main()
