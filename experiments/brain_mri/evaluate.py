"""
Evaluate the trained Brain MRI anomaly detection model.

Usage
-----
    python experiments/brain_mri/evaluate.py --checkpoint experiments/brain_mri/checkpoints/checkpoint_best.pth
    python experiments/brain_mri/evaluate.py --checkpoint <path> --visualize
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from framework.datasets import MedicalImageDataset
from framework.models import AnomalyAutoencoder
from framework.utils import (
    compute_anomaly_score,
    evaluate_anomaly_detection,
    optimal_threshold,
    plot_anomaly_map,
    plot_roc_curve,
)
from sklearn.metrics import roc_curve


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(args):
    cfg = load_config(args.config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg["model"]
    model = AnomalyAutoencoder(
        in_channels=model_cfg.get("in_channels", 1),
        latent_dim=model_cfg.get("latent_dim", 512),
        base_channels=model_cfg.get("base_channels", 32),
        use_skip=model_cfg.get("use_skip", True),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Loaded checkpoint from epoch %d", ckpt["epoch"])

    data_cfg = cfg["data"]
    test_dataset = MedicalImageDataset(
        root=data_cfg["root"],
        split="test",
        grayscale=data_cfg.get("grayscale", True),
        image_size=data_cfg.get("image_size", 128),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["evaluation"].get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )
    logger.info("Test set: %s", test_dataset)

    all_scores, all_labels = [], []
    all_images, all_recons, all_maps = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            out = model(images)
            scores = compute_anomaly_score(
                out["anomaly_map"].cpu().numpy(),
                reduction=cfg["evaluation"].get("anomaly_score_reduction", "percentile95"),
            )
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.numpy().tolist())
            if args.visualize:
                all_images.append(images.cpu().numpy())
                all_recons.append(out["reconstruction"].cpu().numpy())
                all_maps.append(out["anomaly_map"].cpu().numpy())

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    metrics = evaluate_anomaly_detection(scores_arr, labels_arr)
    thresh, precision, recall = optimal_threshold(scores_arr, labels_arr)
    metrics.update({"best_threshold": thresh, "precision_at_thresh": precision, "recall_at_thresh": recall})

    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info("  %-28s %.4f", k, v)
    logger.info("=" * 50)

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fprs, tprs, _ = roc_curve(labels_arr, scores_arr)
    plot_roc_curve(fprs, tprs, metrics["auroc"], experiment_name="Brain MRI",
                   save_path=str(results_dir / "roc_curve.png"))

    if args.visualize:
        vis_dir = results_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        images_cat = np.concatenate(all_images, axis=0)
        recons_cat = np.concatenate(all_recons, axis=0)
        maps_cat = np.concatenate(all_maps, axis=0)
        for i in range(min(args.num_viz, len(images_cat))):
            label_str = "anomaly" if labels_arr[i] == 1 else "normal"
            plot_anomaly_map(
                images_cat[i], recons_cat[i], maps_cat[i],
                score=float(scores_arr[i]),
                title=f"Sample {i} [{label_str}]",
                save_path=str(vis_dir / f"sample_{i:04d}_{label_str}.png"),
            )
        logger.info("Visualizations saved to %s", vis_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Brain MRI anomaly detection model")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument(
        "--checkpoint",
        default="experiments/brain_mri/checkpoints/checkpoint_best.pth",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_viz", type=int, default=20)
    main(parser.parse_args())
