# 🦴 Pelvic MRI Anomaly Detection Experiment

This experiment applies the self-supervised anomaly detection framework to **Pelvic MRI** data.
The model learns the appearance of healthy pelvic anatomy and flags deviations as anomalies at test time.

## Overview

| Property | Value |
|---|---|
| Modality | Pelvic MRI (T2-weighted) |
| Image size | 128 × 128 |
| Latent dimension | 256 |
| Epochs | 150 |
| Batch size | 16 |
| Optimizer | Adam (lr = 1e-4, cosine decay) |
| Loss | L1 + SSIM + Perceptual |

## Quickstart

### 1. Prepare data

Follow the instructions in [`data/README.md`](data/README.md) to place your dataset in the correct structure.

### 2. Train

```bash
python experiments/pelvic_mri/train.py
```

Override any config value from the CLI:

```bash
python experiments/pelvic_mri/train.py --epochs 200 --lr 5e-5
```

Resume from a checkpoint:

```bash
python experiments/pelvic_mri/train.py --resume experiments/pelvic_mri/checkpoints/checkpoint_latest.pth
```

### 3. Evaluate

```bash
python experiments/pelvic_mri/evaluate.py \
    --checkpoint experiments/pelvic_mri/checkpoints/checkpoint_best.pth \
    --visualize
```

Results (AUROC, AUPRC, FPR@95TPR) are printed to the console and saved to:

```
experiments/pelvic_mri/results/
  metrics.json
  roc_curve.png
  visualizations/
    sample_0000_normal.png
    sample_0001_anomaly.png
    ...
```

## Configuration

All hyperparameters live in [`config.yaml`](config.yaml).
Edit the file or use CLI flags to customise the experiment.

## Results

> Fill in after running your experiment.

| Metric | Value |
|--------|-------|
| AUROC | — |
| AUPRC | — |
| FPR @ 95 % TPR | — |
