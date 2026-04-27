# 🧠 Brain MRI Anomaly Detection Experiment

This experiment applies the self-supervised anomaly detection framework to **Brain MRI** data.
The model learns the appearance of healthy brain anatomy (T1/T2-weighted) and detects
tumours, lesions, and other pathologies as anomalies at inference time.

## Overview

| Property | Value |
|---|---|
| Modality | Brain MRI (T1 / T2-weighted) |
| Image size | 128 × 128 |
| Latent dimension | 512 |
| Epochs | 200 |
| Batch size | 16 |
| Optimizer | Adam (lr = 5e-5, cosine decay) |
| Loss | L1 + 1.5×SSIM + 0.2×Perceptual |

The larger latent dimension (512) and stronger SSIM weight capture the higher anatomical
complexity of brain MRI compared to pelvic data.

## Quickstart

### 1. Prepare data

Follow the instructions in [`data/README.md`](data/README.md) to download and prepare
the IXI (healthy) and BraTS (anomalous) datasets.

### 2. Train

```bash
python experiments/brain_mri/train.py
```

Override hyperparameters from the CLI:

```bash
python experiments/brain_mri/train.py --epochs 300 --lr 1e-5
```

Resume from a checkpoint:

```bash
python experiments/brain_mri/train.py --resume experiments/brain_mri/checkpoints/checkpoint_latest.pth
```

### 3. Evaluate

```bash
python experiments/brain_mri/evaluate.py \
    --checkpoint experiments/brain_mri/checkpoints/checkpoint_best.pth \
    --visualize
```

Results are saved to:

```
experiments/brain_mri/results/
  metrics.json
  roc_curve.png
  visualizations/
    sample_0000_normal.png
    sample_0001_anomaly.png
    ...
```

## Configuration

All hyperparameters live in [`config.yaml`](config.yaml).

## Results

> Fill in after running your experiment.

| Metric | Value |
|--------|-------|
| AUROC | — |
| AUPRC | — |
| FPR @ 95 % TPR | — |
