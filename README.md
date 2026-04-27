<div align="center">

<img src="figures/Anomaly_detection_official_logo_noBg.png" alt="Anomaly Detection Logo" width="300" class="center"/>

# Self-Supervised Anomaly Detection for Medical Images

[![Python](https://img.shields.io/badge/Python%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22863a?style=for-the-badge)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-1f6feb?style=for-the-badge)](https://github.com/MustafaKadhim/Anomaly-Detection-Self-supervised-anomaly-detection-for-medical-images/pulls)

*A research-ready framework for detecting anomalies in medical images without requiring any anomalous training samples.*

[🚀 Quickstart](#-quickstart) · [🔖 Citation](#-citation) · [🏗 Framework](#-framework) · [🧪 Experiments](#-experiments) · [📊 Results](#-results) · [📁 Repository Structure](#-repository-structure)

</div>

---

## 🌟 Highlights

- 🔸**No anomalous labels required:** Trains exclusively on healthy images using self-supervised reconstruction and token distribution learning
- 🔸**Plug-and-play:** Fully customizable and allows modifications based on your task needs. 
- 🔸**Two evaluation experiments:** Developed and tested utilizing publically available pelvic and brain MRI datasets. All scripts are provided above for reproducibility.
- 🔸**Public datasets:** We utilized **LUND-PORBE** for pelvis, and **IXI**, **fastMRI**, and **fastMRI+** (for annotations) for brain experiments.

---

## 🔖 Citation
If you find our work interesting, please cite our work:
```
@inproceedings{
placeholder,
title={Catching MRI outliers: etc.......},
author={M. Kadhim. V. Rogiwski etc........},
booktitle={Phiro-2026 ........ },
year={2026},
url={Phiro-webpage ....... }
}
```

## 🏗 Framework

The core idea is fun & simple: train an autoencoder to perfectly reconstruct **healthy** images. At test time, anomalous regions produce high reconstruction error — forming a pixel-level **anomaly map**.

<div align="center">
<img src="figures/Figure 1. Framework overview (10).png" alt="Framework Architecture" width="600"/>
</div>

> **Figure caption:** * To be added later!.*

---
## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/MustafaKadhim/Anomaly-Detection-Self-supervised-anomaly-detection-for-medical-images.git
cd Anomaly-Detection-Self-supervised-anomaly-detection-for-medical-images
pip install -e .
```

### Minimal Example

```python
import torch
from framework import AnomalyAutoencoder

# Build model
model = AnomalyAutoencoder(in_channels=1, latent_dim=256)

# Forward pass
x = torch.randn(1, 1, 128, 128)          # batch of 1 grayscale MRI slice
out = model(x)

print(out["reconstruction"].shape)        # (1, 1, 128, 128)
print(out["anomaly_map"].shape)           # (1, 1, 128, 128)
print(out["latent"].shape)                # (1, 256)
```

### Data Format

```
data/
  train/
    normal/       ← Healthy images only (used for training)
  test/
    normal/       ← Healthy test images  (label = 0)
    anomaly/      ← Anomalous images     (label = 1)
```

---

## 🧪 Experiments

Two independent experiments are provided, each with their own config, training script, and evaluation pipeline.

<table>
<tr>
<td width="50%" valign="top">

### 🦴 Pelvic MRI

| | |
|---|---|
| Modality | T2-weighted Pelvic MRI |
| Dataset | PROMISE12 |
| Image size | 128 × 128 |
| Latent dim | 256 |
| Epochs | 150 |

```bash
# Train
python experiments/pelvic_mri/train.py

# Evaluate
python experiments/pelvic_mri/evaluate.py \
  --checkpoint experiments/pelvic_mri/checkpoints/checkpoint_best.pth \
  --visualize
```

📂 [`experiments/pelvic_mri/`](experiments/pelvic_mri/)

</td>
<td width="50%" valign="top">

### 🧠 Brain MRI

| | |
|---|---|
| Modality | T1/T2-weighted Brain MRI |
| Dataset | BraTS + IXI |
| Image size | 128 × 128 |
| Latent dim | 512 |
| Epochs | 200 |

```bash
# Train
python experiments/brain_mri/train.py

# Evaluate
python experiments/brain_mri/evaluate.py \
  --checkpoint experiments/brain_mri/checkpoints/checkpoint_best.pth \
  --visualize
```

📂 [`experiments/brain_mri/`](experiments/brain_mri/)

</td>
</tr>
</table>

---

## 📊 Results

> Results will be populated after running the experiments.

| Experiment | AUROC | AUPRC | FPR @ 95 % TPR |
|-----------|:-----:|:-----:|:--------------:|
| Pelvic MRI | — | — | — |
| Brain MRI  | — | — | — |

---

## 📁 Repository Structure

```
.
├── framework/                    # 🏗 Core reusable framework
│   ├── models/
│   │   ├── encoder.py            #   Residual convolutional encoder
│   │   ├── decoder.py            #   U-Net style decoder with skip connections
│   │   └── autoencoder.py        #   Full AnomalyAutoencoder model
│   ├── losses/
│   │   └── anomaly_loss.py       #   L1 + SSIM + Perceptual combined loss
│   ├── datasets/
│   │   └── medical_dataset.py    #   Generic medical image dataset loader
│   ├── trainers/
│   │   └── anomaly_trainer.py    #   Training loop with checkpointing + LR scheduling
│   └── utils/
│       ├── metrics.py            #   AUROC, AUPRC, FPR@95TPR, optimal threshold
│       └── visualization.py      #   Anomaly map, ROC curve, training curve plots
│
├── experiments/
│   ├── pelvic_mri/               # 🦴 Pelvic MRI experiment
│   │   ├── config.yaml           #   Hyperparameters
│   │   ├── train.py              #   Training script
│   │   ├── evaluate.py           #   Evaluation script
│   │   └── data/README.md        #   Dataset preparation guide
│   │
│   └── brain_mri/                # 🧠 Brain MRI experiment
│       ├── config.yaml
│       ├── train.py
│       ├── evaluate.py
│       └── data/README.md
│
├── figures/                      # 🎨 Visuals used in README
│   ├── logo.svg
│   └── architecture.svg
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Configuration

Every aspect of each experiment is controlled by its `config.yaml`:

```yaml
model:
  in_channels: 1        # 1 = grayscale, 3 = RGB
  latent_dim: 256       # Bottleneck size
  base_channels: 32     # Feature map width
  use_skip: true        # U-Net skip connections

training:
  num_epochs: 150
  batch_size: 16
  learning_rate: 1.0e-4
  l1_weight: 1.0
  ssim_weight: 1.0
  perceptual_weight: 0.1

evaluation:
  anomaly_score_reduction: "percentile95"  # mean | max | percentile95
```

---
---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ for the medical imaging research community

⭐ Star this repo if you find it useful!

</div>
