<div align="center">

# 🦴 Two-Stage Unsupervised Anomaly Detection for Pelvic MRI

### *LUND-PROBE-focused Implementation*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-orange.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-red.svg)](https://monai.io/)
[![Research](https://img.shields.io/badge/Status-Research_Code-yellow.svg)]()

*A two-stage unsupervised anomaly-detection framework that learns from normal/reference pelvic MRI slices and detects deviations in synthetic and clinical anomaly cohorts.*

</div>

---

## 📑 Table of Contents

<table>
<tr>
<td width="50%" valign="top">

**🎯 Getting Started**
- [Overview & Core Concept](#-overview--core-concept)
- [The AUROC Pipeline](#-the-auroc-pipeline)
- [Repository Structure](#-repository-structure)
- [Environment Setup](#-environment-setup)

**🏗️ Architecture**
- [Method Overview](#-method-overview)
- [Stage 1: RVQ-VAE](#-stage-1--rvq-vae)
- [Stage 2: Factorized MaskGIT / Fact-biT](#-stage-2--factorized-maskgit--fact-bit)

</td>
<td width="50%" valign="top">

**📊 Data & Training**
- [Data Format](#-data-format)
- [Data Preparation](#-data-preparation)
- [Training](#-training)

**🔬 Inference & Evaluation**
- [Inference and Calibration](#-inference-and-calibration)
- [ROC / AUROC and PR /  Evaluation](#-roc--auroc-and-pr---evaluation)
- [Synthetic Anomaly Utilities](#-synthetic-anomaly-utilities)
- [LPIPS-Backflow Controls and Ablation](#-lpips-backflow-controls-and-ablation)

**📋 Reference**
- [Reproducibility Checklist](#-exact-replication-checklist)
- [Differences from Brain Version](#-key-differences-from-the-brain-mri-version)
- [Practical Notes](#-practical-notes-for-github-readers)

</td>
</tr>
</table>

---

## 🎯 Overview & Core Concept

This repository contains the cleaned **Pelvic MRI implementation** of a two-stage unsupervised anomaly-detection framework. The models are trained on normal/reference pelvic MRI slices and evaluated on synthetic and clinical anomaly cohorts.

The code has been organized around a **CORE vs. AYNU** concept:

<table>
<tr>
<th width="15%">🟢 CORE</th>
<td>Code that <b>directly contributes</b> to the manuscript AUROC-analysis reproduction pipeline.</td>
</tr>
<tr>
<th width="15%">🟡 AYNU</th>
<td><b>"Available Yet Not AUROC-interesting"</b> — auxiliary code retained for transparency, debugging, training diagnostics, visualizations, calibration generation, alternative scores, and supplementary analyses, but not for the primary AUROC calculations.</td>
</tr>
</table>

> ⚠️ **Ground-truth anomaly labels are NOT used for model training.** They are used only for evaluation, cohort/category assignment, plotting, and optional annotation overlays.

---

## 🔄 The AUROC Pipeline

The primary patient-level ROC/PR path flows through the following stages:

```text
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT SLICE                             │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          Stage 1: RVQ-VAE  →  reconstruction / tokens           │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      Stage 2: Factorized MaskGIT / Fact-biT  →  healing         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      LPIPS (input vs. healed/inpainted)  →  heatmap             │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      perceptual + token-surprisal + LPIPS-backflow fusion       │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│     Final_Binary_sum_of_anomaly_maps (per slice)                │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      Patient-level:  sum_all_bars_score                         │
│     = Σ_slices(Final_Binary_sum_of_anomaly_maps)                │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ROC / AUROC-analysis                     │
└─────────────────────────────────────────────────────────────────┘
```

### 🟢 CORE Output Fields

The per-slice field consumed by the main patient-level ROC/ pipeline is:

| Field | Role |
|---|---|
| `Final_Binary_sum_of_anomaly_maps` | Final binary ALM mask count: **ALM(A) ∪ ALM(B) ∪ LPIPS-backflow** when backflow is enabled. |

`ROC_Curves_Calculations.py` aggregates this field into a patient-level score:

```text
sum_all_bars_score = Σ_slices(Final_Binary_sum_of_anomaly_maps)
```

Then `compute_patient_roc_and_auc(...)` computes patient-level ROC/AUROC from `sum_all_bars_score`.

Important: `Final_Binary_sum_of_anomaly_maps` is the final fused binary mask count when `--binary-include-lpips-backflow` is active. It already includes ALM-B token pixels and LPIPS-backflow by default, so `token_surprisal_hot_px` is retained only for audit/ablation and is **not** added separately to the ROC score. Use `--no-binary-include-lpips-backflow` for a clean no-backflow contribution if needed.

### 🟡 AYNU Examples

<details>
<summary><b>Click to expand — useful for debugging but not for the primary AUROC-analysis </b></summary>

- `clamped_pixel_sum`
- `lpips_input_recon_sum_mask`
- Sharpness totals and sharpness-based plots
- Per-patient bar plots of intermediate quantities
- Stage 2 alternative anomaly-map methods
- Token-frequency diagnostics
- Calibration-generation figures
- Annotation overlay figures
- Synthetic-data generation utilities

</details>

---

## 📁 Repository Structure

```text
Final_Clean_to_Github_Pelvis/
│
├── 🟢 CORE — Models & Pipeline
│   ├── Model_Stage_1.py                         # Stage 1 RVQ-VAE model
│   ├── Model_Stage_2.py                         # Stage 2 Factorized MaskGIT / Fact-biT model
│   ├── Train_framework.py                       # PyTorch Lightning training entry point
│   ├── dataset.py                               # .npy slice Dataset/DataModule
│   ├── Inference_Pelvis_Experiments.py          # Recursive-AutoMask V4 inference + calibration
│   └── ROC_Curves_Calculations.py               # Patient-level ROC/ and category analyses
│
├── ⚙️  Configuration & Documentation
│   ├── config_yaml.yaml                         # Reference config summary (not auto-loaded)
│   ├── Train_Val_Test_Exact_DataSplits_LUND_PROBE.json  # Recorded split information
│   └── Pelvis_Experiments_requirements.txt      # Pinned Python environment from the experiment
│
└── 🔧 Data Preparation, External Cohorts & Synthetic Utilities
    ├── preslice_volumes.py                      # NIfTI volume → per-slice .npy preprocessing
    ├── External_dataset.py                      # External cohort preprocessing/loading utilities
    ├── Simulation_inference_v4_extended_CJG.py  # Synthetic anomaly generation helpers
    └── Simluation_inference_v3_support_CJG.py   # Support code for synthetic data generation
```

> ⚠️ **Important:** Several scripts contain absolute local default paths from the original experiment environment. For a new machine or GitHub user, **pass explicit CLI paths** or edit the defaults before running.

---

## 🏗️ Method Overview

The framework has **two learned stages**:

<table>
<tr>
<th>Stage</th>
<th>Model</th>
<th>Purpose</th>
</tr>
<tr>
<td align="center"><b>1️⃣<br>Stage 1</b></td>
<td><b>RVQ-VAE</b><br><sub>ViT encoder, residual vector quantization, PixelShuffle decoder</sub></td>
<td>Learns a discrete latent representation of normal/reference pelvic MRI appearance</td>
</tr>
<tr>
<td align="center"><b>2️⃣<br>Stage 2</b></td>
<td><b>Factorized MaskGIT / Fact-biT transformer</b></td>
<td>Learns token distributions and heals masked/suspect tokens using bidirectional masked prediction</td>
</tr>
</table>

### 🎛️ Recursive-AutoMask V4 (Inference)

At inference, **Recursive-AutoMask V4** computes three complementary anomaly signals:

<table>
<tr>
<th>Signal</th>
<th>Description</th>
</tr>
<tr>
<td><b>Token surprisal</b></td>
<td>Repeated random masking of Stage 1 L1 tokens, followed by Stage 2 prediction and NLL scoring of the true tokens.</td>
</tr>
<tr>
<td><b>LPIPS healing heatmap</b></td>
<td>Stage 2 heals checkerboard-masked tokens; spatial LPIPS compares the input image with healed/inpainted images; calibrated Z-score thresholding converts the heatmap to a binary detection mask.</td>
</tr>
<tr>
<td><b>LPIPS-backflow</b></td>
<td>After targeted inpainting, spatial LPIPS compares <b>Input vs. Inpainted</b>. The map is thresholded to a binary backflow mask and OR-unioned into the final binary ALM mask when enabled.</td>
</tr>
</table>

Unlike the cleaned Brain README, the Pelvis inference code’s main LPIPS branch is **input-referenced**:

| Phase | LPIPS Reference |
|---|---|
| **Calibration** | LPIPS(input, healed) |
| **Inference iteration 0** | LPIPS(input, healed) |
| **Targeted backflow iterations** | LPIPS(input, inpainted) |

> 💡 `lpips_input_recon` is computed for auxiliary diagnostics/visualizations, not for the main AUROC score.

---

## 🧩 Stage 1 — RVQ-VAE

📄 **File:** `Model_Stage_1.py`

`Stage1RVQVAE` maps a 2D grayscale pelvic MRI slice to RVQ tokens and reconstructs the slice.

### Architecture

| Component | Implementation detail |
|---|---|
| **Input** | Single-channel 2D pelvic MRI slice, typically `1 × 256 × 256` |
| **Patch embedding** | `Conv2d` with `kernel_size = stride = patch_size` |
| **Patch size in training script** | `8` → `32 × 32 = 1024` tokens |
| **Encoder** | ViT-style Transformer encoder, depth 8, 8 heads |
| **Quantizer** | `ResidualVQ`, 2 quantizers, codebook size 192 in the training script |
| **Decoder** | PixelShuffle decoder back to one image channel; **output NOT clamped** (returned directly from the decoder) |
| **Forward output** | `recon`, `indices`, `commit_loss`, `quant_error_map` |
| **Multi-scale encoder** | Available if needed in the model yet not used; instantiated in the **AYNU block** of `__init__` — used only by auxiliary `encode_multiscale()` path, not the primary CORE encode path |

### Training Loss

```text
L1 reconstruction loss
+ BiomedCLIP perceptual loss (weight 0.9)
+ RVQ commitment loss
```

Training uses AdamW with a cosine annealing learning-rate schedule. Stage 1 also contains training-time augmentation and validation visualization code, which is AYNU relative to the inference AUROC path.

### Augmentations

**Stage 1 internal augmentation** (applied in `training_step`, always active during training):

<table>
<tr>
<td>🎨 Intensity scaling</td>
<td>🔄 Affine rotation/translation</td>
</tr>
</table>

> ⚠️ **Only 2 transforms.** Intensity scaling uses prob=0.33, scale range 0.5–1.5. Affine uses rotation ±5°, horizontal-only translation ±5 px, prob=0.33. There is **no horizontal flip, no contrast adjustment, no Gaussian noise, and no zoom**. The affine transform does not apply any zoom/scale factor.

**DataModule augmentation** (applied when `augment=True` is passed to `SliceDataModule`):

<table>
<tr>
<td>↔️ Random flip</td>
<td>🔄 Random rotation</td>
</tr>
</table>

> ⚠️ **Note:** Random flip uses prob=0.5; random rotation ±5° uses prob=0.3. In the current `Train_framework.py`, `SliceDataModule` is instantiated **without** `augment=True`, so DataModule augmentations are **disabled by default**. Only the 2 Stage 1 internal transforms (intensity + affine) are active by default.

---

## 🧩 Stage 2 — Factorized bi-directional transformer (Fact-biT)

📄 **File:** `Model_Stage_2.py`

`FactorizedMaskGIT` predicts masked RVQ tokens from Stage 1.

### Architecture

| Component | Implementation detail |
|---|---|
| **Token levels** | Separate L1 and L2 token streams |
| **Codebook size** | 192 per level when loaded from the Stage 1 checkpoint used by `Train_framework.py` |
| **Transformer** | SDPA transformer with RMSNorm and SwiGLU |
| **Position encoding** | **3D RoPE** over row, column, and slice position |
| **Sequence length** | Derived from Stage 1 image size and patch size; typically 1024 |
| **Stage 1 during training** | Frozen and set to eval mode |

### Training Loss

```text
CE(masked L1 tokens) + 0.25 × CE(masked L2 tokens)
```

With **label smoothing** (`0.05`) and a **mixed random/block masking strategy**.

> 📌 A key pelvic-specific feature is slice-position conditioning through 3D RoPE. The model extracts slice indices from filenames such as `patient_id_slice_045.npy`. Slice indices are used for Stage 2 anatomical position encoding and for optional per-slice calibration statistics.

---

## 🛠️ Environment Setup

For the original environment, install the pinned requirements:

```bash
pip install -r Pelvis_Experiments_requirements.txt
```

Important dependencies include:

- PyTorch / PyTorch Lightning
- MONAI
- `vector-quantize-pytorch`
- `transformers` and/or `open_clip_torch` for BiomedCLIP during Stage 1 training
- `lpips` for spatial perceptual heatmaps during inference
- nibabel, NumPy, SciPy, scikit-image, matplotlib, tqdm, W&B

---

## 📦 Data Format

### Training Slice Files

Training uses individual `.npy` slices. The expected filename pattern is:

```text
{patient_id}_slice_{idx:03d}.npy
```

Files that do not contain `_slice_` are ignored by `SliceDataModule`.

### `dataset.py` Preprocessing

Each `.npy` slice is loaded and transformed as follows:

<details>
<summary><b>📋 Main Preprocessing Steps</b></summary>

1. Load `float32` NumPy array
2. Rotate with `np.rot90(arr, k=-1)`
3. Add channel dimension with MONAI `EnsureChannelFirstD`
4. Resize to `320 × 320` using area interpolation
5. Center crop to `256 × 256`
6. Convert to tensor
7. Optionally apply DataModule training augmentation if `augment=True`

</details>

### Recommended Directory Structure

Preserve one folder per patient/case whenever possible so downstream aggregation can group slices correctly:

```text
data_dir/
├── patient_001/
│   ├── patient_001_slice_030.npy
│   └── patient_001_slice_031.npy
└── patient_002/
    └── patient_002_slice_045.npy
```

> ⚠️ **Pelvis-specific requirement:** Preserve filename slice indices (`_slice_###`) because they affect Stage 2 3D RoPE conditioning and optional per-slice calibration lookup.

---

## 🧪 Data Preparation

### 🔹 Pre-slicing LUND-PROBE / Normal-Reference NIfTI Data

`preslice_volumes.py` converts 3D NIfTI volumes into `.npy` slices.

```bash
python preslice_volumes.py
```

Main behavior:

- Reads source NIfTI paths from the script/config environment
- Z-score normalizes per volume
- Saves every axial slice as `{patient_id}_slice_{idx:03d}.npy`
- Writes `preslice_metadata.json`

> ⚠️ Because `preslice_volumes.py` is script/config driven, check or edit its path settings before running.

### 🔹 External / Synthetic / Clinical Cohorts

`External_dataset.py` contains utilities for processing external NIfTI cohorts and preserving category/case metadata. Its preprocessing utilities include:

<table>
<tr>
<td>📥 NIfTI loading</td>
<td>📊 Slice normalization</td>
<td>📐 Resize/crop</td>
</tr>
<tr>
<td>💾 Per-slice saving</td>
<td>🏷️ Category tracking</td>
<td>🗂️ Case-folder tracking</td>
</tr>
</table>

The downstream ROC code uses patient/case/category identifiers to stratify results into synthetic, clinical, or normal testing groups.

---

## 🚂 Training

Entry point: **`Train_framework.py`**

### 1️⃣ Stage 1 Training

```bash
python Train_framework.py --stage1 \
    --data-dir /path/to/PreSliced \
    --batch-size 128 \
    --num-workers 8 \
    --max-epochs 100 \
    --lr 1e-4 \
    --precision 32 \
    --wandb-project RVQ-MaskGIT \
    --wandb-run-name Stage1-RVQ-VAE
```

**Current script-level settings/defaults:**

| Setting | Value |
|---|---:|
| `embed_dim` | 256 |
| `codebook_size` | 192 |
| `commitment_cost` | 0.25 |
| learning rate default | `1e-4` |
| batch size default | 128 |
| max epochs default | 100 |
| checkpoint monitor | `val/loss` |
| checkpoint top-k | 3 |
| trainer device default | `[1]` |

### 2️⃣ Stage 2 Training

```bash
python Train_framework.py --stage2 \
    --data-dir /path/to/PreSliced \
    --stage1-ckpt /path/to/stage1.ckpt \
    --batch-size 128 \
    --num-workers 8 \
    --max-epochs 100 \
    --lr 1e-4 \
    --precision 32 \
    --wandb-run-name Stage2-Factorized-MaskGIT
```

> ⚠️ Stage 2 requires a valid Stage 1 checkpoint. **The Stage 1 model is frozen during Stage 2 training.**

Stage 2 also filters training slices to the anatomically relevant slice-index range, documented in the config as approximately:

```text
train_slice_min = 30
train_slice_max = 60
```

This depends on correctly encoded `_slice_###` filenames.

> ⚠️ The current `Train_framework.py` / `SliceDataModule` path still seed-splits `data_dir` into train/validation subsets at runtime. The split manifest JSON is a recorded reference, not an automatically enforced input to the current training entry point.

### 📊 Logging and Checkpointing

`Train_framework.py` uses:

- `CSVLogger`
- Optional `WandbLogger`
- `LearningRateMonitor`
- `ModelCheckpoint`

Disable W&B with:

```bash
--wandb-off
```

> 🔐 **Privacy note:** Do not log patient-identifiable information to W&B, filenames, plots, or shared logs.

---

## 🔬 Inference and Calibration

Main script: **`Inference_Pelvis_Experiments.py`**

### 🔧 Model Loading

`load_models(stage1_ckpt, stage2_ckpt, device)` loads Stage 1 and Stage 2 checkpoints. Stage 1 perceptual-loss keys are stripped during inference loading, so **BiomedCLIP weights are not required for checkpoint loading at inference time**.

### Step 1️⃣ — Normal / Reference Calibration

Calibration estimates per-pixel normal/reference LPIPS statistics:

```text
mu[h, w]    = mean LPIPS(input, healed) over calibration slices
sigma[h, w] = std  LPIPS(input, healed) over calibration slices
```

```bash
python Inference_Pelvis_Experiments.py \
    --calibration-mode \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --data-dir /path/to/normal_reference_slices \
    --calibration-map /path/to/zscore_calibration.npz \
    --smoothing-kernel 15 \
    --heatmap-aggregation geomean \
    --heal-patterns "2,3" \
    --device cuda:0
```

The calibration file may contain:

| Key | Meaning |
|---|---|
| `mu` | Global per-pixel mean LPIPS map |
| `sigma` | Global per-pixel std LPIPS map |
| `n_samples` | Number of calibration slices |
| `smoothing_kernel` | Smoothing kernel used before statistics |
| per-slice entries | Optional slice-index-specific statistics when enough samples exist |

> 🚨 **Critical reproducibility rule:** Use the **same** `--smoothing-kernel` during calibration and inference.

Calibration generation itself is marked AYNU in the refactor blueprint because the AUROC path loads an existing calibration map; nevertheless, calibration is required to create that map for a new dataset/reference population.

### Step 2️⃣ — Anomaly Inference

Example using current inspected script defaults as a starting point:

```bash
python Inference_Pelvis_Experiments.py \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --data-dir /path/to/test_or_anomaly_slices \
    --output-dir /path/to/inference_results \
    --calibration-map /path/to/zscore_calibration.npz \
    --z-threshold 2.0 \
    --smoothing-kernel 15 \
    --heatmap-aggregation geomean \
    --heal-patterns "2,3" \
    --token-surprisal-samples 50 \
    --token-surprisal-mask-ratio 0.90 \
    --token-surprisal-clamp 8.0 \
    --binary-threshold 0.60 \
    --device cuda:0
```

### ⚙️ Current Important Inference Defaults

| Argument | Current default |
|---|---:|
| `--z-threshold` | `2.0` |
| `--smoothing-kernel` | `15` |
| `--heatmap-aggregation` | `geomean` |
| `--batch-size` | `320` |
| `--heal-steps` | `6` |
| `--heal-temperature` | `0.3` |
| `--heal-patterns` | `"2,3"` |
| `--binary-threshold` | `0.60` |
| `--binary-include-lpips-backflow` | enabled by default |
| `--LPIPS-in-inp-threshold-back-to-binary-token-map` | `"(99, 0)"` |
| `--token-surprisal-samples` | `50` |
| `--token-surprisal-mask-ratio` | `0.90` |
| `--token-surprisal-clamp` | `8.0` |
| `--use-tta` | enabled by default |
| `--num-iterations` | `1` |
| `--inpaint-steps` | `12` |
| `--inpaint-temperature` | `0.3` |
| `--device` | `cuda:1` |

> 💾 These defaults are experiment-specific. **Save the exact CLI command with each run.**

### 🔄 Main Inference Steps

Inside `recursive_automask_v4_zscore(...)`, the CORE flow is:

```text
1. Load input slice and extract slice position from filename when available
        ↓
2. Compute Stage 1 reconstruction and RVQ tokens (obtain L1-and L2-tokens)
        ↓
3. Compute token surprisal through random repeated L1-token masking  →  ALM-B binary mask
        ↓
4. Heal checkerboard-masked tokens using Stage 2
        ↓
5. Compute LPIPS between input and healed image(s)  →  ALM-A binary mask
        ↓
6. Aggregate native and TTA heatmaps, commonly with geomean
        ↓
7. Smooth and threshold by Z-score using the calibration map → binarized maps
        ↓
8. Final_Binary_sum_of_anomaly_maps = sum(ALM-A ∪ ALM-B ∪ LPIPS-backflow when enabled)
        ↓
9. Write per-slice JSON fields, including Final_Binary_sum_of_anomaly_maps and token_surprisal_hot_px
        ↓
10. Aggregate patient scores in ROC_Curves_Calculations.py
```

Targeted inpainting and multi-iteration refinement are implemented and preserved, but the current AUROC default is `--num-iterations 1`, so later refinement iterations are not part of the default CORE score.

### 📤 Output JSON Fields

Main output file: **`results_v4_zscore.json`**

| Field | Type | Meaning |
|---|:---:|---|
| `Final_Binary_sum_of_anomaly_maps` | 🟢 CORE | Per-slice final fused binary ALM count used in `sum_all_bars_score`; includes ALM-B and LPIPS-backflow by default |
| `token_surprisal_hot_px` | 🟡 AYNU | Count of hot token-surprisal pixels after clamping, retained for audit/ablation only; not added separately by the ROC script |
| `Final_Binary_sum_of_anomaly_maps_Base` | 🟡 AYNU | ALM-A binary count before token/backflow union |
| `Final_Binary_sum_of_anomaly_maps_Token` | 🟡 AYNU | ALM-B token-surprisal binary count |
| `Final_Binary_sum_of_anomaly_maps_Overlap` | 🟡 AYNU | ALM-A ∩ ALM-B overlap |
| `Final_Binary_sum_of_anomaly_maps_LPIPS_Backflow` | 🟡 AYNU | LPIPS(Input, Inpainted) backflow binary count before union |
| `path`, `filename` | 🔹 meta | Source slice path/name |
| `category` | 🔹 meta | Cohort/anomaly category metadata |
| `case_folder` | 🔹 meta | Patient/case grouping metadata |
| `used_zscore` | 🔹 meta | Whether calibration Z-score thresholding was active |
| `z_threshold` | 🔹 meta | Z-score threshold used if active |
| `clamped_pixel_sum` | 🟡 AYNU | Auxiliary LPIPS-derived score; not the primary AUROC score |
| `lpips_input_recon_sum_mask` | 🟡 AYNU | Auxiliary reconstruction diagnostic |
| `sharpness_score`, `artifact_flag` | 🟡 AYNU | Auxiliary artifact/sharpness diagnostics |
| `iteration_metrics` | 🟡 AYNU | Auxiliary iteration/refinement diagnostics |

### 🖼️ Visualization Outputs

| File | Figures shown | Purpose |
|---|---|---|
| `_Final_ALM_Arithmetic.png` | Input · normalized ALM-A · normalized ALM-B · normalized LPIPS-backflow if available · arithmetic overlay · binary component maps | Default qualitative figure; visual review only, not used by ROC |
| `_Final_ALM_Heatmap.png` | Input · Score · **ALM-A (LPIPS binary)** · **ALM-B (Token binary)** · Overlay · final binary mask | Optional figure enabled by `--save-alm-heatmap-png`; can be interesting for debugging. |
| `_Anomaly_Overlay.png` | Input · Healed · Heatmap overlay · LPIPS+Token overlay · LPIPS raw · **Binary+Token map** | Optional full analysis figure enabled by `--include-full-analysis-figure` |
| `_full.png` | Multi-panel diagnostic figure | Full pipeline diagnostics for full overview of performance (🟡 AYNU) |

> The ROC scripts read JSON fields, not PNGs. `_Final_ALM_Arithmetic.png` independently normalizes components before visualization, so it does not impact the numeric ROC score.

---

## 📈 ROC / AUROC and PR /  Evaluation

Main script: **`ROC_Curves_Calculations.py`**

### 🟢 CORE Merged-ROC Workflow

```bash
python ROC_Curves_Calculations.py \
    --run-merged-roc \
    --roc-input /path/to/results_v4_zscore_a.json /path/to/results_v4_zscore_b.json \
    --output-dir /path/to/roc_outputs
```

If `--roc-input` is not supplied, the script may fall back to `DEFAULT_ROC_INPUT_PATHS`, which are local experiment paths and should usually be overridden.

The CORE functions are:

```text
aggregate_patient_sum_of_all_bars(...)
merge_json_payloads_for_roc(...)
compute_patient_roc_and_auc(...)
```

### 🟢 CORE Aggregation

The patient-level score is:

```text
sum_all_bars_score
  = Σ_slices(Final_Binary_sum_of_anomaly_maps)
```

`compute_patient_roc_and_auc(...)` then computes patient-level ROC/AUROC and PR/ using:

| Variable | Definition |
|---|---|
| `score` | `sum_all_bars_score` |
| `label = 0` | Normal/reference patients (`orig` identifier) |
| `label = 1` | Anomaly patients |

> 📌 Labels are assigned from patient/case identifiers: `orig` cases are treated as normal/reference and all other cases as anomaly. Confirm/change this naming convention before using the ROC script on new cohorts.

### 📊 Outputs

The ROC script can produce:

- Merged JSON payloads
- ROC curve figures
- Precision-recall curve figures
- AUROC metrics JSON
- Bootstrap confidence intervals
- Threshold tables
- Synthetic-vs-clinical split curves
- Category-stratified sensitivity tables

The script also contains many AYNU plotting functions for intermediate patient-level summaries.

### ⚠️ Deprecation Notice

> Older code or notes may refer to `clamped_pixel_sum`, `lpips_input_recon_sum_mask`, or `Binary_Sum_Heatmap` as the primary patient-level score. **That is not the current Pelvis ROC path.**
>
> The primary ROC path uses:
> ```text
> sum_all_bars_score = Σ_slices(Final_Binary_sum_of_anomaly_maps)
> ```
> `token_surprisal_hot_px` is retained only as an audit/ablation field and is **not** added separately to `sum_all_bars_score`.

---

## 📄 Reference Configuration File

`config_yaml.yaml` documents the intended experiment settings in a structured format. It is useful as a reference, but the inspected main Python scripts do **NOT** automatically load it as the sole runtime source of truth.

### Active behavior is controlled by:

1. CLI arguments
2. Hard-coded defaults inside Python scripts
3. Checkpoint hyperparameters
4. Filename conventions

---

## 🧪 Synthetic Anomaly Utilities

The repository includes helper scripts for synthetic anomaly generation:

- `Simulation_inference_v4_extended_CJG.py`
- `Simluation_inference_v3_support_CJG.py`

These are not on the primary AUROC computation path, but they document and support generation of synthetic variations such as blur/noise/inserted structures used in the broader experimental workflow. They are optional for you to use in case you would like to simulate global or prostate-local anomalies. 

---

## ✅ Framework Replication Checklist

Use this checklist when trying to reproduce the Pelvis experiment:

- [ ] Normal/reference NIfTI volumes were pre-sliced to `.npy` with the expected `_slice_###` naming convention
- [ ] Training/validation/test splits are recorded and reused in JSON file (***Train_Val_Test_Exact_DataSplits_LUND_PROBE.json***)
- [ ] Stage 1 checkpoint path is recorded
- [ ] Stage 2 checkpoint path is recorded
- [ ] Calibration data are normal/reference and independent from anomaly evaluation cohorts
- [ ] `--smoothing-kernel` is identical between calibration and inference
- [ ] Slice indices are preserved in filenames for 3D RoPE during training, inference, and per-slice calibration lookup
- [ ] `NpySliceDataset.__getitem__` applies `np.rot90(arr, k=-1)` at load time before MONAI transforms — verify this rotation is consistent across training, calibration, and inference data loaders for your specific data. 
- [ ] DataModule applies `Resize(320×320, area)` → `CenterSpatialCrop(256×256)` at load time — verify input `.npy` slices are compatible with this pipeline and that it works correctly for your data and not cropping important info.
- [ ] Stage 1 decoder output is **NOT clamped** — the reconstruction is returned directly from the decoder; note this differs from the Brain version which clamps to `[-3, 3]`. Nonetheless, it's up to you to decide based on task. 
- [ ] Stage 1 internal augmentation uses only 2 transforms (intensity + affine); you might need other or more intense augmentations based on task.
- [ ] A `results_v4_zscore.json` file is obtained and saved for each cohort. Easy to keep track of cohorts this way.
- [ ] ROC is computed from `sum_all_bars_score = Σ_slices(Final_Binary_sum_of_anomaly_maps)` per patient/case
- [ ] Make sure that `orig`/normal identifiers for "Normal test samples" are correct before ROC-analysis. Change naming otherwise. 
- [ ] Ground-truth anomaly categories are not used for model training, as only normal/reference shall be used for training. 
- [ ] No patient-identifying information is exposed in public logs, filenames, figures, or W&B runs. 

---

## 🔄 Key Differences from the Brain MRI Version

| Aspect | 🦴 Pelvic MRI (this folder) | 🧠 Brain MRI |
|---|---|---|
| **Anatomy/domain** | Pelvic MRI | Brain MRI |
| **Main data source** | LUND-PROBE-style pelvic MRI workflow | IXI + fastMRI-style brain workflow |
| **Stage 2 positional encoding** | **3D RoPE** over row, column, slice | 2D RoPE over row/column |
| **Codebook size** | 192 per RVQ level | 256 per RVQ level |
| **Main ROC score** | `sum_all_bars_score = Σ_slices(Final_Binary_sum_of_anomaly_maps)` | Same final-binary-only sum-all-bars definition |
| **Binary fusion** | ALM-A ∪ ALM-B ∪ LPIPS-backflow by default; edge erosion **disabled by default** | ALM-A ∪ ALM-B, optional LPIPS-backflow; edge erosion **disabled by default** |
| **Primary LPIPS reference** | Input-vs-healed/inpainted | Reconstruction-vs-healed/inpainted |
| **File format** | `.npy` slices | Primarily `.npz` with key `arr` |
| **Slice index importance** | Used for 3D RoPE and per-slice calibration | Not used for 2D RoPE |
| **Default qualitative output** | `_Final_ALM_Arithmetic.png` | Same |
| **`_Anomaly_Overlay.png`** | 6 panels incl. Binary+Token map (A∪B) | 7 panels incl. Binary+Token map (A∪B) (erosion panel unchanged when disabled) |
| **Decoder output clamping** | **NOT clamped** — decoder output returned directly | Clamped to `[-3, 3]` via `torch.clamp` |
| **DataModule preprocessing pipeline** | `np.rot90(k=-1)` at load time → `Resize(320×320, area)` → `CenterSpatialCrop(256×256)` | No resize/crop — images pre-sized during volume preprocessing; rotation applied at NIfTI export time |
| **Stage 1 internal augmentation** | **2 transforms**: intensity scaling (prob=0.33) + affine rotation ±5°/horiz-only translation (prob=0.33) — no flip, no contrast, no noise, no zoom | **5 transforms**: intensity scaling, contrast (gamma 0.5–1.5), Gaussian noise (prob=0.50), affine ±15°/±15 px bidirectional/zoom 0.8–1.2, flip (prob=0.50) |
| **Stage 2 mask implementation** | Python `for`-loop per sample in `_apply_mask()` and `_apply_block_mask()` | Fully vectorized tensor operations; no per-sample CPU syncs |
| **`_masked_ce()` fallback** | Falls back to all tokens when no masked tokens exist | Avoids fallback via `clamp(min=1)` denominator; never leaves the GPU |
| **MultiScaleEncoder code role** | Instantiated in **AYNU block** of `__init__`; `encode_multiscale()` in AYNU methods section; not on the CORE `encode_tokens()` path | Instantiated in **AYNU block** of `__init__`; `encode_multiscale()` in AYNU methods section; not on the CORE `encode_tokens()` path |
| **Dataset class name** | `NpySliceDataset` | `SliceDataset` |
| **DataModule directory input** | Single `data_dir` with seeded train/val split only | Supports separate `train_dir`/`val_dir` **or** single `data_dir` with seeded split |

---

## 📌 Practical Notes for GitHub Readers

> ⚕️ **This is research code, not a clinically validated tool.**
> **Do NOT use the model output for clinical decisions.**

<table>
<tr>
<td width="5%">🧩</td>
<td>The code assumes <b>2D slice-based processing</b>; patient-level evaluation is produced by aggregating slice scores.</td>
</tr>
<tr>
<td>🚫</td>
<td>Avoid slice-level train/test leakage. Splits should be <b>patient-level</b> whenever possible.</td>
</tr>
<tr>
<td>🔢</td>
<td>Preserve filename slice indices (<code>_slice_###</code>) because they affect Stage 2 slice conditioning and optional per-slice calibration lookup.</td>
</tr>
<tr>
<td>🧭</td>
<td>Be careful with <b>orientation, rotation, resizing, and category/case naming conventions</b> when preparing new data.</td>
</tr>
<tr>
<td>📂</td>
<td>Many defaults are local to the original workstation; <b>override paths explicitly</b>.</td>
</tr>
<tr>
<td>🔍</td>
<td>The <b>CORE/AYNU comments</b> in the Python files are meant to help readers identify exactly what contributes to the reported AUROC.</td>
</tr>
</table>


---

## Thresholding and fusion definitions

| Component | Continuous source | Thresholding | Binary/fusion result | Main CLI controls |
|---|---|---|---|---|
| **ALM(A)** LPIPS/healing arm | Iteration-0 `LPIPS(Input, Healed)` heatmap, usually `geomean` TTA aggregation | Z-score mask from calibration: `z = (LPIPS - μ) / (σ + ε)`, default `z_threshold=2.0`; then masked score is binarized by `--binary-threshold`, default `0.60` | `binary_mask_base_i = (heatmap × z_mask) > binary_threshold` | `--z-threshold`, `--smoothing-kernel`, `--heatmap-aggregation`, `--binary-threshold` |
| **ALM(B)** token-surprisal arm | Pseudo-PLL/NLL token-surprisal map from repeated random token masking | Values `<= --token-surprisal-clamp` are zeroed. Default clamp is `8.0`. In the final binary map, any remaining positive token pixel is included via `token_binary_i = token_surprisal_i > 0` | OR-unioned into ALM(A): `ALM(A) ∪ ALM(B)`; also stored separately as `token_surprisal_hot_px` | `--token-surprisal-samples`, `--token-surprisal-mask-ratio`, `--token-surprisal-clamp`, `--no-token-surprisal` |
| **LPIPS-backflow** | Final `LPIPS(Input, Inpainted)` map from targeted inpainting | `build_lpips_backflow_mask(lpips_map, selector)`. Current CLI default selector `(99, 0)` means per-slice 99th percentile cutoff; valid alternatives are `(percentile, 0)`, `(0, fixed_threshold)`, or scalar fixed threshold. The helper fallback in `run_inference_v4_zscore(...)` still keeps a legacy `(97, 0)` default for direct calls. | OR-unioned after ALM(A)/ALM(B): `ALM(A) ∪ ALM(B) ∪ LPIPS-backflow`; contributes directly to `Final_Binary_sum_of_anomaly_maps` | `--binary-include-lpips-backflow` enabled by default, `--no-binary-include-lpips-backflow`, `--LPIPS-in-inp-threshold-back-to-binary-token-map` |


## 🔁 LPIPS-Backflow Controls and Ablation

The current Pelvis experiment code includes LPIPS-backflow in the latest/default inference path. It mirrors the Brain binary-backflow mechanism, but uses the Pelvis-specific image pair:

```text
LPIPS(Input, Inpainted) → threshold → binary backflow mask → inserted into the final ALM mask counted by Final_Binary_sum_of_anomaly_maps
```


Selector behavior for LPIPS-backflow:

```python
# Example Percentile mode, current CLI default
--LPIPS-in-inp-threshold-back-to-binary-token-map "(99, 0)"
cutoff = percentile(LPIPS(Input, Inpainted), 99)
backflow_mask = LPIPS(Input, Inpainted) > cutoff

# Example Fixed-threshold mode
--LPIPS-in-inp-threshold-back-to-binary-token-map "(0, 0.60)"
cutoff = 0.60
backflow_mask = LPIPS(Input, Inpainted) > cutoff
```

The parser rejects selectors where both percentile and fixed threshold are positive, because that would be ambiguous.

### Useful ablation commands

Default/latest Pelvis inference with LPIPS-backflow enabled and fast quantitative inference:

```bash
python Final_Clean_to_Github_Pelvis/Inference_Pelvis_Experiments.py \
  --data-dir /path/to/pelvis_npy_folder \
  --output-dir /path/to/output \
  --device cuda:1 \
  --no-figures-only-json #this command saves time to get quantitative results and no plots. 
```

No-backflow ablation with all other defaults unchanged:

```bash
python Final_Clean_to_Github_Pelvis/Inference_Pelvis_Experiments.py \
  --data-dir /path/to/pelvis_npy_folder \
  --output-dir /path/to/output_no_backflow \
  --device cuda:1 \
  --no-binary-include-lpips-backflow \
  --no-figures-only-json
```

Change only the LPIPS-backflow thresholding rule:

```bash
python Final_Clean_to_Github_Pelvis/Inference_Pelvis_Experiments.py \
  --data-dir /path/to/pelvis_npy_folder \
  --output-dir /path/to/output_fixed_backflow \
  --device cuda:1 \
  --LPIPS-in-inp-threshold-back-to-binary-token-map "(0, 0.60)" \
  --no-figures-only-json
```

### JSON diagnostics for ablations

Each slice-level result now includes many metrics (useful for debugging):

```text
Final_Binary_sum_of_anomaly_maps                         final fused binary count used by ROC
Final_Binary_sum_of_anomaly_maps_Base                    ALM(A) binary count before token/backflow
Final_Binary_sum_of_anomaly_maps_Token                   ALM(B) token binary count
Final_Binary_sum_of_anomaly_maps_Overlap                 ALM(A) ∩ ALM(B) overlap
Final_Binary_sum_of_anomaly_maps_LPIPS_Backflow          LPIPS(Input, Inpainted) backflow binary count
Final_Binary_sum_of_anomaly_maps_Overlap_LPIPS_Backflow
Binary_Include_LPIPS_Backflow
LPIPS_in_inp_threshold_back_to_binary_token_map
LPIPS_in_inp_threshold_cutoff_value
```
---
