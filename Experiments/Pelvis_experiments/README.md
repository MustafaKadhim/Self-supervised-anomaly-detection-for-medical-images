<div align="center">

# 🦴 Two-Stage Unsupervised Anomaly Detection for Pelvic MRI

### *LUND-PROBE Implementation*

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
- [The AUROC/AUPRC Pipeline](#-the-aurocauprc-pipeline)
- [Repository Structure](#-repository-structure)
- [Environment Setup](#-environment-setup)

**🏗️ Architecture**
- [Method Overview](#-method-overview)
- [Stage 1: RVQ-VAE](#-stage-1--rvq-vae)
- [Stage 2: Fact-biT / Fact-biT](#-stage-2--factorized-maskgit--fact-bit)

</td>
<td width="50%" valign="top">

**📊 Data & Training**
- [Data Format](#-data-format)
- [Data Preparation](#-data-preparation)
- [Training](#-training)

**🔬 Inference & Evaluation**
- [Inference and Calibration](#-inference-and-calibration)
- [ROC / AUROC and PR / AUPRC Evaluation](#-roc--auroc-and-pr--auprc-evaluation)
- [Synthetic Anomaly Utilities](#-synthetic-anomaly-utilities)

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
<td>Code that <b>directly contributes</b> to the manuscript AUROC/AUPRC reproduction pipeline.</td>
</tr>
<tr>
<th width="15%">🟡 AYNU</th>
<td><b>"Available Yet Not AUROC-interesting"</b> — auxiliary code retained for transparency, debugging, training diagnostics, visualizations, calibration generation, alternative scores, and supplementary analyses, but not for the primary AUROC/AUPRC calculations.</td>
</tr>
</table>

> ⚠️ **Ground-truth anomaly labels are NOT used for model training.** They are used only for evaluation, cohort/category assignment, plotting, and optional annotation overlays.

---

## 🔄 The AUROC/AUPRC Pipeline

The primary patient-level ROC/PR path flows through the following stages:

```text
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT SLICE                             │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          Stage 1: RVQ-VAE  →  reconstruction image / tokens     │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      Stage 2:  Fact-biT  →  healing                             │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      LPIPS (input vs. healed/inpainted)  →  heatmap             │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      Binary + token-surprisal fusion                            │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌──────────────────────┐         ┌──────────────────────────────┐
│  Binary_Sum_Heatmap  │         │  token_surprisal_hot_px      │
│      (per slice)     │         │         (per slice)          │
└──────────┬───────────┘         └──────────────┬───────────────┘
           │                                    │
           └────────────────┬───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Patient-level:  sum_all_bars_score                             │
│  = Σ_slices (token_surprisal_hot_px + Binary_Sum_Heatmap)       │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ROC / AUROC and PR / AUPRC                     │
└─────────────────────────────────────────────────────────────────┘
```

### 🟢 CORE Output Fields

The two per-slice fields consumed by the main patient-level ROC/AUPRC pipeline are:

| Field | Role |
|---|---|
| `Binary_Sum_Heatmap` | Binary-positive masked LPIPS/healing heatmap count |
| `token_surprisal_hot_px` | Token-surprisal hot-pixel count after clamping |

`ROC_Curves_Calculations.py` aggregates these fields into a patient-level score:

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

Then `compute_patient_roc_and_auc(...)` computes patient-level ROC/AUROC and PR/AUPRC from `sum_all_bars_score`.

### 🟡 AYNU Examples

<details>
<summary><b>Click to expand — useful but not the primary AUROC/AUPRC score</b></summary>

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
│   ├── Model_Stage_2.py                         # Stage 2 Fact-biT / Fact-biT model
│   ├── Train_framework.py                       # PyTorch Lightning training entry point
│   ├── dataset.py                               # .npy slice Dataset/DataModule
│   ├── Inference_Pelvis_Experiments.py          # Recursive-AutoMask V4 inference + calibration
│   └── ROC_Curves_Calculations.py               # Patient-level ROC/AUPRC and category analyses
│
├── ⚙️  Configuration & Documentation
│   ├── config_yaml.yaml                         # Reference config summary (not auto-loaded)
│   ├── Instructions.md                          # Internal CORE/AYNU refactor blueprint
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
<td><b>Fact-biT / Fact-biT transformer</b></td>
<td>Learns token distributions and heals masked/suspect tokens using bidirectional masked prediction</td>
</tr>
</table>

### 🎛️ Recursive-AutoMask V4 (Inference)

At inference, **Recursive-AutoMask V4** computes two complementary anomaly signals:

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
</table>

Unlike the cleaned Brain README, the Pelvis inference code’s main LPIPS branch is **input-referenced**:

| Phase | LPIPS Reference |
|---|---|
| **Calibration** | LPIPS(input, healed) |
| **Inference iteration 0** | LPIPS(input, healed) |
| **Refinement iterations** | LPIPS(input, inpainted) |

> 💡 `lpips_input_recon` is computed for auxiliary diagnostics/visualizations, not for the main AUROC/AUPRC score.

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
| **Multi-scale encoder** | Present in the model and used by auxiliary/multiscale paths |
| **Quantizer** | `ResidualVQ`, 2 quantizers, codebook size 192 in the training script |
| **Decoder** | PixelShuffle decoder back to one image channel |
| **Forward output** | `recon`, `indices`, `commit_loss`, `quant_error_map` |

### Training Loss

```text
L1 reconstruction loss
+ BiomedCLIP perceptual loss (weight 0.9)
+ RVQ commitment loss
```

Training uses AdamW with a cosine annealing learning-rate schedule. Stage 1 also contains training-time augmentation and validation visualization code, which is AYNU relative to the inference AUROC/AUPRC path.

### Augmentations

<table>
<tr>
<td>↔️ Horizontal flip</td>
<td>🔄 Small rotation around ±5°</td>
<td>🎨 Stage 1 internal augmentation logic</td>
</tr>
</table>

> ⚠️ **Note:** In the current `Train_framework.py`, `SliceDataModule` is instantiated without passing `augment=True`, so DataModule augmentations are disabled unless the script is modified. Stage 1 also has its own internal augmentation logic in `training_step`. **Record the exact command/script state used for reproducibility.**

---

## 🧩 Stage 2 — Fact-biT / Fact-biT

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

> 🔒 **Reproducibility:** If you need exact checkpoint reproduction, use the pinned environment rather than unpinned latest package versions.

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

The downstream ROC code uses patient/case/category identifiers to stratify results into synthetic and clinical groups.

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

### Step 1️⃣ — Healthy / Reference Calibration

Calibration estimates per-pixel healthy/reference LPIPS statistics:

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
2. Compute Stage 1 reconstruction and RVQ tokens
        ↓
3. Compute token surprisal through repeated L1 token masking  →  ALM-B binary mask
        ↓
4. Heal checkerboard-masked tokens using Stage 2
        ↓
5. Compute LPIPS between input and healed image(s)  →  ALM-A binary mask
        ↓
6. Aggregate native and TTA heatmaps, commonly with geomean
        ↓
7. Smooth and threshold by Z-score using the calibration map
        ↓
8. Binary_Sum_Heatmap = (ALM-A ∪ ALM-B) white-pixel count ##known as unified heatmap in Figure 1.d) in paper
        ↓
9. Write per-slice JSON fields, including Binary_Sum_Heatmap and token_surprisal_hot_px
        ↓
10. Aggregate patient scores in ROC_Curves_Calculations.py
```

Targeted inpainting and multi-iteration refinement are implemented and preserved, but the current AUROC default is `--num-iterations 1`, so later refinement iterations are not part of the default CORE score.

### 📤 Output JSON Fields

Main output file: **`results_v4_zscore.json`**

| Field | Type | Meaning |
|---|:---:|---|
| `Binary_Sum_Heatmap` | 🟢 CORE | Count of binary-positive masked LPIPS pixels used in `sum_all_bars_score` |
| `token_surprisal_hot_px` | 🟢 CORE | Count of hot token-surprisal pixels after clamping used in `sum_all_bars_score` |
| `path`, `filename` | 🔹 meta | Source slice path/name |
| `category` | 🔹 meta | Cohort/anomaly category metadata |
| `case_folder` | 🔹 meta | Patient/case grouping metadata |
| `used_zscore` | 🔹 meta | Whether calibration Z-score thresholding was active |
| `z_threshold` | 🔹 meta | Z-score threshold used if active |
| `clamped_pixel_sum` | 🟡 AYNU | Auxiliary LPIPS-derived score; not the primary AUROC/AUPRC score |
| `lpips_input_recon_sum_mask` | 🟡 AYNU | Auxiliary reconstruction diagnostic |
| `sharpness_score`, `artifact_flag` | 🟡 AYNU | Auxiliary artifact/sharpness diagnostics |
| `iteration_metrics` | 🟡 AYNU | Auxiliary iteration/refinement diagnostics |

### 🖼️ CORE Visualization Outputs (per slice, saved by default)

| File | Panels | Purpose |
|---|---|---|
| `_Final_ALM_Heatmap.png` | 🟢 Input · Score · **ALM-A (LPIPS binary)** · **ALM-B (Token binary)** · Overlay · **A∪B combined mask** | Direct visual verification that the saved binary mask matches `Binary_Sum_Heatmap` |
| `_Anomaly_Overlay.png` | Input · Healed · Heatmap overlay · LPIPS+Token overlay · LPIPS raw · **Binary+Token map (A∪B)** | Clinical-style overlay with annotation boxes |
| `_full.png` | Multi-panel diagnostic figure | Full pipeline diagnostics (AYNU) |

> ✅ `_Final_ALM_Heatmap.png` is the recommended figure for verifying anomaly localization: the rightmost panel (A∪B combined mask) is the exact binary map whose white-pixel count equals `Binary_Sum_Heatmap`.

---

## 📈 ROC / AUROC and PR / AUPRC Evaluation

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
  = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

Labels are assigned from patient/case identifiers: `orig` cases are treated as normal/reference (`label = 0`) and all other cases as anomaly (`label = 1`). Confirm this naming convention before using the ROC script on new cohorts.

### 📊 Outputs

The ROC script can produce:

- Merged JSON payloads
- ROC curve figures
- Precision-recall curve figures
- AUROC/AUPRC metrics JSON
- Bootstrap confidence intervals
- Threshold tables
- Synthetic-vs-clinical split curves
- Category-stratified sensitivity tables

The script also contains many AYNU plotting functions for intermediate patient-level summaries.

---

## 📄 Reference Configuration File

`config_yaml.yaml` documents the intended experiment settings in a structured format. It is useful as a reference, but the inspected main Python scripts do **NOT** automatically load it as the sole runtime source of truth.

### Active behavior is controlled by:

1. CLI arguments
2. Hard-coded defaults inside Python scripts
3. Checkpoint hyperparameters
4. Filename conventions
5. The actual input JSON/data paths selected at runtime

### 💾 For reproducibility, keep:

- [x] Exact training commands
- [x] Exact inference commands
- [x] Checkpoint paths / checkpoint hashes if available
- [x] Calibration `.npz` file
- [x] `results_v4_zscore.json`
- [x] Merged ROC JSON/metrics outputs
- [x] The version of this code folder
- [x] The split manifest JSON

---

## 🧪 Synthetic Anomaly Utilities

The repository includes helper scripts for synthetic anomaly generation:

- `Simulation_inference_v4_extended_CJG.py`
- `Simluation_inference_v3_support_CJG.py`

These are not on the primary AUROC/AUPRC computation path, but they document and support generation of synthetic variations such as blur/noise/inserted structures used in the broader experimental workflow.

> 🔐 Be careful not to write identifiable DICOM metadata or patient information to public outputs when using these utilities.

---

## ✅ Exact Replication Checklist

Use this checklist when trying to reproduce the Pelvis experiment:

- [ ] Normal/reference NIfTI volumes were pre-sliced to `.npy` with the expected `_slice_###` naming convention
- [ ] Training/validation/test splits are recorded and reused
- [ ] Stage 1 checkpoint path is recorded
- [ ] Stage 2 checkpoint path is recorded
- [ ] Calibration data are normal/reference and independent from anomaly evaluation cohorts
- [ ] `--smoothing-kernel` is identical between calibration and inference
- [ ] Slice indices are preserved in filenames for 3D RoPE and per-slice calibration lookup
- [ ] The exact inference CLI command is saved
- [ ] `results_v4_zscore.json` is retained for each cohort
- [ ] ROC is computed from `sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)` per patient/case
- [ ] `orig`/normal identifiers are correct before assigning ROC labels
- [ ] Ground-truth/category labels are not used for model training
- [ ] No patient-identifying information is exposed in public logs, filenames, figures, or W&B runs

---

## 🔄 Key Differences from the Brain MRI Version

| Aspect | 🦴 Pelvic MRI (this folder) | 🧠 Brain MRI |
|---|---|---|
| **Anatomy/domain** | Pelvic MRI | Brain MRI |
| **Main data source** | LUND-PROBE-style pelvic MRI workflow | IXI + fastMRI-style brain workflow |
| **Stage 2 positional encoding** | **3D RoPE** over row, column, slice | 2D RoPE over row/column |
| **Codebook size** | 192 per RVQ level | 256 per RVQ level |
| **Main ROC score** | `sum_all_bars_score = Σ_slices(...)` | Same sum-all-bars definition |
| **Binary fusion** | ALM-A ∪ ALM-B (no erosion) | ALM-A ∪ ALM-B + edge erosion |
| **Primary LPIPS reference** | Input-vs-healed/inpainted | Reconstruction-vs-healed/inpainted |
| **File format** | `.npy` slices | Primarily `.npz` with key `arr` |
| **Slice index importance** | Used for 3D RoPE and per-slice calibration | Not used for 2D RoPE |
| **CORE heatmap output** | `_Final_ALM_Heatmap.png` (6 panels: input, score, ALM-A, ALM-B, overlay, A∪B) | Same |
| **`_Anomaly_Overlay.png`** | 6 panels incl. Binary+Token map (A∪B) | 7 panels incl. Binary+Token map (A∪B) and erosion effect |

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
<td>The <b>CORE/AYNU comments</b> in the Python files are meant to help readers identify exactly what contributes to the reported AUROC/AUPRC.</td>
</tr>
</table>

---

<div align="center">

### 🦴 Two-Stage Unsupervised Anomaly Detection — Pelvic MRI

*Research code for medical image analysis using deep learning.*

</div>
