<div align="center">

# 🧠 Two-Stage Unsupervised Anomaly Detection for Brain MRI

### *IXI / fastMRI Implementation*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-orange.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-red.svg)](https://monai.io/)
[![Research](https://img.shields.io/badge/Status-Research_Code-yellow.svg)]()

*A two-stage unsupervised anomaly-detection framework that learns only from normal T1-weighted brain MRI slices and detects deviations from the learned normal distribution.*

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
- [Stage 2: Fact-biT](#-stage-2--fact-bit)

</td>
<td width="50%" valign="top">

**📊 Data & Training**
- [Data Format](#-data-format)
- [Data Preparation](#-data-preparation)
- [Training](#-training)

**🔬 Inference & Evaluation**
- [Inference and Calibration](#-inference-and-calibration)
- [Annotation & Bounding Boxes](#-annotation-and-bounding-box-evaluation)
- [ROC / AUROC Evaluation](#-roc--auroc-evaluation)

**📋 Reference**
- [Reproducibility Checklist](#-exact-replication-checklist)
- [Differences from Pelvic Version](#-key-differences-from-the-pelvic-mri-version)
- [Practical Notes](#-practical-notes-for-github-readers)

</td>
</tr>
</table>

---

## 🎯 Overview & Core Concept

This repository contains the cleaned **Brain MRI implementation** of a two-stage unsupervised anomaly-detection framework. The training pipeline learns only from normal T1-weighted brain MRI slices, and the inference pipeline detects deviations from the learned normal distribution.

The code has been organized around a **CORE vs. AYNU** concept:

<table>
<tr>
<th width="15%">🟢 CORE</th>
<td>Code that <b>directly contributes</b> to the AUROC-producing pipeline.</td>
</tr>
<tr>
<th width="15%">🟡 AYNU</th>
<td><b>"Available Yet Not AUROC-interesting"</b> — auxiliary code retained for reproducibility, debugging, training diagnostics, visualizations, bounding-box analysis, and alternative scores, but not for the primary AUROC calculations.</td>
</tr>
</table>

> ⚠️ **Ground-truth anomaly labels and bounding boxes are NOT used for model training.** They are used only for evaluation, filtering, folder curation, and visualization.

---

## 🔄 The AUROC Pipeline

The primary AUROC path flows through the following stages:

```text
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT SLICE                              │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          Stage 1: RVQ-VAE  →  reconstruction / tokens            │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      Stage 2: Fact-biT  →  healing / inpainting                   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      LPIPS (reconstruction vs. healed)  →  heatmap               │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           Binary + token-surprisal fusion                        │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌──────────────────────┐         ┌──────────────────────────────┐
│  Binary_Sum_Heatmap  │         │  token_surprisal_hot_px      │
│      (per slice)     │         │         (per slice)           │
└──────────┬───────────┘         └──────────────┬───────────────┘
           │                                    │
           └────────────────┬───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Patient-level:  sum_all_bars_score                              │
│  = Σ_slices (token_surprisal_hot_px + Binary_Sum_Heatmap)        │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ROC  /  AUROC                              │
└─────────────────────────────────────────────────────────────────┘
```

### 🟢 CORE Output Fields

The per-slice fields consumed by the primary patient-level ROC pipeline:

| Field | Role |
|---|---|
| `Binary_Sum_Heatmap` | Binary/perceptual fused pixel count |
| `token_surprisal_hot_px` | Token-surprisal hot-pixel count |

`ROC_Curve_Calculations.py` aggregates these fields into a patient-level score:

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

This is intentionally identical to the pelvis pipeline definition. `binary_token_score` may still appear in Brain outputs as a **backward-compatible alias**, but it aliases the corrected `sum_all_bars_score` in the current code.

Then `compute_fastmri_roc_and_auc(...)` computes ROC/AUROC using `sum_all_bars_score` and cohort labels.

### 🟡 AYNU Examples

<details>
<summary><b>Click to expand — useful but not the primary AUROC score</b></summary>

- Bounding-box overlap metrics
- Per-slice precision / F1 localization metrics
- Reconstruction-quality figures
- Stage 2 alternative anomaly-map methods
- Token-frequency summaries
- Heatmap idea figures
- Per-patient bar plots and threshold tables
- `clamped_pixel_sum` and other auxiliary JSON fields

</details>

---

## 📁 Repository Structure

```text
Final_Clean_to_Github_Brain/
│
├── 🟢 CORE — Models & Pipeline
│   ├── Model_Stage1.py                      # Stage 1 RVQ-VAE model
│   ├── Model_Stage_2.py                     # Stage 2 Fact-biT model
│   ├── Train_framework.py                   # PyTorch Lightning training entry point
│   ├── dataset.py                           # Slice Dataset/DataModule for .npz/.png
│   ├── Inference_Brain_Experiments.py       # Recursive-AutoMask V4 inference + calibration
│   └── ROC_Curve_Calculations.py            # Patient-level ROC/AUROC metrics
│
├── ⚙️  Configuration & Documentation
│   ├── config_yaml.yaml                     # Reference config summary (not auto-loaded)
│   ├── Instructions_Brain.md                # Internal CORE/AYNU refactor instructions
│   └── Train_Val_Test_Exact_DataSplits_IXI_fastMRI.json   # Recorded data splits
│
└── 🔧 Data Preparation & Utilities
    ├── IXI_dataset_overview.py              # IXI NIfTI → training-ready .npz
    ├── Render_patient_slices_from_csv.py    # fastMRI .h5 → .npz/PNG rendering
    ├── collect_normal_slices.py             # Select normal slices from annotation CSVs
    ├── build_patient_Global_label_folders.py   # Build study/global-label folders
    ├── build_patient_Local_label_folders.py    # Build per-slice/local-label folders
    └── Inference_heatmaps_ideas_generator.py   # Optional heatmap visualization helper
```

> ⚠️ **Important:** Many scripts still contain absolute local default paths from the original experiment environment. For a new machine or GitHub user, **pass explicit CLI paths** instead of relying on defaults.

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
<td><b>RVQ-VAE</b><br><sub>ViT encoder, multi-scale encoder, residual vector quantization, PixelShuffle decoder</sub></td>
<td>Learns a discrete latent representation of normal brain appearance</td>
</tr>
<tr>
<td align="center"><b>2️⃣<br>Stage 2</b></td>
<td><b>Fact-biT transformer</b></td>
<td>Learns distributions over Stage 1 codebook tokens and heals masked/suspect tokens</td>
</tr>
</table>

### 🎛️ Recursive-AutoMask V4 (Inference)

At inference, **Recursive-AutoMask V4** performs calibration, healing, perceptual comparison, binary-mask fusion, and optional targeted inpainting. The main heatmap branch is **reconstruction-referenced**:

| Phase | LPIPS Reference |
|---|---|
| **Calibration** | LPIPS(Stage 1 reconstruction, healed reconstruction) |
| **Inference iteration 0** | LPIPS(Stage 1 reconstruction, healed reconstruction) |
| **Refinement iterations** | LPIPS(Stage 1 reconstruction, inpainted reconstruction) |

> 💡 The script also computes `lpips_input_recon`, but this is mainly for auxiliary visualization/analysis rather than the primary AUROC path.

---

## 🧩 Stage 1 — RVQ-VAE

📄 **File:** `Model_Stage1.py`

`Stage1RVQVAE` maps a 2D grayscale MRI slice to RVQ tokens and reconstructs the image.

### Architecture

| Component | Implementation detail |
|---|---|
| **Input** | Single-channel 2D slice, typically `1 × 256 × 256` |
| **Patch embedding** | `Conv2d` with `kernel_size = stride = patch_size` |
| **Default patch size** | `8` → `32 × 32 = 1024` tokens |
| **Encoder** | ViT-style Transformer encoder, depth 8, 8 heads |
| **Multi-scale encoder** | Convolutional feature pyramid fused with attention |
| **Quantizer** | `ResidualVQ`, 2 quantizers, codebook size 256 |
| **Decoder** | PixelShuffle decoder, output clamped to `[-3, 3]` |
| **Forward output** | `recon`, `indices`, `commit_loss`, `quant_error_map` |

### Training Loss

```text
L1 reconstruction loss
+ BiomedCLIP perceptual loss (weight 0.5)
+ RVQ commitment loss
```

### MONAI Augmentations (when enabled)

<table>
<tr>
<td>🎨 Intensity scaling</td>
<td>🌗 Contrast adjustment</td>
<td>📡 Gaussian noise</td>
</tr>
<tr>
<td>🔄 Affine rotation/translation/zoom</td>
<td>↔️ Horizontal flip</td>
<td></td>
</tr>
</table>

> ⚠️ **Note:** In `Train_framework.py`, Stage 1 is instantiated with `use_augmentations=False` in the current script body even though the DataModule can apply augmentations through `--augment`. The active augmentation source depends on the training command/script settings. **Record the exact command used for reproducibility.**

---

## 🧩 Stage 2 — Fact-biT

📄 **File:** `Model_Stage_2.py`

`FactorizedMaskGIT` predicts masked RVQ tokens from Stage 1.

### Architecture

| Component | Implementation detail |
|---|---|
| **Token levels** | Separate L1 and L2 token embeddings |
| **Codebook size** | 256 per level (loaded from trained Stage 1) |
| **Transformer** | SDPA transformer with RMSNorm and SwiGLU |
| **Position encoding** | 2D rotary embeddings over row/column token positions |
| **Sequence length** | Derived from Stage 1 image size and patch size; typically 1024 |
| **Stage 1 during training** | Frozen and set to eval mode |

### Training Loss

```text
CE(masked L1 tokens) + 0.25 × CE(masked L2 tokens)
```

With **label smoothing** (`0.05`) and a **mixed random/block masking strategy**.

> ⚠️ **Backward compatibility note:** Some comments/docstrings in the model still mention "3D RoPE" for the earlier pelvis/volumetric code. The Brain implementation uses **2D RoPE** (row and column only). `slice_pos` may be accepted by call signatures but is not part of the Brain positional encoding.

---

## 🛠️ Environment Setup

The original experiment used PyTorch Lightning, MONAI, vector quantization, LPIPS, and medical/scientific Python packages.

```bash
pip install torch pytorch-lightning monai vector-quantize-pytorch \
    transformers open_clip_torch lpips nibabel h5py pandas pillow \
    numpy scipy scikit-learn matplotlib tqdm imageio
```

> 🔒 **Reproducibility:** If you need exact reproduction, pin versions from the environment used for the experiment. The README intentionally does not guarantee that the newest package versions will reproduce old checkpoints exactly.

### 📝 Notes

- Stage 1 training can use **BiomedCLIP** via `transformers` / `open_clip_torch`.
- Inference uses **LPIPS** (`lpips` package, VGG backbone) for the spatial perceptual heatmap.
- CUDA device defaults in scripts may point to `cuda:1` → override with CLI arguments if needed.

---

## 📦 Data Format

### Training / Inference Slice Files

The standard saved array format:

```text
.npz file containing key: arr
```

where `arr` is a 2D `float32` image slice.

- `dataset.py` supports `.npz` and image files such as `.png` for training/validation (via `--file-ext`).
- The inference dataloader in `Inference_Brain_Experiments.py` **recursively** supports `.npz` and `.npy` files.

### Recommended Directory Structure for Inference

The inference script stores the immediate parent folder name as `case_folder` in the output JSON. For patient-level aggregation, use **one folder per patient/case**:

```text
data_dir/
├── patient_001/
│   ├── patient_001_slice_003.npz
│   └── patient_001_slice_004.npz
└── patient_002/
    └── patient_002_slice_005.npz
```

`ROC_Curve_Calculations.py` can then aggregate slices by patient/case and filter by `case_folder` or category.

---

## 🧪 Data Preparation

### 🔹 IXI normal Training Data

Convert IXI T1 NIfTI volumes to 2D `.npz` slices:

```bash
python IXI_dataset_overview.py \
    --input-dir /path/to/IXI-T1 \
    --output-npy-dir /path/to/Training_samples_FastMRI_IXI \
    --training-ready \
    --training-slice-start 128 \
    --training-slice-end 188 \
    --z-clip "-3,3" \
    --intensity-scale none \
    --pattern "*.nii.gz" \
    --recursive
```

<details>
<summary><b>📋 Main Preprocessing Steps</b></summary>

1. Load NIfTI volume
2. Reorient to closest canonical orientation
3. Z-score normalize per volume
4. Clip to the requested range, commonly `[-3, 3]`
5. Crop/pad in-plane to `256 × 256`
6. Rotate exported slices with `np.rot90(..., k=1)`
7. Save `.npz` files with key `arr`

</details>

> 💡 The default slice range `128–188` was used to focus on **informative axial brain slices** and avoid many non-informative superior/inferior slices.

### 🔹 fastMRI Rendering / Anomaly Folder Preparation

`Render_patient_slices_from_csv.py` converts fastMRI `.h5` volumes into `.npz` and/or PNG slices. It can read either:

- a CSV containing patient/slice requests, or
- a label-root folder produced by the anomaly folder builders.

```bash
python Render_patient_slices_from_csv.py \
    --label-root /path/to/FastMRI_Local_Anomalies_ByLabel \
    --include-label "Mass" \
    --best-box-only \
    --data-root /path/to/fastMRI_h5_root \
    --series-type AXT1 \
    --output-dir /path/to/rendered_pngs \
    --output-npy-dir /path/to/rendered_npz \
    --annotation-csv /path/to/Annotated_fastMRI_Brains_Detailed.csv
```

> ⚠️ **Important preprocessing difference:**
> - **IXI training slices** come from NIfTI files.
> - **fastMRI anomaly/evaluation slices** come from `.h5` `reconstruction_rss` volumes, are normalized per volume, may be vertically flipped for orientation/display consistency, then resized/cropped to the saved 2D representation.

### 🔹 Normal-Slice Collection for Calibration

```bash
python collect_normal_slices.py \
    --annotation-csv /path/to/Annotated_fastMRI_Brains_Detailed.csv \
    --patient-list /path/to/Annotated_FastMRI_Brains.csv \
    --series-type AXT1 \
    --slice-start 0 \
    --slice-end 5 \
    --png-root /path/to/normal_png_root \
    --output-csv /path/to/normal_slices.csv
```

A slice can be treated as normal if it is study-level normal, contains the configured normal-label keyword, or has no annotation depending on script settings.

### 🔹 Global and Local Anomaly Label Folders

<details>
<summary><b>📁 Global / Study-level Labels</b></summary>

```bash
python build_patient_Global_label_folders.py \
    --anomalies-dir /path/to/FastMRI_Anomalies_Collection \
    --detailed-csv /path/to/Annotated_fastMRI_Brains_Detailed.csv \
    --output-dir /path/to/FastMRI_Global_Anomalies_ByLabel \
    --use-detailed
```

</details>

<details>
<summary><b>📁 Local / Per-slice Labels</b></summary>

```bash
python build_patient_Local_label_folders.py \
    --anomalies-dir /path/to/FastMRI_Anomalies_Collection \
    --detailed-csv /path/to/Annotated_fastMRI_Brains_Detailed.csv \
    --output-dir /path/to/FastMRI_Local_Anomalies_ByLabel
```

</details>

These scripts create label-specific folders and patient/slice CSVs useful for running inference category by category.

---

## 🚂 Training

Entry point: **`Train_framework.py`**

### 1️⃣ Stage 1 Training

```bash
python Train_framework.py --stage1 \
    --train-dir /path/to/Training_samples_FastMRI_IXI \
    --val-dir /path/to/Validation_samples_FastMRI \
    --file-ext .npz \
    --batch-size 192 \
    --num-workers 12 \
    --max-epochs 100 \
    --lr 2e-4 \
    --precision 32 \
    --augment
```

**Current script-level constants/defaults:**

| Setting | Value |
|---|---:|
| `embed_dim` | 256 |
| `codebook_size` | 256 |
| `commitment_cost` | 0.25 |
| learning rate | `2e-4` |
| default max epochs | 100 |
| checkpoint monitor | `val/loss` |
| checkpoint top-k | 3 |
| trainer device default | `[1]` |

> 📌 Checkpoints are written under the hard-coded experiment checkpoint directory unless the script is edited.

### 2️⃣ Stage 2 Training

```bash
python Train_framework.py --stage2 \
    --train-dir /path/to/Training_samples_FastMRI_IXI \
    --val-dir /path/to/Validation_samples_FastMRI \
    --file-ext .npz \
    --batch-size 158 \
    --num-workers 12 \
    --max-epochs 100 \
    --lr 2e-4 \
    --stage1-ckpt /path/to/stage1.ckpt \
    --augment
```

> ⚠️ Stage 2 requires a valid Stage 1 checkpoint. **The Stage 1 model is frozen during Stage 2 training.**

### 📊 Logging

`Train_framework.py` uses:
- `CSVLogger`
- Optional `WandbLogger`
- `LearningRateMonitor`
- `ModelCheckpoint`

Disable W&B with `--wandb-off`.

> 🔐 **Privacy note:** Do not log patient-identifiable information to W&B, filenames, plots, or shared logs.

---

## 🔬 Inference and Calibration

Main script: **`Inference_Brain_Experiments.py`**

### 🔧 Model Loading

`load_models(stage1_ckpt, stage2_ckpt, device)` loads Stage 1 and Stage 2 checkpoints. Stage 1 perceptual-loss keys are stripped during inference loading, so **BiomedCLIP is not required for inference**.

### Step 1️⃣ — normal Calibration

Calibration estimates per-pixel normal LPIPS statistics:

```text
mu[h, w]    = mean LPIPS (reconstruction vs. healed) over normal calibration slices
sigma[h, w] = std  LPIPS (reconstruction vs. healed) over normal calibration slices
```

```bash
python Inference_Brain_Experiments.py \
    --calibration-mode \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --data-dir /path/to/normal_calibration_slices \
    --output-dir /path/to/calibration_output \
    --calibration-map /path/to/zscore_calibration.npz \
    --smoothing-kernel 7 \
    --heal-patterns "4" \
    --device cuda:0
```

Calibration writes an audit file `calibration_input_files.txt`. **Keep this file** with the experiment outputs because it records which slices were used for calibration.

> 🚨 **Critical reproducibility rule:** Use the **same** `--smoothing-kernel` during calibration and inference.

### Step 2️⃣ — Anomaly Inference

```bash
python Inference_Brain_Experiments.py \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --data-dir /path/to/test_or_anomaly_slices \
    --output-dir /path/to/inference_results \
    --calibration-map /path/to/zscore_calibration.npz \
    --z-threshold "(-2.5, 6.0)" \
    --smoothing-kernel 7 \
    --num-iterations 1 \
    --heal-patterns "4" \
    --token-surprisal-samples 100 \
    --token-surprisal-mask-ratio 0.820 \
    --annotation-csv /path/to/Annotated_fastMRI_Brains_Detailed.csv \
    --device cuda:0
```

The script supports many additional switches for filtering, batch running over label folders, annotation coordinate handling, TTA, binary fusion, LPIPS backflow, visualization, and output control.

> 💾 **Because defaults are experiment-specific, save the exact CLI command with each run.**

### ⚙️ Current Important Inference Defaults

| Argument | Current default |
|---|---:|
| `--z-threshold` | `"(-2.5 , 6.0)"` |
| `--smoothing-kernel` | `7` |
| `--num-iterations` | `1` |
| `--inter-iteration-dilation` | `1` |
| `--heal-steps` | `6` |
| `--heal-temperature` | `0.9` |
| `--heal-patterns` | `"4"` |
| `--token-surprisal-samples` | `100` |
| `--token-surprisal-mask-ratio` | `0.820` |
| `--inpaint-steps` | `12` |
| `--inpaint-temperature` | `0.5` |
| `--batch-size` | `320` |
| `--device` | `cuda:1` |

> ⚠️ These are **not necessarily** identical to paper-like/recommended ablation settings. Treat them as the code defaults at the time of this README update.

### 🔄 Main Inference Steps

Inside `recursive_automask_v4_zscore(...)`, the CORE flow is:

```text
1. Compute Stage 1 reconstruction and RVQ tokens
        ↓
2. Compute sharpness map/score (motion-blur awareness)
        ↓
3. Compute token surprisal via Monte Carlo token masking  →  ALM-B binary mask
        ↓
4. Heal masked token patterns using Stage 2 MaskGIT
        ↓
5. Compute LPIPS (Stage 1 recon ↔ healed/inpainted recon)  →  ALM-A binary mask
        ↓
6. Aggregate ensemble heatmaps
        ↓
7. Threshold with calibration Z-score (or fallback percentile logic)
        ↓
8. Optionally refine with targeted token inpainting
        ↓
9. Binary_Sum_Heatmap = (ALM-A ∪ ALM-B) + edge cleanup
        ↓
10. Write per-slice JSON fields (including Binary_Sum_Heatmap and token_surprisal_hot_px)
```

### 📤 Output JSON Fields

Main output file: **`results_v4_zscore.json`**

| Field | Type | Meaning |
|---|:---:|---|
| `Binary_Sum_Heatmap` | 🟢 CORE | Per-slice binary/perceptual fused pixel count used in `sum_all_bars_score` |
| `token_surprisal_hot_px` | 🟢 CORE | Per-slice token-surprisal hot-pixel count used in `sum_all_bars_score` |
| `Binary_Sum_Heatmap_Base` | 🟡 AYNU | Base binary component |
| `Binary_Sum_Heatmap_Token` | 🟡 AYNU | Token-surprisal binary component |
| `Binary_Sum_Heatmap_Overlap` | 🟡 AYNU | Overlap component |
| `case_folder` | 🔹 meta | Immediate parent folder of the slice file |
| `category` | 🔹 meta | CLI-supplied or inferred batch category |
| `sharpness_score` | 🟡 AYNU | Motion/blur-related auxiliary score |
| `clamped_pixel_sum` | 🟡 AYNU | Auxiliary LPIPS-derived score; **not the primary AUROC score** |
| `has_ground_truth_bbox` | 🟡 AYNU | Whether a matched annotation box exists |
| `num_true_positive_bboxes` | 🟡 AYNU | Auxiliary bounding-box localization count |
| `inside_bbox_detection_ratio` | 🟡 AYNU | Auxiliary localization ratio |
| `precision`, `f1_score` | 🟡 AYNU | Auxiliary per-slice localization metrics |

### 🖼️ CORE Visualization Outputs (per slice, saved by default)

| File | Panels | Purpose |
|---|---|---|
| `_Final_ALM_Heatmap.png` | 🟢 Input · Score · **ALM-A (LPIPS binary)** · **ALM-B (Token binary)** · Overlay · **A∪B combined mask** | Direct visual verification that the saved binary mask matches `Binary_Sum_Heatmap` |
| `_Anomaly_Overlay.png` | Input · Healed · Heatmap overlay · LPIPS+Token overlay · LPIPS raw · **Binary+Token map (A∪B)** · Erosion effect | Clinical-style overlay with annotation boxes and detection status |
| `_full.png` | Multi-panel diagnostic figure | Full pipeline diagnostics (AYNU) |

> ✅ `_Final_ALM_Heatmap.png` is the recommended figure for verifying anomaly localization: the rightmost panel (A∪B combined mask) is the exact binary map whose white-pixel count equals `Binary_Sum_Heatmap`.

---

## 🎯 Annotation and Bounding-Box Evaluation

Annotation CSVs are expected to contain columns such as:

```text
file, slice, x, y, width, height, label, study_level, base_size
```

### Coordinate Preprocessing Modes

`Inference_Brain_Experiments.py` supports multiple modes:

- `legacy`
- `render_fastmri`
- `mask_pipeline`

### Optional Annotation Flips

- `--annotation-flip-vertical`
- `--annotation-flip-horizontal`

> 📌 Bounding-box metrics are useful for **localization analysis** but are AYNU relative to the primary patient-level AUROC path.

---

## 📈 ROC / AUROC Evaluation

Main script: **`ROC_Curve_Calculations.py`**

```bash
python ROC_Curve_Calculations.py \
    --input /path/to/results_v4_zscore.json \
    --output-dir /path/to/roc_outputs
```

### 🟢 CORE Aggregation

Performed by `aggregate_fastmri_binary_token_patient_scores(...)`, which sums **both** `token_surprisal_hot_px` and `Binary_Sum_Heatmap` over all included slices for each patient/case:

```text
sum_all_bars_score
  = Σ_slices (token_surprisal_hot_px + Binary_Sum_Heatmap)
```

`compute_fastmri_roc_and_auc(...)` then computes patient-level ROC/AUROC using:

| Variable | Definition |
|---|---|
| `score` | `sum_all_bars_score` |
| `label = 0` | Included test-normal patients |
| `label = 1` | Anomaly patients |

> 📌 Validation normals may be excluded depending on script options/policy. This prevents validation-normal slices from being mixed into the final test-normal ROC cohort unless intentionally enabled.

### ⚠️ Deprecation Notice

> Older README text referred to patient-level aggregation of `clamped_pixel_sum`, or to using only `Binary_Sum_Heatmap` / `binary_token_score`. **That is not the CORE AUROC path in the current cleaned Brain code.**
>
> The primary ROC path uses:
> ```text
> sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
> ```
> `binary_token_score` is retained only as a backward-compatible alias for this corrected combined score.

---

## 📄 Reference Configuration File

`config_yaml.yaml` is a structured reference summary of experiment settings. It is **useful documentation**, but the main Python scripts inspected here do **NOT** automatically load it as the runtime source of truth.

### Active behavior is controlled by:

1. CLI arguments
2. Hard-coded defaults inside the Python scripts
3. Checkpoint hyperparameters
4. The actual data files selected at runtime

### 💾 For reproducibility, keep:

- [x] The exact CLI command
- [x] The checkpoint paths / checkpoint hashes if available
- [x] `calibration_input_files.txt`
- [x] The produced `results_v4_zscore.json`
- [x] The version of this code folder
- [x] The train/validation/test split JSON if relevant

---

## ✅ Exact Replication Checklist

Use this checklist when trying to reproduce the Brain experiment:

- [ ] IXI normal T1 volumes were preprocessed with `IXI_dataset_overview.py`
- [ ] Saved training arrays are `.npz` files with key `arr`
- [ ] Training/validation/test patient or slice splits are recorded and reused
- [ ] Stage 1 checkpoint path is recorded
- [ ] Stage 2 checkpoint path is recorded
- [ ] Calibration slices are normal/normal and independent from anomaly evaluation data
- [ ] `--smoothing-kernel` is identical between calibration and inference
- [ ] The exact inference CLI command is saved
- [ ] `calibration_input_files.txt` is retained
- [ ] `results_v4_zscore.json` is retained
- [ ] ROC is computed from `sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)` per patient/case
- [ ] Ground-truth labels are used only for evaluation, not training/calibration model fitting
- [ ] No patient-identifying information is exposed in public logs, filenames, figures, or W&B runs

---

## 🔄 Key Differences from the Pelvic MRI Version

| Aspect | 🧠 Brain MRI (this folder) | 🦴 Pelvic Version |
|---|---|---|
| **Anatomy/domain** | Brain MRI | Pelvic MRI |
| **Data sources** | IXI normal + fastMRI-style brain evaluation | LUND-PROBE + clinical pelvis workflow |
| **Stage 2 positional encoding** | 2D RoPE (row/column) | 3D RoPE |
| **Codebook size** | 256 per RVQ level | 192 per level |
| **Main ROC score** | `sum_all_bars_score = Σ_slices(...)` | Same sum-all-bars definition |
| **Binary fusion** | ALM-A ∪ ALM-B + edge erosion | ALM-A ∪ ALM-B (no erosion) |
| **Primary LPIPS reference** | Reconstruction-vs-healed/inpainted | Input-vs-healed/inpainted |
| **File format** | Primarily `.npz` with key `arr` | `.npy` slices |
| **CORE heatmap output** | `_Final_ALM_Heatmap.png` (6 panels: input, score, ALM-A, ALM-B, overlay, A∪B) | Same |
| **`_Anomaly_Overlay.png`** | 7 panels incl. Binary+Token map (A∪B) and erosion effect | 6 panels incl. Binary+Token map (A∪B) |

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
<td>🧭</td>
<td>Be careful with <b>orientation, flipping, resizing, and annotation coordinate modes</b> when comparing heatmaps to boxes.</td>
</tr>
<tr>
<td>📂</td>
<td>Many defaults are local to the original workstation; <b>override paths explicitly</b>.</td>
</tr>
<tr>
<td>🔍</td>
<td>The <b>CORE comments</b> in the Python files are meant to help readers identify exactly what contributes to the reported AUROC.</td>
</tr>
</table>

---

<div align="center">

### 🧠 Two-Stage Unsupervised Anomaly Detection — Brain MRI

*Research code for medical image analysis using deep learning.*

</div>
