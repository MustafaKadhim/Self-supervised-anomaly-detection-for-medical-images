# 🧠 Two-Stage Anomaly Detection for Brain MRI

Unsupervised anomaly detection framework for T1-weighted brain MRI, adapted from the pelvic pipeline. The model trains exclusively on healthy subjects and detects anomalies at inference time by measuring divergence from the learned distribution.

**Datasets:** fastMRI / fastMRI+ / IXI

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Architecture](#architecture)
  - [Stage 1 — RVQ-VAE](#stage-1--rvq-vae)
  - [Stage 2 — Fact-biT](#stage-2--fact-bit)
- [Environment](#environment)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference Pipeline](#inference-pipeline)
- [Evaluation](#evaluation)
- [Key Differences vs. Pelvic MRI](#key-differences-vs-pelvic-mri)
- [Exact Replication Checklist](#exact-replication-checklist)
- [Code Audit Addendum](#code-audit-addendum)

---

| Stage | Model | Purpose |
|:-----:|:-----:|:--------|
| **Stage 1** | RVQ-VAE (ViT encoder + PixelShuffle decoder) | Learn tokens of healthy brain appearance |
| **Stage 2** | Factorized MaskGIT (bidirectional masked transformer) | Learn token distributions; estimate per-token surprise at inference |

---

## Repository Structure

```
Final_Code_Phiro_Brain_MRI/
├── 📄 FastMRI_model_stage1.py            # Stage 1: RVQ-VAE
├── 📄 FastMRI_model_stage2.py            # Stage 2: Fact-biT
├── 📄 Train_frameworks.py                # Training of both stages
├── 📄 FastMRI_Inference.py               # Detailed inference and visualization pipeline
├── 📄 Heatmaps_ideas_generator.py        # Colormap visualisation utility
├── 📄 FastMRI_ROC_Curve_Calculations.py  # Patient-level ROC analysis
├── 📄 IXI_dataset_overview_collection.py # IXI NIfTI → .npz pre-processing
├── 📄 collect_normal_slices.py           # Filter normal slices from annotation CSVs
├── 📄 Render_patient_slices_from_csv.py  # Render/export slices from CSV or label folders
├── 📄 dataset.py                         # Shared dataset utilities (external dependency)
└── 📄 config.yaml.yaml                   # Centralised configuration reference
```

> ⚠️ **Note:** `dataset.py` is imported by `Train_frameworks.py` but lives **outside** this folder. Ensure it is available in your Python path.

---

## Architecture

### Stage 1 — RVQ-VAE (`FastMRI_model_stage1.py`)

#### Architecture Flow

```
Input (B, 1, 256, 256)
    │
    ▼
┌─────────────────────────────────────┐
│  PatchEmbedding                     │
│  Conv2d(kernel=stride=patch_size) │
│  (B, 1, 256, 256) → (B, seq, dim) │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  ViTEncoder                         │
│  • Depth: 8                         │
│  • Heads: 8                         │
│  • Activation: GELU                 │
│  • Dropout: 0.1                     │
│  • Positional Embedding: learned    │
│    (randn × 0.02)                   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  ResidualVQ                         │
│  • Levels: 2                        │
│  • Codebook size: 256 per level     │
│  • kmeans_init, EMA decay: 0.85     │
│  • Orthogonal reg: 0.1              │
│  • Dead code threshold: 0.1         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PixelShuffleDecoder                │
│  • Stem: Conv2d → SiLU              │
│  • 3 ResBlocks (3×Conv + GN8 + SiLU)│
│  • Upsample: Conv → PixelShuffle×2  │
│  • num_upsample = log₂(patch_size)  │
│  • base_channels = embed_dim        │
└──────────────┬──────────────────────┘
               │
               ▼
    Output (clamped: [-3.0, 3.0])
```

> **Note:** `MultiScaleEncoder` is included but **not active**. When enabled, it builds a feature pyramid at strides 1/2/4 fused via cross-attention (8 heads).

#### Component Details

| Component | Configuration |
|:----------|:-------------|
| **Input** | Grayscale T1 brain MRI slice, `256×256` |
| **PatchEmbedding** | `Conv2d` with `kernel_size = stride = patch_size` |
| **ViTEncoder** | `TransformerEncoder`, depth=`8`, heads=`8`, `GELU`, dropout=`0.1` |
| **Positional Embedding** | Learned, initialized from `randn × 0.02` |
| **MultiScaleEncoder** | *(Included but unused)* Feature pyramid with `Conv2d` projections at stride `1/2/4`, fused via cross-attention (`8` heads) |
| **ResidualVQ** | 2-level residual quantization; `codebook_size=256` per level; `kmeans_init`, EMA decay=`0.85`, `orthogonal_reg_weight=0.1`, `threshold_ema_dead_code=0.1` |
| **PixelShuffleDecoder** | Stem → 3 residual blocks → PixelShuffle upsample → 1-ch head; `num_upsample=log₂(patch_size)`; `base_channels=embed_dim` (self-sizing) |
| **Output Clamp** | `torch.clamp(recon, -3.0, 3.0)` |

#### 🔑 Key differences between Brain vs. Pelvic pipelines

| Feature | **Brain (This)** | Pelvic |
|:--------|:----------------:|:------:|
| `codebook_size` | **`256`** | `192` |
| `embed_dim` | `256` | `256` |
| Augmentation | **Much Richer** | Basic |

**Augmentation structure via MONAI:**

```text
RandScaleIntensity(0.1, p=0.33)
    → RandAdjustContrast(γ∈[0.5, 1.5], p=0.33)
    → RandGaussianNoise(std=0.30, p=0.50)      ← Heavier noise
    → RandAffine(±15°, ±15px, zoom 0.8–1.2, p=0.33)
    → RandFlip(horizontal, p=0.5)
```

> 📸 **Sanity Check:** Augmentation preview saved to `FastMRI_RQC_ValExamples/augmentations_preview.png` on the first training batch.

#### Training Configuration

| Parameter | Value |
|:----------|:------|
| **Reconstruction Loss** | `L1` |
| **Perceptual Loss** | BiomedCLIP Cosine Feature Similarity (frozen vision encoder) |
| **Perceptual Weight** | `0.5` |
| **VQ Commitment Loss** | Weight = `0.25` |
| **Optimizer** | `AdamW(β=(0.9, 0.95), weight_decay=1e-4)` |
| **Scheduler** | `CosineAnnealingLR(T_max=max_epochs)` |
| **Gradient Clipping** | `1.0` |
| **Precision** | `32` (float32) |

#### Validation Visualization ***(sanity check)***

Every epoch, **4 random samples** are saved to `FastMRI_RQC_ValExamples/` as `2×2` panels:

| Position | Content |
|:---------|:--------|
| **Top-Left** | Input |
| **Top-Right** | Reconstruction |
| **Bottom-Left** | Q1 Codebook Indices |
| **Bottom-Right** | Q2 Codebook Indices |

*PSNR reported per sample.*

---

### Stage 2 — Fact-biT (`FastMRI_model_stage2.py`)

#### Architecture Flow

```
L1 Tokens ──→ l1_embed ──┐
                          ├──→ Task Embed ──→ Transformer Stack ──→ L1/L2 Logits
L2 Tokens ──→ l2_embed ──┤          ↑
                          │    2D RoPE (row, col)
Task ID  ──→ task_embed ──┘    RMSNorm + SwiGLU + SDPA
```

#### Component Details

| Component | Configuration |
|:----------|:-------------|
| **Positional Embedding** | **2D RoPE** (row + column only, no slice axis); head_dim split into `row_dim + col_dim` |
| **RotaryEmbedding2D** | `max_positions=seq_hw+1`, `base=25000`; precomputed cos/sin buffers |
| **RMSNorm** | `eps=1e-6`, replaces LayerNorm |
| **SwiGLU** | `w1/w2/w3` linear layers; `silu(w1(x)) * w2(x)` → `w3`; dropout=`0.0` |
| **TransformerBlockSDPA** | Pre-norm → QKV → 2D RoPE on Q,K → `F.scaled_dot_product_attention` → out_proj → pre-norm → SwiGLU |
| **Stack Depth** | `8` blocks, `8` heads |
| **Token Embeddings** | `l1_embed` (`codebook_size+1`, `embed_dim`), `l2_embed` (`codebook_size+1`, `embed_dim`), `task_embed` (`2`, `embed_dim`) |
| **seq_len** | `(image_size // patch_size)²` = `1024` (patch=`8`) |

> **Key Difference vs. Pelvic:** Brain MRI Stage 2 uses **2D RoPE** (row, col) rather than 3D RoPE (row, col, slice). The slice position argument is accepted but silently ignored for backward compatibility.

#### Masking Strategy

| Token | Masking |
|:------|:--------|
| **L1** | 70% of time: ratio ∈ [0.50, 0.75]; 30% of time: ratio ∈ [0.20, 0.50] |
| **L2** | ratio ∈ [0.15, 0.55] from `β(4, 4)` distribution |
| **Block Masking** | 50% of batches use random rectangles (union of overlapping blocks) instead of random masking |
| **Constraint** | All masks guarantee ≥1 masked token per sample (vectorised enforcement) |
| **Validation** | Random masking (fixed ratio=0.20) + center mask (inner 67%×67%) |

#### Training Configuration

| Parameter | Value |
|:----------|:------|
| **Loss** | `CE(L1 masked) + 0.25 × CE(L2 masked)` |
| **Label Smoothing** | `0.05` |
| **Optimizer** | `AdamW(β=(0.9, 0.98), wd=0.01)` |
| **Param Groups** | Separate groups (no decay on bias/norm/embed) |
| **Scheduler** | `LambdaLR`: 2000-step linear warmup → cosine decay |

#### Token Frequency Tracking

- Codebook utilisation
- Distribution entropy
- "Lift" (`acc − majority-class` baseline)

Logged every 1000 batches. Call `print_token_frequency_summary()` post-training to verify learning beyond mode-guessing.

---

## Environment

Install with pip (exact versions):

```bash
pip install torch==2.8.0+cu128 pytorch-lightning==2.5.5 monai==1.5.1 \
    vector-quantize-pytorch==1.27.15 transformers==4.57.2 \
    open_clip_torch==3.2.0 lpips==0.1.4 nibabel==5.3.2 imageio scipy tqdm matplotlib
```

> BiomedCLIP is loaded via `transformers` (`CLIPVisionModel`) with an automatic fallback to `open_clip`. The perceptual loss model is **frozen** during all training.

---

## Data Preparation

### Training Data: IXI Dataset (Healthy T1 Brain Volumes)

Use `IXI_dataset_overview_collection.py` to convert NIfTI volumes to `.npz` slices:

```bash
python IXI_dataset_overview_collection.py \
    --input-dir /path/to/IXI-T1/ \
    --output-npy-dir /path/to/Training_samples_FastMRI_IXI \
    --training-ready \
    --training-slice-start 128 \
    --training-slice-end 188 \
    --z-clip "-3,3" \
    --intensity-scale none \
    --pattern "*.nii.gz"
```

#### Preprocessing Pipeline per Slice

| Step | Operation |
|:-----|:----------|
| 1 | Load NIfTI, re-orient to closest-canonical axes (`nib.as_closest_canonical`) |
| 2 | Pad/crop in-plane to `256×256` (`center_crop_or_pad`) |
| 3 | Z-score normalise per volume: `(vol − μ) / σ`, clipped to `[−3, 3]` |
| 4 | Rotate 90° CCW (`np.rot90(arr, k=1)`) |
| 5 | Center crop or pad to `256×256` |
| 6 | Save as `.npz` with key `arr` (float32); naming: `{file_id}_slice_{idx:03d}.npz` |

> Slices `128–188` correspond to informative axial brain slices (avoiding skull cap / base-of-brain).

**Intensity scale options:** `none` (keep raw z-score), `minus1_1` (rescale to `[−1,1]`), `zero1` (rescale to `[0,1]`).

### FastMRI Validation / Test Data

FastMRI Brain (T1) slices are organized by patient. For each volume:
- Apply the same z-score normalisation and `256×256` sizing
- Save as `.npz` slices with the same naming convention

### Anomaly Annotation CSVs

For evaluation, annotations are stored in CSVs with columns:

| Column | Description |
|:-------|:------------|
| `file` | Patient identifier |
| `slice` | Slice index |
| `x, y, width, height` | Bounding box coordinates |
| `label` | Anomaly label |
| `study_level` | Global anomaly flag |
| `base_size` | Reference image dimensions |

### Example for Collecting Slices from fastMRI+

```bash
python collect_normal_slices.py \
    --annotation-csv path/to/brain.csv \
    --patient-list Annotated_FastMRI_Brains.csv \
    --series-type AXT1 \
    --slice-start 0 \
    --slice-end 10 \
    --png-root /path/to/Normal_Brains_pngs \
    --output-csv normal_slices_0_10.csv
```

A slice is classified as normal if:
- `study_level == "yes"`, **or**
- label contains the `--normal-label-keyword`, **or**
- it has no annotation at all (according to the official fastMRI+ guidelines). 

### Rendering Patient Slices

`Render_patient_slices_from_csv.py` bridges FastMRI `.h5` source volumes to 2D slice files/figures.

#### What the Script Does

| Step | Operation |
|:-----|:----------|
| Scan | Recursively scans `--data-root` for FastMRI `.h5` files; filters by series type (`AXT1`) |
| Input | Accepts CSV with `file[,slice,reason]` OR `--label-root` folder with patient subfolders |
| Process | Loads `reconstruction_rss`, pads/crops to `320×320`, per-volume z-score normalisation, clips `[−3,3]` |
| Transform | `np.flipud(...)`, center-crop to `320×320`, resize to `256×256` |
| Export | `.npz` (key: `arr`) and/or PNG previews with optional annotation overlays |
| Filter | `--include-label` and `--best-box-only` for curated anomaly slices |

#### Typical Outputs

```
label/
└── patient_xxx/
    ├── patient_xxx_slice_003.npz
    └── patient_xxx_slice_003.png
```

#### Example: Best Slice per Patient for a Selected Label "Mass"

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

### Building Anomaly Label Folders

#### Global Labels (Study-Level)

```bash
python build_patient_Global_label_folders.py \
    --anomalies-dir /path/to/FastMRI_Anomalies_Collection \
    --detailed-csv Annotated_fastMRI_Brains_Detailed.csv \
    --output-dir /path/to/FastMRI_Global_Anomalies_ByLabel \
    --use-detailed
```

**Default global labels:**
- `Motion artifact`
- `Possible artifact`
- `Colpocephaly`
- `Extra-axial collection`
- `Global label: Small vessel chronic white matter ischemic change` ***Excluded in this study***

#### Local Labels (Per-Slice Pathologies)

```bash
python build_patient_Local_label_folders.py \
    --anomalies-dir /path/to/FastMRI_Anomalies_Collection \
    --detailed-csv Annotated_fastMRI_Brains_Detailed.csv \
    --output-dir /path/to/FastMRI_Local_Anomalies_ByLabel
```

**Default local labels:**
`Edema`, `Enlarged ventricles`, `Craniotomy`, `Mass`, `Nonspecific lesion`, `Resection cavity`, `Intraventricular substance`, `Paranasal sinus opacification`, `Posttreatment change`, `Nonspecific white matter lesion`, `Encephalomalacia`, `Dural thickening`, `Absent septum pellucidum`, `Lacunar infarct`, `Likely cysts`

### Inference Data Directory Structure

The inference dataloader searches `--data-dir` recursively for `.npz` (or `.npy`) files. The `case_folder` field is set to the **immediate parent directory name**.

```
data_dir/
├── patient_abc/
│   ├── patient_abc_slice_003.npz
│   └── patient_abc_slice_004.npz
└── patient_xyz/
    └── patient_xyz_slice_005.npz
```

The `category` field defaults to `"FastMRI"`. Use `--category "Mass"` to tag all files with a specific anomaly label for stratified ROC analysis.

---

## Training

### Stage 1 Training

```bash
python Train_frameworks.py --stage1 \
    --train-dir /path/to/Training_samples_FastMRI_IXI \
    --val-dir   /path/to/Validation_samples_FastMRI \
    --file-ext .npz \
    --batch-size 192 \
    --num-workers 12 \
    --max-epochs 100 \
    --lr 2e-4 \
    --precision 32 \
    --augment
```

#### Hyperparameters

| Parameter | Value |
|:----------|:------|
| `embed_dim` | `256` |
| `codebook_size` | `256` (both levels) |
| `commitment_cost` | `0.25` |
| `perceptual_weight` | `0.5` |
| `lr` | `2e-4` |
| `batch_size` | `192` |
| `gradient_clip_val` | `1.0` |
| `precision` | `32` (float32) |
| GPU device | `[1]` |

Checkpoints saved to `FastMRI_IXI_Augmented_lightningCheckpoints/` as `FastMRI_stage1-{epoch:03d}-{val/loss:.4f}.ckpt`. Top-3 by `val/loss` are kept.

#### Fine-Tuning from Existing Checkpoint 
This could be powerful in case you work with very limited dataset samples and want the model to learn quickly. You can use your previous checkpoints instead of starting from scratch! (⚠️ make sure to have the same model design)  
```bash
python Train_frameworks.py --stage1 \
    --pretrained-stage1-ckpt /path/to/previous_stage1.ckpt \
    [... other args ...]
```

Loaded with `strict=False`; augmentations are then disabled during fine-tuning in needed; otherwise (`strict=True, use_augmentations=True`).

### Stage 2 Training

```bash
python Train_frameworks.py --stage2 \
    --train-dir /path/to/Training_samples_FastMRI_IXI \
    --val-dir   /path/to/Validation_samples_FastMRI \
    --file-ext .npz \
    --batch-size 158 \
    --num-workers 12 \
    --max-epochs 100 \
    --lr 2e-4 \
    --stage1-ckpt /path/to/FastMRI_stage1-epoch=099-val/loss=0.0891.ckpt \
    --augment
```

#### Hyperparameters

| Parameter | Value |
|:----------|:------|
| `embed_dim` | `256` |
| `codebook_size_level1/2` | `256` |
| `depth` | `8` |
| `num_heads` | `8` |
| `warmup_steps` | `2000` |
| `weight_decay` | `0.01` |
| `l2_loss_weight` | `0.25` |
| `q_error_weight` | `0.10` |
| `label_smoothing` | `0.05` |
| `batch_size` | `158` |
| `mask_ratio` (val) | `0.20` |
| `mask_ratio_min/max` (train) | `0.15` / `0.75` |

Stage 1 is loaded from `--stage1-ckpt`, frozen, and set to `eval()`. Stage 2 weights can optionally be warm-started via `--pretrained-stage2-ckpt` (very nice to use in case you want faster convergence and you have not changed the model design).

### WandB Logging of Experiments 

```bash
# Enable logging
python Train_frameworks.py --stage2 [args] \
    --wandb-project RVQ-MaskGIT-FastMRI-IXI \
    --wandb-run-name "Stage2-Augmented-FastMRI-IXI"

# Disable logging
python Train_frameworks.py --stage2 [args] --wandb-off
```

#### Logged Metrics (Stage 2)

| Category | Metrics |
|:---------|:--------|
| **Training** | `train/loss`, `train/loss_l1`, `train/loss_l2`, `train/acc_l1`, `train/acc_l2` |
| **Lift** | `train/lift_l1`, `train/lift_l2`, `train/baseline_l1`, `train/baseline_l2` |
| **Codebook** | `train/l1_codebook_utilization`, `train/l2_codebook_utilization`, `train/l1_entropy` |
| **Validation** | `val/loss`, `val/loss_center`, `val/acc_center`, `val/lift_center` |

---

## Inference Pipeline

### Recursive-AutoMask V4

#### Step 1 — Model Loading

```python
stage1, stage2 = load_models(stage1_ckpt, stage2_ckpt, device)
```

#### Step 2 — Calibration (You shall only include Normal/Refernce samples in this step)

```bash
python FastMRI_Inference.py \
    --calibration-mode \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --data-dir /path/to/healthy_slices \
    --output-dir /path/to/calib_output \
    --calibration-map /path/to/zscore_calibration.npz \
    --smoothing-kernel 15 \
    --heal-patterns "0,1"
```

**Per Normal/Refernce slice:**
1. Ensemble heal (checkerboard patterns, optional TTA flip enabled)
2. `LPIPS(original or reconstructed, healed)` → perceptual heatmap
3. Average-pool smooth with `kernel=15`
4. Accumulate across your Normal/Reference population → compute μ (mean) and σ (std) per pixel to obtain a calibration map. 

> ⚠️ **Critical:** `--smoothing-kernel` must be identical between calibration and inference.

#### Step 3 — Z-Score Inference

```bash
python FastMRI_Inference.py \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --data-dir /path/to/test_slices \
    --output-dir /path/to/results \
    --calibration-map /path/to/zscore_calibration.npz \
    --z-threshold 2.0 \
    --smoothing-kernel 15 \
    --num-iterations 3 \
    --heal-patterns "0,1" \        # You can choose whatever pattern you like :D
    --heatmap-aggregation mean \   # I have included other aggregation options in case you dont like "mean"
    --annotation-csv /path/to/brain.csv
```

### Output Fields per Slice for In-depth Analysis

| Field | Description |
|:------|:------------|
| `Binary_Sum_Heatmap`         | Pixel count of the fused binary detection mask |
| `clamped_pixel_sum`          | Sum of LPIPS values above clamp threshold |
| `token_surprisal_hot_px`     | Count of high-surprisal tokens upsampled to pixel space |
| `lpips_input_recon_sum_mask` | Total LPIPS inside the anomaly mask region |
| `anomaly_pixel_count`        | Pixel count of the anomaly mask |
| `has_ground_truth_bbox`      | Whether an annotation box is present (optional) |
| `num_true_positive_bboxes`   | Bounding-box TP count (ratio ≥ 10% overlap) (optional) |
| `inside_bbox_detection_ratio` | Mean fraction of GT box pixels that are flagged (optional) |
| `precision`, `f1_score`      | Per-slice localisation quality metrics (optional) |
| `sharpness_score`            | Laplacian variance as motion/blur artifact proxy (optional) |

Results saved to `results_v4_zscore.json` (one entry per slice) plus per-slice detailed PNG figures with analysis. The JSON file contains many score details that might be of interest for you, and can also be used for ROC-analysis below.  

## Evaluation

### ROC Curves and Detection Metrics

## 📊 ROC Analysis Implementation

> **Prerequisite:** This section describes the exact implementation found in `FastMRI_ROC_Curve_Calculations.py`. All scores, thresholds, and aggregation logic are documented here for exact replication.

---

### Overview

The ROC analysis operates at the **patient level**, not slice level. It uses `Binary_Sum_Heatmap` (or Binary+Token combined) as the anomaly score and treats **Test_Samples_FastMRI** patients as the normal class. Validation samples are excluded by default.

```
Slice-level JSON results
    │
    ├──→ Aggregate per patient (sum Binary_Sum_Heatmap)
    │
    ├──→ Classify patients:
    │       • Test_Samples_FastMRI → Normal (label=0)
    │       • Validation_samples → Excluded (default)
    │       • Everything else → Anomaly (label=1)
    │
    └──→ Compute ROC curve + AUC + bootstrap CI
```

### Score Definition: What Is the ROC Score?

The primary score for ROC generation is derived from **`Binary_Sum_Heatmap`** at the **slice level**, aggregated to the **patient level**.

#### Score Aggregation Formula

```python
# Per patient:
patient_score = Σ (Binary_Sum_Heatmap) across all slices for that patient. (Patient ID is derived from the filename stem before `_slice_`)
```


#### Binary+Token Combined Score (Optional)

When inference is run with `--binary-include-token-surprisal`, the `Binary_Sum_Heatmap` field already contains the fused Binary+Token value. However, the ROC-script also supports computing a combined score explicitly if needed:

```python
combined_score_per_slice = token_surprisal_hot_px + Binary_Sum_Heatmap
patient_combined_score = Σ combined_score_per_slice
```
---

### Patient Classification: Normal vs. Anomaly Ground Truth

#### Classification Rules

| Criterion | Label | Included in ROC |
|:----------|:-----:|:---------------|
| Patient ID or category contains `"test_samples_fastmri"` | `0` (Normal/Reference) | ✅ Yes |
| Patient ID or cetegory contains `"orig"`                 | `0` (Normal/Reference) | ✅ Yes |
| Patient ID or category contains `"validation_samples"`   | `0` (Normal)           | ❌ No (excluded by default) |
| Everything else                                          | `1` (Anomaly)          | ✅ Yes |

> **Critical:** The ROC uses **only Test samples** as negatives by default. Use `--include-validation-in-roc` to also include Validation samples if needed.

---

#### Step 2: Compute ROC Curve and AUC

```python
compute_fastmri_roc_and_auc(
    patient_scores=merged_patient_scores,
    expected_test_normals=30,           # Sanity check: expected # of test normals
    bootstrap_samples=2000,             # For confidence intervals
    confidence_level=0.95,
    bootstrap_random_seed=42,
    ci_fpr_grid_size=201,               # Grid resolution for CI band
)
```

Outputs:
- `sensitivity` = TP / (TP + FN)
- `specificity` = TN / (TN + FP)
- `FPR` = 1-specificity 
- `auc_mean`, `auc_std`, `auc_ci_lower`, `auc_ci_upper`
- `tpr_ci_lower`, `tpr_ci_upper`, `tpr_median` at each false-positive ratio (FPR) point

---

### Threshold Selection Strategies (Optional)

#### 1. Best Threshold by Youden's J Index

```python
best_threshold = max(roc_points, key=lambda p: p["youden_j"])
# youden_j = sensitivity - fpr
```

#### 2. Fixed FPR Threshold

```python
select_threshold_for_target_fpr(roc_metrics, target_fpr=0.20)
# Returns threshold with highest sensitivity where FPR <= target_fpr
```

#### 3. Fixed Threshold Evaluation

```python
evaluate_threshold_on_patient_scores(patient_scores, threshold=1019.0)
# Evaluates the specific threshold used during inference
```

---

### Output Files and Figures

| Output | Description |
|:-------|:------------|
| `FastMRI_ROC_binary_token_curve.png`                | ROC curve with AUC, bootstrap CI band, chance line |
| `FastMRI_ROC_Merged_BinaryToken.json`               | Merged patient scores and metadata |
| `FastMRI_ROC_binary_token_metrics.json`             | Full ROC metrics, thresholds, and evaluations |
| `FastMRI_ROC_Threshold_Table.csv`                   | Per-threshold TP/FP/TN/FN/sensitivity/FPR/precision |
| `FastMRI_ROC_Threshold_Table_page_*.png`            | Paginated threshold table figures |
| `FastMRI_Category_Stratified_Performance_Table.json`| Per-anomaly-category performance |
| `FastMRI_Category_Stratified_Performance_Table.csv` | CSV of category stratified results |
| `FastMRI_Category_Stratified_Performance_Table.png` | Visual table of category performance |


---

### Command-Line Usage

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--input`                     | `results_v4_zscore.json` | Path to single JSON or root folder |
| `--output-dir`                | Input parent | Output directory for all plots and tables |
| `--binary-token-patient-threshold` | `1019.0` | Fixed threshold for Binary+Token patient sum |
| `--expected-test-normal-cases` | `30` | Expected number of Test normal patients (sanity check) |
| `--roc-target-fpr`            | `0.20` | Target max FPR for recommended threshold |
| `--roc-ci-bootstrap-samples`  | `2000` | Bootstrap samples for AUC CI (0 to disable) |
| `--roc-ci-confidence-level`   | `0.95` | Confidence level for bootstrap CI |
| `--roc-ci-random-seed`        | `42` | Random seed for reproducible bootstrap (negative = non-deterministic) |
| `--roc-ci-fpr-grid-size`      | `201` | FPR grid resolution for CI band |
| `--include-validation-in-roc` | False | Include Validation_samples in ROC normal class |
| `--disable-fastmri-roc`       | False | Skip ROC/AUC generation entirely |
| `--show-best-j-marker`        | False | Show Best-Youden marker on ROC curve |
| `--category`                  | None | Filter to categories containing substring (repeatable) |
| `--case-folder`               | None | Filter to case_folders containing substring (repeatable) |
| `--top-n`                     | None | Limit to first N patients in plots |

#### Basic ROC Analysis (Single JSON)

```bash
python FastMRI_ROC_Curve_Calculations.py \
    --input /path/to/results_v4_zscore.json \
    --output-dir /path/to/roc_output \
    --expected-test-normal-cases 30 \
    --roc-target-fpr 0.20 \
    --roc-ci-bootstrap-samples 2000 \
    --roc-ci-confidence-level 0.95 \
    --roc-ci-random-seed 42
```

#### With Category Filtering

```bash
python FastMRI_ROC_Curve_Calculations.py \
    --input /path/to/results_v4_zscore.json \
    --category "Mass" \
    --expected-test-normal-cases 30 \
    --roc-target-fpr 0.20 \
    --roc-ci-bootstrap-samples 2000 \
    --roc-ci-confidence-level 0.95 \
    --roc-ci-random-seed 42
```

#### Include Validation in ROC Normal Class

```bash
python FastMRI_ROC_Curve_Calculations.py \
    --input /path/to/results_v4_zscore.json \
    --include-validation-in-roc \
    --expected-test-normal-cases 50
```

#### Disable ROC (Plots Only)

```bash
python FastMRI_ROC_Curve_Calculations.py \
    --input /path/to/results_v4_zscore.json \
    --disable-fastmri-roc
```
---


### Annotation Box Evaluation (Optional)
`I have included this option in case you want to investigate the overlap of your anomaly heatmaps with any available bounding boxes). This was inspired by Bercea, C.I et.al. 2025`

| Feature | Description |
|:--------|:------------|
| **Input** | `--annotation-csv` with columns: `file, slice, x, y, width, height, label, study_level, base_size` |
| **TP Criterion** | Predicted mask covers ≥5% of GT box area |
| **FP Ratio** | Predicted pixels outside healthy region / predicted pixels inside GT box |
| **Precision** | `TP / (TP + FP_ratio)` |
| **F1** | `2P / (P+1)` for `TP=1` |

---

## Code Audit Addendum

> This section documents implementation details discovered during code inspection that affect interpretation and reproducibility.

### What Is Actively Used vs. Reference Material

| Status | Files |
|:-------|:------|
| **Active workflow scripts** | `Train_frameworks.py`, `FastMRI_model_stage1.py`, `FastMRI_model_stage2.py`, `FastMRI_Inference.py`, `IXI_dataset_overview_collection.py`, `FastMRI_ROC_Curve_Calculations.py` |
| **Reference/config summary** | `config.yaml.yaml` — **not automatically loaded at runtime** |

### ⚠️ Important Implementation Clarifications

#### 1. In Our Work: Primary Anomaly Heatmap Is Calculated as Reconstruction-Healed 

| Stage | Reference |
|:------|:----------|
| Calibration           | `LPIPS(reconstruction, healed)` |
| Inference Iteration 0 | `LPIPS(reconstruction, healed)` |

Nonehteless, our inference script computes `lpips_input_recon` for additional auxiliary analysis only to allow you to get an idea about the performance.

#### 2. Script Defaults In This Work

Current defaults in `FastMRI_Inference.py`:

| Parameter | Default |
|:----------|:--------|
| `--z-threshold` | `"(-2.5, 6.0)"` (two-sided) |
| `--smoothing-kernel` | `7` |
| `--num-iterations` | `1` |
| `--inter-iteration-dilation` | `1` |
| `--heal-steps` | `6` |
| `--heal-temperature` | `0.9` |
| `--heal-patterns` | `"4"` |
| `--token-surprisal-samples` | `100` |
| `--token-surprisal-mask-ratio` | `0.820` |

