# 🧠 Two-Stage Anomaly Detection for Brain MRI

Unsupervised anomaly detection framework for T1-weighted brain MRI, adapted from the pelvic pipeline. The model trains exclusively on healthy subjects and detects anomalies at inference time by measuring divergence from the learned distribution. Datasets used are **(fastMRI/fastMRI+/IXI)**.

---

## Overview

The framework consists of two sequentially trained stages:

| Stage | Model | Purpose |
|-------|-------|---------|
| **Stage 1** | RVQ-VAE (ViT encoder + PixelShuffle decoder) | Learn a discrete codebook of healthy brain appearance |
| **Stage 2** | Factorized MaskGIT (bidirectional masked transformer) | Learn joint token distributions; estimate per-token surprise at inference |

At inference, the **Recursive-AutoMask V4** pipeline applies ensemble healing, LPIPS-based perceptual comparison, Z-score calibration against a healthy-volunteer population, and targeted inpainting over multiple iterations. Anomaly scoring also incorporates token surprisal (pseudo-PLL) computed independently of the healing branch.

---

## Repository Structure

```
Final_Code_Phiro_Brain_MRI/
├── FastMRI_model_stage1.py                              # Stage 1: RVQ-VAE
├── FastMRI_model_stage2.py                              # Stage 2: Factorized MaskGIT
├── FastMRI_train.py                                     # Training entry-point
├── Inference_FastMRI_SOTA_5p_Rec_Heal_5p0_automatic_final.py  # Inference pipeline
├── Inference_heatmaps_ideas_generator.py                # Colormap visualisation utility
├── fastMRI_ROC_Curve_Calculations.py                    # Patient-level ROC / detection metrics
├── IXI_dataset_overview.py                              # IXI NIfTI → .npz pre-processing
├── collect_normal_slices.py                             # Filter normal slices from annotation CSVs
├── build_patient_Global_label_folders.py                # Organise global-label anomaly folders
├── build_patient_Local_label_folders.py                 # Organise per-slice local-label anomaly folders
├── Render_patient_slices_from_csv.py                    # Render/export FastMRI slices from CSV or label folders
└── fastMRI_brain_config.yaml                            # Centralised configuration (this repo)
```

Training data (not included):
```
FastMRI_Sample_Work/
├── Training_samples_FastMRI_IXI/     # Healthy .npz slices for training (from IXI_dataset_overview.py)
├── Validation_samples_FastMRI/       # Healthy .npz slices for validation
└── FastMRI_IXI_Augmented_lightningCheckpoints/  # Checkpoint output directory
```

---

## Architecture

### Stage 1 — RVQ-VAE (`FastMRI_model_stage1.py`)

| Component | Detail |
|-----------|--------|
| Input | Grayscale T1 brain MRI slice, 256×256 |
| `PatchEmbedding` | Conv2d (kernel=stride=patch_size), maps (B,1,256,256) → (B, seq_len, embed_dim) |
| `ViTEncoder` | TransformerEncoder, depth=8, heads=8, GELU, dropout=0.1; learned positional embedding (randn×0.02) |
| `MultiScaleEncoder` | Feature pyramid with Conv2d projections at stride 1/2/4; fused via cross-attention (8 heads) |
| `ResidualVQ` | 2-level residual quantization; codebook_size=256 per level; kmeans_init, EMA decay=0.85, orthogonal_reg_weight=0.1, threshold_ema_dead_code=0.1 |
| `PixelShuffleDecoder` | stem (Conv2d → SiLU) → 3 residual blocks (3 conv + GroupNorm8 + SiLU) → upsample blocks (Conv→PixelShuffle×2→SiLU) → 1-ch head; num_upsample=log2(patch_size) |
| Output clamp | `torch.clamp(recon, -3.0, 3.0)` |

**Key difference vs. Pelvic version:**
- `embed_dim=256` (same) but `codebook_size=256` (vs 192 for pelvic)
- **Augmentation is much richer**: RandScaleIntensity(0.1, p=0.33) + RandAdjustContrast(γ∈[0.5,1.5], p=0.33) + **RandGaussianNoise(p=0.50, std=0.30)** + RandAffine(p=0.33, ±15°, ±15px, zoom 0.8–1.2) + RandFlip(horizontal, p=0.5)
- Translation is **bi-directional** (horizontal + vertical), not horizontal-only
- An augmentation preview is saved to `FastMRI_RQC_ValExamples/augmentations_preview.png` on the first training batch (sanity check)
- Decoder `base_channels=embed_dim` (self-sizing), so the 3rd residual block adds an extra conv pass for richer texture reconstruction

**Loss:** L1 reconstruction + BiomedCLIP perceptual loss (cosine feature similarity, frozen vision tower, `perceptual_weight=0.5`) + VQ commitment loss (weight=0.25).

**Optimizer:** AdamW(β=(0.9, 0.95), wd=1e-4) + CosineAnnealingLR(T_max=max_epochs).

**Validation visualisation:** 4 random samples per epoch saved to `FastMRI_RQC_ValExamples/`; each 2×2 panel shows Input / Reconstruction / Q1 codebook indices / Q2 codebook indices with PSNR.

---

### Stage 2 — Factorized MaskGIT (`FastMRI_model_stage2.py`)

| Component | Detail |
|-----------|--------|
| Positional embedding | **2D RoPE** (row + column only, no slice axis); head_dim split into row_dim + col_dim |
| `RotaryEmbedding2D` | max_positions=seq_hw+1, base=25000; precomputed cos/sin buffers |
| `RMSNorm` | eps=1e-6, replaces LayerNorm |
| `SwiGLU` | w1/w2/w3 linear layers; `silu(w1(x)) * w2(x)` → w3; dropout=0.0 |
| `TransformerBlockSDPA` | pre-norm → QKV → 2D RoPE on Q,K → `F.scaled_dot_product_attention` → out_proj → pre-norm → SwiGLU |
| Stack depth | 8 blocks, 8 heads |
| Token embeddings | `l1_embed` (codebook_size+1, embed_dim), `l2_embed` (codebook_size+1, embed_dim), `task_embed` (2, embed_dim) |
| seq_len | Derived automatically: (image_size // patch_size)² = 1024 (patch=8) |

**Key difference vs. Pelvic version:** The Brain MRI stage 2 uses **2D RoPE** (row, col) rather than 3D RoPE (row, col, slice). The slice position argument is accepted but silently ignored, ensuring full backward compatibility with inference code that passes `slice_pos`.

**Masking strategy:**
- L1 tokens: 70% of time ratio ∈ [0.50, 0.75]; 30% of time ratio ∈ [0.20, 0.50]
- L2 tokens: ratio ∈ [0.15, 0.55] from β(4, 4) distribution
- 50% of training batches use block masking (random rectangles, union of overlapping blocks) instead of random masking
- All masks guarantee ≥1 masked token per sample (vectorised enforcement)
- Validation uses random masking (fixed ratio=0.20) + center mask (inner 67%×67%)

**Training loss:** CE(L1 masked) + 0.25 × CE(L2 masked), label_smoothing=0.05.

**Anomaly maps at inference** (four modes):

| Method | Description |
|--------|-------------|
| `compute_anomaly_map` | All L1/L2 tokens masked; Z-score normalized NLL per component |
| `compute_anomaly_map_sliding` | Sliding window (4×4, stride 2) + 8 MC passes per window; averaged NLL across overlaps |
| `compute_anomaly_map_contextual` | Random 15% of tokens masked; only masked positions scored |
| `compute_anomaly_map_iterative` | Iterative predict refinement (6 steps, initial_mask_ratio=0.70) → final NLL |

**Combined score:** `zscore(nll_l1) + l2_loss_weight × zscore(nll_l2) + q_error_weight × zscore(q_error)`, where z-scoring is per-image over spatial dimensions.

**Token frequency tracking:** Codebook utilisation, distribution entropy, and "lift" (acc − majority-class baseline) logged every 1000 batches. `print_token_frequency_summary()` can be called post-training to verify the model learned beyond mode-guessing.

**Optimizer:** AdamW(β=(0.9, 0.98), wd=0.01) with separate param groups (no decay on bias/norm/embed); LambdaLR with 2000-step linear warmup → cosine decay.

---

## Environment

Install with pip (exact versions):

```bash
pip install torch==2.8.0+cu128 pytorch-lightning==2.5.5 monai==1.5.1 \
    vector-quantize-pytorch==1.27.15 transformers==4.57.2 \
    open_clip_torch==3.2.0 lpips==0.1.4 nibabel==5.3.2 imageio scipy tqdm matplotlib
```

BiomedCLIP is loaded via `transformers` (`CLIPVisionModel`) with an automatic fallback to `open_clip`. The perceptual loss model is frozen during all training.

---

## Data Preparation

### Training data: IXI dataset (healthy T1 brain volumes)

Use `IXI_dataset_overview.py` to convert NIfTI volumes to `.npz` slices:

```bash
python IXI_dataset_overview.py \
    --input-dir /path/to/IXI-T1/ \
    --output-npy-dir /path/to/Training_samples_FastMRI_IXI \
    --training-ready \
    --training-slice-start 128 \
    --training-slice-end 188 \
    --z-clip "-3,3" \
    --intensity-scale none \
    --pattern "*.nii.gz"
```

**Preprocessing pipeline per slice:**
1. Load NIfTI, re-orient to closest-canonical axes (`nib.as_closest_canonical`)
2. Pad/crop in-plane to 256×256 (`center_crop_or_pad`)
3. Z-score normalise per volume: `(vol − μ) / σ`, clipped to [−3, 3]
4. Rotate 90° CCW (`np.rot90(arr, k=1)`)
5. Center crop or pad to 256×256
6. Save as `.npz` with key `arr` (float32); naming pattern: `{file_id}_slice_{idx:03d}.npz`

Slices 128–188 correspond to informative axial brain slices (avoiding skull cap / base-of-brain).

**Intensity scale options:** `none` (keep raw z-score), `minus1_1` (rescale to [−1,1]), `zero1` (rescale to [0,1]).

### FastMRI validation / test data

FastMRI Brain (T1) slices are organized by patient. For each volume:
- Apply the same z-score normalisation and 256×256 crop
- Save as `.npz` slices with the same naming convention

### Anomaly annotation CSVs

For evaluation, annotations are stored in CSVs with columns: `file`, `slice`, `x`, `y`, `width`, `height`, `label`, `study_level`, `base_size`. Three preprocessing modes are supported at inference time: `legacy`, `render_fastmri`, and `mask_pipeline`.

### Collecting normal slices for calibration

```bash
python collect_normal_slices.py \
    --annotation-csv path/to/brain.csv \
    --patient-list Annotated_FastMRI_Brains.csv \
    --series-type AXT1 \
    --slice-start 0 \
    --slice-end 5 \
    --png-root /path/to/Normal_Brains_pngs \
    --output-csv normal_slices_0_5.csv
```

A slice is classified as normal if: it has `study_level == "yes"`, its label contains the `--normal-label-keyword`, or it has no annotation at all.

### Rendering patient slices from CSV / label folders

`Render_patient_slices_from_csv.py` is the bridge between the FastMRI `.h5` source volumes and the 2D slice files/figures used for inspection, anomaly folder curation, and downstream inference inputs.

**What the script does:**
- Recursively scans `--data-root` for FastMRI `.h5` files and keeps only the requested series type (default `AXT1`) using HDF5 metadata.
- Accepts either:
  - a CSV with `file[,slice,reason]`, or
  - a `--label-root` folder created by `build_patient_Global_label_folders.py` / `build_patient_Local_label_folders.py`, where each label folder contains patient subfolders and optional `slices.csv` files.
- Loads the `reconstruction_rss` volume, pads/crops every slice in-plane to **320×320**, performs **per-volume z-score normalisation**, clips intensities with `--z-clip` (default `-3,3`), optionally rescales to `[-1,1]` or `[0,1]`, then exports the requested slices.
- Before saving each slice, applies `np.flipud(...)`, center-crops/pads to **320×320**, and resizes to **256×256**.
- Saves arrays as `.npz` by default (key: `arr`) and/or PNG previews. PNGs can overlay detailed annotation boxes and label text from `Annotated_fastMRI_Brains_Detailed.csv`.
- Supports `--include-label` and `--best-box-only` to keep only slices containing a requested local label and optionally choose the slice with the largest annotated bounding box per patient.

**Typical outputs:**
- Curated per-patient anomaly folders such as `label/patient_xxx/patient_xxx_slice_003.npz`
- Matching PNG renderings for qualitative review
- An optional updated CSV with `png_path` / `npy_path` columns

**Example: render the best slice per patient for one local anomaly label**
```bash
python Render_patient_slices_from_csv.py \
    --label-root /path/to/FastMRI_Local_Anomalies_ByLabel \
    --include-label "Intraventricular substance" \
    --best-box-only \
    --data-root /path/to/fastMRI_h5_root \
    --series-type AXT1 \
    --output-dir /path/to/rendered_pngs \
    --output-npy-dir /path/to/rendered_npz \
    --annotation-csv /path/to/Annotated_fastMRI_Brains_Detailed.csv
```

This rendering/export preprocessing is important because it differs slightly from `IXI_dataset_overview.py`: IXI training slices are written directly from NIfTI volumes, whereas FastMRI slices for anomaly review are generated from `.h5` reconstructions, normalised per volume, flipped vertically for display/orientation consistency, and then resized to the final 256×256 saved representation.


### Building anomaly label folders

**Global labels** (study-level, e.g. motion artifact):
```bash
python build_patient_Global_label_folders.py \
    --anomalies-dir /path/to/FastMRI_Anomalies_Collection \
    --detailed-csv Annotated_fastMRI_Brains_Detailed.csv \
    --output-dir /path/to/FastMRI_Global_Anomalies_ByLabel \
    --use-detailed
```
Default global labels: `Motion artifact`, `Possible artifact`, `Colpocephaly`, `Extra-axial collection`, `Global label: Small vessel chronic white matter ischemic change`.

**Local labels** (per-slice pathologies, e.g. mass, edema):
```bash
python build_patient_Local_label_folders.py \
    --anomalies-dir /path/to/FastMRI_Anomalies_Collection \
    --detailed-csv Annotated_fastMRI_Brains_Detailed.csv \
    --output-dir /path/to/FastMRI_Local_Anomalies_ByLabel
```
Default local labels: `Edema`, `Enlarged ventricles`, `Craniotomy`, `Mass`, `Nonspecific lesion`, `Resection cavity`, `Intraventricular substance`, `Paranasal sinus opacification`, `Posttreatment change`, `Nonspecific white matter lesion`, `Encephalomalacia`, `Dural thickening`, `Absent septum pellucidum`, `Lacunar infarct`, `Likely cysts`.

### Inference data directory structure and `case_folder`

The inference dataloader searches `--data-dir` recursively for `.npz` (or `.npy`) files. The `case_folder` field in each result JSON entry is set to the **immediate parent directory name** of each slice file. This allows grouping by patient or category when the data directory is structured as:

```
data_dir/
├── patient_abc/
│   ├── patient_abc_slice_003.npz
│   └── patient_abc_slice_004.npz
└── patient_xyz/
    └── patient_xyz_slice_005.npz
```

The `category` field in the JSON is set to the value passed via `--category` (or `"FastMRI"` by default). Use `--category "Mass"` to tag all files in a run with a specific anomaly label for stratified ROC analysis.

---

## Training

### Stage 1

```bash
python FastMRI_train.py --stage1 \
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

| Hyperparameter | Value |
|----------------|-------|
| `embed_dim` | 256 |
| `codebook_size` | 256 (both levels) |
| `commitment_cost` | 0.25 |
| `perceptual_weight` | 0.5 |
| `lr` | 2e-4 |
| `batch_size` | 192 (Stage 1) |
| `gradient_clip_val` | 1.0 |
| `precision` | 32 (float32) |
| GPU device | [1] |

Checkpoints saved to `FastMRI_IXI_Augmented_lightningCheckpoints/` as `FastMRI_stage1-{epoch:03d}-{val/loss:.4f}.ckpt`. Top-3 by `val/loss` are kept.

**Fine-tuning from an existing Stage 1 checkpoint** (e.g. LUND-PROBE → FastMRI transfer):
```bash
python FastMRI_train.py --stage1 \
    --pretrained-stage1-ckpt /path/to/previous_stage1.ckpt \
    [... other args ...]
```
Loaded with `strict=False`; augmentations are disabled during fine-tuning (`use_augmentations=False`).

### Stage 2

```bash
python FastMRI_train.py --stage2 \
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

| Hyperparameter | Value |
|----------------|-------|
| `embed_dim` | 256 |
| `codebook_size_level1/2` | 256 |
| `depth` | 8 |
| `num_heads` | 8 |
| `warmup_steps` | 2000 |
| `weight_decay` | 0.01 |
| `l2_loss_weight` | 0.25 |
| `q_error_weight` | 0.10 |
| `label_smoothing` | 0.05 |
| `batch_size` | 158 (Stage 2) |
| `mask_ratio` (val) | 0.20 |
| `mask_ratio_min/max` (train) | 0.15 / 0.75 |

Stage 1 is loaded from `--stage1-ckpt`, frozen, and set to `eval()`. Stage 2 weights can optionally be warm-started from a previous run via `--pretrained-stage2-ckpt`.

### WandB logging

```bash
python FastMRI_train.py --stage2 [args] \
    --wandb-project RVQ-MaskGIT-FastMRI-IXI \
    --wandb-run-name "Stage2-Augmented-FastMRI-IXI"
# To disable WandB:
    --wandb-off
```

**Logged metrics (Stage 2):** `train/loss`, `train/loss_l1`, `train/loss_l2`, `train/acc_l1`, `train/acc_l2`, `train/lift_l1`, `train/lift_l2`, `train/baseline_l1`, `train/baseline_l2`, `train/l1_codebook_utilization`, `train/l2_codebook_utilization`, `train/l1_entropy`, `val/loss`, `val/loss_center`, `val/acc_center`, `val/lift_center`.

---

## Inference Pipeline: Recursive-AutoMask V4

The main inference script is `Inference_FastMRI_SOTA_5p_Rec_Heal_5p0_automatic_final.py`.

### Step 1 — Model loading

```python
stage1, stage2 = load_models(stage1_ckpt, stage2_ckpt, device)
```

`perceptual_loss.*` keys are stripped from the Stage 1 state dict before loading (`strict=False`) so inference does not require BiomedCLIP.

### Step 2 — Calibration (healthy volunteers)

```bash
python Inference_FastMRI_SOTA_5p_Rec_Heal_5p0_automatic_final.py \
    --calibration-mode \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --data-dir /path/to/healthy_slices \
    --output-dir /path/to/calib_output \
    --calibration-map /path/to/zscore_calibration.npz \
    --smoothing-kernel 15 \
    --heal-patterns "0,1"
```

Per healthy slice:
1. Ensemble heal (checkerboard patterns 0 and 1, optional TTA flip)
2. LPIPS(original, healed) → spatial heatmap
3. Average-pool smooth with kernel=15
4. Accumulate across healthy population → compute μ (mean) and σ (std) per pixel

**Critical constraint:** `--smoothing-kernel` must be identical between calibration and inference.

### Step 3 — Z-score inference

```bash
python Inference_FastMRI_SOTA_5p_Rec_Heal_5p0_automatic_final.py \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --data-dir /path/to/test_slices \
    --output-dir /path/to/results \
    --calibration-map /path/to/zscore_calibration.npz \
    --z-threshold 2.0 \
    --smoothing-kernel 15 \
    --num-iterations 3 \
    --heal-patterns "0,1" \
    --heatmap-aggregation mean \
    --annotation-csv /path/to/brain.csv
```

### Pipeline steps inside `recursive_automask_v4_zscore`

1. **Sharpness scoring** — Laplacian variance (`compute_sharpness_score`) flags motion-blurred slices (blur_threshold=0.002).
2. **Token surprisal** — 50 MC masking passes (mask_ratio=0.15); accumulates NLL of the true L1 token only on masked positions; clamps values below 5.0 to zero. Returns a (B,1,Ht,Wt) token-resolution map.
3. **Ensemble healing** — For each checkerboard pattern in `heal_patterns` [0,1]: MaskGIT heals masked tokens (12 steps, temperature=0.8). L1 tokens healed first with fully-masked L2 context; L2 tokens healed conditioned on the healed L1. Optionally repeated with horizontal-flip TTA.
4. **LPIPS heatmap** — `PerceptualLoss` (VGG backbone, spatial=True) computes pixel-wise perceptual distance between original and healed reconstruction.
5. **Aggregation** — `aggregate_heatmaps` over ensemble branches (mean/max/logsumexp/geomean).
6. **Iteration 0 masking** — If calibration loaded: Z = (LPIPS − μ) / (σ + ε); threshold Z > z_threshold. Otherwise: percentile threshold at 95th percentile → morphological dilation (kernel=3) → remove small regions (<5 pixels).
7. **Targeted inpainting** (iterations 1–2) — `build_token_mask` converts the pixel anomaly mask to a token mask (mode: max/avg/topk); `targeted_inpaint` regenerates only flagged tokens (12 steps, temperature=0.9); non-flagged tokens locked exactly. Between iterations, the mask is dilated (kernel=5).
8. **Binary + token surprisal fusion** — `build_final_lpips_binary_token_eval_mask` fuses the LPIPS map (above 60th percentile) with Gaussian-smoothed token surprisal (55% LPIPS + 45% max of LPIPS/surprisal).

### Output fields per slice

| Field | Description |
|-------|-------------|
| `Binary_Sum_Heatmap` | Pixel count of the fused binary detection mask |
| `clamped_pixel_sum` | Sum of LPIPS values above clamp threshold |
| `token_surprisal_hot_px` | Count of high-surprisal tokens upsampled to pixel space |
| `sharpness_score` | Laplacian variance (motion artifact proxy) |
| `lpips_input_recon_sum_mask` | Total LPIPS inside the anomaly mask region |
| `anomaly_pixel_count` | Pixel count of the anomaly mask |
| `has_ground_truth_bbox` | Whether an annotation box is present |
| `num_true_positive_bboxes` | Bounding-box TP count (ratio ≥ 10% overlap) |
| `inside_bbox_detection_ratio` | Mean fraction of GT box pixels that are flagged |
| `precision`, `f1_score` | Per-slice localisation quality metrics |

Results are saved to `results_v4_zscore.json` (one entry per slice) plus per-slice PNG figures.

### Annotation box evaluation

The inference script supports bounding-box evaluation directly:
- Loads annotation boxes from `--annotation-csv` (columns: file, slice, x, y, width, height, label, study_level, base_size)
- Computes `compute_bbox_detection_metrics` per slice: TP if predicted mask covers ≥10% of GT box area
- FP ratio = predicted pixels outside healthy region / predicted pixels inside GT box
- Precision = TP / (TP + FP_ratio); F1 = 2P / (P+1) for TP=1

Three preprocessing modes handle coordinate system differences:
- `legacy`: direct scale from base_size to image size
- `render_fastmri`: y-axis flipped (FastMRI renders top-down)
- `mask_pipeline`: rasterise box as binary mask then resize with nearest-neighbour

---

## Evaluation — ROC Curves and Detection Metrics (`fastMRI_ROC_Curve_Calculations.py`)

```bash
python fastMRI_ROC_Curve_Calculations.py \
    --input /path/to/results_v4_zscore.json \
    --output-dir /path/to/roc_figures \
    [--category "Mass"] \
    [--case-folder "patient_abc"]
```

**Patient-level aggregation:** Slice-level `clamped_pixel_sum` values are summed per patient (derived from filename stem before `_slice_`). Patient is flagged anomalous if the total exceeds a threshold.

**FP pixel ratio:** Computed over slices without any GT bounding box: `highlighted_pixels / (num_no_bbox_slices × 256 × 256) × 100`.

**Localisation metrics from paper:**
- `compute_paper_precision_f1(tp_detected, inside_pixels, outside_pixels)`: reproduces the exact FP ratio, precision, and F1 formulation used in the paper

**Category filtering:** `--category` performs substring matching on the `category` field in the JSON. `--case-folder` filters by patient folder name.

---

## Key Differences vs. the Pelvic MRI Version

| Aspect | Brain MRI (FastMRI/IXI) | Pelvic MRI (LUND-PROBE) |
|--------|------------------------|------------------------|
| Dataset | FastMRI T1 + IXI T1 | LUND-PROBE T2 pelvis |
| Codebook size | **256** per level | 192 per level |
| Augmentation | Rich: noise(p=0.5, std=0.30), contrast, zoom | Minimal: intensity+affine only |
| RoPE | **2D** (row, col) | 3D (row, col, slice) |
| Perceptual weight | 0.5 | 0.9 |
| LR | 2e-4 | 1e-4 |
| Slice filtering | None (all slices from data dirs) | Training slices 30–60 only |
| Evaluation | Bounding-box TP/FP/F1 per slice | Patient-level ROC by category |
| File format | `.npz` (key `arr`) | `.npy` |
| Calibration data | FastMRI "Normal" annotated slices | Healthy LUND-PROBE volunteers |

---

## Exact Replication Checklist

- [ ] IXI T1 volumes pre-processed with `IXI_dataset_overview.py`, slices 128–188, z-clip [−3,3], 256×256 crop, 90° CCW rotation, saved as `.npz`
- [ ] Stage 1 trained 100 epochs, batch=192, lr=2e-4, embed_dim=256, codebook_size=256, perceptual_weight=0.5, full augmentation suite enabled
- [ ] Stage 2 trained 100 epochs, batch=158, lr=2e-4, embed_dim=256, codebook_size=256 both levels, 2000-step LR warmup, Stage 1 frozen
- [ ] Calibration run on "Normal" FastMRI slices with smoothing_kernel=15, heal_patterns=[0,1]
- [ ] Inference uses identical smoothing_kernel=15, z_threshold=2.0, num_iterations=3, heal_steps=12, heal_temperature=0.8, inpaint_steps=12, inpaint_temperature=0.9, token_surprisal_samples=50, mask_ratio=0.15
- [ ] Annotation boxes loaded from `brain.csv` with the appropriate `--annotation-preprocess-mode`
- [ ] Patient-level scores aggregated from `clamped_pixel_sum` over all evaluated slices
- [ ] ROC curves generated with `fastMRI_ROC_Curve_Calculations.py` per anomaly category

---

## Code Audit Addendum (README update)

This section was added after cross-checking the repository code against the original README so readers do not miss implementation details that affect interpretation and reproducibility.

### What is actively used at runtime vs. what is reference material

- `FastMRI_train.py`, `FastMRI_model_stage1.py`, `FastMRI_model_stage2.py`, `Inference_FastMRI_SOTA_5p_Rec_Heal_5p0_automatic_final.py`, `IXI_dataset_overview.py`, and `fastMRI_ROC_Curve_Calculations.py` are the main active workflow scripts.
- `fastMRI_brain_config.yaml` is a **reference/config summary**, but the main scripts inspected here do **not** automatically load it at runtime.
- Several important training and inference settings are defined directly inside Python scripts and CLI defaults rather than being read from YAML.

### Important implementation clarifications

#### 1. The primary FastMRI anomaly heatmap is reconstruction-referenced

This is the most important clarification for readers.

In the currently inspected brain pipeline:

- calibration uses **LPIPS(reconstruction, healed)**
- inference iteration 0 uses **LPIPS(reconstruction, healed)**
- refinement iterations use **LPIPS(reconstruction, inpainted)**

So the main anomaly heatmap is referenced to the **Stage 1 reconstruction**, not directly to the original input image.

The script still computes `lpips_input_recon`, but that is mainly used for auxiliary analysis/visualization outputs rather than as the primary iterative heatmap signal.

This differs conceptually from the pelvic README framing and should be kept in mind when interpreting results.

#### 2. `fastMRI_brain_config.yaml` is not the active runtime controller

Although the YAML file documents the intended settings well, the currently inspected scripts define the active behavior directly.

Examples:

- `FastMRI_train.py` defines its own CLI defaults for paths, batch size, LR, checkpoint loading, WandB, etc.
- `Inference_FastMRI_SOTA_5p_Rec_Heal_5p0_automatic_final.py` defines its own inference defaults directly
- the YAML is therefore best understood as a structured record/reference, not the sole runtime source of truth

#### 3. Training depends on a shared/external dataset module

`FastMRI_train.py` imports:

```python
from dataset import SliceDataModule
```

but there is no `dataset.py` file inside `Final_Code_Phiro_Brain_MRI/`.

So readers should know that the brain training workflow depends on a dataset module that lives outside this folder, likely shared across the broader project environment.

#### 4. Inference dataloader is more flexible than the README originally emphasizes

`create_inference_dataloader(...)` currently:

- searches recursively
- supports both `.npz` and `.npy`
- filters by filename substring through:
  - `category_filter`
  - `case_filter`
  - `patient_filter`
- sets `case_folder` from the immediate parent directory through metadata construction

This makes the inference pipeline suitable for both native FastMRI-style `.npz` inputs and external `.npy` slice datasets.

#### 5. Calibration mode writes an input audit log

During calibration, the script writes:

- `calibration_input_files.txt`

This file records the exact selected calibration files and active filters. It is a very useful reproducibility artifact and should be retained with experiments.

#### 6. Script defaults may differ from the recommended/paper-like settings shown in examples

The README examples are useful, but readers should distinguish them from the script defaults.

Examples of current defaults in `Inference_FastMRI_SOTA_5p_Rec_Heal_5p0_automatic_final.py` include:

- `--z-threshold "(-2.5 , 6.0)"` for two-sided thresholding
- `--smoothing-kernel 7`
- `--num-iterations 1`
- `--inter-iteration-dilation 1`
- `--heal-steps 6`
- `--heal-temperature 0.9`
- `--heal-patterns "4"`
- `--token-surprisal-samples 100`
- `--token-surprisal-mask-ratio 0.820`

These are not the same as the simpler example configuration shown earlier in the README.

#### 7. Annotation/evaluation behavior is richer than a simple bbox overlay

The inference script contains several evaluation-focused controls that are worth surfacing explicitly:

- three annotation preprocessing modes:
  - `legacy`
  - `render_fastmri`
  - `mask_pipeline`
- optional annotation flips:
  - `--annotation-flip-vertical`
  - `--annotation-flip-horizontal`
- optional batch inference over all anomaly label folders with `--run-all-anomaly-folders`
- LPIPS backflow / binary-token fusion controls
- edge-aware erosion controls for binary masks
- optional heatmap-ideas figure generation

These are important for reproducing figure-generation and evaluation behavior.

#### 8. Token surprisal exists in both token-space and image-space forms in the workflow

The token surprisal map is computed from token-level Monte Carlo masking, but it is resized to image resolution before several downstream operations and visualizations.

So readers should interpret it as:

- originating in token space
- but often used in image space after interpolation

### Practical reproducibility notes for readers

- Many defaults point to absolute local paths and should usually be overridden.
- If exact reproducibility matters, keep both:
  - the CLI command used
  - the generated `calibration_input_files.txt`
- Do not assume the YAML file alone reproduces the run; the active CLI/Python defaults matter.
- When interpreting FastMRI anomaly maps, remember that the main LPIPS branch is **reconstruction-vs-healed**, not input-vs-healed.


