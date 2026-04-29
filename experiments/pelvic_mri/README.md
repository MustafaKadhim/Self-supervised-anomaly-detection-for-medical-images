# Two-Stage Anomaly Detection Framework for Pelvic MRI 

## Overview

This repository implements a two-stage, unsupervised anomaly detection framework for pelvic T2-weighted MRI.
The framework is trained exclusively on healthy subjects and detects anomalies at inference time by measuring how surprising a given image is relative to the learned normal distribution — no anomaly labels are required during training.

### Core Idea

**Stage 1 — RVQ-VAE (`Model_Stage_1.py`):** A ViT-based encoder compresses each 2D slice into a discrete token grid via Residual Vector Quantization (RVQ). A PixelShuffle decoder reconstructs the image from quantized tokens.

**Stage 2 — Factorized MaskGIT (`Model_Stage_2.py`):** A bidirectional masked generative transformer learns the joint distribution over the Stage-1 token sequences. The model jointly predicts two codebook levels (structure L1 and texture L2) using factorized task conditioning and 3D Rotary Position Embeddings (RoPE) that encode row, column, and slice-axis positions.

**Inference —  (`Inference_Pelvis_Experiments.py`):** 
At inference, the framework:
1. "Heals" the input by regenerating tokens with deterministic checkerboard masks,
2. Computes an LPIPS perceptual difference map between input and healed image,
3. Converts the map to Z-scores using population statistics from a calibration set of healthy volunteers,
4. Optional: Iteratively refines the anomaly mask with targeted inpainting if the user wants,
5. Aggregats the final perceptual score with token surprisal (pseudo-PLL) score to achive the final anomaly score.

**Evaluation — ROC curves (`ROC_Curves_Calculations.py`):** Patient-level ROC-curves stratified by anomaly category (synthetic, clinical, etc.).

---

## Repository Structure

```
Final_Code_Phiro_Pelvic_MRI/
├── model_Stage_1.py                                    # Stage 1: ViT-RVQ-VAE
├── Model_Stage_2.py                                    # Stage 2: Factorized MaskGIT
├── Framework_train.py                                 # Training entry-point (both stages)
├── dataset.py                                         # NpySliceDataset + SliceDataModule
├── preslice_volumes.py                                # Pre-slicing NIfTI volumes -> .npy
├── Inference_Pelvis_Experiments.py.py                 # Full inference pipeline (Recursive-AutoMask V4)
├── ROC_Curves_Calculations.py                         # ROC / AUC evaluation utilities
├── External_dataset.py                                # External cohort dataset loader
├── Simulate_local_prostate_anomalies.py               # Simulate local anomalies influencing the prostate 
├── Simulated_anomalies_and_Clinical_dataset.py        # Convert DICOM to nifti and generate global synthetic anomalies. 
├── Train_Val_Test_Exact_DataSplits_LUND_PROBE.json    # Exact patient-level train/val/test splits for the pelvis experiments
├── Pelvis_experiments_requirements.txt                # Full needed Python packages 
└── config_yaml.yaml                                   # Centralised configuration (all hyperparameters)
```

---

## Architecture Details

### Stage 1 — `Stage1RVQVAE`

| Component | Details |
|-----------|---------|
| Input | Single-channel 2D MRI slice, 256×256 px (z-score normalised per volume) |
| Patch embedding | Non-overlapping 2D convolution; default `patch_size=8` → 32×32 token grid (1 024 tokens) |
| Encoder | `ViTEncoder`: standard `TransformerEncoder` with GELU activation, 8 layers, 8 heads, `embed_dim=256`; learned absolute position embeddings |
| Multi-scale encoder | `MultiScaleEncoder`: three Conv2d projections at stride 1, 2, 4 fused with a cross-attention layer |
| Quantizer | `ResidualVQ` (vector-quantize-pytorch): 2 codebook levels, 192 codes each, commitment weight=0.25, orthogonal regularisation weight=0.1, k-means init, EMA decay=0.85, dead-code threshold=0.1 |
| Decoder | `PixelShuffleDecoder`: Conv2d stem (embed_dim → 2×base_ch, SiLU) → 3-block residual stack (GroupNorm 8 groups + SiLU) → 3 PixelShuffle×2 upsample blocks → 1-channel head |
| Training loss | L1 reconstruction + BiomedCLIP perceptual loss (weight=0.9) + VQ commitment loss |
| Augmentation | MONAI `RandScaleIntensity` (factor ±0.1, p=0.33) + `RandAffine` (±5°, horizontal-only ±5 px, p=0.33, border padding) |
| Optimiser | AdamW (β=(0.9, 0.95), weight_decay=1e-4) + cosine annealing LR over max_epochs |
| Precision | float32 matmul set to "medium" via `torch.set_float32_matmul_precision` |

**BiomedCLIP perceptual loss** — The frozen vision tower of
`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (or an open_clip equivalent)
extracts pooled features from both reconstructed and target images. Loss = 1 − cosine_similarity.
Grayscale slices are replicated to 3 channels and resized to 224×224 with minmax normalisation before encoding.
A lightweight L1 fallback (`PerceptualLossStub`) is used if BiomedCLIP is unavailable.

**Quantisation error map** — The per-token squared L2 distance between pre- and post-quantisation token
embeddings is reshaped to the 2D token grid (32×32 for patch_size=8) and upsampled to pixel resolution.
This map forms the third component of the anomaly score during inference.

**Validation visualisations** — At the end of every validation epoch, up to 4 samples from slices 40–48
are visualised (input, reconstruction, Q1 and Q2 codebook index maps, PSNR) and saved to `RQC_ValExamples/`.
An augmentation preview is saved once at the first training batch.

### Stage 2 — `Factorized Transformer (Fact-biT)`

| Component | Details |
|-----------|---------|
| Input tokens | `(B, seq_len, 2)` RVQ indices from Stage 1; `seq_len = (image_size // patch_size)² = 1 024` |
| Token embeddings | Separate `nn.Embedding` tables for L1 (size: codebook_size+1) and L2 (size: codebook_size+1); the +1 index is the learnable mask token |
| Task conditioning | Learned `task_embed` (2 entries): index 0 = predict L1 structure tokens, index 1 = predict L2 texture tokens; broadcast-added to all sequence positions |
| Positional encoding | **3D Rotary Position Embeddings (RoPE)** — the head_dim is split into thirds: row, column, and slice (z-axis). Row and column use sinusoidal frequencies for up to 64 spatial positions; slice uses up to 92 positions. Base frequency = 25 000. No learned position embeddings. |
| Transformer | `TransformerSDPA`: stack of `TransformerBlockSDPA` blocks — pre-norm with `RMSNorm` (eps=1e-6), QKV via a single linear then `F.scaled_dot_product_attention` (Flash SDP / memory-efficient SDP enabled on PyTorch 2+), `SwiGLU` FFN (hidden_dim = 4×embed_dim, dropout=0), final `RMSNorm` |
| Prediction heads | Two independent linear layers: `head_l1` (embed_dim → codebook_size) and `head_l2` (embed_dim → codebook_size) |
| Masking strategy | Mixed: with probability 0.5 per batch, block masking (random rectangles until target coverage) is used; otherwise random masking. L1 masks prefer 50–75% ratio (70% of the time) or 20–50%; L2 masks: 15–55% ratio. β(4, 4) distribution is used as a prior, clamped to [mask_ratio_min, mask_ratio_max]. At least 1 token masked per sample. |
| Validation masking | Standard random masking (mask_ratio=0.20) + fixed centre-region mask (inner ~67%×67%) for domain-relevant evaluation |
| Training loss | Cross-entropy on masked L1 tokens + `l2_loss_weight × CE` on masked L2 tokens, both with label smoothing |
| Monitoring | Token frequency tracking; majority-class baseline accuracy and "lift" (model accuracy − baseline) logged every 1 000 batches; codebook utilisation and entropy reported |
| Slice filtering | Only batches containing slices in `[train_slice_min=30, train_slice_max=60]` are used; out-of-range slices are dropped from the batch before encoding |
| Optimiser | AdamW (β=(0.9, 0.98)); weight decay=0.01 for non-embedding/bias/norm params, 0.0 otherwise; linear warmup (2 000 steps) → cosine decay |

**Anomaly scoring at inference:**

Three signals are computed and combined with per-image z-score normalisation across spatial positions:

```
nll_l1    = -log p(true L1 token | all L1 masked, all L2 masked)
nll_l2    = -log p(true L2 token | L1 visible, all L2 masked)
q_error   = ||pre-quant token - quantized token||²

anomaly_score = zscore(nll_l1)
              + l2_loss_weight × zscore(nll_l2)
              + q_error_weight × zscore(q_error)
```

The combined map is upsampled to pixel resolution by `scale_factor = patch_size` (bilinear).

**Four inference scoring variants** are implemented in `model_stage2.py`:

| Method | Function | Description |
|--------|----------|-------------|
| Standard | `compute_anomaly_map` | All tokens masked simultaneously |
| Sliding window | `compute_anomaly_map_sliding` | Local window masking with Monte Carlo dropout, aggregated over all window positions |
| Contextual | `compute_anomaly_map_contextual` | Partial masking (15%) to preserve neighbourhood context |
| Iterative | `compute_anomaly_map_iterative` | MaskGIT iterative refinement, then NLL of true tokens |

---

## Installation

```bash
# Python 3.10+, CUDA 12.8 (or compatible)
pip install -r LUNDPROBE_requirements.txt
```

Key dependencies (pinned versions):

| Package | Version |
|---------|---------|
| torch | 2.8.0+cu128 |
| pytorch-lightning | 2.5.5 |
| monai | 1.5.1 |
| vector-quantize-pytorch | 1.27.15 |
| transformers | 4.57.2 |
| open_clip_torch | 3.2.0 |
| lpips | 0.1.4 |
| nibabel | 5.3.2 |
| scipy | 1.15.3 |
| scikit-image | 0.25.2 |
| wandb | 0.22.1 |
| numpy | 2.1.2 |

---

## Data Preparation

### 1. Pre-slice NIfTI volumes to .npy

Training operates on 2D axial slices stored as individual `.npy` files for fast I/O.

#### Training data (healthy volunteers)

```bash
python preslice_volumes.py
```

The script:
1. Loads each 3D NIfTI volume (`.nii` / `.nii.gz`) via nibabel.
2. Applies per-volume **z-score normalisation**: `(x − mean) / std` (std clipped to ≥ 1e-8).
3. Saves every axial slice as `{patient_id}_slice_{idx:03d}.npy` (float32) along the third axis.
4. Writes a `preslice_metadata.json` summary (total volumes, total slices, per-patient slice count).

#### Test / inference data (anomalous cohorts)

Use `External_dataset.py` for external NIfTI cohorts. This script applies **identical preprocessing** (z-score, 90° CCW rotation, resize 320→crop 256) and uses the following naming convention that encodes category metadata directly in the filename:

```
{category}_{case_folder}_{volume_name}_slice_{idx:03d}.npy
# Example: ClinicalVariations_T2_CUBE_FemaleBrachy_Cube1_slice_045.npy
```

The `category` and `case_folder` segments are parsed by `ROC_Curves_Calculations.py` to assign each slice to an anomaly category for stratified ROC analysis. The expected folder structure for the external cohorts is:

```
<cohort_root>/
├── ClinicalVariations/
│   ├── Spacer/          *.nii.gz
│   ├── MAVRIC_protes/   *.nii.gz
│   └── ...
└── SyntheticVariations/
    ├── Noise/           *.nii.gz
    ├── Motion/          *.nii.gz
    └── ...
```

Configure the source glob and output directory via `dictConfig["dataPath"]` (from a local `config` module) or by editing the script directly. Default output:

```
/home/mluser1/Musti_Anomaly_Detection/Data/PreSliced/
```

**Naming convention is critical:** The slice index embedded in the filename (e.g. `_slice_045`) is parsed at training time (Stage 2 slice filtering) and inference time (slice-position RoPE encoding, per-slice calibration lookup). Files that do not match `*_slice_*.npy` are silently ignored by `SliceDataModule`.

### 2. Dataset splits

`Train_Val_Test_Exact_DataSplits_LUND_PROBE.json` documents the exact patient-level train / validation / test split used in the paper experiments:

| Split | Patients |
|-------|----------|
| Train | 384 |
| Validation | (derived from train set, 10% random with seed 42) |
| Test | Separate cohorts in dedicated inference directories |

When no JSON split is used, `SliceDataModule` performs a random 90/10 train/val split seeded with 42 over all valid `.npy` files in `data_dir`.

### 3. Preprocessing pipeline (applied during training data loading)

Each `.npy` slice passes through the following pipeline in `dataset.py`:

| Step | Transform | Parameters |
|------|-----------|------------|
| 1 | 90° CCW rotation (`np.rot90`, in `__getitem__`) | `k=-1` (matches anatomical orientation) |
| 2 | `EnsureChannelFirstD` | Adds channel dim |
| 3 | `Resized` | `(320, 320)`, area interpolation |
| 4 | `CenterSpatialCropd` | `(256, 256)` |
| 5 | `ToTensorD` | → float32 tensor |
| 6 (train only) | `RandFlipD` | Horizontal flip, p=0.5 |
| 7 (train only) | `RandRotateD` | ±5° (0.0873 rad), keep_size=True, p=0.3 |

The Stage 1 model additionally applies MONAI augmentations **inside `training_step`** (after the DataLoader transforms), allowing the perceptual loss to receive the augmented image.

---

## Training

### Stage 1 — RVQ-VAE

```bash
python train.py --stage1 \
    --data-dir /path/to/PreSliced \
    --batch-size 128 \
    --num-workers 8 \
    --max-epochs 100 \
    --lr 1e-4 \
    --precision 32 \
    --log-dir logs \
    --wandb-project RVQ-MaskGIT \
    --wandb-run-name Stage1-RVQ-VAE
```

**Exact hyperparameters used in the paper (passed directly in `train.py`):**

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `embed_dim` | 256 | Token embedding dimension |
| `patch_size` | 8 | Patch size → 32×32 token grid |
| `encoder_depth` | 8 | Transformer encoder layers |
| `encoder_heads` | 8 | Attention heads |
| `codebook_size` | 192 | Codes per RVQ level |
| `num_quantizers` | 2 | Number of RVQ residual levels |
| `commitment_cost` | 0.25 | VQ commitment loss weight |
| `perceptual_weight` | 0.9 | BiomedCLIP perceptual loss weight |
| `lr` | 1e-4 | Initial learning rate |
| `max_epochs` | 100 | Training epochs |
| `batch_size` | 128 | Batch size |
| `gradient_clip_val` | 1.0 | Gradient norm clipping |
| `use_augmentations` | True | MONAI intra-step augmentation |

The Stage 1 checkpoint used for Stage 2 training in the paper:
```
lightningCheckpoints_Modified/Modified_stage1-epoch=094-val/loss=0.8587.ckpt
```

### Stage 2 — Factorized MaskGIT

Stage 2 requires a trained Stage 1 checkpoint. Stage 1 weights are **completely frozen** (gradients disabled).

```bash
python train.py --stage2 \
    --data-dir /path/to/PreSliced \
    --stage1-ckpt /path/to/stage1.ckpt \
    --batch-size 128 \
    --num-workers 8 \
    --max-epochs 100 \
    --lr 1e-4 \
    --precision 32 \
    --wandb-run-name Stage2-Factorized-MaskGIT
```

**Exact hyperparameters used in the paper (FactorizedMaskGIT defaults):**

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `embed_dim` | 256 | Transformer embedding dimension |
| `depth` | 8 | Transformer blocks |
| `num_heads` | 8 | Attention heads |
| `mask_ratio` | 0.20 | Fixed mask ratio used at validation |
| `mask_ratio_min` | 0.15 | β-distribution clip lower bound |
| `mask_ratio_max` | 0.75 | β-distribution clip upper bound |
| `beta_alpha` | 4.0 | β distribution α parameter |
| `beta_beta` | 4.0 | β distribution β parameter |
| `l2_loss_weight` | 0.25 | Weight of L2 token CE loss |
| `q_error_weight` | 0.1 | Weight of quantisation error in anomaly score |
| `label_smoothing` | 0.05 | Cross-entropy label smoothing (training only) |
| `warmup_steps` | 2 000 | Linear LR warmup steps |
| `weight_decay` | 0.01 | AdamW weight decay (params); 0.0 for biases/norms/embeds |
| `lr` | 1e-4 | Peak learning rate |
| `train_slice_min` | 30 | Minimum slice index included in training |
| `train_slice_max` | 60 | Maximum slice index included in training |
| `max_epochs` | 100 | Training epochs |
| `batch_size` | 128 | Batch size |
| `gradient_clip_val` | 1.0 | Gradient norm clipping |

**GPU:** Training uses `devices=[1]` (single GPU, index 1). Edit `make_trainer` in `train.py` if your setup differs.

### Logging and checkpointing

| Logger | Output | Notes |
|--------|--------|-------|
| CSVLogger | `logs/<stage>/` | Always active |
| WandbLogger | wandb run | Disable with `--wandb-off`; set project with `--wandb-project` |

Checkpoint pattern:
```
lightningCheckpoints_Modified/Modified_Checkerboard_<stage>-epoch=<E>-val/loss=<L>.ckpt
```
Top-3 checkpoints by `val/loss` (min) are retained.

---

## Inference

The inference pipeline (`Inference_LUNDPROBE_final.py`) implements **Recursive-AutoMask V4 with Z-Score Normalization**.

### Calibration Mode — required before inference

Run calibration on a held-out set of **healthy volunteers** to learn per-pixel μ and σ of the LPIPS reconstruction error under normal conditions:

```bash
python Inference_LUNDPROBE_final.py \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --data-dir /path/to/healthy_slices \
    --calibration-mode \
    --calibration-output /path/to/calibration.npz \
    --heal-steps 12 \
    --heal-temperature 0.8 \
    --smoothing-kernel 15 \
    --use-tta \
    --use-per-slice-stats
```

The calibration `.npz` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `mu` | (H, W) | Per-pixel mean LPIPS across all healthy slices (after spatial smoothing) |
| `sigma` | (H, W) | Per-pixel standard deviation |
| `n_samples` | scalar | Number of healthy slices used |
| `smoothing_kernel` | scalar | Kernel size applied during calibration (must match inference) |
| `mu_slice_<idx>` | (H, W) | Per-slice-index mean (when `use_per_slice_stats=True`, ≥3 samples required) |
| `sigma_slice_<idx>` | (H, W) | Per-slice-index std |

A visualisation (`calibration_visualization.png`) showing μ, σ, μ/σ ratio and false-positive-prone regions is also saved automatically.

**Key design decision:** Spatial smoothing (avg-pool, `smoothing_kernel=15`) is applied to both the calibration heatmaps and the inference heatmaps identically. This makes statistics represent anatomical regions rather than exact pixel locations, which is critical because patient-to-patient registration differences are expected.

### Inference Mode

```bash
python Inference_LUNDPROBE_final.py \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --data-dir /path/to/test_slices \
    --calibration-map /path/to/calibration.npz \
    --output-dir /path/to/results \
    --z-threshold 2.0 \
    --num-iterations 3 \
    --heal-steps 12 \
    --heal-temperature 0.8 \
    --inpaint-steps 12 \
    --inpaint-temperature 0.9 \
    --use-tta \
    --token-surprisal-samples 50
```

### Inference Pipeline — Step-by-Step

The core function is `recursive_automask_v4_zscore`. Below is a full description of each step:

**Step 1 — Sharpness computation**

Laplacian variance (`compute_sharpness_score`) and spatial Laplacian energy map (`compute_sharpness_map`) are computed for the input image. Slices with sharpness < `blur_threshold=0.002` are flagged as motion-blurred artifacts; the final mask for those slices is overridden to all-ones (maximum anomaly flag) as a safety measure.

**Step 2 — Token surprisal (pseudo-PLL)**

`compute_token_surprisal_map` approximates the pseudo-perplexity of L1 tokens via 50 independent random-masking passes (mask_ratio=0.15):
- For each pass: randomly mask 15% of L1 tokens, run Stage 2 transformer, compute NLL of the true token.
- Only masked positions accumulate NLL; unmasked positions remain zero.
- Scores are averaged across passes.
- Values > `token_surprisal_clamp=5.0` are retained; values below are zeroed (clamp filter for noise suppression).
- The map is upsampled to pixel resolution (bilinear) and counted as hot pixels (`token_surprisal_hot_px`).

This signal is independent of the healing branch — it measures how unexpected each token is in isolation.

**Step 3 — Ensemble healing**

`ensemble_heal` applies two deterministic checkerboard masks (patterns 0 and 1 in `_build_checker_mask`) and heals each independently:

- Patterns 0/1: alternating black/white at single-pixel granularity (pixel checkerboard)
- Patterns 2/3: 2×2 blocks (if used)
- Patterns 4/5: 4×4 blocks (if used)

For each mask pattern, `_heal_with_mask` iteratively unmasks tokens in 12 MaskGIT steps at temperature=0.8. In each step, the most confident masked positions are unmasked first (confidence-ordered). L1 and L2 codebook levels are healed sequentially (L1 first, then L2 conditioned on healed L1).

TTA (test-time augmentation): the input is also horizontally flipped, healed identically, and the healed image is flipped back. This provides a second healing trajectory that captures different spatial contexts.

**Step 4 — LPIPS heatmap (Iteration 0)**

VGG-LPIPS (spatial mode) measures pixel-wise perceptual distance between input and healed images.
Multiple healed versions from both the native and TTA branches are aggregated:
- `mean`: average LPIPS across all healed versions
- `max`: maximum LPIPS
- `logsumexp`: soft-max aggregation with temperature
- `geomean`: geometric mean of the two ensemble branch heatmaps

**Step 5 — Z-score thresholding (Iteration 0 only)**

If a calibration map is provided:
```
Z[h, w] = (LPIPS_smooth[h, w] − μ[h, w]) / (σ[h, w] + ε)
```
where spatial smoothing (avg-pool, kernel=15) is applied first. Per-slice-index statistics are used when the slice index is present in the calibration and matches a stored entry; otherwise the global μ/σ is used.

Binary mask: `Z > z_threshold` (default 2.0). Connected-component filtering removes regions < `min_region_size=5` pixels. Morphological dilation with kernel=3 expands the mask.

**Step 6 — Targeted inpainting (all iterations)**

`targeted_inpaint` converts the binary mask to a token-space mask (one of: max-pool, avg-pool above 0.5, top-k ratio) and regenerates only the flagged tokens via 12 MaskGIT steps. Non-flagged tokens are **locked** (their indices are preserved exactly). After inpainting, L1 and L2 changes are recorded (`l1_change`, `l2_change` fraction of tokens changed).

**Step 7 — Refinement (Iterations 1–2)**

LPIPS(input, inpainted) replaces the healed heatmap. Percentile thresholding (95th percentile over the image) is used instead of Z-scoring. An inter-iteration dilation (kernel=5) expands the mask slightly before the next inpainting pass.

**Step 8 — Scalar scores per slice**

Each slice in the output JSON receives:

| Field | Description |
|-------|-------------|
| `Binary_Sum_Heatmap` | Number of pixels in the final masked heatmap > 0.10 |
| `clamped_pixel_sum` | Sum of heatmap values above a clamp threshold, weighted by the anomaly mask |
| `token_surprisal_hot_px` | Number of hot pixels in the token surprisal map (NLL > clamp) |
| `sharpness_score` | Laplacian variance of the input slice |
| `anomaly_pixel_count` | Number of pixels flagged in the binary anomaly mask |
| `lpips_input_recon_sum_mask` | LPIPS(input, Stage1 reconstruction) summed over the mask |
| `mean_heal_change_l1/l2` | Fraction of L1/L2 tokens changed during healing |
| `mask_coverage` | Fraction of pixels in the final anomaly mask |

**Patient-level score (for ROC evaluation):**
```
patient_score = sum over slices 38-49 of (token_surprisal_hot_px + Binary_Sum_Heatmap)
```

### Model loading for inference

Stage 1 is loaded by filtering out the perceptual loss weights (not needed at inference):
```python
filtered_state = {k: v for k, v in state_dict.items()
                  if not k.startswith("perceptual_loss.")}
stage1 = Stage1RVQVAE(**hparams)
stage1.load_state_dict(filtered_state, strict=False)
```
Stage 2 is loaded with `FactorizedMaskGIT.load_from_checkpoint(..., stage1=stage1, strict=False)`.
Both models are set to `eval()` with all parameters frozen.

---

## ROC Curve Evaluation

`ROC_Curves_Calculations.py` reads per-slice JSON result files and computes patient-level ROC/AUC curves stratified by anomaly category.

### Anomaly categories in the LUND-PROBE dataset

| Category | Type | Description |
|----------|------|-------------|
| `RandomGhosting` | Synthetic | Ghosting artifact |
| `RandomNoise` | Synthetic | Additive noise |
| `RandomSpike` | Synthetic | k-space spike artifact |
| `RandomMotion` | Synthetic | Motion artifact |
| `WholeImageGaussian` | Synthetic | Whole-image Gaussian blur |
| `Stor_T2_till_sCT` | Clinical | sCT-related sequence variation |
| `ClinicalVariations` | Clinical | Real protocol deviations |
| `Spacer` | Clinical | Gel spacer implant |
| `Unknown` | Clinical | Unclassified anomaly |

**Groupings used in the paper:**
- *Synthetic global:* `RandomGhosting, RandomNoise, RandomSpike, RandomMotion, WholeImageGaussian`
- *Clinical:* `Unknown, Spacer, Stor_T2_till_sCT, ClinicalVariations`

### Default result JSON paths

The script reads from five result directories by default (edit `DEFAULT_ROC_INPUT_PATHS` to match your setup):

```python
DEFAULT_ROC_INPUT_PATHS = [
    ".../Inference_Results_LUND_PROBE_Christian_Clinical/results_v4_zscore.json",
    ".../Inference_Results_LUND_PROBE_ESTROTestdata_CervixBrachy/results_v4_zscore.json",
    ".../Inference_Results_LUND_PROBE_Global_Local_Clinical/results_v4_zscore.json",
    ".../Inference_Results_test_LUND_PROBE_extended_npy/results_v4_zscore.json",
    ".../Inference_Results_LUND_PROBE_SpacerResampled/results_v4_zscore.json",
]
```

### Patient-level aggregation strategies

| Function | Metric used | Description |
|----------|-------------|-------------|
| `aggregate_patient_clamp_from_results` | `clamped_pixel_sum` | Sums clamped LPIPS pixel intensities across all slices |
| `collect_patient_binary_sums` | `Binary_Sum_Heatmap` | Counts binary anomaly pixels per patient |
| `collect_patient_sharpness_totals` | `sharpness_score` | Aggregated Laplacian sharpness (negative control) |
| `aggregate_patient_status` | `combined_score > threshold` | Votes by number of flagged slices exceeding threshold |

### Running ROC evaluation

```bash
python ROC_Curves_Calculations.py \
    --input /path/to/results_v4_zscore.json \
    --output-dir /path/to/roc_figures \
    --categories all
```

---

## Configuration

All hyperparameters are centralised in `config_yaml.yaml`. The file is structured by task:
- `data`: paths, splits, preprocessing
- `stage1`: RVQ-VAE architecture and training
- `stage2`: MaskGIT architecture and training
- `training`: shared trainer settings (epochs, batch size, devices, logging)
- `calibration`: calibration-mode settings
- `inference`: Recursive-AutoMask V4 settings
- `evaluation`: ROC and scoring thresholds

---

## Output Files and Directories

| Path | Description |
|------|-------------|
| `RQC_ValExamples/` | Per-epoch validation reconstructions and codebook index maps |
| `RQC_ValExamples/augmentations_preview.png` | First-batch augmentation sanity check (4 samples × 3 augmented versions) |
| `lightningCheckpoints_Modified/` | PyTorch Lightning checkpoints (top-3 by `val/loss`) |
| `logs/<stage>/` | CSVLogger training curves (loss, accuracy, baseline, lift, codebook utilisation) |
| `<output-dir>/results_v4_zscore.json` | Per-slice inference results (all scalar metrics) |
| `<output-dir>/figures/` | Per-slice visualisation grids: input, reconstruction, healed, healed-TTA, heatmap, Z-score map, mask, inpainted, token surprisal |
| `<calibration>.npz` | Z-score calibration maps (μ, σ, per-slice stats, n_samples, smoothing_kernel) |
| `<calibration>_visualization.png` | Six-panel calibration visualisation (μ, σ, μ/σ ratio, histograms, FP-prone regions) |

---

## Exact Replication Checklist

To exactly replicate the published experiments:

1. **Environment:** Install all packages from `LUNDPROBE_requirements.txt` (CUDA 12.8, PyTorch 2.8.0+cu128).
2. **Data:** Pre-slice healthy pelvic MRI volumes with `preslice_volumes.py` (z-score per volume, all axial slices, naming: `{patient_id}_slice_{idx:03d}.npy`).
3. **Patient splits:** Use `Train_Val_Test_Exact_DataSplits_LUND_PROBE.json` to assign 384 patients to the training set.
4. **Stage 1:** Train with `embed_dim=256, codebook_size=192, num_quantizers=2, patch_size=8, perceptual_weight=0.9, commitment_cost=0.25, lr=1e-4, 100 epochs, batch=128, gradient_clip=1.0`.
5. **Stage 2:** Load Stage 1 from step 4; train with `embed_dim=256, depth=8, num_heads=8, l2_loss_weight=0.25, q_error_weight=0.1, label_smoothing=0.05, warmup_steps=2000, lr=1e-4, weight_decay=0.01, 100 epochs, batch=128, slice_min=30, slice_max=60`.
6. **Calibration:** Run on a held-out set of healthy volunteers with `smoothing_kernel=15, heal_steps=12, temperature=0.8, heal_patterns=[0,1], use_tta=True, use_per_slice_stats=True, heatmap_aggregation=mean`.
7. **Inference:** Run with calibration from step 6; use `z_threshold=2.0, num_iterations=3, heal_steps=12, heal_temperature=0.8, heal_patterns=[0,1], inpaint_steps=12, inpaint_temperature=0.9, token_surprisal_samples=50, token_surprisal_mask_ratio=0.15, token_surprisal_clamp=5.0, smoothing_kernel=15, use_tta=True`.
8. **Evaluation:** Aggregate slice scores over slices 38–49 per patient and compute ROC with `ROC_Curves_Calculations.py`.

---

## Practical Notes

- `train.py` sets `devices=[1]`. Update `make_trainer` if your GPU index differs.
- `preslice_volumes.py` imports `dictConfig` from a local `config` module. Supply that module or edit the `SOURCE_GLOB` variable directly.
- Many default paths in the inference and ROC scripts point to absolute local directories. Pass CLI arguments explicitly in a different environment.
- Stage 2 uses `torch.backends.cuda.enable_flash_sdp(True)` at import time; Flash Attention is used automatically when available.
- The 90° CCW rotation in `dataset.py` (`np.rot90(arr, k=-1)`) is applied at every data load. Ensure your NIfTI volumes are oriented so that the axial plane is the third axis (`vol[:, :, slice_idx]`).

---

## Citation

If you use this code, please cite our work. BibTeX entry to be added upon publication.

---

## Acknowledgements

- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) — ResidualVQ implementation.
- [MONAI](https://monai.io/) — Medical imaging transforms and data utilities.
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) — Domain-adapted perceptual loss during Stage 1 training.
- [lpips](https://github.com/richzhang/PerceptualSimilarity) — Spatial LPIPS used in the inference heatmap.
- [PyTorch Lightning](https://lightning.ai/) — Training framework.
