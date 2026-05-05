# Two-Stage Anomaly Detection Framework for Pelvic MRI (LUND-PROBE)

### Core Idea

**Stage 1 — RVQ-VAE (`Model_Stage_1.py`):** A ViT-based encoder compresses each 2D slice into a discrete token grid via Residual Vector Quantization (RVQ). A PixelShuffle decoder reconstructs the image from quantized tokens.

**Stage 2 — Factorized MaskGIT (`Model_Stage_2.py`):** A bidirectional masked generative transformer learns the joint distribution over the Stage-1 token sequences. The model jointly predicts two codebook levels (structure L1 and texture L2) using factorized task conditioning and 3D Rotary Position Embeddings (RoPE) that encode row, column, and slice-axis positions.

**Inference — Recursive-AutoMask V4 (`Inference_Pelvis_Experiments.py`):** At inference, the framework:
1. "Heals" the input by regenerating tokens with deterministic checkerboard masks (ensemble),
2. Computes an LPIPS perceptual difference map between input and healed image,
3. Converts the map to Z-scores using population statistics from a calibration set of healthy volunteers,
4. Iteratively refines the anomaly mask with targeted inpainting,
5. Augments the final score with token surprisal (pseudo-PLL) for complementary evidence.

**Evaluation — ROC curves (`ROC_Curves_Calculations.py`):** Patient-level ROC curves stratified by anomaly category (synthetic, clinical, spacer, etc.).

---

## Repository Structure

```
Final_Code_Phiro_Pelvic_MRI/
├── 🧠 Models
│   ├── Model_Stage_1.py                   # Stage 1: ViT-RVQ-VAE
│   └── Model_Stage_2.py                   # Stage 2: Factorized MaskGIT
├── 🔥 Training & Inference
│   ├── Train_frameworks.py                # Training entry-point (both stages)
│   └── Inference_Pelvis_Experiments.py    # Recursive-AutoMask V4 pipeline
├── 📊 Evaluation & Data
│   ├── ROC_Curves_Calculations.py         # ROC / AUC evaluation utilities
│   ├── dataset.py                         # NpySliceDataset + SliceDataModule
│   ├── preslice_volumes.py                # Pre-slicing NIfTI volumes → .npy
│   ├── External_dataset.py                # External cohort dataset loader
│   └── Train_Val_Test_Exact_DataSplits... # Exact patient-level splits (JSON)
├── ⚙️ Config
│   ├── config.yaml                        # Centralised configuration
│   └── Pelvis_experiments_requirements.txt# Pinned Python environment
```

---

## Architecture Details

### Stage 1 — RVQ-VAE (`Model_Stage_1.py`)

#### Architecture Flow

```
Input (B, 1, 256, 256)
    │
    ▼
┌─────────────────────────────────────┐
│  PatchEmbedding                     │
│  Conv2d(kernel=stride=patch_size) │
│  (B, 1, 256, 256) → (B, 1024, 256)│
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
│    (absolute)                       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  MultiScaleEncoder                │
│  • 3 Conv2d projections (stride   │
│    1, 2, 4) fused via cross-attn  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  ResidualVQ                         │
│  • Levels: 2                        │
│  • Codebook size: 192 per level     │
│  • kmeans_init, EMA decay: 0.85     │
│  • Orthogonal reg: 0.1              │
│  • Dead code threshold: 0.1         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PixelShuffleDecoder                │
│  • Stem: Conv2d → SiLU              │
│  • 3 ResBlocks (GN8 + SiLU)         │
│  • 3× PixelShuffle×2 upsample      │
│  • 1-channel head                   │
└──────────────┬──────────────────────┘
               │
               ▼
    Output (B, 1, 256, 256)
```


#### Component Details

| Component | Configuration |
|:----------|:-------------|
| **Input** | Single-channel 2D T2 pelvic MRI slice, `256×256` (z-score normalised per volume) |
| **PatchEmbedding** | Non-overlapping `Conv2d`; `patch_size=8` → `32×32` token grid (`1024` tokens) |
| **ViTEncoder** | `TransformerEncoder`, depth=`8`, heads=`8`, `GELU`, dropout=`0.1`; learned **absolute** position embeddings |
| **MultiScaleEncoder** | Three `Conv2d` projections at stride `1/2/4`, fused via cross-attention |
| **ResidualVQ** | 2-level residual quantization; `codebook_size=192` per level; `kmeans_init`, EMA decay=`0.85`, `orthogonal_reg_weight=0.1`, `threshold_ema_dead_code=0.1` |
| **PixelShuffleDecoder** | Stem → 3 residual blocks → 3× `PixelShuffle×2` upsample → 1-ch head |
| **Precision** | `float32` matmul set to `"medium"` via `torch.set_float32_matmul_precision` |

#### Training Configuration

| Parameter | Value |
|:----------|:------|
| **Reconstruction Loss** | `L1` |
| **Perceptual Loss** | BiomedCLIP Cosine Feature Similarity (frozen vision tower) |
| **Perceptual Weight** | `0.9` |
| **VQ Commitment Loss** | Weight = `0.25` |
| **Optimizer** | `AdamW(β=(0.9, 0.95), weight_decay=1e-4)` |
| **Scheduler** | Cosine annealing LR over `max_epochs` |
| **Gradient Clipping** | `1.0` |
| **Precision** | `32` (float32) |
| **Augmentation** | MONAI `RandScaleIntensity` (±0.1, p=0.33) + `RandAffine` (±5°, horizontal-only ±5px, p=0.33) |

> **BiomedCLIP Details:** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (or `open_clip` equivalent). Grayscale slices replicated to 3 channels, resized to `224×224` with minmax normalisation. Loss = `1 − cosine_similarity`. Fallback: `PerceptualLossStub` (L1) if BiomedCLIP unavailable.

#### Validation Visualization

At the end of every validation epoch, up to **4 samples** from slices `40–48` are visualised as `2×2` panels:

| Position | Content |
|:---------|:--------|
| **Top-Left** | Input |
| **Top-Right** | Reconstruction |
| **Bottom-Left** | Q1 Codebook Index Map |
| **Bottom-Right** | Q2 Codebook Index Map |

*PSNR reported per sample.* Saved to `RQC_ValExamples/`. Augmentation preview saved once at first training batch.


### Stage 2 — Fact-biT (`Model_Stage_2.py`)

#### Architecture Flow

```
L1 Tokens ──→ l1_embed (192+1) ──┐
                                  ├──→ Task Embed ──→ Transformer Stack ──→ L1/L2 Logits (predictions)
L2 Tokens ──→ l2_embed (192+1) ──┤          ↑
                                  │    3D RoPE (row, col, slice)
Task ID  ──→ task_embed (2) ──────┘    RMSNorm + SwiGLU + SDPA
```


#### Component Details

| Component | Configuration |
|:----------|:-------------|
| **Input Tokens** | `(B, seq_len, 2)` RVQ indices from Stage 1; `seq_len = 1024` |
| **Token Embeddings** | Separate `nn.Embedding` for L1 (`codebook_size+1`) and L2 (`codebook_size+1`); `+1` = learnable mask token |
| **Task Conditioning** | Learned `task_embed` (`2` entries): index `0` = predict L1, index `1` = predict L2; broadcast-added to all positions |
| **Positional Encoding** | **3D RoPE** — head_dim split into thirds: `row`, `column`, `slice` (z-axis). Row/col: up to `64` positions; slice: up to `92` positions. Base frequency = `25,000`. No learned position embeddings. |
| **Transformer** | `TransformerSDPA`: `TransformerBlockSDPA` stack — pre-norm `RMSNorm` (`eps=1e-6`), QKV linear → `F.scaled_dot_product_attention` (Flash SDP), `SwiGLU` FFN (`hidden_dim=4×embed_dim`, dropout=`0.0`) |
| **Prediction Heads** | `head_l1` (`embed_dim → codebook_size`), `head_l2` (`embed_dim → codebook_size`) |
| **Stack Depth** | `8` blocks, `8` heads |

> **Key Difference vs. Brain MRI:** Pelvic uses **3D RoPE** (row, col, slice) rather than 2D RoPE. The slice axis encodes the anatomical position along the z-axis.

#### Training Configuration

| Parameter | Value |
|:----------|:------|
| **Loss** | `CE(L1 masked) + l2_loss_weight × CE(L2 masked)` |
| **Label Smoothing** | `0.05` |
| **L2 Loss Weight** | `0.25` |
| **Q Error Weight** | `0.1` |
| **Optimizer** | `AdamW(β=(0.9, 0.98))` |
| **Weight Decay** | `0.01` (non-embedding/bias/norm params); `0.0` otherwise |
| **Scheduler** | Linear warmup (`2000` steps) → cosine decay |
| **Slice Filtering** | Only slices `[30, 60]` used in training; out-of-range dropped |

#### Masking Strategy of Tokens During Training

| Token | Masking |
|:------|:--------|
| **L1** | 70% of time: ratio ∈ [0.50, 0.75]; 30% of time: ratio ∈ [0.20, 0.50] |
| **L2** | ratio ∈ [0.15, 0.55] from `β(4, 4)` distribution |
| **Block Masking** | 50% of batches use random rectangles (union of overlapping blocks) instead of random masking |
| **Constraint** | At least 1 token masked per sample |
| **Validation** | Random masking (fixed `mask_ratio=0.20`) + fixed centre-region mask (inner ~67%×67%) |


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
pip install -r Pelvis_experiments_requirements.txt
```

#### Key Dependencies (Pinned Versions)

| Package | Version |
|:--------|:--------|
| `torch` | `2.8.0+cu128` |
| `pytorch-lightning` | `2.5.5` |
| `monai` | `1.5.1` |
| `vector-quantize-pytorch` | `1.27.15` |
| `transformers` | `4.57.2` |
| `open_clip_torch` | `3.2.0` |
| `lpips` | `0.1.4` |
| `nibabel` | `5.3.2` |
| `scipy` | `1.15.3` |
| `scikit-image` | `0.25.2` |
| `wandb` | `0.22.1` |
| `numpy` | `2.1.2` |

---

## Data Preparation

### 1. Pre-slice NIfTI volumes to .npy

Training operates on 2D axial slices stored as individual `.npy` files for fast I/O.

#### Training data (Only Normal/Reference cases, no anomalies)

```bash
python preslice_volumes.py
```

**Pipeline per volume:**

| Step | Operation |
|:-----|:----------|
| 1 | Load 3D NIfTI (`.nii` / `.nii.gz`) via nibabel |
| 2 | Per-volume **z-score normalisation**: `(x − mean) / std` (std clipped to ≥ 1e-8) |
| 3 | Save every axial slice as `{patient_id}_slice_{idx:03d}.npy` (float32) |
| 4 | Write `preslice_metadata.json` summary |

Default output: `../xxx/Data/PreSliced/`

#### Test / inference data (anomalous cohorts)

Use `External_dataset.py` for external NIfTI cohorts. This script applies **identical preprocessing** (z-score, 90° CCW rotation, resize 320→crop 256) and uses the following naming convention that encodes category metadata directly in the filename:

```
{category}_{case_folder}_{volume_name}_slice_{idx:03d}.npy
# Example: ClinicalVariations_T2_CUBE_FemaleBrachy_Cube1_slice_045.npy
```

The `category` and `case_folder` segments are parsed by `ROC_Curves_Calculations.py` to assign each slice to an anomaly category for stratified ROC analysis. The expected folder structure for the evlauation cohorts is:

```
<cohort_root>/
├── ClinicalVariations/
│   ├── Spacer/          *.nii.gz
│   ├── Hip_implants/   *.nii.gz
│   └── ...
└── SyntheticVariations/
    ├── Noise/           *.nii.gz
    ├── Motion/          *.nii.gz
    └── ...
```

**Naming convention is critical if you want to use our exact code:** The slice index embedded in the filename (e.g. `_slice_045`) is parsed at training time (Stage 2 slice filtering) and inference time (slice-position RoPE encoding, per-slice calibration lookup). Files that do not match `*_slice_*.npy` are silently ignored by `SliceDataModule`. Please make sure to modify our code if you follow other naming style. 


### 2. Dataset splits

`Train_Val_Test_Exact_DataSplits_LUND_PROBE.json` documents the exact patient-level train / validation / test split used in our paper:

| Split | Patients |
|:------|:---------|
| **Train** | `384` |
| **Validation** | 10% random from train (seed `42`) |
| **Test** | Separate cohorts in dedicated inference directories |


### 3. Preprocessing pipeline (applied during training data loading)

Each `.npy` slice passes through `dataset.py`:

| Step | Transform | Parameters |
|:-----|:----------|:-----------|
| 1 | 90° CCW rotation (`np.rot90`) | `k=-1` (anatomical orientation) |
| 2 | `EnsureChannelFirstD` | Adds channel dim |
| 3 | `Resized` | `(320, 320)`, area interpolation |
| 4 | `CenterSpatialCropd` | `(256, 256)` |
| 5 | `ToTensorD` | → float32 tensor |
| 6 (train only) | `RandFlipD` | Horizontal flip, `p=0.5` |
| 7 (train only) | `RandRotateD` | ±5° (0.0873 rad), `keep_size=True`, `p=0.3` |

> Stage 1 applies the augmentations **inside `training_step`** (after DataLoader), known as "online" augmentation.

---

## Training

### Stage 1 — RVQ-VAE

```bash
python Train_frameworks.py --stage1 \
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

| Parameter | Value | Description |
|:----------|:------|:------------|
| `embed_dim` | `256` | Token embedding dimension |
| `patch_size` | `8` | Patch size → `32×32` token grid |
| `encoder_depth` | `8` | Transformer encoder layers |
| `encoder_heads` | `8` | Attention heads |
| `codebook_size` | `192` | Codes per RVQ level |
| `num_quantizers` | `2` | Number of RVQ residual levels |
| `commitment_cost` | `0.25` | VQ commitment loss weight |
| `perceptual_weight` | `0.9` | BiomedCLIP perceptual loss weight |
| `lr` | `1e-4` | Initial learning rate |
| `max_epochs` | `100` | Training epochs |
| `batch_size` | `128` | Batch size |
| `gradient_clip_val` | `1.0` | Gradient norm clipping |
| `use_augmentations` | `True` | MONAI intra-step augmentation |

> **GPU:** Training uses `devices=[1]` (single GPU, index 1). Edit `make_trainer` if your setup differs.


### Stage 2 — Fact-biT 

Stage 2 requires a trained Stage 1 checkpoint. Stage 1 weights are **completely frozen** (gradients disabled).

```bash
python Train_frameworks.py --stage2 \
    --data-dir /path/to/PreSliced \
    --stage1-ckpt /path/to/stage1.ckpt \
    --batch-size 128 \
    --num-workers 8 \
    --max-epochs 100 \
    --lr 1e-4 \
    --precision 32 \
    --wandb-run-name Stage2-Factorized-MaskGIT
```

#### Exact hyperparameters used in our paper:

| Parameter | Value | Description |
|:----------|:------|:------------|
| `embed_dim` | `256` | Transformer embedding dimension |
| `depth` | `8` | Transformer blocks |
| `num_heads` | `8` | Attention heads |
| `mask_ratio` | `0.20` | Fixed mask ratio at validation |
| `mask_ratio_min` | `0.15` | β-distribution clip lower bound |
| `mask_ratio_max` | `0.75` | β-distribution clip upper bound |
| `beta_alpha` | `4.0` | β distribution α parameter |
| `beta_beta` | `4.0` | β distribution β parameter |
| `l2_loss_weight` | `0.25` | Weight of L2 token CE loss |
| `q_error_weight` | `0.1` | Weight of quantisation error in anomaly score |
| `label_smoothing` | `0.05` | Cross-entropy label smoothing |
| `warmup_steps` | `2000` | Linear LR warmup steps |
| `weight_decay` | `0.01` | AdamW weight decay (non-embed/bias/norm); `0.0` otherwise |
| `lr` | `1e-4` | Peak learning rate |
| `train_slice_min` | `30` | Minimum slice index for training |
| `train_slice_max` | `60` | Maximum slice index for training |
| `max_epochs` | `100` | Training epochs |
| `batch_size` | `128` | Batch size |
| `gradient_clip_val` | `1.0` | Gradient norm clipping |

**GPU:** Training uses `devices=[1]` (single GPU, index 1). Edit `make_trainer` in `train.py` if your setup differs.

### Logging and Checkpointing

| Logger | Output | Notes |
|:-------|:-------|:------|
| **CSVLogger** | `logs/<stage>/` | Always active |
| **WandbLogger** | wandb run | Disable with `--wandb-off`; set project with `--wandb-project` |

**Checkpoint pattern:**
```
lightningCheckpoints_Modified/Modified_Checkerboard_<stage>-epoch=<E>-val/loss=<L>.ckpt
```
Top-3 checkpoints by `val/loss` (min) are retained.

---



## Inference Pipeline

### Recursive-AutoMask V4 with Z-Score Normalization

#### Calibration Mode (Required Before Inference)

Run on held-out **Normal/Reference data** to learn per-pixel μ and σ of LPIPS reconstruction error within your reference population:

```bash
python Inference_Pelvis_Experiments.py \
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

**Calibration `.npz` contents:**

| Key | Shape | Description |
|:-----|:------|:------------|
| `mu` | `(H, W)` | Per-pixel mean LPIPS across healthy slices (after spatial smoothing) |
| `sigma` | `(H, W)` | Per-pixel standard deviation |
| `n_samples` | scalar | Number of healthy slices used |
| `smoothing_kernel` | scalar | Kernel size (must match inference) |
| `mu_slice_<idx>` | `(H, W)` | Per-slice-index mean (when `use_per_slice_stats=True`, ≥3 samples) |
| `sigma_slice_<idx>` | `(H, W)` | Per-slice-index std |

A visualisation (`calibration_visualization.png`) showing μ, σ, μ/σ ratio and false-positive (FP)-prone regions is saved automatically.

> **Key Design:** Spatial smoothing (`avg-pool, kernel=15`) is applied identically to calibration and inference heatmaps. This makes statistics represent coarse anatomical regions rather than exact pixel locations; critical as patient-to-patient registration and anatomical differences are expected.

#### Inference Mode

```bash
python Inference_Pelvis_Experiments.py \
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

**Step 1 — (Optional) Sharpness computation**

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

Each slice in the output JSON have:

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
patient-level score = sum of "token_surprisal_hot_px + Binary_Sum_Heatmap" over selected slices.
ROC-analysis utilizes this score. 
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

## ROC-analysis 

`ROC_Curves_Calculations.py` reads per-slice JSON result files and computes patient-level ROC/AUC curves stratified by anomaly category.

### Default result JSON paths

The script reads from different directories by default (edit `DEFAULT_ROC_INPUT_PATHS` to match your setup):

```python
DEFAULT_ROC_INPUT_PATHS = [
    ".../Inference_Results_LUND_PROBE_Volunteer_Clinical/results_v4_zscore.json",
    ".../Inference_Results_LUND_PROBE_CervixBrachy/results_v4_zscore.json",
    ".../Inference_Results_LUND_PROBE_Clinical/results_v4_zscore.json",
    ".... etc. "
]
```

### Patient-Level Aggregation Strategies

| Function | Metric Used | Description |
|:---------|:------------|:------------|
| `aggregate_patient_clamp_from_results` | `clamped_pixel_sum` | Sums clamped LPIPS pixel intensities across all slices |
| `collect_patient_binary_sums` | `Binary_Sum_Heatmap` | Counts binary anomaly pixels per patient |
| `collect_patient_sharpness_totals` | `sharpness_score` | Aggregated Laplacian sharpness (negative control) |
| `aggregate_patient_status` | `combined_score > threshold` | Votes by number of flagged slices exceeding threshold |


### 📊 Running ROC-analysis 

```bash
python ROC_Curves_Calculations.py \
    --input /path/to/results_v4_zscore.json \
    --output-dir /path/to/roc_figures \
    --categories all
```

#### Step 2: Compute ROC, AUC, and CI-bands

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


## Acknowledgements

- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) — ResidualVQ implementation
- [MONAI](https://monai.io/) — Medical imaging transforms and data utilities
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) — Domain-adapted perceptual loss during Stage 1 training
- [lpips](https://github.com/richzhang/PerceptualSimilarity) — Spatial LPIPS used in the inference heatmap
- [PyTorch Lightning](https://lightning.ai/) — Training framework


