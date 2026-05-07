# Two-Stage Unsupervised Anomaly Detection for Pelvic MRI (LUND-PROBE)

This repository contains the cleaned Pelvic MRI implementation of a two-stage unsupervised anomaly-detection framework. The models are trained on normal/reference pelvic MRI slices and evaluated on synthetic and clinical anomaly cohorts.

The code has been organized around a **CORE vs. AYNU** concept:

- **CORE** = code that directly contributes to the manuscript AUROC/AUPRC reproduction pipeline.
- **AYNU** = “Available Yet Not AUROC-interesting” auxiliary code retained for transparency, debugging, training diagnostics, visualizations, calibration generation, alternative scores, and supplementary analyses, but not on the primary AUROC path.

The primary AUROC path in the cleaned Pelvis code is:

```text
input slice
  → Stage 1 RVQ-VAE tokens/reconstruction
  → Stage 2 Fact-biT healing
  ├─ token-surprisal branch → token_surprisal_hot_px
  └─ LPIPS healing branch → Binary_Sum_Heatmap
  → per-patient sum_all_bars_score
       = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
  → ROC / AUROC and PR / AUPRC
```

Ground-truth anomaly labels are **not used for model training**. They are used only for evaluation, cohort/category assignment, plotting, and optional annotation overlays.

---

## Repository contents

```text
Final_Clean_to_Github_Pelvis/
├── Model_Stage_1.py                         # Stage 1 RVQ-VAE model
├── Model_Stage_2.py                         # Stage 2 Factorized MaskGIT / Fact-biT model
├── Train_framework.py                       # PyTorch Lightning training entry point
├── dataset.py                               # .npy slice Dataset/DataModule
├── Inference_Pelvis_Experiments.py          # Recursive-AutoMask V4 inference + calibration
├── ROC_Curves_Calculations.py               # Patient-level ROC/AUPRC and category analyses
├── config_yaml.yaml                         # Reference configuration summary; not auto-loaded by scripts
├── Instructions.md                          # Internal CORE/AYNU refactor blueprint
├── Train_Val_Test_Exact_DataSplits_LUND_PROBE.json # Recorded train/val/test split information                                          
├── preslice_volumes.py                      # NIfTI volume → per-slice .npy preprocessing
├── External_dataset.py                      # External cohort preprocessing/loading utilities
├── Simulation_inference_v4_extended_CJG.py  # Synthetic anomaly generation helpers
├── Simluation_inference_v3_support_CJG.py   # Support code for synthetic data generation
└── Pelvis_Experiments_requirements.txt      # Pinned Python environment from the experiment
```

> **Important:** Several scripts contain absolute local default paths from the original experiment environment. For a new machine or GitHub user, pass explicit CLI paths or edit the defaults before running.

---

## CORE concept and main score

The cleaned code is deliberately arranged so a reader can identify the AUROC-producing path quickly.

### CORE per-slice fields

The two per-slice JSON fields consumed by the main patient-level ROC pipeline are:

```text
token_surprisal_hot_px
Binary_Sum_Heatmap
```

`ROC_Curves_Calculations.py` aggregates them with:

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

The patient-level ROC/AUPRC is computed from `sum_all_bars_score` by `compute_patient_roc_and_auc(...)`.

### AYNU examples

The following are useful and preserved, but they do not define the primary AUROC score unless explicitly used in separate analyses:

- `clamped_pixel_sum`
- `lpips_input_recon_sum_mask`
- sharpness totals and sharpness-based plots
- per-patient bar plots of intermediate quantities
- Stage 2 alternative anomaly-map methods
- token-frequency diagnostics
- calibration-generation figures
- annotation overlay figures
- synthetic-data generation utilities

---

## Method overview

The framework has two learned stages.

| Stage | Model | Purpose |
|---|---|---|
| **Stage 1** | RVQ-VAE with ViT encoder, residual vector quantization, and PixelShuffle decoder | Learns a discrete latent representation of normal/reference pelvic MRI appearance |
| **Stage 2** | Factorized MaskGIT / Fact-biT transformer | Learns token distributions and heals masked/suspect tokens using bidirectional masked prediction |

At inference, **Recursive-AutoMask V4** computes two complementary anomaly signals:

1. **Token surprisal:** repeated random masking of Stage 1 L1 tokens, followed by Stage 2 prediction and NLL scoring of the true tokens.
2. **LPIPS healing heatmap:** Stage 2 heals checkerboard-masked tokens; spatial LPIPS compares the input image with healed/inpainted images; calibrated Z-score thresholding converts the heatmap to a binary detection mask.

Unlike the cleaned Brain README, the Pelvis inference code’s main LPIPS branch is **input-referenced**:

- calibration uses **LPIPS(input, healed)**
- inference iteration 0 uses **LPIPS(input, healed)**
- refinement iterations use **LPIPS(input, inpainted)**

`lpips_input_recon` is computed for auxiliary diagnostics/visualizations, not for the main AUROC score.

---

## Architecture details

### Stage 1 — `Model_Stage_1.py`

`Stage1RVQVAE` maps a 2D grayscale pelvic MRI slice to RVQ tokens and reconstructs the slice.

| Component | Implementation detail |
|---|---|
| Input | Single-channel 2D pelvic MRI slice, typically `1 × 256 × 256` |
| Patch embedding | Conv2d with `kernel_size = stride = patch_size` |
| Patch size in training script | `8`, giving `32 × 32 = 1024` tokens |
| Encoder | ViT-style Transformer encoder, depth 8, 8 heads |
| Multi-scale encoder | Present in the model and used by auxiliary/multiscale paths |
| Quantizer | `ResidualVQ`, 2 quantizers, codebook size 192 in the training script |
| Decoder | PixelShuffle decoder back to one image channel |
| Forward output | `recon`, `indices`, `commit_loss`, `quant_error_map` |

Stage 1 training loss:

```text
L1 reconstruction loss
+ BiomedCLIP perceptual loss, weight 0.9
+ RVQ commitment loss
```

Training uses AdamW with a cosine annealing learning-rate schedule. Stage 1 also contains training-time augmentation and validation visualization code, which is AYNU relative to the inference AUROC path.

### Stage 2 — `Model_Stage_2.py`

`FactorizedMaskGIT` predicts masked RVQ tokens from Stage 1.

| Component | Implementation detail |
|---|---|
| Token levels | Separate L1 and L2 token streams |
| Codebook size | 192 per level when loaded from the Stage 1 checkpoint used by `Train_framework.py` |
| Transformer | SDPA transformer with RMSNorm and SwiGLU |
| Position encoding | **3D RoPE** over row, column, and slice position |
| Sequence length | Derived from Stage 1 image size and patch size; typically 1024 |
| Stage 1 during Stage 2 training | Frozen and set to eval mode |

Stage 2 training loss:

```text
CE(masked L1 tokens) + 0.25 × CE(masked L2 tokens)
```

with label smoothing (`0.05`) and a mixed random/block masking strategy.

A key pelvic-specific feature is slice-position conditioning through 3D RoPE. The model extracts slice indices from filenames such as:

```text
patient_id_slice_045.npy
```

Slice indices are used for Stage 2 anatomical position encoding and for optional per-slice calibration statistics.

---

## Environment

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

If you need exact checkpoint reproduction, use the pinned environment rather than unpinned latest package versions.

---

## Data format and preprocessing

### Training slice format

Training uses individual `.npy` slices. The expected filename pattern is:

```text
{patient_id}_slice_{idx:03d}.npy
```

Files that do not contain `_slice_` are ignored by `SliceDataModule`.

### `dataset.py` preprocessing

Each `.npy` slice is loaded and transformed as follows:

1. load `float32` NumPy array
2. rotate with `np.rot90(arr, k=-1)`
3. add channel dimension with MONAI `EnsureChannelFirstD`
4. resize to `320 × 320` using area interpolation
5. center crop to `256 × 256`
6. convert to tensor
7. optional training augmentation if `augment=True` in the DataModule:
   - horizontal flip
   - small rotation around ±5°

In the current `Train_framework.py`, `SliceDataModule` is instantiated without passing `augment=True`, so DataModule augmentations are disabled unless the script is modified. Stage 1 also has its own internal augmentation logic in `training_step`.

### Pre-slicing LUND-PROBE / normal-reference NIfTI data

`preslice_volumes.py` converts 3D NIfTI volumes into `.npy` slices.

```bash
python preslice_volumes.py
```

Main behavior:

- reads source NIfTI paths from the script/config environment
- z-score normalizes per volume
- saves every axial slice as `{patient_id}_slice_{idx:03d}.npy`
- writes `preslice_metadata.json`

Because `preslice_volumes.py` is script/config driven, check or edit its path settings before running.

### External/synthetic/clinical cohorts

`External_dataset.py` contains utilities for processing external NIfTI cohorts and preserving category/case metadata. Its preprocessing utilities include:

- NIfTI loading
- slice normalization
- resize/crop
- per-slice saving
- category and case-folder tracking

The downstream ROC code uses patient/case/category identifiers to stratify results into synthetic and clinical groups.

---

## Training

The training entry point is:

```text
Train_framework.py
```

### Stage 1 training

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

Current script-level settings:

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

### Stage 2 training

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

Stage 2 requires a valid Stage 1 checkpoint. Stage 1 is frozen during Stage 2 training.

Stage 2 also filters training slices to the anatomically relevant slice-index range, documented in the config as approximately:

```text
train_slice_min = 30
train_slice_max = 60
```

This depends on correctly encoded `_slice_###` filenames.

### Logging and checkpointing

`Train_framework.py` uses:

- `CSVLogger`
- optional `WandbLogger`
- `LearningRateMonitor`
- `ModelCheckpoint`

Disable W&B with:

```bash
--wandb-off
```

Privacy note: do not log patient-identifiable information to W&B, filenames, plots, or shared logs.

---

## Inference and calibration

The main inference script is:

```text
Inference_Pelvis_Experiments.py
```

### Model loading

`load_models(stage1_ckpt, stage2_ckpt, device)` loads Stage 1 and Stage 2 checkpoints. Stage 1 perceptual-loss keys are stripped during inference loading so that BiomedCLIP weights are not required for checkpoint loading at inference time.

### Step 1 — healthy/reference calibration

Calibration estimates per-pixel healthy/reference LPIPS statistics:

```text
mu[h, w]    = mean LPIPS(input, healed) value over calibration slices
sigma[h, w] = std  LPIPS(input, healed) value over calibration slices
```

Example:

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
| `mu` | global per-pixel mean LPIPS map |
| `sigma` | global per-pixel std LPIPS map |
| `n_samples` | number of calibration slices |
| `smoothing_kernel` | smoothing kernel used before statistics |
| per-slice entries | optional slice-index-specific statistics when enough samples exist |

> **Critical reproducibility rule:** use the same `--smoothing-kernel` during calibration and inference.

Calibration generation itself is marked AYNU in the refactor blueprint because the AUROC path loads an existing calibration map; nevertheless, calibration is required to create that map for a new dataset/reference population.

### Step 2 — anomaly inference

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

### Current important inference defaults

The inspected script currently defines defaults including:

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

These defaults are experiment-specific. Save the exact CLI command with each run.

### Main inference steps

Inside `recursive_automask_v4_zscore(...)`, the CORE flow is:

1. load input slice and extract slice position from filename when available
2. compute Stage 1 tokens/reconstruction
3. compute token surprisal through repeated L1 token masking
4. heal checkerboard-masked tokens using Stage 2
5. compute LPIPS between input and healed image(s)
6. aggregate native and TTA heatmaps, commonly with `geomean`
7. smooth and threshold by Z-score using the calibration map
8. compute `Binary_Sum_Heatmap` from the masked heatmap and binary threshold
9. write per-slice JSON fields
10. aggregate patient scores in `ROC_Curves_Calculations.py`

Targeted inpainting and multi-iteration refinement are implemented and preserved, but the current AUROC default is `--num-iterations 1`, so later refinement iterations are not part of the default CORE score.

### Output

The main JSON output is:

```text
results_v4_zscore.json
```

Important fields include:

| Field | Meaning |
|---|---|
| `token_surprisal_hot_px` | CORE count of hot token-surprisal pixels after clamping |
| `Binary_Sum_Heatmap` | CORE count of binary-positive masked LPIPS pixels |
| `path`, `filename` | source slice path/name |
| `category` | cohort/anomaly category metadata |
| `case_folder` | patient/case grouping metadata |
| `used_zscore` | whether calibration Z-score thresholding was active |
| `z_threshold` | Z-score threshold used if active |
| `clamped_pixel_sum` | AYNU auxiliary LPIPS-derived score |
| `lpips_input_recon_sum_mask` | AYNU reconstruction diagnostic |
| `sharpness_score`, `artifact_flag` | AYNU artifact/sharpness diagnostics |
| `iteration_metrics` | AYNU iteration/refinement diagnostics |

---

## ROC / AUPRC evaluation

The ROC script is:

```text
ROC_Curves_Calculations.py
```

### Main merged-ROC workflow

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

The patient-level score is:

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

Labels are assigned from patient/case identifiers: “orig” cases are treated as normal/reference (`label=0`) and all other cases as anomaly (`label=1`). Confirm this naming convention before using the ROC script on new cohorts.

### Outputs

The ROC script can produce:

- merged JSON payloads
- ROC curve figures
- precision-recall curve figures
- AUROC/AUPRC metrics JSON
- bootstrap confidence intervals
- threshold tables
- synthetic-vs-clinical split curves
- category-stratified sensitivity tables

The script also contains many AYNU plotting functions for intermediate patient-level summaries.

---

## Reference configuration file

`config_yaml.yaml` documents the intended experiment settings in a structured format. It is useful as a reference, but the inspected main Python scripts do **not** automatically load it as the sole runtime source of truth.

Active behavior is controlled by:

- CLI arguments
- hard-coded defaults inside Python scripts
- checkpoint hyperparameters
- filename conventions
- actual input JSON/data paths selected at runtime

For reproducibility, keep:

1. exact training commands
2. exact inference commands
3. checkpoint paths/checkpoint hashes if available
4. calibration `.npz` file
5. `results_v4_zscore.json`
6. merged ROC JSON/metrics outputs
7. the code version
8. the split manifest JSON

---

## Synthetic anomaly utilities

The repository includes helper scripts for synthetic anomaly generation:

- `Simulation_inference_v4_extended_CJG.py`
- `Simluation_inference_v3_support_CJG.py`

These are not on the primary AUROC computation path, but they document and support generation of synthetic variations such as blur/noise/inserted structures used in the broader experimental workflow.

Be careful not to write identifiable DICOM metadata or patient information to public outputs when using these utilities.

---

## Exact replication checklist

Use this checklist when trying to reproduce the Pelvis experiment.

- [ ] Normal/reference NIfTI volumes were pre-sliced to `.npy` with the expected `_slice_###` naming convention.
- [ ] Training/validation/test splits are recorded and reused.
- [ ] Stage 1 checkpoint path is recorded.
- [ ] Stage 2 checkpoint path is recorded.
- [ ] Calibration data are normal/reference and independent from anomaly evaluation cohorts.
- [ ] `--smoothing-kernel` is identical between calibration and inference.
- [ ] Slice indices are preserved in filenames for 3D RoPE and per-slice calibration lookup.
- [ ] The exact inference CLI command is saved.
- [ ] `results_v4_zscore.json` is retained for each cohort.
- [ ] ROC is computed from `sum_all_bars_score = token_surprisal_hot_px + Binary_Sum_Heatmap` summed over patient slices.
- [ ] “orig”/normal identifiers are correct before assigning ROC labels.
- [ ] Ground-truth/category labels are not used for model training.
- [ ] No patient-identifying information is exposed in public logs, filenames, figures, or W&B runs.

---

## Key differences from the Brain MRI version

| Aspect | Pelvic MRI code in this folder | Brain MRI code |
|---|---|---|
| Anatomy/domain | Pelvic MRI | Brain MRI |
| Main data source | LUND-PROBE-style pelvic MRI workflow | IXI + fastMRI-style brain workflow |
| Stage 2 positional encoding | **3D RoPE** over row, column, slice | 2D RoPE over row/column |
| Codebook size used by training script | 192 per RVQ level | 256 per RVQ level |
| Main ROC score | patient sum of `token_surprisal_hot_px + Binary_Sum_Heatmap` | patient sum of `Binary_Sum_Heatmap` |
| Main LPIPS reference | input-vs-healed/inpainted | reconstruction-vs-healed/inpainted |
| File format | `.npy` slices | primarily `.npz` with key `arr` |
| Slice index importance | used for 3D RoPE and per-slice calibration | not used for 2D RoPE |

---

## Practical notes for GitHub readers

- This is research code, not a clinically validated tool.
- Do not use model outputs for clinical decisions.
- The code is slice-based; patient-level scores are produced by aggregating slice scores.
- Avoid slice-level leakage. Splits should be patient-level whenever possible.
- Preserve filename slice indices (`_slice_###`) because they affect Stage 2 slice conditioning and calibration lookup.
- Be careful with orientation, rotation, resizing, and category/case naming conventions when preparing new data.
- Many defaults are local to the original workstation; override paths explicitly.
- The CORE/AYNU comments in the Python files are meant to help readers identify exactly what contributes to the reported AUROC/AUPRC.
