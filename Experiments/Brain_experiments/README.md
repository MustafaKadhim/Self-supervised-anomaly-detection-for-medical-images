# Two-Stage Unsupervised Anomaly Detection for Brain MRI (IXI / fastMRI)

This repository contains the cleaned Brain MRI implementation of a two-stage unsupervised anomaly-detection framework. The training pipeline learns only from normal/reference T1-weighted brain MRI slices and the inference pipeline detects deviations from the learned normal/reference distribution.

The code has been organized around a **CORE vs. AYNU** concept:

- **CORE** = code that directly contributes to the AUROC-producing pipeline.
- **AYNU** = “Available Yet Not Used” auxiliary code retained for reproducibility, debugging, training diagnostics, visualizations, bounding-box analysis, and alternative scores, but not on the primary AUROC path.

The primary AUROC path is:

```text
input slice
  → Stage 1 RVQ-VAE reconstruction / tokens
  → Stage 2 MaskGIT healing / inpainting
  → LPIPS reconstruction-vs-healed heatmap
  → binary + token-surprisal fusion
  → per-slice Binary_Sum_Heatmap
  → patient-level sum of Binary_Sum_Heatmap
  → ROC / AUROC
```

Ground-truth anomaly labels and bounding boxes are **not used for model training**. They are used only for evaluation, filtering, folder curation, and visualization.

---

## Repository contents

```text
Final_Clean_to_Github_Brain/
├── Model_Stage1.py                         # Stage 1 RVQ-VAE model
├── Model_Stage_2.py                         # Stage 2 Fact-biT model
├── Train_framework.py                       # PyTorch Lightning training entry point
├── dataset.py                               # Slice Dataset/DataModule for .npz/.png slices
├── Inference_Brain_Experiments.py           # Recursive-AutoMask V4 inference + calibration
├── ROC_Curve_Calculations.py                # Patient-level ROC/AUROC and related metrics
├── config_yaml.yaml                         # Reference configuration summary; not auto-loaded by scripts
├── Instructions_Brain.md                    # Internal CORE/AYNU refactor instructions
├── Train_Val_Test_Exact_DataSplits_IXI_fastMRI.json # Recorded train/val/test split information
│                                             
├── IXI_dataset_overview.py                  # IXI NIfTI → training-ready .npz preprocessing
├── Render_patient_slices_from_csv.py        # fastMRI .h5 → .npz/PNG rendering utility
├── collect_normal_slices.py                 # Select normal slices from annotation CSVs
├── build_patient_Global_label_folders.py    # Build study/global-label anomaly folders
├── build_patient_Local_label_folders.py     # Build per-slice/local-label anomaly folders
└── Inference_heatmaps_ideas_generator.py    # Optional heatmap visualization helper
```

> **Important:** Many scripts still contain absolute local default paths from the original experiment environment. For a new machine or GitHub user, pass explicit CLI paths instead of relying on defaults.

---

## CORE concept and main score

The codebase intentionally distinguishes the AUROC-producing path from auxiliary analyses.

### CORE output field

The only per-slice field consumed by the primary patient-level ROC pipeline is:

```text
Binary_Sum_Heatmap
```

`ROC_Curve_Calculations.py` aggregates this field into a patient-level score named:

```text
binary_token_score = sum(Binary_Sum_Heatmap over all included slices for a patient)
```

Then `compute_fastmri_roc_and_auc(...)` computes ROC/AUROC using `binary_token_score` and cohort labels.

### AYNU examples

The following are useful and preserved, but they are not relevant for primary AUROC calculations unless explicitly selected in additional analysis:

- bounding-box overlap metrics
- per-slice precision / F1 localization metrics
- reconstruction-quality figures
- Stage 2 alternative anomaly-map methods
- token-frequency summaries
- heatmap idea figures
- per-patient bar plots and threshold tables
- `clamped_pixel_sum` and other auxiliary JSON fields

---

## Method overview

The framework has two learned stages.

| Stage | Model | Purpose |
|---|---|---|
| **Stage 1**   | RVQ-VAE with ViT encoder, multi-scale encoder, residual vector quantization, PixelShuffle decoder | Learns a discrete latent representation of normal/reference anatomy |
| **Stage 2**   | Fact-biT transformer | Learns distributions over Stage 1 tokens and heals masked tokens |

At inference, **Recursive-AutoMask V4** performs calibration, healing, perceptual comparison, binary-mask fusion, and optional targeted inpainting. The main heatmap branch is reconstruction-referenced:

- calibration uses **LPIPS(Stage 1 reconstruction, healed reconstruction)**
- inference iteration uses **LPIPS(Stage 1 reconstruction, healed reconstruction)**
- refinement iterations use **LPIPS(Stage 1 reconstruction, inpainted reconstruction)**

The script also computes `lpips_input_recon`, but this is mainly for auxiliary visualization/analysis rather than the primary path.

---

## Architecture details

### Stage 1 — `Model_Stage1.py`

`Stage1RVQVAE` maps a 2D grayscale MRI slice to RVQ tokens and reconstructs the image.

| Component | Implementation detail |
|---|---|
| Input | Single-channel 2D slice, typically `1 × 256 × 256` |
| Patch embedding | Conv2d with `kernel_size = stride = patch_size` |
| Default patch size in training script | `8`, giving `32 × 32 = 1024` tokens |
| Encoder | ViT-style Transformer encoder, depth 8, 8 heads |
| Multi-scale encoder | Convolutional feature pyramid fused with attention |
| Quantizer | `ResidualVQ`, 2 quantizers, codebook size 256 in the training script |
| Decoder | PixelShuffle decoder, output clamped to `[-3, 3]` |
| Forward output | `recon`, `indices`, `commit_loss`, `quant_error_map` |

Training loss:

```text
L1 reconstruction loss (MAE)
+ BiomedCLIP perceptual loss, with weight 0.5
+ RVQ commitment loss
```

Stage 1 training also includes rich MONAI augmentations when enabled:

- intensity scaling
- contrast adjustment
- Gaussian noise
- affine rotation/translation/zoom
- horizontal flip

> In `Train_framework.py`, Stage 1 is instantiated with `use_augmentations=False` in the current script body even though the DataModule can apply augmentations through `--augment`. Therefore, the active augmentation source depends on the training command/script settings. Record the exact command used for reproducibility.

### Stage 2 — `Model_Stage_2.py`

`FactorizedMaskGIT` predicts masked RVQ tokens from Stage 1.

| Component | Implementation detail |
|---|---|
| Token levels | Separate L1 and L2 token embeddings |
| Codebook size | 256 per level when loaded from the trained Stage 1 used by `Train_framework.py` |
| Transformer | SDPA transformer with RMSNorm and SwiGLU |
| Position encoding | 2D rotary embeddings over row/column token positions |
| Sequence length | Derived from Stage 1 image size and patch size; typically 1024 |
| Stage 1 during Stage 2 training | Frozen and set to eval mode |

Training loss:

```text
CE(masked L1 tokens) + 0.25 × CE(masked L2 tokens)
```

with label smoothing (`0.05`) and a mixed random/block masking strategy.

> Some comments/docstrings in the model still mention “3D RoPE” for backward compatibility with the earlier pelvis/volumetric code. The Brain implementation in this folder uses 2D RoPE: row and column only. `slice_pos` may be accepted by call signatures but is not part of the Brain positional encoding.

---

## Environment

The experiments used Python packages available in the requirement.txt file above. 

Notes:

- Stage 1 training can use BiomedCLIP via `transformers` / `open_clip_torch`.
- Inference uses LPIPS (`lpips` package, VGG backbone) for the spatial perceptual heatmap.
- CUDA device defaults in scripts may point to `cuda:1`; override with CLI arguments if you work with single GPU.

---

## Data format

### Training/inference slice files

The standard saved array format is:

```text
.npz file containing key: arr
```

where `arr` is a 2D `float32` image slice.

`dataset.py` supports `.npz` and image files such as `.png` for training/validation, depending on `--file-ext`. The inference dataloader in `Inference_Brain_Experiments.py` recursively supports `.npz` and `.npy` files.

### Recommended directory structure for inference

The inference script stores the immediate parent folder name as `case_folder` in the output JSON. For patient-level aggregation, use one folder per patient/case when possible:

```text
case_folder_dir/
├── patient_001/
│   ├── patient_001_slice_003.npz
│   └── patient_001_slice_004.npz
└── patient_002/
    └── patient_002_slice_005.npz
```

`ROC_Curve_Calculations.py` can then aggregate slices by patient/case and filter by `case_folder` or category.

---

## Data preparation

### IXI normal/reference training data

Use `IXI_dataset_overview.py` to convert IXI T1 NIfTI volumes to 2D `.npz` slices.

Example:

```bash
python IXI_dataset_overview.py \
    --input-dir /path/to/IXI-T1 \
    --output-npy-dir /path/to/Training_samples_FastMRI_IXI \
    --training-ready \
    --training-slice-start 128 \
    --training-slice-end 188 \
    --z-clip "-3,3" \     #here we selected z-score normalization with clamp(-3,3) 
    --intensity-scale none \
    --pattern "*.nii.gz" \
    --recursive
```

Main preprocessing steps:

1. load NIfTI volume
2. reorient to closest canonical orientation
3. z-score normalize per volume
4. clip to the requested range, commonly `[-3, 3]`
5. crop/pad in-plane to `256 × 256`
6. rotate exported slices with `np.rot90(..., k=1)`
7. save `.npz` files with key `arr`

The default documented slice range `128–188` was used to focus on informative axial brain slices and avoid many non-informative superior/inferior slices.

### fastMRI rendering / anomaly folder preparation

`Render_patient_slices_from_csv.py` converts fastMRI `.h5` volumes into `.npz` and/or PNG slices for review and inference.

It can read either:

- a CSV containing patient/slice requests, or
- a label-root folder produced by `build_patient_Global_label_folders.py` or `build_patient_Local_label_folders.py`.

Example:

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

Important preprocessing difference:

- IXI training slices come from NIfTI files.
- fastMRI anomaly/evaluation slices come from `.h5` `reconstruction_rss` volumes, are normalized per volume, may be vertically flipped for orientation/display consistency, then resized/cropped to the saved 2D representation.

### Normal-slice collection for calibration

`collect_normal_slices.py` helps identify normal slices from annotation CSVs.

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

### Global and local anomaly label folders

Global/study-level labels:

```bash
python build_patient_Global_label_folders.py \
    --anomalies-dir /path/to/FastMRI_Anomalies_Collection \
    --detailed-csv /path/to/Annotated_fastMRI_Brains_Detailed.csv \
    --output-dir /path/to/FastMRI_Global_Anomalies_ByLabel \
    --use-detailed
```

Local/per-slice labels:

```bash
python build_patient_Local_label_folders.py \
    --anomalies-dir /path/to/FastMRI_Anomalies_Collection \
    --detailed-csv /path/to/Annotated_fastMRI_Brains_Detailed.csv \
    --output-dir /path/to/FastMRI_Local_Anomalies_ByLabel
```

These scripts create label-specific folders and patient/slice CSVs useful for running inference category by category.

---

## Training

The training entry point is:

```text
Train_framework.py
```

### Stage 1 training

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

Current script-level constants/defaults:

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

Checkpoints are written under the hard-coded experiment checkpoint directory unless the script is edited.

### Stage 2 training

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

Stage 2 requires a valid Stage 1 checkpoint. The Stage 1 model is frozen during Stage 2 training.

### Logging

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
Inference_Brain_Experiments.py
```

### Model loading

`load_models(stage1_ckpt, stage2_ckpt, device)` loads Stage 1 and Stage 2 checkpoints. Stage 1 perceptual-loss keys are stripped during inference loading so that BiomedCLIP is not required for inference checkpoint loading.

### Step 1 — normal/reference calibration

Calibration estimates per-pixel normal/reference LPIPS statistics:

```text
mu[h, w]    = mean LPIPS reconstruction-vs-healed value over normal/reference calibration slices
sigma[h, w] = std  LPIPS reconstruction-vs-healed value over normal/reference calibration slices
```

Example:

```bash
python Inference_Brain_Experiments.py \
    --calibration-mode \
    --stage1-ckpt /path/to/stage1.ckpt \
    --stage2-ckpt /path/to/stage2.ckpt \
    --data-dir /path/to/normal/reference_calibration_slices \
    --output-dir /path/to/calibration_output \
    --calibration-map /path/to/zscore_calibration.npz \
    --smoothing-kernel 7 \
    --heal-patterns "4" \
    --device cuda:0
```

Calibration writes an audit file:

```text
calibration_input_files.txt
```

Keep this file with the experiment outputs because it records which slices were used for calibration.

> **Critical reproducibility rule:** use the same `--smoothing-kernel` during calibration and inference.

### Step 2 — anomaly inference

Example using the current script-style defaults as a starting point:

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

The script supports many additional switches for filtering, batch running over label folders, annotation coordinate handling, TTA, binary fusion, LPIPS backflow, visualization, and output control. Because defaults are experiment-specific, save the exact CLI command with each run.

### Current important inference defaults

The inspected script currently defines defaults including:

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

These are not necessarily identical to paper-like/recommended ablation settings. Treat them as the code defaults at the time of this README update.

### Main inference steps

Inside `recursive_automask_v4_zscore(...)`, the CORE flow is:

1. compute Stage 1 reconstruction and RVQ tokens
2. compute sharpness map/score for motion-blur awareness
3. compute token surprisal by Monte Carlo token masking
4. heal masked token patterns using Stage 2 MaskGIT
5. compute LPIPS between Stage 1 reconstruction and healed/inpainted reconstruction
6. aggregate ensemble heatmaps
7. threshold with calibration Z-score or fallback percentile logic
8. optionally refine with targeted token inpainting
9. fuse LPIPS binary mask, token-surprisal mask, LPIPS backflow, and edge cleanup
10. write per-slice JSON fields, including `Binary_Sum_Heatmap`

### Output

The main JSON output is:

```text
results_v4_zscore.json
```

Important fields include:

| Field | Meaning |
|---|---|
| `Binary_Sum_Heatmap` | CORE per-slice pixel count used for patient-level AUROC aggregation |
| `Binary_Sum_Heatmap_Base` | base binary component |
| `Binary_Sum_Heatmap_Token` | token-surprisal binary component |
| `Binary_Sum_Heatmap_Overlap` | overlap component |
| `case_folder` | immediate parent folder of the slice file |
| `category` | CLI-supplied category or inferred batch category |
| `sharpness_score` | motion/blur-related auxiliary score |
| `clamped_pixel_sum` | auxiliary LPIPS-derived score; not the primary AUROC score |
| `has_ground_truth_bbox` | whether a matched annotation box exists |
| `num_true_positive_bboxes` | auxiliary bounding-box localization count |
| `inside_bbox_detection_ratio` | auxiliary localization ratio |
| `precision`, `f1_score` | auxiliary per-slice localization metrics |

---

## Annotation and bounding-box evaluation

Annotation CSVs are expected to contain columns such as:

```text
file, slice, x, y, width, height, label, study_level, base_size
```

`Inference_Brain_Experiments.py` supports multiple coordinate preprocessing modes, including:

- `legacy`
- `render_fastmri`
- `mask_pipeline`

It also supports optional annotation flips:

- `--annotation-flip-vertical`
- `--annotation-flip-horizontal`

Bounding-box metrics are useful for localization analysis but are AYNU relative to the primary patient-level AUROC path.

---

## ROC / AUROC evaluation

The ROC script is:

```text
ROC_Curve_Calculations.py
```

Example:

```bash
python ROC_Curve_Calculations.py \
    --input /path/to/results_v4_zscore.json \
    --output-dir /path/to/roc_outputs
```

The CORE aggregation is performed by:

```text
aggregate_fastmri_binary_token_patient_scores(...)
```

which sums `Binary_Sum_Heatmap` over all included slices for each patient/case.

`compute_fastmri_roc_and_auc(...)` then computes patient-level ROC/AUROC using:

```text
score = binary_token_score
label = 0 for included test-normal patients
label = 1 for anomaly patients
```

Validation normals may be excluded depending on script options/policy. This prevents validation-normal slices from being mixed into the final test-normal ROC cohort unless intentionally enabled.

> Older README text referred to patient-level aggregation of `clamped_pixel_sum`. That is not the CORE AUROC path in the current cleaned Brain code. The primary ROC path uses `Binary_Sum_Heatmap` → `binary_token_score`.

---

## Reference configuration file

`config_yaml.yaml` is a structured reference summary of experiment settings. It is useful documentation, but the main Python scripts inspected here do **not** automatically load it as the runtime source of truth.

Active behavior is controlled by:

- CLI arguments
- hard-coded defaults inside the Python scripts
- checkpoint hyperparameters
- the actual data files selected at runtime

For reproducibility, keep:

1. the exact CLI command
2. the checkpoint paths/checkpoint hashes if available
3. `calibration_input_files.txt`
4. the produced `results_v4_zscore.json`
5. the version of this code folder
6. the train/validation/test split JSON if relevant

---

## Exact replication checklist

Use this checklist when trying to reproduce the Brain experiment.

- [ ] IXI normal/reference T1 volumes were preprocessed with `IXI_dataset_overview.py`.
- [ ] Saved training arrays are `.npz` files with key `arr`.
- [ ] Training/validation/test patient or slice splits are recorded and reused.
- [ ] Stage 1 checkpoint path is recorded.
- [ ] Stage 2 checkpoint path is recorded.
- [ ] Calibration slices are normal/reference/normal and independent from anomaly evaluation data.
- [ ] `--smoothing-kernel` is identical between calibration and inference.
- [ ] The exact inference CLI command is saved.
- [ ] `calibration_input_files.txt` is retained.
- [ ] `results_v4_zscore.json` is retained.
- [ ] ROC is computed from `Binary_Sum_Heatmap` summed per patient/case.
- [ ] Ground-truth labels are used only for evaluation, not training/calibration model fitting.
- [ ] No patient-identifying information is exposed in public logs, filenames, figures, or W&B runs.

---

## Key differences from the pelvic MRI version

| Aspect | Brain MRI code in this folder | Pelvic version concept |
|---|---|---|
| Anatomy/domain | Brain MRI | Pelvic MRI |
| Data sources | IXI normal/reference training data + fastMRI-style brain evaluation/rendering workflow | LUND-PROBE pelvis workflow |
| Stage 2 positional encoding | 2D RoPE over row/column | 3D RoPE in the pelvic code |
| Codebook size used by training script | 256 per RVQ level | 192 per level in the pelvic setup |
| Main ROC score | patient sum of `Binary_Sum_Heatmap` | may differ by pelvis analysis script/version |
| Primary LPIPS reference | reconstruction-vs-healed/inpainted | earlier descriptions may frame input-vs-healed |
| File format | primarily `.npz` with key `arr` | pelvis code may use `.npy` depending on version |

---

## Practical notes for GitHub readers

- This is research code, not a clinically validated tool.
- Do not use the model output for clinical decisions.
- The code assumes 2D slice-based processing; patient-level evaluation is produced by aggregating slice scores.
- Avoid slice-level train/test leakage. Splits should be patient-level whenever possible.
- Be careful with orientation, flipping, resizing, and annotation coordinate modes when comparing heatmaps to boxes.
- Many defaults are local to the original workstation; override paths explicitly.
- The CORE comments in the Python files are meant to help readers identify exactly what contributes to the reported AUROC.
