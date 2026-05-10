<div align="center">

# 🧠🦴 Brain vs. Pelvic MRI — CORE-Relevant Experiment Differences

*What is shared, what differs, and what matters for AUROC reproduction*

</div>

---

This document summarizes the **CORE-relevant** differences between the two cleaned experiment folders:

<table>
<tr>
<th width="10%">🧠</th>
<td><code>Brain_Experiments/</code> — IXI / fastMRI brain MRI implementation</td>
</tr>
<tr>
<th>🦴</th>
<td><code>Pelvis_Experiments/</code> — LUND-PROBE pelvic MRI implementation</td>
</tr>
</table>

<table>
<tr>
<th width="15%">🟢 CORE</th>
<td>Code and settings that <b>directly affect the primary patient-level ROC/AUROC path.</b></td>
</tr>
<tr>
<th width="15%">🟡 AYNU</th>
<td><b>"Available Yet Not AUROC-interesting"</b> — auxiliary code useful for debugging, visualization, training diagnostics, localization/bounding-box analysis, synthetic-data utilities, or alternative analyses, but <b>not</b> what defines the primary reported AUROC score unless explicitly selected.</td>
</tr>
</table>

> 📌 **Most important shared point:** Both ROC scripts use the same patient-level score:
>
> ```text
> sum_all_bars_score = Σ_slices(Final_Binary_sum_of_anomaly_maps)
> ```
>
> `token_surprisal_hot_px` is written to JSON for audit/ablation only and is **not** added to the primary ROC score in either experiment. `Final_Binary_sum_of_anomaly_maps` already includes ALM-B (binarized token-surprisal) pixels when token fusion is enabled — adding `token_surprisal_hot_px` again would double-count token evidence.

> ℹ️ Unless explicitly noted otherwise, the defaults below refer to the current CLI parser defaults. A few helper signatures still keep legacy fallback values for direct programmatic calls, so the CLI behavior is the authoritative reference for reproduction.

---

## 📑 Table of Contents

<table>
<tr>
<td width="50%" valign="top">

**🎯 Overview**
- [Executive Summary](#-executive-summary)
- [Shared Current ROC Score Definition](#-shared-current-roc-score-definition)
- [Main Code Differences in Files](#-main-code-differences-in-files)

**📁 Data & Models**
- [Data Sources and File Formats](#-data-sources-and-file-formats)
- [Dataset Class and DataModule Design](#-dataset-class-and-datamodule-design)
- [Model Architecture Differences](#-model-architecture-differences)

</td>
<td width="50%" valign="top">

**🔬 Inference & Evaluation**
- [Inference and Heatmap Differences](#-inference-and-heatmap-differences)
- [Binary Mask Fusion Details](#-binary-mask-fusion-details)
- [Default CLI Parameters](#-default-cli-parameters)
- [ROC / AUROC / AUPRC Evaluation](#-roc--auroc--auprc-evaluation)
- [Calibration and Preprocessing](#-calibration-and-preprocessing)
- [Labels, Cohorts, and Patient Aggregation](#-labels-cohorts-and-patient-aggregation)

**📋 Reference**
- [Reproducibility Checklist](#-reproducibility-checklist)
- [What Not to Confuse with the CORE Score](#-what-not-to-confuse-with-the-core-score)
- [Final Takeaway](#-final-takeaway)

</td>
</tr>
</table>

---

## 🎯 Executive Summary

| Topic | 🧠 Brain experiment | 🦴 Pelvis experiment | CORE consequence |
|---|---|---|---|
| Anatomy/domain | Brain MRI (axial T1) | Pelvic MRI | Different data distributions and preprocessing. |
| Normal/training source | IXI + fastMRI normal cases (T1-weighted scans) | LUND-PROBE normal pelvic cases (T2-weighted) | Different MRI cohorts and preparation scripts. |
| Evaluation/anomaly workflow | fastMRI-style brain rendering/annotation workflow | Synthetic and clinical pelvic anomaly cohorts | Different cohort-label logic and folder conventions. |
| Runtime split handling | `Train_framework.py` prefers explicit `train_dir` / `val_dir` inputs and only falls back to a seeded random split when they are omitted | `Train_framework.py` seed-splits `data_dir` into train/val at runtime | Split JSONs are documentation artifacts unless the script is changed to consume them directly. |
| Main ROC score | `sum_all_bars_score` | `sum_all_bars_score` | Same score formula in both experiments. |
| ROC score formula | `Σ_slices(Final_Binary_sum_of_anomaly_maps)` | `Σ_slices(Final_Binary_sum_of_anomaly_maps)` | `token_surprisal_hot_px` is written to JSON for audit/ablation only and is **not** added to the ROC score. |
| Stage 1 codebook size | **256** per RVQ level | **192** per RVQ level | Checkpoints are not interchangeable. |
| Stage 2 positional encoding | **2D RoPE** over row/column only | **3D RoPE** over row/column/slice | Pelvis depends on slice-index conditioning; Brain does not. |
| Main LPIPS reference | LPIPS(Stage 1 reconstruction, healed/inpainted reconstruction) | LPIPS(input image, healed/inpainted input) | The perceptual heatmap branch is not numerically identical. |
| LPIPS backflow (default) | **Disabled** by default (`--binary-include-lpips-backflow` is False) | **Enabled** by default (`--binary-include-lpips-backflow` is True; CLI selector defaults to 99th percentile) | Final binary mask composition differs by default. |
| Post-fusion edge erosion | **Disabled by default** (`--binary-edge-erosion-iters 0`); enable with `--binary-edge-erosion-iters N` | **Disabled by default** (`--binary-edge-erosion-iters 0`); enable with `--binary-edge-erosion-iters N` | Both scripts now share `apply_edge_to_center_erosion()`; disabled by default in both for harmonized inference. |
| Stage 1 decoder output clamping | **Clamped** to `[-3, 3]` in `decode()` | **Not clamped** — raw decoder output returned | Affects reconstruction range; calibration maps built from these outputs reflect different value distributions. |
| Main slice format | `.npz` with key `arr` | `.npy` (float32 direct array) | Dataset loaders and preprocessing differ. |
| Dataset class name | `SliceDataset` / `SliceDataModule` | `NpySliceDataset` / `SliceDataModule` | Different APIs; `NpySliceDataset` only supports `.npy`, `SliceDataset` supports `.npz` and `.png`. |
| DataModule directory layout | Supports separate `train_dir` / `val_dir` **or** single `data_dir` with split | Single `data_dir` only, always splits randomly | Brain can use pre-separated directories; Pelvis always performs a runtime seed-split. |
| DataModule preprocessing pipeline | No in-transform Resize or Crop — images already 256×256 from `IXI_dataset_overview.py` | Resize to 320×320 (area interpolation) → CenterSpatialCrop to 256×256 in every DataModule `setup()` call | Different processing chains; Brain images must be pre-sized correctly; Pelvis accepts any ≥256 input. |
| Slice-level rotation | `np.rot90(k=1)` applied **at preprocessing save time** by `IXI_dataset_overview.py`; DataModule `__getitem__` does **not** rotate | `np.rot90(k=-1)` applied **at load time** in `NpySliceDataset.__getitem__` | Rotation happens at different pipeline stages; do not apply both. |
| Stage 1 internal augmentation breadth | 5 transforms: intensity, contrast, Gaussian noise (prob=0.50, std=0.30), affine (±15°, ±15px bidirectional, zoom 0.8–1.2), horizontal flip (prob=0.50) | 2 transforms: intensity scaling + affine (±5° rotation, ±5px horizontal-only translation) — **no contrast, no noise, no zoom, no flip** | Brain receives much richer augmentation to handle greater acquisition variability. |
| Patient score slice range | All included JSON rows; no slice-index restriction | All included JSON rows; no slice-index restriction in the merged CORE ROC aggregator | Older/auxiliary Pelvis clamped-sum plots use slices 38–49, but `aggregate_patient_sum_of_all_bars(...)` does not. |
| AUPRC reported | **No** (AUROC only) | **Yes** (both AUROC and AUPRC) | Pelvis ROC script reports Precision–Recall AUC additionally. Nonetheless, feel free to create your own AUPRC for Brain :D |

---

## 🔄 Shared Current ROC Score Definition

Both experiments use the same patient-level score in the primary ROC/AUROC scripts:

```text
patient-level score:
sum_all_bars_score = Σ_slices(Final_Binary_sum_of_anomaly_maps)
```

`Final_Binary_sum_of_anomaly_maps` is computed per slice from the final binary anomaly maps (A & B) union:

```text
Final_Binary_sum_of_anomaly_maps = count(ALM-A ∪ ALM-B [∪ LPIPS-backflow if enabled] [after edge erosion if enabled])
```

ALM-A is the thresholded and binarized LPIPS heatmap arm; ALM-B is the thresholded and binarized token-surprisal arm. Their union (and optional LPIPS-backflow refinement) defines the final binary mask whose white-pixel count is summed across slices to give `sum_all_bars_score`. That is the score that is used for ROC-analysis. 

### 🟢 Field meanings

| Field | Meaning | Used in CORE AUROC? |
|---|---|:---:|
| `Final_Binary_sum_of_anomaly_maps` | Count of white pixels in the final binary ALM mask after ALM-A ∪ ALM-B fusion, optional LPIPS-backflow, and optional edge erosion. | ✅ Yes — this is the **only** field used by the CORE ROC-analysis |
| `sum_all_bars_score` | Patient-level sum of `Final_Binary_sum_of_anomaly_maps` across slices. | ✅ Yes — this is the patient-level AUROC ranking field |
| `token_surprisal_hot_px` | Count of pixels where the upsampled token-surprisal map is > 0 after NLL clamp filtering. Written to JSON for audit and ablation. | ❌ **Not** included to the primary CORE ROC-analysis |


### Why this matters

For fair Brain-vs-Pelvis comparison the same score definition must be used:

```text
Brain  ROC score = Σ_slices(Final_Binary_sum_of_anomaly_maps)
Pelvis ROC score = Σ_slices(Final_Binary_sum_of_anomaly_maps)
```

The formula is identical, but the **numerical values** of `Final_Binary_sum_of_anomaly_maps` differ between experiments due to: different LPIPS references (reconstruction-healed vs. input-healed), and if you choose to enable or disable backflow (defaults enabled). Edge erosion is disabled by default in both.

---

## 📊 Main Code Differences in Files

| Category | 🧠 Brain folder | 🦴 Pelvis folder |
|---|---|---|
| Stage 1 file | `Model_Stage_1.py` | `Model_Stage_1.py` |
| Stage 2 file | `Model_Stage_2.py` | `Model_Stage_2.py` |
| Inference file | `Inference_Brain_Experiments.py` | `Inference_Pelvis_Experiments.py` |
| ROC file | `ROC_Curve_Calculations.py` | `ROC_Curves_Calculations.py` |
| Training entry point | `Train_framework.py` | `Train_framework.py` |
| Dataset file | `dataset.py` | `dataset.py` |
| Split manifest | `Train_Val_Test_Exact_DataSplits_IXI_fastMRI.json` | `Train_Val_Test_Exact_DataSplits_LUND_PROBE.json` |
| Reference config | `config_yaml.yaml` | `config_yaml.yaml` |
| Main preprocessing utilities | `IXI_dataset_overview.py`, `Render_patient_slices_from_csv.py`, `collect_normal_slices.py`, label-folder builders | `preslice_volumes.py`, `External_dataset.py`, simulation helper scripts |
| Pinned requirements | `Brain_Experiments_requirements.txt` | `Pelvis_Experiments_requirements.txt` |

---

## 📁 Data Sources and File Formats

### 🧠 Brain experiment

The Brain repository is organized around:

- Normal **IXI T1** NIfTI volumes for training-style slice generation;
- fastMRI-style `.h5` rendering / annotation workflows for evaluation/anomaly folders;
- Saved 2D slices as `.npz` files containing key `arr`.

| Purpose | File |
|---|---|
| IXI NIfTI to training-ready slices | `IXI_dataset_overview.py` |
| fastMRI `.h5` rendering to `.npz` / PNG | `Render_patient_slices_from_csv.py` |
| Normal-slice selection for calibration | `collect_normal_slices.py` |
| Global/local anomaly folder construction | `build_patient_Global_label_folders.py`, `build_patient_Local_label_folders.py` |

Brain preprocessing (performed once by `IXI_dataset_overview.py` before training):

1. Load NIfTI volume and reorient to closest canonical orientation
2. Z-score normalize per volume, clip to `[-3, 3]`
3. Extract axial slice range `128–188` (inclusive)
4. Rotate 90° CCW (`np.rot90(arr, k=1)`) at save time
5. Center-crop-or-pad to `256 × 256`
6. Save as `.npz` with key `"arr"` (float32)

### 🦴 Pelvis experiment

The Pelvis repository is organized around:

- LUND-PROBE-style pelvic MRI normal/reference data;
- `.npy` slice files (direct float32 arrays);
- Filename slice indices using the `_slice_###` convention.

| Purpose | File |
|---|---|
| NIfTI volume to per-slice `.npy` preprocessing | `preslice_volumes.py` |
| External cohort preprocessing/loading utilities | `External_dataset.py` |
| Synthetic global/local anomaly simulations | `Simulation_inference_v4_extended_CJG.py`, `Simluation_inference_v3_support_CJG.py` |

Pelvis preprocessing (performed once by `preslice_volumes.py` before training):

1. Load float32 NIfTI volume
2. Z-score normalize per volume
3. Save each axial slice as `{patient_id}_slice_{idx:03d}.npy` (float32)
4. The **dataset loader** applies `np.rot90(arr, k=-1)` in `__getitem__` at training time
5. The **DataModule** applies Resize(320×320) → CenterSpatialCrop(256×256) during training transforms

### 🟢 CORE implication

| Issue | 🧠 Brain | 🦴 Pelvis | Why it matters |
|---|---|---|---|
| File format | `.npz` with key `"arr"` (also supports `.png` for training) | `.npy` | Dataset loaders and preprocessing assumptions differ. |
| Slice index in filename | Present when available; not required for Brain 2D RoPE | **Required** — used for Pelvis 3D RoPE and per-slice calibration lookup | Pelvis filenames must preserve `_slice_###`. |
| Rotation direction and stage | `np.rot90(k=1)` at **save time** by `IXI_dataset_overview.py`; DataModule `__getitem__` does **not** rotate | `np.rot90(k=-1)` at **load time** in `NpySliceDataset.__getitem__` | Rotation happens at different pipeline stages; do not double-rotate or cross-mix prepared data. |
| Source cohorts | IXI + fastMRI/fastMRI+ brain workflow | LUND-PROBE, synthetic, and clinical pelvic workflow | Cohort labels and patient grouping differ. |

---

## 🗂️ Dataset Class and DataModule Design

Both experiments share the filename `dataset.py` and the class name `SliceDataModule`, but the internal implementations differ significantly.

### Class names

| | 🧠 Brain | 🦴 Pelvis |
|---|---|---|
| Dataset class | `SliceDataset` | `NpySliceDataset` |
| DataModule class | `SliceDataModule` | `SliceDataModule` |

### Supported file formats

| | 🧠 Brain | 🦴 Pelvis |
|---|---|---|
| Training format | `.npz` (key `arr`) and `.png` — configurable via `--file-ext` | `.npy` only — hardcoded |
| File filtering | No `_slice_` filename filter required | Files without `_slice_` in the name are skipped |

### DataModule directory input

| | 🧠 Brain | 🦴 Pelvis |
|---|---|---|
| Directory layout | Supports `train_dir` + `val_dir` (preferred) **or** single `data_dir` with seeded random split | Single `data_dir` only; always performs seeded random split |
| Val split fraction | Configurable (`--val-split`, default 0.10) | Configurable (`val_split=0.10` default) |

### DataModule preprocessing pipeline in `setup()`

| Transform | 🧠 Brain | 🦴 Pelvis |
|---|---|---|
| Rotation in `__getitem__` | **None** — images saved pre-rotated | `np.rot90(arr, k=-1)` applied before transforms |
| EnsureChannelFirstD | ✅ | ✅ |
| Resize | ❌ — images are already 256×256 | ✅ Resize to 320×320 (area interpolation) |
| CenterSpatialCrop | ❌ | ✅ Crop to 256×256 |
| ToTensorD | ✅ | ✅ |

### DataModule augmentation (when enabled at training time)

When `augment=True` is passed to the Brain DataModule, or `--augment` is passed on the CLI:

| Augmentation transform | 🧠 Brain DataModule (`augment=True`) | 🦴 Pelvis DataModule (`augment=True`) |
|---|---|---|
| RandScaleIntensity | ✅ (factor 0.10, prob 0.33) | ❌ |
| RandAdjustContrast | ✅ (gamma 0.5–1.5, prob 0.33) | ❌ |
| RandGaussianNoise | ✅ (prob 0.50, std 0.30) | ❌ |
| RandAffine rotation | ✅ ±15° (prob 0.33) | ✅ ±5° (prob 0.3) |
| RandAffine translation | ✅ ±15px both axes (prob 0.33) | ❌ (not in DataModule augment) |
| RandAffine zoom | ✅ 0.8–1.2 (prob 0.33) | ❌ |
| RandFlip | ✅ horizontal (prob 0.50) | ✅ horizontal (prob 0.50) |

> ⚠️ Note: Both models can also apply augmentations internally in their `training_step` via `self.train_aug` when `use_augmentations=True`. The model-internal augmentations and DataModule augmentations are separate and can stack. The Brain model's internal augmentation pipeline is considerably richer than the Pelvis model's (see Stage 1 architecture section below).

---

## 🏗️ Model Architecture Differences

Both experiments use the same broad two-stage idea:

<table>
<tr>
<th>Stage</th>
<th>Model</th>
<th>Purpose</th>
</tr>
<tr>
<td align="center"><b>1️⃣<br>Stage 1</b></td>
<td><b>RVQ-VAE</b><br><sub>ViT encoder, residual vector quantization, PixelShuffle decoder</sub></td>
<td>Learns a discrete latent representation of normal anatomy appearance</td>
</tr>
<tr>
<td align="center"><b>2️⃣<br>Stage 2</b></td>
<td><b>Factorized bi-directional transformer (Fact-biT / Factorized MaskGIT)</b></td>
<td>Learns token distributions and heals masked/suspect tokens using bidirectional masked prediction</td>
</tr>
</table>

### Stage 1 differences

| Feature | 🧠 Brain | 🦴 Pelvis | CORE relevance |
|---|---|---|---|
| Codebook size | **256** per RVQ level | **192** per RVQ level | Checkpoints and token distributions are not interchangeable. |
| Number of RVQ levels | 2 (L1 structure + L2 texture) | 2 (L1 structure + L2 texture) | Same design. |
| ViT embed_dim | 256 | 256 | Same. |
| ViT patch size | 8 → 32×32 = 1 024 tokens | 8 → 32×32 = 1 024 tokens | Same token grid. |
| ViT encoder depth | 8 layers | 8 layers | Same. |
| ViT encoder heads | 8 | 8 | Same. |
| **Decoder output clamping** | **Clamped to `[-3, 3]`** via `torch.clamp(recon, min=-3.0, max=3.0)` in `decode()` | **Not clamped** — raw decoder output returned from `decode()` | Affects the reconstruction value range; LPIPS calibration maps built from Brain reconstructions will reflect this clipping. |
| BiomedCLIP perceptual weight | **0.5** | **0.9** | Pelvis training is more strongly regularized toward medical visual features. |
| Training LR | **2.0 × 10⁻⁴** | **1.0 × 10⁻⁴** | Brain uses 2× higher learning rate. |
| Adam betas | [0.9, 0.95] | [0.9, 0.95] | Same. |
| Batch size | **192** | **128** | Different GPU memory budgets. |
| **Internal model augmentation** | 5 transforms: intensity (prob=0.33), contrast (gamma 0.5–1.5, prob=0.33), Gaussian noise (prob=**0.50**, std=0.30), affine (prob=0.33, rotation **±15°**, translate **±15px both axes**, zoom **0.8–1.2**), horizontal flip (prob=**0.50**) | 2 transforms: intensity (prob=0.33) + affine (prob=0.33, rotation **±5°**, translate **±5px horizontal only**) — **no contrast, no noise, no zoom, no flip** | Brain receives far richer augmentation; Pelvis model's internal augmentation is minimal. |
| Augmentation sanity check | `sanity_check_aug=True` parameter saves a preview grid on the first training batch | No sanity check parameter | Brain only. |

### Stage 2 differences

| Feature | 🧠 Brain | 🦴 Pelvis | CORE relevance |
|---|---|---|---|
| **Positional encoding** | **2D RoPE** (row, col) — `rope_max_positions=33`, `rope_base=25000` | **3D RoPE** (row, col, slice) — `rope_max_positions=64`, `rope_max_slices=92`, `rope_base=25000` | Major CORE difference — Brain ignores slice position entirely. |
| Slice position conditioning | `slice_pos` parameter accepted but **silently ignored** in `forward()` and all AYNU scoring methods (backward compat only) | `slice_pos` **actively expanded** into attention computation per head at every forward pass | Pelvis strongly depends on correct `_slice_###` filenames to extract slice indices. |
| embed_dim | 256 | 256 | Same. |
| Transformer depth | 8 layers | 8 layers | Same. |
| Transformer heads | 8 | 8 | Same. |
| Normalization | RMSNorm | RMSNorm | Same. |
| FFN | SwiGLU | SwiGLU | Same. |
| Codebook sizes (from Stage 1) | 256 / 256 | 192 / 192 | Must match Stage 1 checkpoint — not interchangeable. |
| Batch size | **158** | **128** | Different memory budget (Brain Stage 2 uses higher batch for seq_len=1024). |
| Training LR | **2.0 × 10⁻⁴** | **1.0 × 10⁻⁴** | Same 2× ratio as Stage 1. |
| Training slice filter | None — all slices in the batch are used | **Slices 30–60 only** — batches outside this range are dropped before Stage 1 encoding via `_filter_training_slices()` | Pelvis focuses training on anatomically relevant pelvic slices. |
| `_extract_slice_indices()` | Not present — slice position is never extracted or used | Static method present; extracts `_slice_###` from filenames and returns a tensor of slice indices | Pelvis-specific infrastructure for 3D RoPE. |
| `_apply_mask()` implementation | **Vectorized** — avoids per-sample Python for-loops to prevent CPU syncs during training | **Python for-loop per sample** — ensures at least one token masked per item but causes GPU–CPU syncs | Different GPU efficiency characteristics. |
| `_apply_block_mask()` implementation | **Fully vectorized tensor operations** — generates all block corners in parallel and unions via `any()` | **Python for-loop per sample** — keeps adding random rectangles until coverage target is met | Brain's block masking is more GPU-friendly. |
| `_masked_ce()` implementation | Masked reduction using `(loss_all * flat_mask.float()).sum() / total_masked.clamp(min=1)` — avoids any `.item()` CPU sync | Falls back to full-sequence CE if no tokens are masked; indexed selection otherwise | Brain's loss computation avoids CPU syncs during training. |

---

## 🔬 Inference and Heatmap Differences

Both experiments use Recursive-AutoMask V4-style inference with token surprisal, LPIPS heatmaps, binary threshold/fusion, and JSON output containing `token_surprisal_hot_px` and `Final_Binary_sum_of_anomaly_maps`.

| Inference component | 🧠 Brain | 🦴 Pelvis |
|---|---|---|
| Main inference script | `Inference_Brain_Experiments.py` | `Inference_Pelvis_Experiments.py` |
| **LPIPS reference (main heatmap)** | **LPIPS(Stage1 reconstruction, healed reconstruction)** | **LPIPS(input image, healed input)** |
| **LPIPS reference (backflow/refinement)** | **LPIPS(Stage1 reconstruction, inpainted reconstruction)** | **LPIPS(input image, inpainted input)** |
| Post-fusion edge erosion | **Disabled by default** (`iters=0`); opt-in via `--binary-edge-erosion-iters` | **Disabled by default** (`iters=0`); opt-in via `--binary-edge-erosion-iters` |
| LPIPS backflow default | **False** (opt-in via `--binary-include-lpips-backflow`) | **True** (opt-out via `--no-binary-include-lpips-backflow`; CLI selector defaults to 99th percentile) |
| LPIPS backflow threshold | Backflow is disabled by default; the vestigial CLI selector default remains **97th percentile** `(97.0, 0.0)` and the direct `run_inference_v4_zscore(...)` helper keeps a legacy `(0.0, 0.585)` fallback | CLI selector default is **99th percentile** `(99.0, 0.0)`; the direct `run_inference_v4_zscore(...)` helper keeps a legacy `(97.0, 0.0)` fallback |
| Binary threshold (ALM-A) | **0.585** (fixed LPIPS value) | **0.60** (fixed LPIPS value) |
| Binary token surprisal threshold (ALM-B) | **5.0** after a **5.0** NLL clamp; with defaults this is equivalent to including `token_surprisal_map > 0` | `> 0` (any non-zero surprisal after clamp) |
| Smoothing kernel | **7** | **15** |
| Heatmap aggregation | **mean** | **geomean** |
| Heal patterns | **"4"** (single 4×4 checkerboard mask) | **"2,3"** (two 2×2 checkerboard masks) |
| Heal steps | 6 | 6 |
| Heal temperature | 0.9 | 0.3 |
| Inpaint temperature | 0.5 | 0.3 |
| Token surprisal samples | **100** Monte Carlo passes | **50** Monte Carlo passes |
| Token mask ratio | **0.80** (80% of L1 tokens masked per pass) | **0.90** (90% of L1 tokens masked per pass) |
| Token surprisal NLL clamp | 5.0 | **8.0** |
| Z-score threshold | **(-2.5, 6.0)** two-sided | 2.0 (one-sided) |
| Num iterations | 1 | 1 |
| Inter-iteration dilation | **1** | **5** |

> 💡 Because the LPIPS reference differs, `Final_Binary_sum_of_anomaly_maps` is **not numerically identical** between experiments even though it plays the same role in the final score formula. Additionally, the LPIPS backflow being enabled by default in Pelvis but disabled by default in Brain means the binary mask composition is structurally different.

---

## 🔀 Binary Mask Fusion Details

Both experiments combine binary anomaly arms into a final mask using boolean union. This final mask is what is counted in `Final_Binary_sum_of_anomaly_maps`.

### 🧠 Brain

```text
ALM-A = selected masked_LPIPS_score > 0.585
        default selected iteration is 0, so the heatmap is LPIPS(recon, healed)
        where recon = Stage 1 reconstruction (clamped to [-3, 3])
ALM-B = token_surprisal_NLL > 5.0 after the token_surprisal_clamp=5.0 filter
LPIPS-backflow = thresholded LPIPS(recon, inpainted)
                 only if --binary-include-lpips-backflow is enabled (disabled by default)

pre-union = ALM-A ∪ ALM-B  [∪ LPIPS-backflow if enabled]

Final_Binary_sum_of_anomaly_maps = apply_edge_to_center_erosion(pre-union)
                                   default: iters=0 (disabled); enable with --binary-edge-erosion-iters N
```

### 🦴 Pelvis

```text
ALM-A = selected masked_LPIPS_score > 0.60
        default selected iteration is 0, so the heatmap is LPIPS(input, healed)
        (input image — raw, not Stage 1 reconstruction)
ALM-B = token_surprisal_NLL > 0  (any non-zero surprisal after NLL clamp; default clamp=8.0)
LPIPS-backflow = LPIPS(input, inpainted) at 99th percentile  (enabled by default)

pre-union = ALM-A ∪ ALM-B ∪ LPIPS-backflow

Final_Binary_sum_of_anomaly_maps = apply_edge_to_center_erosion(pre-union)
                                   default: iters=0 (disabled); enable with --binary-edge-erosion-iters N
```

### `token_surprisal_hot_px`

Both experiments define this identically:
```text
token_surprisal_hot_px = count of pixels where token_surprisal_map > 0
```
(after the NLL clamp filter is applied; the default clamp is 5.0 in Brain and 8.0 in Pelvis)

This field is written to JSON for audit and ablation only. It is also represented inside `Final_Binary_sum_of_anomaly_maps` through the ALM-B arm, and is **not** added separately to `sum_all_bars_score`.

---

## ⚙️ Default CLI Parameters

These are the **argparse defaults** as coded in each inference script — the authoritative source for what is used when no CLI override is provided.

| Parameter | 🧠 Brain default | 🦴 Pelvis default |
|---|---|---|
| `--binary-threshold` | **0.60** | **0.60** |
| `--binary-include-lpips-backflow` | **False** | **True** |
| `--smoothing-kernel` | **7** | **15** |
| `--heatmap-aggregation` | **mean** | **geomean** |
| `--heal-patterns` | **"4"** (4×4 checkerboard) | **"2,3"** (2×2 checkerboard pair) |
| `--heal-steps` | 6 | 6 |
| `--heal-temperature` | **0.9** | **0.3** |
| `--inpaint-temperature` | **0.5** | **0.3** |
| `--token-surprisal-samples` | **100** | **50** |
| `--token-surprisal-mask-ratio` | **0.80** | **0.90** |
| `--token-surprisal-clamp` | 5.0 | **8.0** |
| `--binary-token-surprisal-threshold` | **5.0** | N/A (uses `> 0` after clamp) |
| `--binary-edge-erosion-iters` | **0** (disabled by default) | **0** (disabled by default) |
| `--binary-edge-erosion-kernel` | **13** | **13** |
| `--binary-center-protect-radius-ratio` | **0.35** | **0.35** |
| `--z-threshold` | **"(-2.5 , 6.0)"** (two-sided) | **2.0** (one-sided) |
| `--num-iterations` | 1 | 1 |
| `--inter-iteration-dilation` | **1** | **5** |
| `--binary-mask-iteration` | 0 | 0 |
| `--batch-size` | **320** | **320** |
| `--device` | cuda:1 | cuda:1 |

> ⚠️ **Smoothing kernel must match between calibration and inference** within the same experiment. The value used when building the calibration `.npz` must equal the value used at inference, or Z-score normalization will be applied with mismatched spatial smoothing.

---

## 📈 ROC / AUROC / AUPRC Evaluation

### 🟢 Shared scoring rule

| Item | 🧠 Brain | 🦴 Pelvis |
|---|---|---|
| Per-slice field used | `Final_Binary_sum_of_anomaly_maps` | `Final_Binary_sum_of_anomaly_maps` |
| Patient score | `sum_all_bars_score` | `sum_all_bars_score` |
| Formula | `Σ_slices(Final_Binary_sum_of_anomaly_maps)` | `Σ_slices(Final_Binary_sum_of_anomaly_maps)` |
| `token_surprisal_hot_px` role | Written to JSON for audit/ablation; **not** added to CORE ROC score | Written to JSON for audit/ablation; **not** added to CORE ROC score |

### ROC script behavior

| Feature | 🧠 Brain | 🦴 Pelvis |
|---|---|---|
| ROC script | `ROC_Curve_Calculations.py` | `ROC_Curves_Calculations.py` |
| Main aggregation function | `aggregate_fastmri_binary_token_patient_scores(...)` | `aggregate_patient_sum_of_all_bars(...)` |
| Main ROC function | `compute_fastmri_roc_and_auc(...)` | `compute_patient_roc_and_auc(...)` |
| Main patient-score key | `sum_all_bars_score` | `sum_all_bars_score` |
| **AUPRC reported** | **No** — AUROC only | **Yes** — both AUROC and AUPRC (`compute_auprc_step()` using right-step PR integration) |
| **Normal label logic** | Patient ID contains `"test_samples_fastmri"` → label = 0 (normal) | Patient has `is_orig = True` (identifier contains `"orig"`) → label = 0 (normal) |
| Anomaly label logic | All other patients → label = 1 | All other patients → label = 1 |
| **Patient score slice range** | All included rows in the JSON | All included rows in the JSON |

> ⚠️ The current Pelvis merged ROC path does **not** apply a hard-coded 38–49 slice restriction. Functions later in `ROC_Curves_Calculations.py` do use slices 38–49 for older/auxiliary clamped-sum plots, but those are not the `aggregate_patient_sum_of_all_bars(...)` CORE ROC aggregator.

---

## 🎛️ Calibration and Preprocessing

| Topic | 🧠 Brain | 🦴 Pelvis | CORE relevance |
|---|---|---|---|
| LPIPS calibration reference | LPIPS(Stage 1 reconstruction, healed reconstruction) | LPIPS(input, healed input) | Different heatmap semantics — calibration maps are not interchangeable. |
| Reconstruction range going into calibration | Clamped to `[-3, 3]` by Stage 1 decoder | Unclamped decoder output | Affects the dynamic range of LPIPS values in the calibration map. |
| Smoothing kernel (argparse default) | **7** | **15** | Must match between calibration and inference. |
| Heal patterns (argparse default) | **"4"** (4×4 checkerboard) | **"2,3"** (two 2×2 checkerboards) | Must match between calibration and inference. |
| Token-surprisal samples (argparse default) | **100** | **50** | Affects token surprisal stability/counts. |
| Token mask ratio (argparse default) | **0.80** | **0.90** | Affects token-surprisal branch. |
| Heatmap aggregation (argparse default) | **mean** | **geomean** | Aggregation affects `Final_Binary_sum_of_anomaly_maps`. |
| Per-slice Z-score calibration | Supported (`use_per_slice_stats=true`, ≥3 samples required per slice index) | Supported (`use_per_slice_stats=true`, ≥3 samples required per slice index) | Both support anatomically varying calibration maps. |

---

## 🏷️ Labels, Cohorts, and Patient Aggregation

### 🧠 Brain — fastMRI-style label logic

- Normal (label = 0): patient ID matches `"test_samples_fastmri"` substring
- Anomaly (label = 1): all other patients
- Validation cohort: excluded from ROC by default (`include_in_roc = False`)
- All slices contribute to the patient-level score (no slice-range restriction)
- Anomaly categories include both global (Motion artifact, Possible artifact, Colpocephaly, Extra-axial collection, White matter changes) and local/focal findings (Edema, Enlarged ventricles, Mass, Craniotomy, Nonspecific lesion, Resection cavity, etc.)

### 🦴 Pelvis — LUND-PROBE label logic

- Normal (label = 0): patient has `is_orig = True` identifier (contains `"orig"`)
- Anomaly (label = 1): all other patients
- **Slice range restriction**: no hard-coded slice restriction in the current merged CORE ROC aggregator; all included JSON rows contribute to `sum_all_bars_score`
- Anomaly categories include both synthetic (`RandomGhosting`, `RandomNoise`, `RandomSpike`, `RandomMotion`, `WholeImageGaussian`) and clinical (`ClinicalVariations`, `Spacer`, `Unknown`, `Stor_T2_till_sCT`) cohorts
- Stage 2 training slice filter (slices 30–60) is applied during training only, not during inference or evaluation

---

## 🖼️ Default Visualization Output

Both experiments output `_Final_ALM_Arithmetic.png` as the **default qualitative figure** per slice when visualizations are enabled. This figure shows:

- **Top row**: independently normalized ALM-A (masked LPIPS), ALM-B (token surprisal), LPIPS-backflow (if available), and their visual arithmetic sum overlaid on the input image within the final binary-mask support.
- **Bottom row**: final binary map (`Final_Binary_sum_of_anomaly_maps` support), binarized ALM-A, binarized ALM-B, and binarized backflow.

> ⚠️ The Brain `_Anomaly_Overlay.png` (when enabled via `--include-full-analysis-figure`) has **7 panels** while the Pelvis version has **6 panels**. The 7th Brain panel was originally an erosion visualization; since erosion is disabled by default in both experiments, this panel will show an unchanged mask when erosion is not enabled.

This arithmetic figure is qualitative only. The ROC scripts do not read the PNG or the continuous arithmetic overlay; they read JSON fields.

---

## ✅ Reproducibility Checklist

### Shared checks

- [ ] Confirm both ROC scripts use the correct patient-level score: `sum_all_bars_score = Σ_slices(Final_Binary_sum_of_anomaly_maps)`
- [ ] Confirm `token_surprisal_hot_px` is retained in JSON for audit/ablation only and is **not** added separately to `sum_all_bars_score`
- [ ] Save the exact inference CLI command (especially `--binary-threshold`, `--heal-patterns`, `--smoothing-kernel`, `--token-surprisal-samples`, `--binary-include-lpips-backflow`)
- [ ] Save the exact ROC CLI command
- [ ] Keep the generated `results_v4_zscore.json` files
- [ ] Keep calibration `.npz` files and calibration input lists
- [ ] Keep Stage 1 and Stage 2 checkpoint paths / hashes
- [ ] Preserve patient-level train/validation/test split manifests
- [ ] Verify no patient/case leakage between training, calibration, validation, and test/anomaly cohorts
- [ ] Verify patient/case identifiers before ROC label assignment
- [ ] Confirm smoothing kernel matches between calibration and inference
- [ ] Confirm heal patterns match between calibration and inference
- [ ] Do not expose patient-identifying information in public logs, plots, filenames, W&B, or shared outputs

### 🧠 Brain-specific checks

- [ ] IXI T1 preprocessing settings are recorded (slice range 128–188, rot90 k=1 at save time, clip [-3, 3])
- [ ] fastMRI `.h5` rendering settings are recorded
- [ ] `.npz` files contain key `"arr"` and images are pre-sized to 256×256 (no resize/crop in DataModule)
- [ ] Verify DataModule does NOT apply rotation (rotation already baked into saved `.npz` files)
- [ ] Stage 1 decoder clamping to [-3, 3] is noted — calibration maps reflect this range
- [ ] Edge erosion status recorded: disabled by default (`iters=0`); if enabled, record `iters`, `kernel`, and `center_protect` values used
- [ ] LPIPS backflow status recorded (default: **disabled**)
- [ ] `--binary-threshold 0.585`, `--smoothing-kernel 7`, `--heal-patterns 4`, `--token-surprisal-samples 100`, `--token-surprisal-mask-ratio 0.80` confirmed
- [ ] Validation normals are included/excluded intentionally from ROC

### 🦴 Pelvis-specific checks

- [ ] `.npy` filenames preserve `_slice_###` indices (required for 3D RoPE and per-slice calibration lookup)
- [ ] `np.rot90(k=-1)` applied in `NpySliceDataset.__getitem__` — do not double-rotate data
- [ ] DataModule applies Resize(320×320) → CenterSpatialCrop(256×256) — verify input NIfTI slices have sufficient resolution
- [ ] Stage 1 decoder output is NOT clamped — calibration maps may have a wider effective range
- [ ] Confirm whether the intended Pelvis ROC should use all included JSON rows (current code) or a restricted slice range; rerun if the intended analysis changes
- [ ] `is_orig` / normal identifiers are correct before ROC label assignment
- [ ] LPIPS backflow status recorded (default: **enabled** at 99th percentile)
- [ ] Edge erosion status recorded: disabled by default (`iters=0`); if enabled, record `iters`, `kernel`, and `center_protect` values used
- [ ] `--binary-threshold 0.60`, `--smoothing-kernel 15`, `--heal-patterns 2,3`, `--token-surprisal-samples 50`, `--token-surprisal-mask-ratio 0.90` confirmed
- [ ] Stage 2 training slice filter (30–60) is noted for reproducibility context
- [ ] Synthetic and clinical cohorts are not accidentally mixed unless intended
- [ ] AUROC and AUPRC outputs are interpreted together when using the Pelvis merged ROC workflow

---

## ⚠️ What Not to Confuse with the CORE Score

The code in both folders contains many useful diagnostic and auxiliary outputs. These are important for debugging and scientific interpretation, but they should not be reported as the primary AUROC score unless a separate analysis explicitly selects them.

<details>
<summary><b>Click to expand — 🟡 AYNU outputs that are NOT the primary CORE AUROC score</b></summary>

| Auxiliary item | Why it is not the primary CORE score |
|---|---|
| `clamped_pixel_sum` | Useful LPIPS-derived diagnostic, but not the patient-level ROC score in the cleaned CORE path. |
| `lpips_input_recon_sum_mask` | Reconstruction diagnostic / auxiliary analysis field. |
| Sharpness scores / artifact flags | Useful for quality control and artifact analysis, not the main AUROC score. |
| Bounding-box precision/F1/inside-ratio metrics | Localization evaluation (Brain only); labels/boxes are not used to train the model and do not define patient-level AUROC. |
| Per-patient bar plots | Visual summaries of intermediate quantities; they do not by themselves define the ROC score unless they use `sum_all_bars_score`. |
| `compute_anomaly_map()` / `compute_anomaly_map_sliding()` / `compute_anomaly_map_contextual()` / `compute_anomaly_map_iterative()` | Alternative Stage 2 scoring methods in the AYNU section; preserved for transparency/ablation but not the primary cleaned AUROC path. |
| `_Final_ALM_Heatmap.png` figures | AYNU qualitative output (requires `--save-alm-heatmap-png`); the default qualitative output is `_Final_ALM_Arithmetic.png`. |
| Synthetic anomaly generation utilities (Pelvis) | Support cohort generation and experiments; not part of the ROC score calculation itself. |

</details>

---

## 📌 Final Takeaway

The most important CORE distinction is not the ROC score formula — that is **the same** in Brain and Pelvis:

```text
sum_all_bars_score = Σ_slices(Final_Binary_sum_of_anomaly_maps)
```

`Final_Binary_sum_of_anomaly_maps` is the fused final ALM binary map (ALM-A ∪ ALM-B, optional LPIPS-backflow, optional edge erosion — disabled by default in both). `token_surprisal_hot_px` is written to JSON for audit/ablation and is **not** added separately to the patient-level score.

The important differences are:

<table>
<tr>
<td>1️⃣</td>
<td>Anatomy and data sources (IXI/fastMRI brain vs. LUND-PROBE pelvic)</td>
</tr>
<tr>
<td>2️⃣</td>
<td>File formats (<code>.npz / "arr"</code> Brain vs. <code>.npy</code> Pelvis) and rotation stage (save-time Brain vs. load-time Pelvis)</td>
</tr>
<tr>
<td>3️⃣</td>
<td>DataModule preprocessing: <b>🧠 Brain — no resize/crop (images pre-sized 256×256)</b> vs. <b>🦴 Pelvis — Resize(320×320) → CenterSpatialCrop(256×256)</b></td>
</tr>
<tr>
<td>4️⃣</td>
<td>Stage 1 codebook size: <b>🧠 Brain 256</b> vs. <b>🦴 Pelvis 192</b></td>
</tr>
<tr>
<td>5️⃣</td>
<td>Stage 1 decoder output: <b>🧠 Brain clamped to [-3, 3]</b> vs. <b>🦴 Pelvis unclamped</b></td>
</tr>
<tr>
<td>6️⃣</td>
<td>Stage 1 internal model augmentation: <b>🧠 Brain — 5 transforms (intensity, contrast, noise, affine ±15°/±15px/zoom, flip)</b> vs. <b>🦴 Pelvis — 2 transforms (intensity + affine ±5°/±5px horizontal-only, no flip/zoom/noise/contrast)</b></td>
</tr>
<tr>
<td>7️⃣</td>
<td>Stage 2 positional encoding: <b>🧠 Brain 2D RoPE (row, col only)</b> vs. <b>🦴 Pelvis 3D RoPE (row, col, slice)</b></td>
</tr>
<tr>
<td>8️⃣</td>
<td>LPIPS reference: <b>🧠 Brain uses Stage 1 reconstruction as reference</b> vs. <b>🦴 Pelvis uses the raw input image as reference</b></td>
</tr>
<tr>
<td>9️⃣</td>
<td>LPIPS backflow: <b>🧠 Brain disabled by default</b> vs. <b>🦴 Pelvis enabled by default at 99th percentile</b></td>
</tr>
<tr>
<td>🔟</td>
<td>Post-fusion edge erosion: <b>disabled by default in both experiments</b> (`--binary-edge-erosion-iters 0`); `apply_edge_to_center_erosion()` is available in both inference scripts and can be enabled identically</td>
</tr>
<tr>
<td>1️⃣1️⃣</td>
<td>Default CLI parameters differ: binary threshold (0.60 vs. 0.60), smoothing kernel (7 vs. 15), heal patterns ("4" vs. "2,3"), heal/inpaint temperature (0.9/0.5 vs. 0.3/0.3), token surprisal samples (100 vs. 50), token mask ratio (0.80 vs. 0.90), heatmap aggregation (mean vs. geomean), inter-iteration dilation (1 vs. 5)</td>
</tr>
<tr>
<td>1️⃣2️⃣</td>
<td>ROC evaluation: <b>🧠 Brain reports AUROC only</b> vs. <b>🦴 Pelvis reports both AUROC and AUPRC</b></td>
</tr>
</table>

> ⚠️ When reporting AUROC, confirm both experiments use the same **sum-all-bars** patient score (`Σ_slices(Final_Binary_sum_of_anomaly_maps)`). Document whether LPIPS backflow was enabled/disabled and whether edge erosion was applied, because these directly affect the `Final_Binary_sum_of_anomaly_maps` count. 

---

<div align="center">

### That's it, have fun!!

*Research code for unsupervised medical image anomaly detection and analysis using deep learning.*

</div>
