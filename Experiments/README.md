<div align="center">

# 🔬 CORE-Relevant Differences Between Brain & Pelvis Experiments

[![Brain MRI](https://img.shields.io/badge/🧠_Brain-IXI_/_fastMRI-blue.svg)]()
[![Pelvis MRI](https://img.shields.io/badge/🦴_Pelvis-LUND--PROBE/Clinical-purple.svg)]()
[![CORE](https://img.shields.io/badge/Focus-CORE_AUROC_Calculations-green.svg)]()
</div>

<div align="left">
A comparative reference for the cleaned Brain and Pelvis anomaly-detection experiments, focused on what directly affects the primary patient-level ROC/AUROC path.

</div>

---

## 🎯 Scope of This Document

This document summarizes the **CORE-relevant** differences between the cleaned Brain and Pelvis experiments.

<table>
<tr>
<th width="15%">🟢 CORE</th>
<td>Code and settings that <b>directly affect</b> the primary patient-level ROC/AUROC calculations.</td>
</tr>
<tr>
<th width="15%">🟡 AYNU</th>
<td>Auxiliary code useful for debugging, visualization, training diagnostics, localization/bounding-box analysis, synthetic-data utilities, or alternative analyses — but <b>does not define</b> the primary reported ROC/AUROC score unless explicitly selected.</td>
</tr>
</table>

> 🚨 **Most important shared point:** both experiments should use the same patient-level ROC score definition:
>
> ```text
> sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
> ```
>
> This means ROC/AUROC must use the sum of **both** the token-surprisal arm **and** the binary/perceptual heatmap arm — not only one of them.

---

## 📑 Table of Contents

<table>
<tr>
<td width="50%" valign="top">

**🎯 Core Concepts**
- [1. Executive Summary](#1--executive-summary)
- [2. Shared CORE AUROC Definition](#2--shared-core-auroc-definition)
- [3. Main Differences at a Glance](#3--main-differences-at-a-glance)

**📊 Data & Architecture**
- [4. Data Sources and File Formats](#4--data-sources-and-file-formats)
- [5. Model Architecture Differences](#5--model-architecture-differences)
- [6. Inference and Heatmap Differences](#6--inference-and-heatmap-differences)

</td>
<td width="50%" valign="top">

**📈 Evaluation & Reproducibility**
- [7. ROC / AUROC / AUPRC Evaluation](#7--roc--auroc--auprc-evaluation-differences)
- [8. Calibration and Preprocessing](#8--calibration-and-preprocessing-differences)
- [9. Labels, Cohorts, and Aggregation](#9--labels-cohorts-and-patient-aggregation)

**✅ Reference**
- [10. Reproducibility Checklist](#10--reproducibility-checklist)
- [11. What NOT to Confuse with CORE](#11--what-not-to-confuse-with-the-core-score)
- [🏁 Final Takeaway](#-final-takeaway)

</td>
</tr>
</table>

---

## 1. 📋 Executive Summary

| Topic | 🧠 Brain Experiment | 🦴 Pelvis Experiment | 🎯 CORE Consequence |
|---|---|---|---|
| **Anatomy/domain** | Brain MRI | Pelvic MRI | Different data distributions and preprocessing assumptions |
| **Main training source** | IXI T1 NIfTI-derived slices | LUND-PROBE-style pelvic slices | Different source cohorts and preparation scripts |
| **Evaluation/anomaly workflow** | fastMRI-style brain rendering/annotation | Synthetic + clinical pelvic anomaly cohorts | Different cohort-label logic and folder conventions |
| **Main ROC score** | `sum_all_bars_score` | `sum_all_bars_score` | ✅ **Same intended score formula** |
| **ROC score formula** | `Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)` | `Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)` | Both arms must be included |
| **Stage 1 codebook size** | 256 per RVQ level | 192 per RVQ level | ⚠️ Checkpoints are **not** interchangeable |
| **Stage 2 positional encoding** | 2D RoPE (row/column) | 3D RoPE (row/column/slice) | Pelvis depends on slice-index conditioning |
| **Main LPIPS reference** | Recon-vs-healed / recon-vs-inpainted | Input-vs-healed / input-vs-inpainted | Perceptual heatmap branches differ |
| **Main slice format** | Primarily `.npz` (key `arr`); inference reads `.npy` too | `.npy` with `_slice_###` filename convention | Different preprocessing assumptions |
| **ROC outputs** | ROC/AUROC-focused fastMRI brain outputs | ROC/AUROC **plus** PR/AUPRC | Pelvis reports PR/AUPRC in main workflow |

---

## 2. 🎯 Shared CORE AUROC Definition

Both cleaned experiments use the **same intended patient-level score** for primary ROC/AUROC evaluation:

```text
per-slice score contribution = token_surprisal_hot_px + Binary_Sum_Heatmap

per-patient score:
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

### 📊 Field Meanings

| Field | Meaning | CORE? |
|---|---|:---:|
| `token_surprisal_hot_px` | Count of hot pixels from the token-surprisal branch | 🟢 Yes |
| `Binary_Sum_Heatmap` | Count of binary-positive pixels from the perceptual/binary heatmap branch after experiment-specific fusion/thresholding | 🟢 Yes |
| `sum_all_bars_score` | Patient-level sum of the two per-slice quantities above | 🟢 Yes |
| `binary_token_score` | In Brain code, retained as a backward-compatible **alias** for the corrected combined score | ⚠️ Do not treat as a separate score |

### 💡 Why This Matters

The CORE score is **not** only the perceptual/binary heatmap side, and is **not** only the token-surprisal side. For fair Brain-vs-Pelvis comparison, both arms must contribute identically at patient aggregation:

```text
🧠 Brain  ROC score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
🦴 Pelvis ROC score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

---

## 3. 🔍 Main Differences at a Glance

| Category | 🧠 Brain Folder | 🦴 Pelvis Folder |
|---|---|---|
| **Stage 1 file** | `Model_Stage1.py` | `Model_Stage_1.py` |
| **Stage 2 file** | `Model_Stage_2.py` | `Model_Stage_2.py` |
| **Inference file** | `Inference_Brain_Experiments.py` | `Inference_Pelvis_Experiments.py` |
| **ROC file** | `ROC_Curve_Calculations.py` | `ROC_Curves_Calculations.py` |
| **Training entry point** | `Train_framework.py` | `Train_framework.py` |
| **Dataset file** | `dataset.py` | `dataset.py` |
| **Split manifest** | `Train_Val_Test_Exact_DataSplits_IXI_fastMRI.json` | `Train_Val_Test_Exact_DataSplits_LUND_PROBE.json` |
| **Reference config** | `config_yaml.yaml` | `config_yaml.yaml` |
| **Preprocessing utilities** | `IXI_dataset_overview.py`, `Render_patient_slices_from_csv.py`, `collect_normal_slices.py`, label-folder builders | `preslice_volumes.py`, `External_dataset.py`, simulation helpers |

---

## 4. 📦 Data Sources and File Formats

### 4.1 🧠 Brain Experiment

The Brain repository is organized around:

- 🔹 normal/reference **IXI T1** NIfTI volumes for training-style slice generation
- 🔹 **fastMRI-style** `.h5` rendering / annotation workflows for evaluation/anomaly folders
- 🔹 Saved 2D slices, primarily as **`.npz`** files containing key `arr`

**Relevant Brain files:**

| Purpose | File |
|---|---|
| IXI NIfTI → training-ready slices | `IXI_dataset_overview.py` |
| fastMRI `.h5` rendering → `.npz` / PNG | `Render_patient_slices_from_csv.py` |
| Normal-slice selection for calibration | `collect_normal_slices.py` |
| Global/local anomaly folder construction | `build_patient_Global_label_folders.py`, `build_patient_Local_label_folders.py` |

<details>
<summary><b>📋 Brain Preprocessing Details</b></summary>

- NIfTI loading and canonical reorientation for IXI
- Per-volume z-score normalization
- Clipping commonly to `[-3, 3]`
- Crop/pad to `256 × 256`
- `.npz` output with key `arr`
- fastMRI anomaly/evaluation slices rendered from `.h5` `reconstruction_rss` volumes

</details>

### 4.2 🦴 Pelvis Experiment

The Pelvis repository is organized around:

- 🔹 **LUND-PROBE-style** pelvic MRI normal/reference data
- 🔹 **`.npy`** slice files
- 🔹 Filename slice indices using the `_slice_###` convention

**Relevant Pelvis files:**

| Purpose | File |
|---|---|
| NIfTI volume → per-slice `.npy` preprocessing | `preslice_volumes.py` |
| External cohort preprocessing/loading utilities | `External_dataset.py` |
| Synthetic anomaly support | `Simulation_inference_v4_extended_CJG.py`, `Simluation_inference_v3_support_CJG.py` |

<details>
<summary><b>📋 Pelvis Preprocessing Details</b></summary>

- Loading `float32` `.npy` slices
- Rotation with `np.rot90(arr, k=-1)` in `dataset.py`
- Resize to `320 × 320` and center crop to `256 × 256`
- Saved slice naming such as `{patient_id}_slice_{idx:03d}.npy`

</details>

### 4.3 🎯 CORE Implication

| Issue | 🧠 Brain | 🦴 Pelvis | Why It Matters |
|---|---|---|---|
| **File format** | Mostly `.npz` (key `arr`); inference reads `.npy` too | `.npy` | Dataset loaders and preprocessing assumptions differ |
| **Slice index** | Present in filenames; not used for 2D RoPE | Required for 3D RoPE and per-slice calibration lookup | Pelvis filenames **must** preserve `_slice_###` |
| **Orientation/rotation** | IXI and fastMRI have their own orientation steps | Dataset loader rotates with `np.rot90(arr, k=-1)` | ⚠️ Do not mix prepared data without checking orientation |
| **Source cohorts** | IXI + fastMRI-style brain workflow | LUND-PROBE-style pelvic workflow | Cohort labels and patient grouping differ |

---

## 5. 🏗️ Model Architecture Differences

Both experiments share the same broad two-stage idea:

<table>
<tr>
<td align="center" width="15%"><b>1️⃣<br>Stage 1</b></td>
<td><b>RVQ-VAE</b> — learns a discrete latent representation</td>
</tr>
<tr>
<td align="center"><b>2️⃣<br>Stage 2</b></td>
<td><b>Factorized MaskGIT / Fact-biT</b> — predicts/heals masked tokens</td>
</tr>
</table>

However, several **CORE-relevant** implementation details differ.

### 5.1 🧩 Stage 1 Differences

| Feature | 🧠 Brain | 🦴 Pelvis | CORE Relevance |
|---|---|---|---|
| **Input** | `1 × 256 × 256` brain slice | `1 × 256 × 256` pelvic slice | Same spatial target, different anatomy/preprocessing |
| **Patch size** | 8 | 8 | Both typically produce `32 × 32 = 1024` tokens |
| **Encoder** | ViT-style, depth 8, 8 heads | ViT-style, depth 8, 8 heads | Broadly similar |
| **RVQ levels** | 2 | 2 | Broadly similar |
| **Codebook size** | **256** per RVQ level | **192** per RVQ level | ⚠️ Checkpoints/token distributions **not** interchangeable |
| **Training LR** | `2e-4` | `1e-4` | Training dynamics differ |
| **BiomedCLIP perceptual weight** | 0.5 | 0.9 | Training objective differs |

### 5.2 🧩 Stage 2 Differences

| Feature | 🧠 Brain | 🦴 Pelvis | CORE Relevance |
|---|---|---|---|
| **Token streams** | L1/L2 | L1/L2 | Similar broad design |
| **Codebook size (from Stage 1)** | 256 per level | 192 per level | Stage 2 must match its Stage 1 checkpoint |
| **Positional encoding** | **2D RoPE** (row, column) | **3D RoPE** (row, column, slice) | ⭐ **Major CORE difference** |
| **Slice position conditioning** | `slice_pos` accepted in signatures but not part of 2D RoPE | Slice index used for anatomical position encoding | Pelvis depends on correct slice-index filenames |
| **Stage 2 loss** | `CE(masked L1) + 0.25 × CE(masked L2)` | Same documented loss form | Broadly similar training target |
| **Label smoothing** | 0.05 | 0.05 | Similar |

---

## 6. 🔬 Inference and Heatmap Differences

Both experiments use **Recursive-AutoMask V4**-style inference with:

<table>
<tr>
<td>✅ Stage 1 reconstruction/tokens</td>
<td>✅ Stage 2 token healing/inpainting</td>
</tr>
<tr>
<td>✅ Token surprisal</td>
<td>✅ LPIPS heatmaps</td>
</tr>
<tr>
<td>✅ Binary threshold/fusion</td>
<td>✅ JSON output with <code>token_surprisal_hot_px</code> and <code>Binary_Sum_Heatmap</code></td>
</tr>
</table>

The **main CORE difference** is the LPIPS reference used by the primary heatmap branch.

### 🔄 Inference Component Comparison

| Inference Component | 🧠 Brain | 🦴 Pelvis |
|---|---|---|
| **Main inference script** | `Inference_Brain_Experiments.py` | `Inference_Pelvis_Experiments.py` |
| **LPIPS calibration reference** | `LPIPS(Stage 1 recon, healed recon)` | `LPIPS(input, healed)` |
| **LPIPS iteration 0 reference** | `LPIPS(Stage 1 recon, healed recon)` | `LPIPS(input, healed)` |
| **Refinement/inpainting LPIPS** | `LPIPS(Stage 1 recon, inpainted recon)` | `LPIPS(input, inpainted)` |
| **Token surprisal branch** | Monte Carlo token masking / Stage 2 prediction | Monte Carlo token masking / Stage 2 prediction |
| **Per-slice token output** | `token_surprisal_hot_px` | `token_surprisal_hot_px` |
| **Per-slice binary output** | `Binary_Sum_Heatmap` | `Binary_Sum_Heatmap` |

### ⚠️ Important Interpretation

> Because the LPIPS reference differs, the **meaning** of `Binary_Sum_Heatmap` is not numerically identical between experiments, even though it plays the same role in the final score formula.

<table>
<tr>
<td width="50%" valign="top">

**🧠 In Brain**
`Binary_Sum_Heatmap` is driven by a **reconstruction-referenced** perceptual comparison.

</td>
<td width="50%" valign="top">

**🦴 In Pelvis**
`Binary_Sum_Heatmap` is driven by an **input-referenced** perceptual comparison.

</td>
</tr>
</table>

But in both cases, the final patient score is still:

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

---

## 7. 📈 ROC / AUROC / AUPRC Evaluation Differences

### 7.1 🟢 Shared Scoring Rule

| Item | 🧠 Brain | 🦴 Pelvis |
|---|---|---|
| **Per-slice fields** | `token_surprisal_hot_px`, `Binary_Sum_Heatmap` | `token_surprisal_hot_px`, `Binary_Sum_Heatmap` |
| **Patient score** | `sum_all_bars_score` | `sum_all_bars_score` |
| **Formula** | `Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)` | `Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)` |

### 7.2 ⚙️ ROC Script Behavior

| Feature | 🧠 Brain | 🦴 Pelvis |
|---|---|---|
| **ROC script** | `ROC_Curve_Calculations.py` | `ROC_Curves_Calculations.py` |
| **Main aggregation function** | `aggregate_fastmri_binary_token_patient_scores(...)` | `aggregate_patient_sum_of_all_bars(...)` |
| **Main ROC function** | `compute_fastmri_roc_and_auc(...)` | `compute_patient_roc_and_auc(...)` |
| **Main patient-score key** | `sum_all_bars_score` | `sum_all_bars_score` |
| **Backward-compatible alias** | `binary_token_score` aliases combined score | Not needed as primary name |
| **PR/AUPRC** | ROC/AUROC-focused CORE path | ✅ Explicitly computes ROC/AUROC **and** PR/AUPRC |
| **Normal-label policy** | Included test-normal fastMRI patients → `label=0`; validation normals may be excluded | `orig` cases → normal/reference (`label=0`); all others → anomaly (`label=1`) |

### 7.3 🏷️ Labeling Differences

Brain and Pelvis differ in how labels are derived for ROC:

| Labeling Issue | 🧠 Brain | 🦴 Pelvis |
|---|---|---|
| **Normal class** | Included fastMRI test-normal patients | Cases identified as `orig` / normal-reference |
| **Validation normals** | Can be excluded from ROC unless intentionally included | Different fastMRI validation/test policy does not apply |
| **Anomaly class** | Non-test-normal / non-excluded validation categories in fastMRI brain workflow | Non-`orig` cases, including synthetic/clinical anomaly cohorts |

> 🚨 **Always verify patient/case naming conventions before computing ROC on new data.**

---

## 8. 🎛️ Calibration and Preprocessing Differences

| Topic | 🧠 Brain | 🦴 Pelvis | CORE Relevance |
|---|---|---|---|
| **Calibration statistic** | Per-pixel normal/reference LPIPS stats | Per-pixel normal/reference LPIPS stats | Both rely on calibration maps for Z-score thresholding |
| **LPIPS calibration reference** | Reconstruction-vs-healed | Input-vs-healed | Different heatmap semantics |
| **Smoothing kernel default** | `7` | `15` | ⚠️ Must match between calibration and inference *within* each experiment |
| **Heal patterns** | `"4"` | `"2,3"` | Changes healing masks and heatmap generation |
| **Token-surprisal samples** | 100 | 50 | Changes token-surprisal stability/counts |
| **Token mask ratio** | 0.820 | 0.90 | Changes token-surprisal branch |
| **Heatmap aggregation** | Ensemble heatmap aggregation; no geomean default emphasized | `geomean` documented as current default | Aggregation affects `Binary_Sum_Heatmap` |
| **TTA** | Supports TTA/visualization switches | `--use-tta` enabled by default | TTA affects heatmap aggregation if active |

---

## 9. 🏷️ Labels, Cohorts, and Patient Aggregation

### 9.1 🧠 Brain

Brain aggregation is built around fastMRI-style fields such as:

- `filename`
- `path`
- `category`
- `case_folder`
- Inferred patient ID from filename/case metadata
- Test-normal / validation-normal policy

**CORE patient score:**

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

### 9.2 🦴 Pelvis

Pelvis aggregation is built around:

- Filename/case identifiers
- `_slice_###` filename conventions
- `orig` naming to identify normal/reference cases
- Synthetic/clinical category metadata depending on input JSONs

**CORE patient score:**

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

### 9.3 ⚠️ Practical Warning

> The same patient-score formula does **NOT** mean the cohorts are labeled the same way. The label assignment policies are **experiment-specific** and must be checked before interpreting AUROC.

---

## 10. ✅ Reproducibility Checklist

Use this checklist when comparing or rerunning both experiments.

### 🔄 Shared Checks

- [ ] Confirm that both ROC scripts use:
  ```text
  sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
  ```
- [ ] Save the exact inference CLI command
- [ ] Save the exact ROC CLI command
- [ ] Keep the generated `results_v4_zscore.json` files
- [ ] Keep calibration `.npz` files and calibration input lists where available
- [ ] Keep Stage 1 and Stage 2 checkpoint paths / hashes
- [ ] Preserve patient-level train/validation/test split manifests
- [ ] Verify no patient/case leakage between training, calibration, validation, and test/anomaly cohorts
- [ ] Verify patient/case identifiers before ROC label assignment
- [ ] Do not expose patient-identifying information in public logs, plots, filenames, W&B, or shared outputs

### 🧠 Brain-Specific Checks

- [ ] IXI T1 preprocessing settings are recorded
- [ ] fastMRI `.h5` rendering settings are recorded
- [ ] `.npz` files contain key `arr`
- [ ] Calibration and inference use the same `--smoothing-kernel`
- [ ] Validation normals are included/excluded intentionally
- [ ] Brain `binary_token_score`, if present, is treated **only** as a backward-compatible alias for `sum_all_bars_score`

### 🦴 Pelvis-Specific Checks

- [ ] `.npy` filenames preserve `_slice_###` indices
- [ ] Slice indices are correct for 3D RoPE and per-slice calibration lookup
- [ ] `orig`/normal identifiers are correct before ROC label assignment
- [ ] Calibration and inference use the same `--smoothing-kernel`
- [ ] Synthetic and clinical cohorts are not accidentally mixed unless intended
- [ ] AUROC **and** AUPRC outputs are interpreted together when using the Pelvis merged ROC workflow

---

## 11. 🚫 What NOT to Confuse with the CORE Score

The code in both folders contains many useful diagnostic and auxiliary outputs. These are important for debugging and scientific interpretation, but they should **not** be reported as the primary AUROC score unless a separate analysis explicitly selects them.

| 🟡 Auxiliary Item | Why It Is NOT the Primary CORE Score |
|---|---|
| `clamped_pixel_sum` | Useful LPIPS-derived diagnostic, but not the patient-level ROC score in the cleaned CORE path |
| `lpips_input_recon_sum_mask` | Reconstruction diagnostic / auxiliary analysis field |
| Sharpness scores / artifact flags | Useful for quality control and artifact analysis, not the main AUROC score |
| Bounding-box precision/F1/inside-ratio metrics | Localization evaluation; labels/boxes are not used to train the model and do not define patient-level AUROC |
| Per-patient bar plots | Visual summaries of intermediate quantities; they do not by themselves define the ROC score unless they use `sum_all_bars_score` |
| Alternative Stage 2 anomaly maps | Preserved for transparency/ablation but not the primary cleaned AUROC path |
| Synthetic anomaly generation utilities | Support cohort generation and experiments; not part of the ROC score calculation itself |

---

## 🏁 Final Takeaway

The most important CORE distinction is **not** the final ROC score formula, because that should now be the **same** in Brain and Pelvis:

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

### 🎯 The important differences are instead:

<table>
<tr>
<td width="5%">1️⃣</td>
<td><b>Anatomy and data sources</b></td>
</tr>
<tr>
<td>2️⃣</td>
<td><b>Preprocessing and file formats</b></td>
</tr>
<tr>
<td>3️⃣</td>
<td><b>Stage 1 codebook size</b> — 256 (Brain) vs 192 (Pelvis)</td>
</tr>
<tr>
<td>4️⃣</td>
<td><b>Stage 2 positional encoding</b> — 2D RoPE (Brain) vs 3D RoPE (Pelvis)</td>
</tr>
<tr>
<td>5️⃣</td>
<td><b>LPIPS reference</b> — reconstruction-referenced (Brain) vs input-referenced (Pelvis)</td>
</tr>
<tr>
<td>6️⃣</td>
<td><b>Cohort-label policies</b> for ROC</td>
</tr>
<tr>
<td>7️⃣</td>
<td><b>Pelvis explicitly reports PR/AUPRC</b> in the main merged ROC workflow</td>
</tr>
</table>

> ⚠️ **When reporting AUROC, make sure both experiments are compared using the same sum-all-bars patient score — and not an outdated one-arm score.**

---

<div align="center">

### 🔬 CORE-Relevant Differences — Brain vs. Pelvis

*A consistency reference for cross-experiment comparison and reproducibility.*

</div>
