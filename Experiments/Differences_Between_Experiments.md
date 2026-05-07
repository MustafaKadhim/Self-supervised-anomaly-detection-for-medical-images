# CORE-Relevant Differences Between the Brain and Pelvis Experiments

This document summarizes the **CORE-relevant** differences between the cleaned Brain and Pelvis experiments.
Here, **CORE** means the code and settings that directly affect the primary patient-level ROC/AUROC path. **AYNU** means auxiliary code that is useful for debugging, visualization, training diagnostics, localization/bounding-box analysis, synthetic-data utilities, or alternative analyses, but does not define the primary reported ROC/AUROC score unless explicitly selected in a separate analysis.

> **Most important shared point:** both experiments should use the same patient-level ROC score definition:
>
> ```text
> sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
> ```
>
> This means ROC/AUROC must use the sum of both the token-surprisal arm and the binary/perceptual heatmap arm, not only one of them.

---

## Table of contents

- [1. Executive summary](#1-executive-summary)
- [2. Shared CORE AUROC definition](#2-shared-core-auroc-definition)
- [3. Main differences at a glance](#3-main-differences-at-a-glance)
- [4. Data sources and file formats](#4-data-sources-and-file-formats)
- [5. Model architecture differences](#5-model-architecture-differences)
- [6. Inference and heatmap differences](#6-inference-and-heatmap-differences)
- [7. ROC / AUROC / AUPRC evaluation differences](#7-roc--auroc--auprc-evaluation-differences)
- [8. Calibration and preprocessing differences](#8-calibration-and-preprocessing-differences)
- [9. Labels, cohorts, and patient aggregation](#9-labels-cohorts-and-patient-aggregation)
- [10. Reproducibility checklist](#10-reproducibility-checklist)
- [11. What not to confuse with the CORE score](#11-what-not-to-confuse-with-the-core-score)

---

## 1. Executive summary

| Topic | Brain experiment | Pelvis experiment | CORE consequence |
|---|---|---|---|
| Anatomy/domain | Brain MRI | Pelvic MRI | Different data distributions and preprocessing assumptions. |
| Main healthy/training source | IXI T1 NIfTI-derived slices | LUND-PROBE-style normal/reference pelvic MRI slices | Different source cohorts and file preparation scripts. |
| Evaluation/anomaly workflow | fastMRI-style brain rendering/annotation workflow | Synthetic and clinical pelvic anomaly cohorts | Different cohort-label logic and folder conventions. |
| Main ROC score | `sum_all_bars_score` | `sum_all_bars_score` | **Same intended score formula in both experiments.** |
| ROC score formula | `Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)` | `Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)` | Both token surprisal and binary/perceptual arms must be included. |
| Stage 1 codebook size in training script | 256 per RVQ level | 192 per RVQ level | Checkpoints are not interchangeable. |
| Stage 2 positional encoding | 2D RoPE over row/column | 3D RoPE over row/column/slice | Pelvis depends on slice-index conditioning; Brain does not use slice position in RoPE. |
| Main LPIPS reference | Reconstruction-vs-healed / reconstruction-vs-inpainted | Input-vs-healed / input-vs-inpainted | The perceptual heatmap branch is not identical. |
| Main slice format | Primarily `.npz` with key `arr`; inference can also read `.npy` | `.npy` slices with `_slice_###` filename convention | Different preprocessing and dataset assumptions. |
| ROC outputs | ROC/AUROC-focused FastMRI brain outputs | ROC/AUROC plus PR/AUPRC outputs | Pelvis ROC script explicitly reports PR/AUPRC in the main merged workflow. |

---

## 2. Shared CORE AUROC definition

Both cleaned experiments use the same intended patient-level score for primary ROC/AUROC evaluation:

```text
per-slice score contribution = token_surprisal_hot_px + Binary_Sum_Heatmap

per-patient score:
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

### Field meanings

| Field | Meaning | Used in CORE AUROC? |
|---|---|---:|
| `token_surprisal_hot_px` | Count of hot pixels from the token-surprisal branch. | Yes |
| `Binary_Sum_Heatmap` | Count of binary-positive pixels from the perceptual/binary heatmap branch after the experiment-specific fusion/thresholding logic. | Yes |
| `sum_all_bars_score` | Patient-level sum of the two per-slice quantities above. | Yes |
| `binary_token_score` | In the Brain code, retained as a backward-compatible alias for the corrected combined score. | Do not treat as a separate new score. |

### Why this matters

The CORE score is **not** only the perceptual/binary heatmap side and is **not** only the token-surprisal side. For fair Brain-vs-Pelvis comparison, both arms must contribute identically at patient aggregation:

```text
Brain  ROC score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
Pelvis ROC score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

---

## 3. Main differences at a glance

| Category | Brain folder | Pelvis folder |
|---|---|---|
| Stage 1 file | `Model_Stage1.py` | `Model_Stage_1.py` |
| Stage 2 file | `Model_Stage_2.py` | `Model_Stage_2.py` |
| Inference file | `Inference_Brain_Experiments.py` | `Inference_Pelvis_Experiments.py` |
| ROC file | `ROC_Curve_Calculations.py` | `ROC_Curves_Calculations.py` |
| Training entry point | `Train_framework.py` | `Train_framework.py` |
| Dataset file | `dataset.py` | `dataset.py` |
| Split manifest | `Train_Val_Test_Exact_DataSplits_IXI_fastMRI.json` | `Train_Val_Test_Exact_DataSplits_LUND_PROBE.json` |
| Reference config | `config_yaml.yaml` | `config_yaml.yaml` |
| Main preprocessing utilities | `IXI_dataset_overview.py`, `Render_patient_slices_from_csv.py`, `collect_normal_slices.py`, label-folder builders | `preslice_volumes.py`, `External_dataset.py`, simulation helper scripts |

---

## 4. Data sources and file formats

### 4.1 Brain experiment

The Brain repository is organized around:

- healthy **IXI T1** NIfTI volumes for training-style slice generation;
- fastMRI-style `.h5` rendering / annotation workflows for evaluation/anomaly folders;
- saved 2D slices, primarily as `.npz` files containing key `arr`.

Relevant Brain files:

| Purpose | File |
|---|---|
| IXI NIfTI to training-ready slices | `IXI_dataset_overview.py` |
| fastMRI `.h5` rendering to `.npz` / PNG | `Render_patient_slices_from_csv.py` |
| normal-slice selection for calibration | `collect_normal_slices.py` |
| global/local anomaly folder construction | `build_patient_Global_label_folders.py`, `build_patient_Local_label_folders.py` |

Brain preprocessing details documented in the README include:

- NIfTI loading and canonical reorientation for IXI;
- per-volume z-score normalization;
- clipping commonly to `[-3, 3]`;
- crop/pad to `256 × 256`;
- `.npz` output with key `arr`;
- fastMRI anomaly/evaluation slices rendered from `.h5` `reconstruction_rss` volumes.

### 4.2 Pelvis experiment

The Pelvis repository is organized around:

- LUND-PROBE-style pelvic MRI normal/reference data;
- `.npy` slice files;
- filename slice indices using the `_slice_###` convention.

Relevant Pelvis files:

| Purpose | File |
|---|---|
| NIfTI volume to per-slice `.npy` preprocessing | `preslice_volumes.py` |
| external cohort preprocessing/loading utilities | `External_dataset.py` |
| synthetic anomaly support | `Simulation_inference_v4_extended_CJG.py`, `Simluation_inference_v3_support_CJG.py` |

Pelvis preprocessing details documented in the README include:

- loading `float32` `.npy` slices;
- rotation with `np.rot90(arr, k=-1)` in `dataset.py`;
- resize to `320 × 320` and center crop to `256 × 256`;
- saved slice naming such as `{patient_id}_slice_{idx:03d}.npy`.

### 4.3 CORE implication

| Issue | Brain | Pelvis | Why it matters |
|---|---|---|---|
| File format | Mostly `.npz` with key `arr`; inference can also read `.npy` | `.npy` | Dataset loaders and preprocessing assumptions differ. |
| Slice index | Present in filenames when available, but not used for Brain 2D RoPE | Required/important for Pelvis 3D RoPE and per-slice calibration lookup | Pelvis filenames must preserve `_slice_###`. |
| Orientation/rotation | IXI and fastMRI workflows have their own orientation/rendering steps | Dataset loader rotates with `np.rot90(arr, k=-1)` | Do not mix prepared data without checking orientation. |
| Source cohorts | IXI + fastMRI-style brain workflow | LUND-PROBE-style pelvic workflow | Cohort labels and patient grouping differ. |

---

## 5. Model architecture differences

Both experiments use the same broad two-stage idea:

1. **Stage 1 RVQ-VAE** learns a discrete latent representation.
2. **Stage 2 Factorized MaskGIT / Fact-biT** predicts/heals masked tokens.

However, several CORE-relevant implementation details differ.

### 5.1 Stage 1 differences

| Feature | Brain | Pelvis | CORE relevance |
|---|---|---|---|
| Input | Single-channel 2D brain slice, typically `1 × 256 × 256` | Single-channel 2D pelvic slice, typically `1 × 256 × 256` | Same final spatial target, different anatomy/preprocessing. |
| Patch size | 8 | 8 | Both typically produce `32 × 32 = 1024` tokens. |
| Encoder | ViT-style encoder, depth 8, 8 heads | ViT-style encoder, depth 8, 8 heads | Broadly similar. |
| RVQ levels | 2 | 2 | Broadly similar. |
| Codebook size | 256 per RVQ level | 192 per RVQ level | Checkpoints/token distributions are not interchangeable. |
| Stage 1 training LR | `2e-4` in Brain README/script summary | `1e-4` in Pelvis README/script summary | Training dynamics differ. |
| BiomedCLIP perceptual weight | 0.5 | 0.9 | Training objective differs. |

### 5.2 Stage 2 differences

| Feature | Brain | Pelvis | CORE relevance |
|---|---|---|---|
| Token streams | L1/L2 token streams | L1/L2 token streams | Similar broad design. |
| Codebook size expected from Stage 1 | 256 per level | 192 per level | Stage 2 must match its Stage 1 checkpoint. |
| Positional encoding | **2D RoPE** over row and column | **3D RoPE** over row, column, and slice position | Major CORE difference. |
| Slice position conditioning | `slice_pos` may be accepted by signatures but is not part of Brain 2D RoPE | Slice index is used for anatomical position encoding | Pelvis depends more strongly on correct slice-index filenames. |
| Stage 2 loss | `CE(masked L1) + 0.25 × CE(masked L2)` | Same documented loss form | Broadly similar training target. |
| Label smoothing | 0.05 | 0.05 | Similar. |

---

## 6. Inference and heatmap differences

Both experiments use Recursive-AutoMask V4-style inference with:

- Stage 1 reconstruction/tokens;
- Stage 2 token healing/inpainting;
- token surprisal;
- LPIPS heatmaps;
- binary threshold/fusion;
- JSON output containing `token_surprisal_hot_px` and `Binary_Sum_Heatmap`.

The main CORE difference is the LPIPS reference used by the primary heatmap branch.

| Inference component | Brain | Pelvis |
|---|---|---|
| Main inference script | `Inference_Brain_Experiments.py` | `Inference_Pelvis_Experiments.py` |
| Main LPIPS calibration reference | `LPIPS(Stage 1 reconstruction, healed reconstruction)` | `LPIPS(input, healed)` |
| Main LPIPS inference iteration 0 reference | `LPIPS(Stage 1 reconstruction, healed reconstruction)` | `LPIPS(input, healed)` |
| Refinement/inpainting LPIPS reference | `LPIPS(Stage 1 reconstruction, inpainted reconstruction)` | `LPIPS(input, inpainted)` |
| Token surprisal branch | Monte Carlo token masking / Stage 2 prediction | Monte Carlo token masking / Stage 2 prediction |
| Per-slice token output | `token_surprisal_hot_px` | `token_surprisal_hot_px` |
| Per-slice binary/perceptual output | `Binary_Sum_Heatmap` | `Binary_Sum_Heatmap` |

### Important interpretation

Because the LPIPS reference differs, the meaning of `Binary_Sum_Heatmap` is not numerically identical between experiments even though it plays the same role in the final score formula.

- In Brain, `Binary_Sum_Heatmap` is driven by a reconstruction-referenced perceptual comparison.
- In Pelvis, `Binary_Sum_Heatmap` is driven by an input-referenced perceptual comparison.

But in both cases, the final patient score is still:

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

---

## 7. ROC / AUROC / AUPRC evaluation differences

### 7.1 Shared scoring rule

| Item | Brain | Pelvis |
|---|---|---|
| Per-slice fields used | `token_surprisal_hot_px`, `Binary_Sum_Heatmap` | `token_surprisal_hot_px`, `Binary_Sum_Heatmap` |
| Patient score | `sum_all_bars_score` | `sum_all_bars_score` |
| Formula | `Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)` | `Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)` |

### 7.2 ROC script behavior

| Feature | Brain | Pelvis |
|---|---|---|
| ROC script | `ROC_Curve_Calculations.py` | `ROC_Curves_Calculations.py` |
| Main aggregation function | `aggregate_fastmri_binary_token_patient_scores(...)` | `aggregate_patient_sum_of_all_bars(...)` |
| Main ROC function | `compute_fastmri_roc_and_auc(...)` | `compute_patient_roc_and_auc(...)` |
| Main patient-score key | `sum_all_bars_score` | `sum_all_bars_score` |
| Backward-compatible alias | `binary_token_score` may alias corrected combined score | Not needed as primary name in the cleaned pelvis ROC path |
| PR/AUPRC | Brain ROC script is ROC/AUROC-focused in the inspected CORE path | Pelvis ROC script explicitly computes ROC/AUROC and PR/AUPRC in `compute_patient_roc_and_auc(...)` |
| Normal-label policy | Included test-normal fastMRI patients are `label=0`; validation normals may be excluded by policy | “orig” cases are treated as normal/reference (`label=0`); all others as anomaly (`label=1`) |

### 7.3 Labeling differences

Brain and Pelvis differ in how labels are derived for ROC:

| Labeling issue | Brain | Pelvis |
|---|---|---|
| Normal class | Included fastMRI test-normal patients | Cases identified as `orig` / normal-reference |
| Validation normals | Can be excluded from ROC unless intentionally included | Not the same fastMRI validation/test policy |
| Anomaly class | Non-test-normal / non-excluded validation categories in the FastMRI brain workflow | Non-`orig` cases, including synthetic/clinical anomaly cohorts depending on inputs |

Always verify patient/case naming conventions before computing ROC on new data.

---

## 8. Calibration and preprocessing differences

| Topic | Brain | Pelvis | CORE relevance |
|---|---|---|---|
| Calibration statistic | Per-pixel healthy LPIPS statistics | Per-pixel normal/reference LPIPS statistics | Both rely on calibration maps for Z-score thresholding. |
| LPIPS calibration reference | reconstruction-vs-healed | input-vs-healed | Different heatmap semantics. |
| Documented smoothing kernel default | 7 | 15 | Must match between calibration and inference within each experiment. |
| Documented heal patterns | `"4"` | `"2,3"` | Changes healing masks and heatmap generation. |
| Documented token-surprisal samples | 100 | 50 | Changes token-surprisal stability/counts. |
| Documented token mask ratio | 0.820 | 0.90 | Changes token-surprisal branch. |
| Documented heatmap aggregation | Brain README describes ensemble heatmap aggregation; no geomean default emphasized in the same way | `geomean` documented as current default | Aggregation affects `Binary_Sum_Heatmap`. |
| Documented TTA | Brain script supports TTA/visualization switches | Pelvis README says `--use-tta` is enabled by default | TTA affects heatmap aggregation if active. |

---

## 9. Labels, cohorts, and patient aggregation

### 9.1 Brain

Brain aggregation is built around fastMRI-style fields such as:

- `filename`
- `path`
- `category`
- `case_folder`
- inferred patient ID from filename/case metadata
- test-normal / validation-normal policy

CORE patient score:

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

### 9.2 Pelvis

Pelvis aggregation is built around:

- filename/case identifiers;
- `_slice_###` filename conventions;
- `orig` naming to identify normal/reference cases;
- synthetic/clinical category metadata depending on input JSONs.

CORE patient score:

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

### 9.3 Practical warning

The same patient-score formula does **not** mean the cohorts are labeled the same way. The label assignment policies are experiment-specific and must be checked before interpreting AUROC.

---

## 10. Reproducibility checklist

Use this checklist when comparing or rerunning both experiments.

### Shared checks

- [ ] Confirm that both ROC scripts use:

  ```text
  sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
  ```

- [ ] Save the exact inference CLI command.
- [ ] Save the exact ROC CLI command.
- [ ] Keep the generated `results_v4_zscore.json` files.
- [ ] Keep calibration `.npz` files and calibration input lists where available.
- [ ] Keep Stage 1 and Stage 2 checkpoint paths / hashes.
- [ ] Preserve patient-level train/validation/test split manifests.
- [ ] Verify no patient/case leakage between training, calibration, validation, and test/anomaly cohorts.
- [ ] Verify patient/case identifiers before ROC label assignment.
- [ ] Do not expose patient-identifying information in public logs, plots, filenames, W&B, or shared outputs.

### Brain-specific checks

- [ ] IXI T1 preprocessing settings are recorded.
- [ ] fastMRI `.h5` rendering settings are recorded.
- [ ] `.npz` files contain key `arr`.
- [ ] Calibration and inference use the same `--smoothing-kernel`.
- [ ] Validation normals are included/excluded intentionally.
- [ ] Brain `binary_token_score`, if present, is treated only as a backward-compatible alias for `sum_all_bars_score`.

### Pelvis-specific checks

- [ ] `.npy` filenames preserve `_slice_###` indices.
- [ ] Slice indices are correct for 3D RoPE and per-slice calibration lookup.
- [ ] `orig`/normal identifiers are correct before ROC label assignment.
- [ ] Calibration and inference use the same `--smoothing-kernel`.
- [ ] Synthetic and clinical cohorts are not accidentally mixed unless intended.
- [ ] AUROC and AUPRC outputs are interpreted together when using the Pelvis merged ROC workflow.

---

## 11. What not to confuse with the CORE score

The code in both folders contains many useful diagnostic and auxiliary outputs. These are important for debugging and scientific interpretation, but they should not be reported as the primary AUROC score unless a separate analysis explicitly selects them.

| Auxiliary item | Why it is not the primary CORE score |
|---|---|
| `clamped_pixel_sum` | Useful LPIPS-derived diagnostic, but not the patient-level ROC score in the cleaned CORE path. |
| `lpips_input_recon_sum_mask` | Reconstruction diagnostic / auxiliary analysis field. |
| Sharpness scores / artifact flags | Useful for quality control and artifact analysis, not the main AUROC score. |
| Bounding-box precision/F1/inside-ratio metrics | Localization evaluation; labels/boxes are not used to train the model and do not define patient-level AUROC. |
| Per-patient bar plots | Visual summaries of intermediate quantities; they do not by themselves define the ROC score unless they use `sum_all_bars_score`. |
| Alternative Stage 2 anomaly maps | Preserved for transparency/ablation but not the primary cleaned AUROC path. |
| Synthetic anomaly generation utilities | Support cohort generation and experiments; not part of the ROC score calculation itself. |

---

## Final takeaway

The most important CORE distinction is not the final ROC score formula, because that should now be the **same** in Brain and Pelvis:

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
```

The important differences are instead:

1. anatomy and data sources;
2. preprocessing and file formats;
3. Stage 1 codebook size;
4. Stage 2 positional encoding: Brain 2D RoPE vs Pelvis 3D RoPE;
5. LPIPS reference: Brain reconstruction-referenced vs Pelvis input-referenced;
6. cohort-label policies for ROC;
7. Pelvis explicitly reporting PR/AUPRC in the main merged ROC workflow.

When reporting AUROC, make sure both experiments are compared using the same **sum-all-bars** patient score and not an outdated one-arm score.
