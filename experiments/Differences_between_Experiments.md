# Framework Differences: Pelvic MRI vs. Brain MRI

This document describes every design decision that differs between the two instantiations of the two-stage anomaly detection framework. Both frameworks share the same conceptual architecture — Stage 1 RVQ-VAE followed by Stage 2 Factorized MaskGIT — but are adapted to the imaging modality, dataset characteristics, and evaluation protocol of each domain.

Full architecture documentation for each variant lives in its own README:
- Pelvic MRI: `Final_Code_Phiro_Pelvic_MRI/README.md`
- Brain MRI: `Final_Code_Phiro_Brain_MRI/README.md`

---

## 1. Dataset and Imaging Modality

| | Pelvic MRI | Brain MRI |
|---|---|---|
| **Dataset** | LUND-PROBE and internal clinical cases | fastMRI (normal for training, validation and testing) and (fastMRI+ annotated anomaly) + IXI (healthy training) |
| **MRI sequence** | T2-weighted | T1-weighted |
| **Normal training subjects** | 384 patients (from 467 total); splits from `Train_Val_Test_Exact_DataSplits_LUND_PROBE.json` | IXI healthy (581 cases, axial slices 128–188), fastMRI normal (172 cases, 0-10 slices) `Train_validation_test_anomaly_splits_brain.json` |
| **Anomaly test subjects** | Same LUND-PROBE cohort (held-out patients) and clinical cases | fastMRI+ annotated T1 brain scans |

---

## 2. Data Format and Preprocessing

| | Pelvic MRI | Brain MRI |
|---|---|---|
| **Slice file format** | `.npy` (single float32 array) | `.npz` (compressed; array key: `arr`) |
| **Preprocessing script** | `preslice_volumes.py` | `IXI_dataset_overview.py` for IXI training data; `Render_patient_slices_from_csv.py` for FastMRI `.h5` slice export / curation |
| **Z-score normalisation** | Per volume: `(vol − μ) / σ`; σ clipped to ≥ 1e-8 | Per volume: `(vol − μ) / σ`; clipped to [−3, 3] |
| **In-plane target size** | Resize to 320×320 (area) → CenterCrop to 256×256 | IXI: center-crop-or-pad to 256×256. fastMRI: pad/crop to 320×320, then resize to 256×256 |
| **Naming convention** | `{patient_id}_slice_{idx:03d}.npy` | `{file_id}_slice_{idx:03d}.npz` |
| **External cohort script** | `External_dataset.py` (names as `{category}_{case_folder}_{volume_name}_slice_{idx:03d}.npy`) | FastMRI export utility also supports PNG + NPZ writing from CSV or per-label folder structure |

---

## 3. Train / Validation Split

| | Pelvic MRI | Brain MRI |
|---|---|---|
| **Split method** | Single directory; 10% held out internally (`val_split=0.10`, `seed=42`) | Separate `--train-dir` and `--val-dir` (pre-split on disk) |
| **Split manifest** | Optional our JSON (`Train_Val_Test_Exact_DataSplits_LUND_PROBE.json`) | Optional our JSON (`Train_validation_test_anomaly_splits_brain.json`) |

---

## 4. Stage 1 — RVQ-VAE Hyperparameters

| | Pelvic MRI | Brain MRI |
|---|---|---|
| **Codebook size** | **192** codes per quantizer level | **256** codes per quantizer level |
| **Perceptual loss weight** | **0.9** (strong BiomedCLIP supervision) | **0.5** (moderate BiomedCLIP supervision) |
| **Learning rate** | **1e-4** | **2e-4** |
| **AdamW betas** | (0.9, 0.95) | (0.9, 0.95) |
| **embed_dim** | 256 | 256 |
| **patch_size** | 8 (→ 32×32 = 1 024 tokens) | 8 (→ 32×32 = 1 024 tokens) |
| **encoder_depth / heads** | 8 / 8 | 8 / 8 |
| **num_quantizers** | 2 | 2 |
| **commitment_cost** | 0.25 | 0.25 |

---

## 5. Stage 1 — Data Augmentation

The augmentation settings between the two frameworks during training.

| Augmentation | Pelvic MRI | Brain MRI |
|---|---|---|
| **RandScaleIntensity** | factor=0.10, prob=0.33 | factor=0.10, prob=0.33 |
| **RandAdjustContrast** | — | γ ∈ [0.5, 1.5], prob=0.33 |
| **RandGaussianNoise** | — | prob=**0.50**, std=**0.30** |
| **RandAffine — rotation** | ±**5°**, prob=0.33 | ±**15°**, prob=0.33 |
| **RandAffine — translation** | ±**5 px**, horizontal only | ±**15 px**, horizontal **and vertical** |
| **RandAffine — zoom** | — | 0.80×–1.20× |
| **RandFlip** | horizontal, prob=0.50 | horizontal, prob=0.50 |
| **Location of augmentation** | Applied inside `training_step` | Applied inside `training_step` |

**Rationale:** Brain MRI exhibits substantially higher inter-subject variability in intensity, contrast, and field-of-view alignment. Gaussian noise and intensity augmentations simulate protocol acquisition variations. Augmentations may prevent over-fitting.

---

## 6. Stage 2 — Positional Encoding (Key Architectural Difference)

| | Pelvic MRI | Brain MRI |
|---|---|---|
| **RoPE class** | `RotaryEmbedding3D` | `RotaryEmbedding2D` |
| **Axes encoded** | **Row, column, and slice (z)** | **Row and column only** |
| **head_dim split** | Thirds: row_dim, col_dim, slice_dim | Halves: row_dim, col_dim |
| **max_positions** | 64 (spatial) | seq_hw + 1 = 33 (spatial) |
| **max_slices** | 92 | — (not applicable) |
| **slice_pos argument** | Actively used during training and inference | Accepted by API but silently ignored (backward compatibility) |

**Rationale:** Pelvic MRI slices are scanned continuously along a single anatomical axis (inferior–superior), so the slice index carries strong positional semantics for anatomical structures. Brain MRI slices in the FastMRI dataset are processed slice-independently without volumetric context, making the slice dimension uninformative/unconsistent for the model.

---

## 7. Stage 2 — Training Slice Filtering

| | Pelvic MRI | Brain MRI |
|---|---|---|
| **Slice range filter** | Training batches restricted to axial slices **30-60** (`_filter_training_slices`) | IXI healthy (axial slices **128–188**), fastMRI normal (axial slices **0-10**) |
| **Behaviour when batch fails filter** | Batch is skipped entirely | Not applicable |

**Rationale:** Pelvic T2 slices 0–29 and 61+ capture non-pelvic anatomy (abdomen above, femur below) with very different appearance. Restricting to [30, 60] focuses learning on the prostate/bladder/rectum region. Brain slices 128–188 (selected during preprocessing) are all anatomically relevant and require no further filtering.

---

## 8. Stage 2 — Training Configuration

| | Pelvic MRI | Brain MRI |
|---|---|---|
| **Batch size — Stage 1** | 128 | **192** |
| **Batch size — Stage 2** | 128 | **158** |
| **num_workers** | 8 | 12 |
| **Learning rate** | **1e-4** | **2e-4** |
| **l2_loss_weight** | 0.25 | 0.25 |
| **q_error_weight** | 0.10 | 0.10 |
| **label_smoothing** | 0.05 | 0.05 |
| **warmup_steps** | 2000 | 2000 |
| **Fine-tuning warm-start** | Not supported | `--pretrained-stage2-ckpt` loads previous Stage 2 with `strict=False` |

---

## 9. Inference — Patient-Level Scoring Range

| | Pelvic MRI | Brain MRI |
|---|---|---|
| **Slice range for patient score** | Slices **38–49** only (`patient_score_slice_min/max`) | **All slices** evaluated (no restriction) |
| **Rationale** | Slices 38–49 cover the prostate/central pelvic structures; only these slices carry diagnostic information for the target pathologies | Brain anomalies (mass, edema, infarct, etc.) can occur at any axial location; no a-priori restriction is valid |

---

## 10. Inference — Evaluation Method and Metrics

| | Pelvic MRI | Brain MRI |
|---|---|---|
| **Primary evaluation unit** | Patient-level | Slice-level (aggregated to patient-level) |
| **Ground truth format** | Category label in filename; no spatial annotation | Bounding-box CSV (`x, y, width, height, label, study_level, base_size`) |
| **TP definition** | Patient correctly flagged as anomalous | Predicted mask covers ≥ 10% of GT bounding-box area (`tp_inside_ratio_threshold`) |
| **FP ratio** | Not applicable at patient level | Predicted pixels outside healthy region / predicted pixels inside GT box |
| **Metrics reported** | Patient-level AUC per anomaly category | Per-slice precision, F1, `inside_bbox_detection_ratio` + patient-level `clamped_pixel_sum` |
| **Annotation coordinate modes** | — | `legacy`, `render_fastmri` (y-axis flipped), `mask_pipeline` (rasterise then resize) |
| **LPIPS backflow fusion** | Not present | `build_final_lpips_binary_token_eval_mask`: fuses LPIPS (60th-percentile threshold) with token surprisal (55% LPIPS + 45% max of LPIPS/surprisal) |
| **Edge-to-center erosion** | Not present | `apply_edge_to_center_erosion`: stronger boundary erosion, weaker near image centre |

---

## 11. Anomaly Categories

### Pelvic MRI (LUND-PROBE)
Synthetic and clinical categories encoded in filenames, parsed by substring rules:

| Category | Type |
|----------|------|
| RandomGhosting | Synthetic |
| RandomNoise | Synthetic |
| RandomSpike | Synthetic |
| RandomMotion | Synthetic |
| WholeImageGaussian | Synthetic |
| Stor_T2_till_sCT | Synthetic (MRI-to-CT conversion) |
| ClinicalVariations | Clinical |
| Spacer | Clinical (SpaceOAR hydrogel) |
| Unknown | — |

### Brain MRI (FastMRI)

**Global labels** (study-level, from `build_patient_Global_label_folders.py`):
Motion artifact, Possible artifact, Colpocephaly, Extra-axial collection, Small vessel chronic white matter ischemic change.

**Local labels** (per-slice, from `build_patient_Local_label_folders.py`):
Edema, Enlarged ventricles, Craniotomy, Mass, Nonspecific lesion, Resection cavity, Intraventricular substance, Paranasal sinus opacification, Posttreatment change, Nonspecific white matter lesion, Encephalomalacia, Dural thickening, Absent septum pellucidum, Lacunar infarct, Likely cysts.

---

## 12. Data Annotation and Label-Building Tools

| | Pelvic MRI | Brain MRI |
|---|---|---|
| **Category encoding** | In filename prefix (e.g., `RandomGhosting_patient_slice_045.npy`); no external annotation needed | External CSV with per-slice bounding boxes |
| **Normal slice collection** | Healthy-volunteer split; all slices treated as normal | `collect_normal_slices.py` — filters FastMRI annotation CSV for `study_level=yes`, normal label keyword, or unannotated slices |
| **Label organisation** | Not applicable (categories in filename) | `build_patient_Global_label_folders.py` and `build_patient_Local_label_folders.py` create per-label, per-patient folder trees with `patients.csv` and optional `slices.csv` |
| **Rendered slice export utility** | Not required as a separate step | `Render_patient_slices_from_csv.py` reads FastMRI `.h5` `reconstruction_rss`, selects slices from CSV or label folders, optionally overlays annotation boxes, and writes PNG/NPZ outputs |

In the brain pipeline, this extra rendering/export script matters because part of the documented preprocessing is not only IXI NIfTI → training `.npz`, but also FastMRI `.h5` → curated 2D slice exports for anomaly review and label-folder generation. That step includes series filtering (`AXT1`), per-volume z-scoring with clipping, vertical flipping, optional PNG box overlays, and final 256×256 export.

---

## 13. Summary Comparison Table

| Aspect | Pelvic MRI (LUND-PROBE) | Brain MRI (FastMRI / IXI) |
|--------|------------------------|--------------------------|
| MRI sequence | T2-weighted | T1-weighted |
| Codebook size | **192** | **256** |
| Perceptual loss weight | **0.9** | **0.5** |
| Learning rate | **1e-4** | **2e-4** |
| Augmentation richness | 2 transforms (scale + affine) | 5 transforms (+contrast, +noise, +zoom, +flip) |
| Augmentation rotation | **±5°** | **±15°** |
| Augmentation translation | **±5 px horizontal only** | **±15 px horizontal + vertical** |
| Gaussian noise | — | **p=0.50, std=0.30** |
| Dataset augmentation (MONAI) | RandFlip + RandRotate in `dataset.py` | None in dataloader |
| Positional encoding | **3D RoPE** (row, col, slice) | **2D RoPE** (row, col only) |
| Training slice filter | Slices **30–60** only | No filtering |
| Batch size | 128 (both stages) | 192 (S1) / 158 (S2) |
| num_workers | 8 | 12 |
| Train/val split | Internal (10%) | Separate directories |
| File format | `.npy` | `.npz` (key `arr`) |
| Rotation applied | At load time | At save time |
| Patient-level slice range | 38–49 | All slices |
| Evaluation | Patient-level AUC by category | Per-slice bounding-box TP/F1 + patient clamp-sum |
| Annotation format | Filename-encoded category | External bounding-box CSV |
| Fine-tuning CLI support | — | `--pretrained-stage1/2-ckpt` |
| Render/export utility | Not needed as a separate documented step | `Render_patient_slices_from_csv.py` for CSV- or label-folder-driven `.h5` → PNG/NPZ slice generation |
| LPIPS backflow fusion | — | Yes (55%/45% blending) |
| Edge-to-center erosion | — | Yes |
