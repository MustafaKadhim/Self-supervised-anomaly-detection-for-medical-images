


<div align="center">

<img src="figures/Anomaly_detection_official_logo_noBg.png" alt="Anomaly Detection Logo" width="400" class="center"/>
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Inter&weight=600&size=28&duration=3000&pause=1000&color=58A6FF&center=true&vCenter=true&width=600&lines=Self-Supervised+Anomaly+Detection;Medical+Image+Analysis+Framework;Train+on+Normal+Data+Only" alt="Typing SVG" />
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-Lightning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch Lightning"></a>
  <a href="https://monai.io/"><img src="https://img.shields.io/badge/MONAI-Medical_AI-69D3A7?style=flat-square" alt="MONAI"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Research_Code-f1c232?style=flat-square" alt="Research code"></a>
</p>


  
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22863a?style=flat-square" alt="License"></a>
  <a href="https://github.com/MustafaKadhim/Anomaly-Detection-Self-supervised-anomaly-detection-for-medical-images/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-1f6feb?style=flat-square" alt="PRs"></a>
</p>


<div align="center">

<p>
  <a href="https://github.com/MustafaKadhim/Anomaly-Detection-Self-supervised-anomaly-detection-for-medical-images/stargazers ">⭐ Star this repo</a> &nbsp;•&nbsp;
  <a href="https://arxiv.org/abs/2605.24609">📄 Read the paper</a>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake.svg">
  <img alt="github contribution grid snake animation" src="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake.svg">
</picture>

</div>


*A two-stage self-supervised framework for unsupervised anomaly detection in medical MRI that learns from normal/reference images and detects deviations from the learned normal distribution using token surprisal and perceptual analysis.*

[🚀 Quickstart](#-quickstart) · [🏗 Framework](#-framework) · [🧪 Experiments](#-experiments) · [📊 Evaluation](#-evaluation) · [📁 Repository Structure](#-repository-structure) · [🔖 Citation](#-citation)

</div>

---
## ⭐ A Brief Description of Our Work
Two-stage self-supervised framework for unsupervised anomaly detection in medical MRI.

The framework learns a compact, discrete representation of normal anatomy through two jointly trained but sequentially designed stages. Stage 1 is a Residual Vector Quantized Variational Autoencoder (RVQ-VAE) with a Vision Transformer encoder, a PixelShuffle decoder, and a two-level residual codebook. It is trained on normal reference MRI slices using L1 reconstruction loss regularized by a BiomedCLIP perceptual term, producing a discrete token grid that captures structural and textural anatomy at multiple levels of abstraction.

Stage 2 is a factorized bidirectional masked transformer (Fact-biT / Factorized MaskGIT) that operates on the discrete tokens produced by the frozen Stage 1 encoder. It is trained via masked token prediction — a visual analogue of BERT — learning the joint distribution of normal anatomy tokens across spatial positions. For pelvic MRI, it extends positional encoding to three dimensions (row, column, slice index) via 3D Rotary Position Embeddings (RoPE), allowing the model to condition its predictions on anatomical depth.

At inference, neither stage has ever seen anomalous data. Anomaly detection is driven by three complementary signals, fused under the Recursive-AutoMask V4 protocol. The first signal is token surprisal: the true Stage 1 tokens are repeatedly masked at random, predicted by Stage 2, and scored by their negative log-likelihood — regions where the normal distribution assigns low probability to the observed tokens are flagged. The second signal is an LPIPS healing heatmap: Stage 2 heals checkerboard-masked tokens, decodes a "healed" image, and the spatial perceptual distance (LPIPS) between the input and the healed image is thresholded via a calibrated Z-score map built from normal reference slices. The third, optional signal is LPIPS backflow: after targeted inpainting of suspect regions, the perceptual distance between the input and the inpainted image provides a final refinement mask.

These three binary masks are unioned into a single per-slice anomaly map (Final_Binary_sum_of_anomaly_maps), and a patient-level score is formed by summing this count across slices. ROC/AUROC-analysis are computed from this patient-level score against ground-truth cohort labels; labels that were never exposed to the model during training, calibration, or scoring.

Our framework design intentionally separates what contributes to the AUROC (the binary union mask) from auxiliary diagnostics retained for transparency and ablation by implementing a CORE vs. AYNU distinction. The CORE vs. AYNU code annotation makes the primary evaluation path auditable and reproducible, while preserving intermediate outputs that are useful for understanding model behaviour, tuning thresholds, and generating qualitative visualizations. This facilitate understanding what is important without getting lost in what actually matter in the end for heatmap visualizations and ROC-analysis.


---
## 🌟 Why This Repository?

<table>
<tr>
<td width="50%" valign="top">

### ✅ Normal-data training

The learned models are trained on **normal/healthy or reference MRI slices**. Ground-truth anomaly labels are **not used for model training**; labels are used only for evaluation, cohort assignment, plotting, and optional annotation overlays.

</td>
<td width="50%" valign="top">

### 🧩 Two-stage token framework

Both experiments use a **Stage 1 RVQ-VAE** to learn discrete image tokens and a **Stage 2 bi-directional  transformer (Fact-biT)** to model token distributions and heal masked/suspect regions.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📈 Reproducible experiment folders

The cleaned Pelvis and Brain folders include model code, training entry points, inference/calibration scripts, ROC/AUPRC evaluation code, recorded split manifests, and experiment-specific README files.

</td>
<td width="50%" valign="top">

### 🏥 Medical MRI research workflows

The repository documents two MRI research workflows: **Pelvic MRI** based on a LUND-PROBE-style workflow and **Brain MRI** using IXI healthy data with fastMRI-style evaluation data.

</td>
</tr>
</table>

> ⚕️ **Research code only.** This repository is not a clinically validated tool and must not be used for clinical decision-making.

---

## 🏗 Framework

The current cleaned code is **not** a single generic autoencoder package. It is organized around two complete experiment implementations that share the same high-level anomaly-detection idea:

```text
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT MRI SLICE                         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          Stage 1: RVQ-VAE  →  reconstruction / RVQ tokens       │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          Stage 2: Fact-biT →  token healing                     │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      LPIPS perceptual scoring + token surprisal scoring         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Patient-level score:                                           │
│  sum_all_bars_score = Σ_slices(Final_Binary_sum_of_anomaly_maps)│
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ROC / AUROC or PR / AUPRC                      │
└─────────────────────────────────────────────────────────────────┘
```

### 🟢 CORE vs. 🟡 AYNU

<table>
<tr>
<th width="15%">🟢 CORE</th>
<td>Code and output fields that directly contribute to the primary patient-level AUROC/AUPRC pipeline.</td>
</tr>
<tr>
<th width="15%">🟡 AYNU</th>
<td><b>Available Yet Not AUROC-interesting</b>: auxiliary scripts, diagnostics, plots, localization overlays, alternative scores, and visualization utilities preserved for transparency and reproducibility.</td>
</tr>
</table>

The primary per-slice CORE field is:

| Field | Meaning |
|---|---|
| `Final_Binary_sum_of_anomaly_maps` | Count of white pixels in the final binary ALM mask after ALM-A ∪ ALM-B fusion, optional LPIPS-backflow, and optional edge erosion (disabled by default). |

`token_surprisal_hot_px` is written to JSON for debugging and ablation only — it is **not** used for the ROC-analysis. `Final_Binary_sum_of_anomaly_maps` already contains ALM-B (binarized token-surprisal) pixels through the fusion; adding `token_surprisal_hot_px` again would double-count token evidence.

Both cleaned experiments aggregate the CORE field as:

```text
sum_all_bars_score = Σ_slices(Final_Binary_sum_of_anomaly_maps)
```

---

## 🚀 Quickstart

### 1️⃣ Clone the repository

```bash
git clone https://github.com/MustafaKadhim/Anomaly-Detection-Self-supervised-anomaly-detection-for-medical-images.git
cd Anomaly-Detection-Self-supervised-anomaly-detection-for-medical-images
```

### 2️⃣ Choose an experiment

<table>
<tr>
<td width="50%" valign="top">

#### 🦴 Pelvic MRI

```bash
cd Pelvis_Experiments
pip install -r Pelvis_Experiments_requirements.txt
```

Start with:

```text
Pelvis_Experiments/README.md
```

</td>
<td width="50%" valign="top">

#### 🧠 Brain MRI

The Brain folder README lists the required scientific Python packages:

```text
Brain_Experiments/README.md
```

Install the packages listed there, then run the Brain-specific training, calibration, inference, and ROC scripts with explicit paths.

</td>
</tr>
</table>

> ⚠️ Many scripts still contain absolute local default paths from the original experiment environments. For a new machine or GitHub user, **pass explicit CLI paths** and save the exact command used for each run.

---

## 🧪 Experiments

Two independent cleaned experiment folders are provided.

<table>
<tr>
<td width="50%" valign="top">

### 🦴 Pelvic MRI

| | |
|---|---|
| **Domain** | Pelvic MRI |
| **Data workflow** | LUND-PROBE-style pelvic MRI workflow |
| **Training format** | `.npy` slices with `_slice_###` filename indices |
| **Stage 1** | RVQ-VAE, codebook size 192 per RVQ level |
| **Stage 2** | Fact-biT with **3D RoPE** |
| **Main LPIPS reference** | Input-vs-healed/inpainted |
| **Binary fusion** | ALM-A ∪ ALM-B ∪ LPIPS-backflow by default; edge erosion disabled by default |
| **Evaluation** | Patient-level ROC/AUROC and PR/AUPRC |

Core entry points:

```text
Model_Stage_1.py
Model_Stage_2.py
Train_framework.py
Inference_Pelvis_Experiments.py
ROC_Curves_Calculations.py
```

</td>
<td width="50%" valign="top">

### 🧠 Brain MRI

| | |
|---|---|
| **Domain** | Brain MRI |
| **Data workflow** | IXI healthy T1 data + fastMRI-style brain evaluation |
| **Training format** | Primarily `.npz` files with key `arr` |
| **Stage 1** | RVQ-VAE, codebook size 256 per RVQ level |
| **Stage 2** | Fact-biT with **2D RoPE** |
| **Main LPIPS reference** | Reconstruction-vs-healed/inpainted |
| **Binary fusion** | ALM-A ∪ ALM-B, optional LPIPS-backflow (disabled by default); edge erosion disabled by default |
| **Evaluation** | Patient-level ROC/AUROC |

Core entry points:

```text
Model_Stage1.py
Model_Stage_2.py
Train_framework.py
Inference_Brain_Experiments.py
ROC_Curve_Calculations.py
```

</td>
</tr>
</table>

---

## 📊 Evaluation

### Primary patient-level score

Both experiments use the same primary score definition:

```text
sum_all_bars_score = Σ_slices(Final_Binary_sum_of_anomaly_maps)
```

### Evaluation scripts

| Experiment | Script | Main role |
|---|---|---|
| 🦴 Pelvis | `Pelvis_Experiments/ROC_Curves_Calculations.py` | Patient-level ROC/AUPRC, merged ROC workflow, category analyses |
| 🧠 Brain | `Brain_Experiments/ROC_Curve_Calculations.py` | Patient-level ROC/AUROC for Brain outputs |

### Benchmark table

This front page intentionally does **not** report numeric benchmark values. Use the experiment-specific output JSON files and ROC scripts to reproduce metrics from the exact checkpoints, calibration maps, split manifests, and inference commands used in a run.

| Experiment | Primary score | Status |
|:---|:---|:---:|
| **Pelvic MRI** | `sum_all_bars_score` | Reproduce from experiment outputs |
| **Brain MRI** | `sum_all_bars_score` | Reproduce from experiment outputs |

---

## 📁 Repository Structure

```text
.
├── Difference_Between_Experiments.md
│
├── Pelvis_Experiments/
│   ├── README.md
│   ├── Model_Stage_1.py
│   ├── Model_Stage_2.py
│   ├── Train_framework.py
│   ├── dataset.py
│   ├── Inference_Pelvis_Experiments.py
│   ├── ROC_Curves_Calculations.py
│   ├── config_yaml.yaml
│   ├── Train_Val_Test_Exact_DataSplits_LUND_PROBE.json
│   ├── preslice_volumes.py
│   ├── External_dataset.py
│   ├── Simulation_inference_v4_extended_CJG.py
│   ├── Simluation_inference_v3_support_CJG.py
│   └── Pelvis_Experiments_requirements.txt
│
├── Brain_Experiments/
│   ├── README.md
│   ├── Model_Stage1.py
│   ├── Model_Stage_2.py
│   ├── Train_framework.py
│   ├── dataset.py
│   ├── Inference_Brain_Experiments.py
│   ├── ROC_Curve_Calculations.py
│   ├── config_yaml.yaml
│   ├── Train_Val_Test_Exact_DataSplits_IXI_fastMRI.json
│   ├── IXI_dataset_overview.py
│   ├── Render_patient_slices_from_csv.py
│   ├── collect_normal_slices.py
│   ├── build_patient_Global_label_folders.py
│   ├── build_patient_Local_label_folders.py
│   └── Inference_heatmaps_ideas_generator.py


```

---

## ⚙️ Reproducibility Notes

For each experiment, keep:

- [x] Exact training command(s)
- [x] Exact inference command(s)
- [x] Stage 1 and Stage 2 checkpoint paths / hashes if available
- [x] Calibration `.npz` file and calibration settings
- [x] Produced `results_v4_zscore.json`
- [x] ROC/AUPRC output JSON and figures
- [x] Train/validation/test split manifest
- [x] The exact code folder version

Important safeguards:

- Avoid slice-level leakage; use patient-level separation whenever possible.
- Keep calibration/reference data independent from anomaly evaluation cohorts.
- Preserve Pelvis `_slice_###` filename indices because they affect 3D RoPE and optional per-slice calibration lookup.
- Do not log or publish patient-identifying information in filenames, W&B runs, plots, screenshots, or JSON outputs.

---

## 🔖 Citation

A formal citation will be added when the associated manuscript/preprint is available.

```bibtex
Please visit the paper webpage to obtain the DOI. 
```

---

Made with ❤️ for the medical imaging research community


</div>
