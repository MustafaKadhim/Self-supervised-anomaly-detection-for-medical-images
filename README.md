


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
  <a href="https://github.com/MustafaKadhim/Anomaly-Detection-Self-supervised-anomaly-detection-for-medical-images/stargazers">⭐ Star this repo</a> &nbsp;•&nbsp;
  <a href="https://github.com/MustafaKadhim/Anomaly-Detection-Self-supervised-anomaly-detection-for-medical-images/pulls">🤝 Contribute</a>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake.svg">
  <img alt="github contribution grid snake animation" src="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake.svg">
</picture>

</div>











*A research-ready framework for detecting anomalies in medical images, exclusively using normal training samples.*

*A two-stage framework that learns from normal/reference MRI slices and detects deviations from the learned healthy distribution using token surprisal and perceptual healing heatmaps.*

[🚀 Quickstart](#-quickstart) · [🏗 Framework](#-framework) · [🧪 Experiments](#-experiments) · [📊 Evaluation](#-evaluation) · [📁 Repository Structure](#-repository-structure) · [🔖 Citation](#-citation)

</div>

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

Both experiments use a **Stage 1 RVQ-VAE** to learn discrete image tokens and a **Stage 2 Factorized MaskGIT / Fact-biT transformer** to model token distributions and heal masked/suspect regions.

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
│      Stage 2: Fact-biT  →  token healing                        │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│      LPIPS perceptual scoring + token surprisal scoring         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Patient-level score:                                           │
│  sum_all_bars_score = Σ_slices(token_surprisal_hot_px           │
│                                + Binary_Sum_Heatmap)            │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ROC / AUROC and PR / AUPRC                     │
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

The primary per-slice CORE fields are:

| Field | Meaning |
|---|---|
| `Binary_Sum_Heatmap` | Binary/perceptual healing heatmap contribution |
| `token_surprisal_hot_px` | Token-surprisal hot-pixel contribution |

Both cleaned experiments aggregate them as:

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
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
cd Pelvis_Experimentss
pip install -r Pelvis_Experiments_requirements.txt
```

Start with:

```text
Pelvis_Experimentss/README_updated.md
```

</td>
<td width="50%" valign="top">

#### 🧠 Brain MRI

The Brain folder README lists the required scientific Python packages:

```text
Brain_Experiments/README_updated.md
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
| **Stage 2** | Factorized MaskGIT / Fact-biT with **3D RoPE** |
| **Main LPIPS reference** | Input-vs-healed/inpainted |
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
| **Stage 2** | Factorized MaskGIT with **2D RoPE** |
| **Main LPIPS reference** | Reconstruction-vs-healed/inpainted |
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

The latest cleaned Pelvis and Brain READMEs both document the same primary score definition:

```text
sum_all_bars_score = Σ_slices(token_surprisal_hot_px + Binary_Sum_Heatmap)
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
├── Front_page_README.md
├── Difference_Between_Experiments.md
│
├── Pelvis_Experimentss/
│   ├── README_updated.md
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
│   ├── README_updated.md
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
│
├── Final_Code_Phiro_Pelvic_MRI/       # Previous/final Phiro pelvic code snapshot
├── Final_Code_Phiro_Brain_MRI/        # Previous/final Phiro brain code snapshot
└── zz_Gammalt_*/                      # Older archived development folders
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
@misc{kadhim_two_stage_mri_anomaly_detection,
  title  = {Two-Stage Unsupervised Anomaly Detection for MRI},
  author = {Kadhim, Mustafa and collaborators},
  note   = {Research code; citation details to be updated upon publication}
}
```

---

Made with ❤️ for the medical imaging research community


</div>
