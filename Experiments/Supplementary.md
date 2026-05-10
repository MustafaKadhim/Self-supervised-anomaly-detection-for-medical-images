# Supplementary Information

## Unsupervised Anomaly Detection in Medical MRI via Residual Vector-Quantised Tokenisation and Masked Token Prediction

---

## Table of Contents

1. [Framework Overview and Design Rationale](#1-framework-overview-and-design-rationale)
2. [Stage 1 — Residual Vector-Quantised Variational Autoencoder](#2-stage-1--residual-vector-quantised-variational-autoencoder)
3. [Stage 2 — Factorised Masked Token Predictor](#3-stage-2--factorised-masked-token-predictor)
4. [Inference — Recursive AutoMask Scoring](#4-inference--recursive-automask-scoring)
5. [Data Augmentation](#5-data-augmentation)
6. [Domain-Specific Adaptations: Pelvic and Brain MRI](#6-domain-specific-adaptations-pelvic-and-brain-mri)
7. [Data Preprocessing and Patient-Level Splits](#7-data-preprocessing-and-patient-level-splits)
8. [Evaluation Methodology](#8-evaluation-methodology)
9. [Supplementary Tables](#9-supplementary-tables)

---

## 1. Framework Overview and Design Rationale

Detecting anomalies in medical images without annotated pathology is a long-standing challenge. Supervised approaches require curated labels that are expensive, inconsistent across institutions, and inherently tied to known disease categories — making them poorly suited to open-world detection where the lesion type is unknown in advance. Our framework sidesteps this limitation entirely: it is trained exclusively on images from healthy individuals and learns, in an unsupervised manner, what normal anatomy looks like at both the structural and textural level.

The core intuition is that a model trained to represent and predict normal anatomy will, if it is doing its job, struggle to represent anomalous anatomy faithfully. By deliberately withholding portions of an image from the model and measuring how surprised it is by what lies beneath the mask — and separately measuring how well it can restore those regions to their expected normal appearance — we obtain two complementary and interpretable signals of abnormality. Neither signal requires a pathology label, a threshold tuned on diseased data, or any prior assumption about the type of anomaly.

The framework is organised into two sequential stages:

1. **Stage 1 (RVQ-VAE):** A Vision Transformer encoder compresses each 256 × 256 MRI slice into a compact grid of discrete tokens through residual vector quantisation. A convolutional decoder with pixel-shuffle upsampling reconstructs the image from these tokens, guided by a biomedical perceptual loss that steers the codebook towards clinically meaningful representations.

2. **Stage 2 (Fact-BiT):** A bidirectional transformer-based masked token predictor learns the joint distribution of token sequences arising from healthy anatomy. At inference time, masked subsets of tokens are filled in by the model; discrepancies between predicted and observed token distributions — measured as surprisal — localise regions the model considers unlikely given its prior over normal anatomy.

Both stages are trained on healthy data only. Anomaly scoring involves two parallel branches — **token surprisal** and **perceptual healing** — whose outputs are fused into a single patient-level score used for receiver operating characteristic (ROC) analysis.

---

## 2. Stage 1 — Residual Vector-Quantised Variational Autoencoder

### 2.1 Encoder Architecture

Each two-dimensional MRI slice (256 × 256, single channel) is projected into a sequence of non-overlapping patch tokens via a learned patch embedding (convolutional projection, patch size 8 × 8). This produces a spatial grid of 32 × 32 = 1 024 tokens, each residing in an embedding space of dimension 256.

The token sequence is processed by a Vision Transformer (ViT) encoder comprising 8 transformer layers with 8 attention heads and a dropout rate of 0.1, with learned positional embeddings. The encoder maps each image to a dense feature map preserving the 32 × 32 spatial structure.

### 2.2 Residual Vector Quantisation Module

The encoded feature map is passed through a **Residual Vector Quantisation (RVQ)** module with two quantisation levels. The first level captures coarse structural content; the second captures finer texture. Each level maintains an independently learned codebook and assigns each spatial position to its nearest codebook entry by Euclidean distance. Codebook entries are updated via exponential moving average (EMA, decay 0.85) rather than gradient descent, which stabilises training and avoids codebook collapse. An orthogonal regularisation penalty (weight 0.10) further encourages codebook diversity, and entries that fall below an EMA activation threshold are re-initialised to prevent dead codes. The commitment loss (weight 0.25) is applied with a stop-gradient on the quantised targets, encouraging encoder outputs to stay close to the codebook entries without updating the codebook through backpropagation.

The codebook size per level is set to 192 for the pelvic model and 256 for the brain model, reflecting differences in anatomical diversity across imaging domains (see Section 6).

### 2.3 Decoder Architecture

The decoder reverses the encoding process through three sequential pixel-shuffle upsampling blocks, each doubling spatial resolution. The channel progression is as follows: a stem convolution processes the quantised features at 512 channels, followed by three residual blocks (512 channels, GroupNorm with 8 groups, SiLU activations). Three upsampling stages then progressively reduce the channel count from 512 to 256, 256 to 128, and 128 to 64, each followed by a SiLU activation. A final convolutional head (64 → 1 channel) produces the grayscale reconstruction at 256 × 256 resolution. For the brain model, reconstruction values are clamped to [−3, 3] to match the input normalisation range; for the pelvic model, the decoder output is unclamped.

The **quantisation error map** — the spatial distribution of L2 distances between encoded features and their assigned codebook vectors — is available as a secondary diagnostic signal but is not incorporated into the primary anomaly score.

### 2.4 Training Objective

Stage 1 is trained end-to-end using a composite loss:

$$\mathcal{L}_{\text{Stage1}} = \mathcal{L}_{\text{L1}} + \lambda_{\text{perc}} \cdot \mathcal{L}_{\text{BiomedCLIP}} + \mathcal{L}_{\text{commit}}$$

- **Pixel reconstruction loss** $\mathcal{L}_{\text{L1}}$: Mean absolute error between input and reconstructed image, providing a direct pixel-level fidelity constraint.

- **Perceptual loss** $\mathcal{L}_{\text{BiomedCLIP}}$: Feature-space cosine distance computed using the pooled output of a BiomedCLIP Vision Transformer pretrained on biomedical image–text pairs. The perceptual network is fully frozen during Stage 1 training and not updated at any point. Inputs are preprocessed to 224 × 224 and normalised with ImageNet statistics before feature extraction. The loss is computed as the mean cosine distance (1 − cosine similarity) between L2-normalised feature vectors of the reconstruction and target. This loss steers the codebook towards representations that align with human-meaningful biomedical features rather than low-level pixel statistics. The perceptual weight $\lambda_{\text{perc}}$ is 0.9 for pelvic MRI and 0.5 for brain MRI, reflecting the greater importance of fine structural boundaries in the pelvis.

- **Commitment loss** $\mathcal{L}_{\text{commit}}$ (weight 0.25): Encourages encoder outputs to remain close to their assigned codebook entries, with a stop-gradient applied to the quantised targets.

### 2.5 Training Protocol

Stage 1 is trained using AdamW (β₁ = 0.9, β₂ = 0.95, weight decay 10⁻⁴) with cosine annealing over the full training duration (T_max = 100 epochs, η_min = 0). Training runs for a maximum of 100 epochs; the top three checkpoints ranked by validation loss are retained for downstream use, without early stopping. Full 32-bit precision is used throughout. Both models were trained on a single GPU.

---

## 3. Stage 2 — Factorised Masked Token Predictor

### 3.1 Motivation

Once Stage 1 has compressed each image into a discrete token grid, the modelling task becomes learning the joint distribution of token sequences arising from healthy anatomy. A transformer trained on this task learns rich spatial dependencies — which structural configurations are likely, which juxtapositions of texture tokens are common — without any explicit supervision. At inference time, a token that is unlikely given its spatial context is a candidate anomaly signal.

We adopt a MaskGIT-style training objective: a random subset of token positions is masked, and the model is trained to predict the masked tokens from the remaining context via cross-entropy. This approach has two advantages over autoregressive modelling. First, bidirectional attention allows every unmasked token to inform every masked prediction, naturally encoding the holistic spatial structure of anatomy. Second, at inference time any arbitrary masking pattern can be applied, allowing systematic exploration of local and regional spatial contexts.

### 3.2 Architecture

The Fact-BiT transformer receives as input the full 1 024-token grid from Stage 1, with a configurable fraction of positions replaced by a learned mask token. A **task embedding** (two entries: Level 1 or Level 2 prediction) is added to all positions, enabling the same model weights to predict tokens from either quantisation level. The forward pass produces two parallel prediction heads — one per quantisation level — each projecting the transformer hidden state to a distribution over codebook entries.

The backbone uses 8 transformer layers with 8 attention heads, no dropout (the Stage 2 transformer is applied without dropout regularisation), and RMSNorm layer normalisation. Feedforward layers employ **SwiGLU** activations (Swish-gated linear units), which provide improved gradient flow compared with ReLU-family alternatives. Attention is computed via PyTorch's Scaled Dot-Product Attention kernel with Flash Attention support when available. The hidden dimension is 256 throughout, matching the Stage 1 embedding dimension.

### 3.3 Positional Encoding: 2D and 3D Rotary Position Embeddings

Positional encoding is a critical design choice because MRI slices have explicit spatial relationships that differ fundamentally from the sequential structure of language tokens.

For **brain MRI**, where each slice is analysed as an independent two-dimensional image, we use **2D Rotary Position Embeddings (RoPE)** that encode the row and column indices of the 32 × 32 token grid (rope_base = 25 000). RoPE applies frequency-domain rotations to query and key vectors, encoding relative positions directly into the attention mechanism rather than as additive biases, which is robust to the variable slice positioning that arises from different MRI protocols.

For **pelvic MRI**, anatomical context extends meaningfully across the axial stack: pelvic organs and surrounding musculature shift in predictable ways from superior to inferior slices. To exploit this inter-slice structure, we extend RoPE to a **3D formulation** that encodes the row index, column index, and axial slice index simultaneously (rope_max_positions = 64, rope_max_slices = 92, rope_base = 25 000), with each spatial dimension occupying one-third of the rotary embedding dimension. The slice index is extracted from a standardised naming convention and passed as an explicit conditioning argument during both training and inference. This 3D RoPE enables the transformer to model inter-slice consistency, allowing it to flag regions that are anomalous not just in-plane but relative to the expected volumetric anatomy at that axial position.

### 3.4 Masking Strategy

Training employs a **mixed masking schedule** that combines two complementary masking modes, each sampled with equal probability:

- **Block masking (50%):** Structured 2 × 2 or 4 × 4 spatial block patterns are applied to the token grid. These patterns enforce that the model cannot simply copy neighbouring tokens into masked positions but must reason about broader spatial context to reconstruct the missing structure.

- **Random masking (50%):** Token positions are independently masked with a probability drawn from a Beta(4, 4) distribution clipped to [0.15, 0.75], producing smoothly varying amounts of context at training time and preventing the model from over-fitting to any single masking density.

The combination ensures that the model is equally competent at predicting tokens from dense spatial context and from sparse long-range cues — both scenarios that arise in the inference pipeline.

### 3.5 Training Objective

The training loss for Stage 2 is a weighted cross-entropy over both quantisation levels, with label smoothing (ε = 0.05) to prevent overconfident predictions:

$$\mathcal{L}_{\text{Stage2}} = \mathcal{L}_{\text{CE}}^{(L1)} + 0.25 \cdot \mathcal{L}_{\text{CE}}^{(L2)}$$

The lower weight on Level 2 (texture tokens) reflects that texture is harder to predict precisely from context and contributes proportionally less signal to the joint distribution model.

### 3.6 Training Protocol

Stage 2 is trained with Stage 1 weights fully frozen. AdamW (β₁ = 0.9, β₂ = 0.98, weight decay 0.01) is used with a linear warmup over the first 2 000 gradient steps followed by cosine annealing to the end of training. Learning rates are 10⁻⁴ for the pelvic model and 2 × 10⁻⁴ for the brain model (see Table S2). Training runs for a maximum of 100 epochs; the top three checkpoints by validation loss are retained without early stopping.

For pelvic MRI, only slices within the anatomically informative axial range (indices 30–60) are included in Stage 2 training, ensuring that the token distribution model is not diluted by near-empty slices at the superior and inferior extents of the volume.

---

## 4. Inference — Recursive AutoMask Scoring

Anomaly detection at inference time involves two parallel scoring branches: **token surprisal** and **perceptual healing**. Both operate on each two-dimensional slice and produce a per-slice anomaly signal; patient-level scores are obtained by aggregating across all slices.

### 4.1 Offline Calibration on Healthy Reference Images

Before evaluating any test case, a calibration pass is performed on a held-out set of healthy reference slices drawn exclusively from the training distribution. For each reference slice:

1. Stage 1 encodes the image to Level 1 and Level 2 token grids.
2. Stage 2 heals the image under the domain-specific checkerboard mask patterns (pelvic model: 2 × 2 block patterns; brain model: 4 × 4 block pattern), producing a healed reconstruction.
3. The perceptual distance (Learned Perceptual Image Patch Similarity, LPIPS, with VGG backbone) is computed, yielding a per-pixel heatmap.
4. Spatial smoothing is applied to the heatmap via average pooling (kernel: 15 pixels for the pelvic model, 7 pixels for the brain model).

This process accumulates — at every pixel position — a distribution of LPIPS values from healthy reference subjects and slices. The per-pixel mean μ and standard deviation σ are stored in a calibration map. When sufficient samples are available per axial slice position (≥ 3), per-slice statistics are computed; otherwise global statistics are used. This calibration map encodes the baseline perceptual reconstruction variability at every spatial location for normal anatomy, and the smoothing kernel must be identical between calibration and inference.

### 4.2 Token Surprisal Branch

For each test slice, a statistical probe of the Stage 2 token distribution is performed:

1. Stage 1 encodes the test image to Level 1 tokens.
2. A random masking pattern (mask ratio: 0.90 for the pelvic model, 0.82 for the brain model) is applied to the Level 1 token grid, and Stage 2 predicts the masked positions.
3. The **surprisal** of the true observed token at each masked position is computed as the negative log-probability assigned to it by the model: $s_i = -\log p(\hat{t}_i = t_i^{\ast} \mid \text{context})$, where $t_i^{\ast}$ is the true token at position $i$.
4. Steps 2–3 are repeated $N$ times with independent random masks (50 repetitions for the pelvic model, 100 for the brain model), and surprisal values are averaged across repetitions at each spatial position.
5. Surprisal values at or below a domain-specific threshold (8.0 for the pelvic model, 5.0 for the brain model) are zeroed, retaining only positions that the model finds genuinely surprising.
6. The surprisal map is upsampled from the 32 × 32 token grid to pixel space (256 × 256).

The resulting binary anomaly map (ALM-B) records positions where the token surprisal exceeds the domain threshold. This branch is sensitive to structural anomalies that alter token identity — masses, resection cavities, or implanted spacers that produce token sequences never encountered in healthy training data.

### 4.3 Perceptual Healing Branch

The healing branch exploits the model's ability to inpaint masked regions:

1. Stage 1 encodes the test image to Level 1 and Level 2 token grids.
2. A checkerboard mask pattern is applied; Stage 2 fills the masked positions iteratively over 6 healing steps. At each step, token predictions are generated, and the most confident positions are committed to the reconstructed grid, with a healing softmax temperature controlling the sharpness of the prediction distribution.
3. The full healed token grid (observed unmasked tokens combined with predicted masked tokens) is decoded by Stage 1 back to pixel space, producing a **healed reconstruction**.
4. **Test-time augmentation:** The same procedure is applied to the horizontally flipped image. For the pelvic model, original and flipped LPIPS heatmaps are combined by geometric mean; for the brain model, they are combined by arithmetic mean.
5. The LPIPS heatmap undergoes spatial smoothing with the domain-specific average-pooling kernel, matching the kernel used at calibration.
6. **Z-score normalisation:** Each pixel's heatmap value is transformed to $z = (h - \mu) / (\sigma + \epsilon)$ using the per-pixel calibration statistics from Section 4.1.
7. Pixels with $z$ outside the domain-specific threshold bounds are assigned to the healing-derived binary anomaly mask (ALM-A): a one-sided upper bound of $z > 2.0$ for the pelvic model, and a two-sided interval (lower = −2.5, upper = 6.0) for the brain model.

The LPIPS reference differs between domains. For **pelvic MRI**, LPIPS is computed between the original input image and the healed reconstruction, measuring how much the healed version diverges from what was observed — a direct indicator of local anatomical implausibility. For **brain MRI**, LPIPS is computed between the Stage 1 reconstruction (before healing) and the healed reconstruction, which isolates the change introduced specifically by the healing process and is less sensitive to global intensity differences inherent to brain MRI protocols.

For the pelvic model, a supplementary **backflow** refinement step is applied by default: pixels with LPIPS values above the 99th percentile (computed during healing) are additionally included in ALM-A, providing a secondary mechanism to capture regions with systematically high perceptual error that may fall below the primary Z-score threshold.

### 4.4 Binary Mask Fusion and Patient-Level Scoring

The final per-slice binary anomaly map is the union of the healing mask (ALM-A, §4.3) and the token surprisal mask (ALM-B, §4.2):

$$\text{ALM}_{\text{final}} = \text{ALM-A} \cup \text{ALM-B}$$

The **patient-level anomaly score** is obtained by summing the count of detected pixels in the final binary mask across all evaluated slices:

$$S_{\text{patient}} = \sum_{\text{slices}} \left| \text{ALM}_{\text{final}} \right|$$

This additive formulation is intentional: both branches capture complementary aspects of abnormality — distributional surprise versus perceptual reconstruction discrepancy — and their union leverages both signals without requiring hand-tuned fusion weights. Patients with higher $S_{\text{patient}}$ are ranked as more anomalous.

---

## 5. Data Augmentation

Augmentation is applied separately at two levels: within the data loading pipeline (DataModule augmentation, applied per mini-batch during training) and within the Stage 1 model's own training step (internal model augmentation). These two augmentation sources are independent and may be applied in combination.

### 5.1 Brain MRI Augmentation

Brain MRI data exhibit substantial scanner- and protocol-induced intensity variability across the IXI and fastMRI cohorts. To prevent the Stage 1 codebook from treating scanner noise or intensity drift as distinctive anatomical features, a richer augmentation strategy is applied.

**DataModule augmentation (applied per training batch when enabled):**

| Transform | Parameters |
|---|---|
| Random intensity scaling | Scale factor ± 0.10, probability 0.33 |
| Random contrast adjustment | γ ∈ [0.5, 1.5], probability 0.33 |
| Random Gaussian noise | σ = 0.30, probability 0.50 |
| Random affine — rotation | ± 15°, probability 0.33 |
| Random affine — translation | ± 15 pixels (both axes), probability 0.33 |
| Random affine — zoom | [0.80, 1.20], probability 0.33 |
| Random horizontal flip | Probability 0.50 |

**Internal model augmentation (applied within Stage 1 training step):**

The same five augmentation families (intensity scaling, contrast, Gaussian noise, affine, horizontal flip) are available as an internal model augmentation module, with parameters matching the DataModule configuration. In the brain model's training procedure, the DataModule augmentation is the primary augmentation source; when the DataModule pipeline is active, the internal model augmentation block is disabled to avoid stacking identical transforms.

### 5.2 Pelvic MRI Augmentation

Pelvic MRI data from the LUND-PROBE cohort were acquired under a standardised institutional protocol, resulting in substantially lower inter-subject intensity variability than the multi-scanner brain cohort. Augmentation is accordingly more conservative, targeting realistic within-protocol geometric variation only.

**Internal model augmentation (applied within Stage 1 training step):**

| Transform | Parameters |
|---|---|
| Random intensity scaling | Probability 0.33 |
| Random affine — rotation | ± 5°, probability 0.33 |
| Random affine — translation | ± 5 pixels (horizontal axis only), probability 0.33 |

No contrast adjustment, Gaussian noise injection, zoom, or horizontal flip is applied during pelvic model training. The pelvic DataModule augmentation, when enabled, adds only a horizontal flip (probability 0.50) and a small affine rotation (± 5°, probability 0.30); noise and contrast transforms are absent.

### 5.3 Rationale for Asymmetric Augmentation

The asymmetry in augmentation intensity is a deliberate methodological choice. The objective of Stage 1 training is to build a codebook that reliably represents the normal anatomy of a specific imaging domain. For the brain, where scanner field strength, coil configuration, and reconstruction protocol vary considerably across subjects, richer augmentation is needed to ensure that the learned token distribution reflects anatomy rather than acquisition artefacts. For the pelvis, where the acquisition is tightly controlled, aggressive augmentation risks teaching the codebook to be invariant to genuine anatomical variation that should be encoded as distinct tokens, reducing the discriminative power of the token surprisal branch at inference time.

---

## 6. Domain-Specific Adaptations: Pelvic and Brain MRI

While the two-stage framework, patient-level scoring formula, and core algorithmic pipeline are identical across both anatomical domains, several implementation choices were calibrated to the characteristics of each imaging context. These are not independent architectural variants but principled adaptations of the same design.

### 6.1 Codebook Size and Perceptual Loss Weight

The pelvic model uses 192 codebook entries per quantisation level, reflecting the anatomically constrained imaging field (bladder, uterus/prostate, rectum, and surrounding musculature) acquired under a standardised protocol. The brain model uses 256 entries, appropriate for the greater morphological diversity of cortical and subcortical structures across a multi-scanner cohort. The perceptual loss weight is higher for pelvic MRI (0.9 versus 0.5), reflecting the greater diagnostic importance of fine structural boundaries — such as the capsule of the prostate or the uterine wall — where biomedical perceptual alignment during training yields a more informative codebook.

### 6.2 Positional Encoding Dimensionality

The most architecturally distinctive difference between the two models is the dimensionality of the Stage 2 positional encoding. Pelvic anatomy exhibits strong and predictable axial organisation: the prostate base lies superior to the apex, the uterine fundus superior to the cervix, and each pelvic structure occupies a characteristic range of axial positions. The 3D RoPE encodes this prior directly into the attention mechanism, enabling the Stage 2 model to leverage inter-slice consistency as an additional context cue — an anomalous slice that disrupts the expected axial progression is rated as more surprising than a slice that is locally unusual but consistent with its volumetric neighbourhood.

Brain MRI is processed as independent 2D slices, suited to the fastMRI evaluation workflow where slice thickness, spacing, and orientation vary considerably across subjects. The 2D RoPE encodes only in-plane spatial relationships, making the model robust to this protocol heterogeneity.

### 6.3 LPIPS Reference and Decoder Clamping

The perceptual healing branch uses different reference images for the LPIPS computation in each domain. For the pelvic model, LPIPS is computed between the original input image and the healed reconstruction — quantifying how much the healed output deviates from the observed signal. For the brain model, LPIPS is computed between the Stage 1 reconstruction (before healing) and the healed reconstruction — isolating the change introduced specifically by the healing process, rather than any discrepancy between the model's compressed representation and the raw input. This choice is well-suited to the brain domain, where global intensity differences across scanners would inflate input-vs-healed LPIPS regardless of local anomaly content.

Correspondingly, the Stage 1 decoder output for the brain model is clamped to [−3, 3], matching the input normalisation range used during IXI preprocessing. The pelvic decoder output is not clamped. This means that the calibration maps for the two models are derived from distributions with different dynamic ranges and should not be interchanged.

### 6.4 Healing Mask Patterns and Temperatures

The checkerboard masking pattern used during healing differs between domains. The pelvic model applies two complementary 2 × 2 block checkerboard masks (covering all positions across both passes), which is well-suited to the moderate-resolution anatomical detail in pelvic MRI. The brain model applies a single 4 × 4 block checkerboard mask, whose coarser structure is better matched to the spatial scale of relevant brain pathology (masses, resection cavities, and perilesional oedema) and the higher spatial frequency content of brain MRI.

The healing softmax temperature (0.3 for pelvis, 0.9 for brain) and the inpainting temperature (0.3 for pelvis, 0.5 for brain) reflect these domain differences: the lower pelvic temperatures produce sharper, more deterministic token predictions, appropriate for a dataset with less acquisition variability, while the higher brain temperatures preserve more uncertainty in the prediction, which is important for downstream surprisal estimation in a heterogeneous multi-scanner setting.

### 6.5 Token Surprisal Sampling and Thresholding

Token surprisal sampling is doubled for the brain model (100 versus 50 Monte Carlo repetitions) to reduce variance in the surprisal estimate, which is particularly important for the heterogeneous fastMRI anomaly cohort where single-pass estimates may be noisy. The mask ratio used for surprisal sampling is higher for the pelvic model (0.90 versus 0.82), ensuring that a greater fraction of tokens is probed in each pass; this is appropriate given the lower contextual ambiguity of the more uniform pelvic anatomy.

The NLL clamp threshold — which zeroes low-surprisal positions before binarisation — is higher for the pelvic model (8.0 versus 5.0), reflecting the lower baseline token entropy expected from a model trained on less variable anatomy.

### 6.6 LPIPS Backflow Refinement

For the pelvic model, a backflow refinement step is enabled by default: regions with LPIPS values at or above the 99th percentile of the per-slice healing heatmap are additionally included in the binary anomaly mask. This provides a secondary pathway to capture high-perceptual-error regions that may not reach the primary Z-score threshold. For the brain model, this backflow step is disabled by default to avoid inflating false positives in acquisitions with spatially varying intensity artefacts.

### 6.7 Training Slice Restriction

During Stage 2 training of the pelvic model, only slices within the anatomically informative axial range (indices 30–60) are included. This focuses the token distribution model on the pelvic floor, bladder, and reproductive organ regions and avoids diluting the learned distribution with near-empty slices at the superior and inferior extent of the imaging volume. No equivalent restriction is applied to the brain model, as IXI preprocessing retains only the informative axial range (slices 128–188) at the preprocessing stage.

---

## 7. Data Preprocessing and Patient-Level Splits

### 7.1 Pelvic MRI (LUND-PROBE Dataset)

Raw MRI volumes are Z-score normalised per volume (zero mean, unit variance). Each axial slice is saved as a two-dimensional array with a standardised naming convention that encodes the patient identifier and the axial slice index (zero-padded to three digits). This naming convention is required by the Stage 2 3D RoPE and by the per-slice calibration lookup, both of which read the slice index directly from the identifier. During training, slices are loaded, rotated 90° clockwise, resized to 320 × 320 using area interpolation, and centre-cropped to 256 × 256.

Patient-level train/validation/test splits are stored in a dedicated split manifest. All splitting is performed at the patient level prior to any preprocessing, ensuring strict separation between sets. No patient appears in more than one partition. A random seed-based split is used at runtime (seed = 42, 10% validation fraction).

Synthetic anomalies — MRI reconstruction artefacts simulating random ghosting, noise injection, spike artefacts, random motion, and whole-image Gaussian blurring — are generated on held-out test patients only and are never seen during training. Clinical anomaly cases (implanted spacers, clinical anatomical variations, and cases of unknown aetiology) are evaluated on a separate held-out cohort.

### 7.2 Brain MRI (IXI + fastMRI Datasets)

**Healthy training data (IXI):** T1-weighted volumes are reoriented to the closest canonical orientation, Z-score normalised per volume, and intensity-clipped to [−3, 3]. Axial slices within the range 128–188 are retained, as these contain informative brain parenchyma. A 90° counter-clockwise rotation is applied at the preprocessing stage to align the IXI T1 orientation convention with the fastMRI evaluation workflow. Each slice is saved as a two-dimensional float32 array, already at 256 × 256 resolution; no further resizing or cropping is applied during training data loading.

**Anomaly evaluation data (fastMRI):** Reconstructed MRI volumes are rendered to 256 × 256 two-dimensional slices. Annotation bounding boxes, where available, are stored alongside the slice data and used for localisation evaluation only — they are never used for training or for computing the primary patient-level AUROC.

Pre-separated training and validation directories are used for the brain model, rather than a runtime random split. Patient-level splits are recorded in a dedicated split manifest consistent with the directory organisation.

---

## 8. Evaluation Methodology

### 8.1 Patient-Level AUROC

All performance results are reported at the **patient level**. Each patient receives a single anomaly score $S_{\text{patient}}$ (§4.4). Binary labels are assigned as follows: patients from the healthy cohort receive label 0, and patients from any anomaly cohort receive label 1. The receiver operating characteristic (ROC) curve is computed from these patient-level scores and labels using the trapezoidal rule, and the area under the curve (AUROC) is the primary performance metric.

Patient-level reporting is a deliberate choice that avoids the statistical inflation inherent in slice-level evaluation, where treating each slice of a single patient as an independent observation artificially inflates the effective sample size and can mask patient-level performance heterogeneity.

### 8.2 Bootstrap Confidence Intervals

Uncertainty in AUROC estimates is quantified via **stratified bootstrap resampling**: bootstrap samples are drawn by resampling separately from the healthy and anomaly cohorts (with replacement, maintaining the original class proportions). A minimum of 2 000 bootstrap samples is used. The 95% confidence interval is reported as the 2.5th and 97.5th percentiles of the bootstrap AUROC distribution. This stratified approach avoids bootstrap samples containing only a single class and reflects uncertainty in both sensitivity and specificity.

### 8.3 Category-Stratified Sensitivity Analysis

Beyond global AUROC, sensitivity at fixed false positive rates (FPR = 0.05, 0.10, 0.20) is reported separately for each anomaly category. For pelvic MRI, categories include synthetic artefact types (random ghosting, random noise, random spike, random motion, whole-image Gaussian) and clinical anomaly groups (implanted spacers, clinical variations, and cases of unknown aetiology). For brain MRI, anomalies are stratified into global (study-level) and local (per-slice) labels from the fastMRI clinical annotation metadata.

### 8.4 Precision–Recall Analysis

AUPRC (area under the precision–recall curve) is reported as a secondary metric for pelvic MRI evaluation, where it complements the AUROC in characterising the trade-off between detection sensitivity and false discovery rate under class imbalance. The AUPRC is computed using right-step integration over the precision–recall curve. For brain MRI, AUROC is the primary reported metric; AUPRC may be computed separately as a supplementary sensitivity analysis.

### 8.5 Localisation Evaluation (Brain MRI)

For the subset of fastMRI patients with bounding-box annotations, the spatial correspondence between the predicted binary anomaly mask and the annotated region is evaluated. Per-slice metrics include the fraction of annotated bounding boxes for which at least one detected pixel falls inside the box (recall at the box level), pixel-level precision (fraction of detected pixels inside any annotated box), and F1 score. These localisation metrics are reported as exploratory findings and are not used for model selection or primary performance comparison. Annotations and bounding boxes are never used during training or calibration.

---

## 9. Supplementary Tables

### Table S1 — Stage 1 (RVQ-VAE) Architecture and Training Hyperparameters

| Hyperparameter | Pelvic MRI | Brain MRI |
|---|---|---|
| Input resolution | 256 × 256 | 256 × 256 |
| Patch size | 8 × 8 | 8 × 8 |
| Token grid size | 32 × 32 (1 024 tokens) | 32 × 32 (1 024 tokens) |
| Encoder depth | 8 transformer layers | 8 transformer layers |
| Attention heads | 8 | 8 |
| Embedding dimension | 256 | 256 |
| Encoder attention dropout | 0.1 | 0.1 |
| RVQ levels | 2 (structure + texture) | 2 (structure + texture) |
| Codebook size (per level) | 192 | 256 |
| EMA decay | 0.85 | 0.85 |
| Orthogonal regularisation weight | 0.10 | 0.10 |
| Commitment cost (stop-gradient) | 0.25 | 0.25 |
| Decoder upsampling stages | 3 (8× total) | 3 (8× total) |
| Decoder channel progression | 512 → 256 → 128 → 64 → 1 | 512 → 256 → 128 → 64 → 1 |
| Decoder output clamping | None (unclamped) | [−3, 3] |
| Reconstruction loss | L1 | L1 |
| Perceptual loss backbone | BiomedCLIP ViT (frozen) | BiomedCLIP ViT (frozen) |
| Perceptual feature | Pooled global representation | Pooled global representation |
| Perceptual loss function | Cosine distance (1 − cos sim) | Cosine distance (1 − cos sim) |
| Perceptual loss weight (λ) | 0.9 | 0.5 |
| Optimiser | AdamW | AdamW |
| Learning rate | 1 × 10⁻⁴ | 2 × 10⁻⁴ |
| β₁, β₂ | 0.9, 0.95 | 0.9, 0.95 |
| Weight decay | 1 × 10⁻⁴ | 1 × 10⁻⁴ |
| LR scheduler | Cosine annealing (T_max = 100, η_min = 0) | Cosine annealing (T_max = 100, η_min = 0) |
| Training precision | Full (32-bit) | Full (32-bit) |
| Batch size | 128 | 192 |
| Maximum epochs | 100 | 100 |
| Early stopping | None (top-3 checkpoints by val. loss retained) | None (top-3 checkpoints by val. loss retained) |
| Validation fraction | 10% (random split, seed = 42) | Pre-separated validation directory |
| DataModule: rotation | 90° clockwise at load time | 90° counter-clockwise at preprocessing save time |
| DataModule: resize + crop | Resize to 320 × 320 (area), centre-crop to 256 × 256 | None (images pre-saved at 256 × 256) |
| DataModule: augmentation (rotation) | ± 5° (prob. 0.33) | ± 15° (prob. 0.33) |
| DataModule: augmentation (translation) | Not applied | ± 15 pixels, both axes (prob. 0.33) |
| DataModule: augmentation (zoom) | Not applied | [0.80, 1.20] (prob. 0.33) |
| DataModule: augmentation (flip) | Horizontal (prob. 0.50) | Horizontal (prob. 0.50) |
| DataModule: augmentation (contrast) | Not applied | γ ∈ [0.5, 1.5] (prob. 0.33) |
| DataModule: augmentation (noise) | Not applied | σ = 0.30 (prob. 0.50) |

---

### Table S2 — Stage 2 (Fact-BiT) Architecture and Training Hyperparameters

| Hyperparameter | Pelvic MRI | Brain MRI |
|---|---|---|
| Positional encoding | 3D RoPE (row, column, axial slice) | 2D RoPE (row, column) |
| RoPE base frequency | 25 000 | 25 000 |
| RoPE max positions | 64 (spatial); 92 (slice) | 33 |
| Transformer depth | 8 layers | 8 layers |
| Attention heads | 8 | 8 |
| Embedding dimension | 256 | 256 |
| Attention mechanism | Flash Attention (SDPA) | Flash Attention (SDPA) |
| Feedforward activation | SwiGLU | SwiGLU |
| Layer normalisation | RMSNorm | RMSNorm |
| Dropout (Stage 2 transformer) | None | None |
| Task embedding entries | 2 (Level 1, Level 2) | 2 (Level 1, Level 2) |
| Masking strategy | Mixed: 50% block, 50% random | Mixed: 50% block, 50% random |
| Block mask shapes | 2 × 2, 4 × 4 | 2 × 2, 4 × 4 |
| Random mask ratio distribution | Beta(4, 4), clipped [0.15, 0.75] | Beta(4, 4), clipped [0.15, 0.75] |
| L1 cross-entropy loss weight | 1.0 | 1.0 |
| L2 cross-entropy loss weight | 0.25 | 0.25 |
| Label smoothing | 0.05 | 0.05 |
| Optimiser | AdamW | AdamW |
| Learning rate | 1 × 10⁻⁴ | 2 × 10⁻⁴ |
| β₁, β₂ | 0.9, 0.98 | 0.9, 0.98 |
| Weight decay | 0.01 | 0.01 |
| LR scheduler | Linear warmup (2 000 steps) + cosine annealing | Linear warmup (2 000 steps) + cosine annealing |
| LR schedule interval | Per gradient step | Per gradient step |
| Training precision | Full (32-bit) | Full (32-bit) |
| Batch size | 128 | 158 |
| Maximum epochs | 100 | 100 |
| Stage 1 weights during Stage 2 training | Fully frozen | Fully frozen |
| Training slice range (axial) | Indices 30–60 (within-batch filter) | Determined by preprocessing (indices 128–188) |

---

### Table S3 — Inference Hyperparameters

| Hyperparameter | Pelvic MRI | Brain MRI |
|---|---|---|
| Healing mask pattern | 2 × 2 block checkerboard (two complementary passes) | 4 × 4 block checkerboard (single pass) |
| Healing steps | 6 | 6 |
| Healing softmax temperature | 0.3 | 0.9 |
| Inpainting softmax temperature | 0.3 | 0.5 |
| Test-time augmentation | Horizontal flip; heatmaps combined by geometric mean | Horizontal flip; heatmaps combined by arithmetic mean |
| LPIPS reference (healing branch) | Original input vs. healed reconstruction | Stage 1 reconstruction vs. healed reconstruction |
| LPIPS backflow refinement | Enabled by default (99th percentile threshold) | Disabled by default |
| Smoothing kernel | 15-pixel average pooling | 7-pixel average pooling |
| Z-score threshold (ALM-A) | > 2.0 (one-sided upper) | Outside [−2.5, 6.0] (two-sided) |
| LPIPS binary threshold (ALM-A) | 0.60 | 0.60 |
| Token surprisal Monte Carlo samples | 50 | 100 |
| Token surprisal mask ratio | 0.90 | 0.82 |
| NLL clamp threshold | 8.0 | 5.0 |
| Token surprisal binary threshold (ALM-B) | > 0 after NLL clamp | > 5.0 after NLL clamp |
| Post-fusion edge erosion | Disabled by default (opt-in) | Disabled by default (opt-in) |
| Edge erosion kernel (if enabled) | 13 pixels | 13 pixels |
| Central protection radius ratio (if erosion enabled) | 0.35 | 0.35 |
| Inter-iteration dilation | 5 pixels | 1 pixel |
| Patient-level score | $\sum_{\text{slices}} \lvert \text{ALM-A} \cup \text{ALM-B} \rvert$ | $\sum_{\text{slices}} \lvert \text{ALM-A} \cup \text{ALM-B} \rvert$ |

---

### Table S4 — Dataset and Splitting Overview

| | Pelvic MRI | Brain MRI |
|---|---|---|
| Healthy training cohort | LUND-PROBE (healthy volunteers) | IXI T1-weighted (healthy volunteers) |
| Anomaly evaluation cohort | LUND-PROBE anomaly cases + synthetic artefacts | fastMRI annotated clinical cases |
| MRI weighting | T2-weighted | T1-weighted |
| Preprocessing normalisation | Z-score per volume | Z-score per volume + clip [−3, 3] |
| Retained axial slice range | All slices (stage 2 training filtered to 30–60) | 128–188 (at preprocessing) |
| Training DataModule: resize + crop | Resize 320 × 320 → centre-crop 256 × 256 | Pre-saved at 256 × 256 (no resize) |
| Split granularity | Patient level (no slice-level leakage) | Patient level (no slice-level leakage) |
| Validation strategy | 10% random patient split (seed = 42) | Pre-separated validation directory |
| Synthetic anomaly categories | Random ghosting, noise, spike, motion, whole-image Gaussian blurring | Not applicable |
| Clinical anomaly categories | Implanted spacers, clinical variations, unknown aetiology | Global findings (motion, white matter changes, extra-axial collection) and local findings (mass, oedema, resection cavity, lesion) |
| Bounding-box annotations | Not available | Available for a subset (fastMRI, used for localisation evaluation only) |
| Calibration cohort | Held-out healthy subjects from training distribution | Held-out healthy subjects from training distribution |

---

### Table S5 — Key Domain-Specific Differences: Summary

| Design dimension | Pelvic MRI | Brain MRI | Rationale |
|---|---|---|---|
| Codebook size per level | 192 | 256 | Greater anatomical diversity in multi-scanner brain cohort requires larger codebook capacity |
| Perceptual loss weight | 0.9 | 0.5 | Fine structural boundaries are diagnostically critical in the pelvis |
| Stage 2 positional encoding | 3D RoPE (row, column, slice) | 2D RoPE (row, column) | Predictable axial organisation in the pelvis; variable protocol in brain warrants 2D-only encoding |
| Decoder output clamping | None | [−3, 3] | Matches IXI preprocessing clip range; enables consistent calibration map dynamic range |
| LPIPS reference image | Original input | Stage 1 reconstruction | Brain protocol variability inflates input-vs-healed LPIPS independently of pathology |
| LPIPS backflow (default) | Enabled (99th percentile) | Disabled | Reduces false positives from acquisition artefacts in multi-scanner brain data |
| Healing mask pattern | 2 × 2 block checkerboard | 4 × 4 block checkerboard | Coarser pattern matches the spatial scale of relevant brain pathology |
| Healing temperature | 0.3 | 0.9 | Lower temperature produces sharper predictions for lower-variability pelvic acquisitions |
| Token surprisal samples | 50 | 100 | More samples required to reduce variance in heterogeneous multi-scanner setting |
| Surprisal mask ratio | 0.90 | 0.82 | Higher ratio for pelvis given lower contextual ambiguity in anatomically constrained domain |
| NLL clamp threshold | 8.0 | 5.0 | Lower baseline token entropy in pelvic model permits a higher clamp without masking true anomalies |
| Z-score threshold | > 2.0 (one-sided) | [−2.5, 6.0] (two-sided) | Two-sided bound captures both anomalously high and anomalously low reconstruction discrepancy in brain |
| Augmentation scope | Minimal (rotation + intensity scaling) | Rich (rotation, translation, zoom, contrast, noise, flip) | Scanner variability in brain data requires stronger augmentation to build anatomy-focused representations |

---

*Correspondence regarding methods should be directed to the corresponding author. Code for both the pelvic and brain MRI implementations is available at the associated repository.*
