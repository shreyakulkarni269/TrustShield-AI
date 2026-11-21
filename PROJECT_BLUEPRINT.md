# MPBA G514 Deep Learning for Business — Project Blueprint & Context Document

**Domain:** Financial Technology / Digital KYC & Identity Verification  
**Topic:** Explainable Deepfake Detection Pipeline with Robustness Stress-Testing and Cost-Optimized Managerial Triage  
**Course Code:** MPBA G514 Deep Learning for Business (MBA Business Analytics, Semester 3)  
**Date Created:** November 2026  

---

## 1. Project Overview & Formal Proposal Statement

### One-Liner Description (for project submission form)
> *"Developing an explainable deep learning pipeline using EfficientNet-B0 on benchmark facial datasets to classify synthetic manipulations, assessing performance resilience against image compression, and designing a cost-optimized three-tier decision framework for automated identity verification."*

### Business Problem Grounding
* **Industry Context:** Automated Digital KYC (Know Your Customer) Onboarding for a Fintech / Neobank.
* **The Problem:** Fraudsters leverage face-swaps and generative tools to bypass KYC checks, opening synthetic accounts for loan fraud, money laundering, and mule operations. 
* **The Inefficiency:** Manual verification costs \$5–\$10 per applicant and introduces a 12–24 hour onboarding delay, leading to customer drop-off. A binary deep learning filter helps automate approvals while keeping human fraud investigators focused on ambiguous cases.

---

## 2. Technical Architecture & End-to-End Pipeline

```
[Raw Data: OpenForensics / FaceForensics++]
                │
                ▼
[Face Cropping & Alignment (224x224 RGB)]
                │
                ▼
[Data Splits: 70% Train (14k) | 15% Val (3k) | 15% Test (3k)]
                │
                ▼
[Model Training: Pretrained EfficientNet-B0 + Dropout(0.3) + Linear(1)]
   • Loss: BCEWithLogitsLoss
   • Optimizer: AdamW (lr=1e-4)
                │
                ▼
[Clean Baseline Evaluation]
   • Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
                │
       ┌────────┴───────────────────────────┐
       ▼                                    ▼
[Model Diagnostics: Grad-CAM]     [Robustness Stress-Test]
 • Heatmaps on final conv layer    • JPEG Quality Sweep: Q ∈ [90, 70, 50, 30]
 • Visual verification of seams    • Degradation curve: Accuracy vs. Quality Factor
       │                                    │
       └────────────────┬───────────────────┘
                        ▼
         [Managerial Decision Analytics]
          • Cost-Matrix Loss Optimization (τ*)
          • Three-Tier Risk Triage (Green / Yellow / Red)
          • Operational Capacity & ROI Modeling
```

---

## 3. Dataset Specifications & Acquisition

### Preferred Academic Benchmark Sources (Avoiding unvetted synthetic scrapes)
1. **OpenForensics (ICCV 2021) — Primary Choice:**
   * Direct download via official repository without waiting for credentials: [https://github.com/billytai1/OpenForensics](https://github.com/billytai1/OpenForensics).
   * Annotations contain bounding boxes and category labels (`category_id: 1` = Real, `category_id: 2` = Forged).
2. **FaceForensics++ (ICCV 2019) — Benchmark / Generalization Test:**
   * Highly cited academic gold standard from TUM / Google: [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics).
   * Approval via automated academic form typically takes 12–48 hours.
3. **Pristine Real-Face Baselines (if needed):**
   * **FFHQ (NVIDIA):** 70,000 real high-res faces: [https://github.com/NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).

### Dataset Organization on Disk
Target size: Balanced **20,000 images** (10,000 Real, 10,000 Manipulated).
```text
dataset/
├── train/
│   ├── real/          # 7,000 images
│   └── manipulated/   # 7,000 images
├── val/
│   ├── real/          # 1,500 images
│   └── manipulated/   # 1,500 images
└── test/
    ├── real/          # 1,500 images
    └── manipulated/   # 1,500 images
```

### Input Preprocessing Specs
* **Size:** $224 \times 224$ pixels.
* **Color Channels:** 3-channel RGB.
* **Normalization:** ImageNet constants ($\text{Mean} = [0.485, 0.456, 0.406]$, $\text{Std} = [0.229, 0.224, 0.225]$).
* **Train Augmentations:** Random horizontal flip ($p=0.5$), slight random rotation ($\pm 10^\circ$).

---

## 4. Modeling & Experimental Details

### EfficientNet-B0 Backbone (PyTorch)
```python
import torch.nn as nn
from torchvision.models import efficientnet_b0

model = efficientnet_b0(weights="DEFAULT")
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, 1)  # Raw single logit for binary classification
)
```

### Diagnostic 1: Grad-CAM Interpretability
* **Library:** `pytorch-grad-cam`
* **Target Layer:** Final convolutional block (`model.features[-1]`).
* **Objective:** Produce visual saliency maps for:
  * **True Positive (TP):** Confirming the network detects blend boundaries, eye reflections, or edge artifacts rather than background shortcuts.
  * **False Positive (FP):** Identifying what innocent visual features (lighting, shadows, glasses) triggered false alarms.

### Diagnostic 2: Transmission Robustness (JPEG Sweep)
* **Hypothesis:** Uploading compressed selfies (e.g., via WhatsApp or mobile browsers) smooths out high-frequency manipulation artifacts, degrading detection accuracy.
* **Method:** Re-save the test partition using PIL at quality levels $QF \in [90, 70, 50, 30]$:
  ```python
  from PIL import Image
  img.save("compressed_test.jpg", "JPEG", quality=q)
  ```
* **Output:** Plot of **Test Accuracy vs. JPEG Quality Factor** demonstrating the operational breakdown point.

---

## 5. MBA Managerial & Decision Framework

### 1. Cost-Sensitive Threshold Optimization
Rather than using a generic $0.50$ probability threshold, calculate total expected enterprise loss across candidate thresholds $\tau \in [0.05, 0.95]$:
* **Cost of False Negative ($C_{\text{FN}}$):** Fake ID accepted $\rightarrow$ Average loan fraud loss = **\$2,500**.
* **Cost of False Positive ($C_{\text{FP}}$):** Real user blocked $\rightarrow$ Manual review ticket + customer friction = **\$15**.

$$\text{Expected Business Cost}(\tau) = C_{\text{FN}} \cdot N_{\text{FN}}(\tau) + C_{\text{FP}} \cdot N_{\text{FP}}(\tau)$$
$$\tau^* = \arg\min_\tau \text{Expected Business Cost}(\tau)$$

### 2. Three-Tier Operational Triage Workflow
```
Applicant Selfie Upload
          │
          ▼
   [EfficientNet-B0] ── Output probability p ∈ [0, 1]
          │
   ┌──────┴──────────────────────────────────────┐
   ▼                                             ▼
[Green Zone: p < τ_low]               [Yellow Zone: τ_low ≤ p ≤ τ_high]      [Red Zone: p > τ_high]
• Volume: ~80-85%                     • Volume: ~10-15%                      • Volume: ~3-5%
• Action: Instant Auto-Approve        • Action: Human Analyst Review Queue   • Action: Auto-Reject
• Latency: < 3 seconds                • Tooling: Grad-CAM overlay display    • Action: Fraud Log
```

### 3. Unit Economics & ROI Simulation (Monthly volume: 100,000 applicants)
* **Status Quo (100% Manual Review):**  
  $100,000 \times \$6.00 = \mathbf{\$600,000/\text{month}}$.
* **Proposed DL + Three-Tier Model:**  
  $14,000 \text{ manual reviews (Yellow Zone)} \times \$6.00 = \mathbf{\$84,000/\text{month}}$.  
  * **Direct Monthly Operational Savings:** $\mathbf{\$516,000/\text{month}}$ (~86% reduction in review costs).
  * **Customer Conversion:** 82% of authentic applicants onboarded instantaneously (3 seconds vs. 24 hours).

### 4. Enterprise Risk Governance & Policy Guidelines
1. **In-App Capture Policy:** The native application must enforce direct in-app camera capture and reject images uploaded via messaging services (WhatsApp/Telegram) which drop compression below $QF < 70$.
2. **Model Risk Governance:** Schedule quarterly canary testing against newly released generative models to monitor and mitigate cross-generator domain drift.
3. **Analyst Augmentation:** Yellow-zone review dashboards must render the Grad-CAM heatmap overlay to reduce human investigation time from 5 minutes to 90 seconds.

---

## 6. Execution Roadmap & Milestones

1. **Phase 1 (Data):** Download OpenForensics / FF++, crop faces with bounding boxes, structure 70/15/15 splits.
2. **Phase 2 (Model):** Train EfficientNet-B0 on Colab T4 GPU, record baseline metrics (Accuracy, F1, ROC-AUC).
3. **Phase 3 (Diagnostics):** Run Grad-CAM heatmap script on selected test cases; execute the 4-tier JPEG compression sweep.
4. **Phase 4 (Analytics):** Compute the cost curve vs. threshold $\tau$, determine $\tau^*$ and triage boundaries, format ROI tables.
5. **Phase 5 (Reporting):** Build the final presentation slide deck (technical validation + managerial outcomes) and optional Streamlit demonstration.
