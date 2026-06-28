# Chest X-Ray Multi-Label Disease Classification

Multi-label classification of 21 chest conditions from X-ray images.

## Dataset

| Property | Value |
|---|---|
| Total images | 7,470 PNG X-rays |
| Labels | 21 conditions (multi-label) |
| Train / Val / Test | 5,229 / 1,120 / 1,121 |
| Split strategy | Stratified on primary label |

**Conditions:** Atelectasis, Cardiomegaly, Effusion, Pneumonia, Pneumothorax, Edema, Emphysema, Fibrosis, Infiltration, Mass, Nodule, Hernia, Fracture, Pleural_Thickening, Opacity, Consolidation, Granuloma, Calcinosis, Scoliosis, Atherosclerosis, Normal

> Dataset is heavily imbalanced — Normal class has 4,166 samples while Mass has only 41.

---

## Model 1 — DenseNet-121 (`cv_model_01.ipynb`)

### Architecture

- **Backbone:** DenseNet-121 pretrained on ImageNet
- **Classifier head:** `Linear(1024→512) → ReLU → Dropout(0.5) → Linear(512→21)`
- **Loss:** BCEWithLogitsLoss with per-class positive weights
- **Optimizer:** AdamW — lr=1e-4, weight_decay=1e-4
- **Scheduler:** CosineAnnealingLR

### Training

| Hyperparameter | Value |
|---|---|
| Image size | 224×224 |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping patience | 5 |

**Augmentations (train only):** horizontal flip, rotation ±15°, random affine translate, color jitter

Training stopped early at epoch 17 (no AUC improvement for 5 consecutive epochs).

### Results

| Metric | Value |
|---|---|
| Test Mean AUC-ROC | **0.7802** |
| Test Loss | 1.2076 |
| Best Val AUC (epoch 12) | 0.7787 |

#### Per-class AUC-ROC

| Condition | AUC |
|---|---|
| Emphysema | 0.942 |
| Hernia | 0.928 |
| Consolidation | 0.916 |
| Cardiomegaly | 0.885 |
| Effusion | 0.868 |
| Edema | 0.858 |
| Atelectasis | 0.822 |
| Atherosclerosis | 0.820 |
| Opacity | 0.810 |
| Infiltration | 0.806 |
| Fibrosis | 0.793 |
| Pneumothorax | 0.786 |
| Mass | 0.769 |
| Pneumonia | 0.768 |
| Normal | 0.758 |
| Pleural_Thickening | 0.751 |
| Calcinosis | 0.675 |
| Granuloma | 0.672 |
| Fracture | 0.635 |
| Nodule | 0.590 |
| Scoliosis | 0.532 |

---

## Model 2 — ViT (`cv_model_vit.ipynb`)

### Architecture

- **Backbone:** `codewithdark/vit-chest-xray` (ViT-base pretrained on chest X-rays)
- **Classifier head:** `Linear(768→512) → ReLU → Dropout(0.5) → Linear(512→21)`
- **Input:** [CLS] token from the last hidden state
- **Loss:** BCEWithLogitsLoss with per-class positive weights
- **Optimizer:** AdamW — lr=5e-5, weight_decay=1e-4
- **Scheduler:** CosineAnnealingLR

### Training

| Hyperparameter | Value |
|---|---|
| Image size | 224×224 |
| Batch size | 16 |
| Max epochs | 20 |
| Early stopping patience | 5 |
| Normalization | mean=0.5, std=0.5 |

**Augmentations (train only):** horizontal flip, rotation ±10°, color jitter

### Results

| Metric | Value |
|---|---|
| Test Mean AUC-ROC | **0.7731** |
| Test Loss | 1.3912 |

#### Per-class AUC-ROC

| Condition | AUC |
|---|---|
| Consolidation | 0.923 |
| Hernia | 0.915 |
| Emphysema | 0.912 |
| Cardiomegaly | 0.896 |
| Edema | 0.874 |
| Effusion | 0.870 |
| Infiltration | 0.843 |
| Atelectasis | 0.830 |
| Opacity | 0.810 |
| Mass | 0.806 |
| Atherosclerosis | 0.802 |
| Pneumonia | 0.778 |
| Normal | 0.762 |
| Pleural_Thickening | 0.737 |
| Fibrosis | 0.717 |
| Pneumothorax | 0.678 |
| Calcinosis | 0.663 |
| Granuloma | 0.635 |
| Nodule | 0.612 |
| Fracture | 0.611 |
| Scoliosis | 0.561 |

---

## Model Comparison

| Condition | DenseNet-121 | ViT | Delta |
|---|---|---|---|
| Emphysema | 0.942 | 0.912 | -0.030 |
| Hernia | 0.928 | 0.915 | -0.013 |
| Consolidation | 0.916 | 0.923 | +0.007 |
| Cardiomegaly | 0.885 | 0.896 | +0.011 |
| Effusion | 0.868 | 0.870 | +0.002 |
| Edema | 0.858 | 0.874 | +0.016 |
| Atelectasis | 0.822 | 0.830 | +0.008 |
| Atherosclerosis | 0.820 | 0.802 | -0.018 |
| Opacity | 0.810 | 0.810 | 0.000 |
| Infiltration | 0.806 | 0.843 | +0.037 |
| Fibrosis | 0.793 | 0.717 | -0.076 |
| Pneumothorax | 0.786 | 0.678 | -0.108 |
| Mass | 0.769 | 0.806 | +0.037 |
| Pneumonia | 0.768 | 0.778 | +0.010 |
| Normal | 0.758 | 0.762 | +0.004 |
| Pleural_Thickening | 0.751 | 0.737 | -0.014 |
| Calcinosis | 0.675 | 0.663 | -0.012 |
| Granuloma | 0.672 | 0.635 | -0.037 |
| Fracture | 0.635 | 0.611 | -0.024 |
| Nodule | 0.590 | 0.612 | +0.022 |
| Scoliosis | 0.532 | 0.561 | +0.029 |
| **Mean AUC** | **0.7802** | **0.7731** | **-0.0071** |

### Analysis

ViT performs better on **diffuse/global patterns** (Infiltration, Mass, Edema, Cardiomegaly) where global attention across the whole image is beneficial. DenseNet-121 performs better on **structural/edge findings** (Pneumothorax, Fibrosis, Emphysema) where fine local texture and spatial cues matter — ViT's 16×16 patches miss peripheral absence-of-markings that signal Pneumothorax.
