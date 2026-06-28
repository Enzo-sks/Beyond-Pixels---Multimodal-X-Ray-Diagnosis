## Multimodal Fusion — Implementation & Results

### Overview

The `multimodal_fusion/` module combines a custom BERT text encoder and a Vision Transformer (ViT) image encoder into a single multi-label classifier for 21 chest pathologies, using bidirectional cross-attention as the fusion mechanism.

---

### Architecture (`model.py`)

**`BidirectionalCrossAttention`**

Both modalities are projected to a common space (d=512) before attention:

- Text tokens `(B, L, 256)` → Linear projection → `(B, L, 512)`
- Image tokens `(B, 197, 768)` → Linear projection → `(B, 197, 512)`
- Two cross-attention directions computed **in parallel**:
  - Text → Image (`t2i`): text queries attend over image keys/values
  - Image → Text (`i2t`): image queries attend over text keys/values
- Each branch has a residual connection, FFN (512 → 1024 → 512), and LayerNorm

**`MultimodalFusion`**

| Component | Details |
|-----------|---------|
| Text encoder | Custom BERT (d=256, 6 layers, 8 heads, RoPE+RMSNorm+SiLU) — 4.3M params |
| Image encoder | `codewithdark/vit-chest-xray` ViT-B/16 (d=768, 197 tokens) — 86M params |
| Cross-attention | BidirectionalCrossAttention, 8 heads, d_model=512 |
| Text pooling | CLS token (index 0) of fused text sequence |
| Image pooling | Learned soft-attention over all 197 fused image tokens |
| Classification head | Linear(1024 → 512) + LayerNorm + GELU + Dropout + Linear(512 → 21) |
| **Total parameters** | **95,931,925** |

---

### Dataset (`dataset.py`)

- Source: `dataset_labeled.csv` — 7,470 paired rows (image_id + findings + 21 binary labels)
- After filtering empty findings: **6,473 samples**
- Split: MultilabelStratifiedShuffleSplit — **Train 4,529 / Val 976 / Test 968**
- Text tokenized via custom BPE tokenizer (vocab: 4,359 tokens), padded per-batch
- Images: PIL PNG → Resize(224×224) → Normalize([0.5, 0.5, 0.5])

---

### Training (`train.py`)

**Loss:** AsymmetricLoss (Ben-Baruch et al., 2021) — γ⁻=4, γ⁺=1, clip=0.05

**Two-phase strategy:**

| Phase | Epochs | Trainable params | LR |
|-------|--------|------------------|----|
| Phase 1 | 1–3 | Cross-attn + head only (5.3M) | Warmup 5e-6 → 5e-5 |
| Phase 2 | 4–30 | All params (95.9M) | LR_head=5e-5, LR_backbone=5e-6, cosine → 1e-6 |

**Other settings:** AdamW, weight_decay=1e-4, gradient clipping 1.0, early stopping patience=7, batch size=16

**Training stopped at epoch 22** (early stopping triggered).

---

### Results

**Training curve (key epochs):**

| Epoch | Phase | Val AUC |
|-------|-------|---------|
| 1 | Frozen | 0.6247 |
| 3 | Frozen | 0.9206 |
| 4 | Unfrozen | 0.9596 |
| 8 | Unfrozen | 0.9879 ★ |
| 15 | Unfrozen | 0.9881 ★ best |
| 22 | Unfrozen | early stop |

**Per-class AUC on test set (968 samples):**

| Pathology | AUC |
|-----------|-----|
| Atelectasis | 0.9875 |
| Cardiomegaly | 0.9887 |
| Effusion | 0.9731 |
| Pneumonia | 0.9670 |
| Pneumothorax | 0.9969 |
| Edema | 0.9922 |
| Emphysema | 0.9604 |
| Fibrosis | 0.9550 |
| Infiltration | 0.9839 |
| Mass | 0.9946 |
| Nodule | 0.9918 |
| Hernia | 0.9997 |
| Fracture | 0.9922 |
| Pleural_Thickening | 1.0000 |
| Opacity | 0.9999 |
| Consolidation | 0.9936 |
| Granuloma | 0.9999 |
| Calcinosis | 0.9974 |
| Scoliosis | 0.9950 |
| Atherosclerosis | 1.0000 |
| Normal | 0.9920 |
| **Mean AUC** | **0.9886** |

**Best val AUC: 0.9881 — Test AUC: 0.9886**

Checkpoint saved at: `checkpoints/multimodal_fusion.pt`

Checkpoint for the multimodal_fusion + tokenizer: 
https://drive.google.com/drive/folders/1NBXp-Foy-2Rym3vY2mclVxde-ZcBop7t?usp=sharing


