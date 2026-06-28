## Projector LLM — ViT → MLP Projector → MedGemma 4B + LoRA

### Overview

Le module `projector_llm/` implémente l'approche **Famille 4 — Projector + LLM** (inspirée de LLaVA-Med / MedGemma).

Un ViT pré-entraîné extrait des tokens visuels. Un **Projector MLP à 2 couches** les projette vers l'espace d'embedding de MedGemma 4B. Les tokens image projetés sont **prepend** à la séquence de tokens texte. MedGemma traite la séquence mixte et le dernier token causal alimente une tête de classification multi-label (21 pathologies).

**Pourquoi MedGemma 4B et pas Gemma 3 1B ?**
`google/medgemma-4b-pt` est pré-entraîné par Google DeepMind sur des données médicales massives (imagerie + texte clinique). Son espace d'embedding est déjà aligné sur la terminologie radiologique, ce qui donne un avantage direct pour les rapports CXR vs un LLM généraliste.

---

### Architecture (`model.py`)

```
pixel_values (B, 3, 224, 224)
      │
      ▼
  ViT encoder  (B, 197, 768)   ← GELÉ
      │
      ▼
  Projector MLP                ← ENTRAÎNÉ
    Linear(768 → 2560)
    GELU + Dropout
    Linear(2560 → 2560)
      │  (B, 197, 2560)
      │
  prepend à text_embeds (B, L, 2560)
      │
      ▼
  MedGemma 4B + LoRA (q_proj, v_proj)   ← LoRA ENTRAÎNÉ, reste GELÉ
      │
  last_hidden_state[:, -1, :]   (B, 2560)
      │
      ▼
  Linear(2560 → 21)   ← ENTRAÎNÉ
```

| Composant | Détails |
|-----------|---------|
| ViT | `codewithdark/vit-chest-xray` ViT-B/16 (768d, 197 tokens) — 86M params — gelé |
| Projector | MLP 2 couches : 768→2560→2560 + GELU + Dropout — ~7.9M params — entraîné |
| LLM | MedGemma 4B (`google/medgemma-4b-pt`, 2560d) — ~4B params — gelé sauf LoRA |
| LoRA | rank=16, alpha=32, cibles : q_proj + v_proj — ~8-12M params — entraîné |
| Head | Linear(2560 → 21) — ~54k params — entraîné |
| **Entraînables** | **~16-20M / ~4.09B total (~0.4%)** |

---

### Dataset (`dataset.py`)

Réutilise exactement `multimodal_fusion/dataset.py` :
- 6 473 échantillons après filtrage
- Split identique : Train 4 529 / Val 976 / Test 968
- Tokenizer BPE maison (vocab 4 359 tokens)
- Images : Resize(224×224) + Normalize([0.5, 0.5, 0.5])

---

### Entraînement (`train.py`)

**Phase unique** — LoRA s'adapte directement sur la connaissance médicale pré-entraînée :

| Paramètre | Valeur |
|-----------|--------|
| Loss | AsymmetricLoss (γ⁻=4, γ⁺=1, clip=0.05) |
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR → 1e-6 |
| Batch size | **4** (MedGemma 4B, ~10-12 GB VRAM) |
| Gradient clipping | 1.0 |
| AMP | ✅ float16 sur CUDA |
| Early stopping | patience=7 |
| LoRA rank/alpha | 16 / 32 |

**Sauvegarde légère** : uniquement projector + matrices LoRA + head (~60 MB vs ~16 GB pour les poids complets 4B).

---

### Accès au modèle (gated)

MedGemma est un modèle à accès restreint :

```bash
# 1. Accepter la licence sur HuggingFace
# https://huggingface.co/google/medgemma-4b-pt

# 2. Se connecter
huggingface-cli login
# coller ton token HF (Settings → Access Tokens)
```

---

### Usage

```bash
# Minimal
python train.py --image_dir /content/Png --csv /content/drive/MyDrive/dataset_labeled.csv

# Avec checkpoint ViT et LoRA rank plus élevé
python train.py \
  --image_dir /content/Png \
  --csv dataset_labeled.csv \
  --vit_checkpoint /content/best_vit_chest_04.pth \
  --lora_rank 32 \
  --lora_alpha 64 \
  --batch_size 2   # si VRAM < 16 GB

# LLM alternatif (ex: retour à Gemma 3 1B)
python train.py \
  --image_dir /content/Png \
  --csv dataset_labeled.csv \
  --gemma_name google/gemma-3-1b-pt \
  --batch_size 8
```

---

### Installation des dépendances

```bash
pip install peft transformers accelerate iterative-stratification
# Accepter la licence sur https://huggingface.co/google/medgemma-4b-pt
# puis : huggingface-cli login
```

---

### Comparaison avec multimodal_fusion

| Critère | multimodal_fusion | projector_llm |
|---------|-------------------|---------------|
| Fusion | Cross-attention bidirectionnel | Prepend + self-attention LLM |
| LLM backbone | ✗ | MedGemma 4B (médical, ~4B params) |
| Params entraînables | ~5.3M → 95.9M | ~16-20M (stable) |
| VRAM estimée | ~4 GB | ~10-12 GB (batch=4, AMP) |
| Checkpoint | ~366 MB | ~60 MB |
| Avantage domaine | ViT fine-tuné CXR | ViT CXR + LLM médical |
| Test AUC | **0.9886** | à mesurer |