# Architecture 3 — Q-Former label-aligné (famille 3)

Fusion image + texte par **distillation de requêtes latentes** (Q-Former, cf.
`docs/architectures_fusion_multimodale_v3.md` §Famille 3). Conçue pour
l'interprétabilité : **une requête apprenable par pathologie**, alignée
positionnellement sur les labels.

## Principe

```
indication+findings --CXR-BERT (gelé)--> T (B, L≤256, 768) + masque padding
image PNG           --ViT      (gelé)--> Z (B, 197, 768)
14 requêtes label-alignées --N blocs Q-Former--> H (B, 14, 768)
H --tête diagonale--> 14 logits --Asymmetric Loss--> entraînement
```

Chaque bloc (`JointQueryBlock`) : **SA des requêtes sur `[Q; T]`** (seules les
requêtes sont mises à jour) → **CA des requêtes vers `Z`** → **FFN**, chacune
résiduelle + LayerNorm. La tête est **diagonale** : `logit_j = w_j · H_j + b_j`,
donc le logit de la pathologie *j* ne lit que la requête *j* (aucun pooling).

| Composant | Détail |
|-----------|--------|
| Encodeur texte | `microsoft/BiomedVLP-CXR-BERT-specialized` (768), gelé par défaut |
| Encodeur image | `codewithdark/vit-chest-xray` ViT-B/16 (768, 197 tokens), gelé |
| Q-Former | N blocs (défaut 3), 14 requêtes label-alignées |
| Loss | Asymmetric Loss (γ⁻=4, γ⁺=1, clip=0.05) |
| Labels | 14 NIH (`LABELS_14`, ordre canonique de `text_classification`) |
| Anti-fuite | texte = `indication` + `findings` (jamais `impression`) |

## Variante « branchement profond » (`--text_feature_mode deep`)

Au lieu de donner la dernière couche de CXR-BERT à tous les blocs, le bloc *l*
lit la *l*-ième couche cachée (les N dernières, coarse→fine) — style BLIP-2
hiérarchique. Coût nul (BERT calcule déjà toutes ses couches). Exposé comme
**ablation** ; le défaut reste `last`.

## Fichiers

- `model.py` — `JointQueryBlock`, `QFormerHead`, `FusionQFormer`, `build_cxr_tokenizer`
  (encodeurs CXR-BERT + ViT chargés ici)
- `train.py` — entraînement PyTorch Lightning + évaluation test (AUC par label)
- `job_qformer.sh` — soumission SLURM

Données, loss (ASL) et transforms image sont partagés via `../common/`
(`common.data`, `common.losses`, `common.transforms`) avec les autres
architectures multimodales.

## Entraînement

```bash
# Local / Colab
python train.py --image_dir /content/Png

# Variante profonde, 4 blocs
python train.py --image_dir /content/Png --text_feature_mode deep --n_layers 4

# Sanity check (surapprentissage sur 2 batches)
python train.py --image_dir /content/Png --overfit_batches 2 --epochs 50

# Cluster
IMAGE_DIR=~/data/Png sbatch job_qformer.sh 14 major
```

Le split est **groupé par texte** (sans fuite train/val/test) via
`text_classification/data/splits.py`.
