# Modèles entraînés — `models/`

Tous les fichiers de ce dossier sont gérés via **Git LFS** (backend GitLab ENST).

## Convention

```
models/
├── <module>/
│   └── <model_name>.<ext>        # pth, ckpt, safetensors...
```

Un sous-dossier par module Python (`text_classification`, `image_classification`, `dicom`, `api`).

## Modèles actuels

| Modèle | Module | Taille | Usage |
|---|---|---|---|
| `api/model_full.pth` | api | 16 MB | Modèle servi par l'API FastAPI (medical-api) |

## Charger un modèle

```python
import torch

model = torch.load("models/api/model_full.pth", map_location="cpu")
# ou pour PyTorch Lightning :
from mmmia.image_classification.own_model.models.tinyvgg import TinyVGG
model = TinyVGG.load_from_checkpoint("models/image_classification/best.ckpt")
```

## Ajouter un modèle

1. Sauvegarder dans le bon sous-dossier (`models/<module>/<nom_descriptif>.pth`)
2. **Nom descriptif** : inclure architecture + dataset + métrique principale
   - ✅ `densenet121_openi_auc0.91.pth`
   - ❌ `model_v3.pth`
3. `git add` puis commit — LFS s'occupe automatiquement du transfert
4. Documenter dans ce README (table ci-dessus)

## ⚠️ Ne PAS commit

- Checkpoints intermédiaires d'entraînement (`epoch=*.ckpt`) → `lightning_logs/` (gitignored)
- Modèles expérimentaux jetables → garder en local jusqu'à validation
