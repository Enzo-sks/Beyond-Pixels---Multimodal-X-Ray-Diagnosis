#!/bin/bash

#SBATCH --job-name=mm-sst
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zarrougui-25@telecom-paris.fr

set -euo pipefail

# ---- Arguments -------------------------------------------------------------
MODE="${1:-14}"
LABELS="${2:-major}"
TEXT_MODE="${3:-last}"

# ---- Racine = dossier de soumission (sbatch copie le script dans un spool) --
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
# Pour entraîner le modèle SSTopensource, on utilise le dossier `multimodal head`
ARCH_PKG="$REPO_ROOT/multimodal head"
[ -d "$ARCH_PKG" ] || { echo "ERREUR: $ARCH_PKG introuvable — lance sbatch depuis la racine du repo." >&2; exit 1; }

# ---- CSV labellisé (image-level) ; major = ablation anti-circularité --------
if [ "$LABELS" = "major" ]; then
  CSV="$REPO_ROOT/data/shared/dataset_labeled.csv"
fi

# ---- Chemin des images (À FOURNIR : variable d'env IMAGE_DIR) ---------------
IMAGE_DIR="${IMAGE_DIR:-}"

# ---- Environnement ---------------------------------------------------------
source ~/envs/artishow/bin/activate
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false

cd "$REPO_ROOT"
mkdir -p logs

# ---- Cache HuggingFace (CXR-BERT + ViT téléchargés depuis HF) --------------
export HF_HOME="$REPO_ROOT/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Pré-télécharger une fois sur un nœud CPU avec réseau :
#   HF_HOME="$PWD/.hf_cache" python -c "from transformers import AutoModel, AutoTokenizer, ViTModel; \
#     n='microsoft/BiomedVLP-CXR-BERT-specialized'; \
#     AutoModel.from_pretrained(n, trust_remote_code=True); \
#     AutoTokenizer.from_pretrained(n, trust_remote_code=True); \
#     ViTModel.from_pretrained('codewithdark/vit-chest-xray'); print('modèles mis en cache')"

# ---- Préflight : échoue tôt et clairement si une dépendance manque ---------
err=0
if [ -z "$IMAGE_DIR" ] || [ ! -d "$IMAGE_DIR" ]; then
  echo "MANQUE: IMAGE_DIR='$IMAGE_DIR' (dossier des PNG Open-i). Fournis-le : IMAGE_DIR=... sbatch ..." >&2; err=1
fi
if [ ! -f "$CSV" ]; then
  echo "MANQUE: CSV '$CSV'. Génère-le : python src/mmmia/text_classification/data/labeliser.py --labels $LABELS" >&2; err=1
fi
[ "$err" -eq 0 ] || { echo "=> Préflight échoué : fournis les dépendances ci-dessus puis resoumets." >&2; exit 2; }

# ---- Infos de debug --------------------------------------------------------
echo "=== Job $SLURM_JOB_ID sur $(hostname) à $(date) ==="
echo "mode=$MODE | labels=$LABELS | text_mode=$TEXT_MODE | image_dir=$IMAGE_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# ---- Entraînement + évaluation (split groupé déjà câblé dans train.py) ------
# Prépare le répertoire multimodal head pour que `train.py` puisse s'exécuter
# 1) Assure que le module attendu `model` existe en copiant SSTopensource.py
# 2) Pointe `dataset_labeled.csv` vers le CSV généré à la racine
if [ -f "$ARCH_PKG/SSTopensource.py" ]; then
  cp -f "$ARCH_PKG/SSTopensource.py" "$ARCH_PKG/model.py"
  echo "Copie: SSTopensource.py -> $ARCH_PKG/model.py"
fi
if [ -f "$CSV" ]; then
  ln -sf "$CSV" "$ARCH_PKG/dataset_labeled.csv"
  echo "Lien symbolique: $CSV -> $ARCH_PKG/dataset_labeled.csv"
fi

cd "$ARCH_PKG"
echo "=== python -u train.py (dans $ARCH_PKG) ==="
python -u train.py

echo "=== Terminé à $(date) — checkpoint dans $ARCH_PKG/checkpoints/ ==="
