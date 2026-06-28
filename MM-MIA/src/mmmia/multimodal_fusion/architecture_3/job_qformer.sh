#!/bin/bash
# ============================================================================
#  job_qformer.sh — Entraîne le Q-Former label-aligné (CXR-BERT + ViT) sur le
#  cluster GPU Télécom Paris. Conventions : cf. docs/cluster_ml_pipeline.md et
#  src/mmmia/multimodal_fusion/job_multimodal.sh (architecture_1).
#
#  Usage (depuis la racine du projet, ~/mon_projet) :
#      IMAGE_DIR=/chemin/vers/les/png \
#        sbatch src/mmmia/multimodal_fusion/architecture_3/job_qformer.sh [MODE] [LABELS] [TEXT_MODE]
#
#      MODE      : 5 | 14 | 21          (défaut 14 — NIH)
#      LABELS    : all | major          (défaut major — anti-circularité)
#      TEXT_MODE : last | deep          (défaut last ; deep = branchement profond)
#
#  Exemples :
#      IMAGE_DIR=~/data/Png sbatch .../job_qformer.sh 14 major
#      IMAGE_DIR=~/data/Png sbatch .../job_qformer.sh 14 major deep
# ============================================================================
#SBATCH --job-name=mm-qformer
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hugo.hennion@telecom-paris.fr

set -euo pipefail

# ---- Arguments -------------------------------------------------------------
MODE="${1:-14}"
LABELS="${2:-major}"
TEXT_MODE="${3:-last}"
# Flags train.py supplémentaires (variante anti-collapse / modality dropout), ex :
#   EXTRA_ARGS="--norm pre --reinject_identity --text_dropout 0.3"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ---- Racine = dossier de soumission (sbatch copie le script dans un spool) --
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
ARCH_PKG="$REPO_ROOT/src/mmmia/multimodal_fusion/architecture_3"
[ -d "$ARCH_PKG" ] || { echo "ERREUR: $ARCH_PKG introuvable — lance sbatch depuis ~/mon_projet." >&2; exit 1; }

# ---- CSV labellisé (image-level) ; major = ablation anti-circularité --------
if [ "$LABELS" = "major" ]; then
  CSV="$REPO_ROOT/data/shared/dataset_labeled_major.csv"
else
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
cd "$ARCH_PKG"
echo "=== python train.py --csv $CSV --image_dir $IMAGE_DIR --mode $MODE --text_feature_mode $TEXT_MODE $EXTRA_ARGS ==="
python -u train.py --csv "$CSV" --image_dir "$IMAGE_DIR" \
       --mode "$MODE" --text_feature_mode "$TEXT_MODE" --batch_size 16 $EXTRA_ARGS

echo "=== Terminé à $(date) — checkpoint dans $ARCH_PKG/checkpoints/ ==="
