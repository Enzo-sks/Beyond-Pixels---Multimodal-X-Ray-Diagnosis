#!/bin/bash
# ============================================================================
#  job_projector_llm.sh — Entraîne ProjectorLLM (ViT → Projector → MedGemma 4B + LoRA)
#  Version optimisée pour GPU P100 (16 Go VRAM)
# ============================================================================

#SBATCH --job-name=mm-projector-medgemma
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=enzo.sakhinis@telecom-paris.fr

set -euo pipefail

# ---- Arguments -------------------------------------------------------------
LABELS="${1:-major}"

# ---- Racine du repo --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../../../" >/dev/null 2>&1 && pwd)}"
PKG_DIR="$REPO_ROOT/src/mmmia/multimodal_fusion/projector_llm"
[ -d "$PKG_DIR" ] || { echo "ERREUR: $PKG_DIR introuvable."; exit 1; }

# ---- CSV -------------------------------------------------------------------
CSV="$REPO_ROOT/data/shared/dataset_labeled.csv"

# ---- Chemin des images -----------------------------------------------------
IMAGE_DIR="/home/infres/esakhinis-25/ArtishowLLM/data/Png"

# ---- Environnement Python --------------------------------------------------
source ~/envs/artishow/bin/activate
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false

cd "$REPO_ROOT"
mkdir -p logs

# ---- Cache HF --------------------------------------------------------------
export HF_HOME="$REPO_ROOT/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- Préflight -------------------------------------------------------------
err=0
[ -d "$IMAGE_DIR" ] || { echo "MANQUE: dossier images $IMAGE_DIR"; err=1; }
[ -f "$CSV" ]       || { echo "MANQUE: CSV $CSV"; err=1; }
[ "$err" -eq 0 ] || exit 2

# ---- Debug -----------------------------------------------------------------
echo "=== Job $SLURM_JOB_ID sur $(hostname) ==="
echo "labels=$LABELS | image_dir=$IMAGE_DIR | csv=$CSV"
nvidia-smi || true

# ---- Entraînement ----------------------------------------------------------
cd "$PKG_DIR"
echo "=== Lancement entraînement ==="

python -u train.py \
    --csv        "$CSV"                    \
    --image_dir  "$IMAGE_DIR"              \
    --text_ckpt  "$REPO_ROOT/src/mmmia/text_classification/checkpoints" \
    --gemma_name "google/gemma-3-1b-pt"    \
    --batch_size 4                         \
    --lora_rank  16                        \
    --lora_alpha 32

echo "=== Terminé — checkpoints dans $PKG_DIR/checkpoints/ ==="