#!/bin/bash
#SBATCH --job-name=dicom_3d_finetuning
#SBATCH --output=logs/%x_%j.out      # stdout dans logs/
#SBATCH --error=logs/%x_%j.err       # stderr dans logs/
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00              # max 36h pour les etudiants
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=prenom.nom@telecom-paris.fr

# --- Variables ---
# A MODIFIER: chemins selon votre dossier sur le cluster
PROJECT_DIR="$HOME/artishow/dicom_gpu_telecom"
ENV_DIR="$HOME/artishow/mon_projet_ia"
CSV_PATH="$HOME/artishow/data/dataset_labeled_volumes_3d_combined.csv" 
VOLUMES_DIR="$HOME/artishow" # dataset.py ajoutera les sous-dossiers v1 et v2

# --- Setup ---
source $ENV_DIR/bin/activate
cd $PROJECT_DIR

mkdir -p logs checkpoints results splits

# --- Infos de debug utiles dans les logs ---
echo "=== Job $SLURM_JOB_ID demarre sur $(hostname) a $(date) ==="
echo "GPU alloue :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Verification de l'acces GPU par PyTorch ---
echo "Test de l'acces GPU via PyTorch..."
python -c "import torch; assert torch.cuda.is_available(), 'Erreur: Aucun GPU detecte par PyTorch!'"
if [ $? -ne 0 ]; then
    echo "🚨 Le test GPU a echoue. L'entrainement tournerait sur CPU. Arret du script."
    exit 1
fi
echo "✅ GPU detecte avec succes par PyTorch."

# --- 1. Preparer les donnees (force recreation des splits) ---
echo "Creation des splits train/val/test..."
rm -f splits/split_train_3d.csv splits/split_val_3d.csv splits/split_test_3d.csv
python prepare_data.py \
    --input-csv "$CSV_PATH" \
    --output-dir "splits/"

# --- 2. Lancer l'entrainement v7 ---
echo "Demarrage de l'entrainement v7..."
python train.py \
    --data-dir splits/ \
    --volumes-dir "$VOLUMES_DIR" \
    --output-dir checkpoints/ \
    --epochs 25 \
    --batch-size 8 \
    --lr 1e-3 \
    --patience 5

# --- 3. Evaluer sur le set de test ---
echo "Evaluation du meilleur modele..."
python eval.py \
    --data-dir splits/ \
    --volumes-dir "$VOLUMES_DIR" \
    --model-path checkpoints/best_model_3d.pth \
    --output-dir results/

echo "=== Job termine a $(date) ==="
