#!/bin/bash
#SBATCH --job-name=artishow
#SBATCH --gres=gpu:1          # demander 1 GPU
#SBATCH --cpus-per-task=4     # doit correspondre à num_workers
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Charger l'environnement
source ~/.bashrc
conda activate mon_env   # remplace par ton env réel

# Lancer ton script
#python artishow_lightning.py

import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
