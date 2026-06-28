# Pipeline ML sur le cluster — Télécom Paris

## Vue d'ensemble

```
[Ta machine locale]          [Cluster]
       │                         │
       │  1. SSH / VS Code ───►  login node (gpu-gw)
       │  2. rsync code ──────►  /home/ids/<user>/mon_projet/
       │                         │
       │                    3. sbatch / sinteractive
       │                         │
       │                    4. Compute node (GPU)
       │                         │  - charge dataset depuis /projects/common/
       │                         │  - entraîne le modèle
       │                         │  - sauvegarde checkpoints + logs
       │                         │
       │  5. rsync résultats ◄─  /home/ids/<user>/results/
```

---

## Étape 0 — Setup SSH (une seule fois)

Ajoute ça dans `~/.ssh/config` sur **ta machine locale** :

```
Host cluster
    HostName gpu-gw.enst.fr
    User <tp-username>
    ServerAliveInterval 60

Host cluster-store
    HostName ids-store.enst.fr
    User <tp-username>
    ServerAliveInterval 60
```

Ensuite tu peux juste faire `ssh cluster` pour te connecter.

---

## Étape 1 — Préparer ton projet en local

Structure recommandée :

```
mon_projet/
├── data/               # scripts de chargement uniquement, PAS les données
│   └── dataset.py
├── models/
│   └── resnet.py
├── train.py
├── eval.py
├── requirements.txt
└── job.sh              # script sbatch
```

> **Ne mets jamais les données dans ton repo.** Les datasets lourds restent sur le cluster.

---

## Étape 2 — Envoyer le code sur le cluster

```bash
# Première fois : copie complète
rsync -avz --exclude='__pycache__/' \
           --exclude='*.pyc' \
           --exclude='.git/' \
           --exclude='venv/' \
    mon_projet/ cluster-store:~/mon_projet/

# Après chaque modification : sync incrémentale (très rapide)
rsync -avz --exclude='__pycache__/' mon_projet/ cluster-store:~/mon_projet/
```

---

## Étape 3 — Setup de l'environnement Python (une seule fois)

Connecte-toi au login node, puis demande un nœud CPU pour ne pas surcharger le login :

```bash
ssh cluster
sinteractive -p CPU --time 0:30:00
```

Une fois sur le nœud CPU :

```bash
cd ~/mon_projet

# Créer le virtualenv
python3 -m venv ~/envs/mon_projet
source ~/envs/mon_projet/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

exit   # libère le nœud CPU
```

> L'environnement est stocké dans ton home, il persiste entre les sessions.

---

## Étape 4 — Trouver / préparer le dataset

### Vérifier les datasets déjà disponibles

```bash
ls /projects/common/
# ex: imagenet/  coco/  cifar/  ...
```

Si ton dataset est là, **utilise directement ce chemin** dans ton code :

```python
# dataset.py
DATA_ROOT = "/projects/common/imagenet"
```

### Si ton dataset n'est pas disponible

```bash
# Option A : le télécharger dans ton home (attention au quota)
ssh cluster
cd ~/mon_projet/data
wget https://...  # ou via Python script

# Option B : le déposer depuis ta machine locale
rsync -avz --progress mon_dataset.tar.gz cluster-store:~/
# Puis sur le cluster :
tar xzf mon_dataset.tar.gz -C ~/data/
```

> Si le dataset est très gros (>50 GB), demande accès à `/projects/` ou `/tsi/` via le support.

---

## Étape 5 — Tester en interactif (debug rapide)

Avant de soumettre un long job, teste que tout fonctionne avec quelques batches :

```bash
ssh cluster
sinteractive -p P100 --gpus 1 --time 1:00:00
```

Sur le nœud de calcul :

```bash
source ~/envs/mon_projet/bin/activate
cd ~/mon_projet

# Test rapide : 1 epoch, 10 batches
python train.py --epochs 1 --max-batches 10 --debug

# Vérifier que le GPU est bien utilisé
nvidia-smi
```

Si ça plante ici, tu évites de gaspiller du temps de batch.

```bash
exit   # libère le GPU dès que le debug est fini
```

---

## Étape 6 — Lancer un job batch

Crée `job.sh` dans ton projet :

```bash
#!/bin/bash
#SBATCH --job-name=mon_training
#SBATCH --output=logs/%x_%j.out      # stdout dans logs/
#SBATCH --error=logs/%x_%j.err       # stderr dans logs/
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00              # max 36h pour les étudiants
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=prenom.nom@telecom-paris.fr

# --- Setup ---
source ~/envs/mon_projet/bin/activate
cd ~/mon_projet
mkdir -p logs checkpoints

# --- Infos de debug utiles dans les logs ---
echo "=== Job $SLURM_JOB_ID démarré sur $(hostname) à $(date) ==="
echo "GPU alloué :"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Lancer l'entraînement ---
python train.py \
    --data-root /projects/common/imagenet \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --checkpoint-dir checkpoints/ \
    --log-dir logs/

echo "=== Job terminé à $(date) ==="
```

```bash
# Soumettre
sbatch job.sh

# Vérifier qu'il est dans la queue
squeue -u $USER
```

---

## Étape 7 — Suivre l'entraînement

### Lire les logs en direct

```bash
# Depuis le login node
tail -f logs/mon_training_12345.out
```

### Surveiller le GPU (si job interactif ou pour vérification)

```bash
watch -n 2 nvidia-smi
```

### TensorBoard en remote

Sur le cluster, dans ton job ou en interactif :

```bash
tensorboard --logdir logs/ --port 6006
```

Sur ta machine locale, ouvre un tunnel SSH :

```bash
ssh -L 6006:localhost:6006 cluster
# Puis ouvre http://localhost:6006 dans ton navigateur
```

---

## Étape 8 — Sauvegarder des checkpoints

Dans ton code Python, sauvegarde régulièrement pour ne pas tout perdre si le job est interrompu :

```python
# train.py
import torch, os

CHECKPOINT_DIR = "checkpoints/"

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        ckpt = torch.load(path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"], ckpt["loss"]
    return 0, None

# Dans la boucle d'entraînement :
for epoch in range(start_epoch, total_epochs):
    train_one_epoch(...)
    if epoch % 5 == 0:   # sauvegarde toutes les 5 epochs
        save_checkpoint(model, optimizer, epoch, loss,
                        f"{CHECKPOINT_DIR}/epoch_{epoch}.pt")
```

---

## Étape 9 — Récupérer les résultats

```bash
# Depuis ta machine locale
rsync -avz cluster-store:~/mon_projet/checkpoints/ ./checkpoints/
rsync -avz cluster-store:~/mon_projet/logs/        ./logs/
```

---

## Étape 10 — Modifier le code et relancer

Le workflow quotidien se résume à :

```bash
# 1. Modifier le code en local (VS Code, etc.)
# 2. Sync
rsync -avz --exclude='__pycache__/' mon_projet/ cluster-store:~/mon_projet/

# 3. Relancer le job
ssh cluster
sbatch ~/mon_projet/job.sh

# 4. Suivre les logs
tail -f ~/mon_projet/logs/mon_training_*.out
```

Ou utilise **VS Code Remote SSH** pour éditer directement sur le cluster sans sync manuel.

---

## Cheatsheet quotidien

```bash
# Connexion
ssh cluster

# Statut de mes jobs
squeue -u $USER

# GPU dispo sur une partition
squeue -p A40

# Session de debug
sinteractive -p P100 --gpus 1

# Soumettre
sbatch job.sh

# Logs en direct
tail -f logs/mon_training_*.out

# Annuler un job
scancel <JOBID>

# Mon fairshare (> 0.5 = priorité haute)
sshare -u $USER
```
