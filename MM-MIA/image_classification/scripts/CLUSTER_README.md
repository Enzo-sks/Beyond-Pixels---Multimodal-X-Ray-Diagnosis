# Artishow — TinyVGG avec PyTorch Lightning
hello
Classification d'images médicales (radiographies thoraciques) avec un réseau TinyVGG, entraîné via PyTorch Lightning. Compatible entraînement local et cluster GPU (SLURM / Telecom Paris).

---

## Table des matières

1. [Architecture du projet](#architecture-du-projet)
2. [Installation](#installation)
3. [Structure des données](#structure-des-données)
4. [Utilisation](#utilisation)
5. [Configuration](#configuration)
6. [Description des composants](#description-des-composants)
7. [Résultats et visualisation](#résultats-et-visualisation)
8. [Lancement sur cluster SLURM](#lancement-sur-cluster-slurm)
9. [Bug connu](#bug-connu)

---

## Architecture du projet

```
artishow/
├── artishow_lightning.py   # Script principal
├── requirements.txt        # Dépendances Python
├── job.sh                  # Template SLURM (optionnel)
├── train/                  # Données d'entraînement (ImageFolder)
│   ├── Atelectasis/
│   ├── Cardiomegaly/
│   └── ...
└── test/                   # Données de test (ImageFolder)
    ├── Atelectasis/
    ├── Cardiomegaly/
    └── ...
```

---

## Installation

### Prérequis

- Python >= 3.10
- pip

### Installer les dépendances

```bash
pip install -r requirements.txt
```

### Avec CUDA (cluster GPU, Linux)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning pandas matplotlib Pillow
```

---

## Structure des données

Le script utilise `torchvision.datasets.ImageFolder`. Chaque sous-dossier de `train/` et `test/` correspond à une classe.

```
train/
├── Atelectasis/
│   ├── image_001.png
│   └── ...
├── Cardiomegaly/
│   └── ...
test/
├── Atelectasis/
│   └── ...
└── Cardiomegaly/
    └── ...
```

> Les images corrompues sont ignorées automatiquement grâce au `safe_loader`.

---

## Utilisation

### Entraînement local

```bash
python artishow_lightning.py
```

Lightning détecte automatiquement le device disponible (CPU, GPU CUDA, Apple MPS).

### Inférence sur une image custom

Modifier la variable `custom_image_path` en bas du script :

```python
custom_image_path = "chemin/vers/mon/image.png"
```

---

## Configuration

Tous les hyperparamètres sont centralisés en haut du fichier :

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `BATCH_SIZE` | `12` | Taille du batch |
| `NUM_WORKERS` | `4` | Workers DataLoader (mettre `0` sur Windows) |
| `NUM_EPOCHS` | `5` | Nombre d'époques |
| `LEARNING_RATE` | `0.01` | Taux d'apprentissage (Adam) |
| `TRAIN_DIR` | `"train"` | Chemin vers les données d'entraînement |
| `TEST_DIR` | `"test"` | Chemin vers les données de test |

---

## Description des composants

### `safe_loader`

Charge les images PIL en ignorant les fichiers corrompus (`UnidentifiedImageError`), évitant ainsi l'interruption de l'entraînement sur un dataset bruité.

### `ArtiShowDataModule` (`pl.LightningDataModule`)

Encapsule la création des datasets et des DataLoaders. Lightning appelle `setup()` automatiquement avant `fit()`, ce qui garantit la compatibilité avec l'entraînement multi-GPU.

**Transformations appliquées :**

- **Train** : redimensionnement à 64×64 pixels, conversion en tenseur. *(La ligne `TrivialAugmentWide` est commentée mais disponible pour activer l'augmentation de données.)*
- **Test** : redimensionnement à 64×64 pixels, conversion en tenseur uniquement.

### `TinyVGGLightning` (`pl.LightningModule`)

Implémentation du réseau TinyVGG wrappé dans un `LightningModule`.

**Architecture :**

```
Input (3, 64, 64)
    │
    ▼
Conv Block 1 : Conv2d(3→H) → ReLU → Conv2d(H→H) → ReLU → MaxPool2d
    │
    ▼
Conv Block 2 : Conv2d(H→H) → ReLU → Conv2d(H→H) → ReLU → MaxPool2d
    │
    ▼
Classifier   : Flatten → Linear(H×13×13 → N_classes)
```

`H` = `hidden_units` (10 par défaut), `N_classes` = nombre de dossiers dans `train/`.

**Ce que Lightning gère automatiquement :**

- Déplacement des tenseurs sur le bon device (`.to(device)`)
- Boucle d'optimisation (`zero_grad` / `backward` / `step`)
- Basculement `train()` / `eval()`
- Logging (TensorBoard, CSV Logger)

### `plot_loss_curves`

Lit le fichier `metrics.csv` généré par le `CSVLogger` de Lightning et trace les courbes de loss et d'accuracy par époque. Le graphique est sauvegardé dans `loss_curves.png`.

### `pred_and_plot_image`

Charge une image depuis le disque, la prétraite, passe le modèle en mode inférence (`torch.inference_mode()`), et affiche la prédiction avec la probabilité associée.

---

## Résultats et visualisation

Après l'entraînement, deux sorties sont générées :

- `loss_curves.png` — courbes train/val loss et accuracy par époque
- Affichage matplotlib de la prédiction sur l'image custom

Les logs détaillés (par step et par époque) sont disponibles dans le dossier `lightning_logs/`.

---

## Lancement sur cluster SLURM

Un template de job SLURM est fourni en commentaire en bas du script. Pour l'utiliser :

1. Copier le bloc commenté dans un fichier `job.sh`
2. Adapter le nom de l'environnement conda (`mon_env`) et les ressources demandées
3. Soumettre le job :

```bash
sbatch job.sh
```

**Paramètres SLURM à ajuster selon le cluster :**

| Directive | Valeur exemple | Description |
|---|---|---|
| `--gres=gpu:1` | `gpu:1` ou `gpu:4` | Nombre de GPUs |
| `--cpus-per-task` | `4` | Doit correspondre à `NUM_WORKERS` |
| `--mem` | `16G` | RAM par nœud |
| `--time` | `02:00:00` | Durée max du job |

> Pour utiliser plusieurs GPUs, passer `devices=-1` dans le `Trainer` et `--gres=gpu:N` dans le job SLURM.

---
