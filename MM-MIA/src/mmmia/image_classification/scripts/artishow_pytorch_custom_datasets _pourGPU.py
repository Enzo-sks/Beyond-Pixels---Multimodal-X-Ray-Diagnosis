# -*- coding: utf-8 -*-
"""
04. PyTorch Custom Datasets - Artishow
TinyVGG avec PyTorch Lightning — compatible cluster GPU (Telecom Paris)

Installation :
    pip install torch torchvision pytorch-lightning

Lancement local :
    python artishow_lightning.py

Lancement sur cluster SLURM (exemple) :
    sbatch job.sh   # voir le template SLURM en bas de fichier
"""

# ─── Imports ────────────────────────────────────────────────────────────────

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─── Config ─────────────────────────────────────────────────────────────────

BATCH_SIZE = 12
NUM_WORKERS = 4   # Sur cluster : augmenter (ex. 4-8). Sur Windows local : mettre 0
NUM_EPOCHS = 5
LEARNING_RATE = 0.01

TRAIN_DIR = "train"
TEST_DIR = "test"

# ─── Dataset & DataModule ────────────────────────────────────────────────────


def safe_loader(path: str) -> Image.Image | None:
    """Charge une image en ignorant les fichiers corrompus."""
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        print(f"⚠️  Image corrompue ignorée : {path}")
        return None


class ArtiShowDataModule(pl.LightningDataModule):
    """
    LightningDataModule : encapsule la création des datasets et dataloaders.
    Lightning appelle automatiquement setup() avant fit() sur le bon process.
    """

    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_names: List[str] = []

        self.train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            # transforms.TrivialAugmentWide(num_magnitude_bins=31),
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def setup(self, stage: str = None):
        self.train_data = datasets.ImageFolder(
            root=self.train_dir,
            transform=self.train_transform,
            loader=safe_loader,
        )
        self.test_data = datasets.ImageFolder(
            root=self.test_dir,
            transform=self.test_transform,
            loader=safe_loader,
        )
        self.class_names = self.train_data.classes
        print(f"Classes : {self.class_names}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,   # accélère le transfert CPU → GPU
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ─── Model (LightningModule) ─────────────────────────────────────────────────


class TinyVGGLightning(pl.LightningModule):
    """
    TinyVGG wrappé dans un LightningModule.

    Lightning gère automatiquement :
      - le .to(device) de tous les tenseurs
      - optimizer.zero_grad() / loss.backward() / optimizer.step()
      - model.train() / model.eval()
      - le logging (TensorBoard, CSV, etc.)
    """

    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
        learning_rate: float = LEARNING_RATE,
    ):
        super().__init__()
        self.save_hyperparameters()  # sauvegarde les hparams dans le checkpoint
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 13 * 13, output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        X, y = batch
        logits = self(X)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        # prog_bar=True affiche la métrique dans la barre de progression
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def plot_loss_curves(trainer: pl.Trainer) -> None:
    """
    Reconstruit les courbes loss/accuracy à partir des métriques loggées
    par le CSVLogger de Lightning (fichier metrics.csv).
    """
    import pandas as pd

    log_dir = Path(trainer.logger.log_dir) / "metrics.csv"
    if not log_dir.exists():
        print("Pas de fichier metrics.csv trouvé.")
        return

    df = pd.read_csv(log_dir)

    # Lightning log par step ET par epoch ; on garde seulement les lignes epoch
    df_epoch = df.dropna(subset=["epoch"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for col, ax, title in [
        (["train_loss", "val_loss"], ax1, "Loss"),
        (["train_acc",  "val_acc"],  ax2, "Accuracy"),
    ]:
        for c in col:
            data = df_epoch[["epoch", c]].dropna()
            ax.plot(data["epoch"], data[c], label=c)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150)
    plt.show()


def pred_and_plot_image(
    model: TinyVGGLightning,
    image_path: str,
    class_names: List[str],
    device: str = "cpu",
) -> None:
    """Prédit et affiche la classe d'une image custom."""
    transform = transforms.Compose([transforms.Resize((64, 64))])

    image = torchvision.io.read_image(image_path).type(torch.float32) / 255.0
    image = transform(image)

    model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(image.unsqueeze(0).to(device))

    probs = torch.softmax(logits, dim=1)
    label = torch.argmax(probs, dim=1).cpu()

    plt.imshow(image.permute(1, 2, 0))
    plt.title(f"Pred: {class_names[label]} | Prob: {probs.max():.3f}")
    plt.axis(False)
    plt.show()
    print(f"Probabilities: {probs} | Classes: {class_names}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # DataModule
    dm = ArtiShowDataModule(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    dm.setup()

    # Modèle
    model = TinyVGGLightning(
        input_shape=3,
        hidden_units=10,
        output_shape=len(dm.class_names),
        learning_rate=LEARNING_RATE,
    )

    # Trainer
    # accelerator="gpu"  → utilise le(s) GPU détecté(s) automatiquement
    # devices=1          → 1 GPU. Mettre -1 pour tous les GPUs dispo sur le nœud
    # precision="16-mixed" → active l'AMP (entraînement moitié-précision, ~2x plus rapide)
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",   # "gpu" sur le cluster, "cpu" en fallback local
        devices="auto",
        precision="16-mixed", # retirer si le cluster ne supporte pas l'AMP
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)

    # Courbes
    plot_loss_curves(trainer)

    # Inférence sur une image custom
    custom_image_path = "drive/MyDrive/artishow/Dataset/test/Atelectasis/CXR1053_IM-0040-3003.png"
    pred_and_plot_image(
        model=model,
        image_path=custom_image_path,
        class_names=dm.class_names,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


# ─── Template SLURM ──────────────────────────────────────────────────────────
# Copier dans un fichier job.sh et soumettre avec : sbatch job.sh
#
# #!/bin/bash
# #SBATCH --job-name=artishow
# #SBATCH --gres=gpu:1          # nombre de GPUs demandés
# #SBATCH --cpus-per-task=4     # doit correspondre à NUM_WORKERS
# #SBATCH --mem=16G
# #SBATCH --time=02:00:00
# #SBATCH --output=logs/%j.out
# #SBATCH --error=logs/%j.err
#
# source ~/.bashrc
# conda activate mon_env          # ou : module load python pytorch
#
# python artishow_lightning.py