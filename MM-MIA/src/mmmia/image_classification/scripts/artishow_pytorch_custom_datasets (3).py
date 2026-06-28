# -*- coding: utf-8 -*-
"""
04. PyTorch Custom Datasets - Artishow
TinyVGG avec PyTorch Lightning — version locale (CPU)

Installation :
    pip install torch torchvision pytorch-lightning torchmetrics scikit-learn pandas matplotlib

Lancement :
    python artishow_lightning.py
"""

# ─── Imports ────────────────────────────────────────────────────────────────

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import roc_curve, auc as sk_auc
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import AUROC
from torchvision import datasets, transforms

# ─── Config ─────────────────────────────────────────────────────────────────

BATCH_SIZE = 12
NUM_WORKERS = 0    # 0 obligatoire sur Windows (pas de fork)
NUM_EPOCHS = 5
LEARNING_RATE = 0.01

TRAIN_DIR = "train"
TEST_DIR = "test"

# ─── Dataset & DataModule ────────────────────────────────────────────────────


def safe_loader(path: str) -> Image.Image:
    """Charge une image. Lève une exception si le fichier est corrompu
    (ImageFolder filtrera automatiquement ces entrées via is_valid_file)."""
    return Image.open(path).convert("RGB")


def is_valid_image(path: str) -> bool:
    """Retourne False si l'image est corrompue ou illisible, True sinon.
    Passé à ImageFolder via is_valid_file= pour exclure les fichiers invalides
    avant que les transforms ne soient appliquées.
    """
    try:
        Image.open(path).verify()  # vérifie l'intégrité sans décoder l'image entière
        return True
    except Exception:
        print(f"⚠️  Image corrompue ignorée : {path}")
        return False


class ArtiShowDataModule(pl.LightningDataModule):
    """Encapsule datasets et dataloaders."""

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
            is_valid_file=is_valid_image,
        )
        self.test_data = datasets.ImageFolder(
            root=self.test_dir,
            transform=self.test_transform,
            loader=safe_loader,
            is_valid_file=is_valid_image,
        )
        self.class_names = self.train_data.classes
        print(f"Classes : {self.class_names}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# ─── Model (LightningModule) ─────────────────────────────────────────────────


class TinyVGGLightning(pl.LightningModule):
    """TinyVGG wrappé dans un LightningModule."""

    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
        learning_rate: float = LEARNING_RATE,
    ):
        super().__init__()
        self.save_hyperparameters()
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

        # average="macro" : moyenne non pondérée sur toutes les classes
        # (pertinent si dataset déséquilibré, comme en imagerie médicale)
        self.train_auroc = AUROC(task="multiclass", num_classes=output_shape, average="macro")
        self.val_auroc   = AUROC(task="multiclass", num_classes=output_shape, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        X, y = batch
        logits = self(X)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_epoch=True)

        if stage == "train":
            self.train_auroc.update(probs, y)
        else:
            self.val_auroc.update(probs, y)

        return loss

    def on_train_epoch_end(self):
        self.log("train_auc", self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        # try/except : protection contre le sanity check de Lightning qui lance
        # une mini-validation avant l'epoch 1 et peut ne voir qu'une seule classe
        try:
            self.log("val_auc", self.val_auroc.compute(), prog_bar=True)
        except Exception:
            pass
        self.val_auroc.reset()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _find_col(df: "pd.DataFrame", candidates: List[str]) -> str | None:
    """Retourne le premier nom de colonne trouvé dans le DataFrame parmi les candidats.
    Lightning peut nommer les métriques 'train_loss', 'train_loss_epoch', etc.
    selon la version et la config — on cherche donc par préfixe.
    """
    for name in candidates:
        if name in df.columns:
            return name
    # fallback : cherche par préfixe (ex. 'train_loss_epoch' pour 'train_loss')
    for name in candidates:
        matches = [c for c in df.columns if c.startswith(name)]
        if matches:
            return matches[0]
    return None


def plot_training_curves(trainer: pl.Trainer) -> None:
    """Trace Loss, Accuracy et AUC par epoch à partir du CSV loggé par Lightning."""
    log_dir = Path(trainer.logger.log_dir) / "metrics.csv"
    if not log_dir.exists():
        print(f"Pas de fichier metrics.csv trouvé dans {log_dir}")
        return

    df = pd.read_csv(log_dir)
    df_epoch = df.dropna(subset=["epoch"])

    print("Colonnes disponibles dans metrics.csv :", df.columns.tolist())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        (["train_loss", "train_loss_epoch"], ["val_loss"], "Loss"),
        (["train_acc",  "train_acc_epoch"],  ["val_acc"],  "Accuracy"),
        (["train_auc",  "train_auc_epoch"],  ["val_auc"],  "AUC (macro)"),
    ]
    for (train_candidates, val_candidates, title), ax in zip(specs, axes):
        for candidates, label in [(train_candidates, train_candidates[0]), (val_candidates, val_candidates[0])]:
            col = _find_col(df_epoch, candidates)
            if col is None:
                print(f"  ⚠️  Colonne introuvable pour {label}, ignorée.")
                continue
            data = df_epoch[["epoch", col]].dropna()
            ax.plot(data["epoch"], data[col], label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


def plot_roc_curves(
    model: TinyVGGLightning,
    dataloader: DataLoader,
    class_names: List[str],
) -> None:
    """Trace la courbe ROC one-vs-rest pour chaque classe sur le jeu de validation."""
    all_probs, all_labels = [], []

    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            probs = torch.softmax(model(X), dim=1)
            all_probs.append(probs)
            all_labels.append(y)

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    num_classes = len(class_names)

    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5))
    if num_classes == 1:
        axes = [axes]

    for i, (ax, cls) in enumerate(zip(axes, class_names)):
        y_bin = (all_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, all_probs[:, i])
        roc_auc = sk_auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.set_title(f"ROC — {cls}")
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    plt.show()


def pred_and_plot_image(
    model: TinyVGGLightning,
    image_path: str,
    class_names: List[str],
) -> None:
    """Prédit et affiche la classe d'une image custom."""
    transform = transforms.Compose([transforms.Resize((64, 64))])

    image = torchvision.io.read_image(image_path).type(torch.float32) / 255.0
    image = transform(image)

    model.eval()
    with torch.inference_mode():
        logits = model(image.unsqueeze(0))

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

    # Trainer — CPU local
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="cpu",
        num_sanity_val_steps=0,  # évite le crash AUC avant l'epoch 1
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=dm)

    # Courbes loss / accuracy / AUC
    plot_training_curves(trainer)

    # Courbes ROC par classe
    plot_roc_curves(
        model=model,
        dataloader=dm.val_dataloader(),
        class_names=dm.class_names,
    )

    # Inférence sur une image custom
    custom_image_path = "Dataset/test/Atelectasis/CXR1053_IM-0040-3003.png"
    pred_and_plot_image(
        model=model,
        image_path=custom_image_path,
        class_names=dm.class_names,
    )