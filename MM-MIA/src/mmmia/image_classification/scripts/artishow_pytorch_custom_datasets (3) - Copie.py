"""
ArtiShow — Modèle amélioré
==========================
Problèmes identifiés dans la version originale :
  - TinyVGG trop petit (10 hidden units) → underfitting massif
  - Pas d'augmentation de données → mauvaise généralisation
  - Learning rate fixe sans scheduler → convergence lente
  - AUC ~0.50 = modèle aléatoire → le modèle n'apprend rien

Solutions appliquées :
  1. Transfer Learning avec EfficientNet-B0 (pré-entraîné sur ImageNet)
     → feature extractor puissant, même avec peu de données
  2. Fine-tuning progressif : d'abord classifieur seul, puis débloque tout
  3. Augmentation forte : RandomHFlip, ColorJitter, RandomRotation, Cutout
  4. Scheduler CosineAnnealingLR + warmup
  5. Class weights automatiques si dataset déséquilibré
  6. Early stopping + ModelCheckpoint sur val_auc
  7. Images 224×224 (taille attendue par EfficientNet)
  8. Label smoothing dans CrossEntropyLoss

Installation :
    pip install torch torchvision pytorch-lightning torchmetrics scikit-learn pandas matplotlib

Lancement :
    python artishow_improved.py
"""

# ─── Imports ────────────────────────────────────────────────────────────────

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import AUROC, Accuracy, F1Score
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ─── Config ─────────────────────────────────────────────────────────────────

BATCH_SIZE   = 32          # plus grand = gradient plus stable
NUM_WORKERS  = 0           # 0 sur Windows
NUM_EPOCHS   = 2          # early stopping s'active avant si plateau
LR_HEAD      = 3e-3        # lr pour le classifieur seul (phase 1)
LR_FINETUNE  = 5e-5        # lr pour le fine-tuning complet (phase 2)
FREEZE_EPOCHS = 5          # nb d'epochs avec backbone gelé
IMAGE_SIZE   = 224         # taille attendue par EfficientNet

TRAIN_DIR = "train"
TEST_DIR  = "test"

# ─── Helpers dataset ────────────────────────────────────────────────────────

def is_valid_image(path: str) -> bool:
    try:
        Image.open(path).verify()
        return True
    except Exception:
        print(f"⚠️  Image corrompue ignorée : {path}")
        return False


def safe_loader(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ─── DataModule ─────────────────────────────────────────────────────────────

class ArtiShowDataModule(pl.LightningDataModule):
    """
    DataModule avec :
    - Augmentation forte sur le train
    - Normalisation ImageNet (obligatoire pour EfficientNet pré-entraîné)
    - WeightedRandomSampler si classes déséquilibrées
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        train_dir: str = TRAIN_DIR,
        test_dir: str  = TEST_DIR,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        image_size: int = IMAGE_SIZE,
        use_weighted_sampler: bool = True,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir  = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.use_weighted_sampler = use_weighted_sampler
        self.class_names: List[str] = []
        self.class_weights: Optional[torch.Tensor] = None

        # ── Augmentation train (forte) ───────────────────────────────────
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Cutout
        ])

        # ── Transform validation (pas d'augmentation) ────────────────────
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def setup(self, stage: str = None):
        self.train_data = datasets.ImageFolder(
            root=self.train_dir,
            transform=self.train_transform,
            loader=safe_loader,
            is_valid_file=is_valid_image,
        )
        self.val_data = datasets.ImageFolder(
            root=self.test_dir,
            transform=self.val_transform,
            loader=safe_loader,
            is_valid_file=is_valid_image,
        )
        self.class_names = self.train_data.classes
        n_classes = len(self.class_names)
        print(f"\n📂 Classes ({n_classes}) : {self.class_names}")
        print(f"   Train : {len(self.train_data)} images")
        print(f"   Val   : {len(self.val_data)} images\n")

        # ── Class weights (pour CrossEntropyLoss + dataset déséquilibré) ─
        labels = [s[1] for s in self.train_data.samples]
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(n_classes),
            y=labels,
        )
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"   Class weights : {dict(zip(self.class_names, weights.round(3)))}\n")

        # ── WeightedRandomSampler (sur-échantillonne les classes rares) ──
        if self.use_weighted_sampler:
            sample_weights = torch.tensor([weights[l] for l in labels])
            self._sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
        else:
            self._sampler = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            sampler=self._sampler,
            shuffle=(self._sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ─── Model ──────────────────────────────────────────────────────────────────

class ArtiShowModel(pl.LightningModule):
    """
    EfficientNet-B0 pré-entraîné, fine-tuné en 2 phases :
      Phase 1 (epochs 0..FREEZE_EPOCHS-1) : backbone gelé, seul le classifieur s'entraîne
      Phase 2 (epochs FREEZE_EPOCHS..)    : tout le réseau s'entraîne avec lr très faible
    """

    def __init__(
        self,
        num_classes: int,
        freeze_epochs: int = FREEZE_EPOCHS,
        lr_head: float = LR_HEAD,
        lr_finetune: float = LR_FINETUNE,
        class_weights: Optional[torch.Tensor] = None,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.freeze_epochs = freeze_epochs
        self.lr_head = lr_head
        self.lr_finetune = lr_finetune
        self.num_classes = num_classes

        # ── Backbone EfficientNet-B0 pré-entraîné ────────────────────────
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Remplace le classifieur d'origine (1000 classes) par le nôtre
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        # Gèle le backbone au démarrage
        self._freeze_backbone(freeze=True)

        # ── Loss avec label smoothing + class weights ─────────────────────
        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1,   # réduit l'overconfidence
        )

        # ── Métriques torchmetrics ────────────────────────────────────────
        for split in ("train", "val"):
            setattr(self, f"{split}_auroc",
                    AUROC(task="multiclass", num_classes=num_classes, average="macro"))
            setattr(self, f"{split}_acc",
                    Accuracy(task="multiclass", num_classes=num_classes))
            setattr(self, f"{split}_f1",
                    F1Score(task="multiclass", num_classes=num_classes, average="macro"))

    # ── Freeze / unfreeze ────────────────────────────────────────────────────

    def _freeze_backbone(self, freeze: bool):
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze
        status = "gelé" if freeze else "dégelé"
        print(f"\n🔒 Backbone {status}\n")

    def on_train_epoch_start(self):
        """Débloque le backbone à l'epoch FREEZE_EPOCHS."""
        if self.current_epoch == self.freeze_epochs:
            self._freeze_backbone(freeze=False)
            # Met à jour le learning rate dans l'optimiseur
            for param_group in self.optimizers().param_groups:
                param_group["lr"] = self.lr_finetune
            print(f"  ➡️  Fine-tuning complet activé (lr={self.lr_finetune})")

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ── Shared step ──────────────────────────────────────────────────────────

    def _step(self, batch, split: str):
        X, y = batch
        logits = self(X)
        loss   = self.loss_fn(logits, y)
        probs  = torch.softmax(logits, dim=1)

        # Update métriques
        getattr(self, f"{split}_auroc").update(probs, y)
        getattr(self, f"{split}_acc").update(probs, y)
        getattr(self, f"{split}_f1").update(probs, y)

        self.log(f"{split}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def _epoch_end(self, split: str):
        for metric_name in ("auroc", "acc", "f1"):
            metric = getattr(self, f"{split}_{metric_name}")
            try:
                val = metric.compute()
                key = f"{split}_auc" if metric_name == "auroc" else f"{split}_{metric_name}"
                self.log(key, val, prog_bar=True)
            except Exception:
                pass
            metric.reset()

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    # ── Optimiseur + Scheduler ───────────────────────────────────────────────

    def configure_optimizers(self):
        """
        AdamW (meilleure régularisation que Adam) + CosineAnnealingLR
        qui réduit progressivement le lr jusqu'à presque 0 sur NUM_EPOCHS.
        """
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr_head,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ─── Visualisation ──────────────────────────────────────────────────────────

def plot_training_curves(trainer: pl.Trainer) -> None:
    log_dir = Path(trainer.logger.log_dir) / "metrics.csv"
    if not log_dir.exists():
        print(f"⚠️  metrics.csv introuvable : {log_dir}")
        return

    df = pd.read_csv(log_dir)
    df_e = df.dropna(subset=["epoch"])

    def get(df, *candidates):
        for c in candidates:
            if c in df.columns:
                return df[["epoch", c]].dropna()
            matches = [x for x in df.columns if x.startswith(c)]
            if matches:
                return df[["epoch", matches[0]]].dropna()
        return None

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("ArtiShow — Courbes d'entraînement (EfficientNet-B0)", fontsize=14)

    specs = [
        ("Loss",     "train_loss", "val_loss"),
        ("Accuracy", "train_acc",  "val_acc"),
        ("AUC macro","train_auc",  "val_auc"),
        ("F1 macro", "train_f1",   "val_f1"),
    ]
    colors = {"train": "#2B5EA7", "val": "#C1522A"}

    for ax, (title, train_key, val_key) in zip(axes, specs):
        for key, label in [(train_key, "train"), (val_key, "val")]:
            data = get(df_e, key)
            if data is not None:
                col = data.columns[1]
                ax.plot(data["epoch"], data[col],
                        label=label, color=colors[label], linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves_improved.png", dpi=150)
    plt.show()
    print("✅  Courbes sauvegardées → training_curves_improved.png")


def plot_roc_curves(
    model: ArtiShowModel,
    dataloader: DataLoader,
    class_names: List[str],
) -> None:
    all_probs, all_labels = [], []
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            probs = torch.softmax(model(X), dim=1)
            all_probs.append(probs)
            all_labels.append(y)

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    n = len(class_names)

    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, cls in enumerate(class_names):
        ax = axes[i]
        y_bin = (all_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, all_probs[:, i])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color="#C1522A", label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Faux positifs"); ax.set_ylabel("Vrais positifs")
        ax.set_title(f"ROC — {cls}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("roc_curves_improved.png", dpi=150)
    plt.show()
    print("✅  Courbes ROC sauvegardées → roc_curves_improved.png")


def show_predictions(
    model: ArtiShowModel,
    dataloader: DataLoader,
    class_names: List[str],
    n: int = 16,
) -> None:
    """Affiche une grille de prédictions avec vrai label vs prédit."""
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

    model.eval()
    images, labels, preds, confs = [], [], [], []

    with torch.inference_mode():
        for X, y in dataloader:
            logits = model(X)
            probs  = torch.softmax(logits, dim=1)
            pred   = probs.argmax(dim=1)
            conf   = probs.max(dim=1).values
            images.extend(X.cpu()); labels.extend(y.cpu())
            preds.extend(pred.cpu()); confs.extend(conf.cpu())
            if len(images) >= n:
                break

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flatten()

    for i in range(n):
        img = images[i] * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        true_cls = class_names[labels[i]]
        pred_cls = class_names[preds[i]]
        color    = "green" if labels[i] == preds[i] else "red"
        axes[i].imshow(img)
        axes[i].set_title(
            f"Vrai: {true_cls}\nPrédit: {pred_cls} ({confs[i]:.0%})",
            color=color, fontsize=8
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Prédictions sur le jeu de validation", fontsize=13)
    plt.tight_layout()
    plt.savefig("predictions_grid.png", dpi=150)
    plt.show()
    print("✅  Grille sauvegardée → predictions_grid.png")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. DataModule ────────────────────────────────────────────────────────
    dm = ArtiShowDataModule(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        use_weighted_sampler=True,
    )
    dm.setup()

    # ── 2. Modèle ────────────────────────────────────────────────────────────
    model = ArtiShowModel(
        num_classes=len(dm.class_names),
        freeze_epochs=FREEZE_EPOCHS,
        lr_head=LR_HEAD,
        lr_finetune=LR_FINETUNE,
        class_weights=dm.class_weights,
        dropout=0.4,
    )

    # ── 3. Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            monitor="val_auc",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_auc:.3f}",
            verbose=True,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=6,           # stop si pas d'amélioration pendant 6 epochs
            verbose=True,
            min_delta=0.001,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── 4. Trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",       # GPU si dispo, sinon CPU
        devices=1,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        log_every_n_steps=5,
        precision="16-mixed",     # mixed precision → 2x plus rapide sur GPU
                                  # si erreur sur CPU, remplacer par "32"
    )

    # ── 5. Entraînement ──────────────────────────────────────────────────────
    trainer.fit(model, datamodule=dm)
    print(f"\n✅  Meilleur checkpoint : {trainer.checkpoint_callback.best_model_path}")

    # ── 6. Charge le meilleur checkpoint ─────────────────────────────────────
    best_model = ArtiShowModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        num_classes=len(dm.class_names),
        class_weights=dm.class_weights,
    )
    torch.save(best_model.state_dict(), "model_full.pth")
    print("✅  Meilleur modèle sauvegardé → model_full.pth")

    # ── 7. Visualisations ────────────────────────────────────────────────────
    plot_training_curves(trainer)
    plot_roc_curves(best_model, dm.val_dataloader(), dm.class_names)
    show_predictions(best_model, dm.val_dataloader(), dm.class_names, n=16)

    # ── 8. Résumé final ──────────────────────────────────────────────────────
    results = trainer.validate(best_model, datamodule=dm, verbose=True)
    print("\n📊  Résultats finaux :")
    for k, v in results[0].items():
        print(f"   {k:20s} : {v:.4f}")