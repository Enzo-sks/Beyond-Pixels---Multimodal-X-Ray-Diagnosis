"""train.py — Entraînement du ProjectorLLM (ViT → Projector → Gemma 3 1B + LoRA).

Stratégie d'entraînement :
  Phase unique — ViT gelé, Gemma gelé sauf LoRA.
  Seuls le Projector, les matrices LoRA et la head sont entraînés.

  Pourquoi pas de 2 phases ici ?
  Le LLM pré-entraîné a déjà une représentation riche. Dégeler le ViT
  n'apporterait pas grand chose (il est déjà domain-specific via codewithdark).
  Si tu veux expérimenter, tu peux ajouter un dégel du ViT en phase 2.

Usage:
    python train.py --image_dir /content/Png --csv /content/drive/MyDrive/dataset_labeled.csv
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTModel
from tokenizers import Tokenizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
PARENT      = os.path.dirname(ROOT)
FUSION_ROOT = os.path.join(PARENT, "multimodal_fusion")

for p in [ROOT, PARENT, FUSION_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model   import build_model                                          # noqa: E402
from dataset import MultimodalDataset, multimodal_collate, LABEL_COLS, load_paired_df  # noqa: E402


# ── AsymmetricLoss (identique aux autres modules) ────────────────────────────

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss — Ben-Baruch et al. (2021)."""

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, logits, targets):
        p     = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0)

        log_p   = torch.log(p.clamp(min=self.eps))
        log_1_p = torch.log((1 - p_neg).clamp(min=self.eps))

        loss_pos = (1 - p)  ** self.gamma_pos * log_p
        loss_neg =  p_neg   ** self.gamma_neg  * log_1_p

        return -(targets * loss_pos + (1 - targets) * loss_neg).mean()


# ── Transforms (identiques à multimodal_fusion) ──────────────────────────────

VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD  = [0.5, 0.5, 0.5]

TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.85, 1.15), shear=10),
    transforms.ElasticTransform(alpha=40.0, sigma=5.0),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
])

VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
])


# ── Boucles train / eval ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total = 0.0
    for ids, pixels, labels in loader:
        ids, pixels, labels = ids.to(device), pixels.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision (AMP) pour économiser la VRAM
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(ids, pixels)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(ids, pixels)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

        total += loss.item()

    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, all_probs, all_labels = 0.0, [], []

    for ids, pixels, labels in loader:
        ids, pixels, labels = ids.to(device), pixels.to(device), labels.to(device)

        with torch.autocast(device_type=device.type if hasattr(device, "type") else "cpu",
                            dtype=torch.float16, enabled=(str(device) != "cpu")):
            logits = model(ids, pixels)

        total += criterion(logits.float(), labels).item()
        all_probs.append(logits.float().sigmoid().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs = np.vstack(all_probs)
    gt    = np.vstack(all_labels)
    valid = [i for i in range(gt.shape[1]) if len(np.unique(gt[:, i])) > 1]
    auc   = np.nanmean([roc_auc_score(gt[:, i], probs[:, i]) for i in valid])

    return total / len(loader), auc


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP     = DEVICE.type == "cuda"
    N_LABELS    = len(LABEL_COLS)
    LR          = args.lr
    EPOCHS      = args.epochs
    BATCH       = args.batch_size
    PATIENCE    = args.patience
    NUM_WORKERS = 2
    LORA_RANK   = args.lora_rank
    LORA_ALPHA  = args.lora_alpha

    CKPT_DIR  = os.path.join(ROOT, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)
    CKPT_PATH = os.path.join(CKPT_DIR, "projector_llm.pt")

    print(f"Device : {DEVICE}  |  AMP : {USE_AMP}")
    print(f"LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}")

    # ── Données ──────────────────────────────────────────────────────────
    df = load_paired_df(args.csv)
    print(f"Dataset total : {len(df)} images")

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(msss.split(df, df[LABEL_COLS]))
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(msss2.split(df.iloc[temp_idx], df.iloc[temp_idx][LABEL_COLS]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[temp_idx].iloc[val_idx].reset_index(drop=True)
    test_df  = df.iloc[temp_idx].iloc[test_idx].reset_index(drop=True)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # Tokenizer texte (BPE maison, identique aux autres modules)
    # NB: text_classification/ vit sous src/mmmia/, pas sous multimodal_fusion/.
    # PARENT = .../src/mmmia/multimodal_fusion -> il faut remonter un niveau
    # de plus pour atteindre .../src/mmmia/text_classification/checkpoints.
    MMMIA_ROOT = os.path.dirname(PARENT)  # .../src/mmmia
    TEXT_CKPT = args.text_ckpt or os.path.join(MMMIA_ROOT, "text_classification", "checkpoints")
    tok = Tokenizer.from_file(os.path.join(TEXT_CKPT, "tokenizer.json"))

    train_ds = MultimodalDataset(train_df, LABEL_COLS, tok, args.image_dir, TRAIN_TF)
    val_ds   = MultimodalDataset(val_df,   LABEL_COLS, tok, args.image_dir, VAL_TF)
    test_ds  = MultimodalDataset(test_df,  LABEL_COLS, tok, args.image_dir, VAL_TF)

    train_loader = DataLoader(train_ds, BATCH, shuffle=True,
                              collate_fn=multimodal_collate, num_workers=NUM_WORKERS, pin_memory=USE_AMP)
    val_loader   = DataLoader(val_ds,   BATCH, shuffle=False,
                              collate_fn=multimodal_collate, num_workers=NUM_WORKERS, pin_memory=USE_AMP)
    test_loader  = DataLoader(test_ds,  BATCH, shuffle=False,
                              collate_fn=multimodal_collate, num_workers=NUM_WORKERS, pin_memory=USE_AMP)

    # ── ViT ──────────────────────────────────────────────────────────────
    print("Chargement ViT (codewithdark/vit-chest-xray)...")
    vit = ViTModel.from_pretrained("codewithdark/vit-chest-xray")

    if args.vit_checkpoint and os.path.exists(args.vit_checkpoint):
        state = torch.load(args.vit_checkpoint, map_location=DEVICE, weights_only=True)
        vit_state = {k.replace("vit.", ""): v for k, v in state.items() if k.startswith("vit.")}
        missing, unexpected = vit.load_state_dict(vit_state, strict=False)
        print(f"ViT checkpoint chargé — missing: {len(missing)}, unexpected: {len(unexpected)}")
    else:
        print("ViT : poids HuggingFace (pas de checkpoint fine-tuné fourni)")

    # ── Construction du modèle complet (Projector + LoRA + head) ─────────
    model = build_model(
        vit=vit,
        gemma_name=args.gemma_name,
        n_labels=N_LABELS,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        dropout=0.1,
        device=str(DEVICE),
    )

    # ── Optimizer — uniquement les params entraînables ────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion  = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    scaler     = torch.cuda.amp.GradScaler() if USE_AMP else None

    best_auc, no_improve = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    # ── Boucle principale ─────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        train_loss        = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss, val_auc = evaluate(model, val_loader, criterion, DEVICE)
        current_lr        = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | LR {current_lr:.2e} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | AUC={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc, no_improve = val_auc, 0
            # Sauvegarde : projector + LoRA + head uniquement (pas les poids gelés)
            torch.save(
                {
                    "projector":  model.projector.state_dict(),
                    "lora":       {k: v for k, v in model.gemma.state_dict().items() if "lora_" in k},
                    "head":       model.head.state_dict(),
                    "epoch":      epoch,
                    "val_auc":    val_auc,
                },
                CKPT_PATH,
            )
            print(f"  => Nouveau meilleur AUC {best_auc:.4f} — sauvegardé ({CKPT_PATH})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping à l'epoch {epoch}")
                break

    # ── Évaluation finale ─────────────────────────────────────────────────
    print("\n=== Évaluation sur le test set ===")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    model.projector.load_state_dict(ckpt["projector"])
    model.head.load_state_dict(ckpt["head"])
    lora_state = model.gemma.state_dict()
    lora_state.update(ckpt["lora"])
    model.gemma.load_state_dict(lora_state)

    test_loss, test_auc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss : {test_loss:.4f}  |  Test Mean AUC : {test_auc:.4f}")
    print(f"\nMeilleur val AUC : {best_auc:.4f}")
    print(f"Checkpoint : {CKPT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projector + LoRA + Gemma 3 1B — CXR multi-label")
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(PARENT), "image_preprocess", "dataset_labeled.csv"),
        help="Chemin vers dataset_labeled.csv",
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Répertoire contenant les fichiers PNG",
    )
    parser.add_argument(
        "--vit_checkpoint",
        default=None,
        help="Chemin vers les poids fine-tunés du ViT (best_vit_chest_04.pth)",
    )
    parser.add_argument(
        "--text_ckpt",
        default=None,
        help="Répertoire contenant tokenizer.json",
    )
    parser.add_argument(
        "--gemma_name",
        default="google/gemma-3-1b-pt",
        help="Identifiant HuggingFace du LLM (défaut: google/gemma-3-1b-pt)",
    )
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=4,
                        help="Batch size — Gemma 3 1B est léger, peut être augmenté si la VRAM le permet")
    parser.add_argument("--patience",   type=int,   default=7)
    parser.add_argument("--lora_rank",  type=int,   default=16,
                        help="Rang des matrices LoRA (plus grand = plus expressif mais plus lourd)")
    parser.add_argument("--lora_alpha", type=int,   default=32,
                        help="Alpha LoRA (mise à l'échelle = alpha/rank)")
    args = parser.parse_args()
    main(args)