"""train.py — Entraînement two-phase du modèle MultimodalFusion.

Phase 1 (epochs 1-3)  : Backbones gelés. Seuls cross-attention + tête sont entraînés.
                         LR warmup linéaire 0.1 → 1.0 × LR_HEAD.
Phase 2 (epochs 4-30) : Backbones dégelés à LR_BACK = 0.1 × LR_HEAD.
                         Cosine annealing jusqu'à 1e-6.

Usage (Colab ou local):
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
from transformers import ViTModel
from tokenizers import Tokenizer
from sklearn.metrics import roc_auc_score

# ── Chemins vers les modules existants ──────────────────────────────────────
# Convention du repo (cf. text_classification/scripts) : on insère le dossier du
# module courant + celui de text_classification dans sys.path, puis import bare.
ROOT = os.path.dirname(os.path.abspath(__file__))
MM_ROOT = os.path.dirname(ROOT)                          # multimodal_fusion (common/)
TEXT_ROOT = os.path.join(ROOT, "..", "..", "text_classification")
sys.path.insert(0, ROOT)        # model.py (local)
sys.path.insert(0, MM_ROOT)     # common/ (partagé)
sys.path.insert(0, TEXT_ROOT)   # models/ (BERTForMLM)

from models import BERTForMLM    # noqa: E402  (import après sys.path)
from model import MultimodalFusion
from common.data import (FusionDataset, fusion_collate, load_paired_df,  # noqa: E402
                         resolve_label_cols, build_groups, grouped_train_val_test)
from common.transforms import TRAIN_TF, VAL_TF           # noqa: E402
from common.losses import AsymmetricLoss                 # noqa: E402


# ── Boucle d'entraînement ────────────────────────────────────────────────────

def _apply_modality(ids, pixels, modality):
    """Ablation de modalité : neutralise l'entrée non utilisée.

    - 'image' : texte effacé (ids → PAD=0) → mesure la perf SANS le rapport,
                i.e. ce que vaut vraiment la branche image (baseline anti-leak).
    - 'text'  : image effacée (pixels → 0) → perf texte seul.
    - 'fusion': aucune modification (défaut).
    """
    if modality == "image":
        ids = torch.zeros_like(ids)
    elif modality == "text":
        pixels = torch.zeros_like(pixels)
    return ids, pixels


def train_epoch(model, loader, optimizer, criterion, device, modality="fusion"):
    model.train()
    total = 0.0
    for ids, pixels, labels in loader:
        ids, pixels, labels = ids.to(device), pixels.to(device), labels.to(device)
        ids, pixels = _apply_modality(ids, pixels, modality)
        loss = criterion(model(ids, pixels), labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, modality="fusion"):
    model.eval()
    total, all_probs, all_labels = 0.0, [], []
    for ids, pixels, labels in loader:
        ids, pixels, labels = ids.to(device), pixels.to(device), labels.to(device)
        ids, pixels = _apply_modality(ids, pixels, modality)
        logits = model(ids, pixels)
        total += criterion(logits, labels).item()
        all_probs.append(logits.sigmoid().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs  = np.vstack(all_probs)
    gt     = np.vstack(all_labels)
    valid  = [i for i in range(gt.shape[1]) if len(np.unique(gt[:, i])) > 1]
    auc    = np.nanmean([roc_auc_score(gt[:, i], probs[:, i]) for i in valid])
    return total / len(loader), auc


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODE       = 21              # arch_1 : jeu de labels complet IU-XRay (ordre canonique)
    D_MODEL    = 512
    N_HEADS    = 8
    DROPOUT    = 0.1
    LR_HEAD    = 5e-5
    LR_BACK    = LR_HEAD * 0.1   # backbone dégelé à 1/10 du LR tête
    EPOCHS     = 30
    WARMUP     = 3                # Phase 1 : 3 epochs gel
    BATCH      = args.batch_size
    PATIENCE   = 7
    NUM_WORKERS = 2

    CKPT_DIR = os.path.join(ROOT, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)
    CKPT_NAME = "multimodal_fusion_no_i2t.pt" if args.no_i2t else "multimodal_fusion.pt"
    CKPT_PATH = os.path.join(CKPT_DIR, CKPT_NAME)

    TEXT_CKPT = args.text_ckpt or os.path.join(TEXT_ROOT, "checkpoints")

    print(f"Device : {DEVICE}")
    print(f"Image dir : {args.image_dir}")

    # ── Données ──────────────────────────────────────────────────────────
    df = load_paired_df(args.csv)
    label_cols = resolve_label_cols(df, MODE)   # ordre canonique (source : text_classification)
    N_LABELS = len(label_cols)
    print(f"Dataset total : {len(df)} images | {N_LABELS} labels")

    # Split GROUPÉ par texte indication+findings (70 / 15 / 15) — aucune image
    # d'un même rapport, ni template dupliqué, partagé entre train/val/test.
    groups = build_groups(df)
    train_idx, val_idx, test_idx = grouped_train_val_test(groups, train=0.70, val=0.15)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}  (split groupé, modalité={args.modality})")

    tok = Tokenizer.from_file(os.path.join(TEXT_CKPT, "tokenizer.json"))

    train_ds = FusionDataset(train_df, label_cols, tok, args.image_dir, TRAIN_TF)
    val_ds   = FusionDataset(val_df,   label_cols, tok, args.image_dir, VAL_TF)
    test_ds  = FusionDataset(test_df,  label_cols, tok, args.image_dir, VAL_TF)

    train_loader = DataLoader(train_ds, BATCH, shuffle=True,
                              collate_fn=fusion_collate, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH, shuffle=False,
                              collate_fn=fusion_collate, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  BATCH, shuffle=False,
                              collate_fn=fusion_collate, num_workers=NUM_WORKERS, pin_memory=True)

    # ── Modèles pré-entraînés ─────────────────────────────────────────────
    bert = BERTForMLM(tok.get_vocab_size(), 256, 8, 6, 512)
    bert.load_state_dict(
        torch.load(os.path.join(TEXT_CKPT, "bert_pretrained.pt"),
                   map_location=DEVICE, weights_only=True)
    )
    text_encoder = bert.encoder
    print(f"Text encoder chargé : {sum(p.numel() for p in text_encoder.parameters()):,} params")

    vit = ViTModel.from_pretrained("codewithdark/vit-chest-xray")

    # Charger les poids fine-tunés du ViT si disponibles
    if args.vit_checkpoint and os.path.exists(args.vit_checkpoint):
        state = torch.load(args.vit_checkpoint, map_location=DEVICE, weights_only=True)
        # Extraire uniquement les paramètres ViT (sans la tête de classif)
        vit_state = {k.replace("vit.", ""): v for k, v in state.items() if k.startswith("vit.")}
        missing, unexpected = vit.load_state_dict(vit_state, strict=False)
        print(f"ViT checkpoint chargé — missing: {len(missing)}, unexpected: {len(unexpected)}")
    else:
        print("ViT : poids HuggingFace (pas de checkpoint fine-tuné fourni)")

    model = MultimodalFusion(
        text_encoder=text_encoder, vit=vit,
        n_labels=N_LABELS, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT,
        use_i2t=not args.no_i2t,
    ).to(DEVICE)
    if args.no_i2t:
        print("Variante leakage-resistant : direction Image→Texte (i2t) désactivée.")

    total     = sum(p.numel() for p in model.parameters())
    print(f"Modèle total : {total:,} paramètres")

    # ── Phase 1 : backbones gelés ─────────────────────────────────────────
    def freeze_backbones():
        for p in model.text_encoder.parameters(): p.requires_grad = False
        for p in model.vit.parameters():          p.requires_grad = False

    def unfreeze_backbones():
        for p in model.text_encoder.parameters(): p.requires_grad = True
        for p in model.vit.parameters():          p.requires_grad = True

    freeze_backbones()
    head_params = (
        list(model.cross_attn.parameters()) +
        list(model.image_pool_w.parameters()) +
        list(model.head.parameters())
    )
    trainable = sum(p.numel() for p in head_params)
    print(f"Phase 1 — entraînables : {trainable:,} (cross-attn + tête)")

    optimizer = optim.AdamW(head_params, lr=LR_HEAD, weight_decay=1e-4)

    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP]
    )

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

    best_auc, no_improve = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    # ── Boucle principale ─────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        # Passage en Phase 2
        if epoch == WARMUP + 1:
            print(f"\n[Epoch {epoch}] → Phase 2 : backbone dégelé (LR={LR_BACK:.1e})")
            unfreeze_backbones()
            backbone_params = (
                list(model.text_encoder.parameters()) +
                list(model.vit.parameters())
            )
            optimizer.add_param_group({
                "params": backbone_params,
                "lr": LR_BACK,
                "weight_decay": 1e-4,
            })
            # Étendre les base_lrs du cosine scheduler pour inclure le backbone
            cosine_sched.base_lrs.append(LR_BACK)

        train_loss         = train_epoch(model, train_loader, optimizer, criterion, DEVICE, args.modality)
        val_loss, val_auc  = evaluate(model, val_loader, criterion, DEVICE, args.modality)
        current_lr         = scheduler.get_last_lr()[0]
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
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  => Nouveau meilleur AUC {best_auc:.4f} — sauvegardé")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping à l'epoch {epoch}")
                break

    # ── Évaluation finale ─────────────────────────────────────────────────
    print("\n=== Évaluation sur le test set ===")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    test_loss, test_auc = evaluate(model, test_loader, criterion, DEVICE, args.modality)
    print(f"Test Loss : {test_loss:.4f}  |  Test Mean AUC : {test_auc:.4f}")
    print(f"\nMeilleur val AUC : {best_auc:.4f}")
    print(f"Checkpoint : {CKPT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=os.path.join(ROOT, "..", "image_preprocess", "dataset_labeled.csv"),
        help="Chemin vers dataset_labeled.csv",
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Répertoire contenant les fichiers PNG (ex. /content/Png)",
    )
    parser.add_argument(
        "--vit_checkpoint",
        default=None,
        help="Chemin vers les poids fine-tunés de ViTChestClassifier (best_vit_chest_04.pth)",
    )
    parser.add_argument(
        "--text_ckpt",
        default=None,
        help="Répertoire contenant bert_pretrained.pt et tokenizer.json "
             "(défaut : Text classification/checkpoints/)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
    )
    args = parser.parse_args()
    main(args)
