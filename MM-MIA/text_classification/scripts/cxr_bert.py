"""
+==============================================================+
|  Fine-tuning CXR-BERT-specialized (multi-label)               |
|                                                                |
|  Epoch 1  : classifier head only (encoder frozen)             |
|  Epoch 2+ : layers 10-11 + pooler + head (0-9 frozen)        |
|                                                                |
|  Usage :  python scripts/cxr_bert.py 5                        |
|           python scripts/cxr_bert.py 14                       |
|           python scripts/cxr_bert.py 21                       |
|                                                                |
|  Requiert : pip install pytorch-lightning transformers         |
+==============================================================+
"""

import os, sys, torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from sklearn.metrics import (f1_score, roc_auc_score,
                             classification_report, hamming_loss)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import CXRBertClassifier
from models.cxr_bert_classifier import CXR_BERT_NAME
from data import LabelDataset, pad_collate, load_reports

OUT_DIR  = os.path.join(ROOT, "outputs", "CXR-BERT-specialized")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ======================================================================
#  CONFIG
# ======================================================================

mode = int(sys.argv[1]) if len(sys.argv) > 1 else 21
assert mode in (5, 14, 21), "Usage : python scripts/cxr_bert.py [5|14|21]"

MAX_LEN  = 256
EPOCHS   = 30
LR       = 2e-5
BATCH    = 16
PATIENCE = 5

# ======================================================================
#  LIGHTNING MODULE
# ======================================================================

class LitCXRBert(pl.LightningModule):
    def __init__(self, n_labels, pos_weight, pad_id=0):
        super().__init__()
        self.clf = CXRBertClassifier(n_labels, pad_id)
        self.clf.freeze_encoder()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.val_probs, self.val_labels = [], []
        self.history = {"train": [], "val": []}
        self._train_losses, self._val_losses = [], []

    def forward(self, ids):
        return self.clf(ids)

    def on_train_epoch_start(self):
        # current_epoch is 0-indexed :
        #   0 → epoch 1 → head only (frozen from __init__)
        #   1 → epoch 2 → unfreeze layers 10-11 + pooler
        if self.current_epoch == 1:
            self.clf.unfreeze_top_layers()
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Epoch {self.current_epoch + 1}: {n:,} trainable params")

    def training_step(self, batch, _):
        ids, lab = batch
        loss = self.loss_fn(self(ids), lab)
        self._train_losses.append(loss.detach())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.history["train"].append(
            torch.stack(self._train_losses).mean().item())
        self._train_losses.clear()

    def validation_step(self, batch, _):
        ids, lab = batch
        logits = self(ids)
        loss = self.loss_fn(logits, lab)
        self._val_losses.append(loss.detach())
        self.log("val_loss", loss, prog_bar=True)
        self.val_probs.append(logits.sigmoid().cpu())
        self.val_labels.append(lab.cpu())

    def on_validation_epoch_end(self):
        if self._val_losses:
            self.history["val"].append(
                torch.stack(self._val_losses).mean().item())
            self._val_losses.clear()
        if self.val_probs:
            probs = torch.cat(self.val_probs).numpy()
            labs  = torch.cat(self.val_labels).numpy()
            valid = [i for i in range(labs.shape[1])
                     if len(np.unique(labs[:, i])) > 1]
            if valid:
                auc = roc_auc_score(labs[:, valid], probs[:, valid],
                                    average="macro")
                self.log("val_auc", auc, prog_bar=True)
        self.val_probs.clear()
        self.val_labels.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        return [opt], [sched]

# ======================================================================
#  DONNEES
# ======================================================================

np.random.seed(42); torch.manual_seed(42)

texts, labels_np, label_cols, pos_weight, mode_name = load_reports(
    mode=mode, text_cols="findings", pw_clip=5)

print(f"Mode    : {mode_name}")
print(f"Dataset : {len(texts)} rapports | {len(label_cols)} labels")
print(f"Labels  : {label_cols}")

hf_tok = AutoTokenizer.from_pretrained(CXR_BERT_NAME, trust_remote_code=True)
PAD_ID = hf_tok.pad_token_id

tok = hf_tok.backend_tokenizer
tok.enable_truncation(max_length=MAX_LEN)

full_ds = LabelDataset(texts, labels_np.tolist(), tok)
n_val = max(1, len(full_ds) // 5)
train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
print(f"Train   : {len(train_ds)} | Val : {n_val}")

train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=pad_collate)
val_loader   = DataLoader(val_ds,   BATCH, collate_fn=pad_collate)

# ======================================================================
#  ENTRAINEMENT
# ======================================================================

lit = LitCXRBert(n_labels=len(label_cols), pos_weight=pos_weight, pad_id=PAD_ID)

ckpt_cb = pl.callbacks.ModelCheckpoint(
    dirpath=CKPT_DIR, filename=f"cxr_bert_{mode}",
    monitor="val_loss", mode="min", save_top_k=1,
)
early_cb = pl.callbacks.EarlyStopping("val_loss", patience=PATIENCE, mode="min")

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    callbacks=[ckpt_cb, early_cb],
    enable_model_summary=True,
    num_sanity_val_steps=0,
    logger=False,
)
trainer.fit(lit, train_loader, val_loader)

# Reload best checkpoint
best = torch.load(ckpt_cb.best_model_path, weights_only=False)
lit.load_state_dict(best["state_dict"])

# ======================================================================
#  EVALUATION
# ======================================================================

lit.eval()
DEVICE = lit.device
all_probs, all_labels = [], []
with torch.no_grad():
    for ids, lab in val_loader:
        ids = ids.to(DEVICE)
        all_probs.append(lit(ids).sigmoid().cpu().numpy())
        all_labels.append(lab.numpy())

probs      = np.vstack(all_probs)
labels_arr = np.vstack(all_labels)

best_t = max(np.arange(0.1, 0.6, 0.05),
             key=lambda t: f1_score(labels_arr, (probs > t).astype(int),
                                    average="macro", zero_division=0))
preds = (probs > best_t).astype(int)

f1_mac = f1_score(labels_arr, preds, average="macro", zero_division=0)
f1_mic = f1_score(labels_arr, preds, average="micro", zero_division=0)
f1_sam = f1_score(labels_arr, preds, average="samples", zero_division=0)
h_loss = hamming_loss(labels_arr, preds)
valid  = [i for i in range(labels_arr.shape[1])
          if len(np.unique(labels_arr[:, i])) > 1]
auc_mac = roc_auc_score(labels_arr[:, valid], probs[:, valid], average="macro")
auc_mic = roc_auc_score(labels_arr[:, valid], probs[:, valid], average="micro")
auc_per = roc_auc_score(labels_arr[:, valid], probs[:, valid], average=None)
valid_names = [label_cols[i] for i in valid]

print(f"\n{'='*50}")
print(f"  {mode_name} -- CXR-BERT-specialized -- seuil={best_t:.2f}")
print(f"{'='*50}")
print(f"  F1 macro   : {f1_mac:.4f}")
print(f"  F1 micro   : {f1_mic:.4f}")
print(f"  F1 samples : {f1_sam:.4f}")
print(f"  AUC macro  : {auc_mac:.4f}")
print(f"  AUC micro  : {auc_mic:.4f}")
print(f"  Hamming    : {h_loss:.4f}")

print(f"\n-- AUC par label --")
for name, auc in sorted(zip(valid_names, auc_per),
                         key=lambda x: x[1], reverse=True):
    print(f"  {name:25s} {auc:.4f}  {'#' * int(auc * 20)}")

display_names = label_cols.copy()
if mode == 5:
    display_names = [n if n != "Effusion" else "Pleural Effusion"
                     for n in display_names]
print(f"\n{classification_report(labels_arr, preds, target_names=display_names, zero_division=0)}")

# ======================================================================
#  PLOTS
# ======================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(lit.history["train"], "o-", label="Train", color="tab:red")
axes[0].plot(lit.history["val"],   "s-", label="Val",   color="tab:orange")
axes[0].set(title="Train vs Val Loss", xlabel="Epoch", ylabel="Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

names_g = ["F1\nmacro", "F1\nmicro", "F1\nsamples", "AUC\nmacro", "AUC\nmicro"]
vals_g  = [f1_mac, f1_mic, f1_sam, auc_mac, auc_mic]
bars = axes[1].bar(names_g, vals_g, color=["steelblue"]*3 + ["coral"]*2)
axes[1].set_ylim(0, 1); axes[1].set_title("Metriques globales")
for bar, val in zip(bars, vals_g):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=9)
axes[1].grid(axis="y", alpha=0.3)

sp = sorted(zip(valid_names, auc_per), key=lambda x: x[1])
ln, av = zip(*sp)
colors = ["coral" if a < 0.7 else "steelblue" for a in av]
axes[2].barh(ln, av, color=colors)
axes[2].axvline(x=0.7, color="gray", linestyle="--", linewidth=0.8)
axes[2].set_xlim(0, 1); axes[2].set_title("AUC par label"); axes[2].set_xlabel("AUC")
for i, (_, val) in enumerate(sp):
    axes[2].text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=8)

fig.suptitle(f"CXR-BERT-specialized -- {mode_name}", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, f"ft_{mode}.png"), dpi=150)
plt.show()

print(f"\n=> Sauvegarde : outputs/CXR-BERT-specialized/ft_{mode}.png")
print(f"=> Checkpoint : checkpoints/cxr_bert_{mode}.ckpt")
