"""
+==============================================================+
|  Fine-tuning multi-label sur le dataset Artishow (IU X-Ray)   |
|                                                                |
|  Usage :  python scripts/finetune.py 5   (CheXpert)           |
|           python scripts/finetune.py 14  (NIH ChestX-ray14)   |
|           python scripts/finetune.py 21  (tous labels IU)     |
+==============================================================+
"""

import os, sys, random, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report, hamming_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import BERTForMLM, Classifier
from data import LabelDataset, pad_collate, load_reports

CKPT_DIR = os.path.join(ROOT, "checkpoints")
OUT_DIR  = os.path.join(ROOT, "outputs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================================
#  CONFIG
# ======================================================================

mode = int(sys.argv[1]) if len(sys.argv) > 1 else 21
assert mode in (5, 14, 21), "Usage : python scripts/finetune.py [5|14|21]"

D, H, N, D_FF = 256, 8, 6, 512
FREEZE_ENCODER = False
EPOCHS   = 30
LR       = 2e-4
BATCH    = 32
PATIENCE = 5
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================================
#  DONNEES
# ======================================================================

random.seed(42); torch.manual_seed(42); np.random.seed(42)

texts, labels_np, label_cols, pos_weight, mode_name = load_reports(
    mode=mode, text_cols="findings", pw_clip=5)
labels = labels_np.tolist()
pos_weight = pos_weight.to(DEVICE)

print(f"Mode     : {mode_name}")
print(f"Dataset  : {len(texts)} rapports | {len(label_cols)} labels | Device : {DEVICE}")
print(f"Labels   : {label_cols}")

# -- Tokenizer + encodeur pre-entraine --------------------------------

tok = Tokenizer.from_file(os.path.join(CKPT_DIR, "tokenizer.json"))
V = tok.get_vocab_size()

bert = BERTForMLM(V, D, H, N, D_FF)
bert.load_state_dict(torch.load(os.path.join(CKPT_DIR, "bert_pretrained.pt"),
                                map_location=DEVICE, weights_only=True))
print(f"Encodeur charge : {sum(p.numel() for p in bert.encoder.parameters()):,} params")

# ======================================================================
#  CLASSIFIEUR + POS_WEIGHT
# ======================================================================

clf = Classifier(bert.encoder, n_labels=len(label_cols)).to(DEVICE)
if FREEZE_ENCODER:
    for p in clf.encoder.parameters(): p.requires_grad = False
print(f"Params entrainables : {sum(p.numel() for p in clf.parameters() if p.requires_grad):,}")

full_ds = LabelDataset(texts, labels, tok)
n_val = max(1, len(full_ds) // 5)
train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
print(f"Train : {len(train_ds)} | Val : {n_val}")
print(f"pos_weight : {pos_weight.cpu().numpy().round(2)}")

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
train_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=pad_collate)
val_loader   = DataLoader(val_ds, BATCH, collate_fn=pad_collate)

# ======================================================================
#  ENTRAINEMENT
# ======================================================================

opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad, clf.parameters()),
                          lr=LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
best_val, wait = float("inf"), 0
history = {"train": [], "val": []}

for ep in range(1, EPOCHS + 1):
    clf.train(); tl = 0
    for ids, lab in train_loader:
        ids, lab = ids.to(DEVICE), lab.to(DEVICE)
        loss = loss_fn(clf(ids), lab)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(clf.parameters(), 1.0); opt.step()
        tl += loss.item()
    sched.step()

    clf.eval(); vl = 0
    with torch.no_grad():
        for ids, lab in val_loader:
            ids, lab = ids.to(DEVICE), lab.to(DEVICE)
            vl += loss_fn(clf(ids), lab).item()

    tl /= len(train_loader); vl /= max(len(val_loader), 1)
    history["train"].append(tl); history["val"].append(vl)
    print(f"  Epoch {ep:2d}/{EPOCHS} | train={tl:.4f} | val={vl:.4f}")

    if vl < best_val:
        best_val, wait = vl, 0
        torch.save(clf.state_dict(), os.path.join(CKPT_DIR, f"classifier_{mode}.pt"))
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"  Early stopping (epoch {ep})"); break

# ======================================================================
#  EVALUATION
# ======================================================================

clf.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"classifier_{mode}.pt"),
                               weights_only=True))
clf.eval()

all_probs, all_labels = [], []
with torch.no_grad():
    for ids, lab in val_loader:
        ids, lab = ids.to(DEVICE), lab.to(DEVICE)
        all_probs.append(clf(ids).sigmoid().cpu().numpy())
        all_labels.append(lab.cpu().numpy())

probs  = np.vstack(all_probs)
labels_arr = np.vstack(all_labels)

# Seuil optimal
best_t = max(np.arange(0.1, 0.6, 0.05),
             key=lambda t: f1_score(labels_arr, (probs > t).astype(int),
                                    average='macro', zero_division=0))
preds = (probs > best_t).astype(int)

# Metriques
f1_mac = f1_score(labels_arr, preds, average='macro', zero_division=0)
f1_mic = f1_score(labels_arr, preds, average='micro', zero_division=0)
f1_sam = f1_score(labels_arr, preds, average='samples', zero_division=0)
h_loss = hamming_loss(labels_arr, preds)
valid  = [i for i in range(labels_arr.shape[1]) if len(np.unique(labels_arr[:, i])) > 1]
auc_mac = roc_auc_score(labels_arr[:, valid], probs[:, valid], average='macro')
auc_mic = roc_auc_score(labels_arr[:, valid], probs[:, valid], average='micro')
auc_per = roc_auc_score(labels_arr[:, valid], probs[:, valid], average=None)
valid_names = [label_cols[i] for i in valid]

print(f"\n{'='*50}")
print(f"  {mode_name} -- seuil={best_t:.2f}")
print(f"{'='*50}")
print(f"  F1 macro   : {f1_mac:.4f}")
print(f"  F1 micro   : {f1_mic:.4f}")
print(f"  F1 samples : {f1_sam:.4f}")
print(f"  AUC macro  : {auc_mac:.4f}")
print(f"  AUC micro  : {auc_mic:.4f}")
print(f"  Hamming    : {h_loss:.4f}")

print(f"\n-- AUC par label --")
for name, auc in sorted(zip(valid_names, auc_per), key=lambda x: x[1], reverse=True):
    bar = '#' * int(auc * 20)
    print(f"  {name:25s} {auc:.4f}  {bar}")

print(f"\n-- Rapport sklearn --")
display_names = label_cols.copy()
if mode == 5:
    display_names = [n if n != "Effusion" else "Pleural Effusion" for n in display_names]
print(classification_report(labels_arr, preds, target_names=display_names, zero_division=0))

# ======================================================================
#  PLOTS
# ======================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history["train"], 'o-', label='Train', color='tab:red')
axes[0].plot(history["val"], 's-', label='Val', color='tab:orange')
axes[0].set(title='Train vs Val Loss', xlabel='Epoch', ylabel='Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

names_g = ['F1\nmacro', 'F1\nmicro', 'F1\nsamples', 'AUC\nmacro', 'AUC\nmicro']
vals_g  = [f1_mac, f1_mic, f1_sam, auc_mac, auc_mic]
bars = axes[1].bar(names_g, vals_g, color=['steelblue']*3 + ['coral']*2)
axes[1].set_ylim(0, 1); axes[1].set_title('Metriques globales')
for bar, val in zip(bars, vals_g):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=9)
axes[1].grid(axis='y', alpha=0.3)

sp = sorted(zip(valid_names, auc_per), key=lambda x: x[1])
ln, av = zip(*sp)
colors = ['coral' if a < 0.7 else 'steelblue' for a in av]
axes[2].barh(ln, av, color=colors)
axes[2].axvline(x=0.7, color='gray', linestyle='--', linewidth=0.8)
axes[2].set_xlim(0, 1); axes[2].set_title('AUC par label'); axes[2].set_xlabel('AUC')
for i, (_, val) in enumerate(sp):
    axes[2].text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)

fig.suptitle(f"Fine-tuning {mode_name} -- IU X-Ray", fontsize=13, fontweight='bold')
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"ft_{mode}.png"), dpi=150); plt.show()

print(f"\n=> Sauvegarde : checkpoints/classifier_{mode}.pt, outputs/ft_{mode}.png")
