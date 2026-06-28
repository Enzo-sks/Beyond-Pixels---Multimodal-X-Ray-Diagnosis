# %% [code cell 1]
from google.colab import drive
drive.mount('/content/drive')

# %% [code cell 2]
from pathlib import Path

# %% [code cell 3]
# ── Copy images from Drive to Colab local disk (50x faster I/O) ─────────────
import shutil

LOCAL_IMG_DIR = Path('/content/Png')
LOCAL_IMG_DIR.mkdir(exist_ok=True)

if not any(LOCAL_IMG_DIR.iterdir()):
    print('Copying images from Drive to local disk (5-10 min, one-time)...')
    shutil.copytree(str(Path('/content/drive/MyDrive/Png')), str(LOCAL_IMG_DIR), dirs_exist_ok=True)
    print(f'Done! {len(list(LOCAL_IMG_DIR.glob("*.png")))} images copied.')
else:
    print(f'Already copied: {len(list(LOCAL_IMG_DIR.glob("*.png")))} images found locally.')

# %% [code cell 4]
# !pip install -q iterative-stratification

# %% [code cell 5]
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from transformers import ViTModel
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# %% [code cell 6]
# ── Paths ────────────────────────────────────────────────────────────────────
IMAGE_DIR = LOCAL_IMG_DIR
CSV_PATH  = Path('/content/drive/MyDrive/dataset_labeled.csv')

# ── Label columns ────────────────────────────────────────────────────────────
LABEL_COLS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule',
    'Hernia', 'Fracture', 'Pleural_Thickening', 'Opacity', 'Consolidation',
    'Granuloma', 'Calcinosis', 'Scoliosis', 'Atherosclerosis', 'Normal'
]
NUM_CLASSES = len(LABEL_COLS)

# ── Hyperparameters ──────────────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 16
NUM_EPOCHS    = 30      # extended from 20 — model was still improving at ep20
LR            = 5e-5
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15
WARMUP_EPOCHS = 3
PATIENCE      = 7       # increased from 5 — gives more room after backbone unfreeze

print(f'Number of classes: {NUM_CLASSES}')
print(f'Labels: {LABEL_COLS}')

# %% [code cell 7]
df = pd.read_csv(CSV_PATH)
print(f'Total rows: {len(df)}')
print(df[LABEL_COLS].sum().sort_values(ascending=False))

# %% [code cell 8]
# Verify images exist and build clean image paths
df['img_path'] = df['image_id'].apply(lambda x: str(IMAGE_DIR / x))

missing = df[~df['img_path'].apply(os.path.exists)]
print(f'Missing images: {len(missing)}')
df = df[df['img_path'].apply(os.path.exists)].reset_index(drop=True)
print(f'Rows after dropping missing: {len(df)}')

# %% [code cell 9]
# Label distribution bar chart
counts = df[LABEL_COLS].sum().sort_values(ascending=False)
plt.figure(figsize=(14, 4))
counts.plot(kind='bar')
plt.title('Label distribution')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [code cell 10]
# Multi-label stratified split
msss = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=42
)
train_idx, temp_idx = next(msss.split(df, df[LABEL_COLS]))
train_df = df.iloc[train_idx].reset_index(drop=True)
temp_df  = df.iloc[temp_idx].reset_index(drop=True)

msss2 = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT), random_state=42
)
val_idx, test_idx = next(msss2.split(temp_df, temp_df[LABEL_COLS]))
val_df  = temp_df.iloc[val_idx].reset_index(drop=True)
test_df = temp_df.iloc[test_idx].reset_index(drop=True)

print(f'Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')

# %% [markdown]
# ## Change 1 — Stronger Geometric Augmentations
# 
# Previous augmentations: `RandomHorizontalFlip`, `RandomRotation(10)`, mild `ColorJitter`.
# 
# Added:
# - `RandomAffine` — simulates patient positioning variance (translation, scale, shear)
# - `ElasticTransform` — realistic soft-tissue deformation
# - `RandomErasing` — applied post-tensor, simulates local occlusion / implants / noise patches
# - Stronger `ColorJitter` — broader brightness/contrast range

# %% [code cell 11]
VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD  = [0.5, 0.5, 0.5]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.10, 0.10),
        scale=(0.85, 1.15),
        shear=10
    ),
    transforms.ElasticTransform(alpha=40.0, sigma=5.0),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
])


class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, label_cols, transform=None):
        self.df         = dataframe.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row['img_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_cols].values.astype(float),
                              dtype=torch.float32)
        return image, labels


train_dataset = ChestXRayDataset(train_df, LABEL_COLS, train_transform)
val_dataset   = ChestXRayDataset(val_df,   LABEL_COLS, val_transform)
test_dataset  = ChestXRayDataset(test_df,  LABEL_COLS, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f'Batches — train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}')

# %% [markdown]
# ## Change 2 — Asymmetric Loss (ASL)
# 
# Previous loss: `BCEWithLogitsLoss(pos_weight=...)`.
# 
# **Why ASL?** Standard BCE with `pos_weight` up-weights positives globally but does not suppress easy negatives. In a multi-label setting the vast majority of labels per sample are 0 (negative), so the gradient is dominated by easy true-negatives that contribute little signal.
# 
# ASL applies **asymmetric focusing**:
# - Positives: mild focal factor (`gamma_pos=1`) — similar to standard focal loss
# - Negatives: stronger focal factor (`gamma_neg=4`) + probability margin shift (`clip=0.05`) — confident easy negatives are down-weighted aggressively
# 
# This lets the model focus on the hard rare-positive cases (Nodule, Fracture, Granuloma) without ignoring common ones.

# %% [code cell 12]
class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    Ref: Ben-Baruch et al. (2021) 'Asymmetric Loss For Multi-Label Classification'
    https://arxiv.org/abs/2009.14119

    gamma_neg >> gamma_pos: hard negatives contribute more than easy negatives;
    clip (m): shifts predicted negative probs down by m before computing loss,
              zeroing out very confident true-negatives entirely.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)

        # Shift negative probabilities down by margin (clamp to 0)
        p_neg = (p - self.clip).clamp(min=0)

        # Stable log probabilities
        log_p     = torch.log(p.clamp(min=self.eps))
        log_1_p   = torch.log((1 - p_neg).clamp(min=self.eps))

        # Focal weights (asymmetric)
        loss_pos = (1 - p)  ** self.gamma_pos * log_p
        loss_neg =  p_neg   ** self.gamma_neg  * log_1_p

        loss = -(targets * loss_pos + (1 - targets) * loss_neg)
        return loss.mean()

# %% [markdown]
# ## Change 3 — Richer Classification Head + Attention Pooling
# 
# Previous head: `Linear(768→512) → ReLU → Dropout(0.5) → Linear(512→21)` over `[CLS]` token only.
# 
# Two upgrades:
# 1. **Attention pooling** over all 197 patch tokens (instead of just `[CLS]`). Each token votes with a learned weight — patch regions relevant to pathology contribute more. This is especially useful for spatially diffuse findings (Scoliosis, Pleural Thickening).
# 2. **Deeper MLP head** with `BatchNorm1d` and `GELU` activations. `BatchNorm` stabilises training after backbone unfreeze; `GELU` is smoother than `ReLU` and aligns with ViT's internal activations. Dropout reduced from 0.5 → 0.3 since we now have two regularisation points.

# %% [code cell 13]
class AttentionPool(nn.Module):
    """Soft attention pooling over all ViT token outputs (including [CLS])."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # hidden_states: [B, seq_len, hidden_size]
        weights = torch.softmax(self.attn(hidden_states), dim=1)  # [B, seq_len, 1]
        return (weights * hidden_states).sum(dim=1)               # [B, hidden_size]


class ViTChestClassifier(nn.Module):
    """
    ViT backbone (codewithdark/vit-chest-xray) with:
      - Attention pooling over all patch tokens (replaces bare [CLS] extraction)
      - Deeper MLP head: 768 → 512 → 256 → num_classes
        with BatchNorm1d + GELU + Dropout(0.3) at each hidden layer
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.vit  = ViTModel.from_pretrained('codewithdark/vit-chest-xray')
        hidden_size = self.vit.config.hidden_size  # 768

        self.pool = AttentionPool(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled  = self.pool(outputs.last_hidden_state)  # [B, 768]
        return self.classifier(pooled)


model = ViTChestClassifier(NUM_CLASSES).to(DEVICE)

# Phase 1: freeze backbone, train head only
for param in model.vit.parameters():
    param.requires_grad = False

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params:     {total_params:,}')
print(f'Trainable params: {trainable_params:,}  (head + attention pool only)')
print(f'Frozen params:    {total_params - trainable_params:,}  (ViT backbone)')

# %% [code cell 14]
criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)

warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_EPOCHS]
)

print('Loss: AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)')
print(f'LR schedule: warmup {WARMUP_EPOCHS} epochs → cosine decay to 1e-6')

# %% [code cell 15]
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, leave=False, desc='Train'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []
    for images, labels in tqdm(loader, leave=False, desc='Eval'):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    auc_scores = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            auc_scores.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
        else:
            auc_scores.append(float('nan'))
    mean_auc = np.nanmean(auc_scores)

    return total_loss / len(loader.dataset), mean_auc, auc_scores

# %% [code cell 16]
history    = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'lr': []}
best_auc   = 0.0
CKPT_PATH  = '/content/drive/MyDrive/best_vit_chest_04.pth'
no_improve = 0

for epoch in range(1, NUM_EPOCHS + 1):

    # Phase 2: unfreeze backbone after warmup
    if epoch == WARMUP_EPOCHS + 1:
        for param in model.vit.parameters():
            param.requires_grad = True
        optimizer.add_param_group({
            'params': model.vit.parameters(),
            'lr': LR * 0.1,
            'weight_decay': WEIGHT_DECAY
        })
        cosine_scheduler.base_lrs.append(LR * 0.1)
        print(f'  [Epoch {epoch}] Backbone unfrozen — backbone lr={LR * 0.1:.1e}')

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_auc, _ = evaluate(model, val_loader, criterion, DEVICE)
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_auc'].append(val_auc)
    history['lr'].append(current_lr)

    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
          f"LR: {current_lr:.2e} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Mean AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc   = val_auc
        no_improve = 0
        torch.save(model.state_dict(), CKPT_PATH)
        print(f'  New best AUC {best_auc:.4f} — checkpoint saved')
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f'  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)')
            break

print(f'Training complete. Best Val AUC: {best_auc:.4f}')

# %% [code cell 17]
epochs = range(1, len(history['train_loss']) + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

axes[0].plot(epochs, history['train_loss'], label='Train')
axes[0].plot(epochs, history['val_loss'],   label='Val')
axes[0].set_title('Loss (ASL)')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(epochs, history['val_auc'], color='green')
axes[1].set_title('Validation Mean AUC-ROC')
axes[1].set_xlabel('Epoch')
axes[1].set_ylim(0, 1)

axes[2].plot(epochs, history['lr'], color='orange')
axes[2].set_title('Learning Rate Schedule')
axes[2].set_xlabel('Epoch')
axes[2].set_yscale('log')

plt.tight_layout()
plt.show()

# %% [code cell 18]
# Load best checkpoint
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

test_loss, test_auc, per_class_auc = evaluate(model, test_loader, criterion, DEVICE)
print(f'Test Loss: {test_loss:.4f}  |  Test Mean AUC: {test_auc:.4f}\n')

# Per-class AUC table with delta vs v3 baseline
V3_AUC = {
    'Atelectasis': 0.735, 'Cardiomegaly': 0.831, 'Effusion': 0.870,
    'Pneumonia': 0.733, 'Pneumothorax': 0.722, 'Edema': 0.793,
    'Emphysema': 0.881, 'Fibrosis': 0.935, 'Infiltration': 0.699,
    'Mass': 0.636, 'Nodule': 0.597, 'Hernia': 0.846,
    'Fracture': 0.600, 'Pleural_Thickening': 0.590, 'Opacity': 0.773,
    'Consolidation': 0.872, 'Granuloma': 0.588, 'Calcinosis': 0.630,
    'Scoliosis': 0.645, 'Atherosclerosis': 0.767, 'Normal': 0.713
}

auc_df = pd.DataFrame({
    'Condition': LABEL_COLS,
    'AUC v4':   per_class_auc,
    'AUC v3':   [V3_AUC[c] for c in LABEL_COLS],
})
auc_df['Delta'] = auc_df['AUC v4'] - auc_df['AUC v3']
auc_df = auc_df.sort_values('AUC v4', ascending=False).reset_index(drop=True)
print(auc_df.to_string(index=False, float_format='{:.3f}'.format))
