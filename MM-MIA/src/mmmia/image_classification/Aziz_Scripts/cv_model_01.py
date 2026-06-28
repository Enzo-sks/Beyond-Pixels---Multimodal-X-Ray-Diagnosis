# %% [code cell 1]
from google.colab import drive
drive.mount('/content/drive')

# %% [code cell 2]
from pathlib import Path


# %% [code cell 3]
# ── Copy images from Drive to Colab local disk (50x faster I/O) ─────────────
import shutil

LOCAL_IMG_DIR = Path('/content/drive/MyDrive/Png')
LOCAL_IMG_DIR.mkdir(exist_ok=True)

if not any(LOCAL_IMG_DIR.iterdir()):  # skip if already copied
    print("Copying images from Drive to local disk (5-10 min, one-time)...")
    shutil.copytree(str(Path('/content/drive/MyDrive/Png')), str(LOCAL_IMG_DIR), dirs_exist_ok=True)
    print(f"Done! {len(list(LOCAL_IMG_DIR.glob('*.png')))} images copied.")
else:
    print(f"Already copied: {len(list(LOCAL_IMG_DIR.glob('*.png')))} images found locally.")


# %% [code cell 4]
# !pip install -q timm scikit-learn

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
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# %% [code cell 6]
# ── Paths ────────────────────────────────────────────────────────────────────
IMAGE_DIR = Path('/content/drive/MyDrive/Png')   # folder with .png images
CSV_PATH  = Path('/content/drive/MyDrive/dataset_labeled.csv')  # upload your CSV here
# If your CSV is in a different Drive location, adjust CSV_PATH above.

# ── Label columns ────────────────────────────────────────────────────────────
LABEL_COLS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule',
    'Hernia', 'Fracture', 'Pleural_Thickening', 'Opacity', 'Consolidation',
    'Granuloma', 'Calcinosis', 'Scoliosis', 'Atherosclerosis', 'Normal'
]
NUM_CLASSES = len(LABEL_COLS)

# ── Hyperparameters ──────────────────────────────────────────────────────────
IMG_SIZE    = 224     # DenseNet-121 expects 224×224
BATCH_SIZE  = 32
NUM_EPOCHS  = 20
LR          = 1e-4
WEIGHT_DECAY= 1e-5
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

print(f'Number of classes: {NUM_CLASSES}')
print(f'Labels: {LABEL_COLS}')

# %% [code cell 7]
df = pd.read_csv(CSV_PATH)
print(f'Total rows: {len(df)}')
print(df[LABEL_COLS].sum().sort_values(ascending=False))

# %% [code cell 8]
# Verify images exist and build clean image paths
df['img_path'] = df['image_id'].apply(lambda x: str(IMAGE_DIR / x))

# Drop rows where image file is missing
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
# Stratify on the most frequent single label to keep proportions
df['primary_label'] = df[LABEL_COLS].idxmax(axis=1)

train_df, temp_df = train_test_split(
    df, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=42,
    stratify=df['primary_label']
)
val_df, test_df = train_test_split(
    temp_df, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT), random_state=42,
    stratify=temp_df['primary_label']
)

print(f'Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}')

# %% [code cell 11]
# Augmentation for training; just normalize for val/test
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, label_cols, transform=None):
        self.df        = dataframe.reset_index(drop=True)
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

# %% [code cell 12]
def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """DenseNet-121 pretrained on ImageNet, head replaced for multi-label output."""
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
        # No sigmoid here — BCEWithLogitsLoss is numerically more stable
    )
    return model


model = build_model(NUM_CLASSES).to(DEVICE)
print(model.classifier)

# %% [code cell 13]
# Positive frequency per class → weight = neg_freq / pos_freq
pos_counts  = train_df[LABEL_COLS].sum().values
neg_counts  = len(train_df) - pos_counts
pos_weight  = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Two-stage training:
#   Phase 1 — only head (faster, avoids destroying pretrained features)
#   Phase 2 — fine-tune all layers with lower LR
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print('Positive weights (first 5):', pos_weight[:5].cpu().numpy())

# %% [code cell 14]
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

    # Per-class AUC (skip classes with only one label value present)
    auc_scores = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            auc_scores.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
        else:
            auc_scores.append(float('nan'))
    mean_auc = np.nanmean(auc_scores)

    return total_loss / len(loader.dataset), mean_auc, auc_scores

# %% [code cell 15]
history   = {"train_loss": [], "val_loss": [], "val_auc": []}
best_auc  = 0.0
CKPT_PATH = '/content/drive/MyDrive/best_densenet121.pth'
PATIENCE  = 5
no_improve = 0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_auc, _ = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_auc"].append(val_auc)

    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Mean AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        no_improve = 0
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"  New best AUC {best_auc:.4f} — checkpoint saved")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print(f"Training complete. Best Val AUC: {best_auc:.4f}")


# %% [code cell 16]
epochs = range(1, len(history["train_loss"]) + 1)  # actual epochs run
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(epochs, history["train_loss"], label="Train")
axes[0].plot(epochs, history["val_loss"],   label="Val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(epochs, history["val_auc"], color="green")
axes[1].set_title("Validation Mean AUC-ROC")
axes[1].set_xlabel("Epoch")
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()


# %% [code cell 17]
# Load best checkpoint
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

test_loss, test_auc, per_class_auc = evaluate(model, test_loader, criterion, DEVICE)
print(f'Test Loss: {test_loss:.4f}  |  Test Mean AUC: {test_auc:.4f}\n')

# Per-class AUC table
auc_df = pd.DataFrame({'Condition': LABEL_COLS, 'AUC-ROC': per_class_auc})
auc_df = auc_df.sort_values('AUC-ROC', ascending=False).reset_index(drop=True)
print(auc_df.to_string(index=False))
