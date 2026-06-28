import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pandas as pd
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
try:
    # new name
    from transformers import ViTImageProcessor as ViTProcessor
except Exception:
    # older transformers
    from transformers import ViTFeatureExtractor as ViTProcessor

from sklearn.model_selection import train_test_split

from model import SingleStreamMMHd


class CSVMultimodalDataset(Dataset):
    """Dataset reading rows from a CSV and returning image + tokenized text + label.

    Expects `dataset_labeled.csv` with an `Image_path` column containing the path
    to the PNG (or a column you pass via `image_col`). Labels are assumed to be
    the one-hot-like columns starting at the column named `Atelectasis` (as in
    the provided CSV). We convert that multi-hot vector to a single class via
    argmax to match the model's CrossEntropyLoss expectation.
    """

    def __init__(self, csv_path="dataset_labeled.csv", split="train",
                 image_col="Image_path", text_col="findings",
                 test_size=0.1, random_state=42, max_length=128):

        self.df = pd.read_csv(csv_path)
        # remember CSV location so we can resolve image paths relative to it
        self.csv_path = os.path.abspath(csv_path)
        self.csv_dir = os.path.dirname(self.csv_path)

        # determine class columns (start at 'Atelectasis' if present)
        if "Atelectasis" in self.df.columns:
            cls_start = self.df.columns.get_loc("Atelectasis")
            self.class_cols = list(self.df.columns[cls_start:])
        else:
            # fallback: take all columns after 'xml_uid' if present
            if "xml_uid" in self.df.columns:
                cls_start = self.df.columns.get_loc("xml_uid") + 1
                self.class_cols = list(self.df.columns[cls_start:])
            else:
                # last 21 columns heuristic
                self.class_cols = list(self.df.columns[-21:])

        train_df, val_df = train_test_split(self.df, test_size=test_size,
                                            random_state=random_state,
                                            shuffle=True)

        self.df = train_df.reset_index(drop=True) if split == "train" else val_df.reset_index(drop=True)

        self.image_col = image_col
        self.text_col = text_col
        self.max_length = max_length

        # Tokenizer and image processor
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.processor = ViTProcessor.from_pretrained('codewithdark/vit-chest-xray')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # Image
        img_path = row.get(self.image_col, None)
        if pd.isna(img_path) or img_path is None:
            raise FileNotFoundError(f"No image path for row {idx}")

        # Resolve image path candidates in order:
        # 1) as given (absolute or relative)
        # 2) relative to current working directory
        # 3) relative to the CSV directory
        # 4) inside the CSV directory's `Png` folder (preferred)
        img_path = str(img_path)
        candidates = []
        if os.path.isabs(img_path):
            candidates.append(img_path)
        else:
            candidates.append(img_path)
            candidates.append(os.path.join(os.getcwd(), img_path.lstrip("/")))
            candidates.append(os.path.join(self.csv_dir, img_path.lstrip("/")))
            candidates.append(os.path.join(self.csv_dir, "Png", img_path.lstrip("/")))

        found = False
        for p in candidates:
            if p and os.path.exists(p):
                img_path = p
                found = True
                break

        if not found:
            raise FileNotFoundError(f"Image file not found for row {idx}. Tried: {candidates}")

        image = Image.open(img_path).convert("RGB")

        # Prepare pixel_values via the ViT processor
        proc = self.processor(images=image, return_tensors="pt")
        pixel_values = proc["pixel_values"].squeeze(0)

        # Text (use findings or caption)
        text = row.get(self.text_col, "")
        tok = self.tokenizer(text,
                             padding="max_length",
                             truncation=True,
                             max_length=self.max_length,
                             return_tensors="pt")
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        # Label: convert multi-hot into single class index via argmax
        label_vector = row[self.class_cols].astype(int).values
        if label_vector.sum() == 0:
            # if no positive label, fallback to last class index
            label = len(self.class_cols) - 1
        else:
            label = int(np.argmax(label_vector))

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 21

BATCH_SIZE = 8
EPOCHS = 20

LR = 1e-4
WEIGHT_DECAY = 1e-2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# DATASETS (CSV-backed)
# ============================================================

# Adjust `csv_path` if your CSV lives elsewhere or you have an extracted zip
CSV_PATH = os.path.join(os.getcwd(), "dataset_labeled.csv")

train_dataset = CSVMultimodalDataset(
    csv_path=CSV_PATH,
    split="train",
    image_col="Image_path",
    text_col="findings",
)

val_dataset = CSVMultimodalDataset(
    csv_path=CSV_PATH,
    split="val",
    image_col="Image_path",
    text_col="findings",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ============================================================
# MODEL
# ============================================================

model = SingleStreamMMHd(
    num_classes=NUM_CLASSES
)

# ------------------------------------------------------------
# Freeze ViT
# ------------------------------------------------------------

for p in model.vit_backbone.parameters():
    p.requires_grad = False

# ------------------------------------------------------------
# Freeze BERT
# ------------------------------------------------------------

for p in model.text_backbone.parameters():
    p.requires_grad = False

model = model.to(DEVICE)

# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()

# ============================================================
# OPTIMIZER
# ============================================================

trainable_params = filter(
    lambda p: p.requires_grad,
    model.parameters()
)

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# ============================================================
# TRAIN
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader)

    for batch in pbar:

        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            pixel_values,
            input_ids,
            attention_mask
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

        pbar.set_description(
            f"loss={loss.item():.4f}"
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in val_loader:

        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(
            pixel_values,
            input_ids,
            attention_mask
        )

        loss = criterion(
            logits,
            labels
        )

        running_loss += loss.item()

        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    val_loss = running_loss / len(val_loader)
    val_acc = correct / total

    return val_loss, val_acc

# ============================================================
# MAIN LOOP
# ============================================================

best_val_acc = 0.0

for epoch in range(EPOCHS):

    print()
    print("=" * 60)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("=" * 60)

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc = validate()

    scheduler.step()

    print(
        f"Train Loss : {train_loss:.4f} | "
        f"Train Acc : {train_acc:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f} | "
        f"Val Acc   : {val_acc:.4f}"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print(
            f"New best model saved "
            f"(acc={best_val_acc:.4f})"
        )

print()
print("Training finished.")