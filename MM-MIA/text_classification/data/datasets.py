"""datasets.py — MLMDataset, LabelDataset, pad_collate, load_reports."""

import os, random, torch, torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from .tokenizer import SP

# ── Label definitions (shared across all scripts) ────────────────────

META_COLS = {"xml_uid", "indication", "findings", "impression",
             "comparison", "image_ids", "num_images"}

LABELS_5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]

LABELS_14 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

MODE_NAMES = {5: "CheXpert-5", 14: "NIH-14", 21: "IU-XRay-21"}


def load_reports(mode=21, text_cols="findings", data_dir=None, pw_clip=5):
    """Load IU X-Ray reports and compute pos_weight.

    Args:
        mode: 5, 14, or 21 labels.
        text_cols: column name (str) or list of column names to concatenate.
        data_dir: path to data directory (default: same dir as this file).
        pw_clip: max value for pos_weight clamping.

    Returns:
        texts, labels_np, label_cols, pos_weight, mode_name
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_csv(os.path.join(data_dir, "dataset_reports.csv"))

    if mode == 5:    label_cols = [c for c in LABELS_5  if c in df.columns]
    elif mode == 14: label_cols = [c for c in LABELS_14 if c in df.columns]
    else:            label_cols = [c for c in df.columns if c not in META_COLS]

    if isinstance(text_cols, str):
        text_cols = [text_cols]

    df = df.dropna(subset=text_cols)
    texts = df[text_cols].fillna("").agg(" ".join, axis=1).str.strip().tolist()
    labels_np = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

    y_all = torch.tensor(labels_np)
    pos = y_all.sum(0); neg = len(y_all) - pos
    pos_weight = (neg / pos.clamp(min=1)).clamp(max=pw_clip)

    return texts, labels_np, label_cols, pos_weight, MODE_NAMES[mode]


class MLMDataset(Dataset):
    def __init__(self, texts, tok, mask_prob=0.15):
        self.enc = tok.encode_batch(texts)
        self.mid = tok.token_to_id(SP["mask"])
        self.skip = {tok.token_to_id(v) for v in SP.values()}
        self.V, self.p = tok.get_vocab_size(), mask_prob
    def __len__(self): return len(self.enc)
    def __getitem__(self, i):
        ids = torch.tensor(self.enc[i].ids, dtype=torch.long); lab = ids.clone()
        for j in range(len(ids)):
            if ids[j].item() not in self.skip and random.random() < self.p:
                r = random.random()
                if r < 0.8:   ids[j] = self.mid
                elif r < 0.9: ids[j] = random.randint(0, self.V-1)
            else: lab[j] = -100
        return ids, lab


class LabelDataset(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok.encode_batch(texts)
        self.lab = torch.tensor(labels, dtype=torch.float)
    def __len__(self): return len(self.enc)
    def __getitem__(self, i):
        return torch.tensor(self.enc[i].ids, dtype=torch.long), self.lab[i]


def pad_collate(batch):
    seqs, labs = zip(*batch)
    ml = max(len(s) for s in seqs)
    seqs = torch.stack([F.pad(s, (0, ml-len(s))) for s in seqs])
    labs = torch.stack(list(labs)) if labs[0].dim() > 0 and labs[0].shape[0] > 1 \
        else torch.stack([F.pad(l, (0, ml-len(l)), value=-100) for l in labs])
    return seqs, labs
