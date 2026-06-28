"""data.py — Dataset pairé texte + image partagé par les modèles multimodaux.

Source : CSV image-level (data/shared/dataset_labeled*.csv), déjà pairé —
  image_id   : nom du fichier PNG (ex. CXR3037_IM-1410-1001.png)
  indication : motif clinique  ┐  entrées texte (jamais `impression` : anti-fuite)
  findings   : corps du rapport ┘
  <labels>   : colonnes binaires 0/1

Le `Dataset` est agnostique de l'encodeur : on lui passe le tokenizer (interface
`encode_batch`), les colonnes labels et le transform image. Chaque architecture
choisit son propre encodeur/tokenizer dans son `model.py`.

Labels et split sans-fuite sont importés de `text_classification` (source de
vérité unique : ordre canonique des labels, regroupement par texte).
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# ── Réutilisation des constantes/utilitaires de text_classification ──────────
# Convention du repo : on insère text_classification dans sys.path puis import bare.
_TEXT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "..", "text_classification")
if _TEXT_ROOT not in sys.path:
    sys.path.insert(0, _TEXT_ROOT)

from data.datasets import LABELS_5, LABELS_14, META_COLS          # noqa: E402
from data.splits import text_group_key, grouped_train_val_test     # noqa: E402

# Liste canonique IU-XRay 21 labels (ordre = colonnes des CSV labellisés, identique
# à l'ancienne constante hardcodée d'architecture_1). On la fige explicitement car
# l'exclusion par META_COLS ne suffit pas sur le CSV image-level, qui porte des
# colonnes supplémentaires (image_id, caption, height, width, Image_path) absentes
# de META_COLS — elles seraient sinon prises pour des labels.
LABELS_21 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Pneumonia", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Hernia", "Fracture", "Pleural_Thickening", "Opacity", "Consolidation",
    "Granuloma", "Calcinosis", "Scoliosis", "Atherosclerosis", "Normal",
]

# Re-export pour que les train.py n'importent que depuis common.
__all__ = [
    "LABELS_5", "LABELS_14", "LABELS_21", "META_COLS", "MODE_LABELS", "TEXT_COLS",
    "resolve_label_cols", "load_paired_df", "build_texts", "build_groups",
    "FusionDataset", "fusion_collate",
    "text_group_key", "grouped_train_val_test",
]

MODE_LABELS = {5: LABELS_5, 14: LABELS_14, 21: LABELS_21}
TEXT_COLS = ["indication", "findings"]      # `impression` EXCLU (anti-fuite)


def resolve_label_cols(df: pd.DataFrame, mode: int = 14) -> list[str]:
    """Colonnes labels dans l'ordre canonique (source de vérité : LABELS_*).

    mode 5/14/21 : sous-ensemble figé filtré sur les colonnes présentes du df.
    Autre valeur : repli sur toutes les colonnes hors méta (ordre du CSV).
    """
    if mode in MODE_LABELS:
        return [c for c in MODE_LABELS[mode] if c in df.columns]
    return [c for c in df.columns if c not in META_COLS]


def load_paired_df(csv_path) -> pd.DataFrame:
    """Charge le CSV image-level et supprime les lignes sans findings."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["findings"]).reset_index(drop=True)
    df["findings"] = df["findings"].astype(str).str.strip()
    df = df[df["findings"] != ""].reset_index(drop=True)
    return df


def build_texts(df: pd.DataFrame, text_cols=None) -> list[str]:
    """Concatène les colonnes texte demandées (défaut : indication + findings).

    `text_cols` permet de restreindre l'entrée texte — p.ex. `["indication"]`
    pour une inférence/validation SANS findings (cf. `--no_findings`), alors que
    l'entraînement garde indication+findings.
    """
    cols = [c for c in (text_cols or TEXT_COLS) if c in df.columns]
    return df[cols].fillna("").agg(" ".join, axis=1).str.strip().tolist()


def build_groups(df: pd.DataFrame):
    """Clé de groupe pour un split sans fuite, par texte = indication + findings.

    Deux échantillons à l'entrée texte STRICTEMENT identique (vues multiples d'un
    même rapport, ou templates recopiés) tombent dans le même split.
    """
    return text_group_key(build_texts(df))


class FusionDataset(Dataset):
    """Retourne (token_ids, pixel_values, labels) pour chaque image.

    Args:
        df         : DataFrame (image_id, indication, findings, <labels>)
        label_cols : colonnes labels à utiliser (ordre = ordre des logits)
        tokenizer  : tokenizer à interface `encode_batch` (HF backend ou tokenizers)
        image_dir  : répertoire des PNG
        transform  : transform torchvision appliqué à chaque image PIL
        text_cols  : colonnes texte à encoder (défaut indication+findings) ; passer
                     `["indication"]` pour un set d'éval sans findings.
    """

    def __init__(self, df, label_cols, tokenizer, image_dir, transform=None,
                 text_cols=None):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.image_dir = Path(image_dir)
        self.transform = transform
        # Encodage texte en batch à l'init (gain de temps).
        self.enc = tokenizer.encode_batch(build_texts(self.df, text_cols))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        ids = torch.tensor(self.enc[idx].ids, dtype=torch.long)

        img_path = self.image_dir / self.df.at[idx, "image_id"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            pixel_values = self.transform(img)
        else:
            import torchvision.transforms.functional as TF
            pixel_values = TF.to_tensor(img.resize((224, 224)))

        labels = torch.tensor(
            self.df.loc[idx, self.label_cols].values.astype(float),
            dtype=torch.float32,
        )
        return ids, pixel_values, labels


def fusion_collate(batch):
    """Pad les séquences texte à la longueur max du batch, stack images + labels."""
    ids_list, pixels_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in ids_list)
    ids_pad = torch.stack([F.pad(ids, (0, max_len - len(ids))) for ids in ids_list])
    pixels = torch.stack(list(pixels_list))
    labels = torch.stack(list(labels_list))
    return ids_pad, pixels, labels
