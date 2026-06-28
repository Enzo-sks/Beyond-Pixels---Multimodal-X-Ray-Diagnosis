"""common — infrastructure partagée par les modèles de fusion multimodale.

Pièces agnostiques de l'architecture (réutilisées par architecture_1, 3, …) :
  - data       : FusionDataset, fusion_collate, load_paired_df, resolve_label_cols,
                 build_groups, split groupé sans fuite (grouped_train_val_test).
  - transforms : pré-traitement image ViT (TRAIN_TF / VAL_TF).
  - losses     : AsymmetricLoss.

Le choix de l'encodeur (ViT, CXR-BERT, BERT custom…) reste dans le `model.py` de
chaque architecture — chaque modèle importe l'encodeur qu'il veut.
"""

from .data import (  # noqa: F401
    FusionDataset, fusion_collate, load_paired_df, build_texts, build_groups,
    resolve_label_cols, grouped_train_val_test, text_group_key,
    LABELS_5, LABELS_14, LABELS_21, META_COLS, MODE_LABELS, TEXT_COLS,
)
from .losses import AsymmetricLoss        # noqa: F401
from .transforms import TRAIN_TF, VAL_TF, VIT_MEAN, VIT_STD  # noqa: F401
