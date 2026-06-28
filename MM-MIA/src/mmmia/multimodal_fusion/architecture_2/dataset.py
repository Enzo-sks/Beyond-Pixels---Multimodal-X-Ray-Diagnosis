"""dataset.py — Dataset pairé texte + image pour Projector LLM.

Réutilise exactement le même MultimodalDataset et multimodal_collate que
multimodal_fusion/dataset.py. Ce fichier est un proxy d'import pour garder
la structure de module cohérente.

Si tu veux des modifications spécifiques (prompt engineering, tokenizer Gemma,
longueur de séquence différente), c'est ici que tu les fais.

Usage:
    from projector_llm.dataset import MultimodalDataset, multimodal_collate, LABEL_COLS, load_paired_df
"""

# ── Import depuis le module multimodal_fusion existant ───────────────────────
# On réexporte tout pour que train.py puisse importer depuis ce module.
import sys
import os

# Résolution du chemin vers src/ (racine du package "mmmia").
# Arborescence : src/mmmia/multimodal_fusion/projector_llm/dataset.py (ce fichier)
# On doit remonter 3 niveaux pour atteindre src/, car "mmmia" est importé
# comme un package à partir de src/, pas depuis multimodal_fusion/.
_HERE = os.path.dirname(os.path.abspath(__file__))               # .../projector_llm
_SRC  = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # .../src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mmmia.multimodal_fusion.architecture_1.dataset import (   # noqa: E402
    MultimodalDataset,
    multimodal_collate,
    LABEL_COLS,
    load_paired_df,
)

__all__ = [
    "MultimodalDataset",
    "multimodal_collate",
    "LABEL_COLS",
    "load_paired_df",
]