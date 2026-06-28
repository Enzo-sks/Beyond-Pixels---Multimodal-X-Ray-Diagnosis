"""splits.py — Découpage train/val/test SANS fuite, groupé par rapport.

Pourquoi : un rapport IU X-Ray = plusieurs images (frontale + latérale) qui
partagent le MÊME texte `findings` et les MÊMES labels. De plus ~23 % des
rapports ont un texte strictement dupliqué. Un découpage aléatoire (random_split,
ShuffleSplit) place donc le même texte des deux côtés → fuite train/test.

`grouped_train_val_test` garantit qu'un `xml_uid` (un rapport) tombe entièrement
dans un seul split. Aucun texte partagé entre train, val et test.
"""

import hashlib

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def text_group_key(texts):
    """Clé de groupe = hash du texte normalisé (strip + lowercase).

    À utiliser comme `groups` pour `grouped_train_val_test`. Garantit que deux
    échantillons au texte STRICTEMENT identique (multi-images d'un même rapport
    OU templates 'normaux' recopiés sur des patients différents) tombent dans le
    même split. C'est un sur-ensemble du groupage par xml_uid.
    """
    return np.array([
        hashlib.md5(str(t).strip().lower().encode("utf-8")).hexdigest()
        for t in texts
    ])


def grouped_train_val_test(groups, train=0.70, val=0.15, seed=42):
    """Indices train/val/test groupés par `groups`, sans chevauchement de groupe.

    Args:
        groups : array-like (len = n_samples) — identifiant de groupe par ligne
                 (ex. xml_uid). Toutes les lignes d'un même groupe restent ensemble.
        train  : proportion (sur les groupes) du train.
        val    : proportion du val. Le test reçoit le reste (1 - train - val).
        seed   : graine pour la reproductibilité.

    Returns:
        (train_idx, val_idx, test_idx) : np.ndarray d'indices de lignes.
    """
    groups = np.asarray(groups)
    n = len(groups)
    idx = np.arange(n)

    # 1) train  vs  (val+test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train, random_state=seed)
    train_pos, temp_pos = next(gss1.split(idx, groups=groups))

    # 2) val  vs  test  (à l'intérieur du reste, toujours groupé)
    rel_val = val / (1.0 - train)            # part de val DANS le reste
    gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val, random_state=seed)
    temp_idx = idx[temp_pos]
    val_rel, test_rel = next(gss2.split(temp_idx, groups=groups[temp_pos]))

    train_idx = idx[train_pos]
    val_idx   = temp_idx[val_rel]
    test_idx  = temp_idx[test_rel]
    return train_idx, val_idx, test_idx
