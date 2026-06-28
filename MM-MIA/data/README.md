# Données — `data/`

Tous les fichiers de ce dossier sont gérés via **Git LFS** (sauf les README et configs).
Backend LFS : **GitLab ENST** (voir `.lfsconfig`).

## Structure

```
data/
├── shared/
│   └── dataset_labeled.csv          # Source de vérité labels 12 pathologies
│                                     # (utilisé par text + image + dicom)
├── text_classification/
│   └── dataset_reports.csv          # Rapports bruts (text uniquement)
│
├── image_classification/
│   ├── dataset/                     # Train/Test split d'images PNG (échantillon)
│   ├── train.zip                    # Dataset complet pour entraînement (LFS, 806 MB)
│   └── test.zip                     # Dataset de test (LFS, 198 MB)
│
├── dicom/
│   ├── raw/                         # Fichiers .dcm originaux (LFS)
│   │   ├── fragment_data_set/       # Sous-ensemble extrait pour dev
│   │   ├── dicom_windowing_test/    # Tests de windowing
│   │   ├── dataset_labeled_enriched.csv
│   │   ├── dataset_labeled_with_png_mapping.csv
│   │   └── png_files_list.csv
│   ├── preprocessed_multiwindow/    # Images multi-window prétraitées (LFS, ~7-10 MB / img)
│   └── metadata/
│       └── merged_df_meta.csv       # Métadonnées DICOM consolidées
│
└── labeling/
    ├── NLMCXR_reports/              # Rapports XML Open-I (NLM)
    │   └── ecgen-radiology/
    └── new_dataset.csv              # Output du pipeline de labellisation
```

## Récupérer les données

### Tout télécharger (peut être lourd — plusieurs Go)
```bash
git lfs pull
```

### Télécharger un sous-ensemble seulement
```bash
git lfs pull --include="data/text_classification/**"
git lfs pull --include="data/dicom/raw/fragment_data_set/**"
```

### Vérifier l'état
```bash
git lfs ls-files            # Liste des fichiers LFS
git lfs ls-files --size     # Avec leur taille
```

## Source des données

- **Rapports texte** : [Open-I (NLM)](https://openi.nlm.nih.gov/)
- **Images PNG** : extraites depuis les fichiers DICOM Open-I (preprocessing maison — voir `docs/dicom_preprocessing.md`)
- **DICOM** : Open-I + données ajoutées par l'équipe

## ⚠️ Confidentialité

Les images DICOM peuvent contenir des **métadonnées patient** (PatientName, PatientID...). Avant tout partage hors GitLab ENST, vérifier l'anonymisation :
```python
import pydicom
ds = pydicom.dcmread("data/dicom/raw/xxx.dcm")
print(ds.PatientName, ds.PatientID)  # doivent être anonymisés
```

## Ne PAS commit dans `data/`

- Caches dérivés → utiliser `data/cache/` (gitignored)
- Données temporaires → `data/interim/` (gitignored)
- Tout fichier > 100 MB qui n'est pas un dataset structurel
