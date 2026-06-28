# DICOM Multi-Window Preprocessing

## 📋 Vue d'ensemble

Ce projet implémente un **preprocessing DICOM Multi-Window** pour transformer des images médicales brutes en PNG optimisées pour le Machine Learning.

### Concept clé

Un DICOM brut contient des pixels 16-bit avec des valeurs entre -1024 et +4000 (Hounsfield Units). C'est beaucoup trop pour qu'un modèle ML l'analyse efficacement.

**Solution** : appliquer 3 **fenêtres (windows)** différentes pour isoler des zones d'intérêt médical :

| Window | Objective | Output |
|--------|-----------|--------|
| **Lungs** | Parenchyme pulmonaire (nodules, infiltrations) | Canal R |
| **Mediastinum** | Cœur et structures médiastinales | Canal G |
| **Bone** | Structures osseuses (fractures) | Canal B |

**Résultat** : 1 PNG RGB 3-canal qui encode les 3 perspectives médicales.

---

## 🚀 Utilisation

### Installation

```bash
pip install pydicom pillow numpy
```

### Exécution

```bash
cd projet_dicom
python preprocessing_dicom_multiwindow.py
```

### Input/Output

- **Input** : Dossier `fragment_data_set/` (fichiers `.dcm`)
- **Output** : Dossier `preprocessed_images_multiwindow/` (PNG RGB 3-canal)

### Exemple

```
fragment_data_set/1/1_IM-0001-3001.dcm
        ↓ (preprocessing)
preprocessed_images_multiwindow/1_IM-0001-3001_multiwindow.png
```

---

## 📊 Architecture du preprocessing

1. **Charger le DICOM** (pydicom)
   - Extraction du pixel array 16-bit
   - Application du rescaling médical si nécessaire

2. **Adapter les fenêtres**
   - Calcul automatique basé sur la plage réelle des pixels
   - Lungs: 60% de la plage
   - Mediastinum: 30% de la plage
   - Bone: 50% de la plage

3. **Appliquer les windows**
   - Clipping des valeurs à la plage [WL-WW/2, WL+WW/2]
   - Normalisation 8-bit (0-255)

4. **Combiner en RGB**
   - Stack [lungs, mediastinum, bone] → RGB
   - Sauvegarder en PNG

---

## 💡 Avantages

✅ **Efficacité** : 1 fichier PNG au lieu de 3
✅ **Stockage** : Même taille avant/après
✅ **ML** : Les modèles CNN reçoivent 3 perspectives
✅ **Médical** : Chaque canal = fenêtre diagnostique réelle

---

## 📁 Fichiers du projet

```
projet_dicom/
├── preprocessing_dicom_multiwindow.py     ← Script principal (production)
├── test_dicom_windowing.py                ← Script de test (avec visualisations)
├── fragment_data_set/                     ← DICOM input
│   ├── 1-20260326T154122Z-3-001/1/
│   ├── 2-20260326T154123Z-3-001/2/
│   └── ...
├── preprocessed_images_multiwindow/       ← PNG output
│   ├── 1_IM-0001-3001_multiwindow.png
│   ├── 1_IM-0001-4001_multiwindow.png
│   └── ...
└── README.md                              ← Ce fichier
```

---

## 🔬 Résultats du fragment test

- **Input** : 10 DICOM
- **Output** : 10 PNG RGB multiwindow
- **Taux de succès** : 100%

---

## 🎯 Prochaines étapes

1. **Google Colab** : Adapter le script pour traiter les 7000 images
2. **Integration** : Utiliser les PNG multiwindow avec vos modèles PyTorch
3. **Benchmarking** : Tester si la performance du modèle s'améliore avec le multi-window

---

## 📝 Notes

- Les fenêtres sont automatiquement adaptées à la plage réelle de chaque image
- Aucune perte d'information cliniquement pertinente
- Compatible avec tous les modèles CNN standard (entrée 3-canal RGB)

---

**Question ?** Consultez `test_dicom_windowing.py` pour voir les images d'explication du concept.
