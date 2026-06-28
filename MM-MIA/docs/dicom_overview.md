# 🫁 Preprocessing DICOM + XML

Pipeline de preprocessing pour linking DICOM ↔ PNG ↔ XML et enrichissement métadonnées médicales.

## 📁 Structure

```
projet_dicom/
├── fragment_data_set/              # Données DICOM brutes
├── NLMCXR_reports/                 # Rapports médicaux XML
├── dicom_preprocessed/             # DICOM multifenêtres (générés)
├── extract_labels.py               # Extraction métadonnées DICOM
├── enrich_dataset_with_xml.py      # Enrichissement dataset XML
├── config.py                       # Configuration
├── requirements.txt                # Dépendances
└── README.md                       # Ce fichier
```

## 🚀 Installation & Utilisation

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Extraire étiquettes DICOM

```bash
python extract_labels.py
```

Génère `dicom_labels.csv` avec métadonnées des fichiers DICOM.

### 3. Enrichir le dataset avec XML

```bash
python enrich_dataset_with_xml.py
```

Génère `dataset_labeled_enriched.csv` avec:
- ✅ Linking PNG ↔ XML via pattern `IM-XXXX-YYYY`
- ✅ Métadonnées texte: indication, findings, impression, comparison
- ✅ MeSH tags diagnostiques
- ✅ Identification fichiers DICOM correspondants

## 📊 Fichiers générés

| Fichier | Contenu |
|---------|---------|
| `dicom_labels.csv` | Métadonnées DICOM extraites |
| `dataset_labeled_enriched.csv` | Dataset enrichi: 7470 images × 39 colonnes |
| `dicom_preprocessed/` | DICOM multifenêtres (à implémenter) |

## 🔧 Configuration

`config.py`:
- `IMAGE_SIZE`: Résolution preprocessing (défaut: 256×256)
- `LUNG_WINDOW`: Center=-150, Width=400
- `MEDIASTINAL_WINDOW`: Center=150, Width=400

## ⚙️ Prochaines étapes

- [ ] Créer script preprocessing DICOM multifenêtres
- [ ] Valider qualité des images resizées
- [ ] Analyser distribution des pathologies
   labels_dict[patient_id] = classe  # 0=Normal, 1=Pathologie, etc.
   ```

2. **Architecture du modèle** dans `main.py`:
   ```python
   trainer = Trainer(model_name="resnet18")  # ou "simple_cnn"
   ```

3. **Hyperparamètres** dans `config.py`:
   ```python
   BATCH_SIZE = 4
   LEARNING_RATE = 1e-4
   NUM_EPOCHS = 50
   ```

## 📈 Sortie du modèle

Les résultats sont sauvegardés dans:
- `models/best_model_*.pth` : Meilleur modèle
- `runs/history_*.json` : Historique d'entraînement

## 💡 Modèles disponibles

- **SimpleCNN**: CNN 4 couches légère et rapide
- **ResNet18**: Modèle pré-entraîné plus puissant

Choisissez selon vos ressources!
