"""Configuration du preprocessing DICOM"""

from pathlib import Path

# src/mmmia/dicom/scripts/config.py → racine 4 niveaux au-dessus
PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATA_DIR = PROJECT_ROOT / "data" / "dicom" / "raw" / "fragment_data_set"
XML_DIR = PROJECT_ROOT / "data" / "labeling" / "NLMCXR_reports" / "ecgen-radiology"
CSV_FILE = PROJECT_ROOT / "data" / "shared" / "dataset_labeled.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "dicom" / "raw" / "dataset_labeled_enriched.csv"
DICOM_OUTPUT_DIR = PROJECT_ROOT / "data" / "dicom" / "preprocessed_multiwindow"

# Créer les dossiers de sortie si besoin
DICOM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration preprocessing DICOM
IMAGE_SIZE = (256, 256)

# Windows radiologiques (Hounsfield Units)
LUNG_WINDOW = {"center": -150, "width": 400}
MEDIASTINAL_WINDOW = {"center": 150, "width": 400}
