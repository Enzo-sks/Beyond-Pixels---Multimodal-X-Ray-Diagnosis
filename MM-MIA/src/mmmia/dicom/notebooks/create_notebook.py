import json
import os

cells = []

def add_code_cell(code):
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in code.split('\n')]
    })

def add_markdown_cell(text):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.split('\n')]
    })

add_markdown_cell('# Preprocessing Avancé (Multimodal) des DICOM 3D\nCe notebook applique les corrections de fenêtrage (Lung Window), extrait les métadonnées (Age, Sexe, Description), redimensionne en 224x224x64, et sauvegarde le tout sur Google Drive.')

add_code_cell('!pip install pydicom scipy tqdm')

add_code_cell('''from google.colab import drive
import os

# Montage du Google Drive
drive.mount('/content/drive')

# Configuration des chemins sur le Drive
DRIVE_FOLDER = '/content/drive/MyDrive/artishow'
DICOM_FOLDER = os.path.join(DRIVE_FOLDER, 'dataset_dicom')
LABELS_CSV = os.path.join(DRIVE_FOLDER, 'dataset_labeled_enriched.csv')

# Nouveaux dossiers de sortie
VOLUMES_FOLDER = os.path.join(DRIVE_FOLDER, 'dicom_volume', 'volumes_preprocessed_3d_v2')
OUTPUT_CSV = os.path.join(DRIVE_FOLDER, 'dicom_volume', 'dataset_labeled_volumes_3d_multimodal.csv')

os.makedirs(VOLUMES_FOLDER, exist_ok=True)
print('✅ Chemins configurés')''')

add_code_cell('''import pandas as pd
import numpy as np
import pydicom
from scipy import ndimage
from tqdm.notebook import tqdm

# --- PARAMETRES DE TRAITEMENT ---
TARGET_SHAPE = (224, 224, 64) # Résolution adaptée pour les CNN 3D modernes
WINDOW_CENTER = -600  # Lung Window
WINDOW_WIDTH = 1500   # Lung Window

print(f'Cible : {TARGET_SHAPE} avec Lung Window')''')

add_code_cell('''def parse_age(age_str):
    """Convert DICOM Age string (e.g., '056Y') to integer."""
    if not age_str:
        return None
    try:
        return int(age_str.replace('Y', '').replace('M', '').replace('D', ''))
    except:
        return None

def process_dicom_series(dicom_path):
    """
    Charge une série DICOM, extrait les métadonnées, corrige les HU, fenètre et redimensionne.
    Retourne (volume_numpy, metadata_dict)
    """
    dcm_files = [f for f in os.listdir(dicom_path) if f.endswith('.dcm')]
    if not dcm_files:
        raise ValueError("Aucun fichier .dcm trouvé")
        
    dcm_files.sort()
    
    # Lire le premier fichier pour extraire les métadonnées
    first_dcm = pydicom.dcmread(os.path.join(dicom_path, dcm_files[0]))
    
    metadata = {
        'patient_age': parse_age(getattr(first_dcm, 'PatientAge', None)),
        'patient_sex': getattr(first_dcm, 'PatientSex', 'Unknown'),
        'study_description': getattr(first_dcm, 'StudyDescription', 'Unknown'),
        'manufacturer': getattr(first_dcm, 'Manufacturer', 'Unknown')
    }
    
    intercept = getattr(first_dcm, 'RescaleIntercept', 0)
    slope = getattr(first_dcm, 'RescaleSlope', 1)
    
    # Charger tout le volume brut
    volume_raw = np.zeros((len(dcm_files), first_dcm.pixel_array.shape[0], first_dcm.pixel_array.shape[1]), dtype=np.float32)
    for i, dcm_file in enumerate(dcm_files):
        dcm = pydicom.dcmread(os.path.join(dicom_path, dcm_file))
        volume_raw[i] = dcm.pixel_array.astype(np.float32)
        
    # 1. Convertir en vraies unités Hounsfield (HU)
    volume_hu = volume_raw * slope + intercept
    
    # 2. Fenêtrage Pulmonaire (Lung Window)
    window_min = WINDOW_CENTER - WINDOW_WIDTH / 2
    window_max = WINDOW_CENTER + WINDOW_WIDTH / 2
    volume_windowed = np.clip(volume_hu, window_min, window_max)
    
    # 3. Normalisation Min-Max [0, 1]
    volume_norm = (volume_windowed - window_min) / (window_max - window_min)
    
    # 4. Redimensionnement (Resize 224x224x64)
    zoom_factors = tuple(t / c for t, c in zip(TARGET_SHAPE, volume_norm.shape))
    volume_resized = ndimage.zoom(volume_norm, zoom_factors, order=1)
    
    return volume_resized, metadata''')

add_code_cell('''if not os.path.exists(LABELS_CSV):
    print(f"❌ Erreur: Le fichier CSV {LABELS_CSV} est introuvable.")
else:
    df = pd.read_csv(LABELS_CSV)
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Traitement des Volumes"):
        image_id = row['image_id']
        patient_id = image_id.split('_')[0].replace('CXR', '')
        patient_folder = os.path.join(DICOM_FOLDER, patient_id)
        
        row_dict = row.to_dict()
        
        try:
            if not os.path.exists(patient_folder):
                raise FileNotFoundError("Dossier introuvable")
                
            volume, meta = process_dicom_series(patient_folder)
            
            # Sauvegarder le volume Numpy
            vol_path = os.path.join(VOLUMES_FOLDER, f"{image_id}.npy")
            np.save(vol_path, volume.astype(np.float32))
            
            row_dict['volume_npy_path'] = vol_path
            row_dict['patient_age'] = meta['patient_age']
            row_dict['patient_sex'] = meta['patient_sex']
            row_dict['study_description'] = meta['study_description']
            row_dict['manufacturer'] = meta['manufacturer']
            row_dict['preprocessing_status'] = 'SUCCESS'
            
        except Exception as e:
            row_dict['preprocessing_status'] = f"FAILED: {str(e)}"
            row_dict['volume_npy_path'] = None
            
        results.append(row_dict)
        
    df_results = pd.DataFrame(results)
    
    # Filtrer uniquement les succès
    df_success = df_results[df_results['preprocessing_status'] == 'SUCCESS']
    df_success.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\\n{'='*50}")
    print("✅ PRÉTRAITEMENT TERMINÉ")
    print(f"{'='*50}")
    print(f"Volumes sauvegardés dans : {VOLUMES_FOLDER}")
    print(f"Nouveau CSV généré : {OUTPUT_CSV}")
    print(f"Succès : {len(df_success)} / {len(df_results)}")''')

notebook = {
    'cells': cells,
    'metadata': {
        'colab': {
            'provenance': []
        },
        'kernelspec': {
            'display_name': 'Python 3',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 0
}

output_path = r'C:\Users\ahmed\Desktop\artishow\artihow _final\MM-MIA\src\mmmia\dicom\notebooks\Advanced_Preprocess_DICOM_to_Volumes_3D.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print('Notebook created successfully')
