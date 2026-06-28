import subprocess
import sys

# Install required packages
packages = ['pandas', 'numpy', 'pydicom', 'scipy', 'tqdm']
for package in packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

import pandas as pd
import numpy as np
import os
from pathlib import Path
import pydicom
from scipy import ndimage
from tqdm import tqdm
import json
from datetime import datetime
import shutil

print("✅ All packages imported successfully!")

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_FOLDER = '/content/drive/MyDrive/artishow'
    print("✅ Google Drive mounted")
except ImportError:
    # Local development
    DRIVE_FOLDER = '/home/infres/ahmed-25/artishow'
    print("⚠️ Running locally (no Google Drive)")

# Configure paths
DICOM_FOLDER = os.path.join(DRIVE_FOLDER, 'dataset_dicom')
LABELS_CSV = os.path.join(DRIVE_FOLDER, 'dataset_labeled_enriched.csv')
VOLUMES_FOLDER = os.path.join(DRIVE_FOLDER, 'dicom_volume', 'volumes_preprocessed_3d')
PREPROCESSING_LOG = os.path.join(DRIVE_FOLDER, 'dicom_volume', 'preprocessing_log_3d')

# Create output folders
os.makedirs(VOLUMES_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSING_LOG, exist_ok=True)

print(f"\n📁 Paths configured:")
print(f"  DICOM folder: {DICOM_FOLDER}")
print(f"  Labels CSV: {LABELS_CSV}")
print(f"  Volumes output: {VOLUMES_FOLDER}")
print(f"  Logs: {PREPROCESSING_LOG}")

# Verify files exist
if os.path.exists(LABELS_CSV):
    print(f"✅ Labels CSV found")
else:
    print(f"❌ Labels CSV NOT found")

if os.path.exists(DICOM_FOLDER):
    num_patients = len([f for f in os.listdir(DICOM_FOLDER) if os.path.isdir(os.path.join(DICOM_FOLDER, f))])
    print(f"✅ DICOM folder found ({num_patients} patient folders)")
else:
    print(f"❌ DICOM folder NOT found")

# Load labels
df_labels = pd.read_csv(LABELS_CSV)

print(f"\n{'='*70}")
print("📊 DATASET OVERVIEW")
print(f"{'='*70}")
print(f"Total studies: {len(df_labels)}")
print(f"Columns: {list(df_labels.columns[:5])} ...")
print(f"\nFirst 3 rows:")
print(df_labels[['image_id', 'dicom_file_name', 'Cardiomegaly']].head(3))

# Define pathology classes
PATHOLOGY_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule',
    'Hernia', 'Fracture', 'Pleural_Thickening', 'Opacity', 'Consolidation',
    'Granuloma', 'Calcinosis', 'Scoliosis', 'Atherosclerosis', 'Normal'
]

print(f"\nNumber of pathology classes: {len(PATHOLOGY_CLASSES)}")

def load_dicom_series(dicom_path):
    """
    Load a DICOM series from a folder containing multiple .dcm files.
    Returns a 3D volume array.
    """
    dcm_files = [f for f in os.listdir(dicom_path) if f.endswith('.dcm')]
    
    if not dcm_files:
        return None
    
    # Load first DICOM to get shape
    dcm = pydicom.dcmread(os.path.join(dicom_path, dcm_files[0]))
    
    # Initialize volume
    volume = np.zeros((len(dcm_files), dcm.pixel_array.shape[0], dcm.pixel_array.shape[1]), dtype=np.float32)
    
    # Load all slices
    for i, dcm_file in enumerate(sorted(dcm_files)):
        dcm = pydicom.dcmread(os.path.join(dicom_path, dcm_file))
        volume[i] = dcm.pixel_array.astype(np.float32)
    
    return volume

def apply_windowing(volume, window_center=40, window_width=400):
    """
    Apply medical windowing (Hounsfield units).
    Typical CT chest: center=40, width=400
    """
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    windowed = np.clip(volume, window_min, window_max)
    windowed = (windowed - window_min) / (window_max - window_min)
    return windowed

def resize_volume(volume, target_shape=(128, 128, 64)):
    """
    Resize 3D volume to target shape.
    """
    current_shape = volume.shape
    zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
    resized = ndimage.zoom(volume, zoom_factors, order=1)
    return resized

def normalize_volume(volume):
    """
    Normalize volume to zero mean and unit variance.
    """
    mean = volume.mean()
    std = volume.std()
    if std > 0:
        return (volume - mean) / std
    return volume

print("✅ Preprocessing functions defined")

# Configuration
VOLUME_SIZE = (128, 128, 64)
WINDOW_CENTER = 40
WINDOW_WIDTH = 400

print(f"\n{'='*70}")
print("🔄 PREPROCESSING ALL DICOM STUDIES")
print(f"{'='*70}")
print(f"Target volume size: {VOLUME_SIZE}")
print(f"Window: center={WINDOW_CENTER}, width={WINDOW_WIDTH}")
print(f"Output folder: {VOLUMES_FOLDER}\n")

# Track results
results = {
    'success': [],
    'failed': [],
    'not_found': []
}

# Process each study
for idx, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Preprocessing DICOM"):
    image_id = row['image_id']
    
    # Extract patient ID (e.g., "CXR3037_IM-1410-1001" -> "3037")
    patient_id = image_id.split('_')[0].replace('CXR', '')
    patient_folder = os.path.join(DICOM_FOLDER, patient_id)
    
    try:
        # Check if folder exists
        if not os.path.exists(patient_folder):
            results['not_found'].append(image_id)
            continue
        
        # Load DICOM series
        volume = load_dicom_series(patient_folder)
        
        if volume is None:
            results['failed'].append((image_id, "No DICOM files found"))
            continue
        
        # Apply preprocessing
        volume = apply_windowing(volume, WINDOW_CENTER, WINDOW_WIDTH)
        volume = resize_volume(volume, VOLUME_SIZE)
        volume = normalize_volume(volume)
        
        # Save as .npy
        volume_path = os.path.join(VOLUMES_FOLDER, f"{image_id}.npy")
        np.save(volume_path, volume.astype(np.float32))
        
        results['success'].append((image_id, volume_path))
    
    except Exception as e:
        results['failed'].append((image_id, str(e)))

# Print results
print(f"\n{'='*70}")
print("✅ PREPROCESSING COMPLETE")
print(f"{'='*70}")
print(f"Successfully processed: {len(results['success'])}")
print(f"Failed: {len(results['failed'])}")
print(f"Not found: {len(results['not_found'])}")
print(f"Total: {len(df_labels)}")

if results['failed']:
    print(f"\nFailed cases (first 5):")
    for image_id, error in results['failed'][:5]:
        print(f"  - {image_id}: {error}")

if results['not_found']:
    print(f"\nNot found (first 5):")
    for image_id in results['not_found'][:5]:
        print(f"  - {image_id}")

# Create index CSV
successful_data = []
for image_id, volume_path in results['success']:
    # Get original row
    orig_row = df_labels[df_labels['image_id'] == image_id].iloc[0].to_dict()
    # Add volume path
    orig_row['volume_npy_path'] = volume_path
    successful_data.append(orig_row)

# Create dataframe
df_volumes = pd.DataFrame(successful_data)

# Save CSV
volumes_csv = os.path.join(DRIVE_FOLDER, 'dataset_labeled_volumes_3d.csv')
df_volumes.to_csv(volumes_csv, index=False)

print(f"\n{'='*70}")
print("💾 CSV INDEX CREATED")
print(f"{'='*70}")
print(f"File: {volumes_csv}")
print(f"Rows: {len(df_volumes)}")
print(f"\nFirst 3 rows:")
print(df_volumes[['image_id', 'Cardiomegaly', 'Normal']].head(3))