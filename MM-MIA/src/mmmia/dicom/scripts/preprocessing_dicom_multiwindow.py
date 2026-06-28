"""
DICOM Multi-Window Preprocessing - Production Script
Processing all DICOM files and generating single PNG 3-channel output
"""

import os
import numpy as np
import pydicom
from PIL import Image
from pathlib import Path

# ===== CONFIGURATION =====
DICOM_DIR = "fragment_data_set"
OUTPUT_DIR = "preprocessed_images_multiwindow"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def adapt_windows_to_pixel_range(pixel_array):
    """Adapte les fenêtres en fonction de la plage réelle des pixels"""
    pixel_min = pixel_array.min()
    pixel_max = pixel_array.max()
    pixel_mean = pixel_array.mean()
    pixel_range = pixel_max - pixel_min
    
    adapted_windows = {
        "lungs": {
            "level": pixel_mean - pixel_range * 0.2,
            "width": pixel_range * 0.6
        },
        "mediastinum": {
            "level": pixel_mean,
            "width": pixel_range * 0.3
        },
        "bone": {
            "level": pixel_mean + pixel_range * 0.15,
            "width": pixel_range * 0.5
        }
    }
    return adapted_windows

def apply_window(pixel_array, window_level, window_width):
    """Applique une fenêtre DICOM et convertit 16-bit → 8-bit"""
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    windowed = np.clip(pixel_array, lower, upper)
    windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    return windowed

def process_dicom(dicom_path):
    """Charge DICOM, applique les 3 windows, retourne l'image RGB"""
    try:
        dcm = pydicom.dcmread(dicom_path)
        pixel_array = dcm.pixel_array.astype(np.float32)
        
        if hasattr(dcm, 'RescaleIntercept') and hasattr(dcm, 'RescaleSlope'):
            pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        
        windows = adapt_windows_to_pixel_range(pixel_array)
        
        lungs = apply_window(pixel_array, windows["lungs"]["level"], windows["lungs"]["width"])
        mediastinum = apply_window(pixel_array, windows["mediastinum"]["level"], windows["mediastinum"]["width"])
        bone = apply_window(pixel_array, windows["bone"]["level"], windows["bone"]["width"])
        
        rgb_image = np.stack([lungs, mediastinum, bone], axis=2)
        return rgb_image, None
    except Exception as e:
        return None, str(e)

# ===== MAIN PROCESSING =====
print("=" * 80)
print("DICOM MULTI-WINDOW PREPROCESSING")
print("=" * 80)

# Find all DICOM files
dicom_files = []
for root, dirs, files in os.walk(DICOM_DIR):
    for file in files:
        if file.endswith('.dcm'):
            dicom_files.append(os.path.join(root, file))

print(f"\n🔍 Found {len(dicom_files)} DICOM files\n")

if not dicom_files:
    print("❌ No DICOM files found!")
    exit(1)

# Process each DICOM
success_count = 0
error_count = 0

for idx, dicom_path in enumerate(sorted(dicom_files), 1):
    filename = os.path.basename(dicom_path)
    filename_base = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{filename_base}_multiwindow.png")
    
    rgb_image, error = process_dicom(dicom_path)
    
    if rgb_image is not None:
        Image.fromarray(rgb_image, mode='RGB').save(output_path, quality=95)
        success_count += 1
        status = "✅"
    else:
        error_count += 1
        status = "❌"
    
    print(f"[{idx:2d}/{len(dicom_files)}] {status} {filename}")

print("\n" + "=" * 80)
print(f"✅ Succès: {success_count}/{len(dicom_files)}")
print(f"❌ Erreurs: {error_count}/{len(dicom_files)}")
print(f"📁 Output: {os.path.abspath(OUTPUT_DIR)}")
print("=" * 80)
