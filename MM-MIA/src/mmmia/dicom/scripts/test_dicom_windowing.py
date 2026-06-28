"""
================================================================================
DICOM MULTI-WINDOW CONCEPT - Test & Visualization Script
================================================================================

CONCEPT :
---------
Un DICOM brut contient des valeurs 16-bit (-1024 à +4000) pour chaque pixel.
C'est trop pour l'oeil humain ou un modèle ML de traiter d'un coup.

SOLUTION : WINDOWING
---------------------
Appliquer une "fenêtre" qui isole une plage d'intérêt :
  • Fenêtre LUNGS      : Met en avant le parenchyme pulmonaire (nodules, infiltrations)
  • Fenêtre MEDIASTINUM: Met en avant le cœur et structures internes (cardiomégalie)
  • Fenêtre BONE       : Met en avant les structures osseuses (fractures)

MULTI-WINDOW RGB
----------------
Au lieu de 3 fichiers PNG séparés, on combine les 3 fenêtres dans 1 PNG RGB :
  • Canal R (Red)   = Lungs window
  • Canal G (Green) = Mediastinum window
  • Canal B (Blue)  = Bone window

RÉSULTAT :
1 DICOM → 1 PNG 3-canal (512×512×3) avec 3 perspectives médicales

AVANTAGE ML :
- Modèles CNN reçoivent 3 vues au lieu de 1
- Même taille de stockage
- Meilleure prédiction grâce au contexte enrichi

================================================================================
"""

import os
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# ===== CONFIGURATION =====
DICOM_DIR = "fragment_data_set"
OUTPUT_DIR = "dicom_windowing_test"

# Windows médicales (seront ajustées selon la plage réelle)
WINDOWS = {
    "lungs": {"level": -500, "width": 1500},        # Parenchyme pulmonaire
    "mediastinum": {"level": 40, "width": 400},     # Cœur et structures
    "bone": {"level": 300, "width": 1500}           # Structures osseuses
}

# Fonction pour adapter les fenêtres à la plage réelle des pixels
def adapt_windows_to_pixel_range(pixel_array):
    """Adapte les fenêtres en fonction de la plage réelle des pixels"""
    pixel_min = pixel_array.min()
    pixel_max = pixel_array.max()
    pixel_mean = pixel_array.mean()
    pixel_range = pixel_max - pixel_min
    
    # Adapter les niveaux de fenêtre à la plage réelle
    adapted_windows = {
        "lungs": {
            "level": pixel_mean - pixel_range * 0.2,      # 20% sous la moyenne
            "width": pixel_range * 0.6                     # Couvrir 60% de la plage
        },
        "mediastinum": {
            "level": pixel_mean,                           # Au centre
            "width": pixel_range * 0.3                     # Couvrir 30% de la plage
        },
        "bone": {
            "level": pixel_mean + pixel_range * 0.15,      # 15% au-dessus de la moyenne
            "width": pixel_range * 0.5                     # Couvrir 50% de la plage
        }
    }
    
    return adapted_windows, pixel_min, pixel_max, pixel_mean

# ===== UTILITAIRES =====
def apply_window(pixel_array, window_level, window_width):
    """Applique une fenêtre DICOM"""
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    windowed = np.clip(pixel_array, lower, upper)
    windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    return windowed

# Créer output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Trouver le premier DICOM
print("=" * 80)
print("TEST DICOM WINDOWING")
print("=" * 80)

dicom_files = []
for root, dirs, files in os.walk(DICOM_DIR):
    for file in files:
        if file.endswith('.dcm'):
            dicom_files.append(os.path.join(root, file))

if not dicom_files:
    print("❌ Aucun DICOM trouvé!")
    exit(1)

# Utiliser le premier DICOM
test_dicom_path = dicom_files[0]
print(f"\n📂 DICOM sélectionné: {test_dicom_path}")

# Charger le DICOM
print("\n📖 Chargement du DICOM...")
dcm = pydicom.dcmread(test_dicom_path)
pixel_array = dcm.pixel_array.astype(np.float32)

# Appliquer rescale si nécessaire
if hasattr(dcm, 'RescaleIntercept') and hasattr(dcm, 'RescaleSlope'):
    pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    print(f"   - Rescale appliquée (slope={dcm.RescaleSlope}, intercept={dcm.RescaleIntercept})")

print(f"   - Dimensions: {pixel_array.shape}")
print(f"   - Min=({pixel_array.min():.0f}), Max=({pixel_array.max():.0f}), Mean=({pixel_array.mean():.0f})")

# Adapter les fenêtres à la plage réelle des pixels
adapted_windows, px_min, px_max, px_mean = adapt_windows_to_pixel_range(pixel_array)

# Appliquer les 3 windows
print("\n🪟 Application des windows (adaptées à la plage réelle)...")
lungs = apply_window(pixel_array, adapted_windows["lungs"]["level"], adapted_windows["lungs"]["width"])
mediastinum = apply_window(pixel_array, adapted_windows["mediastinum"]["level"], adapted_windows["mediastinum"]["width"])
bone = apply_window(pixel_array, adapted_windows["bone"]["level"], adapted_windows["bone"]["width"])

print(f"   ✅ Lungs window: WL={adapted_windows['lungs']['level']:.0f}, WW={adapted_windows['lungs']['width']:.0f}")
print(f"   ✅ Mediastinum window: WL={adapted_windows['mediastinum']['level']:.0f}, WW={adapted_windows['mediastinum']['width']:.0f}")
print(f"   ✅ Bone window: WL={adapted_windows['bone']['level']:.0f}, WW={adapted_windows['bone']['width']:.0f}")

# Sauvegarder les 3 images individuelles
print("\n💾 Sauvegarde des images individuelles...")
filename_base = os.path.splitext(os.path.basename(test_dicom_path))[0]

Image.fromarray(lungs, mode='L').save(os.path.join(OUTPUT_DIR, f"{filename_base}_lungs.png"))
print(f"   ✅ {filename_base}_lungs.png")

Image.fromarray(mediastinum, mode='L').save(os.path.join(OUTPUT_DIR, f"{filename_base}_mediastinum.png"))
print(f"   ✅ {filename_base}_mediastinum.png")

Image.fromarray(bone, mode='L').save(os.path.join(OUTPUT_DIR, f"{filename_base}_bone.png"))
print(f"   ✅ {filename_base}_bone.png")

# Créer l'image RGB combinée (R=Lungs, G=Mediastinum, B=Bone)
print("\n🎨 Création de l'image RGB combinée...")
rgb_image = np.stack([lungs, mediastinum, bone], axis=2)
Image.fromarray(rgb_image, mode='RGB').save(os.path.join(OUTPUT_DIR, f"{filename_base}_multiwindow_RGB.png"))
print(f"   ✅ {filename_base}_multiwindow_RGB.png")
print(f"      (R=Lungs, G=Mediastinum, B=Bone)")

# Visualiser avec matplotlib
print("\n📊 Génération de la visualisation...")
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Image originale (grayscale simple)
original_normalized = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
axes[0, 0].imshow(original_normalized, cmap='gray')
axes[0, 0].set_title('Original (sans windowing)', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Lungs window
axes[0, 1].imshow(lungs, cmap='gray')
axes[0, 1].set_title('Lungs Window\n(WL=-500, WW=1500)', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Mediastinum window
axes[1, 0].imshow(mediastinum, cmap='gray')
axes[1, 0].set_title('Mediastinum Window\n(WL=40, WW=400)', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Bone window
axes[1, 1].imshow(bone, cmap='gray')
axes[1, 1].set_title('Bone Window\n(WL=300, WW=1500)', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}_comparison.png"), dpi=100, bbox_inches='tight')
print(f"   ✅ {filename_base}_comparison.png (comparaison des 4 versions)")

# Visualiser l'image RGB
fig_rgb, ax_rgb = plt.subplots(1, 1, figsize=(8, 8))
ax_rgb.imshow(rgb_image)
ax_rgb.set_title('Multi-Window RGB\n(R=Lungs, G=Mediastinum, B=Bone)', fontsize=14, fontweight='bold')
ax_rgb.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}_rgb_view.png"), dpi=100, bbox_inches='tight')
print(f"   ✅ {filename_base}_rgb_view.png (visualisation RGB)")

print("\n" + "=" * 80)
print("✨ TEST TERMINÉ!")
print("=" * 80)
print(f"\n Fichiers générés dans: {os.path.abspath(OUTPUT_DIR)}")
print(f"   - {filename_base}_lungs.png (grayscale)")
print(f"   - {filename_base}_mediastinum.png (grayscale)")
print(f"   - {filename_base}_bone.png (grayscale)")
print(f"   - {filename_base}_multiwindow_RGB.png (RGB 3-canal)")
print(f"   - {filename_base}_comparison.png (comparaison)")
print(f"   - {filename_base}_rgb_view.png (RGB visualisation)")

print("\n Explications:")
print("   - Lungs: Meilleure visibilité du parenchyme pulmonaire (nodules, infiltrations)")
print("   - Mediastinum: Meilleure visibilité du cœur et structures médiastinales")
print("   - Bone: Meilleure visibilité des structures osseuses")
print("   - RGB combiné: Les 3 windows encodées dans 1 image 3-canal")
# ===== CRÉER UNE IMAGE CONCEPTUELLE =====
print("\n📊 Génération de l'image conceptuelle...")

# Créer une figure explicative du concept
fig_concept = plt.figure(figsize=(14, 10))

# Titre principal
fig_concept.suptitle('DICOM MULTI-WINDOW: Concept & Workflow', fontsize=18, fontweight='bold', y=0.98)

# TextAreas avec explications
concept_text = """
CONCEPT: DICOM Multi-Window

1. DICOM Brut (16-bit)
   → Valeurs de pixels: -1024 à +4000 (Hounsfield Units)
   → Trop d'informations impossibles à traiter d'un coup

2. WINDOWING: Isolation d'une plage d'intérêt
   → Lungs window (WL=-500, WW=1500)     → Parenchyme pulmonaire
   → Mediastinum window (WL=40, WW=400)  → Cœur & structures
   → Bone window (WL=300, WW=1500)       → Structures osseuses

3. MULTI-WINDOW RGB: Combiner les 3 windows
   Canal R = Lungs      (rouge)
   Canal G = Mediastinum (vert)
   Canal B = Bone       (bleu)
   
   → Résultat: 1 PNG 3-canal (512×512×3)
   
4. ML ADVANTAGE:
   ✓ 3 perspectives au lieu d'1
   ✓ Même avant/après stockage
   ✓ Modèle CNN voit plus de détails
   ✓ Meilleure prédiction accurée

VISUAL RESULT:
   Zones Jaune/Clair  = Densité haute (os, structures)
   Zones Rouges       = Parenchyme pulmonaire
   Zones Noires       = Peu d'intérêt médical
"""

ax1 = fig_concept.add_subplot(1, 1, 1)
ax1.text(0.05, 0.95, concept_text, transform=ax1.transAxes, 
         fontfamily='monospace', fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax1.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}_CONCEPT_explanation.png"), dpi=150, bbox_inches='tight', facecolor='white')
print(f"   ✅ {filename_base}_CONCEPT_explanation.png")

# Créer une image de workflow
fig_workflow = plt.figure(figsize=(14, 8))
fig_workflow.suptitle('DICOM Multi-Window: Processing Pipeline', fontsize=16, fontweight='bold')

process_text = """
WORKFLOW: De DICOM aux PNG Multi-Window
════════════════════════════════════════════════════════════════════

INPUT: 1_IM-0001-3001.dcm (Radiographie thoracique brute 16-bit)
        ↓
        
ÉTAPE 1: Charger le DICOM
        └─ Lecture du fichier .dcm
        └─ Extraction du pixel_array (16-bit)
        └─ Plage: Min=12230, Max=32466, Mean=23583
        ↓
        
ÉTAPE 2: Adapter les fenêtres à la plage réelle
        ├─ Fenêtre Lungs:       WL=19535, WW=12142  (60% de la plage)
        ├─ Fenêtre Mediastinum: WL=23583, WW=6071   (30% de la plage)
        └─ Fenêtre Bone:        WL=26618, WW=10118  (50% de la plage)
        ↓
        
ÉTAPE 3: Appliquer les windows (conversion 16-bit → 8-bit)
        ├─ Clipping: lower = WL - WW/2, upper = WL + WW/2
        ├─ Normalisation: (pixel - lower) / (upper - lower) × 255
        ├─ OUTPUT 1: lungs.png (grayscale 8-bit)
        ├─ OUTPUT 2: mediastinum.png (grayscale 8-bit)
        └─ OUTPUT 3: bone.png (grayscale 8-bit)
        ↓
        
ÉTAPE 4: Combiner les 3 windows en RGB
        ├─ Stack [lungs, mediastinum, bone] → RGB array
        └─ OUTPUT: multiwindow_RGB.png (512×512×3)
        ↓
        
ÉTAPE 5: Utiliser en ML
        ├─ Charger: Image.open('multiwindow_RGB.png')
        ├─ Passer dans PyTorch CNN (entrée 3-canal standard)
        └─ Classification avec 3 perspectives médicales
        ↓
        
OUTPUT: 1 PNG RGB 3-canal
        (Les 3 fenêtres encodées dans 1 fichier)

AVANTAGES:
  ✓ Efficacité: 1 fichier au lieu de 3
  ✓ Stockage: Même taille avant/après
  ✓ ML: Modèle reçoit 3 perspectives
  ✓ Médicalement pertinent: Chaque canal = fenêtre diagnostique
"""

ax_workflow = fig_workflow.add_subplot(1, 1, 1)
ax_workflow.text(0.05, 0.95, process_text, transform=ax_workflow.transAxes, 
                fontfamily='monospace', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax_workflow.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}_WORKFLOW_explanation.png"), dpi=150, bbox_inches='tight', facecolor='white')
print(f"   ✅ {filename_base}_WORKFLOW_explanation.png")

print("\n✨ Images conceptuelles générées pour expliquer le concept!")