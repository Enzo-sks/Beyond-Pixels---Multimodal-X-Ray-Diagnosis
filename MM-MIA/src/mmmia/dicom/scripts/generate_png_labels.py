"""
PNG Label Mapping Generator
Extrait les IDs DICOM des PNG et match avec dataset_labeled_enriched.csv
"""

import pandas as pd
import re
import os
from pathlib import Path

# Racine du repo : 4 niveaux au-dessus de ce script (src/mmmia/dicom/scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[4]

CSV_PATH = PROJECT_ROOT / "data" / "dicom" / "raw" / "dataset_labeled_enriched.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dicom" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Liste des PNG (copiée de l'utilisateur)
png_list = """1028_IM-0022-2001_multiwindow.png
1029_IM-0022-1001_multiwindow.png
102_IM-0016-1001_multiwindow.png
102_IM-0016-2001_multiwindow.png
1030_IM-0024-1001_multiwindow.png
1030_IM-0024-2001_multiwindow.png
1031_IM-0025-4004_multiwindow.png
1032_IM-0026-1001-0001_multiwindow.png
1032_IM-0026-1001-0002_multiwindow.png
1033_IM-0027-2001_multiwindow.png
1033_IM-0027-4004_multiwindow.png
1034_IM-0028-1001_multiwindow.png
1034_IM-0028-1002_multiwindow.png
1035_IM-0028-1001_multiwindow.png
1035_IM-0028-2001_multiwindow.png
1036_IM-0029-1001_multiwindow.png
1037_IM-0029-1001_multiwindow.png
1037_IM-0029-2001_multiwindow.png
1038_IM-0029-1001_multiwindow.png
1038_IM-0029-2001_multiwindow.png
1039_IM-0030-1002_multiwindow.png
103_IM-0023-1001_multiwindow.png
103_IM-0023-2001_multiwindow.png
1040_IM-0032-2001_multiwindow.png
1040_IM-0032-4004_multiwindow.png
1041_IM-0033-1001_multiwindow
1072_IM-0052-1001-0002_multiwindow.png
1073_IM-0053-1001_multiwindow.png
1073_IM-0053-2001_multiwindow.png
1074_IM-0054-1001_multiwindow.png
1074_IM-0054-2001_multiwindow.png
1075_IM-0054-1001_multiwindow.png
1075_IM-0054-2001_multiwindow.png
1076_IM-0054-1001_multiwindow.png
1076_IM-0054-3001_multiwindow.png
1077_IM-0054-1001_multiwindow.png
1077_IM-0054-2001_multiwindow.png
1078_IM-0055-1001_multiwindow.png
1078_IM-0055-2001_multiwindow.png
1079_IM-0056-1001_multiwindow.png
1079_IM-0056-2001_multiwindow.png
107_IM-0049-1001_multiwindow.png
107_IM-0049-2001_multiwindow.png
1081_IM-0057-2002_multiwindow.png
1082_IM-0058-1001_multiwindow.png
1083_IM-0058-1001_multiwindow.png
1083_IM-0058-2001_multiwindow.png
1084_IM-0058-1001_multiwindow.png
1084_IM-0058-2001_multiwindow.png
1085_IM-0059-1001_multiwindow.png
1085_IM-0059-2001_multiwindow.png
1086_IM-0
1114_IM-0079-1001_multiwindow.png
1114_IM-0079-2001_multiwindow.png
1115_IM-0079-1001_multiwindow.png
1116_IM-0079-1001_multiwindow.png
1116_IM-0079-2001_multiwindow.png
1117_IM-0079-1001_multiwindow.png
1117_IM-0079-2001_multiwindow.png
1118_IM-0079-1001_multiwindow.png
1118_IM-0079-2001_multiwindow.png
1119_IM-0080-1001_multiwindow.png
1119_IM-0080-2001_multiwindow.png
111_IM-0076-1001_multiwindow.png
111_IM-0076-1002_multiwindow.png
1120_IM-0080-1001_multiwindow.png
1121_IM-0080-1001_multiwindow.png
1121_IM-0080-2001_multiwindow.png
1122_IM-0080-1001-0001_multiwindow.png
1122_IM-0080-1001-0002_multiwindow.png
1122_IM-0080-2001_multiwindow.png
1123_IM-0080-1001_multiwindow.png
1123_IM-0080-2001_multiwindow.png
1124_IM-0081-2001_multiwindow.png
1124_IM-0081-3001_multiwindow.png
1125_IM-0082-1001_multiwindow.png
1125_IM-0082-2001_multiwindow.png"""

print("=" * 80)
print("🎯 PNG LABEL MAPPING GENERATOR")
print("=" * 80)

# Charger le dataset
print(f"\n📂 chargement: {CSV_PATH}")
df_labels = pd.read_csv(CSV_PATH)
print(f"✅ Dataset chargé: {df_labels.shape[0]} lignes")

# Parser les PNG
png_files = [f.strip() for f in png_list.strip().split('\n') if f.strip() and len(f.strip()) > 5]
print(f"\n📋 PNG à traiter: {len(png_files)} fichiers")

# Colonnes de findings
finding_columns = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 
    'Infiltration', 'Mass', 'Nodule', 'Hernia', 'Fracture', 
    'Pleural_Thickening', 'Opacity', 'Consolidation', 'Granuloma', 
    'Calcinosis', 'Scoliosis', 'Atherosclerosis', 'Normal'
]

# Mapping
mapping_data = []
matched = 0
unmatched = []
duplicates = 0

print(f"\n🔗 Extraction et matching des IDs DICOM...\n")

for i, png_file in enumerate(png_files):
    # Nettoyer le nom
    temp_name = png_file.replace('_multiwindow.png', '').replace('.png', '').strip()
    
    # Extraire l'ID DICOM: IM-XXXX-XXXX
    match = re.search(r'(IM-\d+-\d+)', temp_name)
    
    if match:
        dicom_id = match.group(1)
        
        # Recherche dans le dataset
        matching_rows = df_labels[df_labels['dicom_im_pattern'] == dicom_id]
        
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            
            # Récupérer les findings positifs
            findings = [col for col in finding_columns if row[col] == 1]
            primary_label = findings[0] if findings else 'Normal'
            
            mapping_data.append({
                'png_filename': png_file,
                'dicom_id': dicom_id,
                'primary_label': primary_label,
                'all_findings': ','.join(findings),
                'image_id': row['image_id'],
                'findings_text': row['findings'],
                'impression': row['impression']
            })
            matched += 1
            
            if (i + 1) % 50 == 0:
                print(f"  ✓ {i + 1}/{len(png_files)} traités... ({matched} matchés)")
        else:
            unmatched.append((png_file, dicom_id))
    else:
        unmatched.append((png_file, 'NO_PATTERN'))

print(f"\n{'='*80}")
print(f"📊 RÉSULTATS DU MATCHING")
print(f"{'='*80}")
print(f"✅ Matchés:    {matched}/{len(png_files)} ({matched/len(png_files)*100:.1f}%)")
print(f"❌ Non matchés: {len(unmatched)}")

if len(unmatched) > 0:
    print(f"\n⚠️  Premiers non-matchés (max 10):")
    for png_file, dicom_id in unmatched[:10]:
        print(f"   - {png_file} → {dicom_id}")

# Créer DataFrame
df_mapping = pd.DataFrame(mapping_data)

print(f"\n📋 Distribution des labels principaux:")
print(df_mapping['primary_label'].value_counts())

# Sauvegarder les CSVs
mapping_csv = os.path.join(OUTPUT_DIR, 'png_labels_mapping_enriched.csv')
df_mapping.to_csv(mapping_csv, index=False)
print(f"\n💾 CSV enrichi sauvegardé: {mapping_csv}")

simple_csv = os.path.join(OUTPUT_DIR, 'png_labels_mapping_simple.csv')
df_mapping[['png_filename', 'dicom_id', 'primary_label', 'all_findings']].to_csv(simple_csv, index=False)
print(f"💾 CSV simple sauvegardé: {simple_csv}")

print(f"\n{'='*80}")
print(f"✨ GÉNÉRÉ AVEC SUCCÈS!")
print(f"{'='*80}")
print(f"\n📁 Fichiers générés:")
print(f"   1. png_labels_mapping_enriched.csv ({len(df_mapping)} lignes)")
print(f"   2. png_labels_mapping_simple.csv ({len(df_mapping)} lignes)")
