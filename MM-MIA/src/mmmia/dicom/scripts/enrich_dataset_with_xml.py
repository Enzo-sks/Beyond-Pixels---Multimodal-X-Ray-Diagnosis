"""
Enhancement Script: Enrich dataset_labeled.csv with XML metadata and DICOM matching
"""

import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from collections import defaultdict

# src/mmmia/dicom/scripts/enrich_dataset_with_xml.py → racine 4 niveaux au-dessus
PROJECT_ROOT = Path(__file__).resolve().parents[4]
XML_DIR = str(PROJECT_ROOT / "data" / "labeling" / "NLMCXR_reports" / "ecgen-radiology")
DICOM_DIR = str(PROJECT_ROOT / "data" / "dicom" / "raw" / "fragment_data_set")
CSV_FILE = str(PROJECT_ROOT / "data" / "shared" / "dataset_labeled.csv")
OUTPUT_CSV = str(PROJECT_ROOT / "data" / "dicom" / "raw" / "dataset_labeled_enriched.csv")

print("=" * 80)
print("ÉTAPE 1: Extraction des métadonnées XML")
print("=" * 80)

xml_metadata = defaultdict(dict)
xml_files_processed = 0
im_patterns_found = 0

for xml_file in sorted(os.listdir(XML_DIR)):
    if not xml_file.endswith('.xml'):
        continue
    
    try:
        xml_path = os.path.join(XML_DIR, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extraire pmcId
        pmc_elem = root.find(".//pmcId")
        pmc_id = pmc_elem.get("id") if pmc_elem is not None else None
        
        # Extraire Abstract
        abstract_data = {}
        for text_elem in root.findall(".//AbstractText"):
            label = text_elem.get("Label")
            text_content = text_elem.text if text_elem.text else ""
            if label:
                abstract_data[label] = text_content
        
        # Extraire MeSH tags
        mesh_tags = []
        for major_elem in root.findall(".//MeSH/major"):
            if major_elem.text:
                mesh_tags.append(major_elem.text)
        
        # Extraire parentImage IDs avec IM-XXXX-YYYY
        for parent_img in root.findall(".//parentImage"):
            img_id = parent_img.get("id")
            if img_id:
                # Extraire IM-XXXX-YYYY du format CXR1148_IM-1148-1001
                match = re.search(r"IM-(\d{4})-(\d{4})", img_id)
                if match:
                    im_pattern = f"IM-{match.group(1)}-{match.group(2)}"
                    
                    xml_metadata[im_pattern] = {
                        "xml_id": pmc_id,
                        "indication": abstract_data.get("INDICATION", ""),
                        "findings": abstract_data.get("FINDINGS", ""),
                        "impression": abstract_data.get("IMPRESSION", ""),
                        "comparison": abstract_data.get("COMPARISON", ""),
                        "mesh_tags": ",".join(mesh_tags) if mesh_tags else ""
                    }
                    im_patterns_found += 1
        
        xml_files_processed += 1
        
        if xml_files_processed % 500 == 0:
            print(f"  [{xml_files_processed}] fichiers XML traités, {im_patterns_found} patterns IM trouvés")
    
    except Exception as e:
        print(f"  ⚠️  Erreur en parsant {xml_file}: {str(e)}")
        continue

print(f"\n✅ Total: {xml_files_processed} fichiers XML traités")
print(f"✅ Total: {len(xml_metadata)} patterns IM-XXXX-YYYY uniques trouvés\n")

# Construire un dictionnaire des fichiers DICOM disponibles (RECHERCHE RÉCURSIVE)
print("=" * 80)
print("ÉTAPE 2: Inventaire des fichiers DICOM (recherche récursive)")
print("=" * 80)

dicom_files = {}
if os.path.exists(DICOM_DIR):
    for root, dirs, files in os.walk(DICOM_DIR):
        for dicom_file in files:
            if dicom_file.endswith('.dcm'):
                # Extraire IM-XXXX-YYYY
                match = re.search(r"IM-(\d{4})-(\d{4})", dicom_file)
                if match:
                    im_pattern = f"IM-{match.group(1)}-{match.group(2)}"
                    # Stocker le chemin complet relatif
                    full_path = os.path.join(root, dicom_file)
                    dicom_files[im_pattern] = full_path
    
    print(f"✅ {len(dicom_files)} fichiers DICOM trouvés (recherche récursive)")
else:
    print(f"⚠️  Répertoire {DICOM_DIR} non trouvé")

print()

# Charger le CSV
print("=" * 80)
print("ÉTAPE 3: Enrichissement du CSV (mapping COMPLET)")
print("=" * 80)

df = pd.read_csv(CSV_FILE)
print(f"✅ CSV chargé: {len(df)} lignes\n")

# Ajouter les colonnes
new_columns = [
    "xml_report_id",
    "dicom_im_pattern", 
    "dicom_file_name",
    "xml_indication",
    "xml_findings",
    "xml_impression",
    "xml_comparison",
    "xml_mesh_tags"
]

for col in new_columns:
    if col not in df.columns:
        df[col] = None

# Enrichir chaque ligne
matched_count = 0
dicom_matched_count = 0
xml_matched_count = 0

for idx, row in df.iterrows():
    image_id = row["image_id"]
    
    # Extraire IM-XXXX-YYYY du image_id
    match = re.search(r"IM-(\d{4})-(\d{4})", image_id)
    if match:
        im_pattern = f"IM-{match.group(1)}-{match.group(2)}"
        df.at[idx, "dicom_im_pattern"] = im_pattern
        
        # Chercher dans xml_metadata (TOUS les patterns trouvés dans les XML)
        if im_pattern in xml_metadata:
            metadata = xml_metadata[im_pattern]
            df.at[idx, "xml_report_id"] = metadata["xml_id"]
            df.at[idx, "xml_indication"] = metadata["indication"]
            df.at[idx, "xml_findings"] = metadata["findings"]
            df.at[idx, "xml_impression"] = metadata["impression"]
            df.at[idx, "xml_comparison"] = metadata["comparison"]
            df.at[idx, "xml_mesh_tags"] = metadata["mesh_tags"]
            xml_matched_count += 1
        
        # Chercher le fichier DICOM correspondant (fragment du dataset)
        if im_pattern in dicom_files:
            df.at[idx, "dicom_file_name"] = dicom_files[im_pattern]
            dicom_matched_count += 1
            matched_count += 1
    
    if (idx + 1) % 50 == 0:
        print(f"  [{idx + 1}/{len(df)}] lignes traitées...")

print(f"\n✅ Matching terminé:")
print(f"   - Lignes avec pattern IM trouvé: {len(df[df['dicom_im_pattern'].notna()])}/{len(df)}")
print(f"   - Lignes avec metadonnées XML: {xml_matched_count}/{len(df)} ({100*xml_matched_count/len(df):.1f}%)")
print(f"   - Lignes avec fichier DICOM physique: {dicom_matched_count}/{len(df)} ({100*dicom_matched_count/len(df):.1f}%)")

# Identifier les IM patterns des XML qui ne sont pas dans le dataset original
xml_im_patterns = set(xml_metadata.keys())
csv_im_patterns = set(df['dicom_im_pattern'].dropna().unique())
missing_patterns = xml_im_patterns - csv_im_patterns

print(f"\n📊 Analyse des patterns IM:")
print(f"   - Patterns trouvés dans XML: {len(xml_im_patterns)}")
print(f"   - Patterns dans le CSV original: {len(csv_im_patterns)}")
print(f"   - Patterns XML NON dans le CSV: {len(missing_patterns)}")

if missing_patterns:
    print(f"\n⚠️  {len(missing_patterns)} patterns IM trouvés dans XML mais pas dans le CSV")
    print(f"   (Ces images PNG n'existent peut-être pas, mais les rapports XML existent)")

print()

# Sauvegarder le CSV enrichi
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ CSV enrichi sauvegardé: {OUTPUT_CSV}\n")

# Statistiques finales
print("=" * 80)
print("STATISTIQUES FINALES")
print("=" * 80)

print(f"\nColonnes du CSV enrichi ({len(df.columns)} colonnes):")
print(f"  - Colonnes originales: 22 (labels pathologies)")
print(f"  - Colonnes enrichies: {len(new_columns)}")
print(f"    {', '.join(new_columns)}")

print(f"\nCouverture des données:")
print(f"  - Lignes totales: {len(df)}")
print(f"  - Avec IM-XXXX-YYYY: {len(df[df['dicom_im_pattern'].notna()])} ({100*len(df[df['dicom_im_pattern'].notna()])/len(df):.1f}%)")
print(f"  - Avec XML metadata: {xml_matched_count} ({100*xml_matched_count/len(df):.1f}%)")
print(f"  - Avec DICOM file: {dicom_matched_count} ({100*dicom_matched_count/len(df):.1f}%)")

# Afficher un exemple
print(f"\n📋 Exemple de ligne enrichie (ID 0):")
print(f"   image_id: {df.iloc[0]['image_id']}")
print(f"   dicom_im_pattern: {df.iloc[0]['dicom_im_pattern']}")
print(f"   xml_report_id: {df.iloc[0]['xml_report_id']}")
print(f"   dicom_file_name: {df.iloc[0]['dicom_file_name']}")
print(f"   xml_indication: {str(df.iloc[0]['xml_indication'])[:50]}...")

print("\n" + "=" * 80)
print("✨ ENRICHISSEMENT TERMINÉ!")
print("=" * 80)
