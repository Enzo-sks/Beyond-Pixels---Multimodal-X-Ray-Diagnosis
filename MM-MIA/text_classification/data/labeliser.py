import os
import re
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# ── Mapping complet basé sur le vocabulaire officiel NLM Open-i ─────────────
MESH_MAP = {
    "Atelectasis": [
        "pulmonary atelectasis", "atelectasis", "atelectases",
        "collapsed lung", "collapse; pulmonary", "lobar collapse"
    ],
    "Cardiomegaly": [
        "cardiomegaly", "cardiac enlargement", "enlarged heart",
        "enlargement heart", "heart size increased", "megalocardia"
    ],
    "Effusion": [
        "pleural effusion", "fluid in the chest", "pleural cavity effusion",
        "pericardial effusion", "effusion; pericardium", "pericardial fluid"
    ],
    "Pneumonia": [
        "pneumonia", "inflammation lung", "pulmonary inflammation",
        "pneumoniae", "pneumonitides", "pneumonitis"
    ],
    "Pneumothorax": [
        "pneumothorax", "free air in the chest outside the lung",
        "pleural air collection", "hemopneumothorax", "hydropneumothorax"
    ],
    "Edema": [
        "pulmonary edema", "edema lung", "wet lungs", "subpleural edema",
        "pulmonary congestion", "pulmonary vascular congestion",
        "vascular redistribution", "central vascular congestion"
    ],
    "Emphysema": [
        "emphysema", "pulmonary emphysema", "lung emphysema",
        "centrilobular emphysema", "panacinar emphysema",
        "bullous emphysema", "abnormal collection of air in tissues"
    ],
    "Fibrosis": [
        "fibrosis", "pulmonary fibrosis", "fibroses", "fibroplasia",
        "fibrous thickening", "cirrhosis of lung", "fibrosis of lung",
        "idiopathic pulmonary fibrosis"
    ],
    "Infiltration": [
        "infiltrate"
    ],
    "Mass": [
        "mass"
    ],
    "Nodule": [
        "nodule", "nodulus"
    ],
    "Hernia": [
        "hernia, diaphragmatic", "diaphragmatic hernia",
        "hernia, hiatal", "hiatal hernia", "esophageal hiatal hernia"
    ],
    "Fracture": [
        "fractures, bone", "fracture", "broken", "bone fractures"
    ],
    "Pleural_Thickening": [
        "thickening", "thickened", "cuffing"
    ],
    "Opacity": [
        "opacity", "decreased translucency", "increased density",
        "opaque", "haziness", "airspace disease",
        "air space lung disease", "alveolar lung disease"
    ],
    "Consolidation": [
        "consolidation"
    ],
    "Granuloma": [
        "granuloma", "calcified granuloma", "calcifying granuloma",
        "granulomatous calcifications", "granulomatous disease"
    ],
    "Calcinosis": [
        "calcinosis", "calcification", "calcified", "macrocalcification",
        "calcium deposits"
    ],
    "Scoliosis": [
        "scoliosis", "scoliotic", "spine curvature",
        "lateral curvature of the spine"
    ],
    "Atherosclerosis": [
        "atherosclerosis", "atherosclerotic vascular disease", "atheromatosis"
    ],
}


# ── Parsing XML → une ligne par image + une ligne par rapport ────────────────
def parse_xml_dir(xml_dir):
    image_samples = []
    report_samples = []
    for filepath in glob.glob(os.path.join(xml_dir, "*.xml")):
        tree = ET.parse(filepath)
        root = tree.getroot()

        # XML UID
        uid_node = root.find("uId")
        xml_uid = uid_node.attrib.get("id", "") if uid_node is not None else ""

        # MeSH labels (major + automatic)
        labels_major = [n.text.lower() for n in root.findall(".//MeSH/major") if n.text]
        labels_auto  = [n.text.lower() for n in root.findall(".//MeSH/automatic") if n.text]
        all_labels   = "|".join(np.unique(labels_major + labels_auto))

        # Abstract sections
        comparison = ""
        indication = ""
        findings   = ""
        impression = ""
        for abstract in root.findall(".//Abstract/AbstractText"):
            label = abstract.attrib.get("Label", "").upper()
            text  = abstract.text or ""
            if label == "COMPARISON":
                comparison = text
            elif label == "INDICATION":
                indication = text
            elif label == "FINDINGS":
                findings = text
            elif label == "IMPRESSION":
                impression = text

        # Collect image ids for this report
        image_ids = []
        for img in root.findall(".//parentImage"):
            img_id = img.attrib.get("id", "")
            if img_id:
                image_ids.append(img_id + ".png")
                image_samples.append({
                    "image_id": img_id + ".png",
                    "xml_uid": xml_uid,
                    "mesh_all": all_labels,
                })

        # One row per report
        report_samples.append({
            "xml_uid": xml_uid,
            "comparison": comparison,
            "indication": indication,
            "findings": findings,
            "impression": impression,
            "mesh_all": all_labels,
            "image_ids": "|".join(image_ids),
            "num_images": len(image_ids),
        })

    return pd.DataFrame(image_samples), pd.DataFrame(report_samples)


# ── Labellisation ────────────────────────────────────────────────────────────
def label_from_mesh(mesh_df):
    labels = {}
    for pathology, keywords in MESH_MAP.items():
        pattern = "|".join([re.escape(k) for k in keywords])
        mask = mesh_df["mesh_all"].str.contains(pattern, case=False, na=False, regex=True)
        labels[pathology] = mask.values.astype(int)

    labels_df = pd.DataFrame(labels, index=mesh_df.index)
    labels_df["Normal"] = (labels_df[list(MESH_MAP.keys())].sum(axis=1) == 0).astype(int)
    labels_df.insert(0, "image_id", mesh_df["image_id"].values)
    labels_df.insert(1, "xml_uid", mesh_df["xml_uid"].values)
    return labels_df


# ── Labellisation pour les rapports ──────────────────────────────────────────
def label_reports_from_mesh(report_df):
    labels = {}
    for pathology, keywords in MESH_MAP.items():
        pattern = "|".join([re.escape(k) for k in keywords])
        mask = report_df["mesh_all"].str.contains(pattern, case=False, na=False, regex=True)
        labels[pathology] = mask.values.astype(int)

    labels_df = pd.DataFrame(labels, index=report_df.index)
    labels_df["Normal"] = (labels_df[list(MESH_MAP.keys())].sum(axis=1) == 0).astype(int)

    report_out = report_df[["xml_uid", "comparison", "indication", "findings",
                             "impression", "image_ids", "num_images"]].copy()
    return pd.concat([report_out, labels_df], axis=1)


# ── Pipeline ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # src/mmmia/text_classification/data/labeliser.py → racine 5 niveaux au-dessus
    from pathlib import Path
    PROJ_ROOT = Path(__file__).resolve().parents[4]
    XML_DIR   = str(PROJ_ROOT / "data" / "labeling" / "NLMCXR_reports" / "ecgen-radiology")
    CSV_PATH  = str(PROJ_ROOT / "data" / "labeling" / "new_dataset.csv")
    # dataset_labeled.csv est partagé (text + image + dicom) → data/shared/
    SHARED_DIR = PROJ_ROOT / "data" / "shared"
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMAGE_PATH = str(SHARED_DIR / "dataset_labeled.csv")
    # dataset_reports.csv est text-only → data/text_classification/
    TEXT_DIR = PROJ_ROOT / "data" / "text_classification"
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_PATH = str(TEXT_DIR / "dataset_reports.csv")

    print("Parsing XMLs...")
    mesh_df, report_df = parse_xml_dir(XML_DIR)
    print(f"  {len(mesh_df)} images parsées depuis les XML")
    print(f"  {len(report_df)} rapports XML parsés")

    # ── Image-level CSV ──────────────────────────────────────────────────
    print("Labellisation (images)...")
    labels_df = label_from_mesh(mesh_df)

    print("Merge avec le dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} lignes dans le CSV")

    df_final = df.merge(labels_df, on="image_id", how="left")

    LABEL_COLS = list(MESH_MAP.keys()) + ["Normal"]
    df_final[LABEL_COLS] = df_final[LABEL_COLS].fillna(0).astype(int)
    df_final["xml_uid"] = df_final["xml_uid"].fillna("")

    print(f"\nMatched: {df_final[LABEL_COLS[0]].notna().sum()} / {len(df_final)}")
    print("\nLabel distribution (images):")
    print(df_final[LABEL_COLS].sum().sort_values(ascending=False).to_string())
    print(f"\nShape final (images): {df_final.shape}")

    df_final.to_csv(OUT_IMAGE_PATH, index=False)
    print(f"Sauvegardé : {OUT_IMAGE_PATH}")

    # ── Report-level CSV ─────────────────────────────────────────────────
    print("\nLabellisation (rapports)...")
    report_labeled = label_reports_from_mesh(report_df)

    print(f"\nLabel distribution (rapports):")
    print(report_labeled[LABEL_COLS].sum().sort_values(ascending=False).to_string())
    print(f"\nShape final (rapports): {report_labeled.shape}")

    report_labeled.to_csv(OUT_REPORT_PATH, index=False)
    print(f"Sauvegardé : {OUT_REPORT_PATH}")
