import pandas as pd
import os

def fix_labels():
    print("Chargement des CSVs...")
    new_csv_path = r"C:\Users\ahmed\Desktop\artishow\artihow _final\MM-MIA\src\mmmia\dicom\labels\dataset_labeled_volumes_3d_multimodal.csv"
    old_csv_path = r"C:\Users\ahmed\Desktop\artishow\artihow _final\MM-MIA\src\mmmia\dicom\labels\dataset_labeled_volumes_3d_vf.csv"
    output_path = r"C:\Users\ahmed\Desktop\artishow\artihow _final\MM-MIA\src\mmmia\dicom\labels\dataset_labeled_volumes_3d_multimodal_fixed.csv"

    df_new = pd.read_csv(new_csv_path)
    df_old = pd.read_csv(old_csv_path)

    print(f"Lignes dans le nouveau CSV (multimodal) : {len(df_new)}")
    print(f"Lignes dans l'ancien CSV (vf) : {len(df_old)}")

    # Liste des colonnes de labels à transférer de l'ancien vers le nouveau
    label_cols = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule',
        'Hernia', 'Fracture', 'Pleural_Thickening', 'Opacity', 'Consolidation',
        'Granuloma', 'Calcinosis', 'Scoliosis', 'Atherosclerosis', 'Normal'
    ]

    # Créer un dictionnaire de mapping pour chaque label depuis l'ancien CSV
    # On utilise l'image_id comme clé primaire
    df_old_indexed = df_old.set_index('image_id')[label_cols]

    # Mettre à jour les colonnes de labels du nouveau dataframe
    # map_col met à jour uniquement les lignes qui correspondent
    for col in label_cols:
        # Extraire la série du vieux dataset
        mapping_series = df_old_indexed[col]
        # Mettre à jour le nouveau
        df_new[col] = df_new['image_id'].map(mapping_series).fillna(df_new[col])

    # Convertir en entiers (car fillna peut créer des floats)
    for col in label_cols:
        df_new[col] = df_new[col].astype(int)

    df_new.to_csv(output_path, index=False)
    print(f"✅ Mapping terminé ! Fichier sauvegardé sous : {output_path}")

if __name__ == "__main__":
    fix_labels()
