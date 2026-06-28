import pandas as pd
import os

def clean_csv():
    print("="*50)
    print("🧹 NETTOYAGE DU DATASET MULTIMODAL")
    print("="*50)
    
    # Chemins sur le cluster
    csv_path = '/home/infres/ahmed-25/artishow/data/dataset_labeled_volumes_3d_multimodal.csv'
    v_dir = '/home/infres/ahmed-25/artishow/volumes_preprocessed_3d_v2/volumes_preprocessed_3d_v2'

    # 1. Lire le CSV complet
    print("Lecture du CSV complet...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Erreur: Fichier introuvable à {csv_path}")
        return

    # 2. Vérifier chaque image
    print("Vérification de l'existence des images .npy sur le disque...")
    df['exists'] = df['volume_npy_path'].apply(
        lambda x: os.path.exists(os.path.join(v_dir, os.path.basename(x)))
    )

    # 3. Filtrer
    df_clean = df[df['exists']].copy()
    
    # 4. Sauvegarder
    df_clean.drop(columns=['exists'], inplace=True)
    df_clean.to_csv(csv_path, index=False)

    print(f"\n✅ Terminé !")
    print(f"Images ignorées (non téléchargées) : {len(df) - len(df_clean)}")
    print(f"Images valides restantes : {len(df_clean)}")

if __name__ == "__main__":
    clean_csv()
