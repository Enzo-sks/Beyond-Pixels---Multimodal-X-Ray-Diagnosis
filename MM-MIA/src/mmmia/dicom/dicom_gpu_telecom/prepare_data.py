import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from config import PATHOLOGY_CLASSES

def prepare_splits(input_csv, output_dir):
    print(f"Loading data from {input_csv}...")
    df_labels = pd.read_csv(input_csv)
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Total volumes: {len(df_labels)}")

    # === IMPROVED STRATIFIED SPLIT FOR MULTI-LABEL (v4) ===
    # Multi-criteria stratification:
    # 1. Stratify on Normal class presence
    # 2. Stratify on number of pathologies per sample
    
    df_labels['stratify_col_normal'] = df_labels['Normal'].astype(str)
    
    df_labels['num_pathologies'] = df_labels[PATHOLOGY_CLASSES].sum(axis=1)
    df_labels['pathology_count_bin'] = pd.cut(
        df_labels['num_pathologies'],
        bins=[-1, 0, 1, 3, 30],
        labels=['0_pathologies', '1_pathology', '2-3_pathologies', '4+_pathologies']
    )
    
    df_labels['stratify_col'] = (df_labels['stratify_col_normal'].astype(str) + '_' +
                                  df_labels['pathology_count_bin'].astype(str))
    
    # First split: 70% train, 30% temp (val+test)
    train_df, temp_df = train_test_split(
        df_labels,
        test_size=0.3,
        random_state=42,
        stratify=df_labels['stratify_col']
    )
    
    # Second split: Split temp into 50/50 (val and test)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['stratify_col']
    )
    
    # Drop the temporary stratification columns
    cols_to_drop = ['stratify_col_normal', 'pathology_count_bin', 'stratify_col', 'num_pathologies']
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    val_df = val_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"{'='*70}")
    print("🔀 STRATIFIED DATASET SPLIT (70/15/15) - v4 MULTI-CRITERIA")
    print(f"{'='*70}")
    print(f"Train: {len(train_df):4d} ({len(train_df)/len(df_labels)*100:5.1f}%)")
    print(f"Val:   {len(val_df):4d} ({len(val_df)/len(df_labels)*100:5.1f}%)")
    print(f"Test:  {len(test_df):4d} ({len(test_df)/len(df_labels)*100:5.1f}%)")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)}")

    # Save splits
    train_path = os.path.join(output_dir, 'split_train_3d.csv')
    val_path = os.path.join(output_dir, 'split_val_3d.csv')
    test_path = os.path.join(output_dir, 'split_test_3d.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\n✅ Stratified splits saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data splits for 3D DICOM Fine-Tuning")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to the original CSV with all labels and volume paths")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the split CSVs")
    
    args = parser.parse_args()
    prepare_splits(args.input_csv, args.output_dir)
