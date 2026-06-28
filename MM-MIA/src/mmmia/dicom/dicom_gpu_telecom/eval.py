import os
import argparse
import json
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, multilabel_confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import PATHOLOGY_CLASSES, DEFAULT_BATCH_SIZE
from dataset import DICOM3DMultiLabelDataset, get_val_transforms
from model import get_model

def evaluate_model(model, loader, device, thresholds):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    print("\n🧪 Evaluating on test set...")
    with torch.no_grad():
        for volumes, labels, _ in tqdm(loader):
            volumes = volumes.to(device)
            outputs = model(volumes)

            probs = torch.sigmoid(outputs)
            
            # Apply dynamic thresholds per class
            preds = torch.zeros_like(probs)
            for i, cls in enumerate(PATHOLOGY_CLASSES):
                t = thresholds.get(cls, 0.5)
                preds[:, i] = (probs[:, i] >= t).int()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D DICOM Model v6")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing split CSVs")
    parser.add_argument("--volumes-dir", type=str, required=True, help="Directory containing .npy volumes")
    parser.add_argument("--model-path", type=str, required=True, help="Path to best_model_3d.pth")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Test Data
    test_df = pd.read_csv(os.path.join(args.data_dir, 'split_test_3d.csv'))
    test_dataset = DICOM3DMultiLabelDataset(test_df, PATHOLOGY_CLASSES, volumes_dir=args.volumes_dir, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Load Model (pretrained=False because we load our own weights)
    model = get_model(len(PATHOLOGY_CLASSES), pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    # Load Thresholds
    thresholds_path = os.path.join(os.path.dirname(args.model_path), 'optimal_thresholds.json')
    if os.path.exists(thresholds_path):
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)
        print("✅ Loaded optimal thresholds")
    else:
        print("⚠️ No optimal thresholds found, defaulting to 0.5")
        thresholds = {cls: 0.5 for cls in PATHOLOGY_CLASSES}

    all_preds, all_probs, all_labels = evaluate_model(model, test_loader, device, thresholds)

    # Per-class metrics
    print(f"\n{'='*70}")
    print("📊 PER-CLASS RESULTS")
    print(f"{'='*70}")
    print(f"{'Class':<25} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    for i, cls in enumerate(PATHOLOGY_CLASSES):
        cls_prec = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        cls_rec = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        cls_f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        t = thresholds.get(cls, 0.5)
        print(f"{cls:<25} {t:>10.2f} {cls_prec:>10.3f} {cls_rec:>10.3f} {cls_f1:>10.3f}")

    # Global metrics
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    try:
        valid_cols = [i for i in range(all_labels.shape[1]) if len(np.unique(all_labels[:, i])) > 1]
        test_auc = roc_auc_score(all_labels[:, valid_cols], all_probs[:, valid_cols], average='weighted', multi_class='ovr')
    except Exception as e:
        print(f"\n⚠️ AUC calculation failed: {e}")
        test_auc = 0.0

    print(f"\n{'='*70}")
    print("📊 TEST SET RESULTS (v6)")
    print(f"{'='*70}")
    print(f"Accuracy:  {test_acc*100:.2f}%")
    print(f"Precision: {test_precision*100:.2f}%")
    print(f"Recall:    {test_recall*100:.2f}%")
    print(f"F1-Score:  {test_f1*100:.2f}%")
    print(f"AUC-ROC:   {test_auc*100:.2f}%")

    results = {
        'model': 'r3d_18-Kinetics400-v6',
        'version': '6.0 - Pretrained Kinetics + 3-Phase Unfreezing + AUC Selection + ASL',
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_auc_roc': float(test_auc),
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': {
            'batch_size': args.batch_size,
            'gamma_neg': 4,
            'gamma_pos': 1,
            'clip': 0.05,
            'dropout': 0.3,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'unfreezing': '3-phase: head -> layer4 -> layer3'
        },
        'optimal_thresholds': thresholds
    }

    results_path = os.path.join(args.output_dir, 'results_3d_v6.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Results saved to {results_path}")

    # --- CONFUSION MATRICES ---
    print(f"\n{'='*70}")
    print("🔢 Generating confusion matrices...")
    print(f"{'='*70}")

    # 1. Per-class confusion matrices (one 2x2 matrix per disease)
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    
    # Save per-class confusion data as JSON
    confusion_data = {}
    for i, cls in enumerate(PATHOLOGY_CLASSES):
        tn, fp, fn, tp = mcm[i].ravel()
        confusion_data[cls] = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    
    with open(os.path.join(args.output_dir, 'confusion_matrices.json'), 'w') as f:
        json.dump(confusion_data, f, indent=4)
    print(f"✅ Per-class confusion data saved to confusion_matrices.json")

    # 2. Plot a big heatmap: rows = classes, columns = [TP, FP, FN, TN]
    matrix_data = []
    for cls in PATHOLOGY_CLASSES:
        d = confusion_data[cls]
        matrix_data.append([d['true_positive'], d['false_positive'], d['false_negative'], d['true_negative']])
    
    matrix_df = pd.DataFrame(
        matrix_data,
        index=PATHOLOGY_CLASSES,
        columns=['True Pos', 'False Pos', 'False Neg', 'True Neg']
    )
    
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(matrix_df, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5, ax=ax)
    ax.set_title('Confusion Matrix per Pathology (Test Set)', fontsize=14)
    ax.set_ylabel('Pathology')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_heatmap.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Confusion heatmap saved to confusion_heatmap.png")

    # 3. Save per-class metrics as CSV for easy analysis
    per_class_rows = []
    for i, cls in enumerate(PATHOLOGY_CLASSES):
        d = confusion_data[cls]
        cls_prec = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        cls_rec = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        cls_f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        per_class_rows.append({
            'class': cls,
            'threshold': thresholds.get(cls, 0.5),
            'precision': round(cls_prec, 4),
            'recall': round(cls_rec, 4),
            'f1': round(cls_f1, 4),
            'TP': d['true_positive'],
            'FP': d['false_positive'],
            'FN': d['false_negative'],
            'TN': d['true_negative'],
            'support': d['true_positive'] + d['false_negative']  # nombre réel de positifs
        })
    
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(os.path.join(args.output_dir, 'per_class_metrics.csv'), index=False)
    print(f"✅ Per-class metrics saved to per_class_metrics.csv")

if __name__ == "__main__":
    main()
