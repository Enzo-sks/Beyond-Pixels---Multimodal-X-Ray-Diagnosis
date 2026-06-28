import os
import argparse
import json
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score

from config import (PATHOLOGY_CLASSES, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
                    DEFAULT_LR, DEFAULT_PATIENCE, DEFAULT_CHECKPOINT_INTERVAL,
                    UNFREEZE_LAYER4_EPOCH, UNFREEZE_LAYER3_EPOCH, LR_FINETUNE)
from dataset import DICOM3DMultiLabelDataset, get_train_transforms, get_val_transforms
from model import get_model, AsymmetricLoss, unfreeze_layer4, unfreeze_layer3

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for volumes, labels, _ in tqdm(loader, desc="Training"):
        volumes = volumes.to(device)
        labels = labels.to(device)

        outputs = model(volumes)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate_and_get_probs(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for volumes, labels, _ in tqdm(loader, desc="Validating"):
            volumes = volumes.to(device)
            labels = labels.to(device)

            outputs = model(volumes)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return avg_loss, np.array(all_probs), np.array(all_labels)

def find_optimal_thresholds(y_true, y_prob):
    """Find best threshold per class using F1 on validation set."""
    print("\n🔍 Calculating optimal thresholds per class...")
    thresholds = {}
    num_classes = y_true.shape[1]
    
    for i in range(num_classes):
        best_t = 0.5
        best_f1 = 0.0
        for t in np.arange(0.2, 0.8, 0.05):
            preds = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        if best_f1 == 0.0:
            best_t = 0.5
            
        thresholds[PATHOLOGY_CLASSES[i]] = round(float(best_t), 2)
        
    return thresholds

def compute_val_auc(y_true, y_prob):
    """Compute macro AUC-ROC on validation, handling single-class columns."""
    try:
        # Only compute AUC for classes that have both 0s and 1s
        valid_cols = []
        for i in range(y_true.shape[1]):
            if len(np.unique(y_true[:, i])) > 1:
                valid_cols.append(i)
        if len(valid_cols) == 0:
            return 0.0
        return roc_auc_score(y_true[:, valid_cols], y_prob[:, valid_cols], average='macro')
    except:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="3D DICOM Fine-Tuning v6")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing split CSVs")
    parser.add_argument("--volumes-dir", type=str, required=True, help="Directory containing .npy volumes")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs/checkpoints")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")

    # Load splits
    train_df = pd.read_csv(os.path.join(args.data_dir, 'split_train_3d.csv'))
    val_df = pd.read_csv(os.path.join(args.data_dir, 'split_val_3d.csv'))

    train_dataset = DICOM3DMultiLabelDataset(train_df, PATHOLOGY_CLASSES, volumes_dir=args.volumes_dir, transform=get_train_transforms())
    val_dataset = DICOM3DMultiLabelDataset(val_df, PATHOLOGY_CLASSES, volumes_dir=args.volumes_dir, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"\n--- Dataset Summary ---")
    print(f"Train samples: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Val samples:   {len(val_dataset)} ({len(val_loader)} batches)")
    print(f"Batch size:    {args.batch_size}")
    print(f"Num classes:   {len(PATHOLOGY_CLASSES)}")
    print(f"Volume shape:  224x224x64 -> model input (3, 64, 224, 224)")

    model = get_model(len(PATHOLOGY_CLASSES), pretrained=True).to(device)

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    
    # Phase 1: Only train classifier head with high LR
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UNFREEZE_LAYER4_EPOCH, eta_min=1e-5)

    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_val_auc = 0.0
    best_thresholds = {cls: 0.5 for cls in PATHOLOGY_CLASSES}
    patience_counter = 0

    print(f"\n{'='*70}")
    print("🚀 STARTING TRAINING v7 (Pretrained r3d_18 + 3-Phase Unfreezing)")
    print(f"{'='*70}")
    print(f"Phase 1 (epochs 1-{UNFREEZE_LAYER4_EPOCH}): Classifier head only, LR={args.lr}")
    print(f"Phase 2 (epochs {UNFREEZE_LAYER4_EPOCH+1}-{UNFREEZE_LAYER3_EPOCH}): + layer4, LR={LR_FINETUNE}")
    print(f"Phase 3 (epochs {UNFREEZE_LAYER3_EPOCH+1}+): + layer3, LR={LR_FINETUNE/2}")

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Progressive Unfreezing
        if epoch == UNFREEZE_LAYER4_EPOCH:
            model = unfreeze_layer4(model)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_FINETUNE,
                weight_decay=1e-4
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=UNFREEZE_LAYER3_EPOCH - UNFREEZE_LAYER4_EPOCH, eta_min=1e-6
            )
        
        if epoch == UNFREEZE_LAYER3_EPOCH:
            model = unfreeze_layer3(model)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_FINETUNE / 2,
                weight_decay=1e-4
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - UNFREEZE_LAYER3_EPOCH, eta_min=1e-7
            )
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs, val_labels = validate_and_get_probs(model, val_loader, criterion, device)
        val_auc = compute_val_auc(val_labels, val_probs)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {current_lr:.2e}")
        
        # Save best model based on AUC (not loss)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_thresholds = find_optimal_thresholds(val_labels, val_probs)
            
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model_3d.pth'))
            with open(os.path.join(args.output_dir, 'optimal_thresholds.json'), 'w') as f:
                json.dump(best_thresholds, f, indent=4)
                
            print(f"✅ New best model saved! AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE | Best Val AUC: {best_val_auc:.4f}")
    print(f"{'='*70}")

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    ax1.axvline(x=UNFREEZE_LAYER4_EPOCH, color='orange', linestyle='--', alpha=0.7, label='Unfreeze L4')
    ax1.axvline(x=UNFREEZE_LAYER3_EPOCH, color='red', linestyle='--', alpha=0.7, label='Unfreeze L3')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['val_auc'], label='Val AUC-ROC', marker='D', linewidth=2, color='green')
    ax2.axvline(x=UNFREEZE_LAYER4_EPOCH, color='orange', linestyle='--', alpha=0.7, label='Unfreeze L4')
    ax2.axvline(x=UNFREEZE_LAYER3_EPOCH, color='red', linestyle='--', alpha=0.7, label='Unfreeze L3')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC-ROC')
    ax2.set_title('Validation AUC-ROC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history_3d.png'), dpi=100, bbox_inches='tight')

    # Save training history as JSON for later analysis
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    print(f"📊 Training history saved to {os.path.join(args.output_dir, 'training_history.json')}")

if __name__ == "__main__":
    main()
