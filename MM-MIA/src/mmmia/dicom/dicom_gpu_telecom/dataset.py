import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn

from monai.transforms import (
    Compose,
    RandRotate90,
    RandZoom,
    RandGaussianNoise,
    RandFlip
)

def get_train_transforms():
    # MONAI transforms for arrays (H, W, D) or (C, H, W, D).
    # We apply them after converting the volume to (1, D, H, W).
    # MONAI spatial dims expect (C, spatial_dims...).
    # So if we feed (1, D, H, W) it means spatial dims are (D, H, W).
    return Compose([
        RandFlip(prob=0.3, spatial_axis=1), # Flip H
        RandFlip(prob=0.3, spatial_axis=2), # Flip W
        RandRotate90(prob=0.3, max_k=3, spatial_axes=(1, 2)), # Rotate in H-W plane
        RandZoom(prob=0.3, min_zoom=0.9, max_zoom=1.1, padding_mode="constant"),
        RandGaussianNoise(prob=0.2, mean=0.0, std=0.05)
    ])

def get_val_transforms():
    return None

def load_volume_npy(volume_path):
    """
    Load preprocessed volume from .npy file.
    Volumes are already windowed, resized, and normalized!
    """
    try:
        volume = np.load(volume_path)
        return volume
    except Exception as e:
        print(f"Error loading {volume_path}: {e}")
        return None

class DICOM3DMultiLabelDataset(Dataset):
    """Multi-label 3D Dataset loading preprocessed .npy volumes"""

    def __init__(self, dataframe, label_columns, volumes_dir, transform=None):
        """
        Args:
            dataframe: DataFrame with 'volume_npy_path' + label columns
            label_columns: List of 21 pathology column names
            volumes_dir: Path to directory containing .npy files
            transform: Augmentation pipeline
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_columns = label_columns
        self.volumes_dir = volumes_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_id = row['image_id']
        
        # Extract filename from CSV and build absolute path with volumes_dir
        import os
        filename = os.path.basename(row['volume_npy_path'])
        
        # Gérer la lecture combinée v1 et v2
        if 'dataset_version' in row:
            if row['dataset_version'] == 'v1':
                # Modifiez ce chemin relatif si le nom du dossier v1 est différent sur le cluster
                volume_path = os.path.join(self.volumes_dir, 'volumes_preprocessed_3d', filename)
            else:
                # Modifiez ce chemin relatif si le nom du dossier v2 est différent sur le cluster
                volume_path = os.path.join(self.volumes_dir, 'volumes_preprocessed_3d_v2/volumes_preprocessed_3d_v2', filename)
        else:
            volume_path = os.path.join(self.volumes_dir, filename)

        # Load preprocessed volume from .npy
        volume = load_volume_npy(volume_path)

        if volume is None:
            # Create a placeholder volume with the expected numpy shape (H, W, D) for v7 (224x224x64)
            volume = np.zeros((224, 224, 64), dtype=np.float32)

        # Convert to tensor. Original numpy volume is assumed to be (H, W, D)
        volume_tensor = torch.from_numpy(volume).float() # Shape: (H, W, D)

        # Permute to (D, H, W) to match (T, H, W) expectation of video models
        volume_tensor = volume_tensor.permute(2, 0, 1) # Shape: (D, H, W) where D is depth/time

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        volume_tensor = volume_tensor.unsqueeze(0) # Shape: (1, D, H, W)

        # Replicate the single channel to 3 channels for R3D_18 input: (1, D, H, W) -> (3, D, H, W)
        volume_tensor = volume_tensor.repeat(3, 1, 1, 1)

        # FIX: Assurer que tous les volumes (v1 et v2) ont exactement la même taille (64, 224, 224)
        # Si un ancien volume a une taille différente, on le redimensionne à la volée.
        import torch.nn.functional as F
        if volume_tensor.shape[1:] != (64, 224, 224):
            volume_tensor = volume_tensor.unsqueeze(0) # (1, 3, D, H, W)
            volume_tensor = F.interpolate(volume_tensor, size=(64, 224, 224), mode='trilinear', align_corners=False)
            volume_tensor = volume_tensor.squeeze(0) # (3, D, H, W)

        # Apply transforms (augmentation)
        if self.transform:
            volume_tensor = self.transform(volume_tensor)
            # Convertir le MetaTensor de MONAI en Tensor classique pour éviter les erreurs du collate_fn
            if hasattr(volume_tensor, "as_tensor"):
                volume_tensor = volume_tensor.as_tensor()

        # Get labels (21 binary values)
        labels = torch.tensor(row[self.label_columns].values.astype(np.float32))

        return volume_tensor, labels, image_id
