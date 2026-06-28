"""transforms.py — Pré-traitement image partagé par les modèles multimodaux.

Normalisation ViT (mean/std = 0.5) : valable pour tous les modèles dont
l'encodeur image est le ViT `codewithdark/vit-chest-xray`. Un futur encodeur à
normalisation différente (ImageNet, TorchXRayVision [-1024,1024]) définira ses
propres presets plutôt que de réutiliser ceux-ci.
"""

from torchvision import transforms

VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD  = [0.5, 0.5, 0.5]

TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.85, 1.15), shear=10),
    transforms.ElasticTransform(alpha=40.0, sigma=5.0),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
])

VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
])
