import torch
import torch.nn as nn
from torchvision import models

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification with class imbalance.
    v6: gamma_pos=1 (less aggressive on positives), gamma_neg=4 (strong suppression of easy negatives)
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip  # Probability margin for hard negative mining

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)

        # Probability shifting for negatives (hard negative mining)
        probs_neg = probs
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        pos_loss = -targets * torch.pow(1.0 - probs, self.gamma_pos) * torch.log(probs)
        neg_loss = -(1.0 - targets) * torch.pow(probs_neg, self.gamma_neg) * torch.log(1.0 - probs_neg)

        loss = pos_loss + neg_loss
        return loss.mean()

def get_model(num_classes, pretrained=True):
    """
    v6: Back to torchvision r3d_18 WITH pretrained Kinetics-400 weights.
    Random init (MONAI) was the main cause of failure in v5.
    """
    print("\n📥 Loading torchvision r3d_18 (Kinetics-400 pretrained)...")
    
    if pretrained:
        weights = models.video.R3D_18_Weights.KINETICS400_V1
        model = models.video.r3d_18(weights=weights)
        print("✅ Loaded pretrained Kinetics-400 weights")
    else:
        model = models.video.r3d_18(weights=None)
        print("⚠️ No pretrained weights loaded")

    # Freeze ALL backbone layers initially (warmup on classifier only)
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head with reduced dropout
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # v6: reduced from 0.7 to 0.3
        nn.Linear(num_ftrs, num_classes)
    )
    # Classifier is always trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"   Trainable params: {trainable:,} | Frozen params: {frozen:,}")

    return model

def unfreeze_layer4(model):
    """Phase 2: Unfreeze only layer4 (closest to classifier)"""
    print("\n🔓 Phase 2: Unfreezing layer4...")
    for param in model.layer4.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params now: {trainable:,}")
    return model

def unfreeze_layer3(model):
    """Phase 3: Unfreeze layer3 as well"""
    print("\n🔓 Phase 3: Unfreezing layer3...")
    for param in model.layer3.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params now: {trainable:,}")
    return model
