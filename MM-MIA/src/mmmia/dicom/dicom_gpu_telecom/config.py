import os

PATHOLOGY_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Pneumonia',
    'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule',
    'Hernia', 'Fracture', 'Opacity',
    'Granuloma', 'Calcinosis', 'Scoliosis', 'Atherosclerosis', 'Normal'
]

# Hyperparameters v6
DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-3           # Higher LR for classifier head warmup
DEFAULT_PATIENCE = 5        # More patience (v4/v5 had 3, too aggressive)
DEFAULT_CHECKPOINT_INTERVAL = 1

# Progressive Unfreezing Schedule
UNFREEZE_LAYER4_EPOCH = 5   # Phase 2: unfreeze layer4
UNFREEZE_LAYER3_EPOCH = 10  # Phase 3: unfreeze layer3
LR_FINETUNE = 1e-5          # Lower LR for pretrained backbone layers