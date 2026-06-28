"""classifier.py — Multi-label classifier (encoder + [CLS] pooling + head)."""

import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, encoder, n_labels):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(encoder.d, n_labels))
    def forward(self, ids): return self.head(self.encoder(ids)[:, 0])
