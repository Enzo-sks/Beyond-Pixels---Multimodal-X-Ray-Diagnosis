"""losses.py — Pertes partagées par les modèles multimodaux."""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss — Ben-Baruch et al. (2021), arXiv:2009.14119.

    Le projet utilise l'ASL à la place de la BCE pour gérer le déséquilibre
    multi-label sévère d'Open-i.
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, logits, targets):
        p     = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0)

        log_p   = torch.log(p.clamp(min=self.eps))
        log_1_p = torch.log((1 - p_neg).clamp(min=self.eps))

        loss_pos = (1 - p)  ** self.gamma_pos * log_p
        loss_neg =  p_neg   ** self.gamma_neg  * log_1_p

        return -(targets * loss_pos + (1 - targets) * loss_neg).mean()
