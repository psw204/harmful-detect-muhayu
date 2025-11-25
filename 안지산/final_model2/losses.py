# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def bce_logits(logits, labels):
    """
    이진 분류용 BCEWithLogitsLoss
    logits: (B,)
    labels: (B,) float (0 or 1)
    """
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def info_nce(z1, z2, temperature: float = 0.07):
    """
    CLIP-style InfoNCE: 배치 내 같은 인덱스 쌍이 양성 샘플
    z1, z2: (B,D)
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = z1 @ z2.t() / temperature  # (B,B)
    targets = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(logits, targets)
    return loss
