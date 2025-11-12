# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def bce_logits(logits, labels):
    """이진 분류용 BCEWithLogits"""
    return F.binary_cross_entropy_with_logits(logits, labels.float())

def info_nce(z1, z2, temperature=0.07):
    """CLIP-style InfoNCE: 배치 내 정답쌍(동일 샘플의 쌍)이 양성"""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.t() / temperature  # (B,B)
    targets = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(logits, targets)
    return loss
