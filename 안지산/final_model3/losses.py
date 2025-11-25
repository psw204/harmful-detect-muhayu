# final_model3/losses.py
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for logits.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,)
        targets: (B,) float (0 or 1)
        """
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)  # p_t

        # alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = -alpha_t * (1 - pt) ** self.gamma * (
            targets * torch.log(prob + 1e-8) + (1 - targets) * torch.log(1 - prob + 1e-8)
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    lcfg = cfg["loss"]
    loss_type = lcfg.get("type", "bce").lower()
    if loss_type == "focal":
        return FocalLoss(alpha=lcfg.get("alpha", 0.25), gamma=lcfg.get("gamma", 2.0))
    else:
        # default BCE with logits
        return nn.BCEWithLogitsLoss()
