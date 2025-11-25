# final_model3/utils.py
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(
    labels: torch.Tensor, probs: torch.Tensor, threshold: float = 0.5
) -> Tuple[float, float, float, float]:
    """
    labels: (N,) 0/1
    probs:  (N,) [0,1]
    """
    labels_np = labels.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    preds_np = (probs_np >= threshold).astype(int)

    acc = accuracy_score(labels_np, preds_np)
    p, r, f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="binary", zero_division=0
    )
    return acc, p, r, f1


def optimize_threshold(labels: torch.Tensor, probs: torch.Tensor) -> float:
    """
    validation에서 F1 기준으로 best threshold 찾기.
    """
    labels_np = labels.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()

    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.linspace(0.05, 0.95, 19):  # 0.05 ~ 0.95 step 0.05
        preds = (probs_np >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels_np, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr
