# utils.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(labels, preds):
    """
    labels, preds:
        - list of ints (evaluation)
        - OR numpy arrays
        - OR torch tensors (training)
    모든 입력을 안전하게 numpy array로 변환하여 처리.
    """

    # ---------------------------
    # 1) numpy 변환 처리
    # ---------------------------
    # torch.Tensor일 경우
    try:
        # label/pred가 tensor이면 .detach().cpu().numpy() 적용
        if hasattr(labels, "detach"):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.array(labels)

        if hasattr(preds, "detach"):
            preds_np = preds.detach().cpu().numpy()
        else:
            preds_np = np.array(preds)

    except Exception:
        # 그 외 list, python number 등은 바로 numpy 변환
        labels_np = np.array(labels)
        preds_np = np.array(preds)

    # int형으로 통일
    labels_np = labels_np.astype(int)
    preds_np = preds_np.astype(int)

    # ---------------------------
    # 2) 지표 계산
    # ---------------------------
    acc = accuracy_score(labels_np, preds_np)
    prec = precision_score(labels_np, preds_np, zero_division=0)
    rec = recall_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
    }
