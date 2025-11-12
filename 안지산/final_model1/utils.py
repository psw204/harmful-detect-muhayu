# -*- coding: utf-8 -*-
import os, torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

@torch.no_grad()
def evaluate(model, loader, threshold=0.5, device='cuda'):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        v, a, ids, mask, y = [b.to(device) if torch.is_tensor(b) else b for b in batch]
        # 실제 KoBERT CLS는 train.py에서 계산. 여기선 더미 없이 그대로 받도록 맞춤이 이상적이지만,
        # evaluate는 train.py와 동일 forward를 사용하므로 이 함수 내부에선 텍스트 임베딩을 만들지 않음.
        # → 평가 시에는 train.py의 forward 경로를 그대로 사용해야 하므로
        #   이 함수는 train.py에서 호출될 때 model.forward 호출만 수행하도록 설계.
        # 다만 여기서는 간단히 동일 경로를 재현하기 어렵기 때문에,
        # 평가 로직은 train.py 내부에서 호출(텍스트 CLS 생성 후)되도록 구성.
        pass  # 실제 평가는 train.py에서 수행
