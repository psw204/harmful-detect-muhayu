# -*- coding: utf-8 -*-
"""
final_model2 threshold 최적화 스크립트
-------------------------------------
validation 셋에서 0.01~0.99 사이 threshold를 sweep 하여
최대 F1-score를 주는 threshold를 자동으로 찾는다.
"""

import torch
import json
import yaml
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from transformers import AutoTokenizer, AutoModel

from dataset_avt import AVTDataset
from model_avt import AVTModel

def extract_cls(txt_model, ids, mask, device):
    with torch.no_grad():
        output = txt_model(input_ids=ids.to(device),
                           attention_mask=mask.to(device))
        return output.last_hidden_state[:, 0, :]  # (B,768)


def evaluate_thresholds(model, loader, txt_model, device):

    all_labels = []
    all_probs = []

    for vid, wav, ids, mask, y in tqdm(loader, desc="Extracting predictions"):
        vid = vid.to(device)
        wav = wav.to(device)
        y = y.to(device)

        cls = extract_cls(txt_model, ids, mask, device)
        with torch.no_grad():
            logits, _, _ = model(vid, wav, cls)
            probs = torch.sigmoid(logits)

        all_labels.extend(y.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    import numpy as np
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    best_th = 0.5
    best_f1 = 0.0

    print("\n🔍 Threshold Sweeping...")
    for th in [i/100 for i in range(1, 100)]:
        preds = (all_probs >= th).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return best_th, best_f1


if __name__ == "__main__":
    # load config
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer + text encoder
    tokenizer = AutoTokenizer.from_pretrained(cfg["text"]["model_name"])
    txt_model = AutoModel.from_pretrained(cfg["text"]["model_name"]).to(device)
    txt_model.eval()
    for p in txt_model.parameters():
        p.requires_grad = False

    # dataset
    val_ds = AVTDataset(cfg["val_manifest"], cfg, tokenizer)
    val_ld = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=4
    )

    # load model
    ckpt = torch.load("outputs/checkpoints_v2/best_f1_final_model2.pt")
    model = AVTModel().to(device)
    model.load_state_dict(ckpt)
    model.eval()

    # run search
    best_th, best_f1 = evaluate_thresholds(model, val_ld, txt_model, device)

    print("\n==============================")
    print(f"✨ Best Threshold = {best_th:.3f}")
    print(f"✨ Best F1        = {best_f1:.4f}")
    print("==============================")
