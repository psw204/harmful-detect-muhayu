#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

# ✅ transformers가 TF 안 쓰게 힌트
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ✅ numpy 2.x 호환 (typeDict → sctypeDict)
import numpy as np
if not hasattr(np, "typeDict") and hasattr(np, "sctypeDict"):
    np.typeDict = np.sctypeDict

from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import AutoImageProcessor, ViTForImageClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# -------------------------------------------------
# 경로 / 하이퍼파라미터 설정
# -------------------------------------------------
# 🔥 팀원 데이터로 변경
BASE = "/home/jovyan/kau-muhayu-multimodal-harmful-content-detect/임영재/팀원_라벨링/팀원_데이터"

# ⬇⬇⬇ 여기 두 줄만 실제 폴더 이름에 맞게 바꿔줘 ⬇⬇⬇
SAFE_DIR = os.path.join(BASE, "안전_이미지")   # 예시: 팀원_데이터/안전_이미지
HARM_DIR = os.path.join(BASE, "유해_이미지")   # 예시: 팀원_데이터/유해_이미지
# ⬆⬆⬆ 네 폴더 구조에 맞게 수정 ⬆⬆⬆

# 팀원 데이터 전부 사용하므로 max 개수 제한 제거
MAX_SAFE = None      # safe 전체 사용
MAX_HARM = None      # harm 전체 사용
VAL_RATIO = 0.1      # train/val = 0.9 / 0.1

MODEL_ID = "jaranohaal/vit-base-violence-detection"
BATCH_SIZE = 32
EPOCHS = 5
LR_HEAD = 1e-4       # classifier head lr
LR_ENC  = 1e-5       # encoder 마지막 레이어 lr (head보다 10배 작게)
WEIGHT_DECAY = 1e-2
SEED = 42

# -------------------------------------------------
# Seed 고정
# -------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------------------------------------
# Dataset
# -------------------------------------------------
class FolderViolenceDataset(Dataset):
    def __init__(self, items, processor, augment=False):
        """
        items: list of (img_path, label)
        """
        self.items = items
        self.processor = processor

        if augment:
            # 너무 과하지 않은 기본 augment
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # [3, H, W]
        return pixel_values, torch.tensor(label, dtype=torch.long)


# -------------------------------------------------
# 폴더에서 이미지 경로 수집
# -------------------------------------------------
def collect_image_paths(root, max_n=None):
    """
    root 안의 jpg/jpeg/png를 재귀적으로 모두 찾아서
    최대 max_n개까지 반환 (max_n=None이면 전부 사용)
    """
    patterns = []
    for ext in ["jpg", "jpeg", "png"]:
        patterns.append(os.path.join(root, f"**/*.{ext}"))

    paths = []
    for p in patterns:
        paths.extend(glob(p, recursive=True))

    paths = sorted(list(set(paths)))  # 중복 제거 + 정렬
    if max_n is not None and len(paths) > max_n:
        paths = random.sample(paths, max_n)

    return paths


# -------------------------------------------------
# Train / Val 셋 만들기
# -------------------------------------------------
def make_train_val_items():
    safe_paths = collect_image_paths(SAFE_DIR, max_n=MAX_SAFE)
    harm_paths = collect_image_paths(HARM_DIR, max_n=MAX_HARM)

    print(f"✅ safe 이미지: {len(safe_paths)}개 (root={SAFE_DIR})")
    print(f"✅ harm 이미지: {len(harm_paths)}개 (root={HARM_DIR})")

    safe_items = [(p, 0) for p in safe_paths]
    harm_items = [(p, 1) for p in harm_paths]

    all_items = safe_items + harm_items
    random.shuffle(all_items)

    # train/val split
    n_total = len(all_items)
    n_val = int(n_total * VAL_RATIO)
    val_items = all_items[:n_val]
    train_items = all_items[n_val:]

    print(f"📊 전체 샘플: {n_total}개 (train={len(train_items)}, val={len(val_items)})")

    # 클래스 분포 확인
    def count_class(items):
        neg = sum(1 for _, y in items if y == 0)
        pos = sum(1 for _, y in items if y == 1)
        return neg, pos

    neg_tr, pos_tr = count_class(train_items)
    neg_va, pos_va = count_class(val_items)
    print(f"  train: safe={neg_tr}, harm={pos_tr}")
    print(f"  val  : safe={neg_va}, harm={pos_va}")

    return train_items, val_items


# -------------------------------------------------
# Train + Eval 루프
# -------------------------------------------------
def train_and_eval(model, processor, train_items, val_items):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 🔹 encoder 마지막 2개 layer + classifier만 학습
    for name, param in model.named_parameters():
        param.requires_grad = False
        if name.startswith("encoder.layer.10") or name.startswith("encoder.layer.11"):
            param.requires_grad = True
        if name.startswith("classifier"):
            param.requires_grad = True

    # head / encoder 파라미터 그룹 분리
    head_params = []
    enc_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(p)
        else:
            enc_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": LR_HEAD},
            {"params": enc_params, "lr": LR_ENC},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # 클래스 가중치 (train set 기준)
    num_neg = sum(1 for _, y in train_items if y == 0)
    num_pos = sum(1 for _, y in train_items if y == 1)
    total = num_neg + num_pos

    w_neg = total / (2.0 * num_neg) if num_neg > 0 else 1.0
    w_pos = total / (2.0 * num_pos) if num_pos > 0 else 1.0
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)
    print(f"⚖️ class weights: neg={w_neg:.3f}, pos={w_pos:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_dataset = FolderViolenceDataset(train_items, processor, augment=True)
    val_dataset   = FolderViolenceDataset(val_items,   processor, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("\n🚀 Fine-tuning Start (팀원_데이터 기반)")

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        # -------------- Train --------------
        model.train()
        train_losses = []
        train_preds_all = []
        train_labels_all = []

        for imgs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=imgs)
            logits = outputs.logits

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            train_preds_all.extend(preds.cpu().numpy().tolist())
            train_labels_all.extend(labels.cpu().numpy().tolist())

        train_acc = accuracy_score(train_labels_all, train_preds_all)
        train_f1  = f1_score(train_labels_all, train_preds_all)

        # -------------- Val --------------
        model.eval()
        val_losses = []
        val_preds_all = []
        val_labels_all = []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Val   Epoch {epoch+1}"):
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(pixel_values=imgs)
                logits = outputs.logits

                loss = criterion(logits, labels)
                val_losses.append(loss.item())

                preds = torch.argmax(logits, dim=1)
                val_preds_all.extend(preds.cpu().numpy().tolist())
                val_labels_all.extend(labels.cpu().numpy().tolist())

        val_acc = accuracy_score(val_labels_all, val_preds_all)
        val_f1  = f1_score(val_labels_all, val_preds_all)

        print(
            f"\n📍 Epoch {epoch+1}/{EPOCHS} "
            f"| train_loss={np.mean(train_losses):.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} "
            f"| val_loss={np.mean(val_losses):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        # best 모델 저장 기준: val_f1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"💾 New best model (val_f1={best_val_f1:.4f})")

    # best 모델 리턴
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# -------------------------------------------------
# main
# -------------------------------------------------
if __name__ == "__main__":
    print("🔧 데이터 수집 중...")
    train_items, val_items = make_train_val_items()

    print("\n📦 ViT 모델 / processor 로드 중...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID)

    model = train_and_eval(model, processor, train_items, val_items)

    save_path = "./vit_finetuned.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Fine-tuned model saved to {save_path}")
