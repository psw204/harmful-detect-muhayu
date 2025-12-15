#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

# -------------------------------
# 경로 설정
# -------------------------------
RESULT_DIR = (
    "/home/jovyan/kau-muhayu-multimodal-harmful-content-detect/"
    "임영재/팀원_라벨링_모델선정/결과_데이터_training/팀원_데이터/라벨_결과"
)

# 🔧 여기서 어떤 파일을 평가할지 파일 이름만 바꿔주면 됨
# (예: th035 / th063 등)
#11
# VIDEO
VIDEO_VERIFIED_FILE = "verified_video_labels.json"   # harmful GT=1
VIDEO_SAFE_FILE     = "safe_video_labels.json"       # safe   GT=0

# IMAGE
IMAGE_VERIFIED_FILE = "verified_labels.json"         # harmful GT=1
IMAGE_SAFE_FILE     = "safe_labels.json"             # safe   GT=0

VERIFIED_VIDEO_PATH = os.path.join(RESULT_DIR, VIDEO_VERIFIED_FILE)
SAFE_VIDEO_PATH     = os.path.join(RESULT_DIR, VIDEO_SAFE_FILE)

VERIFIED_IMAGE_PATH = os.path.join(RESULT_DIR, IMAGE_VERIFIED_FILE)
SAFE_IMAGE_PATH     = os.path.join(RESULT_DIR, IMAGE_SAFE_FILE)


# -------------------------------
# 유틸 함수
# -------------------------------
def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ JSON LOAD ERROR: {path} ({e})")
            return {}
    print(f"⚠️ 파일 없음: {path}")
    return {}


def calc_metrics(TP, TN, FP, FN):
    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return acc, precision, recall, f1, total


def eval_membership(verified_dict, safe_dict, title=""):
    """verified = GT 1, safe = GT 0 으로 보고 membership 기반으로 metric 계산"""
    TP = TN = FP = FN = 0

    harm_total = len(verified_dict)
    safe_total = len(safe_dict)
    harm_correct = 0
    safe_correct = 0

    # ---------- harmful 그룹 (GT = 1) ----------
    for fname, info in verified_dict.items():
        pred = info.get("pred_label", info.get("final_label", 0))
        try:
            pred = int(pred)
        except:
            pred = 0

        gt = 1

        if gt == 1 and pred == 1:
            TP += 1
            harm_correct += 1
        elif gt == 1 and pred == 0:
            FN += 1

    # ---------- safe 그룹 (GT = 0) ----------
    for fname, info in safe_dict.items():
        pred = info.get("pred_label", info.get("final_label", 0))
        try:
            pred = int(pred)
        except:
            pred = 0

        gt = 0

        if gt == 0 and pred == 0:
            TN += 1
            safe_correct += 1
        elif gt == 0 and pred == 1:
            FP += 1

    acc, prec, rec, f1, total = calc_metrics(TP, TN, FP, FN)

    print(f"\n🔹 {title} Metrics (membership 기반 GT) ===============")
    print(f"TP={TP} | TN={TN} | FP={FP} | FN={FN} | Total={total}")
    print(f"🎯 Accuracy : {acc*100:.2f}%")
    print(f"🎯 Precision: {prec*100:.2f}%")
    print(f"🎯 Recall   : {rec*100:.2f}%")
    print(f"🎯 F1-score : {f1*100:.2f}%\n")

    if harm_total > 0:
        print(f"📦 Harmful(verified)  정확도: {harm_correct}/{harm_total} "
              f"= {harm_correct/harm_total*100:.2f}%")
    if safe_total > 0:
        print(f"📦 Safe(safe) 정확도: {safe_correct}/{safe_total} "
              f"= {safe_correct/safe_total*100:.2f}%")

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "harm_total": harm_total,
        "safe_total": safe_total,
        "harm_correct": harm_correct,
        "safe_correct": safe_correct,
    }


# -------------------------------
# 메인 평가 로직
# -------------------------------
def main():
    # ===== VIDEO =====
    verified_vid = load_json(VERIFIED_VIDEO_PATH)   # harmful GT = 1
    safe_vid     = load_json(SAFE_VIDEO_PATH)       # safe   GT = 0

    print("======================================")
    print("📊 팀원_데이터 VIDEO 간단 평가 (membership GT)")
    print("======================================")
    print(f"📂 verified_video_labels (harm) 개수: {len(verified_vid)}")
    print(f"📂 safe_video_labels (safe) 개수    : {len(safe_vid)}")

    video_stats = eval_membership(verified_vid, safe_vid, title="VIDEO")

    # ===== IMAGE =====
    verified_img = load_json(VERIFIED_IMAGE_PATH)   # harmful GT = 1
    safe_img     = load_json(SAFE_IMAGE_PATH)       # safe   GT = 0

    print("\n======================================")
    print("📊 팀원_데이터 IMAGE 간단 평가 (membership GT)")
    print("======================================")
    print(f"📂 verified_labels (harm) 개수: {len(verified_img)}")
    print(f"📂 safe_labels (safe) 개수    : {len(safe_img)}")

    image_stats = eval_membership(verified_img, safe_img, title="IMAGE")

    print("\n✅ 영상 + 이미지 평가 완료")


if __name__ == "__main__":
    main()
