# -*- coding: utf-8 -*-
"""
final_model2 데이터셋 클래스 분포 확인 스크립트
------------------------------------------------
train.jsonl / val.jsonl 안에 harmful=1이 얼마나 있는지 계산해준다.

사용법:
    python count_class_dist.py
"""

import json
import os

def count_harmful(path):
    harmful = 0
    safe = 0
    total = 0

    if not os.path.exists(path):
        print(f"[ERROR] 파일이 존재하지 않음: {path}")
        return 0, 0, 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            h = item.get("harmful", 0)
            if h == 1:
                harmful += 1
            else:
                safe += 1
            total += 1

    return harmful, safe, total


if __name__ == "__main__":
    # final_model2 기준 상대경로 사용
    train_path = "../팀원_전처리/splits/train.jsonl"
    val_path   = "../팀원_전처리/splits/val.jsonl"

    print("🔍 클래스 분포 계산 중...\n")

    h_train, s_train, t_train = count_harmful(train_path)
    h_val, s_val, t_val = count_harmful(val_path)

    print("=== 📘 Train.jsonl ===")
    print(f"총 clip 수     : {t_train}")
    print(f"harmful(1) 개수: {h_train}")
    print(f"safe(0) 개수   : {s_train}")
    print(f"비율(harmful %) : {h_train / (t_train+1e-9) * 100:.2f}%")

    print("\n=== 📗 Val.jsonl ===")
    print(f"총 clip 수     : {t_val}")
    print(f"harmful(1) 개수: {h_val}")
    print(f"safe(0) 개수   : {s_val}")
    print(f"비율(harmful %) : {h_val / (t_val+1e-9) * 100:.2f}%")

    print("\n✨ 완료! harmful 비율이 낮다면 imbalance 문제로 F1이 0이 나오는 게 정상입니다.")
