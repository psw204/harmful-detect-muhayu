
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO 기반 시각(vision) 분석
- frames 디렉터리의 모든 이미지에 대해 객체 탐지
- 이미지별로 클래스별 최고 confidence만 남겨서 JSON으로 저장
- 무기/피/노출 등은 커스텀 weight(model_path)로 학습해 사용 가능
"""

import os
import json
import argparse
from glob import glob

from ultralytics import YOLO
import torch
from tqdm import tqdm


def run_yolo(frames_dir,
             out_path,
             model_path="yolov8n.pt",
             conf=0.25,
             batch=32,
             device=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 디바이스 설정 (cuda 있으면 0, 없으면 cpu)
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)
    model.to(device)

    # 입력 프레임들 정렬
    imgs = sorted(
        p for p in glob(os.path.join(frames_dir, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    results = {}
    for i in tqdm(range(0, len(imgs), batch), desc="YOLO"):
        chunk = imgs[i:i+batch]
        outs = model(chunk, verbose=False, conf=conf, device=device)
        for path, res in zip(chunk, outs):
            per_class_best = {}
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls_id = int(b.cls.item())
                    score = float(b.conf.item())
                    name = res.names.get(cls_id, str(cls_id))
                    # 클래스별 최고 confidence만 유지
                    if name not in per_class_best or score > per_class_best[name]:
                        per_class_best[name] = score
            results[os.path.basename(path)] = per_class_best

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Vision saved -> {out_path} (images={len(imgs)})")
    if not results:
        print("⚠️ 결과가 비어 있습니다. 입력 프레임이 없거나 conf 임계값이 너무 높을 수 있어요.")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="프레임 이미지 디렉터리")
    ap.add_argument("--out", required=True, help="vision 결과 JSON 경로")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weight 경로(무기 커스텀 모델 가능)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--batch", type=int, default=32, help="batch size")
    ap.add_argument("--device", default=None, help="cuda:0, cpu 등 (기본값: 자동 결정)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse()
    run_yolo(args.frames, args.out, args.model, args.conf, args.batch, args.device)
