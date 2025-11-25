#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ViT 기반 폭력(violence) 장면 분류
- 모델: jaranohaal/vit-base-violence-detection
- 입력: frames 디렉토리의 이미지들
- 출력: 프레임별 violence_prob + 전체 통계 (avg / max / p95)

JSON 예시:
{
  "model": "jaranohaal/vit-base-violence-detection",
  "frames_dir": "...",
  "num_frames_total": 150,
  "num_frames_used": 15,
  "per_frame": {
    "clip_000_frame_000.jpg": { "violence_prob": 0.013 },
    ...
  },
  "overall": {
    "avg_violence_prob": 0.12,
    "max_violence_prob": 0.84,
    "p95_violence_prob": 0.80
  }
}
"""

import os
import json
import argparse
from glob import glob

os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification


MODEL_ID = "jaranohaal/vit-base-violence-detection"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="프레임 이미지들이 있는 디렉토리")
    ap.add_argument("--out", required=True, help="출력 JSON 경로")
    ap.add_argument("--batch", type=int, default=16, help="배치 크기 (기본 16)")
    ap.add_argument(
        "--stride",
        type=int,
        default=10,
        help="프레임 샘플링 간격 (N장 중 1장만 사용, 기본 10)",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="cuda:0 또는 cpu (기본: cuda 가능하면 cuda:0, 아니면 cpu)",
    )
    return ap.parse_args()


def load_model(device: str):
    print(f"🔍 Loading ViT violence model: {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def compute_violence_scores(
    processor,
    model,
    image_paths,
    device: str,
    batch_size: int = 16,
):
    """
    image_paths: 리스트[str]
    반환: dict {filename: {"violence_prob": float}}

    - 모델의 id2label 을 보고 "violence" 라벨의 prob을 violence_prob로 사용
    """
    id2label = model.config.id2label
    # violence 라벨 인덱스 찾기 (이름에 "violence" 가 들어간 것)
    violence_indices = [
        i for i, name in id2label.items()
        if "violence" in name.lower()
    ]
    if not violence_indices:
        # 방어적으로 index=1 을 violence 로 가정
        print("⚠️ id2label 에 'violence' 포함 라벨이 없어 index 1 을 violence 로 사용합니다.")
        violence_indices = [1]
    v_idx = violence_indices[0]

    per_frame = {}

    for i in tqdm(range(0, len(image_paths), batch_size), desc="ViT Violence"):
        chunk = image_paths[i: i + batch_size]
        images = []
        valid_paths = []
        for p in chunk:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception:
                continue

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits  # [B, num_labels]
        probs = torch.softmax(logits, dim=-1)  # softmax

        for path, prob_vec in zip(valid_paths, probs):
            prob_vec = prob_vec.cpu().numpy()
            v_prob = float(prob_vec[v_idx])
            v_prob = float(np.clip(v_prob, 0.0, 1.0))
            fname = os.path.basename(path)
            per_frame[fname] = {"violence_prob": v_prob}

    return per_frame


if __name__ == "__main__":
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    frames_dir = args.frames
    out_path = args.out

    imgs = sorted(
        [
            p
            for p in glob(os.path.join(frames_dir, "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    num_total = len(imgs)
    if num_total == 0:
        print("⚠️ No frames found for ViT violence.")
        result = {
            "model": MODEL_ID,
            "frames_dir": frames_dir,
            "num_frames_total": 0,
            "num_frames_used": 0,
            "per_frame": {},
            "overall": {
                "avg_violence_prob": 0.0,
                "max_violence_prob": 0.0,
                "p95_violence_prob": 0.0,
            },
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)
        print(f"✅ ViT violence saved -> {out_path}")
        raise SystemExit(0)

    # stride 적용해서 일부 프레임만 사용
    imgs_used = imgs[:: max(1, args.stride)]
    num_used = len(imgs_used)
    print(f"🖼  ViT Violence: {num_total} frames 중 {num_used}개 사용 (stride={args.stride})")

    processor, model = load_model(device)
    per_frame = compute_violence_scores(
        processor,
        model,
        imgs_used,
        device=device,
        batch_size=args.batch,
    )

    if per_frame:
        vals = [v["violence_prob"] for v in per_frame.values()]
        avg_v = float(np.mean(vals))
        max_v = float(np.max(vals))
        p95_v = float(np.percentile(vals, 95))
    else:
        avg_v = max_v = p95_v = 0.0

    result = {
        "model": MODEL_ID,
        "frames_dir": frames_dir,
        "num_frames_total": num_total,
        "num_frames_used": num_used,
        "per_frame": per_frame,
        "overall": {
            "avg_violence_prob": avg_v,
            "max_violence_prob": max_v,
            "p95_violence_prob": p95_v,
        },
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"✅ ViT violence saved -> {out_path} | "
        f"avg={avg_v:.3f}, max={max_v:.3f}, p95={p95_v:.3f}"
    )
