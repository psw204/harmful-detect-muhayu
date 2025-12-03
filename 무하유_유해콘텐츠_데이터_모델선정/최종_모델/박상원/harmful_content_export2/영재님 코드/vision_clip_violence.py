#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from glob import glob

os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# -----------------------------
# 1. 프롬프트: harmful / benign 짝 맞추기
# -----------------------------
HARMFUL_PROMPTS = [
    "a violent scene with people fighting, punching or kicking",
    "a person shooting a gun at another person",
    "visible blood, gore or serious injury",
    "a person holding a weapon in a threatening or aggressive way",
    "a brutal fight scene from an action movie",
    "an explicit violent scene that should not be shown to children",
    
]

BENIGN_PROMPTS = [
    "people calmly talking with no fighting or violence",
    "a person holding a harmless everyday object, no threat",
    "no blood or injury, just normal healthy people",
    "a person holding tools or everyday items in a safe way",
    "a normal peaceful scene with people standing or walking",
    "a safe and non-violent scene that is appropriate for all ages",
]


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
    ap.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="softmax temperature (값이 클수록 분포가 평평해짐, 기본 2.0)",
    )
    # 0.85 기준/샤프니스도 CLI에서 바꿀 수 있게 해도 되지만,
    # 지금은 코드 안에 상수로 두고 튜닝하는 방향으로 둠.
    return ap.parse_args()


def load_model(device):
    print("🔍 Loading CLIP model (openai/clip-vit-base-patch32)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()
    return model, processor


# -----------------------------
# 2. 0.85 기준 리스케일 함수
# -----------------------------
# def rescale_clip(p: float, center: float = 0.85, sharpness: float = 25.0) -> float:
#     """
#     p: 0~1 사이의 원본 CLIP harmful 합 점수
#     center: 기준점 (예: 0.85 -> 여기서 0.5가 나오도록)
#     sharpness: 기울기. 클수록 center 주변에서 급격히 0/1로 나뉨
#     """
#     x = p - center        # center보다 크면 양수, 작으면 음수
#     y = 1.0 / (1.0 + np.exp(-sharpness * x))  # 시그모이드
#     return float(np.clip(y, 0.0, 1.0))


# -----------------------------
# 3. CLIP 점수 계산 로직
# -----------------------------
@torch.no_grad()
def compute_clip_scores(
    model,
    processor,
    image_paths,
    device,
    batch_size: int = 16,
    temperature: float = 2.0,
):
    """
    image_paths: 리스트[str]
    반환: dict {filename: violence_prob(float)}

    - harmful + benign 프롬프트를 함께 넣고 softmax
    - harmful 프롬프트들 확률을 합산해서 harm_prob 계산
    - harm_prob를 0.85 기준으로 0~1 스케일로 다시 매핑
    """
    texts = HARMFUL_PROMPTS + BENIGN_PROMPTS
    num_harm = len(HARMFUL_PROMPTS)

    per_frame = {}

    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i: i + batch_size]
        images = []
        valid_paths = []
        for p in chunk:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception:
                # 깨진 이미지 등은 스킵
                continue

        if not images:
            continue

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        out = model(**inputs)
        logits = out.logits_per_image  # (B, T)

        # temperature scaling으로 분포를 덜 뾰족하게
        logits = logits / temperature
        probs = logits.softmax(dim=-1).cpu().numpy()  # softmax over text

        for path, prob_vec in zip(valid_paths, probs):
            # 1) harmful 프롬프트 쪽 확률 합산
            harm_prob = float(np.sum(prob_vec[:num_harm]))
            harm_prob = float(np.clip(harm_prob, 0.0, 1.0))

            # 2) 0.85 기준으로 0~1 다시 매핑
            #harm_score = rescale_clip(harm_prob, center=0.85, sharpness=25.0)

            fname = os.path.basename(path)
            per_frame[fname] = {"violence_prob": harm_prob}

    return per_frame


# -----------------------------
# 4. 메인
# -----------------------------
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
        print("⚠️ No frames found for CLIP.")
        result = {
            "model": "openai/clip-vit-base-patch32",
            "frames_dir": frames_dir,
            "num_frames_total": 0,
            "num_frames_used": 0,
            "prompts": {
                "harmful": HARMFUL_PROMPTS,
                "benign": BENIGN_PROMPTS,
            },
            "per_frame": {},
            "overall": {
                "avg_violence_prob": 0.0,
                "max_violence_prob": 0.0,
                "p95_violence_prob": 0.0,
            },
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)
        print(f"✅ CLIP saved -> {out_path}")
        exit(0)

    # stride 적용해서 일부 프레임만 사용
    imgs_used = imgs[:: max(1, args.stride)]
    num_used = len(imgs_used)
    print(f"🖼  CLIP: {num_total} frames 중 {num_used}개 사용 (stride={args.stride})")

    model, processor = load_model(device)
    per_frame = compute_clip_scores(
        model,
        processor,
        imgs_used,
        device,
        batch_size=args.batch,
        temperature=args.temperature,
    )

    if per_frame:
        vals = [v["violence_prob"] for v in per_frame.values()]
        avg_v = float(np.mean(vals))
        max_v = float(np.max(vals))
        p95_v = float(np.percentile(vals, 95))
    else:
        avg_v = max_v = p95_v = 0.0

    result = {
        "model": "openai/clip-vit-base-patch32",
        "frames_dir": frames_dir,
        "num_frames_total": num_total,
        "num_frames_used": num_used,
        "prompts": {
            "harmful": HARMFUL_PROMPTS,
            "benign": BENIGN_PROMPTS,
        },
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
        f"✅ CLIP saved -> {out_path} | "
        f"avg={avg_v:.3f}, max={max_v:.3f}, p95={p95_v:.3f}"
    )
