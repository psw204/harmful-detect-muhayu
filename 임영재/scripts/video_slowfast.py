#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------

"""
SlowFast R101 기반 행동 인식
 - 32프레임 샘플 → Slow pathway + Fast pathway 처리
 - top5 행동 label/확률 출력
 - violence 관련 행동 자동 tagging

출력 JSON 예시:
{
  "model": "slowfast_r101",
  "frames_dir": "...",
  "frames_per_clip": 32,
  "clip_sec": 2.0,
  "clips": [
    {
      "index": 0,
      "start_sec": 0.0,
      "end_sec": 2.0,
      "topk": [
        { "index": 145, "label": "punching person", "prob": 0.83 },
        ...
      ],
      "top1_prob": 0.83,
      "violence_hint": 0.83
    }
  ],
  "overall": {
    "num_clips": 10,
    "avg_top1_prob": 0.52,
    "max_top1_prob": 0.97,
    "avg_violence_hint": 0.21,
    "max_violence_hint": 0.94
  }
}
"""

import os
import json
import torch
import argparse
import torchvision.transforms as T
from torchvision.io import read_image
from tqdm import tqdm
from typing import List
from pytorchvideo.models.hub import slowfast_r101

# --------------------------
# 0. Kinetics-400 라벨 파일 경로
#    같은 폴더에 'kinetics_400_labels.txt' (한 줄당 하나의 클래스 이름) 두면 됨
# --------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
KINETICS_LABELS_PATH = os.path.join(THIS_DIR, "kinetics_400_labels.txt")

# --------------------------
# 1. 폭력 클래스 키워드 정의
# --------------------------
VIOLENCE_KEYWORDS = [
    "fight", "fighting", "punch", "hit", "kick", "attack",
    "shoot", "shooting", "gun", "weapon", "stab", "strangle",
    "choke", "beat", "assault", "violence"
]


# --------------------------
# 라벨 로더
# --------------------------
def load_kinetics_labels(path: str, num_classes: int) -> list:
    """
    Kinetics-400 라벨을 텍스트 파일에서 로드.
    - 파일이 없거나 개수가 안 맞으면 'class_0' 형식으로 fallback
    """
    labels = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        labels.append(name)
        except Exception:
            labels = []

    if len(labels) != num_classes:
        print(
            f"[SlowFast] ⚠ 라벨 개수 불일치 또는 파일 없음. "
            f"({len(labels)}개) → 'class_i' 형식으로 대체"
        )
        labels = [f"class_{i}" for i in range(num_classes)]

    return labels


# --------------------------
# 2. 프레임 불러오기
# --------------------------
def load_frames(frame_paths: List[str]) -> torch.Tensor:
    """
    frame_paths 리스트에서 이미지를 읽어서 [T, C, H, W] 텐서로 반환
    """
    frames = []
    for p in frame_paths:
        img = read_image(p).float() / 255.0  # [C,H,W], 0~1
        frames.append(img)
    if not frames:
        return torch.empty(0, 3, 224, 224)
    return torch.stack(frames)  # [T,C,H,W]


# --------------------------
# 3. SlowFast 입력 변환
# --------------------------
def slowfast_transform(frames: torch.Tensor, alpha: int = 4):
    """
    frames: [T, C, H, W]

    SlowFast 요구 형태:
      - 각 pathway 입력: [B, C, T, H, W]
      - 여기서는 B=1만 다룸
      - Fast: 전체 T 프레임
      - Slow: T / alpha 프레임 샘플링 (alpha=4가 기본)
    """
    # [T,C,H,W] -> [C,T,H,W]
    frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

    # Fast pathway: 모든 프레임 사용
    fast_pathway = frames  # [C,T,H,W]

    # Slow pathway: T/alpha 만큼 샘플링
    T_len = frames.shape[1]
    num_slow = max(T_len // alpha, 1)
    idxs = torch.linspace(0, T_len - 1, num_slow).long()
    slow_pathway = frames[:, idxs, :, :]  # [C, T_slow, H, W]

    return [slow_pathway, fast_pathway]  # 둘 다 [C,T,H,W]


# --------------------------
# 4. SlowFast 모델 로드
# --------------------------
def load_slowfast_model(device: str):
    print("[SlowFast] Loading SlowFast R101...")
    model = slowfast_r101(pretrained=True)
    model = model.eval().to(device)
    return model


# --------------------------
# 5. main inference
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="Frame directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-per-clip", type=int, default=32)
    ap.add_argument("--clip-sec", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SlowFast] Using device: {device}")

    # 프레임 정렬
    all_frames = sorted(
        f for f in os.listdir(args.frames)
        if f.lower().endswith(".jpg")
    )
    num = len(all_frames)
    if num == 0:
        print("⚠️ No frames found in:", args.frames)
        json.dump(
            {
                "model": "slowfast_r101",
                "frames_dir": args.frames,
                "frames_per_clip": args.frames_per_clip,
                "clip_sec": args.clip_sec,
                "clips": [],
                "overall": {
                    "num_clips": 0,
                    "avg_top1_prob": 0.0,
                    "max_top1_prob": 0.0,
                    "avg_violence_hint": 0.0,
                    "max_violence_hint": 0.0,
                },
            },
            open(args.out, "w"),
            indent=2,
            ensure_ascii=False,
        )
        return

    F = args.frames_per_clip

    # ✅ 실제 fps가 들어오면 그걸 기준으로 클립 길이(sec) 계산
    if args.fps and args.fps > 0:
        clip_sec = F / args.fps
        print(f"[SlowFast] Using real fps={args.fps} → clip_sec={clip_sec:.4f}")
    else:
        clip_sec = args.clip_sec
        print(f"[SlowFast] Using fallback clip_sec={clip_sec}")

    slowfast = load_slowfast_model(device)

    # Kinetics-400 기준 클래스 수
    num_classes = 400
    labels = load_kinetics_labels(KINETICS_LABELS_PATH, num_classes)

    # 텐서용 Resize (CHW)
    transform = T.Resize((224, 224))

    clips_out = []
    top1_list = []
    violence_list = []

    clip_index = 0
    for i in tqdm(range(0, num, F), desc="SlowFast Clips"):
        window = all_frames[i: i + F]
        if len(window) < F:
            break  # 마지막 애매한 클립은 버림(원하면 패딩으로 처리 가능)

        # 경로 생성
        paths = [os.path.join(args.frames, f) for f in window]
        frames = load_frames(paths)  # [T,C,H,W]

        # resize
        frames = torch.stack([transform(fr) for fr in frames])  # [T,C,224,224]

        # SlowFast 입력 변환 (alpha=4)
        slow_pathway, fast_pathway = slowfast_transform(frames)  # [C,T,H,W] each

        # 배치 차원 추가 + 디바이스 이동 → [1,C,T,H,W]
        slow_pathway = slow_pathway.unsqueeze(0).to(device)
        fast_pathway = fast_pathway.unsqueeze(0).to(device)

        inp = [slow_pathway, fast_pathway]

        # Inference
        with torch.no_grad():
            out = slowfast(inp)  # logits [B, num_classes]

        prob = torch.softmax(out, dim=1)[0]  # [num_classes]

        # top5
        top5 = torch.topk(prob, 5)
        top_idx = top5.indices.cpu().tolist()
        top_prob = top5.values.cpu().tolist()

        topk_data = []
        top1_prob = float(top_prob[0])
        top1_list.append(top1_prob)

        # violence 힌트: 라벨에 violence 키워드 포함된 항목 중 최대 prob
        violence_hint = 0.0

        for idx, p in zip(top_idx, top_prob):
            label = labels[idx].lower() if idx < len(labels) else f"class_{idx}"
            p_float = float(p)

            topk_data.append({
                "index": idx,
                "label": label,
                "prob": p_float,
            })

            if any(k in label for k in VIOLENCE_KEYWORDS):
                violence_hint = max(violence_hint, p_float)

        violence_list.append(violence_hint)

        clips_out.append({
            "index": clip_index,
            "start_sec": float(clip_index * clip_sec),
            "end_sec": float((clip_index + 1) * clip_sec),
            "topk": topk_data,
            "top1_prob": top1_prob,
            "violence_hint": violence_hint,
        })

        clip_index += 1

    # overall 통계
    if clips_out:
        avg_top1 = float(sum(top1_list) / len(top1_list))
        max_top1 = float(max(top1_list))
        avg_viol = float(sum(violence_list) / len(violence_list))
        max_viol = float(max(violence_list))
    else:
        avg_top1 = max_top1 = avg_viol = max_viol = 0.0

    out_json = {
        "model": "slowfast_r101",
        "frames_dir": args.frames,
        "frames_per_clip": args.frames_per_clip,
        "clip_sec": clip_sec,
        "clips": clips_out,
        "overall": {
            "num_clips": len(clips_out),
            "avg_top1_prob": avg_top1,
            "max_top1_prob": max_top1,
            "avg_violence_hint": avg_viol,
            "max_violence_hint": max_viol,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out_json, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"✅ SlowFast saved -> {args.out}")


if __name__ == "__main__":
    main()
