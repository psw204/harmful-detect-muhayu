# -*- coding: utf-8 -*-
"""
multimodal.jsonl → train/val 분할 스크립트 (안지산)
---------------------------------------------------
입력:
  1) multimodal.jsonl  : 전처리 결과 매니페스트
  2) verified_video_labels.json : 유해 라벨 (safe는 없음, 나머지 자동 0 처리)
출력:
  - splits/train.jsonl
  - splits/val.jsonl

사용 예시:
python split_manifest.py \
  --manifest "무하유_유해콘텐츠_데이터/4_전처리_결과(개인)/안지산/팀원_전처리/manifests/multimodal.jsonl" \
  --verified "무하유_유해콘텐츠_데이터/3_라벨링_파일(개인)/안지산/라벨_결과/verified_video_labels.json" \
  --out-dir "splits" \
  --ratio 0.8
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

def load_verified(verified_path):
    if not os.path.exists(verified_path):
        print(f"[경고] verified 파일이 없습니다: {verified_path}")
        return set()
    with open(verified_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # verified 파일이 dict 형태일 수도 있고 list일 수도 있음
    if isinstance(data, dict):
        return set(data.keys())
    elif isinstance(data, list):
        return set(data)
    else:
        return set()

def load_manifest(manifest_path):
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items

def group_by_video(items):
    groups = defaultdict(list)
    for it in items:
        vsrc = it["video"]["src"]
        groups[vsrc].append(it)
    return groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--verified", required=True)
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    mani_items = load_manifest(args.manifest)
    verified = load_verified(args.verified)
    groups = group_by_video(mani_items)

    vids = list(groups.keys())
    random.shuffle(vids)

    cut = int(len(vids) * args.ratio)
    train_vids = set(vids[:cut])
    val_vids = set(vids[cut:])

    train_items, val_items = [], []
    for vsrc, clips in groups.items():
        label = 1 if vsrc in verified else 0
        for clip in clips:
            clip["label"] = label
        if vsrc in train_vids:
            train_items.extend(clips)
        else:
            val_items.extend(clips)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for it in train_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for it in val_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[완료] Train={len(train_items)} clips, Val={len(val_items)} clips")
    print(f"→ {train_path}")
    print(f"→ {val_path}")

if __name__ == "__main__":
    main()
