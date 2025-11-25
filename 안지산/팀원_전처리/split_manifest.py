# -*- coding: utf-8 -*-
"""
멀티모달 split_manifest.py (파일명 매칭 완전 해결 버전)
-------------------------------------------------------
verified_video_labels.json 의 key = "파일명.mp4"
전처리 manifest clip_path = ".../파일명.mp4"

→ clip_id 대신 "파일명.mp4" 기준으로 매칭하도록 수정.
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_manifest(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line.strip()))
            except:
                pass
    return items


def load_verified(path):
    """
    verified_video_labels.json 은 dict 형태이며 key=파일명.mp4
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def harmful_from_verified(entry):
    """
    YOLO 기반 harmful 판단 (is_harmful 필드)
    """
    return 1 if entry.get("is_harmful", False) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--approved", required=True)
    ap.add_argument("--safe", required=True)
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # 1) 전처리된 전체 multimodal.jsonl
    mani_items = load_manifest(args.manifest)

    # 2) YOLO 기반 verified / safe 라벨 로딩
    verified = load_verified(args.approved) if os.path.exists(args.approved) else {}
    safe     = load_verified(args.safe) if os.path.exists(args.safe) else {}

    # 두 dict 합치기 (key=파일명.mp4)
    all_labels = {**verified, **safe}

    enriched = []
    missing = 0

    for it in mani_items:
        # clip_path에서 파일명 추출
        clip_path = it["video"].get("clip_path") or it["video"].get("src")
        filename = os.path.basename(clip_path)   # 예: NV_215_4.0_5.0.mp4

        if filename not in all_labels:
            missing += 1
            continue

        lbl = all_labels[filename]
        harmful = harmful_from_verified(lbl)

        it["harmful"] = harmful
        enriched.append(it)

    print(f"[라벨 병합] 총 {len(mani_items)}개 중 {missing}개 라벨 없음 → 제외")
    print(f"[라벨 병합] 최종 병합된 클립 수: {len(enriched)}")

    # 3) video.src 기준으로 그룹화하여 split
    groups = defaultdict(list)
    for it in enriched:
        vsrc = it["video"]["src"]
        groups[vsrc].append(it)

    vids = list(groups.keys())
    random.shuffle(vids)

    cut = int(len(vids) * args.ratio)
    train_vids = set(vids[:cut])
    val_vids   = set(vids[cut:])

    train_items, val_items = [], []
    for vsrc, clips in groups.items():
        if vsrc in train_vids:
            train_items.extend(clips)
        else:
            val_items.extend(clips)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir/"train.jsonl", "w", encoding="utf-8") as f:
        for it in train_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(out_dir/"val.jsonl", "w", encoding="utf-8") as f:
        for it in val_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print("\n[완료]")
    print(f"Train clips: {len(train_items)}")
    print(f"Val clips  : {len(val_items)}")


if __name__ == "__main__":
    main()
