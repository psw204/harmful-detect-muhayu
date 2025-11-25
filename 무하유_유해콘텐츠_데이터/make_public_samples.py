#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import random
import shutil
from pathlib import Path

# -------------------------------
# 설정
# -------------------------------
PUBLIC_ROOT = "1_공개_데이터셋"
OUT_ROOT = "../임영재/팀원_라벨링"

random.seed(42)  # 재현 가능하게 고정

# 샘플 개수
N_HARM_IMGS = 50
N_SAFE_IMGS = 50
N_HARM_VIDS = 50
N_SAFE_VIDS = 50


def pick_files(patterns, n, desc=""):
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if len(files) < n:
        raise ValueError(f"{desc}: 후보 파일이 {len(files)}개뿐이라 {n}개를 뽑을 수 없음")
    print(f"{desc}: 후보 {len(files)}개 중 {n}개 샘플링")
    return random.sample(files, n)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def copy_files(files, dst_dir):
    ensure_dir(dst_dir)
    for src in files:
        fname = os.path.basename(src)
        dst = os.path.join(dst_dir, fname)
        shutil.copy2(src, dst)
    print(f"  -> {dst_dir} 에 {len(files)}개 복사 완료")


def main():
    # ---------------------------
    # 1) 이미지 샘플링
    # ---------------------------
    # 유해 이미지: HOD_Dataset
    harm_img_patterns = [
        os.path.join(PUBLIC_ROOT, "HOD_Dataset", "dataset", "all", "jpg", "*.jpg"),
    ]
    harmful_images = pick_files(harm_img_patterns, N_HARM_IMGS, "유해 이미지 ")

    # 안전 이미지: COCO val2017
    safe_img_patterns = [
        os.path.join(PUBLIC_ROOT, "COCO_Safe_Dataset", "val2017", "*.jpg"),
        os.path.join(PUBLIC_ROOT, "COCO_Safe_Dataset", "val2017", "*.jpeg"),
        os.path.join(PUBLIC_ROOT, "COCO_Safe_Dataset", "val2017", "*.png"),
    ]
    safe_images = pick_files(safe_img_patterns, N_SAFE_IMGS, "안전 이미지 ")

    # ---------------------------
    # 2) 비디오 샘플링
    # ---------------------------
    # 유해 비디오: RWF Fight + RLVS Violence
    harm_vid_patterns = [
        os.path.join(PUBLIC_ROOT, "RWF-2000", "RWF-2000", "train", "Train_Fight", "*.avi"),
        os.path.join(PUBLIC_ROOT, "RLVS", "archive", "Violence", "*.mp4"),
        os.path.join(PUBLIC_ROOT, "RLVS", "archive", "Violence", "*.avi"),
    ]
    harmful_videos = pick_files(harm_vid_patterns, N_HARM_VIDS, "유해 비디오(RWF+RLVS)")

    # 안전 비디오: RWF NonFight + RLVS NonViolence
    safe_vid_patterns = [
        os.path.join(PUBLIC_ROOT, "RWF-2000", "RWF-2000", "train", "Train_NonFight", "*.avi"),
        os.path.join(PUBLIC_ROOT, "RLVS", "archive", "NonViolence", "*.mp4"),
        os.path.join(PUBLIC_ROOT, "RLVS", "archive", "NonViolence", "*.avi"),
    ]
    safe_videos = pick_files(safe_vid_patterns, N_SAFE_VIDS, "안전 비디오(RWF+RLVS)")

    # ---------------------------
    # 3) 폴더에 복사
    # ---------------------------
    img_harm_dir = os.path.join(OUT_ROOT, "유해_이미지")
    img_safe_dir = os.path.join(OUT_ROOT, "안전_이미지")
    vid_harm_dir = os.path.join(OUT_ROOT, "유해_비디오")
    vid_safe_dir = os.path.join(OUT_ROOT, "안전_비디오")

    print("\n[복사 시작]")
    copy_files(harmful_images, img_harm_dir)
    copy_files(safe_images, img_safe_dir)
    copy_files(harmful_videos, vid_harm_dir)
    copy_files(safe_videos, vid_safe_dir)

    print("\n✅ 샘플 데이터셋 생성 완료!")
    print(f"  - 루트 폴더: {os.path.abspath(OUT_ROOT)}")


if __name__ == "__main__":
    main()
