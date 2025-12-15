#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import subprocess
import shutil

# --------------------------------
# 경로 설정 (영상 training 스크립트와 동일 스타일)
# --------------------------------
ROOT_BASE = "/home/jovyan/kau-muhayu-multimodal-harmful-content-detect"

# 팀원들이 만든 이미지 데이터 루트
SRC_ROOT = os.path.join(
    ROOT_BASE,
    "임영재",
    "팀원_라벨링",
    "팀원_데이터",
)

# 결과 저장 루트 (video training 과 동일)
OUT_ROOT = os.path.join(
    ROOT_BASE,
    "임영재",
    "팀원_라벨링_모델선정",
    "결과_데이터_training",
)

# 스크립트 경로
SCRIPT_DIR = os.path.join(ROOT_BASE, "임영재", "scripts")

PY_TORCH = "/home/jovyan/Capstone2/Im/venv_pt/bin/python"
CLIP_PY  = os.path.join(SCRIPT_DIR, "vision_clip_violence.py")
VIT_PY   = os.path.join(SCRIPT_DIR, "vision_vit.py")

# 🔧 이미지 fusion 설정
W_CLIP = 0.8
W_VIT  = 0.2
IMG_TH = 0.35   # 이미지용 threshold (필요하면 여기만 바꿔서 실험)


# --------------------------------
# 공용 유틸
# --------------------------------
def run(cmd):
    print("▶", " ".join(str(x) for x in cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError("Command failed: " + " ".join(str(x) for x in cmd))


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ JSON LOAD ERROR: {path} ({e})")
        return {}


def get_all_images(root):
    """root 내부 전체 폴더에서 jpg/png 탐색 (재귀)"""
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def prepare_flat_dir(files, out_dir):
    """
    파일 리스트를 받아서 out_dir 안에 flat 구조로 복사
    (vision_clip_violence.py / vision_vit.py 가 재귀 탐색 안 한다고 가정)
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for i, src in enumerate(files):
        base = os.path.basename(src)
        dst = os.path.join(out_dir, base)
        if os.path.exists(dst):
            name, ext = os.path.splitext(base)
            dst = os.path.join(out_dir, f"{name}_{i}{ext}")
        shutil.copy2(src, dst)

    return out_dir


def find_score(per_frame_dict, fname):
    """
    per_frame 딕셔너리에서 파일 이름 기준으로 점수 찾기.
    - key 가 fname 과 정확히 일치하면 그대로 사용
    - 아니면 fname 이 key 안에 '포함'되면 그 항목 사용
    """
    if fname in per_frame_dict:
        return per_frame_dict[fname]

    for k, v in per_frame_dict.items():
        if fname in k:
            return v

    return {}  # 없음 → default 0.0 처리


# --------------------------------
# 점수 병합 + 라벨 JSON 생성
# --------------------------------
def merge_scores_and_update_labels(verified_labels, safe_labels, label_dir):
    def r(x):
        return round(float(x), 5)

    clip_harm = load_json(os.path.join(label_dir, "image_clip_harm.json")).get("per_frame", {})
    clip_safe = load_json(os.path.join(label_dir, "image_clip_safe.json")).get("per_frame", {})
    vit_harm  = load_json(os.path.join(label_dir, "image_vit_harm.json")).get("per_frame", {})
    vit_safe  = load_json(os.path.join(label_dir, "image_vit_safe.json")).get("per_frame", {})

    # harmful
    for f in verified_labels.keys():
        clip_info = find_score(clip_harm, f)
        vit_info  = find_score(vit_harm, f)

        clip = r(clip_info.get("violence_prob", 0.0))
        vit  = r(vit_info.get("violence_prob", 0.0))
        fused = r(W_CLIP * clip + W_VIT * vit)

        verified_labels[f] = {
            "label": 1,
            "clip": clip,
            "vit": vit,
            "fused": fused,
            "pred_label": 1 if fused >= IMG_TH else 0,
            "is_harmful": True,
        }

    # safe
    for f in safe_labels.keys():
        clip_info = find_score(clip_safe, f)
        vit_info  = find_score(vit_safe, f)

        clip = r(clip_info.get("violence_prob", 0.0))
        vit  = r(vit_info.get("violence_prob", 0.0))
        fused = r(W_CLIP * clip + W_VIT * vit)

        safe_labels[f] = {
            "label": 0,
            "clip": clip,
            "vit": vit,
            "fused": fused,
            "pred_label": 1 if fused >= IMG_TH else 0,
            "is_safe": True,
            "category": "safe" if fused < IMG_TH else "review",
        }

    # Save
    out_verified = os.path.join(label_dir, "verified_labels.json")
    out_safe     = os.path.join(label_dir, "safe_labels.json")

    with open(out_verified, "w", encoding="utf-8") as f:
        json.dump(verified_labels, f, indent=2, ensure_ascii=False)
    with open(out_safe, "w", encoding="utf-8") as f:
        json.dump(safe_labels, f, indent=2, ensure_ascii=False)

    print(f"✅ 라벨 파일 갱신 완료 → {out_verified}")
    print(f"✅ 라벨 파일 갱신 완료 → {out_safe}")


# --------------------------------
# 팀원_데이터 이미지 전체 처리 (영상 training 스크립트처럼)
# --------------------------------
def process_team_images():
    print("\n==============================")
    print("📸 팀원_데이터 이미지 테스트 시작")
    print("==============================")

    person_src = SRC_ROOT

    # harmful 이미지 폴더 후보
    harm_candidates = ["이미지", "image", "Image"]
    harm_dir = None
    for c in harm_candidates:
        p = os.path.join(person_src, c)
        if os.path.exists(p):
            harm_dir = p
            break
    if harm_dir is None:
        raise FileNotFoundError(f"❌ harmful 이미지 폴더 없음: {harm_candidates}")

    # safe 이미지 폴더 후보
    safe_candidates = ["안전이미지", "안전_이미지", "safe_image", "safe", "Safe"]
    safe_dir = None
    for c in safe_candidates:
        p = os.path.join(person_src, c)
        if os.path.exists(p):
            safe_dir = p
            break
    if safe_dir is None:
        raise FileNotFoundError(f"❌ safe 이미지 폴더 없음: {safe_candidates}")

    harm_files = get_all_images(harm_dir)
    safe_files = get_all_images(safe_dir)

    print(f"📦 harmful 이미지 개수: {len(harm_files)}")
    print(f"📦 safe 이미지 개수   : {len(safe_files)}")

    # 출력 경로 (video training 과 맞춤)
    team_out  = os.path.join(OUT_ROOT, "팀원_데이터")
    label_dir = os.path.join(team_out, "라벨_결과")
    os.makedirs(label_dir, exist_ok=True)

    # key 는 basename 기준 (나중에 membership 평가 / 매칭에 유리)
    verified_labels = {os.path.basename(f): 1 for f in harm_files}
    safe_labels     = {os.path.basename(f): 0 for f in safe_files}

    # 초기 라벨도 보존
    with open(os.path.join(label_dir, "verified_labels_init.json"), "w", encoding="utf-8") as f:
        json.dump(verified_labels, f, indent=2, ensure_ascii=False)
    with open(os.path.join(label_dir, "safe_labels_init.json"), "w", encoding="utf-8") as f:
        json.dump(safe_labels, f, indent=2, ensure_ascii=False)

    # -----------------------------
    # harmful 쪽에 서브폴더가 있을 수 있으니 flat 디렉토리로 복사
    # -----------------------------
    flat_harm_dir = harm_dir
    has_subdir = any(
        os.path.isdir(os.path.join(harm_dir, d)) for d in os.listdir(harm_dir)
    )
    if has_subdir:
        flat_harm_dir = os.path.join(team_out, "flat_harm_images")
        print(f"📂 harmful 이미지 서브폴더 감지 → flat 디렉토리 생성: {flat_harm_dir}")
        prepare_flat_dir(harm_files, flat_harm_dir)
    else:
        print("📂 harmful 이미지 서브폴더 없음 → 원본 폴더 그대로 사용")

    # safe 는 현재 구조상 바로 아래 파일이라 가정 (필요하면 위와 같이 flat 처리 추가)
    flat_safe_dir = safe_dir

    # -----------------------------
    # CLIP & ViT 실행 (기존 스크립트 그대로 사용)
    # -----------------------------
    run([
        PY_TORCH, CLIP_PY,
        "--frames", flat_harm_dir,
        "--out", os.path.join(label_dir, "image_clip_harm.json"),
        "--batch", "16",
        "--stride", "1",
    ])

    run([
        PY_TORCH, CLIP_PY,
        "--frames", flat_safe_dir,
        "--out", os.path.join(label_dir, "image_clip_safe.json"),
        "--batch", "16",
        "--stride", "1",
    ])

    run([
        PY_TORCH, VIT_PY,
        "--frames", flat_harm_dir,
        "--out", os.path.join(label_dir, "image_vit_harm.json"),
        "--batch", "16",
        "--stride", "1",
    ])

    run([
        PY_TORCH, VIT_PY,
        "--frames", flat_safe_dir,
        "--out", os.path.join(label_dir, "image_vit_safe.json"),
        "--batch", "16",
        "--stride", "1",
    ])

    # 점수 병합 + 라벨 JSON 생성
    merge_scores_and_update_labels(verified_labels, safe_labels, label_dir)

    print("🎉 팀원_데이터 이미지 처리 완료!\n")


def main():
    process_team_images()
    print("🎉 final_model_img_training 완료")


if __name__ == "__main__":
    main()
