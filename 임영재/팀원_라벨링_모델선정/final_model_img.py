#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import subprocess
import shutil

# --------------------------------
# 경로 설정
# --------------------------------
BASE_DIR = "/home/jovyan/kau-muhayu-multimodal-harmful-content-detect"

SRC_ROOT = os.path.join(BASE_DIR, "무하유_유해콘텐츠_데이터_모델선정", "2_실제_수집_데이터")
OUT_ROOT = os.path.join(BASE_DIR, "임영재", "팀원_라벨링_모델선정", "결과_데이터_32")
CATEG_ROOT = os.path.join(BASE_DIR, "임영재", "팀원_라벨링_모델선정", "팀원_라벨링")

SCRIPT_DIR = os.path.join(BASE_DIR, "임영재", "scripts")

PY_TORCH = "/home/jovyan/Capstone2/Im/venv_pt/bin/python"
CLIP_PY = os.path.join(SCRIPT_DIR, "vision_clip_violence.py")
VIT_PY  = os.path.join(SCRIPT_DIR, "vision_vit.py")


def run(cmd):
    print("▶", " ".join(str(x) for x in cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError("Command failed: " + " ".join(cmd))


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except:
        print("⚠️ JSON LOAD ERROR:", path)
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
    (CLIP/VIT 스크립트는 재귀탐색 안한다고 가정)
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for i, src in enumerate(files):
        base = os.path.basename(src)
        # 혹시 이름 충돌나면 index 붙이기
        dst = os.path.join(out_dir, base)
        if os.path.exists(dst):
            name, ext = os.path.splitext(base)
            dst = os.path.join(out_dir, f"{name}_{i}{ext}")
        shutil.copy2(src, dst)

    return out_dir


# --------------------------------
# per_frame score find utility
# --------------------------------
def find_score(per_frame_dict, fname):
    """정확히 일치하지 않아도 파일명이 포함되어 있으면 매칭"""
    if fname in per_frame_dict:
        return per_frame_dict[fname]

    for k, v in per_frame_dict.items():
        if fname in k:
            return v    # 제일 처음 등장하는 key 사용

    return {}  # 점수 없음 → default 0.0 처리

def find_category(cat_dict, fname):
    """
    categorized JSON에서 fname에 해당하는 category 정보 찾기
    - key가 정확히 일치하면 그대로 사용
    - 아니면 key 안에 fname이 포함된 첫 번째 것을 사용
    """
    if fname in cat_dict:
        return cat_dict[fname]

    for k, v in cat_dict.items():
        if fname in k:
            return v
    return {}  # 못 찾으면 빈 dict

# --------------------------------
# 점수 병합 함수
# --------------------------------
def merge_scores_and_update_labels(verified_labels, safe_labels, label_dir, categories):
    def r(x):
        return round(float(x), 5)

    clip_harm = load_json(os.path.join(label_dir, "image_clip_harm.json")).get("per_frame", {})
    clip_safe = load_json(os.path.join(label_dir, "image_clip_safe.json")).get("per_frame", {})
    vit_harm  = load_json(os.path.join(label_dir, "image_vit_harm.json")).get("per_frame", {})
    vit_safe  = load_json(os.path.join(label_dir, "image_vit_safe.json")).get("per_frame", {})

    # 가중치 / 임계값
    W_CLIP, W_VIT = 0.8, 0.2
    TH = 0.35

    # harmful
    for f in verified_labels.keys():
        clip_info = find_score(clip_harm, f)
        vit_info  = find_score(vit_harm, f)

        clip = r(clip_info.get("violence_prob", 0.0))
        vit  = r(vit_info.get("violence_prob", 0.0))
        fused = r(W_CLIP * clip + W_VIT * vit)

        # 👇 사람이 만든 categorized JSON에서 category 가져오기
        cat_info = find_category(categories, f)
        category = cat_info.get("category", "unknown")

        verified_labels[f] = {
            "label": 1,              # harmful
            "category": category,    # ← 여기!
            "clip": clip,
            "vit": vit,
            "fused": fused,
            "final_label": 1 if fused >= TH else 0,
        }

    # safe
    for f in safe_labels.keys():
        clip_info = find_score(clip_safe, f)
        vit_info  = find_score(vit_safe, f)

        clip = r(clip_info.get("violence_prob", 0.0))
        vit  = r(vit_info.get("violence_prob", 0.0))
        fused = r(W_CLIP * clip + W_VIT * vit)

        cat_info = find_category(categories, f)
        category = cat_info.get("category", "unknown")

        safe_labels[f] = {
            "label": 0,              # safe
            "category": category,    # ← 여기!
            "clip": clip,
            "vit": vit,
            "fused": fused,
            "pred_label": 1 if fused >= TH else 0,
        }

    # Save
    json.dump(
        verified_labels,
        open(os.path.join(label_dir, "verified_labels.json"), "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(
        safe_labels,
        open(os.path.join(label_dir, "safe_labels.json"), "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    print(f"✅ 라벨 파일 갱신 완료 → {label_dir}")


# --------------------------------
# 사람별 처리
# --------------------------------
def process_person(person_name):
    print("\n==============================")
    print(f"🔍 테스트 시작: {person_name}")
    print("==============================")

    person_src = os.path.join(SRC_ROOT, person_name)

    # 👇 사람별 category 라벨 파일 경로
    categ_path = os.path.join(
        CATEG_ROOT,
        f"{person_name}_labels_categorized.json"
    )
    categories = load_json(categ_path)
    if not categories:
        print(f"⚠️ 카테고리 JSON 없음 또는 빈 파일: {categ_path}")
    else:
        print(f"📄 카테고리 JSON 로드 완료: {categ_path} (keys={len(categories)})")

    # harmful image path guess
    harm_candidates = ["이미지", "image", "Image"]
    harm_dir = None
    for c in harm_candidates:
        p = os.path.join(person_src, c)
        if os.path.exists(p):
            harm_dir = p
            break
    if harm_dir is None:
        raise FileNotFoundError(f"❌ harmful 이미지 폴더 없음: {harm_candidates}")

    harm_files = get_all_images(harm_dir)

    # safe
    safe_candidates = ["안전이미지", "안전_이미지", "safe_image", "safe", "Safe"]
    safe_dir = None
    for c in safe_candidates:
        p = os.path.join(person_src, c)
        if os.path.exists(p):
            safe_dir = p
            break
    if safe_dir is None:
        raise FileNotFoundError(f"❌ safe 이미지 폴더 없음: {safe_candidates}")

    safe_files = get_all_images(safe_dir)

    print(f"📦 harmful 이미지 개수: {len(harm_files)}")
    print(f"📦 safe 이미지 개수   : {len(safe_files)}")

    # output path
    person_out = os.path.join(OUT_ROOT, person_name)
    label_dir = os.path.join(person_out, "라벨_결과")
    os.makedirs(label_dir, exist_ok=True)

    # filename-based keys (basename 기준)
    verified_labels = {os.path.basename(f): 1 for f in harm_files}
    safe_labels     = {os.path.basename(f): 0 for f in safe_files}

    # Save initial
    json.dump(
        verified_labels,
        open(os.path.join(label_dir, "verified_labels_init.json"), "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(
        safe_labels,
        open(os.path.join(label_dir, "safe_labels_init.json"), "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    # -----------------------------
    # 안지산처럼 image/category 구조일 때 대비
    # -----------------------------
    # harm_dir 안에 서브폴더가 있으면 flat 디렉토리 하나 만들어서 CLIP/VIT 는 거기에 돌린다.
    # (safe_image 는 파일 바로 아래라 그대로 사용)
    flat_harm_dir = harm_dir
    has_subdir = any(
        os.path.isdir(os.path.join(harm_dir, d)) for d in os.listdir(harm_dir)
    )
    if has_subdir:
        flat_harm_dir = os.path.join(person_out, "flat_harm_images")
        print(f"📂 harmful 이미지 서브폴더 감지 → flat 디렉토리 생성: {flat_harm_dir}")
        prepare_flat_dir(harm_files, flat_harm_dir)
    else:
        print("📂 harmful 이미지 서브폴더 없음 → 원본 폴더 그대로 사용")

    # safe 도 혹시 모를 상황 대비해서 동일하게 처리하고 싶으면 아래 주석 풀면 됨
    # flat_safe_dir = safe_dir
    # has_safe_subdir = any(
    #     os.path.isdir(os.path.join(safe_dir, d)) for d in os.listdir(safe_dir)
    # )
    # if has_safe_subdir:
    #     flat_safe_dir = os.path.join(person_out, "flat_safe_images")
    #     print(f"📂 safe 이미지 서브폴더 감지 → flat 디렉토리 생성: {flat_safe_dir}")
    #     prepare_flat_dir(safe_files, flat_safe_dir)
    # else:
    #     print("📂 safe 이미지 서브폴더 없음 → 원본 폴더 그대로 사용")
    #
    # 현재 요구사항상 safe_image/파일 구조라서 그대로 사용
    flat_safe_dir = safe_dir

    # -----------------------------
    # Run CLIP & ViT  (기존 스크립트 수정 X)
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

    # 점수 병합 + 라벨 갱신
    merge_scores_and_update_labels(verified_labels, safe_labels, label_dir, categories)

    print(f"🎉 {person_name} 처리 완료\n")


def main():
    people = ["박상원", "안지산", "임영재"]
    for person in people:
        process_person(person)
    print("\n🎉 모든 테스트 완료")


if __name__ == "__main__":
    main()
