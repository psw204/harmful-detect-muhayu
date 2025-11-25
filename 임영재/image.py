#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEAM_ROOT = os.path.join(BASE_DIR, "팀원_라벨링", "팀원_데이터")
IMG_DIR = os.path.join(TEAM_ROOT, "이미지")
SAFE_IMG_DIR = os.path.join(TEAM_ROOT, "안전_이미지")
LABEL_DIR = os.path.join(TEAM_ROOT, "라벨_결과")

SCRIPT_DIR = os.path.join(BASE_DIR, "scripts")

PY_TORCH = "../../Capstone2/Im/venv_pt/bin/python"
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
        return {}

def normalize_scores(d):
    for k, v in d.items():
        if 'clip' in v:
            v['clip'] = float(f"{v['clip']:.8f}")   # 원하는 자리수
        if 'vit' in v:
            v['vit'] = float(f"{v['vit']:.8f}")
    return d

# ==========================
# 1) 폴더 기준 자동 라벨 생성
# ==========================
def auto_generate_labels():
    harm_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    safe_files = sorted([f for f in os.listdir(SAFE_IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    verified_labels = {f: 1 for f in harm_files}
    safe_labels = {f: 0 for f in safe_files}

    # 기본 저장
    json.dump(verified_labels, open(os.path.join(LABEL_DIR, "verified_labels.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(safe_labels, open(os.path.join(LABEL_DIR, "safe_labels.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"🔹 자동 라벨 생성 완료: harmful={len(verified_labels)}, safe={len(safe_labels)}")

    return verified_labels, safe_labels

# ==========================
# 2) CLIP / ViT 실행
# ==========================
def run_clip_vit():
    run([PY_TORCH, CLIP_PY, "--frames", IMG_DIR, "--out", os.path.join(LABEL_DIR, "image_clip_harm.json"), "--batch", "16", "--stride", "1"])
    run([PY_TORCH, CLIP_PY, "--frames", SAFE_IMG_DIR, "--out", os.path.join(LABEL_DIR, "image_clip_safe.json"), "--batch", "16", "--stride", "1"])

    run([PY_TORCH, VIT_PY, "--frames", IMG_DIR, "--out", os.path.join(LABEL_DIR, "image_vit_harm.json"), "--batch", "16", "--stride", "1"])
    run([PY_TORCH, VIT_PY, "--frames", SAFE_IMG_DIR, "--out", os.path.join(LABEL_DIR, "image_vit_safe.json"), "--batch", "16", "--stride", "1"])

# ==========================
# 3) 점수 병합 + 라벨 파일 갱신
# ==========================
def merge_scores_and_update_labels(verified_labels, safe_labels):

    def r(x):
        return round(float(x), 5)

    clip_harm = load_json(os.path.join(LABEL_DIR, "image_clip_harm.json")).get("per_frame", {})
    clip_safe = load_json(os.path.join(LABEL_DIR, "image_clip_safe.json")).get("per_frame", {})
    vit_harm  = load_json(os.path.join(LABEL_DIR, "image_vit_harm.json")).get("per_frame", {})
    vit_safe  = load_json(os.path.join(LABEL_DIR, "image_vit_safe.json")).get("per_frame", {})

    # -----------------------------
    #  🔥 가중치 설정
    # -----------------------------
    W_CLIP = 0.45
    W_VIT  = 0.55

    # -----------------------------
    #  🔥 임계값 설정
    # -----------------------------
    TH = 0.45     # 이 미만 → 안전

    # -----------------------------
    # 1) harmful 이미지 점수 계산
    # -----------------------------
    for f in verified_labels.keys():
        clip = r(clip_harm.get(f, {}).get("violence_prob", 0.0))
        vit  = r(vit_harm.get(f, {}).get("violence_prob", 0.0))
        fused = r(W_CLIP * clip + W_VIT * vit)

        # 최종 판정
        if fused >= TH:
            final = 1   # harmful
        else:
            final = 0   # safe

        verified_labels[f] = {
            "label": 1,
            "clip": clip,
            "vit": vit,
            "fused": fused,
            "final_label": final
        }

    # -----------------------------
    # 2) safe 이미지 점수 계산
    # -----------------------------
    for f in safe_labels.keys():
        clip = r(clip_safe.get(f, {}).get("violence_prob", 0.0))
        vit  = r(vit_safe.get(f, {}).get("violence_prob", 0.0))
        fused = r(W_CLIP * clip + W_VIT * vit)

        # 최종 판정
        if fused >= TH:
            final = 1
        else:
            final = 0

        safe_labels[f] = {
            "label": 0,
            "clip": clip,
            "vit": vit,
            "fused": fused,
            "final_label": final
        }

    # 파일 저장
    json.dump(verified_labels, open(os.path.join(LABEL_DIR, "verified_labels.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(safe_labels, open(os.path.join(LABEL_DIR, "safe_labels.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("✅ safe_labels.json / verified_labels.json 모델 점수 + fused + final_label 포함 갱신됨")

# ==========================
# MAIN
# ==========================
def main():
    verified_labels, safe_labels = auto_generate_labels()

    run_clip_vit()

    merge_scores_and_update_labels(verified_labels, safe_labels)

    print("\n🎉 모든 이미지 라벨링 + 모델 점수 저장 완료!")

if __name__ == "__main__":
    main()
