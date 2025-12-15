#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import sys
import json
import subprocess
from collections import Counter
from glob import glob

# 🔧 numpy 2.x 호환 패치 (typeDict → sctypeDict)
import numpy as np
if not hasattr(np, "typeDict") and hasattr(np, "sctypeDict"):
    np.typeDict = np.sctypeDict

import torch
import numpy as np  # 위에서 이미 임포트했지만, 있어도 상관 없음
from tqdm import tqdm
import torchvision.transforms as T

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

def get_all_videos(root):
    """
    root 아래 모든 하위 폴더까지 뒤져서 동영상 파일 경로를
    root 기준 상대경로(relative path) 리스트로 반환.
      예) category1/clip_001.mp4
    """
    rel_paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(VIDEO_EXTS):
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root)  # root 기준 상대 경로
                rel_paths.append(rel)
    return sorted(rel_paths)


# --------------------------------
# 기본 경로 설정
# --------------------------------
ROOT_BASE = "/home/jovyan/kau-muhayu-multimodal-harmful-content-detect"

# 사람별 원본 비디오가 있는 곳
SRC_ROOT = os.path.join(
    ROOT_BASE,
    "무하유_유해콘텐츠_데이터_모델선정",
    "2_실제_수집_데이터"
)

# 결과 / 프레임 저장 루트
OUT_ROOT = os.path.join(
    ROOT_BASE,
    "임영재",
    "팀원_라벨링_모델선정",
    "결과_데이터_32"
)

CATEG_ROOT = os.path.join(
    ROOT_BASE,
    "임영재",
    "팀원_라벨링_모델선정",
    "팀원_라벨링"
)

# 스크립트 경로
SCRIPT_DIR = os.path.join(ROOT_BASE, "임영재", "scripts")
sys.path.append(SCRIPT_DIR)

# 외부 스크립트들 import (in-process용)
import vision_clip_violence as vc
import vision_vit as vv
import video_slowfast as vsf

# --------------------------------
# 실행 환경 (venv)
# --------------------------------
PY_TORCH = "/home/jovyan/Capstone2/Im/venv_pt/bin/python"
PY_TF    = "/home/jovyan/Capstone2/Im/venv_tf/bin/python"

VIDEO_SPLIT_PY  = os.path.join(SCRIPT_DIR, "video_split.py")
AUDIO_YAMNET_PY = os.path.join(SCRIPT_DIR, "audio_yamnet.py")
TEXT_OCR_PY     = os.path.join(SCRIPT_DIR, "text_toxic.py")
FUSION_PY       = os.path.join(SCRIPT_DIR, "fusion_scores.py")  # YOLO 없는 버전

# 프레임 서브폴더 이름 (사람별 FRAMES_ROOT 아래에 생성)
H_FRAMES_SUBDIR = "비디오"
S_FRAMES_SUBDIR = "안전비디오"

# 전역(사람별로 main에서 바뀜)
HARM_VIDEO_DIR = None
SAFE_VIDEO_DIR = None
LABEL_DIR      = None
FRAMES_ROOT    = None


# =========================================
# 공용 유틸
# =========================================
def run(cmd, env=None):
    print("▶", " ".join(str(x) for x in cmd))
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise RuntimeError("Command failed: " + " ".join(str(x) for x in cmd))


def load_json(path, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 saved -> {path}")


def backup_if_needed(path):
    """실수 방지용 백업: 기존 파일이 있으면 .bak 한 번만 생성"""
    if os.path.exists(path):
        bak = path + ".bak"
        if not os.path.exists(bak):
            os.rename(path, bak)
            print(f"📦 backup created: {bak}")


def find_category(cat_dict, key):
    """
    사람이 만든 categorized JSON(cat_dict) 안에서 key에 해당하는 category 정보 찾기.
    - 1순위: key 완전 일치
    - 2순위: basename 완전 일치
    - 3순위: key / basename 이 cat_dict의 key에 부분 포함
    """
    if key in cat_dict:
        return cat_dict[key]

    base = os.path.basename(key)

    # basename 완전 일치
    for k, v in cat_dict.items():
        if base == os.path.basename(k):
            return v

    # 부분 포함
    for k, v in cat_dict.items():
        if key in k or base in k:
            return v

    return None


def extract_category_value(cat_info):
    """
    cat_info에서 실제 category 문자열 꺼내기.
    예: {"label": 1, "category": "weapon"} 형태를 기본으로 가정.
    구조가 다르면 여기서 키 이름만 맞춰주면 됨.
    """
    if not isinstance(cat_info, dict):
        return None

    if "category" in cat_info:
        return cat_info["category"]

    # 혹시 다른 키명 썼으면 여기 추가
    for k in ["Category", "cat", "카테고리"]:
        if k in cat_info:
            return cat_info[k]

    return None

def round5(x):
    return round(float(x), 5)


# =========================================
# 🔥 In-process 모델 캐시 (CLIP / ViT / SlowFast)
# =========================================
CLIP_MODEL = None
CLIP_PROCESSOR = None
VIT_MODEL = None
VIT_PROCESSOR = None
SLOWFAST_MODEL = None
SLOWFAST_LABELS = None
SLOWFAST_TRANSFORM = T.Resize((224, 224))

if torch.cuda.is_available():
    CLIP_DEVICE = "cuda:0"
    VIT_DEVICE = "cuda:0"
    SLOWFAST_DEVICE = "cuda"
else:
    CLIP_DEVICE = VIT_DEVICE = SLOWFAST_DEVICE = "cpu"


def ensure_clip_model():
    global CLIP_MODEL, CLIP_PROCESSOR
    if CLIP_MODEL is None or CLIP_PROCESSOR is None:
        CLIP_MODEL, CLIP_PROCESSOR = vc.load_model(CLIP_DEVICE)


def ensure_vit_model():
    global VIT_MODEL, VIT_PROCESSOR
    if VIT_MODEL is None or VIT_PROCESSOR is None:
        VIT_PROCESSOR, VIT_MODEL = vv.load_model(VIT_DEVICE)


def ensure_slowfast_model():
    global SLOWFAST_MODEL, SLOWFAST_LABELS
    if SLOWFAST_MODEL is None:
        SLOWFAST_MODEL = vsf.load_slowfast_model(SLOWFAST_DEVICE)
        num_classes = 400  # Kinetics-400
        SLOWFAST_LABELS = vsf.load_kinetics_labels(vsf.KINETICS_LABELS_PATH, num_classes)


# =========================================
# CLIP / ViT / SlowFast in-process 래퍼
# =========================================
def run_clip_inprocess(frames_dir: str, out_path: str, batch: int = 16, stride: int = 10):
    # 🔍 frames_dir 하위 모든 폴더까지 재귀 탐색
    pattern = os.path.join(frames_dir, "**", "*")
    imgs = sorted(
        p
        for p in glob(pattern, recursive=True)
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
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
                "harmful": vc.HARMFUL_PROMPTS,
                "benign": vc.BENIGN_PROMPTS,
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
        return

    # ✅ 균등 32프레임 샘플링
    TARGET = 32
    if num_total <= TARGET:
        imgs_used = imgs[:]  # 전부 사용
    else:
        indices = np.linspace(0, num_total - 1, TARGET, dtype=int)
        imgs_used = [imgs[i] for i in indices]

    num_used = len(imgs_used)
    print(f"🖼  CLIP: {num_total} frames 중 균등 샘플링 {num_used}개 사용 (target={TARGET})")

    ensure_clip_model()
    per_frame = vc.compute_clip_scores(
        CLIP_MODEL,
        CLIP_PROCESSOR,
        imgs_used,
        CLIP_DEVICE,
        batch_size=batch,
        temperature=2.0,
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
            "harmful": vc.HARMFUL_PROMPTS,
            "benign": vc.BENIGN_PROMPTS,
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



def run_vit_inprocess(frames_dir: str, out_path: str, batch: int = 16, stride: int = 10):
    # 🔍 frames_dir 하위 전체에서 이미지 재귀 탐색
    pattern = os.path.join(frames_dir, "**", "*")
    imgs = sorted(
        p
        for p in glob(pattern, recursive=True)
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    num_total = len(imgs)
    if num_total == 0:
        print("⚠️ No frames found for ViT violence.")
        result = {
            "model": vv.MODEL_ID,
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
        return

    # ✅ 균등 32프레임 샘플링
    TARGET = 32
    if num_total <= TARGET:
        imgs_used = imgs[:]
    else:
        indices = np.linspace(0, num_total - 1, TARGET, dtype=int)
        imgs_used = [imgs[i] for i in indices]

    num_used = len(imgs_used)
    print(f"🖼  ViT Violence: {num_total} frames 중 균등 샘플링 {num_used}개 사용 (target={TARGET})")

    ensure_vit_model()
    per_frame = vv.compute_violence_scores(
        VIT_PROCESSOR,
        VIT_MODEL,
        imgs_used,
        device=VIT_DEVICE,
        batch_size=batch,
    )

    if per_frame:
        vals = [v["violence_prob"] for v in per_frame.values()]
        avg_v = float(np.mean(vals))
        max_v = float(np.max(vals))
        p95_v = float(np.percentile(vals, 95))
    else:
        avg_v = max_v = p95_v = 0.0

    result = {
        "model": vv.MODEL_ID,
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


def run_slowfast_inprocess(frames_dir: str, out_path: str, frames_per_clip: int = 32, fps: float = 30.0):
    ensure_slowfast_model()

    # 🔍 frames_dir 하위의 모든 jpg를 재귀 탐색 (full path)
    pattern = os.path.join(frames_dir, "**", "*.jpg")
    all_frames = sorted(glob(pattern, recursive=True))
    num = len(all_frames)
    if num == 0:
        print("⚠️ No frames found in:", frames_dir)
        json.dump(
            {
                "model": "slowfast_r101",
                "frames_dir": frames_dir,
                "frames_per_clip": frames_per_clip,
                "clip_sec": float(frames_per_clip / fps) if fps > 0 else 2.0,
                "clips": [],
                "overall": {
                    "num_clips": 0,
                    "avg_top1_prob": 0.0,
                    "max_top1_prob": 0.0,
                    "avg_violence_hint": 0.0,
                    "max_violence_hint": 0.0,
                },
            },
            open(out_path, "w"),
            indent=2,
            ensure_ascii=False,
        )
        print(f"✅ SlowFast saved -> {out_path}")
        return

    F = frames_per_clip

    # 🎯 전체 영상 길이 (초) 추정
    if fps and fps > 0:
        duration = num / fps
        clip_sec = duration  # 이 32프레임이 전체 영상을 대표한다고 보고 전체 길이를 사용
        print(f"[SlowFast] fps={fps} / frames={num} → duration={duration:.4f} sec")
    else:
        duration = 0.0
        clip_sec = float(F / 30.0)  # fallback
        print(f"[SlowFast] Using fallback clip_sec={clip_sec}")

    slowfast = SLOWFAST_MODEL
    labels = SLOWFAST_LABELS
    transform = SLOWFAST_TRANSFORM

    clips_out = []
    top1_list = []
    violence_list = []

    # ✅ 균등 32프레임 샘플링
    if num <= F:
        sample_paths = all_frames[:]  # 전부 사용
    else:
        indices = np.linspace(0, num - 1, F, dtype=int)
        sample_paths = [all_frames[i] for i in indices]

    # 하나의 clip 생성
    frames = vsf.load_frames(sample_paths)  # [T,C,H,W]
    frames = torch.stack([transform(fr) for fr in frames])  # [T,C,224,224]

    slow_pathway, fast_pathway = vsf.slowfast_transform(frames)

    slow_pathway = slow_pathway.unsqueeze(0).to(SLOWFAST_DEVICE)
    fast_pathway = fast_pathway.unsqueeze(0).to(SLOWFAST_DEVICE)

    inp = [slow_pathway, fast_pathway]

    with torch.no_grad():
        out = slowfast(inp)

    prob = torch.softmax(out, dim=1)[0]

    top5 = torch.topk(prob, 5)
    top_idx = top5.indices.cpu().tolist()
    top_prob = top5.values.cpu().tolist()

    topk_data = []
    top1_prob = float(top_prob[0])
    top1_list.append(top1_prob)

    violence_hint = 0.0

    for idx, p in zip(top_idx, top_prob):
        label = labels[idx].lower() if idx < len(labels) else f"class_{idx}"
        p_float = float(p)

        topk_data.append({
            "index": idx,
            "label": label,
            "prob": p_float,
        })

        if any(k in label for k in vsf.VIOLENCE_KEYWORDS):
            violence_hint = max(violence_hint, p_float)

    violence_list.append(violence_hint)

    clips_out.append({
        "index": 0,
        "start_sec": 0.0,
        "end_sec": float(clip_sec),
        "topk": topk_data,
        "top1_prob": top1_prob,
        "violence_hint": violence_hint,
    })

    if clips_out:
        avg_top1 = float(sum(top1_list) / len(top1_list))
        max_top1 = float(max(top1_list))
        avg_viol = float(sum(violence_list) / len(violence_list))
        max_viol = float(max(violence_list))
    else:
        avg_top1 = max_top1 = avg_viol = max_viol = 0.0

    out_json = {
        "model": "slowfast_r101",
        "frames_dir": frames_dir,
        "frames_per_clip": frames_per_clip,
        "clip_sec": clip_sec,
        "clips": clips_out,
        "overall": {
            "num_clips": len(clips_out),  # 보통 1
            "avg_top1_prob": avg_top1,
            "max_top1_prob": max_top1,
            "avg_violence_hint": avg_viol,
            "max_violence_hint": max_viol,
        },
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out_json, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"✅ SlowFast saved -> {out_path}")



# =========================================
# 1) 자동 라벨 생성 (사람별 디렉토리 기준)
# =========================================
# def auto_generate_video_labels():
    # 상대경로( category/clip_001.mp4 이런 식 ) 리스트
    harm_files = get_all_videos(HARM_VIDEO_DIR)
    safe_files = get_all_videos(SAFE_VIDEO_DIR)

    # key를 상대경로로 사용 (나중에 HARM_VIDEO_DIR/Safe와 합쳐서 full path로 씀)
    verified_video_labels = {f: 1 for f in harm_files}
    safe_video_labels     = {f: 0 for f in safe_files}

    json.dump(
        verified_video_labels,
        open(os.path.join(LABEL_DIR, "verified_video_labels_init.json"), "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(
        safe_video_labels,
        open(os.path.join(LABEL_DIR, "safe_video_labels_init.json"), "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    print(f"🔹 비디오 자동 라벨 생성: harmful={len(verified_video_labels)}, safe={len(safe_video_labels)}")

    return verified_video_labels, safe_video_labels


# =========================================
# 2) 한 영상 처리
# =========================================
def process_one_video(video_path, kind: str):
    """
    kind: "harm" 또는 "safe"
    FRAMES_ROOT / (비디오 | 안전비디오) / <상대경로 기반 이름> 에 프레임 저장
    """
    # 어떤 root 기준인지 결정
    if kind == "safe":
        root = SAFE_VIDEO_DIR
        subdir = S_FRAMES_SUBDIR
    else:
        root = HARM_VIDEO_DIR
        subdir = H_FRAMES_SUBDIR

    # root 기준 상대경로: category1/clip_001.mp4 이런 형태
    rel = os.path.relpath(video_path, root)
    rel_stem, _ = os.path.splitext(rel)           # category1/clip_001
    safe_stem = rel_stem.replace(os.sep, "__")    # category1__clip_001

    video_name = os.path.basename(video_path)     # 로그용

    frames_dir = os.path.join(FRAMES_ROOT, subdir, safe_stem)
    os.makedirs(frames_dir, exist_ok=True)

    FRAMES_JSON   = os.path.join(frames_dir, "meta.json")
    CLIP_JSON     = os.path.join(frames_dir, "clip_result.json")
    VIT_JSON      = os.path.join(frames_dir, "vit_result.json")
    AUDIO_WAV     = os.path.join(frames_dir, f"{safe_stem}_audio.wav")
    AUDIO_JSON    = os.path.join(frames_dir, "audio_result.json")
    TEXT_JSON     = os.path.join(frames_dir, "text_result.json")
    SLOWFAST_JSON = os.path.join(frames_dir, "slowfast_result.json")
    FUSED_JSON    = os.path.join(frames_dir, "fusion_result.json")

    # 1) 비디오 → 프레임 split
    run([
        PY_TORCH, VIDEO_SPLIT_PY,
        "--video", video_path,
        "--out", frames_dir,
        "--clip-sec", "2",
    ])

    meta = load_json(FRAMES_JSON, {})
    meta_meta = meta.get("meta", {}) if isinstance(meta.get("meta", {}), dict) else meta

    fps = meta_meta.get("fps", 30.0)
    try:
        fps = float(fps)
    except:
        fps = 30.0

    orig_total_frames = meta_meta.get("orig_total_frames") or meta_meta.get("total_frames_saved") or meta_meta.get("frames") or 0
    try:
        orig_total_frames = int(orig_total_frames)
    except:
        orig_total_frames = 0

    if "duration" in meta_meta:
        try:
            duration = float(meta_meta["duration"])
        except:
            duration = float(orig_total_frames) / fps if fps > 0 and orig_total_frames > 0 else 0.0
    else:
        duration = float(orig_total_frames) / fps if fps > 0 and orig_total_frames > 0 else 0.0

    total_frames = orig_total_frames

    print(f"[{video_name}] fps = {fps}")

    # 2) CLIP
    run_clip_inprocess(
        frames_dir=frames_dir,
        out_path=CLIP_JSON,
        batch=16,
        stride=10,
    )

    # 3) ViT
    run_vit_inprocess(
        frames_dir=frames_dir,
        out_path=VIT_JSON,
        batch=16,
        stride=10,
    )

    # 4) Audio + YAMNet
    audio_ok = False
    if not os.path.exists(AUDIO_WAV):
        cmd = [
            "ffmpeg", "-hide_banner", "-nostdin",
            "-i", video_path, "-ac", "1", "-ar", "16000", "-vn",
            AUDIO_WAV, "-y",
        ]
        print("▶", " ".join(cmd))
        p = subprocess.run(cmd)
        if p.returncode == 0 and os.path.exists(AUDIO_WAV):
            audio_ok = True
        else:
            print(f"⚠️ 오디오 스트림이 없어서 YAMNet 스킵: {video_path}")
    else:
        audio_ok = True

    if audio_ok:
        tf_env = os.environ.copy()
        tf_env["CUDA_VISIBLE_DEVICES"] = "-1"
        tf_env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        tf_env["TF_CPP_MIN_LOG_LEVEL"] = "2"

        run([
            PY_TF, AUDIO_YAMNET_PY,
            "--audio", AUDIO_WAV,
            "--out", AUDIO_JSON,
        ], env=tf_env)
    else:
        dummy_audio = {
            "overall": {
                "violent_audio_prob": 0.0,
                "has_audio": False,
            }
        }
        with open(AUDIO_JSON, "w", encoding="utf-8") as f:
            json.dump(dummy_audio, f, indent=2, ensure_ascii=False)
        print(f"📝 dummy audio_result.json 생성 -> {AUDIO_JSON}")

    # 5) TEXT (OCR + Toxic)
    frame_imgs = [
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    frame_imgs.sort()

    text_ok = False
    if frame_imgs:
        first_frame = os.path.join(frames_dir, frame_imgs[0])
        try:
            run([
                PY_TORCH,
                TEXT_OCR_PY,
                "--image", first_frame,
                "--out", TEXT_JSON,
            ])
            if os.path.exists(TEXT_JSON):
                text_ok = True
        except RuntimeError:
            print(f"⚠️ 텍스트 분석 실패, dummy로 대체: {video_path}")
            text_ok = False
    else:
        print(f"⚠️ 프레임 이미지가 없어 텍스트 분석 스킵: {video_path}")

    if not text_ok:
        dummy_text = {
            "overall": {
                "hate_prob": 0.0,
                "sexual_text_prob": 0.0,
                "has_text": False,
            }
        }
        with open(TEXT_JSON, "w", encoding="utf-8") as f:
            json.dump(dummy_text, f, indent=2, ensure_ascii=False)
        print(f"📝 dummy text_result.json 생성 -> {TEXT_JSON}")

    # 6) SlowFast
    run_slowfast_inprocess(
        frames_dir=frames_dir,
        out_path=SLOWFAST_JSON,
        frames_per_clip=32,
        fps=fps,
    )

    # 7) Fusion
    run([
        PY_TORCH, FUSION_PY,
        "--clip", CLIP_JSON,
        "--vit", VIT_JSON,
        "--audio", AUDIO_JSON,
        "--text", TEXT_JSON,
        "--slowfast", SLOWFAST_JSON,
        "--out", FUSED_JSON,
    ])

    fused = load_json(FUSED_JSON, {})
    violence_prob = float(
        fused.get("overall", {}).get(
            "violence_prob",
            fused.get("scores", {}).get("final", 0.0),
        )
    )

    # CLIP 사용 프레임 수
    clip_meta = load_json(CLIP_JSON, {})
    sampled_frames = 0

    if "num_frames_used" in clip_meta:
        try:
            sampled_frames = int(clip_meta["num_frames_used"])
        except:
            sampled_frames = 0

    if sampled_frames == 0 and isinstance(clip_meta.get("per_frame"), dict):
        sampled_frames = len(clip_meta["per_frame"])

    if sampled_frames == 0:
        meta_clip = clip_meta.get("meta", {})
        if isinstance(meta_clip, dict) and "sampled_frames" in meta_clip:
            try:
                sampled_frames = int(meta_clip["sampled_frames"])
            except:
                sampled_frames = 0

    if sampled_frames == 0 and "frames" in clip_meta and isinstance(clip_meta["frames"], list):
        sampled_frames = len(clip_meta["frames"])

    # SlowFast 대표 행동
    slow_json = load_json(SLOWFAST_JSON, {})
    estimated_action = None
    clips = slow_json.get("clips") or []
    if clips:
        labels = []
        for c in clips:
            topk = c.get("topk") or []
            if topk:
                labels.append(topk[0].get("label"))
        if labels:
            estimated_action = Counter(labels).most_common(1)[0][0]
    if not estimated_action:
        estimated_action = "unknown"

    violence_prob = round5(violence_prob)
    print(f"[{video_name}] 🔥 violence_prob = {violence_prob}")

    video_stats = {
        "duration": duration,
        "fps": fps,
        "total_frames": total_frames,
        "sampled_frames": sampled_frames,
        "estimated_action": estimated_action,
    }

    return violence_prob, video_stats


# =========================================
# 3) 최종 라벨 갱신 (하나의 사람 기준)
# =========================================
# def merge_scores_and_update_video_labels(verified, safe):
    TH = 0.63

    # 유해 비디오
    for fname in verified.keys():
        video_path = os.path.join(HARM_VIDEO_DIR, fname)
        score, stats = process_one_video(video_path, kind="harm")
        pred = 1 if score >= TH else 0

        verified[fname] = {
            "duration": stats["duration"],
            "fps": stats["fps"],
            "total_frames": stats["total_frames"],
            "sampled_frames": stats["sampled_frames"],
            "estimated_action": stats["estimated_action"],
            "is_harmful": True,
            "label": 1,
            "pred_label": pred,
            "violence_prob": score,
        }

    # 안전 비디오
    for fname in safe.keys():
        video_path = os.path.join(SAFE_VIDEO_DIR, fname)
        score, stats = process_one_video(video_path, kind="safe")
        pred = 1 if score >= TH else 0

        category = "safe" if pred == 0 else "review"

        safe[fname] = {
            "is_safe": True,
            "label": 0,
            "pred_label": pred,
            "category": category,
            "duration": stats["duration"],
            "fps": stats["fps"],
            "total_frames": stats["total_frames"],
            "sampled_frames": stats["sampled_frames"],
            "estimated_action": stats["estimated_action"],
            "violence_prob": score,
        }

    json.dump(
        verified,
        open(os.path.join(LABEL_DIR, "verified_video_labels.json"), "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    json.dump(
        safe,
        open(os.path.join(LABEL_DIR, "safe_video_labels.json"), "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    print("✅ 비디오 라벨 저장 완료 ")


# =========================================
# 사람 한 명 처리
# =========================================
# def process_person(person_name: str):
#     global HARM_VIDEO_DIR, SAFE_VIDEO_DIR, LABEL_DIR, FRAMES_ROOT

#     print("\n==============================")
#     print(f"🎬 비디오 테스트 시작: {person_name}")
#     print("==============================")

#     person_src = os.path.join(SRC_ROOT, person_name)

#     # harmful 비디오 폴더 후보
#     harm_candidates = ["비디오", "video", "Video"]
#     harm_dir = None
#     for c in harm_candidates:
#         p = os.path.join(person_src, c)
#         if os.path.exists(p):
#             harm_dir = p
#             break
#     if harm_dir is None:
#         raise FileNotFoundError(f"❌ harmful 비디오 폴더 없음: {harm_candidates}")

#     # safe 비디오 폴더 후보
#     safe_candidates = ["안전비디오", "안전_비디오", "safe_video", "safe", "Safe"]
#     safe_dir = None
#     for c in safe_candidates:
#         p = os.path.join(person_src, c)
#         if os.path.exists(p):
#             safe_dir = p
#             break
#     if safe_dir is None:
#         raise FileNotFoundError(f"❌ safe 비디오 폴더 없음: {safe_candidates}")

#     # 전역 갱신
#     HARM_VIDEO_DIR = harm_dir
#     SAFE_VIDEO_DIR = safe_dir

#     person_out = os.path.join(OUT_ROOT, person_name)
#     LABEL_DIR = os.path.join(person_out, "라벨_결과")
#     FRAMES_ROOT = os.path.join(person_out, "video_frames")

#     os.makedirs(LABEL_DIR, exist_ok=True)
#     os.makedirs(FRAMES_ROOT, exist_ok=True)

#     print(f"📂 harmful dir: {HARM_VIDEO_DIR}")
#     print(f"📂 safe   dir: {SAFE_VIDEO_DIR}")
#     print(f"📂 label  dir: {LABEL_DIR}")
#     print(f"📂 frames dir: {FRAMES_ROOT}")

#     verified, safe = auto_generate_video_labels()
#     merge_scores_and_update_video_labels(verified, safe)

#     print(f"🎉 {person_name} 비디오 처리 완료!\n")
def process_person(person_name: str):
    """
    사람별로:
      1) person_name_labels_categorized.json을 읽어서
      2) category == 'safe' -> label=0, 그 외 -> label=1 (정답)
      3) 정답이 있는 비디오에만 모델 실행 (process_one_video)
      4) 결과를 verified_video_labels.json / safe_video_labels.json 에 저장
    """
    global HARM_VIDEO_DIR, SAFE_VIDEO_DIR, LABEL_DIR, FRAMES_ROOT

    print("\n==============================")
    print(f"🎬 비디오 평가 시작(카테고리 기반 GT): {person_name}")
    print("==============================")

    # 0) 경로 설정
    person_src = os.path.join(SRC_ROOT, person_name)

    # harmful / safe 비디오 폴더 (물리적 위치용, 의미는 이제 GT에 안 씀)
    harm_candidates = ["비디오", "video", "Video"]
    safe_candidates = ["안전비디오", "안전_비디오", "safe_video", "safe", "Safe"]

    harm_dir = None
    for c in harm_candidates:
        p = os.path.join(person_src, c)
        if os.path.exists(p):
            harm_dir = p
            break

    safe_dir = None
    for c in safe_candidates:
        p = os.path.join(person_src, c)
        if os.path.exists(p):
            safe_dir = p
            break

    if harm_dir is None and safe_dir is None:
        raise FileNotFoundError(f"❌ {person_name}: 비디오 폴더를 찾을 수 없음")

    HARM_VIDEO_DIR = harm_dir
    SAFE_VIDEO_DIR = safe_dir

    person_out = os.path.join(OUT_ROOT, person_name)
    LABEL_DIR = os.path.join(person_out, "라벨_결과")
    FRAMES_ROOT = os.path.join(person_out, "video_frames")

    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(FRAMES_ROOT, exist_ok=True)

    print(f"📂 person_src: {person_src}")
    if HARM_VIDEO_DIR:
        print(f"📂 harm dir : {HARM_VIDEO_DIR}")
    if SAFE_VIDEO_DIR:
        print(f"📂 safe dir : {SAFE_VIDEO_DIR}")
    print(f"📂 label dir: {LABEL_DIR}")
    print(f"📂 frames dir: {FRAMES_ROOT}")

    # 1) 사람이 만든 category JSON (정답 정보)
    categ_path = os.path.join(
        CATEG_ROOT,
        f"{person_name}_labels_categorized.json"
    )
    cat_dict = load_json(categ_path, {})
    if not cat_dict:
        print(f"⚠️ {person_name}: 카테고리 JSON이 비어있거나 없음 → {categ_path}")
        print("   → 이 사람은 스킵합니다.")
        return

    print(f"📄 카테고리 JSON 로드 완료: {categ_path} (keys={len(cat_dict)})")

    # 2) 이 사람 폴더 아래 모든 비디오 목록 수집
    all_video_paths = []

    search_roots = []
    if HARM_VIDEO_DIR:
        search_roots.append(HARM_VIDEO_DIR)
    if SAFE_VIDEO_DIR and SAFE_VIDEO_DIR not in search_roots:
        search_roots.append(SAFE_VIDEO_DIR)
    if not search_roots:
        # 혹시 둘 다 못 찾았으면 person_src 전체를 뒤진다
        search_roots = [person_src]

    for root in search_roots:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(VIDEO_EXTS):
                    full = os.path.join(dirpath, f)
                    all_video_paths.append(full)

    all_video_paths = sorted(set(all_video_paths))
    print(f"🔍 발견한 비디오 개수: {len(all_video_paths)}")

    # 3) 평가 결과를 담을 dict (GT에 따라 나눔)
    verified = {}  # label=1 (유해)
    safe = {}      # label=0 (안전)

    TH = 0.63  # violence_prob threshold

    # 4) 각 비디오에 대해:
    #    - categ JSON에서 category 찾기
    #    - category -> GT label(0/1) 결정
    #    - 정답이 있는 비디오에만 모델 실행
    for video_path in all_video_paths:
        # 사람 폴더 기준 상대경로 (키로 쓰기 좋음)
        rel_from_person = os.path.relpath(video_path, person_src)

        # 사람 라벨 JSON에서 category 찾기
        cat_info = find_category(cat_dict, rel_from_person)
        if not cat_info:
            # 카테고리 JSON에 없으면 이 비디오는 "입력 안 받은 것" → 스킵
            print(f"  ⚠️ 카테고리 없음, 스킵: {rel_from_person}")
            continue

        cat_val = extract_category_value(cat_info)
        if cat_val is None:
            print(f"  ⚠️ category 필드 없음, 스킵: {rel_from_person} -> {cat_info}")
            continue

        cat_str = str(cat_val).lower()

        # ✅ 정답 라벨 결정: category == 'safe' → 0, 그 외 → 1
        is_safe = (cat_str == "safe")
        gt_label = 0 if is_safe else 1

        # frames 디렉토리 구조용 kind (정답 기준으로 나누자)
        kind = "safe" if is_safe else "harm"

        print(f"\n🎥 비디오 처리: {rel_from_person}")
        print(f"   - category: {cat_val} → GT label={gt_label} ({'safe' if is_safe else 'harmful'})")

        # 5) 모델 실행
        score, stats = process_one_video(video_path, kind=kind)
        pred = 1 if score >= TH else 0

        info = {
            "category": cat_val,
            "label": gt_label,          # ✅ 정답 (사람 라벨)
            "pred_label": pred,         # ✅ 모델 예측
            "violence_prob": score,     # 모델 출력
            "duration": stats["duration"],
            "fps": stats["fps"],
            "total_frames": stats["total_frames"],
            "sampled_frames": stats["sampled_frames"],
            "estimated_action": stats["estimated_action"],
            "is_safe": is_safe,
            "is_harmful": not is_safe,
        }

        # GT에 따라 두 개 JSON으로 분리 저장
        if gt_label == 1:
            verified[rel_from_person] = info
        else:
            safe[rel_from_person] = info

        print(
            f"   → violence_prob={score:.3f}, pred_label={pred}, "
            f"{'정답' if pred == gt_label else '오답'}"
        )

    # 6) 결과 저장
    verified_path = os.path.join(LABEL_DIR, "verified_video_labels.json")
    safe_path     = os.path.join(LABEL_DIR, "safe_video_labels.json")

    save_json(verified_path, verified)
    save_json(safe_path, safe)

    print(f"\n✅ {person_name} 비디오 평가 완료")
    print(f"   - harmful(정답=1) 개수: {len(verified)}")
    print(f"   - safe   (정답=0) 개수: {len(safe)}\n")


def update_video_categories_for_person(person_name: str):
    """
    사람이 만든 *_labels_categorized.json을 읽어서
    이미 존재하는 verified_video_labels.json / safe_video_labels.json에
    category 필드만 추가/갱신하는 가벼운 유틸.
    (모델 실행 X)
    """
    print("\n==============================")
    print(f"📝 {person_name} 비디오 category 갱신 시작 (models X)")
    print("==============================")

    # 1) 사람별 categorized JSON
    categ_path = os.path.join(
        CATEG_ROOT,
        f"{person_name}_labels_categorized.json"
    )
    cat_dict = load_json(categ_path, {})
    if not cat_dict:
        print(f"⚠️ {person_name}: 카테고리 JSON이 비어있거나 없음 → {categ_path}")
    else:
        print(f"📄 카테고리 JSON 로드 완료: {categ_path} (keys={len(cat_dict)})")

    # 2) 비디오 라벨 파일들
    label_dir = os.path.join(OUT_ROOT, person_name, "라벨_결과")
    verified_path = os.path.join(label_dir, "verified_video_labels.json")
    safe_path     = os.path.join(label_dir, "safe_video_labels.json")

    verified = load_json(verified_path, {})
    safe     = load_json(safe_path, {})

    if not verified and not safe:
        print(f"⚠️ {person_name}: 기존 비디오 라벨 JSON이 없어 스킵합니다.")
        return

    # 3) 원본 백업
    backup_if_needed(verified_path)
    backup_if_needed(safe_path)

    # 4) harmful 비디오 category 갱신
    updated_harm = 0
    for key, info in verified.items():
        cat_info = find_category(cat_dict, key)
        if not cat_info:
            print(f"  ⚠️ harmful category 매칭 실패: {key}")
            continue

        cat_val = extract_category_value(cat_info)
        if cat_val is None:
            print(f"  ⚠️ harmful category 값 없음: {key} -> {cat_info}")
            continue

        if not isinstance(info, dict):
            info = {}
        info["category"] = cat_val
        verified[key] = info
        updated_harm += 1

    # 5) safe 비디오 category 갱신
    updated_safe = 0
    for key, info in safe.items():
        cat_info = find_category(cat_dict, key)
        if not cat_info:
            print(f"  ⚠️ safe category 매칭 실패: {key}")
            continue

        cat_val = extract_category_value(cat_info)
        if cat_val is None:
            print(f"  ⚠️ safe category 값 없음: {key} -> {cat_info}")
            continue

        if not isinstance(info, dict):
            info = {}
        info["category"] = cat_val
        safe[key] = info
        updated_safe += 1

    # 6) 저장
    save_json(verified_path, verified)
    save_json(safe_path, safe)

    print(f"✅ {person_name} category 갱신 완료 → harmful={updated_harm}, safe={updated_safe}")
# =========================================
# MAIN
# =========================================
def main():
    people = ["박상원", "안지산", "임영재"]

    # 사용법:
    # 1) 전체 파이프라인 실행: python this_script.py
    # 2) category만 갱신:       python this_script.py --update-category
    if "--update-category" in sys.argv:
        print("🔧 MODE: category 업데이트만 수행 (모델 실행 X)")
        for person in people:
            update_video_categories_for_person(person)
        print("\n🎉 모든 사람 category 업데이트 완료")
        return

    # 기본: 전체 비디오 파이프라인 실행
    for person in people:
        process_person(person)
    print("\n🎉 모든 사람 비디오 테스트 완료")


if __name__ == "__main__":
    main()
