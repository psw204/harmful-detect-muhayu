#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import subprocess
import platform
import argparse
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def venv_python(path_unix, path_win):
    return os.path.join(BASE_DIR,
        path_win if platform.system() == "Windows" else path_unix
    )

# Python path for each venv
PY_TORCH = venv_python("venv_pt/bin/python", "venv_pt/Scripts/python.exe")
PY_TF    = venv_python("venv_tf/bin/python", "venv_tf/Scripts/python.exe")

# Paths
INPUTS_DIR  = os.path.join(BASE_DIR, "inputs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FRAMES_DIR  = os.path.join(INPUTS_DIR, "frames")

#비디오 입력
VIDEO_PATH = os.path.join(INPUTS_DIR, "test6.avi")
AUDIO_PATH = os.path.join(INPUTS_DIR, "safe.avi")

VISION_JSON = os.path.join(OUTPUTS_DIR, "vision_results.json")
CLIP_JSON   = os.path.join(OUTPUTS_DIR, "clip_result.json")
AUDIO_JSON  = os.path.join(OUTPUTS_DIR, "audio_result.json")
TEXT_JSON   = os.path.join(OUTPUTS_DIR, "text_result.json")
SLOWFAST_JSON    = os.path.join(OUTPUTS_DIR, "slowfast_result.json")
FUSED_JSON  = os.path.join(OUTPUTS_DIR, "fusion_result.json")
FRAMES_JSON = os.path.join(FRAMES_DIR, "meta.json")
VIT_JSON   = os.path.join(OUTPUTS_DIR, "vit_result.json")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

with open(FRAMES_JSON, "r") as f:
    meta = json.load(f)

fps = meta["meta"]["fps"]
print(f"[orchestrate] real fps from meta.json = {fps}")

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def run(cmd, cwd=None, check=True, env=None):
    print("▶", " ".join(str(c) for c in cmd))
    completed = subprocess.run(cmd, cwd=cwd, env=env)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def extract_audio_if_needed(video_path):
    if os.path.exists(AUDIO_PATH):
        print(f"🎧 Audio already exists, skip -> {AUDIO_PATH}")
        return
    print("🎧 Extracting audio with ffmpeg...")
    proc = subprocess.run([
        "ffmpeg", "-hide_banner", "-nostdin",
        "-i", video_path, "-ac", "1", "-ar", "16000", "-vn",
        AUDIO_PATH, "-y"
    ])
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed to extract audio.")
    print(f"✅ Audio extracted -> {AUDIO_PATH}")


# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default=None,
        help="Run only one module: split, yolo, clip, audio, slowfast, text, fusion")
    args = parser.parse_args()

    # ---------------------------------------------------
    # 1) Video Split
    # ---------------------------------------------------
    if args.test is None or args.test == "split":
        run([
            PY_TORCH,
            os.path.join(BASE_DIR, "scripts/video_split.py"),
            "--video", VIDEO_PATH,
            "--out", FRAMES_DIR,
            "--clip-sec", "2"
        ])

    # ---------------------------------------------------
    # 2) YOLO
    # ---------------------------------------------------
    if args.test is None or args.test == "yolo":
        run([
            PY_TORCH,
            os.path.join(BASE_DIR, "scripts/vision_yolo.py"),
            "--frames", FRAMES_DIR,
            "--out", VISION_JSON,
            "--model", "yolov8n.pt",
            "--conf", "0.25"
        ])

    # ---------------------------------------------------
    # 3) CLIP Violence
    # ---------------------------------------------------
    if args.test is None or args.test == "clip":
        run([
            PY_TORCH,
            os.path.join(BASE_DIR, "scripts/vision_clip_violence.py"),
            "--frames", FRAMES_DIR,
            "--out", CLIP_JSON,
            "--batch", "16",
            "--stride", "10"
        ])
    
    # 3.5) ViT Violence (scene-level)
    # ---------------------------------------------------
    if args.test is None or args.test == "vit":
        run([
            PY_TORCH,
            os.path.join(BASE_DIR, "scripts/vision_vit.py"),
            "--frames", FRAMES_DIR,
            "--out", VIT_JSON,
            "--batch", "16",
            "--stride", "10"
        ])

    # ---------------------------------------------------
    # 4) Audio (YAMNet)
    # ---------------------------------------------------
    # if args.test is None or args.test == "audio":
    #     tf_env = os.environ.copy()
    #     tf_env["CUDA_VISIBLE_DEVICES"] = "-1"
    #     tf_env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    #     tf_env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    #     extract_audio_if_needed(VIDEO_PATH)

    #     run([
    #         PY_TF,
    #         os.path.join(BASE_DIR, "scripts/audio_yamnet.py"),
    #         "--audio", AUDIO_PATH,
    #         "--out", AUDIO_JSON
    #     ], env=tf_env)


    # 5) SlowFast R101 (Action Recognition)
    if args.test is None or args.test == "slowfast":
        run([
            PY_TORCH,
            os.path.join(BASE_DIR, "scripts/video_slowfast.py"),
            "--frames", FRAMES_DIR,
            "--out", SLOWFAST_JSON,       # ← JSON 파일 이름은 유지해도 됨
            "--frames-per-clip", "32",
            #"--clip-sec", "2",
            "--fps", str(fps)
        ])
    # ---------------------------------------------------
    # 6) OCR + Hate/BERT
    # ---------------------------------------------------
    # if args.test is None or args.test == "text":
    #     first_frame = os.path.join(FRAMES_DIR, "clip_000_frame_000.jpg")
    #     run([
    #         PY_TORCH,
    #         os.path.join(BASE_DIR, "scripts/text_ocr_kohate.py"),
    #         "--image", first_frame,
    #         "--out", TEXT_JSON
    #     ])

    # ---------------------------------------------------
    # 7) Fusion
    # ---------------------------------------------------
    # 우선 yamnet, OCR 무시
    if not os.path.exists(AUDIO_JSON):
        json.dump({"overall": {"violence_prob": 0.0}}, open(AUDIO_JSON, "w"))

    if not os.path.exists(TEXT_JSON):
        json.dump({"overall": {}}, open(TEXT_JSON, "w"))
    if args.test is None or args.test == "fusion":
        run([
            PY_TORCH,
            os.path.join(BASE_DIR, "scripts/fusion_scores.py"),
            "--vision", VISION_JSON,
            "--audio", AUDIO_JSON,
            "--text", TEXT_JSON,
            "--clip", CLIP_JSON,
            "--vit", VIT_JSON,
            "--slowfast", SLOWFAST_JSON,
            "--out", FUSED_JSON
        ])


if __name__ == "__main__":
    main()
