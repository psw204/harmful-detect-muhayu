#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================
# 팀원 데이터 경로
# =========================================
TEAM_ROOT       = os.path.join(BASE_DIR, "팀원_라벨링", "팀원_데이터")
HARM_VIDEO_DIR  = os.path.join(TEAM_ROOT, "비디오")
SAFE_VIDEO_DIR  = os.path.join(TEAM_ROOT, "안전_비디오")
LABEL_DIR       = os.path.join(TEAM_ROOT, "라벨_결과")
FRAMES_ROOT     = os.path.join(TEAM_ROOT, "video_frames")

os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(FRAMES_ROOT, exist_ok=True)

# =========================================
# 스크립트 경로
# =========================================
SCRIPT_DIR      = os.path.join(BASE_DIR, "scripts")

VIDEO_SPLIT_PY  = os.path.join(SCRIPT_DIR, "video_split.py")
CLIP_PY         = os.path.join(SCRIPT_DIR, "vision_clip_violence.py")
VIT_PY          = os.path.join(SCRIPT_DIR, "vision_vit.py")
AUDIO_YAMNET_PY = os.path.join(SCRIPT_DIR, "audio_yamnet.py")
SLOWFAST_PY     = os.path.join(SCRIPT_DIR, "video_slowfast.py")
TEXT_OCR_PY     = os.path.join(SCRIPT_DIR, "text_ocr_kohate.py")
FUSION_PY       = os.path.join(SCRIPT_DIR, "fusion_scores.py")  # YOLO 없는 버전

PY_TORCH = "../../Capstone2/Im/venv_pt/bin/python"
PY_TF    = "../../Capstone2/Im/venv_tf/bin/python"

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

def round5(x):
    return round(float(x), 5)

# =========================================
# 1) 자동 라벨 생성
# =========================================
def auto_generate_video_labels():
    harm_files = sorted([f for f in os.listdir(HARM_VIDEO_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))])
    safe_files = sorted([f for f in os.listdir(SAFE_VIDEO_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))])

    verified_video_labels = {f: 1 for f in harm_files}
    safe_video_labels = {f: 0 for f in safe_files}

    json.dump(verified_video_labels,
              open(os.path.join(LABEL_DIR, "verified_video_labels.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    json.dump(safe_video_labels,
              open(os.path.join(LABEL_DIR, "safe_video_labels.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"🔹 비디오 자동 라벨 생성: harmful={len(verified_video_labels)}, safe={len(safe_video_labels)}")

    return verified_video_labels, safe_video_labels


# =========================================
# 2) 한 영상 전체 처리
# =========================================
def process_one_video(video_path):
    video_name = os.path.basename(video_path)
    stem, _ = os.path.splitext(video_name)

    frames_dir = os.path.join(FRAMES_ROOT, stem)
    os.makedirs(frames_dir, exist_ok=True)

    FRAMES_JSON   = os.path.join(frames_dir, "meta.json")
    CLIP_JSON     = os.path.join(frames_dir, "clip_result.json")
    VIT_JSON      = os.path.join(frames_dir, "vit_result.json")
    AUDIO_WAV     = os.path.join(frames_dir, f"{stem}_audio.wav")
    AUDIO_JSON    = os.path.join(frames_dir, "audio_result.json")
    TEXT_JSON     = os.path.join(frames_dir, "text_result.json")
    SLOWFAST_JSON = os.path.join(frames_dir, "slowfast_result.json")
    FUSED_JSON    = os.path.join(frames_dir, "fusion_result.json")

    # 1) 영상 분할
    run([
        PY_TORCH, VIDEO_SPLIT_PY,
        "--video", video_path,
        "--out", frames_dir,
        "--clip-sec", "2"
    ])

    meta = load_json(FRAMES_JSON, {})
    meta_meta = meta.get("meta", {}) if isinstance(meta.get("meta", {}), dict) else meta
    fps = meta_meta.get("fps", 30.0)
    total_frames = meta_meta.get("total_frames_saved") or meta_meta.get("frames") or 0
    try:
        fps = float(fps)
    except:
        fps = 30.0
    try:
        total_frames = int(total_frames)
    except:
        total_frames = 0

    duration = float(total_frames) / fps if fps > 0 and total_frames > 0 else 0.0

    print(f"[{video_name}] fps = {fps}")

    # 2) CLIP
    run([
        PY_TORCH, CLIP_PY,
        "--frames", frames_dir,
        "--out", CLIP_JSON,
        "--batch", "16",
        "--stride", "10"
    ])

    # 3) ViT
    run([
        PY_TORCH, VIT_PY,
        "--frames", frames_dir,
        "--out", VIT_JSON,
        "--batch", "16",
        "--stride", "10"
    ])

    # 4) Audio + YAMNet  (무음 영상 대응)
    audio_ok = False

    if not os.path.exists(AUDIO_WAV):
        cmd = [
            "ffmpeg", "-hide_banner", "-nostdin",
            "-i", video_path, "-ac", "1", "-ar", "16000", "-vn",
            AUDIO_WAV, "-y"
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
        # 실제 YAMNet 실행
        tf_env = os.environ.copy()
        tf_env["CUDA_VISIBLE_DEVICES"] = "-1"
        tf_env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        tf_env["TF_CPP_MIN_LOG_LEVEL"] = "2"

        run([
            PY_TF, AUDIO_YAMNET_PY,
            "--audio", AUDIO_WAV,
            "--out", AUDIO_JSON
        ], env=tf_env)
    else:
        # 더미 audio_result.json 생성 (fusion용)
        dummy_audio = {
            "overall": {
                "violent_audio_prob": 0.0,
                "has_audio": False
            }
        }
        with open(AUDIO_JSON, "w", encoding="utf-8") as f:
            json.dump(dummy_audio, f, indent=2, ensure_ascii=False)
        print(f"📝 dummy audio_result.json 생성 -> {AUDIO_JSON}")

    # 5) TEXT (OCR + KoBERT) — 텍스트 없으면 dummy
    # frames_dir 안에서 아무 JPG/PNG 하나 골라서 OCR에 사용
    frame_imgs = [
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    frame_imgs.sort()

    text_ok = False
    ocr_output_txt = os.path.join(frames_dir, "ocr_text.txt")

    if frame_imgs:
        first_frame = os.path.join(frames_dir, frame_imgs[0])

        try:
            # text_ocr_kohate.py가 "이미지 1장 → text_result.json" 까지 한 방에 하는 구조라면:
            #   --image: 입력 이미지
            #   --out:   text_result.json (overall.hate_prob, overall.sexual_text_prob 등)
            run([
                PY_TORCH,
                TEXT_OCR_PY,
                "--image", first_frame,
                "--out", TEXT_JSON
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
                "has_text": False
            }
        }
        with open(TEXT_JSON, "w", encoding="utf-8") as f:
            json.dump(dummy_text, f, indent=2, ensure_ascii=False)
        print(f"📝 dummy text_result.json 생성 -> {TEXT_JSON}")

     # 6) SlowFast
    run([
        PY_TORCH, SLOWFAST_PY,
        "--frames", frames_dir,
        "--out", SLOWFAST_JSON,
        "--frames-per-clip", "32",
        "--fps", str(fps)
    ])

    # 7) Fusion (YOLO 없음)
    run([
        PY_TORCH, FUSION_PY,
        "--clip", CLIP_JSON,
        "--vit", VIT_JSON,
        "--audio", AUDIO_JSON,
        "--text", TEXT_JSON,
        "--slowfast", SLOWFAST_JSON,
        "--out", FUSED_JSON
    ])

    fused = load_json(FUSED_JSON, {})
    violence_prob = fused.get("overall", {}).get("violence_prob", 0.0)

    # ---- CLIP 사용 프레임 수 추출 (가능한 경우) ----
    clip_meta = load_json(CLIP_JSON, {})
    sampled_frames = 0

    # 1) meta.sampled_frames 우선
    meta_clip = clip_meta.get("meta", {})
    if isinstance(meta_clip, dict) and "sampled_frames" in meta_clip:
        try:
            sampled_frames = int(meta_clip["sampled_frames"])
        except:
            sampled_frames = 0

    # 2) fallback: overall.sampled_frames
    if sampled_frames == 0:
        overall_clip = clip_meta.get("overall", {})
        if isinstance(overall_clip, dict) and "sampled_frames" in overall_clip:
            try:
                sampled_frames = int(overall_clip["sampled_frames"])
            except:
                sampled_frames = 0

    # 3) fallback: frames 리스트 길이
    if sampled_frames == 0 and "frames" in clip_meta and isinstance(clip_meta["frames"], list):
        sampled_frames = len(clip_meta["frames"])


    # ---- SlowFast에서 대표 행동 추출 ----
    slow_json = load_json(SLOWFAST_JSON, {})
    estimated_action = None
    clips = slow_json.get("clips") or []
    if clips:
        # 각 클립에서 top-1 label 뽑아서 가장 많이 나온 걸 대표로 사용
        from collections import Counter
        labels = []
        for c in clips:
            topk = c.get("topk") or []
            if topk:
                labels.append(topk[0].get("label"))
        if labels:
            estimated_action = Counter(labels).most_common(1)[0][0]
    if not estimated_action:
        estimated_action = "unknown"

    # ---- YOLO를 안 쓰고 있으므로 object 관련 필드는 빈 값으로 ----
    detected_objects = []
    total_detections = 0
    frame_detections = []

    violence_prob = round5(violence_prob)
    print(f"[{video_name}] 🔥 violence_prob = {violence_prob}")

    video_stats = {
        "duration": duration,
        "fps": fps,
        "total_frames": total_frames,
        "sampled_frames": sampled_frames,
        "detected_objects": detected_objects,
        "total_detections": total_detections,
        "frame_detections": frame_detections,
        "estimated_action": estimated_action,
    }

    # 👉 이제 점수 + 메타 정보 둘 다 반환
    return violence_prob, video_stats



# =========================================
# 3) 최종 라벨 갱신
# =========================================
def merge_scores_and_update_video_labels(verified, safe):
    TH = 0.45

    # -------------------------------
    # 1) 유해 비디오 (GT=1)
    # -------------------------------
    for fname in verified.keys():
        video_path = os.path.join(HARM_VIDEO_DIR, fname)
        score, stats = process_one_video(video_path)
        pred = 1 if score >= TH else 0

        verified[fname] = {
            "duration": stats["duration"],
            "fps": stats["fps"],
            "total_frames": stats["total_frames"],
            "sampled_frames": stats["sampled_frames"],
            "detected_objects": stats.get("detected_objects", []),
            "total_detections": stats.get("total_detections", 0),
            "frame_detections": stats.get("frame_detections", []),
            "estimated_action": stats["estimated_action"],
            "is_harmful": True,   # GT 라벨
            "label": 1,
            "pred_label": pred,
            "violence_prob": score,
        }

    # -------------------------------
    # 2) 안전 비디오 (GT=0)
    # -------------------------------
    for fname in safe.keys():
        video_path = os.path.join(SAFE_VIDEO_DIR, fname)
        score, stats = process_one_video(video_path)
        pred = 1 if score >= TH else 0

        # 모델이 1로 찍으면 "안전인데 유해로 본 케이스" → review
        category = "safe" if pred == 0 else "review"

        safe[fname] = {
            "is_safe": True,   # GT 기준
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

    # 저장 그대로
    json.dump(verified,
              open(os.path.join(LABEL_DIR, "verified_video_labels.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    json.dump(safe,
              open(os.path.join(LABEL_DIR, "safe_video_labels.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print("✅ 비디오 라벨 저장 완료 ")

# =========================================
# 새로 라벨 파일만 만드는 재빌드 모드
# =========================================
def rebuild_labels_only():
    verified = {}
    safe = {}

    # 기본 TH (필요하면 조정 가능)
    TH = 0.45

    # ---------------------------
    # 1) 유해 비디오(GT=1)
    # ---------------------------
    for fname in sorted(os.listdir(HARM_VIDEO_DIR)):
        if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        stem, _ = os.path.splitext(fname)
        frames_dir = os.path.join(FRAMES_ROOT, stem)

        fusion_path = os.path.join(frames_dir, "fusion_result.json")
        clip_path   = os.path.join(frames_dir, "clip_result.json")
        slow_path   = os.path.join(frames_dir, "slowfast_result.json")
        meta_path   = os.path.join(frames_dir, "meta.json")

        fusion = load_json(fusion_path, {})
        clip   = load_json(clip_path, {})
        slow   = load_json(slow_path, {})
        meta   = load_json(meta_path, {})

        # violence_prob
        violence = float(
            fusion.get("overall", {}).get(
                "violence_prob",
                fusion.get("scores", {}).get("final", 0.0)
            )
        )

        # meta에서 fps, total_frames, duration
        m = meta.get("meta", {}) if isinstance(meta.get("meta", {}), dict) else meta
        fps = float(m.get("fps", 30.0))
        total_frames = int(m.get("total_frames_saved") or m.get("frames") or 0)
        duration = total_frames / fps if fps > 0 else 0.0

        # sampled_frames 추출
        sampled_frames = 0
        mc = clip.get("meta", {})
        if "sampled_frames" in mc:
            sampled_frames = mc["sampled_frames"]
        elif "frames" in clip:
            sampled_frames = len(clip["frames"])

        # estimated_action (slowfast)
        est = "unknown"
        clips = slow.get("clips") or []
        if clips:
            from collections import Counter
            labels = []
            for c in clips:
                topk = c.get("topk") or []
                if topk:
                    lab = topk[0].get("label", "")
                    lab = str(lab).split("\t")[0]  # 번호 제거
                    labels.append(lab)
            if labels:
                est = Counter(labels).most_common(1)[0][0]

        pred = 1 if violence >= TH else 0

        verified[fname] = {
            "duration": duration,
            "fps": fps,
            "total_frames": total_frames,
            "sampled_frames": sampled_frames,
            "detected_objects": [],
            "total_detections": 0,
            "frame_detections": [],
            "estimated_action": est,
            "is_harmful": True,
            "label": 1,
            "pred_label": pred,
            "violence_prob": violence
        }

    # ---------------------------
    # 2) 안전 비디오(GT=0)
    # ---------------------------
    for fname in sorted(os.listdir(SAFE_VIDEO_DIR)):
        if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        stem, _ = os.path.splitext(fname)
        frames_dir = os.path.join(FRAMES_ROOT, stem)

        fusion_path = os.path.join(frames_dir, "fusion_result.json")
        clip_path   = os.path.join(frames_dir, "clip_result.json")
        slow_path   = os.path.join(frames_dir, "slowfast_result.json")
        meta_path   = os.path.join(frames_dir, "meta.json")

        fusion = load_json(fusion_path, {})
        clip   = load_json(clip_path, {})
        slow   = load_json(slow_path, {})
        meta   = load_json(meta_path, {})

        violence = float(
            fusion.get("overall", {}).get(
                "violence_prob",
                fusion.get("scores", {}).get("final", 0.0)
            )
        )

        m = meta.get("meta", {}) if isinstance(meta.get("meta", {}), dict) else meta
        fps = float(m.get("fps", 30.0))
        total_frames = int(m.get("total_frames_saved") or m.get("frames") or 0)
        duration = total_frames / fps if fps > 0 else 0.0

        sampled_frames = 0
        mc = clip.get("meta", {})
        if "sampled_frames" in mc:
            sampled_frames = mc["sampled_frames"]
        elif "frames" in clip:
            sampled_frames = len(clip["frames"])

        est = "unknown"
        clips = slow.get("clips") or []
        if clips:
            from collections import Counter
            labels = []
            for c in clips:
                topk = c.get("topk") or []
                if topk:
                    lab = topk[0].get("label", "")
                    lab = str(lab).split("\t")[0]
                    labels.append(lab)
            if labels:
                est = Counter(labels).most_common(1)[0][0]

        pred = 1 if violence >= TH else 0
        category = "safe" if pred == 0 else "review"

        safe[fname] = {
            "is_safe": True,
            "label": 0,
            "pred_label": pred,
            "category": category,
            "duration": duration,
            "fps": fps,
            "total_frames": total_frames,
            "sampled_frames": sampled_frames,
            "estimated_action": est,
            "violence_prob": violence
        }

    # 저장
    json.dump(verified,
              open(os.path.join(LABEL_DIR, "verified_video_labels.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    json.dump(safe,
              open(os.path.join(LABEL_DIR, "safe_video_labels.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print("🎯 기존 결과 기반 라벨 파일 재생성 완료!")

# =========================================
# MAIN
# =========================================
def main():
    rebuild_labels_only()
    # verified, safe = auto_generate_video_labels()
    # merge_scores_and_update_video_labels(verified, safe)
    # print("\n🎉 전체 비디오 멀티모달 라벨링 완료!")


if __name__ == "__main__":
    main()
