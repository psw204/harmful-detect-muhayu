# -*- coding: utf-8 -*-
"""
비디오 중심 멀티모달 전처리 스크립트 (안지산)
- 입력: 1_공개_데이터셋/{RLVS, RWF-2000}/.../*.mp4 (또는 2_실제_수집_데이터(개인))
- 출력: 4_전처리_결과(개인)/안지산/팀원_전처리/{clips,audio,text,manifests}

요구사항:
  - ffmpeg, ffprobe 설치 필요 (PATH에 있어야 함)
  - pip install opencv-python numpy tqdm srt

기능:
  1) 비디오 스캔 → 지속시간/프레임레이트 메타 수집
  2) 슬라이딩 윈도우로 클립 구간 생성 (win/stride)
  3) ffmpeg로 구간별 mp4, 16kHz mono wav 추출 (오디오 없으면 무음 생성)
  4) 자막(SRT) 있으면 구간 텍스트 생성, 없으면 빈 파일
  5) manifests/multimodal.jsonl 에 클립 단위 엔트리 append (resume 안전)

사용 예시(프로젝트 루트):
python 팀원_전처리/preprocess_videos.py \
  --member 안지산 \
  --public-root 무하유_유해콘텐츠_데이터/1_공개_데이터셋 \
  --out-root   무하유_유해콘텐츠_데이터/4_전처리_결과(개인)/안지산/팀원_전처리 \
  --datasets RLVS RWF-2000 \
  --win 4.0 --stride 2.0 --resume
"""
import os
import re
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

try:
    import srt
except Exception:
    srt = None

# --------------------------
# 경로 기본값
# --------------------------
# 팀원_전처리/ 폴더 안에 이 파일이 있다고 가정
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 상위 3단계 → 프로젝트 루트
PUBLIC_DEFAULT = PROJECT_ROOT / "무하유_유해콘텐츠_데이터" / "1_공개_데이터셋"
OUT_DEFAULT = PROJECT_ROOT / "무하유_유해콘텐츠_데이터" / "4_전처리_결과(개인)" / "안지산" / "팀원_전처리"

CLIPS_DIR = "clips"
AUDIO_DIR = "audio"
TEXT_DIR  = "text"
MANI_DIR  = "manifests"
MANI_NAME = "multimodal.jsonl"

# --------------------------
# 유틸
# --------------------------

def ensure_dirs(root: Path):
    (root / CLIPS_DIR).mkdir(parents=True, exist_ok=True)
    (root / AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    (root / TEXT_DIR ).mkdir(parents=True, exist_ok=True)
    (root / MANI_DIR ).mkdir(parents=True, exist_ok=True)


def list_videos(dirs: List[Path], limit: Optional[int] = None) -> List[Path]:
    vids = []
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.mp4"):
            vids.append(p)
    vids.sort()
    return vids[:limit] if limit else vids


def get_video_meta(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"fps": 0.0, "frames": 0, "duration": 0.0}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if fps > 0 else 0.0
    cap.release()
    return {"fps": float(fps), "frames": int(frames), "duration": float(duration)}


def generate_windows(duration: float, win: float, stride: float) -> List[Tuple[float, float]]:
    out = []
    t = 0.0
    while t + 0.5 < duration:  # 0.5초 이상 남아있을 때만
        s = t
        e = min(t + win, duration)
        if e - s >= 0.5:      # 최소 0.5초
            out.append((round(s, 3), round(e, 3)))
        t += stride
    return out


def run_ffmpeg(cmd: List[str]):
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)


def extract_clip(src: Path, dst: Path, start: float, end: float):
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", str(src), "-c", "copy", str(dst)])


# ---- 오디오 처리(개선): 없으면 무음 생성 ----
def has_audio_stream(src: Path) -> bool:
    """ffprobe로 오디오 스트림 존재 여부 확인"""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error",
             "-select_streams", "a:0",
             "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(src)],
            stderr=subprocess.STDOUT
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


def make_silence_wav(dst: Path, duration: float, sr: int = 16000):
    """오디오가 없을 때 무음 wav 생성"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.01, duration)
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi",
         "-t", str(dur),
         "-i", f"anullsrc=cl=mono:r={sr}",
         str(dst)],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True
    )


def extract_wav(src: Path, dst: Path, start: float, end: float):
    """오디오 추출: 있으면 추출, 없으면 무음 생성"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.01, end - start)
    if has_audio_stream(src):
        # 오디오만 추출(-vn), 첫 번째 오디오 스트림만 선택(-map a:0)
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-to", str(end),
             "-i", str(src), "-vn",
             "-map", "a:0",
             "-ac", "1", "-ar", "16000",
             str(dst)],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True
        )
    else:
        make_silence_wav(dst, dur, sr=16000)


def find_srt_for_video(video_path: Path) -> Optional[Path]:
    # 같은 폴더에 같은 이름 .srt 가 있으면 사용
    cand = video_path.with_suffix('.srt')
    return cand if cand.exists() else None


def read_srt_segments(srt_path: Path):
    if srt is None:
        return []
    try:
        txt = srt_path.read_text(encoding='utf-8')
    except Exception:
        try:
            txt = srt_path.read_text(encoding='cp949')
        except Exception:
            return []
    subs = list(srt.parse(txt))
    segs = []
    for s_sub in subs:
        segs.append({
            "start": s_sub.start.total_seconds(),
            "end": s_sub.end.total_seconds(),
            "text": s_sub.content.replace('\n', ' ').strip()
        })
    return segs


def slice_text_for_window(segs, start: float, end: float) -> str:
    if not segs:
        return ""
    parts = []
    for g in segs:
        # 구간이 겹치면 포함
        if not (g["end"] <= start or g["start"] >= end):
            parts.append(g["text"])
    return " ".join(parts).strip()


def build_clip_basename(video_path: Path, start: float, end: float) -> str:
    base = re.sub(r"[^\w\-]+", "_", video_path.stem)  # 안전한 파일명
    return f"{base}_{start:0.1f}_{end:0.1f}"


def append_manifest(mani_fp, entry: dict):
    mani_fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

# --------------------------
# 메인 파이프라인
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--member', default='안지산')
    ap.add_argument('--public-root', default=str(PUBLIC_DEFAULT))
    ap.add_argument('--datasets', nargs='+', default=['RLVS', 'RWF-2000'])
    ap.add_argument('--out-root', default=str(OUT_DEFAULT))
    ap.add_argument('--win', type=float, default=4.0)
    ap.add_argument('--stride', type=float, default=2.0)
    ap.add_argument('--min-dur', type=float, default=2.0, help='이보다 짧은 영상은 스킵')
    ap.add_argument('--video-limit', type=int, default=None)
    ap.add_argument('--resume', action='store_true', help='이미 존재하는 산출물은 건너뜀')
    args = ap.parse_args()

    public_root = Path(args.public_root)
    out_root = Path(args.out_root)
    ensure_dirs(out_root)

    # 입력 디렉터리 모으기
    in_dirs = [public_root / d for d in args.datasets]
    videos = list_videos(in_dirs, limit=args.video_limit)
    if not videos:
        print(f"입력 비디오가 없습니다. 경로를 확인하세요: {in_dirs}")
        return

    mani_path = out_root / MANI_DIR / MANI_NAME
    mani_path.parent.mkdir(parents=True, exist_ok=True)
    # append 모드 (resume 시 중복 append는 외부 후처리 또는 여기서 별도 체크 필요)
    mani_fp = open(mani_path, 'a', encoding='utf-8')

    pbar = tqdm(videos, desc='전처리', unit='vid')
    for vid in pbar:
        meta = get_video_meta(vid)
        dur = meta['duration']
        if dur < args.min_dur:
            continue

        srt_path = find_srt_for_video(vid)
        segs = read_srt_segments(srt_path) if srt_path else []

        wins = generate_windows(dur, args.win, args.stride)
        for (st, et) in wins:
            base = build_clip_basename(vid, st, et)
            clip_path = out_root / CLIPS_DIR / f"{base}.mp4"
            wav_path  = out_root / AUDIO_DIR / f"{base}.wav"
            txt_path  = out_root / TEXT_DIR  / f"{base}.txt"

            # resume: 존재하면 건너뜀
            if args.resume and clip_path.exists() and wav_path.exists() and txt_path.exists():
                continue

            # 추출
            try:
                extract_clip(vid, clip_path, st, et)
                extract_wav(vid, wav_path, st, et)
                text_content = slice_text_for_window(segs, st, et)
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                txt_path.write_text(text_content, encoding='utf-8')
            except subprocess.CalledProcessError as e:
                print(f"ffmpeg 오류로 스킵: {vid} [{st},{et}] → {e}")
                continue
            except Exception as e:
                print(f"예외로 스킵: {vid} [{st},{et}] → {e}")
                continue

            # 매니페스트 append
            entry = {
                "clip_id": f"{vid.parent.name}/{vid.stem}_{st:.1f}_{et:.1f}",
                "member": args.member,
                "video": {
                    "src": str(vid),
                    "clip_path": str(clip_path),
                    "start": float(st),
                    "end": float(et),
                    "fps": meta['fps'],
                    "duration": float(et - st)
                },
                "audio": {
                    "path": str(wav_path),
                    "sr": 16000,
                    "channels": 1
                },
                "text": {
                    "path": str(txt_path),
                    "source": 'srt' if srt_path else 'none',
                    "content_preview": (text_content[:80] + '...') if text_content else ''
                },
                "labels": {"action": None, "audio": None, "text": None}
            }
            append_manifest(mani_fp, entry)

    mani_fp.close()
    print(f"\n전처리 완료 → 매니페스트: {mani_path}")

if __name__ == '__main__':
    main()
