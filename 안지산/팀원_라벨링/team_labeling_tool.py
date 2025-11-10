# -*- coding: utf-8 -*-
"""
멀티모달 라벨링 툴 (MoViNet + YAMNet + KoBERT 파이프라인용, 진행상황 로그 강화)
- 입력: 전처리 매니페스트 JSONL (preprocess_videos.py가 생성)
- 출력: 3_라벨링_파일(개인)/<팀원>/라벨_결과/ 에 JSONL 2종
    - approved_multimodal_labels.jsonl  (유해/관심 클립: 확정 라벨)
    - safe_multimodal_labels.jsonl      (안전/무관 클립: 안전 기록)

추가/변경점(진행상황 가시화):
  • 시작 시 요약 출력(전체/이미 라벨됨/이번 처리 대상)
  • 매 클립 진입 시 "[i/N] clip_id=..." 로그
  • 저장 시 한 줄 요약 로그(harmful/safe, 선택 라벨)
  • S 키로 스킵 지원(로그 반영)
  • --verbose 옵션으로 경로 상세 출력
  • 종료/완료 시 집계 요약

키바인딩:
  [라벨 지정]
    1..7 : 행동(Action) 클래스 선택
    f/g  : 오디오(Audio) 다음/이전 클래스 순환
    v/b  : 텍스트(Text)  다음/이전 클래스 순환
    0    : 오디오/텍스트를 'none'으로 빠르게 설정 (토글)
    u    : 행동(Action) 라벨 해제(None)

  [확정/제어]
    Y 혹은 Space : '유해/관심'으로 확정 → approved JSONL에 저장
    N            : '안전'으로 확정 → safe JSONL에 저장
    P            : 오디오 재생/토글 (ffplay 필요)
    A            : YOLO 힌트 토글 (중앙 프레임에 박스 표시)
    S            : 현재 클립 스킵
    ←/→         : 이전/다음 클립
    Q            : 종료

실행 예시(팀원_라벨링 폴더 기준):
  python team_labeling_tool.py \
    --member "안지산" \
    --manifest "../../무하유_유해콘텐츠_데이터/4_전처리_결과(개인)/안지산/팀원_전처리/manifests/multimodal.jsonl" \
    --out-root "../../무하유_유해콘텐츠_데이터/3_라벨링_파일(개인)/안지산/라벨_결과" \
    --yolo-hint yolov8n.pt --verbose
"""
import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

# ---------------------- 기본 경로 ----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
DEFAULT_OUT = PROJECT_ROOT / "무하유_유해콘텐츠_데이터" / "3_라벨링_파일(개인)" / "안지산" / "라벨_결과"

APPROVED_NAME = "approved_multimodal_labels.jsonl"
SAFE_NAME     = "safe_multimodal_labels.jsonl"

# ---------------------- 클래스 정의 ----------------------
ACTION_CLASSES = [
    "violence",      # 1
    "assault",       # 2
    "threat",        # 3
    "self_harm",     # 4
    "weapon_use",    # 5
    "sexual",        # 6
    "non_violence"   # 7 (안전한 행동)
]
AUDIO_CLASSES = ["scream", "anger", "threat", "abuse", "none"]
TEXT_CLASSES  = ["threat", "hate", "sexual", "self_harm", "none"]

# ---------------------- 유틸 ----------------------

def load_manifest(jsonl_path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def ensure_outdir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def read_existing_ids(jsonl_path: Path) -> set:
    ids = set()
    if not jsonl_path.exists():
        return ids
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                cid = data.get('clip_id')
                if cid:
                    ids.add(cid)
            except Exception:
                continue
    return ids


def append_jsonl(jsonl_path: Path, obj: Dict[str, Any]):
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def midframe_image(video_path: str, width_limit: int = 1200) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pos = frames // 2 if frames > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    if w > width_limit:
        frame = cv2.resize(frame, (width_limit, int(h * width_limit / w)))
    return frame


def put_multiline_text(img, lines: List[str], x: int = 10, y: int = 28, color=(0, 0, 255)):
    for i, line in enumerate(lines):
        yy = y + i * 24
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# optional YOLO hint
class YOLOHint:
    def __init__(self, model_path: Optional[str]):
        self.enabled = False
        self.model = None
        if model_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            except Exception:
                self.model = None

    def infer(self, frame: np.ndarray):
        if self.model is None:
            return []
        try:
            res = self.model(frame, verbose=False)
            outs = []
            for r in res:
                boxes = r.boxes
                if boxes is None:
                    continue
                for b in boxes:
                    cls = r.names[int(b.cls)]
                    conf = float(b.conf)
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    outs.append((cls, conf, x1, y1, x2, y2))
            return outs
        except Exception:
            return []


# 오디오 재생 토글 (ffplay 비차단)
class AudioPlayer:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None

    def toggle(self, wav_path: str):
        # 이미 재생 중이면 종료
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
            return
        # 새로 재생
        if not Path(wav_path).exists():
            return
        try:
            self.proc = subprocess.Popen([
                'ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', wav_path
            ])
        except Exception:
            self.proc = None


# ---------------------- 메인 ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--member', default='안지산')
    ap.add_argument('--manifest', required=True, help='전처리 매니페스트 JSONL 경로')
    ap.add_argument('--out-root', default=str(DEFAULT_OUT), help='라벨 결과 출력 디렉터리')
    ap.add_argument('--yolo-hint', default=None, help='YOLO 가중치(.pt) 경로 - 선택')
    ap.add_argument('--start', type=int, default=0, help='시작 인덱스')
    ap.add_argument('--limit', type=int, default=None, help='처리할 최대 개수')
    ap.add_argument('--verbose', action='store_true', help='상세 로그 출력')
    args = ap.parse_args()

    member = args.member
    out_dir = Path(args.out_root)
    ensure_outdir(out_dir)
    approved_path = out_dir / APPROVED_NAME
    safe_path = out_dir / SAFE_NAME

    print("\n[로드] 매니페스트:", args.manifest)
    items = load_manifest(Path(args.manifest))
    total_all = len(items)
    if args.limit is not None:
        items = items[args.start: args.start + args.limit]
    else:
        items = items[args.start:]

    # 이미 저장된 clip_id는 스킵
    done_ids = read_existing_ids(approved_path) | read_existing_ids(safe_path)
    total_done = len(done_ids)
    total_todo = sum(1 for it in items if it.get('clip_id') not in done_ids)
    print(f"[요약] 전체={total_all}  이미라벨={total_done}  이번실처리={total_todo}")
    if args.verbose:
        print(f"[출력] approved={approved_path}")
        print(f"[출력] safe    ={safe_path}\n")

    yolo = YOLOHint(args.yolo_hint)
    player = AudioPlayer()

    processed = 0
    saved_harm = 0
    saved_safe = 0
    skipped = 0

    idx = 0
    while idx < len(items):
        it = items[idx]
        cid = it.get('clip_id')
        if cid in done_ids:
            idx += 1
            continue

        vclip = it.get('video', {}).get('clip_path') or it.get('video', {}).get('src')
        wav   = it.get('audio', {}).get('path')
        tpath = it.get('text',  {}).get('path')
        tprev = it.get('text',  {}).get('content_preview', '')

        print(f"\n[진행] [{processed+1}/{total_todo}] clip_id={cid}")
        if args.verbose:
            print(f"       video={vclip}")
            print(f"       audio={wav}")
            print(f"       text ={tpath}")

        action_sel = None
        audio_sel  = None
        text_sel   = None

        frame = midframe_image(vclip)
        if frame is None:
            print("[경고] 프레임 추출 실패 → 스킵")
            idx += 1
            skipped += 1
            processed += 1
            continue

        hint_boxes = []

        while True:
            disp = frame.copy()

            # YOLO 힌트
            if yolo.enabled:
                hint_boxes = yolo.infer(disp)
                for cls, conf, x1, y1, x2, y2 in hint_boxes:
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(disp, f"{cls} {conf:.2f}", (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # 오버레이 텍스트
            lines_top = [
                f"[{processed+1}/{total_todo}] clip_id: {cid}",
                f"Action: {action_sel if action_sel else '-'} | Audio: {audio_sel if audio_sel else '-'} | Text: {text_sel if text_sel else '-'}",
            ]
            put_multiline_text(disp, lines_top, x=10, y=28, color=(0,0,255))

            # 하단 도움말 & 텍스트 프리뷰
            preview = (tprev or '')
            if tpath and Path(tpath).exists():
                try:
                    s = Path(tpath).read_text(encoding='utf-8')[:140]
                    if s:
                        preview = s
                except Exception:
                    pass
            help_lines = [
                "1..7=Action, f/g=Audio±, v/b=Text±, 0=none(A/T), u=unset Action, P=play audio",
                "Y/Space=유해 확정, N=안전, A=YOLO힌트 토글, S=스킵, ←/→=이전/다음, Q=종료",
                f"Text preview: {preview}",
            ]
            put_multiline_text(disp, help_lines, x=10, y=disp.shape[0]-80, color=(255,255,0))

            cv2.imshow('Multimodal Labeling (with progress)', disp)
            key = cv2.waitKey(0) & 0xFF

            # 종료
            if key in [ord('q'), ord('Q')]:
                cv2.destroyAllWindows()
                print("\n[종료] 사용자가 종료함")
                print(f"[요약] harmful={saved_harm}, safe={saved_safe}, 스킵={skipped}, 처리={processed}/{total_todo}")
                return

            # 이전/다음
            if key == 81:  # ←
                idx = max(0, idx-1)
                print("[이동] 이전 클립")
                break
            if key == 83:  # →
                idx = min(len(items)-1, idx+1)
                print("[이동] 다음 클립")
                break

            # 스킵(S)
            if key in [ord('s'), ord('S')]:
                print("[스킵] 사용자 입력으로 현재 클립 스킵")
                skipped += 1
                processed += 1
                idx += 1
                break

            # 라벨링 입력들
            if key in [ord(str(i)) for i in range(1,8)]:
                action_sel = ACTION_CLASSES[key - ord('1')]
                print(f"  · Action = {action_sel}")
                continue
            if key in [ord('f'), ord('F')]:
                cur = AUDIO_CLASSES.index(audio_sel) if audio_sel in AUDIO_CLASSES else -1
                audio_sel = AUDIO_CLASSES[(cur+1) % len(AUDIO_CLASSES)]
                print(f"  · Audio  = {audio_sel}")
                continue
            if key in [ord('g'), ord('G')]:
                cur = AUDIO_CLASSES.index(audio_sel) if audio_sel in AUDIO_CLASSES else 0
                audio_sel = AUDIO_CLASSES[(cur-1) % len(AUDIO_CLASSES)]
                print(f"  · Audio  = {audio_sel}")
                continue
            if key == ord('0'):
                audio_sel = 'none'
                text_sel  = 'none'
                print("  · Audio/Text = none")
                continue
            if key in [ord('v'), ord('V')]:
                cur = TEXT_CLASSES.index(text_sel) if text_sel in TEXT_CLASSES else -1
                text_sel = TEXT_CLASSES[(cur+1) % len(TEXT_CLASSES)]
                print(f"  · Text   = {text_sel}")
                continue
            if key in [ord('b'), ord('B')]:
                cur = TEXT_CLASSES.index(text_sel) if text_sel in TEXT_CLASSES else 0
                text_sel = TEXT_CLASSES[(cur-1) % len(TEXT_CLASSES)]
                print(f"  · Text   = {text_sel}")
                continue
            if key in [ord('u'), ord('U')]:
                action_sel = None
                print("  · Action = None")
                continue
            if key in [ord('p'), ord('P')]:
                if wav:
                    AudioPlayer().toggle(wav)
                    print("  · (오디오 재생 토글)")
                continue
            if key in [ord('a'), ord('A')]:
                yolo.enabled = not yolo.enabled
                print(f"  · YOLO hint = {'ON' if yolo.enabled else 'OFF'}")
                continue

            # 저장
            if key in [32, ord('y'), ord('Y')]:  # harmful
                record = {
                    'clip_id': cid,
                    'member': member,
                    'labels': {
                        'action': action_sel,
                        'audio': audio_sel,
                        'text': text_sel
                    },
                    'meta': {
                        'video': it.get('video', {}),
                        'audio': it.get('audio', {}),
                        'text': it.get('text', {})
                    },
                    'category': 'harmful_or_interest'
                }
                append_jsonl(approved_path, record)
                saved_harm += 1
                processed  += 1
                print(f"[저장] harmful: action={action_sel} audio={audio_sel} text={text_sel}")
                idx += 1
                break

            if key in [ord('n'), ord('N')]:       # safe
                record = {
                    'clip_id': cid,
                    'member': member,
                    'labels': {
                        'action': action_sel if action_sel else 'non_violence',
                        'audio': audio_sel if audio_sel else 'none',
                        'text':  text_sel  if text_sel  else 'none'
                    },
                    'meta': {
                        'video': it.get('video', {}),
                        'audio': it.get('audio', {}),
                        'text': it.get('text', {})
                    },
                    'category': 'safe'
                }
                append_jsonl(safe_path, record)
                saved_safe += 1
                processed  += 1
                print(f"[저장] safe   : action={record['labels']['action']} audio={record['labels']['audio']} text={record['labels']['text']}")
                idx += 1
                break

    cv2.destroyAllWindows()
    print("\n[완료] 모든 대상 처리 종료")
    print(f"[요약] harmful={saved_harm}, safe={saved_safe}, 스킵={skipped}, 처리완료={processed}/{total_todo}")


if __name__ == '__main__':
    main()
