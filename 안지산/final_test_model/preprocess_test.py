# preprocess_test.py
# ------------------------------------------------------------
# ✔ 3명(박상원, 안지산, 임영재)의 개인 수집 데이터 전처리
# ✔ 비디오만 처리
# ✔ clip/audio/text 생성
# ✔ 라벨 JSON 자동 탐색
# ✔ 각 멤버별 manifest 생성 후 test_manifest.jsonl 통합
# ------------------------------------------------------------

import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

DATA_ROOT = (ROOT / "../../무하유_유해콘텐츠_데이터/2_실제_수집_데이터(개인)").resolve()
LABEL_ROOT = (ROOT / "../../무하유_유해콘텐츠_데이터/3_라벨링_파일(개인)").resolve()

OUT_MANIFEST = ROOT / "test_manifest.jsonl"


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def make_dirs(base: Path):
    for d in ["clips", "audio", "text", "manifests"]:
        (base / d).mkdir(exist_ok=True)


def extract_audio(video_path: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (video_path.stem + ".wav")

    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", "16000", "-ac", "1",
        str(out_path)
    ])
    return out_path


def write_text_stub(video_path: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (video_path.stem + ".txt")
    out_path.write_text("", encoding="utf-8")
    return out_path


def write_manifest_jsonl(entries, out_path: Path):
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Split 없이 원본 그대로 사용
# ------------------------------------------------------------
def no_split(video_path: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    dst = out_dir / video_path.name
    if video_path != dst:
        run(["cp", str(video_path), str(dst)])
    return [dst]


# ------------------------------------------------------------
# 멤버 데이터 처리
# ------------------------------------------------------------
def process_member(member_name):
    print(f"\n=== {member_name} 전처리 시작 ===")

    member_dir = DATA_ROOT / member_name
    label_dir = LABEL_ROOT / member_name

    # 라벨 파일 탐색 (verified_video_labels.json 우선)
    label_file = None
    if (label_dir / "verified_video_labels.json").exists():
        label_file = label_dir / "verified_video_labels.json"
    else:
        json_candidates = list(label_dir.glob("*.json"))
        if len(json_candidates) == 0:
            print(f"❗ 라벨 파일 없음: {label_dir}")
            return []
        label_file = json_candidates[0]

    print(f"  ✔ 라벨 파일 사용: {label_file.name}")

    labels = json.loads(label_file.read_text(encoding="utf-8"))

    total_entries = []

    for folder_name in ["비디오", "안전_비디오"]:
        folder = member_dir / folder_name
        if not folder.exists():
            continue
        
        print(f" → 처리 중: {folder_name}")

        clips_dir = folder / "clips"
        audio_dir = folder / "audio"
        text_dir = folder / "text"
        manifest_dir = folder / "manifests"
        make_dirs(folder)

        manifest_entries = []

        # 폴더 안의 mp4 반복 처리
        for f in tqdm(folder.iterdir(), desc=f"{member_name}-{folder_name}"):
            if f.is_dir() or f.suffix.lower() != ".mp4":
                continue

            fname = f.name

            # 라벨 체크
            if fname not in labels:
                print(f"⚠ 라벨 없음 → 스킵: {fname}")
                continue

            info = labels[fname]
            is_harmful = info["is_harmful"]

            # split 없이 원본 그대로 clips 로 복사
            clips = no_split(f, clips_dir)

            for c in clips:
                audio_path = extract_audio(c, audio_dir)
                text_path = write_text_stub(c, text_dir)

                manifest_entries.append({
                    "video_path": str(c),
                    "audio_path": str(audio_path),
                    "text_path": str(text_path),
                    "is_harmful": is_harmful,
                    "source_member": member_name,
                    "source_folder": folder_name
                })

        # 멤버의 개별 manifest 저장
        write_manifest_jsonl(manifest_entries, manifest_dir / "manifest.jsonl")

        total_entries.extend(manifest_entries)

    return total_entries


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("=== Final Test Model 전처리 시작 ===\n")

    members = ["박상원", "안지산", "임영재"]
    all_entries = []

    for m in members:
        entries = process_member(m)
        all_entries.extend(entries)

    print("\n=== 전체 test_manifest.jsonl 생성 ===")
    write_manifest_jsonl(all_entries, OUT_MANIFEST)

    print(f"\n🎉 완료! 총 {len(all_entries)}개 항목 생성됨")
    print(f"✔ 출력: {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
