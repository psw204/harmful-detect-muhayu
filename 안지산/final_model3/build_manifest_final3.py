import json
from pathlib import Path
from tqdm import tqdm

def convert_manifest(in_file, out_file):
    out = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Convert {in_file}"):
            data = json.loads(line)

            video_path = data["video"].get("clip_path")
            audio_path = data["audio"].get("path")
            text_path = data["text"].get("path")
            label = int(data["harmful"])

            if video_path is None:
                continue   # 비디오 없는 경우 제거

            out.append({
                "video_path": video_path,
                "audio_path": audio_path,
                "text_path": text_path,
                "label": label
            })

    # 저장
    with open(out_file, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved: {out_file} ({len(out)} samples)")

if __name__ == "__main__":
    convert_manifest(
        "../팀원_전처리/splits/train.jsonl",
        "../팀원_전처리/splits/train_fm3.jsonl"
    )
    convert_manifest(
        "../팀원_전처리/splits/val.jsonl",
        "../팀원_전처리/splits/val_fm3.jsonl"
    )
