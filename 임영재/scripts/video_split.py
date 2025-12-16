
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, math, cv2, argparse
from tqdm import tqdm
import shutil

def split_video(video_path, output_root, clip_len=2):
    # 비디오 파일명 추출
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # 안전 or 비디오 판별
    lower = video_path.lower()
    if "안전" in lower or "safe" in lower:
        type_name = "안전비디오"
    else:
        type_name = "비디오"

    # 최종 저장 경로 구성
    save_dir = os.path.join(output_root, type_name, base_name)

    # 기존 폴더 삭제 후 생성
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (total_frames / fps) if fps else 0.0

    clips = []
    total_saved = 0

    if fps == 0.0 or total_frames == 0:
        meta = {"video": video_path, "clips": 0, "total_frames_saved": 0, "fps": float(fps)}
        json.dump({"meta": meta, "clips": []},
                  open(os.path.join(save_dir, "meta.json"), "w"),
                  indent=2, ensure_ascii=False)
        print("⚠️ No frames or FPS=0.")
        return

    clip_count = int(math.floor(duration / clip_len))

    for i in tqdm(range(clip_count), desc=f"Splitting {base_name}"):
        start_t = i * clip_len
        end_t = min((i + 1) * clip_len, duration)

        start = int(start_t * fps)
        end   = int(end_t * fps)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = 0
        for _ in range(start, end):
            ret, frame = cap.read()
            if not ret: break
            path = os.path.join(save_dir, f"clip_{i:03d}_frame_{idx:03d}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)
            idx += 1

        total_saved += len(frames)
        clips.append({
            "index": i,
            "start_sec": float(start_t),
            "end_sec": float(end_t),
            "frames": frames
        })

    cap.release()

    meta = {"video": video_path, "clips": len(clips), "total_frames_saved": total_saved, "fps": float(fps)}
    json.dump({"meta": meta, "clips": clips},
              open(os.path.join(save_dir, "meta.json"), "w"),
              indent=2, ensure_ascii=False)

    print(f"🎉 Split complete → {save_dir}")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-sec", type=int, default=2)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse()
    split_video(args.video, args.out, args.clip_sec)
