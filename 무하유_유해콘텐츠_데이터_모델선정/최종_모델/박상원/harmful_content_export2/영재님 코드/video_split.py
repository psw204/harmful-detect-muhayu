

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, math, cv2, argparse
from tqdm import tqdm

def split_video(video_path, output_dir, clip_len=2):
    # 기존 프레임 폴더 완전 삭제
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (total_frames / fps) if fps else 0.0

    clips = []
    total_saved = 0

    if fps == 0.0 or total_frames == 0:
        meta = {"video": video_path, "clips": 0, "total_frames_saved": 0, "fps": float(fps)}
        json.dump({"meta": meta, "clips": []}, open(os.path.join(output_dir, "meta.json"), "w"),
                  indent=2, ensure_ascii=False)
        print("⚠️ No frames or FPS=0.")
        return

    clip_count = int(math.ceil(duration / clip_len))

    for i in tqdm(range(clip_count), desc="Splitting"):
        start_t = i * clip_len
        end_t = min((i + 1) * clip_len, duration)
        start = int(start_t * fps)
        end   = int(end_t * fps)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = 0
        for f in range(start, end):
            ret, frame = cap.read()
            if not ret: break
            path = os.path.join(output_dir, f"clip_{i:03d}_frame_{idx:03d}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)
            idx += 1

        total_saved += len(frames)
        clips.append({"index": i, "start_sec": float(start_t), "end_sec": float(end_t), "frames": frames})

    cap.release()
    meta = {"video": video_path, "clips": len(clips), "total_frames_saved": total_saved, "fps": float(fps)}
    json.dump({"meta": meta, "clips": clips}, open(os.path.join(output_dir, "meta.json"), "w"),
              indent=2, ensure_ascii=False)
    print("✅ Split complete:", meta)

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-sec", type=int, default=2)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse()
    split_video(args.video, args.out, args.clip-sec if hasattr(args, "clip-sec") else args.clip_sec)
