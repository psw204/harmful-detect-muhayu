# -*- coding: utf-8 -*-
"""
무하유 팀원용 데이터 라벨링 도구 (Headless 버전)
- GUI 없이 자동 YOLO 라벨링
- 비디오만 처리
"""

import os, json, cv2
from pathlib import Path
from ultralytics import YOLO

TEAM_MEMBER_NAME = "안지산"

# 전처리 결과 폴더
VIDEO_DIR = "../../무하유_유해콘텐츠_데이터/4_전처리_결과(개인)/안지산/팀원_전처리/clips/"
# 결과 저장 폴더
OUTPUT_DIR = "../../무하유_유해콘텐츠_데이터/3_라벨링_파일(개인)/안지산/라벨_결과/"

# YOLO에서 감지할 유해 객체
HARMFUL_OBJECTS = [
    'knife', 'gun', 'pistol', 'rifle', 'sword', 'axe', 'baseball bat',
    'hammer', 'weapon', 'bottle', 'wine glass', 'beer', 'cup',
    'cigarette', 'scissors', 'fork'
]

def yolo_label_videos(video_dir, yolo_model):
    """비디오에 대해 YOLOv8로 자동 탐지"""
    video_labels = {}
    files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"[INFO] 비디오 {len(files)}개 라벨링 시작")

    for idx, filename in enumerate(files, start=1):
        filepath = os.path.join(video_dir, filename)
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(int(fps), 1)
        sample_frames = min(10, total_frames // frame_interval)

        frame_detections = []
        all_objects = set()

        for i in range(sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
            ret, frame = cap.read()
            if not ret: break
            results = yolo_model(frame, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = result.names[int(box.cls)].lower()
                        for harmful_obj in HARMFUL_OBJECTS:
                            if harmful_obj in cls or cls in harmful_obj:
                                frame_detections.append({
                                    'frame': i * frame_interval,
                                    'object': harmful_obj,
                                    'conf': float(box.conf)
                                })
                                all_objects.add(harmful_obj)
                                break
        cap.release()
        video_labels[filename] = {
            'duration': total_frames / fps if fps > 0 else 0,
            'fps': fps,
            'frames': total_frames,
            'detections': frame_detections,
            'detected_objects': list(all_objects),
            'is_harmful': len(frame_detections) > 0
        }
        print(f"[{idx}/{len(files)}] {filename} → {len(frame_detections)}개 탐지")

    print(f"✓ 비디오 {len(video_labels)}개 자동 라벨링 완료\n")
    return video_labels

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\n무하유 Headless 라벨링 - {TEAM_MEMBER_NAME}")
    yolo = YOLO('yolov8n.pt')

    results = yolo_label_videos(VIDEO_DIR, yolo)

    out_file = os.path.join(OUTPUT_DIR, "verified_video_labels.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"결과 저장 완료 → {out_file}")

if __name__ == "__main__":
    main()
