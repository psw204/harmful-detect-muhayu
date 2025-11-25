# -*- coding: utf-8 -*-
"""
무하유 팀원용 데이터 라벨링 도구
=================================

이 스크립트는 팀원들이 수집한 이미지/비디오 데이터를 동일한 방식으로 라벨링하는 도구입니다.

사용 방법:
1. 수집한 데이터를 지정된 폴더에 저장
2. 하단의 설정 부분에서 폴더 경로 수정
3. python team_labeling_tool.py 실행
4. 화면에 나타나는 이미지/비디오를 보고 Y(유해) / N(안전) 선택

출력 파일:
  - verified_labels.json: 검증된 유해 이미지 라벨
  - safe_labels.json: 안전 이미지 라벨  
  - verified_video_labels.json: 검증된 유해 비디오 라벨
  - safe_video_labels.json: 안전 비디오 라벨

주의사항:
  - ultralytics (YOLOv8) 설치 필요: pip install ultralytics
  - opencv-python 설치 필요: pip install opencv-python
"""

# ============================================================
# 필수 라이브러리 Import
# ============================================================
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

# ============================================================
# ⚙️ 설정 부분 - 팀원이 수정해야 하는 부분
# ============================================================
# 팀원 이름 (출력 파일 구분용)
TEAM_MEMBER_NAME = "임영재"  # 예: "안지산", "임영재" 등으로 수정

# 데이터 폴더 경로 설정
BASE_PATH = './팀원_데이터/'  # 데이터 저장 기본 경로

# 하위 폴더 구조
HARMFUL_IMAGE_DIR = BASE_PATH + '이미지/'          # 유해 이미지 폴더
SAFE_IMAGE_DIR = BASE_PATH + '안전_이미지/'        # 안전 이미지 폴더
HARMFUL_VIDEO_DIR = BASE_PATH + '비디오/'          # 유해 비디오 폴더
SAFE_VIDEO_DIR = BASE_PATH + '안전_비디오/'        # 안전 비디오 폴더
OUTPUT_DIR = BASE_PATH + '라벨_결과/'              # 라벨 파일 출력 폴더

# 유해 객체 목록 (YOLO 탐지용)
HARMFUL_OBJECTS = [
    'knife', 'gun', 'pistol', 'rifle', 'sword', 'axe', 'baseball bat',
    'hammer', 'weapon', 'bottle', 'wine glass', 'beer', 'cup',
    'cigarette', 'scissors', 'fork'
]

# ============================================================
# 폴더 생성
# ============================================================
def create_folders():
    """필요한 폴더 생성"""
    folders = [
        HARMFUL_IMAGE_DIR,
        SAFE_IMAGE_DIR,
        HARMFUL_VIDEO_DIR,
        SAFE_VIDEO_DIR,
        OUTPUT_DIR
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    print("✓ 폴더 구조 확인 완료\n")


# ============================================================
# YOLO 이미지 자동 라벨링
# ============================================================
def yolo_label_images(image_dir, yolo_model):
    """
    이미지에 대해 YOLOv8로 유해 객체 자동 탐지
    
    Args:
        image_dir: 이미지 폴더 경로
        yolo_model: YOLO 모델 객체
        
    Returns:
        dict: {filename: [detections]} 형식의 라벨 딕셔너리
    """
    labels = {}
    
    if not os.path.exists(image_dir):
        return labels
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        return labels
    
    print(f"이미지 YOLO 라벨링 중... ({len(image_files)}개)")
    
    for filename in image_files:
        filepath = os.path.join(image_dir, filename)
        
        # YOLO로 객체 탐지
        results = yolo_model(filepath, verbose=False)
        detections = []
        
        # 탐지 결과 분석
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_name = result.names[int(box.cls)].lower()
                    is_harmful = False
                    matched_class = class_name
                    
                    # 유해 객체 목록과 매칭
                    for harmful_obj in HARMFUL_OBJECTS:
                        if harmful_obj in class_name or class_name in harmful_obj:
                            is_harmful = True
                            matched_class = harmful_obj
                            break
                    
                    # 유해 객체인 경우 상세 정보 저장
                    if is_harmful:
                        detections.append({
                            'class': matched_class,
                            'detected_as': class_name,
                            'confidence': float(box.conf),
                            'bbox': box.xyxy[0].tolist()
                        })
        
        labels[filename] = detections
    
    print(f"✓ {len(labels)}개 이미지 라벨링 완료\n")
    return labels


# ============================================================
# YOLO 비디오 자동 라벨링
# ============================================================
def yolo_label_videos(video_dir, yolo_model):
    """
    비디오에 대해 YOLOv8로 유해 객체 자동 탐지 (프레임 샘플링)
    
    Args:
        video_dir: 비디오 폴더 경로
        yolo_model: YOLO 모델 객체
        
    Returns:
        dict: {filename: {metadata}} 형식의 라벨 딕셔너리
    """
    # 파일명 키워드로 행동 추정
    action_keywords = {
        'violence': 'fighting', 'assault': 'attacking', 'attack': 'attacking',
        'weapon': 'threatening', 'gun': 'shooting', 'knife': 'stabbing',
        'drunk': 'intoxicated', 'drug': 'substance_abuse', 'sexy': 'sexual_content',
        'fight': 'fighting', 'blood': 'injury'
    }
    
    video_labels = {}
    
    if not os.path.exists(video_dir):
        return video_labels
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    if not video_files:
        return video_labels
    
    print(f"비디오 YOLO 라벨링 중... ({len(video_files)}개)")
    
    for filename in video_files:
        filepath = os.path.join(video_dir, filename)
        
        # 비디오 파일 열기
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 프레임 샘플링 간격 (최대 10프레임)
        frame_interval = max(int(fps), 1)
        sample_frames = min(10, total_frames // frame_interval)
        
        # 탐지 결과 저장
        frame_detections = []
        all_objects = set()
        
        # 샘플링된 프레임마다 YOLO 탐지
        for i in range(sample_frames):
            frame_idx = i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO로 객체 탐지
            results = yolo_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_name = result.names[int(box.cls)].lower()
                        
                        # 유해 객체 매칭
                        for harmful_obj in HARMFUL_OBJECTS:
                            if harmful_obj in class_name or class_name in harmful_obj:
                                frame_detections.append({
                                    'frame': frame_idx,
                                    'time': frame_idx / fps if fps > 0 else 0,
                                    'object': harmful_obj,
                                    'confidence': float(box.conf),
                                    'bbox': box.xyxy[0].tolist()
                                })
                                all_objects.add(harmful_obj)
                                break
        
        cap.release()
        
        # 파일명에서 행동 키워드 추정
        estimated_action = 'unknown'
        for keyword, action in action_keywords.items():
            if keyword in filename.lower():
                estimated_action = action
                break
        
        # 비디오 라벨 정보 저장
        video_labels[filename] = {
            'duration': total_frames / fps if fps > 0 else 0,
            'fps': fps,
            'total_frames': total_frames,
            'sampled_frames': sample_frames,
            'detected_objects': list(all_objects),
            'total_detections': len(frame_detections),
            'frame_detections': frame_detections,
            'estimated_action': estimated_action,
            'is_harmful': len(frame_detections) > 0 or estimated_action != 'unknown'
        }
    
    print(f"✓ {len(video_labels)}개 비디오 라벨링 완료\n")
    return video_labels


# ============================================================
# 이미지 검증 인터페이스
# ============================================================
def verify_images(image_dir, auto_labels):
    """
    이미지 직접 검증 (OpenCV GUI)
    
    Args:
        image_dir: 이미지 폴더 경로
        auto_labels: YOLO 자동 라벨 딕셔너리
        
    Returns:
        dict: 검증된 라벨 딕셔너리 {filename: [detections]}
    """
    verified = {}
    
    if not auto_labels:
        print("⚠️ 검증할 이미지가 없습니다.\n")
        return verified
    
    print("="*60)
    print("이미지 검증 시작")
    print("="*60)
    print("Y 또는 Space: 유해 콘텐츠로 승인")
    print("N: 안전 콘텐츠 (거절)")
    print("Q: 검증 중단 및 종료")
    print("="*60 + "\n")
    
    files_to_verify = list(auto_labels.keys())
    
    for idx, filename in enumerate(files_to_verify):
        detections = auto_labels[filename]
        img_path = os.path.join(image_dir, filename)
        
        # 이미지 로드 (한글 경로 처리)
        img_array = cv2.imdecode(
            np.fromfile(img_path, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if img_array is None:
            continue
        
        # 이미지 복사
        display_img = img_array.copy()
        
        # 탐지된 객체에 바운딩 박스 그리기
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # 바운딩 박스 (녹색)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 라벨 텍스트
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(display_img, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 진행 상황 표시
        progress = f"[{idx+1}/{len(files_to_verify)}] {filename}"
        cv2.putText(display_img, progress, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 이미지 크기 조정
        h, w = display_img.shape[:2]
        if w > 1200:
            scale = 1200 / w
            display_img = cv2.resize(display_img, (int(w*scale), int(h*scale)))
        
        # 이미지 표시
        cv2.imshow('Verify - Y:Harmful N:Safe Q:Quit', display_img)
        
        # 콘솔 출력
        print(f"[{idx+1}/{len(files_to_verify)}] {filename}")
        print(f"  탐지 객체: {len(detections)}개")
        if detections:
            print(f"  객체 종류: {', '.join([d['class'] for d in detections[:3]])}")
        
        # 사용자 입력 대기
        key = cv2.waitKey(0) & 0xFF
        
        if key in [ord('y'), ord('Y'), 32]:  # Y 또는 Space: 유해
            verified[filename] = detections
            print("  → 유해 콘텐츠로 승인\n")
        elif key in [ord('n'), ord('N')]:  # N: 안전
            print("  → 안전 콘텐츠로 분류 (제외)\n")
        elif key in [ord('q'), ord('Q')]:  # Q: 종료
            print("\n검증 중단...")
            cv2.destroyAllWindows()
            break
        else:  # 기타: 기본 승인
            verified[filename] = detections
            print("  → 유해 콘텐츠로 승인\n")
        
        cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()
    print(f"\n✓ 검증 완료: {len(verified)}개 유해 이미지 승인\n")
    return verified


# ============================================================
# 비디오 검증 인터페이스
# ============================================================
def verify_videos(video_dir, video_labels):
    """
    비디오 직접 검증 (OpenCV GUI)
    
    Args:
        video_dir: 비디오 폴더 경로
        video_labels: YOLO 자동 라벨 딕셔너리
        
    Returns:
        dict: 검증된 라벨 딕셔너리 {filename: {metadata}}
    """
    verified = {}
    
    if not video_labels:
        print("⚠️ 검증할 비디오가 없습니다.\n")
        return verified
    
    print("="*60)
    print("비디오 검증 시작")
    print("="*60)
    print("Y 또는 Space: 유해 콘텐츠로 승인")
    print("N: 안전 콘텐츠 (거절)")
    print("Q: 검증 중단 및 종료")
    print("="*60 + "\n")
    
    files_to_verify = list(video_labels.keys())
    
    for idx, filename in enumerate(files_to_verify):
        label = video_labels[filename]
        video_path = os.path.join(video_dir, filename)
        
        # 비디오 파일 열기
        cap = cv2.VideoCapture(video_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 중간 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames // 2 if frames > 0 else 0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            continue
        
        # 프레임에 정보 오버레이
        cv2.putText(frame, f"[{idx+1}/{len(files_to_verify)}] {filename}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, f"Action: {label.get('estimated_action', 'unknown')}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        objects_str = ','.join(label.get('detected_objects', [])[:3])
        cv2.putText(frame, f"Objects: {label.get('total_detections', 0)} - {objects_str}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 프레임 크기 조정
        h, w = frame.shape[:2]
        if w > 1000:
            frame = cv2.resize(frame, (1000, int(h * 1000 / w)))
        
        # 프레임 표시
        cv2.imshow("Verify - Y:Harmful N:Safe Q:Quit", frame)
        
        # 콘솔 출력
        print(f"[{idx+1}/{len(files_to_verify)}] {filename}")
        print(f"  탐지 객체: {label.get('total_detections', 0)}개")
        print(f"  추정 행동: {label.get('estimated_action', 'unknown')}")
        
        # 사용자 입력 대기
        key = cv2.waitKey(0) & 0xFF
        
        if key in [ord('y'), ord('Y'), 32]:  # Y 또는 Space: 유해
            label['is_harmful'] = True
            verified[filename] = label
            print("  → 유해 콘텐츠로 승인\n")
        elif key in [ord('n'), ord('N')]:  # N: 안전
            label['is_harmful'] = False
            print("  → 안전 콘텐츠로 분류 (제외)\n")
        elif key in [ord('q'), ord('Q')]:  # Q: 종료
            print("\n검증 중단...")
            cv2.destroyAllWindows()
            break
        else:  # 기타: 기본 승인
            label['is_harmful'] = True
            verified[filename] = label
            print("  → 유해 콘텐츠로 승인\n")
        
        cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()
    print(f"\n✓ 검증 완료: {len(verified)}개 유해 비디오 승인\n")
    return verified


# ============================================================
# 안전 이미지 자동 라벨링
# ============================================================
def label_safe_images(image_dir):
    """
    안전 이미지 자동 라벨링 (검증 불필요)
    
    Args:
        image_dir: 안전 이미지 폴더 경로
        
    Returns:
        dict: {filename: {is_safe, label, category}}
    """
    labels = {}
    
    if not os.path.exists(image_dir):
        return labels
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    for filename in image_files:
        labels[filename] = {
            'is_safe': True,
            'label': 0,
            'category': 'safe'
        }
    
    if labels:
        print(f"✓ 안전 이미지 {len(labels)}개 자동 라벨링 완료\n")
    
    return labels


# ============================================================
# 안전 비디오 자동 라벨링
# ============================================================
def label_safe_videos(video_dir):
    """
    안전 비디오 자동 라벨링 (검증 불필요)
    
    Args:
        video_dir: 안전 비디오 폴더 경로
        
    Returns:
        dict: {filename: {is_safe, label, category, duration, fps, total_frames}}
    """
    video_labels = {}
    
    if not os.path.exists(video_dir):
        return video_labels
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    for filename in video_files:
        filepath = os.path.join(video_dir, filename)
        
        try:
            # 비디오 메타데이터 추출
            cap = cv2.VideoCapture(filepath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            video_labels[filename] = {
                'is_safe': True,
                'label': 0,
                'category': 'safe',
                'duration': duration,
                'fps': fps,
                'total_frames': total_frames
            }
        except:
            pass
    
    if video_labels:
        print(f"✓ 안전 비디오 {len(video_labels)}개 자동 라벨링 완료\n")
    
    return video_labels


# ============================================================
# 메인 실행 함수
# ============================================================
def main():
    """
    전체 라벨링 프로세스 실행
    """
    print("\n" + "="*60)
    print(f"무하유 팀원용 데이터 라벨링 도구 - {TEAM_MEMBER_NAME}")
    print("="*60 + "\n")
    
    # 1. 폴더 생성
    create_folders()
    
    # 2. YOLO 모델 로딩
    print("YOLO 모델 로딩 중...")
    try:
        from ultralytics import YOLO
        yolo = YOLO('yolov8n.pt')
        print("✓ YOLO 모델 로딩 완료\n")
    except ImportError:
        print("✗ ultralytics 설치 필요: pip install ultralytics")
        input("\nEnter를 눌러 종료...")
        return
    except Exception as e:
        print(f"✗ YOLO 모델 로딩 실패: {e}")
        input("\nEnter를 눌러 종료...")
        return
    
    # 3. 유해 이미지 라벨링
    print("\n" + "="*60)
    print("1. 유해 이미지 라벨링")
    print("="*60 + "\n")
    
    harmful_image_labels = yolo_label_images(HARMFUL_IMAGE_DIR, yolo)
    verified_image_labels = verify_images(HARMFUL_IMAGE_DIR, harmful_image_labels)
    
    # 4. 안전 이미지 라벨링
    print("\n" + "="*60)
    print("2. 안전 이미지 라벨링")
    print("="*60 + "\n")
    
    safe_image_labels = label_safe_images(SAFE_IMAGE_DIR)
    
    # 5. 유해 비디오 라벨링
    print("\n" + "="*60)
    print("3. 유해 비디오 라벨링")
    print("="*60 + "\n")
    
    harmful_video_labels = yolo_label_videos(HARMFUL_VIDEO_DIR, yolo)
    verified_video_labels = verify_videos(HARMFUL_VIDEO_DIR, harmful_video_labels)
    
    # 6. 안전 비디오 라벨링
    print("\n" + "="*60)
    print("4. 안전 비디오 라벨링")
    print("="*60 + "\n")
    
    safe_video_labels = label_safe_videos(SAFE_VIDEO_DIR)
    
    # 7. 결과 저장
    print("\n" + "="*60)
    print("결과 저장 중...")
    print("="*60 + "\n")
    
    # verified_labels.json
    output_file = os.path.join(OUTPUT_DIR, 'verified_labels.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(verified_image_labels, f, indent=2, ensure_ascii=False)
    print(f"✓ {output_file}")
    
    # safe_labels.json
    output_file = os.path.join(OUTPUT_DIR, 'safe_labels.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(safe_image_labels, f, indent=2, ensure_ascii=False)
    print(f"✓ {output_file}")
    
    # verified_video_labels.json
    output_file = os.path.join(OUTPUT_DIR, 'verified_video_labels.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(verified_video_labels, f, indent=2, ensure_ascii=False)
    print(f"✓ {output_file}")
    
    # safe_video_labels.json
    output_file = os.path.join(OUTPUT_DIR, 'safe_video_labels.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(safe_video_labels, f, indent=2, ensure_ascii=False)
    print(f"✓ {output_file}")
    
    # 8. 최종 요약
    print("\n" + "="*60)
    print("라벨링 완료!")
    print("="*60)
    print(f"팀원: {TEAM_MEMBER_NAME}")
    print(f"유해 이미지: {len(verified_image_labels)}개")
    print(f"안전 이미지: {len(safe_image_labels)}개")
    print(f"유해 비디오: {len(verified_video_labels)}개")
    print(f"안전 비디오: {len(safe_video_labels)}개")
    print(f"\n출력 폴더: {os.path.abspath(OUTPUT_DIR)}")
    print("="*60 + "\n")


# ============================================================
# 스크립트 실행
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nEnter를 눌러 종료...")

