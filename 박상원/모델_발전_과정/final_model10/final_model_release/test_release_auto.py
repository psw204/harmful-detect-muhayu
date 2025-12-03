"""
Release 모델 자동 평가 스크립트
final_model_release를 사용하여 수집 데이터 평가
"""

import sys
import os
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import clip
from ultralytics import YOLO
from pytorchvideo.models.hub import slowfast_r50
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import time

# Release 모델 경로 추가
RELEASE_DIR = Path(__file__).parent
sys.path.insert(0, str(RELEASE_DIR))

# Release 모델 import
from final_model_release import (
    HarmfulImageClassifier, HarmfulVideoClassifier,
    detect_behavior_with_clip_fast,
    HARMFUL_OBJECTS, HARMFUL_BEHAVIORS, BEHAVIOR_PROMPTS,
    CONTEXTUAL_OBJECTS, ALL_OBJECTS,
    DEVICE, FRAME_SAMPLE, CLIP_MODEL_NAME,
    YOLO_DIM, CLIP_DIM, BEHAVIOR_DIM, SLOWFAST_DIM
)

# Config import
from harmful_content_release.config import (
    IMAGE_MODEL_PATH, VIDEO_MODEL_PATH,
    LABELS_FILE, IMAGE_DIR, SAFE_IMAGE_DIR,
    VIDEO_DIR, SAFE_VIDEO_DIR
)

# ============================================================
# 모델 로드
# ============================================================

def load_models():
    """모델 로드"""
    print("모델 로딩...")
    
    # YOLO 모델
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.to(DEVICE)
    
    # CLIP 모델
    clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    clip_model.eval()
    
    # SlowFast 모델 (비디오용)
    slowfast_model = slowfast_r50(pretrained=True)
    slowfast_model.eval()
    slowfast_model.to(DEVICE)
    
    # 이미지 분류 모델
    image_model = HarmfulImageClassifier(
        yolo_dim=YOLO_DIM,
        clip_dim=CLIP_DIM,
        behavior_dim=BEHAVIOR_DIM
    )
    checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=DEVICE)
    image_model.load_state_dict(checkpoint['model_state_dict'])
    image_model.eval()
    image_model.to(DEVICE)
    image_threshold = checkpoint.get('best_threshold', 0.5)
    
    # 비디오 분류 모델
    video_model = HarmfulVideoClassifier(
        yolo_dim=YOLO_DIM,
        clip_dim=CLIP_DIM,
        slowfast_dim=SLOWFAST_DIM,
        behavior_dim=BEHAVIOR_DIM
    )
    checkpoint = torch.load(VIDEO_MODEL_PATH, map_location=DEVICE)
    video_model.load_state_dict(checkpoint['model_state_dict'])
    video_model.eval()
    video_model.to(DEVICE)
    video_threshold = checkpoint.get('best_threshold', 0.5)
    
    print(f"✓ 모델 로드 완료")
    print(f"  이미지 threshold: {image_threshold:.4f}")
    print(f"  비디오 threshold: {video_threshold:.4f}")
    
    return {
        'yolo': yolo_model,
        'clip': clip_model,
        'clip_preprocess': clip_preprocess,
        'slowfast': slowfast_model,
        'image_model': image_model,
        'video_model': video_model,
        'image_threshold': image_threshold,
        'video_threshold': video_threshold
    }

# ============================================================
# 특징 추출 함수
# ============================================================

def extract_yolo_features(image, yolo_model):
    """YOLO 특징 추출"""
    results = yolo_model(image, verbose=False)
    
    # 객체 카운트 벡터 (19차원)
    feature_vector = torch.zeros(len(ALL_OBJECTS), device=DEVICE)
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # YOLO 클래스명 가져오기
                class_name = yolo_model.names[cls_id]
                
                # ALL_OBJECTS에 있는지 확인
                if class_name in ALL_OBJECTS:
                    idx = ALL_OBJECTS.index(class_name)
                    feature_vector[idx] += conf
    
    return feature_vector

def extract_clip_features(image, clip_model, clip_preprocess):
    """CLIP 특징 추출"""
    image_tensor = clip_preprocess(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features = F.normalize(image_features, p=2, dim=-1)
    
    return image_features.squeeze(0)

def extract_slowfast_features(frames, slowfast_model):
    """SlowFast 특징 추출"""
    # 프레임을 SlowFast 입력 형식으로 변환
    # (T, H, W, C) -> (1, T, C, H, W)
    frames_array = np.array([np.array(frame) for frame in frames])
    
    # 정규화 및 텐서 변환
    frames_tensor = torch.from_numpy(frames_array).float()
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
    frames_tensor = frames_tensor / 255.0
    frames_tensor = frames_tensor.unsqueeze(0)  # (1, T, C, H, W)
    frames_tensor = frames_tensor.to(DEVICE)
    
    with torch.no_grad():
        # SlowFast는 (B, T, C, H, W) 형식 필요
        features = slowfast_model(frames_tensor)
        if isinstance(features, tuple):
            features = features[0]  # 첫 번째 출력 사용
    
    return features.squeeze(0)  # (400,)

# ============================================================
# 예측 함수
# ============================================================

def predict_image_release(image_path, models):
    """Release 모델로 이미지 예측 (순수 모델만)"""
    try:
        image = Image.open(image_path).convert('RGB')
        
        # 특징 추출
        yolo_features = extract_yolo_features(image, models['yolo'])
        clip_features = extract_clip_features(image, models['clip'], models['clip_preprocess'])
        behavior_features = detect_behavior_with_clip_fast(
            image, models['clip'], models['clip_preprocess']
        )
        
        # 특징 결합
        combined = torch.cat([yolo_features, clip_features, behavior_features]).unsqueeze(0)
        
        # 모델 예측
        with torch.no_grad():
            confidence = models['image_model'](combined).item()
        
        # 순수 모델: threshold만 사용
        is_harmful = confidence > models['image_threshold']
        
        return {
            'is_harmful': is_harmful,
            'confidence': confidence,
            'threshold': models['image_threshold']
        }
    except Exception as e:
        print(f"  [오류] {image_path}: {e}")
        return None

def predict_video_release(video_path, models):
    """Release 모델로 비디오 예측 (순수 모델만)"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # 프레임 샘플링 (32개)
        sample_indices = np.linspace(0, frame_count - 1, FRAME_SAMPLE, dtype=int)
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # 각 프레임별 특징 추출
        yolo_features_seq = []
        clip_features_seq = []
        behavior_features_seq = []
        
        for frame in frames:
            yolo_feat = extract_yolo_features(frame, models['yolo'])
            clip_feat = extract_clip_features(frame, models['clip'], models['clip_preprocess'])
            behavior_feat = detect_behavior_with_clip_fast(
                frame, models['clip'], models['clip_preprocess']
            )
            
            yolo_features_seq.append(yolo_feat)
            clip_features_seq.append(clip_feat)
            behavior_features_seq.append(behavior_feat)
        
        # 시퀀스로 변환
        yolo_features_seq = torch.stack(yolo_features_seq)  # (32, 19)
        clip_features_seq = torch.stack(clip_features_seq)  # (32, 512)
        behavior_features_seq = torch.stack(behavior_features_seq)  # (32, 7)
        
        # SlowFast 특징 추출
        slowfast_features = extract_slowfast_features(frames, models['slowfast'])
        slowfast_features_seq = slowfast_features.unsqueeze(0).expand(FRAME_SAMPLE, -1)  # (32, 400)
        
        # 특징 결합
        combined = torch.cat([
            yolo_features_seq,
            clip_features_seq,
            slowfast_features_seq,
            behavior_features_seq
        ], dim=1).unsqueeze(0)  # (1, 32, 938)
        
        # 모델 예측
        with torch.no_grad():
            confidence = models['video_model'](combined).item()
        
        # 순수 모델: threshold만 사용
        is_harmful = confidence > models['video_threshold']
        
        return {
            'is_harmful': is_harmful,
            'confidence': confidence,
            'threshold': models['video_threshold']
        }
    except Exception as e:
        print(f"  [오류] {video_path}: {e}")
        return None

# ============================================================
# 평가 함수
# ============================================================

def evaluate_images(models, image_files):
    """이미지 평가"""
    print("\n" + "=" * 60)
    print("이미지 테스트")
    print("=" * 60)
    
    results = []
    errors = []
    
    for filename, label_info in tqdm(image_files.items(), desc="이미지 처리"):
        label = label_info['label']
        
        # 파일 경로 찾기
        file_path = None
        if label == 1:
            file_path = IMAGE_DIR / filename
        else:
            file_path = SAFE_IMAGE_DIR / filename
        
        if not file_path.exists():
            errors.append(f"{filename}: 파일 없음")
            continue
        
        start_time = time.time()
        pred_result = predict_image_release(file_path, models)
        elapsed = time.time() - start_time
        
        if pred_result is None:
            errors.append(f"{filename}: 예측 실패")
            continue
        
        results.append({
            'filename': filename,
            'true_label': label,
            'pred_label': 1 if pred_result['is_harmful'] else 0,
            'confidence': pred_result['confidence'],
            'elapsed': elapsed
        })
    
    return results, errors

def evaluate_videos(models, video_files):
    """비디오 평가"""
    print("\n" + "=" * 60)
    print("비디오 테스트")
    print("=" * 60)
    
    results = []
    errors = []
    
    for filename, label_info in tqdm(video_files.items(), desc="비디오 처리"):
        label = label_info['label']
        
        # 파일 경로 찾기
        file_path = None
        if label == 1:
            file_path = VIDEO_DIR / filename
        else:
            file_path = SAFE_VIDEO_DIR / filename
        
        if not file_path.exists():
            errors.append(f"{filename}: 파일 없음")
            continue
        
        start_time = time.time()
        pred_result = predict_video_release(file_path, models)
        elapsed = time.time() - start_time
        
        if pred_result is None:
            errors.append(f"{filename}: 예측 실패")
            continue
        
        results.append({
            'filename': filename,
            'true_label': label,
            'pred_label': 1 if pred_result['is_harmful'] else 0,
            'confidence': pred_result['confidence'],
            'elapsed': elapsed
        })
    
    return results, errors

def calculate_metrics(results):
    """성능 지표 계산"""
    if len(results) == 0:
        return None
    
    y_true = [r['true_label'] for r in results]
    y_pred = [r['pred_label'] for r in results]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    avg_time = np.mean([r['elapsed'] for r in results])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'avg_time': avg_time
    }

# ============================================================
# 메인 함수
# ============================================================

def main():
    print("=" * 60)
    print("Release 모델 자동 평가 시작")
    print("=" * 60)
    
    # 라벨 파일 로드
    print("라벨 파일 로딩...")
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    image_files = {}
    video_files = {}
    
    for filename, label_info in labels_data.items():
        file_type = label_info.get('type', 'unknown')
        label = 1 if label_info.get('category') == 'harmful' else 0
        
        if file_type == 'image':
            image_files[filename] = {'label': label, **label_info}
        elif file_type == 'video':
            video_files[filename] = {'label': label, **label_info}
    
    print(f"✓ 총 {len(image_files) + len(video_files)}개 파일")
    print(f"  - 이미지: {len(image_files)}개")
    print(f"  - 비디오: {len(video_files)}개")
    
    # 모델 로드
    models = load_models()
    
    # 이미지 평가
    image_results, image_errors = evaluate_images(models, image_files)
    
    # 비디오 평가
    video_results, video_errors = evaluate_videos(models, video_files)
    
    # 결과 계산
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    
    if image_results:
        img_metrics = calculate_metrics(image_results)
        print("\n[이미지 성능]")
        print(f"정확도 (Accuracy): {img_metrics['accuracy']:.4f} ({img_metrics['accuracy']*100:.2f}%)")
        print(f"정밀도 (Precision): {img_metrics['precision']:.4f}")
        print(f"재현율 (Recall): {img_metrics['recall']:.4f}")
        print(f"F1-Score: {img_metrics['f1']:.4f}")
        print(f"TP: {img_metrics['tp']}, TN: {img_metrics['tn']}, FP: {img_metrics['fp']}, FN: {img_metrics['fn']}")
        print(f"평균 처리 시간: {img_metrics['avg_time']:.2f}초")
    
    if video_results:
        vid_metrics = calculate_metrics(video_results)
        print("\n[비디오 성능]")
        print(f"정확도 (Accuracy): {vid_metrics['accuracy']:.4f} ({vid_metrics['accuracy']*100:.2f}%)")
        print(f"정밀도 (Precision): {vid_metrics['precision']:.4f}")
        print(f"재현율 (Recall): {vid_metrics['recall']:.4f}")
        print(f"F1-Score: {vid_metrics['f1']:.4f}")
        print(f"TP: {vid_metrics['tp']}, TN: {vid_metrics['tn']}, FP: {vid_metrics['fp']}, FN: {vid_metrics['fn']}")
        print(f"평균 처리 시간: {vid_metrics['avg_time']:.2f}초")
    
    # 전체 성능
    all_results = image_results + video_results
    if all_results:
        all_metrics = calculate_metrics(all_results)
        print("\n[전체 성능]")
        print(f"정확도 (Accuracy): {all_metrics['accuracy']:.4f} ({all_metrics['accuracy']*100:.2f}%)")
        print(f"정밀도 (Precision): {all_metrics['precision']:.4f}")
        print(f"재현율 (Recall): {all_metrics['recall']:.4f}")
        print(f"F1-Score: {all_metrics['f1']:.4f}")
        print(f"TP: {all_metrics['tp']}, TN: {all_metrics['tn']}, FP: {all_metrics['fp']}, FN: {all_metrics['fn']}")
    
    # 오류 출력
    all_errors = image_errors + video_errors
    if all_errors:
        print(f"\n[오류] 총 {len(all_errors)}개")
        for err in all_errors[:10]:  # 처음 10개만 출력
            print(f"  - {err}")
        if len(all_errors) > 10:
            print(f"  ... 외 {len(all_errors) - 10}개")
    
    # 결과 저장
    output_file = RELEASE_DIR / "test_release_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'image_results': image_results,
            'video_results': video_results,
            'image_metrics': img_metrics if image_results else None,
            'video_metrics': vid_metrics if video_results else None,
            'overall_metrics': all_metrics if all_results else None,
            'errors': all_errors
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 결과 저장: {output_file}")

if __name__ == "__main__":
    main()

