"""
추론 함수 - Final Model 11 기반 (카테고리 구조)
이미지 및 비디오에 대한 유해 콘텐츠 탐지 추론
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import clip
from ultralytics import YOLO
from pytorchvideo.models.hub import slowfast_r50
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as T
import time

from models import (
    HARMFUL_OBJECTS, BEHAVIOR_CATEGORIES, BEHAVIOR_PROMPTS,
    CONTEXTUAL_OBJECTS, ALL_OBJECTS, OBJECT_MAP
)
from config import (
    DEVICE, FRAME_SAMPLE, CLIP_MODEL_NAME, SLOWFAST_DIM,
    YOLO_DIM, CLIP_DIM, BEHAVIOR_DIM
)

_clip_text_features_cache = None
_clip_weapon_features_cache = None

def set_clip_text_features_cache(cache):
    """CLIP 텍스트 특징 캐시 설정"""
    global _clip_text_features_cache
    _clip_text_features_cache = cache

def set_clip_weapon_features_cache(cache):
    """CLIP 무기 특징 캐시 설정"""
    global _clip_weapon_features_cache
    _clip_weapon_features_cache = cache

# 무기 감지용 prompts
WEAPON_PROMPTS = {
    'gun': "a photo of a gun",
    'firearm': "a photo of a firearm",
    'knife': "a photo of a knife",
    'blade': "a photo of a blade",
    'weapon': "a photo of a weapon"
}

def detect_weapons_with_clip(clip_features, clip_model, weapon_features_cache=None):
    """
    CLIP 기반 무기 감지 (Zero-shot)
    
    Args:
        clip_features: CLIP 특징 벡터 (512차원, 정규화됨)
        clip_model: CLIP 모델
        weapon_features_cache: 캐시된 무기 텍스트 특징
        
    Returns:
        weapon_score: 최대 무기 점수 (cosine similarity, -1~1)
        weapon_type: 감지된 무기 종류
    """
    try:
        with torch.no_grad():
            if weapon_features_cache is not None:
                text_features = weapon_features_cache
            else:
                weapon_prompts_list = list(WEAPON_PROMPTS.values())
                text_tokens = clip.tokenize(weapon_prompts_list).to(DEVICE)
                
                use_amp = DEVICE == 'cuda'
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        text_features = clip_model.encode_text(text_tokens)
                        text_features = F.normalize(text_features, p=2, dim=-1)
                else:
                    text_features = clip_model.encode_text(text_tokens)
                    text_features = F.normalize(text_features, p=2, dim=-1)
            
            image_features = clip_features.unsqueeze(0)
            
            use_amp = DEVICE == 'cuda'
            if use_amp:
                with torch.amp.autocast('cuda'):
                    similarities = (image_features @ text_features.T).squeeze()
            else:
                similarities = (image_features @ text_features.T).squeeze()
            
            similarities_np = similarities.cpu().numpy()
            max_idx = similarities_np.argmax()
            weapon_score = float(similarities_np[max_idx])
            weapon_types = list(WEAPON_PROMPTS.keys())
            weapon_type = weapon_types[max_idx]
            
            return weapon_score, weapon_type
            
    except Exception as e:
        return 0.0, None


def detect_behavior_with_clip_fast_optimized(clip_features, clip_model, text_features_cache=None):
    """
    CLIP 기반 행동 감지 (최적화 버전) - 카테고리별 여러 프롬프트 지원
    이미 추출한 CLIP 특징 재사용하여 속도 향상
    
    Args:
        clip_features: CLIP 특징 벡터 (512차원, 정규화됨)
        clip_model: CLIP 모델
        text_features_cache: 캐시된 텍스트 특징 딕셔너리 {category: features} (선택적)
        
    Returns:
        behavior_scores: 카테고리별 점수 딕셔너리 (0~1, Min-Max 정규화)
    """
    behavior_scores = {}
    
    try:
        with torch.no_grad():
            image_features = clip_features.unsqueeze(0)
            
            # 각 카테고리에 대해 여러 프롬프트의 평균 점수 계산
            for category in BEHAVIOR_CATEGORIES:
                prompts = BEHAVIOR_PROMPTS[category]
                
                if text_features_cache is not None and category in text_features_cache:
                    # 캐시된 텍스트 특징 사용
                    text_features = text_features_cache[category]
                else:
                    # 모든 프롬프트를 한 번에 인코딩
                    text_tokens = clip.tokenize(prompts).to(DEVICE)
                    
                    use_amp = DEVICE == 'cuda'
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            text_features = clip_model.encode_text(text_tokens)
                            text_features = F.normalize(text_features, p=2, dim=-1)
                    else:
                        text_features = clip_model.encode_text(text_tokens)
                        text_features = F.normalize(text_features, p=2, dim=-1)
                
                # 이미지 특징과 모든 프롬프트 특징의 유사도 계산
                use_amp = DEVICE == 'cuda'
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        similarities = (image_features @ text_features.T).squeeze()
                else:
                    similarities = (image_features @ text_features.T).squeeze()
                
                # 여러 프롬프트의 평균 점수
                if len(prompts) == 1:
                    behavior_scores[category] = similarities.item()
                else:
                    behavior_scores[category] = similarities.mean().item()
        
        # Min-Max 정규화 (카테고리 간 상대적 차이 보존)
        min_score = min(behavior_scores.values())
        max_score = max(behavior_scores.values())
        if max_score > min_score:
            for category in behavior_scores:
                behavior_scores[category] = (behavior_scores[category] - min_score) / (max_score - min_score)
    
    except Exception as e:
        behavior_scores = {category: 0.0 for category in BEHAVIOR_CATEGORIES}
    
    return behavior_scores


def detect_behavior_with_clip_fast_from_features(clip_features_seq, clip_model, text_features_cache=None):
    """
    CLIP 기반 행동 감지 (비디오용, 이미 추출한 특징 시퀀스 사용) - 카테고리별 여러 프롬프트 지원
    
    Args:
        clip_features_seq: CLIP 특징 시퀀스 (N, 512차원, 정규화됨)
        clip_model: CLIP 모델
        text_features_cache: 캐시된 텍스트 특징 딕셔너리 {category: features} (선택적)
        
    Returns:
        behavior_scores: 카테고리별 점수 딕셔너리 (0~1, Min-Max 정규화)
    """
    behavior_scores = {}
    
    try:
        with torch.no_grad():
            sample_frames = clip_features_seq[:min(len(clip_features_seq), 4)]
            
            # 각 카테고리에 대해 여러 프롬프트의 평균 점수 계산
            for category in BEHAVIOR_CATEGORIES:
                prompts = BEHAVIOR_PROMPTS[category]
                
                if text_features_cache is not None and category in text_features_cache:
                    # 캐시된 텍스트 특징 사용
                    text_features = text_features_cache[category]
                else:
                    # 모든 프롬프트를 한 번에 인코딩
                    text_tokens = clip.tokenize(prompts).to(DEVICE)
                    
                    use_amp = DEVICE == 'cuda'
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            text_features = clip_model.encode_text(text_tokens)
                            text_features = F.normalize(text_features, p=2, dim=-1)
                    else:
                        text_features = clip_model.encode_text(text_tokens)
                        text_features = F.normalize(text_features, p=2, dim=-1)
                
                # 프레임별 유사도 계산
                use_amp = DEVICE == 'cuda'
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        similarities = sample_frames @ text_features.T
                else:
                    similarities = sample_frames @ text_features.T
                
                # 여러 프롬프트의 평균 점수
                if len(prompts) == 1:
                    frame_scores = similarities[:, 0].cpu().numpy()
                else:
                    frame_scores = similarities.mean(dim=1).cpu().numpy()
                
                behavior_scores[category] = float(np.mean(frame_scores))
        
        # Min-Max 정규화
        min_score = min(behavior_scores.values())
        max_score = max(behavior_scores.values())
        if max_score > min_score:
            for category in behavior_scores:
                behavior_scores[category] = (behavior_scores[category] - min_score) / (max_score - min_score)
    
    except Exception as e:
        behavior_scores = {category: 0.0 for category in BEHAVIOR_CATEGORIES}
    
    return behavior_scores


def detect_behavior_with_clip_fast(image_or_frames, clip_model, clip_preprocess):
    """
    CLIP 기반 행동 감지 (Zero-shot Learning) - 카테고리별 여러 프롬프트 지원
    비디오에서 사용 (원본 버전)
    
    Args:
        image_or_frames: PIL Image 또는 프레임 리스트
        clip_model: CLIP 모델
        clip_preprocess: CLIP 전처리 함수
        
    Returns:
        behavior_scores: 카테고리별 점수 딕셔너리 (0~1 정규화)
    """
    behavior_scores = {}
    
    try:
        # 입력을 리스트로 변환
        if isinstance(image_or_frames, Image.Image):
            frames = [image_or_frames]
        else:
            frames = image_or_frames
        
        # 각 카테고리에 대해 CLIP 유사도 계산 (여러 프롬프트 평균)
        for category in BEHAVIOR_CATEGORIES:
            prompts = BEHAVIOR_PROMPTS[category]  # 프롬프트 리스트
            
            category_scores = []
            for frame in frames[:min(len(frames), 4)]:
                image_input = clip_preprocess(frame).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    # 이미지 특징 추출
                    image_features = clip_model.encode_image(image_input)
                    image_features = F.normalize(image_features, p=2, dim=-1)
                    
                    # 모든 프롬프트에 대한 유사도 계산
                    prompt_scores = []
                    for prompt in prompts:
                        text = clip.tokenize([prompt]).to(DEVICE)
                        text_features = clip_model.encode_text(text)
                        text_features = F.normalize(text_features, p=2, dim=-1)
                        similarity = (image_features @ text_features.T).squeeze()
                        prompt_scores.append(similarity.item())
                    
                    category_scores.append(np.mean(prompt_scores))
            
            behavior_scores[category] = np.mean(category_scores)
        
        # 점수 정규화 (0~1 범위로)
        min_score = min(behavior_scores.values())
        max_score = max(behavior_scores.values())
        if max_score > min_score:
            for category in behavior_scores:
                behavior_scores[category] = (behavior_scores[category] - min_score) / (max_score - min_score)
    
    except Exception as e:
        print(f"  [행동 감지 오류] {e}")
        # 에러 시 모든 카테고리 점수를 0으로 설정
        behavior_scores = {category: 0.0 for category in BEHAVIOR_CATEGORIES}
    
    return behavior_scores


def infer_behavior_from_objects(object_counts: Dict[str, int]) -> List[str]:
    """
    YOLO 객체 카운트 기반으로 유해 카테고리 추론 (규칙 기반 보조 로직)
    
    YOLO로 탐지된 객체를 바탕으로 특정 카테고리를 추론합니다.
    CLIP의 Zero-shot 감지를 보완하는 역할.
    
    Args:
        object_counts: 객체별 감지 개수
            예시: {"cigarette": 2, "wine glass": 1, "knife": 1, "person": 2, ...}
        
    Returns:
        list[str]: 추론된 카테고리 목록 (예: ["smoking", "alcohol"])
    """
    inferred = []
    
    # 규칙 1: 담배 감지 → 흡연
    # - lighter 단독은 흡연으로 보지 않음
    if object_counts.get("cigarette", 0) > 0:
        inferred.append("smoking")
    
    # 규칙 2: 음주 관련 객체 1개 이상 → 음주
    # - wine glass / beer 등
    alcohol_objects = OBJECT_MAP.get("alcohol", [])
    if sum(object_counts.get(obj, 0) for obj in alcohol_objects) >= 1:
        inferred.append("alcohol")
    
    # 규칙 3: 주사기 감지 → 약물
    drug_objects = OBJECT_MAP.get("drugs", [])
    if sum(object_counts.get(obj, 0) for obj in drug_objects) > 0:
        inferred.append("drugs")
    
    # 규칙 4: 혈액/상처 관련 객체 감지 → blood
    blood_objects = OBJECT_MAP.get("blood", [])
    if sum(object_counts.get(obj, 0) for obj in blood_objects) > 0:
        inferred.append("blood")
    
    # 규칙 5: 무기 + 사람 → 위협(threat)
    # - weapon 객체가 있고, person이 최소 1명 이상일 때만 위협으로 추론
    weapon_objects = OBJECT_MAP.get("weapons", [])
    weapon_count = sum(object_counts.get(obj, 0) for obj in weapon_objects)
    person_count = object_counts.get("person", 0)
    if weapon_count > 0 and person_count >= 1:
        inferred.append("threat")
    
    # 중복 제거
    inferred = list(set(inferred))
    
    return inferred


def extract_yolo_features(yolo_results) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    YOLO 탐지 결과에서 특징 벡터 추출
    
    Args:
        yolo_results: YOLO 탐지 결과 (단일 결과 또는 리스트)
    
    Returns:
        feature_vector: 객체별 탐지 개수 벡터 (20차원, ALL_OBJECTS 개수)
        object_counts: 객체별 탐지 개수 딕셔너리
    """
    feature_vector = torch.zeros(YOLO_DIM, device=DEVICE)
    object_counts = {}
    
    if not isinstance(yolo_results, list):
        yolo_results = [yolo_results]
    
    for result in yolo_results:
        if result.boxes is not None:
            for box in result.boxes:
                class_name = result.names[int(box.cls)].lower()
                for i, obj in enumerate(ALL_OBJECTS):
                    if obj in class_name or class_name in obj:
                        feature_vector[i] += 1
                        object_counts[obj] = object_counts.get(obj, 0) + 1
    
    return feature_vector, object_counts


def predict_image(image: Image.Image, image_model, yolo_model, clip_model, 
                  clip_preprocess, threshold: float = 0.4, verbose: bool = True) -> Dict:
    """
    이미지에 대한 유해 콘텐츠 탐지 (학습 시와 동일한 방식)
    
    Args:
        image: PIL Image 객체
        image_model: 학습된 이미지 분류 모델
        yolo_model: YOLO 모델
        clip_model: CLIP 모델
        clip_preprocess: CLIP 전처리 함수
        threshold: 분류 임계값
        
    Returns:
        result: 딕셔너리
            - is_harmful: bool (유해 여부)
            - confidence: float (유해 확률, 0~1)
            - detected_objects: List[str] (감지된 유해 객체)
            - detected_behaviors: List[str] (감지된 유해 행동)
    """
    try:
        if image is None:
            return {
                "is_harmful": False,
                "confidence": 0.0,
                "detected_objects": [],
                "detected_behaviors": [],
                "error": "이미지를 불러올 수 없습니다."
            }
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        yolo_results = yolo_model(image_np, verbose=False, device=DEVICE, imgsz=640, conf=0.25)
        yolo_features, object_counts = extract_yolo_features(yolo_results)
        
        detected_objects = [obj for obj in HARMFUL_OBJECTS if object_counts.get(obj, 0) > 0]
        
        clip_image = clip_preprocess(image).unsqueeze(0).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    clip_features = clip_model.encode_image(clip_image).squeeze()
            else:
                clip_features = clip_model.encode_image(clip_image).squeeze()
            clip_features = F.normalize(clip_features, p=2, dim=-1)
        
        # CLIP 무기 감지 (Zero-shot)
        weapon_score, weapon_type = detect_weapons_with_clip(
            clip_features, clip_model, weapon_features_cache=_clip_weapon_features_cache
        )
        
        behavior_scores = detect_behavior_with_clip_fast_optimized(
            clip_features, clip_model, text_features_cache=_clip_text_features_cache
        )
        inferred_categories = infer_behavior_from_objects(object_counts)
        
        behavior_features = torch.zeros(len(BEHAVIOR_CATEGORIES), device=DEVICE)
        for i, category in enumerate(BEHAVIOR_CATEGORIES):
            clip_score = behavior_scores.get(category, 0.0)
            rule_score = 1.0 if category in inferred_categories else 0.0
            behavior_features[i] = 0.6 * clip_score + 0.4 * rule_score
        
        # detected_behaviors: 규칙 기반 + 매우 높은 CLIP 점수
        # violence는 일상 동작과 혼동되어 제외
        # dangerous, sexual은 더 보수적으로 (오탐 심각)
        detected_behaviors = inferred_categories.copy()
        for category, score in behavior_scores.items():
            if category == 'violence':
                continue  # violence는 제외 (False Positive 너무 많음)
            elif category == 'dangerous' and score >= 0.99:
                # dangerous는 0.99 이상만 (거의 확실한 경우만)
                detected_behaviors.append(category)
            elif category == 'sexual' and score >= 0.98:
                # sexual은 0.98 이상만
                detected_behaviors.append(category)
            elif score >= 0.95 and category not in detected_behaviors:
                # 나머지 카테고리는 0.95 이상
                detected_behaviors.append(category)
        
        yolo_features = yolo_features.to(DEVICE)
        combined = torch.cat([yolo_features, clip_features, behavior_features]).unsqueeze(0)
        
        image_model.eval()
        with torch.no_grad():
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    confidence = image_model(combined).item()
            else:
                confidence = image_model(combined).item()
        
        # CLIP 무기 감지 (Zero-shot): 이미지는 0.3, 비디오는 0.2
        weapon_threshold = 0.3  # 이미지용
        if weapon_score >= weapon_threshold and weapon_type:
            if weapon_type not in detected_objects:
                detected_objects.append(weapon_type)
            if verbose:
                print(f"  [CLIP 무기 감지] {weapon_type}: {weapon_score:.3f}")
        
        # 무기 휴리스틱: knife + threat 또는 CLIP 무기 감지
        weapon_detected = False
        if weapon_score >= weapon_threshold:
            weapon_detected = True
            is_harmful = True
        elif 'knife' in detected_objects and 'threat' in detected_behaviors:
            weapon_detected = True
            is_harmful = True
        
        # 균형잡힌 분류 로직
        if not weapon_detected:
            if len(detected_objects) == 0 and len(detected_behaviors) == 0:
                # 아무것도 감지 안 됨 → 안전
                is_harmful = False
            else:
                # 객체나 행동이 감지됨 → threshold 사용
                # threat만 더 민감하게 반응 (threshold 30% 낮춤)
                if 'threat' in detected_behaviors:
                    adjusted_threshold = threshold * 0.7
                else:
                    adjusted_threshold = threshold
                is_harmful = confidence > adjusted_threshold
        
        return {
            "is_harmful": is_harmful,
            "confidence": confidence,
            "detected_objects": detected_objects,
            "detected_behaviors": detected_behaviors
        }
    
    except Exception as e:
        print(f"이미지 추론 오류: {e}")
        return {
            "is_harmful": False,
            "confidence": 0.0,
            "detected_objects": [],
            "detected_behaviors": [],
            "error": str(e)
        }


def extract_frames_safe(video_path: str) -> Tuple[List[Image.Image], List[torch.Tensor]]:
    """
    안전한 프레임 추출 (손상된 프레임 자동 스킵)
    
    Args:
        video_path: 비디오 파일 경로
        
    Returns:
        frames_pil: PIL Image 리스트 (32개)
        frame_tensors: 프레임 텐서 리스트 (32개)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return [], []
    
    indices = np.linspace(0, total_frames - 1, FRAME_SAMPLE, dtype=int)
    
    frames_pil = []
    frame_tensors = []
    
    for idx in indices:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frames_pil.append(frame_pil)
                
                frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                frame_tensor = T.Resize((256, 256))(frame_tensor)
                frame_tensors.append(frame_tensor)
        except Exception as e:
            continue
    
    cap.release()
    
    while len(frame_tensors) < FRAME_SAMPLE:
        frame_tensors.extend(frame_tensors[:min(len(frame_tensors), FRAME_SAMPLE - len(frame_tensors))])
        frames_pil.extend(frames_pil[:min(len(frames_pil), FRAME_SAMPLE - len(frames_pil))])
    
    frame_tensors = frame_tensors[:FRAME_SAMPLE]
    frames_pil = frames_pil[:FRAME_SAMPLE]
    
    return frames_pil, frame_tensors


def extract_slowfast_features(slowfast_model, frame_tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    SlowFast 모델로 비디오 행동 특징 추출
    
    Args:
        slowfast_model: SlowFast 모델
        frame_tensors: 프레임 텐서 리스트 (32개)
        
    Returns:
        features: SlowFast 특징 벡터 (400차원)
    """
    try:
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
        frame_tensors_normalized = [(f - mean) / std for f in frame_tensors]
        
        fast_pathway = torch.stack(frame_tensors_normalized).unsqueeze(0).to(DEVICE, non_blocking=True)
        fast_pathway = fast_pathway.permute(0, 2, 1, 3, 4)
        
        # Slow pathway: 8개 프레임 샘플링 (인덱스 범위 체크)
        num_frames = len(frame_tensors_normalized)
        if num_frames > 0:
            slow_indices = torch.linspace(0, num_frames - 1, min(8, num_frames)).long()
            slow_tensors = [frame_tensors_normalized[i] for i in slow_indices if i < num_frames]
            if len(slow_tensors) == 0:
                slow_tensors = frame_tensors_normalized[:1]  # 최소 1개는 필요
            slow_pathway = torch.stack(slow_tensors).unsqueeze(0).to(DEVICE, non_blocking=True)
            slow_pathway = slow_pathway.permute(0, 2, 1, 3, 4)
        else:
            # 프레임이 없으면 fast_pathway와 동일하게
            slow_pathway = fast_pathway
        
        with torch.no_grad():
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    features = slowfast_model([slow_pathway, fast_pathway])
            else:
                features = slowfast_model([slow_pathway, fast_pathway])
            features = features.squeeze()
        
        return features
    
    except Exception as e:
        # SlowFast 오류는 조용히 처리 (평가 시 출력 방지)
        return torch.zeros(SLOWFAST_DIM, device=DEVICE)


def predict_video(video_path: str, video_model, yolo_model, slowfast_model,
                  clip_model, clip_preprocess, threshold: float = 0.3, verbose: bool = True) -> Dict:
    """
    비디오에 대한 유해 콘텐츠 탐지 (학습 시와 동일한 방식)
    
    Args:
        video_path: 비디오 파일 경로
        video_model: 학습된 비디오 분류 모델
        yolo_model: YOLO 모델
        slowfast_model: SlowFast 모델
        clip_model: CLIP 모델
        clip_preprocess: CLIP 전처리 함수
        threshold: 분류 임계값
        
    Returns:
        result: 딕셔너리
            - is_harmful: bool (유해 여부)
            - confidence: float (유해 확률, 0~1)
            - detected_objects: List[str] (감지된 유해 객체)
            - detected_behaviors: List[str] (감지된 유해 행동)
    """
    try:
        if video_path is None:
            return {
                "is_harmful": False,
                "confidence": 0.0,
                "detected_objects": [],
                "detected_behaviors": [],
                "error": "비디오 파일을 불러올 수 없습니다."
            }
        
        frames_pil, frame_tensors = extract_frames_safe(video_path)
        
        if len(frames_pil) == 0:
            return {
                "is_harmful": False,
                "confidence": 0.0,
                "detected_objects": [],
                "detected_behaviors": [],
                "error": "비디오에서 프레임을 추출할 수 없습니다."
            }
        
        frame_np_list = [np.array(frame_pil) for frame_pil in frames_pil]
        yolo_results = yolo_model(frame_np_list, verbose=False, device=DEVICE, imgsz=640, conf=0.25)
        
        yolo_features_list = []
        all_object_counts = {}
        
        if isinstance(yolo_results, list):
            for result in yolo_results:
                yolo_feat, obj_counts = extract_yolo_features([result])
                yolo_features_list.append(yolo_feat)
                for obj, count in obj_counts.items():
                    all_object_counts[obj] = all_object_counts.get(obj, 0) + count
        else:
            yolo_feat, obj_counts = extract_yolo_features([yolo_results])
            yolo_features_list.append(yolo_feat)
            for obj, count in obj_counts.items():
                all_object_counts[obj] = all_object_counts.get(obj, 0) + count
        
        yolo_features_seq = torch.stack(yolo_features_list).to(DEVICE)
        detected_objects = [obj for obj in HARMFUL_OBJECTS if all_object_counts.get(obj, 0) > 0]
        
        clip_images = torch.stack([clip_preprocess(frame_pil) for frame_pil in frames_pil]).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    clip_features_seq = clip_model.encode_image(clip_images)
            else:
                clip_features_seq = clip_model.encode_image(clip_images)
            clip_features_seq = F.normalize(clip_features_seq, p=2, dim=-1)
        
        slowfast_features = extract_slowfast_features(slowfast_model, frame_tensors)
        if slowfast_features.device != DEVICE:
            slowfast_features_seq = slowfast_features.unsqueeze(0).expand(FRAME_SAMPLE, -1).to(DEVICE, non_blocking=True)
        else:
            slowfast_features_seq = slowfast_features.unsqueeze(0).expand(FRAME_SAMPLE, -1)
        
        # CLIP 무기 감지 (비디오: 모든 프레임의 평균 특징 사용)
        clip_features_avg = clip_features_seq.mean(dim=0)
        weapon_score, weapon_type = detect_weapons_with_clip(
            clip_features_avg, clip_model, weapon_features_cache=_clip_weapon_features_cache
        )
        
        behavior_scores = detect_behavior_with_clip_fast_from_features(
            clip_features_seq, clip_model, text_features_cache=_clip_text_features_cache
        )
        inferred_categories = infer_behavior_from_objects(all_object_counts)
        
        behavior_features = torch.zeros(len(BEHAVIOR_CATEGORIES), device=DEVICE)
        for i, category in enumerate(BEHAVIOR_CATEGORIES):
            clip_score = behavior_scores.get(category, 0.0)
            rule_score = 1.0 if category in inferred_categories else 0.0
            behavior_features[i] = 0.7 * clip_score + 0.3 * rule_score
        
        # detected_behaviors: 규칙 기반 + 매우 높은 CLIP 점수
        # violence는 일상 동작과 혼동되어 제외
        # dangerous, sexual은 더 보수적으로 (오탐 심각)
        detected_behaviors = inferred_categories.copy()
        for category, score in behavior_scores.items():
            if category == 'violence':
                continue  # violence는 제외 (False Positive 너무 많음)
            elif category == 'dangerous' and score >= 0.99:
                # dangerous는 0.99 이상만 (거의 확실한 경우만)
                detected_behaviors.append(category)
            elif category == 'sexual' and score >= 0.98:
                # sexual은 0.98 이상만
                detected_behaviors.append(category)
            elif score >= 0.95 and category not in detected_behaviors:
                # 나머지 카테고리는 0.95 이상
                detected_behaviors.append(category)
        
        behavior_features_seq = behavior_features.unsqueeze(0).expand(FRAME_SAMPLE, -1)
        
        combined = torch.cat([
            yolo_features_seq,
            clip_features_seq,
            slowfast_features_seq,
            behavior_features_seq
        ], dim=1).unsqueeze(0)
        
        video_model.eval()
        with torch.no_grad():
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    confidence = video_model(combined).item()
            else:
                confidence = video_model(combined).item()
        
        # CLIP 무기 감지 (Zero-shot): 비디오는 0.2 (프레임 평균으로 낮아짐)
        weapon_threshold = 0.2  # 비디오용 (이미지보다 낮음)
        if weapon_score >= weapon_threshold and weapon_type:
            if weapon_type not in detected_objects:
                detected_objects.append(weapon_type)
            if verbose:
                print(f"  [CLIP 무기 감지] {weapon_type}: {weapon_score:.3f}")
        
        # 무기 휴리스틱: knife + threat 또는 CLIP 무기 감지
        weapon_detected = False
        if weapon_score >= weapon_threshold:
            weapon_detected = True
            is_harmful = True
        elif 'knife' in detected_objects and 'threat' in detected_behaviors:
            weapon_detected = True
            is_harmful = True
        
        # 균형잡힌 분류 로직
        if not weapon_detected:
            if len(detected_objects) == 0 and len(detected_behaviors) == 0:
                # 아무것도 감지 안 됨 → 안전
                is_harmful = False
            else:
                # 객체나 행동이 감지됨 → threshold 사용
                # threat만 더 민감하게 반응 (threshold 35% 낮춤)
                if 'threat' in detected_behaviors:
                    adjusted_threshold = threshold * 0.65
                else:
                    adjusted_threshold = threshold
                is_harmful = confidence > adjusted_threshold
        
        return {
            "is_harmful": is_harmful,
            "confidence": confidence,
            "detected_objects": detected_objects,
            "detected_behaviors": detected_behaviors
        }
    
    except Exception as e:
        print(f"비디오 추론 오류: {e}")
        return {
            "is_harmful": False,
            "confidence": 0.0,
            "detected_objects": [],
            "detected_behaviors": [],
            "error": str(e)
        }

