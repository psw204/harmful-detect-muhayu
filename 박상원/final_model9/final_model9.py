"""
무하유 유해 콘텐츠 탐지 시스템 - 행동 인식 확장 버전 9
이미지: YOLOv8 + CLIP + MLP (행동 추론 포함)
비디오: YOLOv8 + SlowFast + CLIP + Transformer (행동 인식 강화)

이 시스템은 이미지와 비디오에서 유해한 콘텐츠와 행동을 자동으로 탐지하는 AI 모델입니다.
- 이미지: YOLO로 객체 탐지 + CLIP으로 맥락 이해 + 행동 추론 + MLP로 분류
- 비디오: YOLO + SlowFast로 행동 인식 + CLIP + Transformer로 시계열 분석 + 행동 분류

[변경점] final_model9: Zero-shot 행동 인식 추가 (추가 라벨링 불필요)
- CLIP 텍스트-비디오 매칭으로 행동 감지
- SlowFast 특징 기반 행동 추론
- 객체+행동 결합 분석으로 위험도 평가

작성자: 박상원
작성일: 2025년 2학기
"""

# 기본 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# 컴퓨터 비전 및 AI 모델
from ultralytics import YOLO  # YOLOv8 객체 탐지 모델
import clip  # CLIP 멀티모달 모델
from pytorchvideo.models.hub import slowfast_r50  # SlowFast 비디오 이해 모델

# 이미지/비디오 처리
from PIL import Image
import cv2

# 데이터 처리 및 분석
import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# 시스템 및 시간
import sys
import datetime

# 데이터 중복 제거
import hashlib

# ============================================================
# 로깅 시스템 설정
# ============================================================
# 현재 시간을 기반으로 로그 파일명 생성
now = datetime.datetime.now()
log_filename = now.strftime("train_log_%Y%m%d_%H%M%S.txt")

class Tee(object):
    """
    표준 출력을 콘솔과 파일에 동시에 출력하는 클래스
    학습 과정을 파일로 저장하여 나중에 분석할 수 있도록 함
    """
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, data):
        sys.__stdout__.write(data)  # 콘솔에 출력
        self.file.write(data)       # 파일에 저장

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()

# 표준 출력과 에러 출력을 모두 로그 파일로 리다이렉트
tee = Tee(log_filename)
sys.stdout = tee
sys.stderr = tee

# ============================================================
# 하이퍼파라미터 설정
# ============================================================
# 데이터 경로 설정
DATA_PATH = './무하유_유해콘텐츠_데이터/'  # 데이터셋 루트 디렉토리

# 디바이스 설정 (GPU 우선, 없으면 CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 배치 크기 설정
BATCH_SIZE = 8          # 이미지 학습 배치 크기
VIDEO_BATCH_SIZE = 4    # 비디오 학습 배치 크기 (메모리 제약으로 작게 설정)

# 학습 에포크 설정
IMAGE_EPOCHS = 10       # 이미지 모델 학습 에포크
VIDEO_EPOCHS = 10       # 비디오 모델 학습 에포크

# ------------------------------------------------------------
# [변경점] final_model9: Model 7의 안정적 파라미터 유지
# ------------------------------------------------------------
# 이유: Model 7이 가장 안정적이고 높은 성능 (F1 0.9656, 에러 0%)
IMAGE_LR = 0.0005       # 이미지 모델 학습률
VIDEO_LR = 0.0001       # 비디오 모델 학습률

# 강력한 정규화 유지
IMAGE_WEIGHT_DECAY = 0.01    # 이미지 모델 가중치 감쇠
VIDEO_WEIGHT_DECAY = 0.01    # 비디오 모델 가중치 감쇠

# 비디오 처리 설정
FRAME_SAMPLE = 32       # 비디오에서 샘플링할 프레임 수

# ============================================================
# [변경점] final_model9: 확실한 위험 물품 + 문맥 기반 위험 행동 정의
# ============================================================
# 확실한 위험 물품 (객체 중심 탐지) - YOLO로 직접 탐지 가능
HARMFUL_OBJECTS = [
    # 날붙이 및 도검류
    'knife', 'dagger', 'machete', 'sword', 'axe',
    # 총기류
    'gun', 'pistol', 'rifle', 'shotgun', 'machine_gun',
    # 폭발물, 사제폭탄류
    'grenade', 'bomb'
]

# 문맥 기반 위험 행동 (텍스트/비디오 기반 탐지) - CLIP + SlowFast로 추론
HARMFUL_BEHAVIORS = [
    'drug_use',        # 마약 복용/투여
    'smoking',         # 흡연 행위
    'drinking',        # 음주 행위 (과도한)
    'violent_act',     # 폭력 행위 (공격, 구타, 싸움)
    'self_harm',       # 자해 행위
    'threatening',     # 위협적 행동/발언
    'sexual_violence'  # 성적 폭력 또는 강압적 성적 행동
]

# ------------------------------------------------------------
# [변경점] final_model9: CLIP을 활용한 행동 인식 프롬프트
# ------------------------------------------------------------
# 이유: 추가 라벨링 없이 Zero-shot으로 행동 감지
BEHAVIOR_PROMPTS = {
    'drug_use': [
        "a person injecting drugs",
        "drug paraphernalia and needles",
        "person using illegal substances"
    ],
    'smoking': [
        "a person smoking cigarette",
        "person holding cigarette and smoking",
        "smoking tobacco"
    ],
    'drinking': [
        "person drinking alcohol excessively",
        "drunk person with bottles",
        "excessive alcohol consumption"
    ],
    'violent_act': [
        "people fighting violently",
        "physical assault and violence",
        "aggressive fighting between people"
    ],
    'self_harm': [
        "person self-harming with object",
        "self-inflicted injury",
        "person cutting themselves"
    ],
    'threatening': [
        "person threatening with weapon",
        "intimidating aggressive behavior",
        "person pointing gun at someone"
    ],
    'sexual_violence': [
        "sexual assault or harassment",
        "non-consensual physical contact",
        "aggressive sexual behavior"
    ]
}

# 보조 객체 (행동 추론에 도움)
CONTEXTUAL_OBJECTS = [
    'bottle', 'wine glass', 'beer', 'cup',  # 음주 관련
    'cigarette', 'lighter',                  # 흡연 관련
    'syringe', 'needle',                     # 약물 관련
]

# 전체 탐지 객체 목록 (위험 물품 + 보조 객체)
ALL_OBJECTS = HARMFUL_OBJECTS + CONTEXTUAL_OBJECTS

# ============================================================
# 이미지 중복 제거 함수
# ============================================================
def compute_image_hash(image_path):
    """
    이미지 파일의 해시값을 계산하여 중복 파일 감지
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        str: 이미지의 MD5 해시값
    """
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def remove_duplicate_images(image_paths, labels):
    """
    중복 이미지를 제거하여 데이터 누수 방지
    
    Args:
        image_paths: 이미지 경로 리스트
        labels: 라벨 리스트
        
    Returns:
        unique_paths, unique_labels: 중복 제거된 경로와 라벨
    """
    seen_hashes = set()
    unique_paths = []
    unique_labels = []
    duplicates = 0
    
    for path, label in zip(image_paths, labels):
        img_hash = compute_image_hash(path)
        if img_hash and img_hash not in seen_hashes:
            seen_hashes.add(img_hash)
            unique_paths.append(path)
            unique_labels.append(label)
        else:
            duplicates += 1
    
    if duplicates > 0:
        print(f"✓ 중복 이미지 {duplicates}개 제거됨")
    
    return unique_paths, unique_labels

# 비디오 파일 유효성 검사 함수
def validate_video(video_path):
    """
    비디오 파일 유효성 검사
    
    손상되었거나 프레임이 없는 비디오 파일을 걸러냅니다.
    데이터 로딩 과정에서 발생할 수 있는 오류를 사전에 방지합니다.
    
    Args:
        video_path: 검사할 비디오 파일 경로
        
    Returns:
        bool: 비디오 파일이 유효하고 프레임이 있으면 True, 그렇지 않으면 False
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():  # 비디오 파일 열기 실패
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames > 0  # 프레임이 1개 이상 있어야 유효
    except:
        return False

# ============================================================
# [변경점] final_model9: Zero-shot 행동 감지 함수
# ============================================================
def detect_behavior_with_clip(image_or_frames, clip_model, clip_preprocess):
    """
    CLIP을 사용한 Zero-shot 행동 감지
    
    추가 라벨링 없이 텍스트 프롬프트로 행동을 감지합니다.
    각 행동 카테고리에 대해 여러 프롬프트의 평균 유사도를 계산합니다.
    
    Args:
        image_or_frames: PIL Image 또는 프레임 리스트
        clip_model: CLIP 모델
        clip_preprocess: CLIP 전처리 함수
        
    Returns:
        behavior_scores: 각 행동별 점수 딕셔너리 (0~1)
    """
    behavior_scores = {}
    
    try:
        # 이미지인 경우 리스트로 변환
        if isinstance(image_or_frames, Image.Image):
            frames = [image_or_frames]
        else:
            frames = image_or_frames
        
        # 각 행동에 대해 CLIP 유사도 계산
        for behavior, prompts in BEHAVIOR_PROMPTS.items():
            scores = []
            
            # 여러 프롬프트에 대해 평균 계산
            for prompt in prompts:
                # 텍스트 인코딩
                text = clip.tokenize([prompt]).to(DEVICE)
                
                frame_scores = []
                for frame in frames[:min(len(frames), 8)]:  # 최대 8개 프레임만 사용
                    # 이미지 인코딩
                    image_input = clip_preprocess(frame).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        image_features = clip_model.encode_image(image_input)
                        text_features = clip_model.encode_text(text)
                        
                        # 코사인 유사도 계산
                        similarity = (image_features @ text_features.T).squeeze()
                        frame_scores.append(similarity.item())
                
                # 프레임 평균
                scores.append(np.mean(frame_scores))
            
            # 여러 프롬프트의 평균 점수
            behavior_scores[behavior] = np.mean(scores)
        
        # 정규화 (0~1 범위로)
        min_score = min(behavior_scores.values())
        max_score = max(behavior_scores.values())
        if max_score > min_score:
            for behavior in behavior_scores:
                behavior_scores[behavior] = (behavior_scores[behavior] - min_score) / (max_score - min_score)
    
    except Exception as e:
        print(f"  [행동 감지 오류] {e}")
        # 오류 시 모두 0으로 초기화
        behavior_scores = {behavior: 0.0 for behavior in HARMFUL_BEHAVIORS}
    
    return behavior_scores

# ============================================================
# [변경점] final_model9: 객체 기반 행동 추론 함수
# ============================================================
def infer_behavior_from_objects(object_counts):
    """
    탐지된 객체로부터 행동 추론 (휴리스틱 규칙)
    
    명시적 행동 라벨 없이 객체 패턴으로 행동을 추론합니다.
    예: cigarette 탐지 → smoking 행동 추론
    
    Args:
        object_counts: 객체별 탐지 횟수 딕셔너리
        
    Returns:
        inferred_behaviors: 추론된 행동 리스트
    """
    inferred_behaviors = []
    
    # 흡연 추론
    if object_counts.get('cigarette', 0) > 0:
        inferred_behaviors.append('smoking')
    
    # 음주 추론
    drinking_objects = ['bottle', 'wine glass', 'beer', 'cup']
    if sum(object_counts.get(obj, 0) for obj in drinking_objects) >= 2:
        inferred_behaviors.append('drinking')
    
    # 약물 사용 추론
    drug_objects = ['syringe', 'needle']
    if sum(object_counts.get(obj, 0) for obj in drug_objects) > 0:
        inferred_behaviors.append('drug_use')
    
    # 위협 추론
    weapon_objects = ['knife', 'gun', 'pistol', 'rifle', 'sword', 'axe']
    if sum(object_counts.get(obj, 0) for obj in weapon_objects) > 0:
        if object_counts.get('person', 0) > 0:  # 사람과 무기가 함께 있으면
            inferred_behaviors.append('threatening')
    
    return inferred_behaviors

# ============================================================
# 이미지 데이터셋
# ============================================================
class HarmfulImageDataset(Dataset):
    """
    유해 이미지 탐지를 위한 데이터셋 클래스 (행동 인식 포함)
    
    YOLO와 CLIP 모델을 사용하여 이미지에서 특징을 추출하고,
    Zero-shot 방식으로 행동을 감지합니다.
    """
    def __init__(self, image_paths, labels, yolo_model, clip_model, clip_preprocess, augment=False):
        self.image_paths = image_paths  # 이미지 파일 경로 리스트
        self.labels = labels           # 라벨 리스트 (0: 안전, 1: 유해)
        self.yolo = yolo_model         # YOLO 객체 탐지 모델
        self.clip_model = clip_model   # CLIP 멀티모달 모델
        self.clip_preprocess = clip_preprocess  # CLIP 전처리 함수
        self.augment = augment         # 데이터 증강 여부
        
        # 데이터 증강 설정 (Model 7 기반)
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.RandomRotation(degrees=15),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        인덱스에 해당하는 이미지와 라벨을 반환
        
        Args:
            idx: 데이터셋 인덱스
            
        Returns:
            combined: YOLO + CLIP + 행동 특징 벡터
            label: 라벨 (0 또는 1)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 이미지 로드
            image = Image.open(img_path).convert('RGB')
            original_image = image.copy()  # 행동 감지용 원본 보존
            
            # YOLO 특징 추출 (원본 이미지 사용)
            yolo_results = self.yolo(img_path, verbose=False)
            yolo_features, object_counts = self._extract_yolo_features(yolo_results)
            
            # CLIP 특징 추출 (데이터 증강 적용)
            if self.augment:
                try:
                    image = self.aug_transform(image)
                except:
                    pass  # 증강 실패 시 원본 사용
            
            clip_image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                clip_features = self.clip_model.encode_image(clip_image).squeeze().cpu()
            
            # ------------------------------------------------------------
            # [변경점] final_model9: Zero-shot 행동 감지 추가
            # ------------------------------------------------------------
            # 행동 감지 (원본 이미지 사용)
            behavior_scores = detect_behavior_with_clip(original_image, self.clip_model, self.clip_preprocess)
            
            # 객체 기반 행동 추론
            inferred_behaviors = infer_behavior_from_objects(object_counts)
            
            # 행동 특징 벡터 생성 (7차원: 각 행동별 점수)
            behavior_features = torch.zeros(len(HARMFUL_BEHAVIORS))
            for i, behavior in enumerate(HARMFUL_BEHAVIORS):
                # CLIP 점수와 객체 추론 결합
                clip_score = behavior_scores.get(behavior, 0.0)
                rule_score = 1.0 if behavior in inferred_behaviors else 0.0
                # 가중 평균 (CLIP 70%, 규칙 30%)
                behavior_features[i] = 0.7 * clip_score + 0.3 * rule_score
            
            # YOLO 특징(객체 수) + CLIP 특징(맥락) + 행동 특징 결합
            # 차원: len(ALL_OBJECTS) + 512 + len(HARMFUL_BEHAVIORS)
            combined = torch.cat([yolo_features, clip_features, behavior_features])
            
            return combined, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 오류 발생 시 제로 벡터 반환
            zero_dim = len(ALL_OBJECTS) + 512 + len(HARMFUL_BEHAVIORS)
            return torch.zeros(zero_dim), torch.tensor(label, dtype=torch.float32)
    
    def _extract_yolo_features(self, results):
        """
        YOLO 탐지 결과에서 객체 특징 벡터 추출
        
        Args:
            results: YOLO 탐지 결과
            
        Returns:
            feature_vector: 객체별 탐지 횟수 벡터
            object_counts: 객체별 탐지 횟수 딕셔너리
        """
        feature_vector = torch.zeros(len(ALL_OBJECTS))
        object_counts = {}
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)].lower()
                    # 전체 객체 목록과 매칭
                    for i, obj in enumerate(ALL_OBJECTS):
                        if obj in class_name or class_name in obj:
                            feature_vector[i] += 1
                            object_counts[obj] = object_counts.get(obj, 0) + 1
        
        return feature_vector, object_counts

# ============================================================
# 비디오 데이터셋
# ============================================================
class HarmfulVideoDataset(Dataset):
    """
    유해 비디오 탐지를 위한 데이터셋 클래스 (행동 인식 강화)
    
    YOLO, SlowFast, CLIP 모델을 사용하여 비디오에서 시공간 특징을 추출하고,
    Zero-shot 방식으로 행동을 감지합니다.
    """
    def __init__(self, video_paths, labels, yolo_model, slowfast_model, clip_model, clip_preprocess, slowfast_dim):
        self.video_paths = video_paths    # 비디오 파일 경로 리스트
        self.labels = labels             # 라벨 리스트 (0: 안전, 1: 유해)
        self.yolo = yolo_model          # YOLO 객체 탐지 모델
        self.slowfast = slowfast_model   # SlowFast 행동 인식 모델
        self.clip_model = clip_model     # CLIP 멀티모달 모델
        self.clip_preprocess = clip_preprocess  # CLIP 전처리 함수
        self.slowfast_dim = slowfast_dim  # SlowFast 출력 차원

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 비디오와 라벨을 반환
        
        Args:
            idx: 데이터셋 인덱스
            
        Returns:
            combined: 시계열 특징 벡터 (FRAME_SAMPLE, feature_dim)
            label: 라벨 (0 또는 1)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # 비디오에서 프레임 추출
            frames_pil, frame_tensors = self._extract_frames(video_path)
            
            if len(frames_pil) == 0:
                raise ValueError("No frames extracted")
            
            # YOLO 특징 추출 (각 프레임별)
            yolo_features_list = []
            all_object_counts = {}
            for i, frame_pil in enumerate(frames_pil):
                # PIL → numpy → YOLO
                frame_np = np.array(frame_pil)
                yolo_results = self.yolo(frame_np, verbose=False)
                yolo_feat, obj_counts = self._extract_yolo_features(yolo_results)
                yolo_features_list.append(yolo_feat)
                
                # 전체 비디오의 객체 누적
                for obj, count in obj_counts.items():
                    all_object_counts[obj] = all_object_counts.get(obj, 0) + count
            
            yolo_features_seq = torch.stack(yolo_features_list)  # (FRAME_SAMPLE, len(ALL_OBJECTS))
            
            # CLIP 특징 추출 (각 프레임별)
            clip_features_list = []
            for frame_pil in frames_pil:
                clip_image = self.clip_preprocess(frame_pil).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    clip_feat = self.clip_model.encode_image(clip_image).squeeze().cpu()
                clip_features_list.append(clip_feat)
            
            clip_features_seq = torch.stack(clip_features_list)  # (FRAME_SAMPLE, 512)
            
            # SlowFast 특징 추출 (전체 비디오)
            slowfast_features = self._extract_slowfast_features(frame_tensors)
            # SlowFast 특징을 각 프레임에 복제
            slowfast_features_seq = slowfast_features.unsqueeze(0).expand(FRAME_SAMPLE, -1)  # (FRAME_SAMPLE, slowfast_dim)
            
            # ------------------------------------------------------------
            # [변경점] final_model9: Zero-shot 행동 감지 (비디오)
            # ------------------------------------------------------------
            # 행동 감지 (여러 프레임 사용)
            behavior_scores = detect_behavior_with_clip(frames_pil, self.clip_model, self.clip_preprocess)
            
            # 객체 기반 행동 추론
            inferred_behaviors = infer_behavior_from_objects(all_object_counts)
            
            # 행동 특징 벡터 생성
            behavior_features = torch.zeros(len(HARMFUL_BEHAVIORS))
            for i, behavior in enumerate(HARMFUL_BEHAVIORS):
                clip_score = behavior_scores.get(behavior, 0.0)
                rule_score = 1.0 if behavior in inferred_behaviors else 0.0
                # 비디오는 CLIP 점수에 더 높은 가중치 (80%)
                behavior_features[i] = 0.8 * clip_score + 0.2 * rule_score
            
            # 행동 특징을 각 프레임에 복제
            behavior_features_seq = behavior_features.unsqueeze(0).expand(FRAME_SAMPLE, -1)  # (FRAME_SAMPLE, len(HARMFUL_BEHAVIORS))
            
            # 모든 특징 결합: YOLO + CLIP + SlowFast + 행동
            # 차원: (FRAME_SAMPLE, len(ALL_OBJECTS) + 512 + slowfast_dim + len(HARMFUL_BEHAVIORS))
            combined = torch.cat([
                yolo_features_seq,
                clip_features_seq,
                slowfast_features_seq,
                behavior_features_seq
            ], dim=1)
            
            return combined, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # 오류 발생 시 제로 벡터 반환
            zero_dim = len(ALL_OBJECTS) + 512 + self.slowfast_dim + len(HARMFUL_BEHAVIORS)
            return torch.zeros(FRAME_SAMPLE, zero_dim), torch.tensor(label, dtype=torch.float32)
    
    def _extract_frames(self, video_path):
        """
        비디오에서 균등하게 샘플링한 프레임 추출
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            frames_pil: PIL Image 리스트
            frame_tensors: Tensor 리스트
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return [], []
        
        # 균등 샘플링
        indices = np.linspace(0, total_frames - 1, FRAME_SAMPLE, dtype=int)
        
        frames_pil = []
        frame_tensors = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # BGR → RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frames_pil.append(frame_pil)
                
                # Tensor 변환 (SlowFast용)
                frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                frame_tensor = T.Resize((256, 256))(frame_tensor)
                frame_tensors.append(frame_tensor)
        
        cap.release()
        
        # 프레임이 부족하면 반복하여 FRAME_SAMPLE 개수 맞추기
        while len(frame_tensors) < FRAME_SAMPLE:
            frame_tensors.extend(frame_tensors[:min(len(frame_tensors), FRAME_SAMPLE - len(frame_tensors))])
            frames_pil.extend(frames_pil[:min(len(frames_pil), FRAME_SAMPLE - len(frames_pil))])
        
        frame_tensors = frame_tensors[:FRAME_SAMPLE]
        frames_pil = frames_pil[:FRAME_SAMPLE]
        
        return frames_pil, frame_tensors
    
    def _extract_slowfast_features(self, frame_tensors):
        """
        SlowFast 모델로 행동 특징 추출
        
        Args:
            frame_tensors: 프레임 Tensor 리스트
            
        Returns:
            features: SlowFast 특징 벡터
        """
        try:
            # SlowFast 정규화
            mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
            frame_tensors_normalized = [(f - mean) / std for f in frame_tensors]
            
            # Fast pathway (32 프레임)
            fast_pathway = torch.stack(frame_tensors_normalized).unsqueeze(0).to(DEVICE)
            fast_pathway = fast_pathway.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            
            # Slow pathway (8 프레임, 균등 샘플링)
            slow_indices = torch.linspace(0, 31, 8).long()
            slow_tensors = [frame_tensors_normalized[i] for i in slow_indices]
            slow_pathway = torch.stack(slow_tensors).unsqueeze(0).to(DEVICE)
            slow_pathway = slow_pathway.permute(0, 2, 1, 3, 4)
            
            # SlowFast 추론
            with torch.no_grad():
                features = self.slowfast([slow_pathway, fast_pathway])
                features = features.squeeze().cpu()
            
            return features
            
        except Exception as e:
            print(f"  [SlowFast 오류] {e}")
            return torch.zeros(self.slowfast_dim)
    
    def _extract_yolo_features(self, results):
        """
        YOLO 탐지 결과에서 객체 특징 벡터 추출
        
        Args:
            results: YOLO 탐지 결과
            
        Returns:
            feature_vector: 객체별 탐지 횟수 벡터
            object_counts: 객체별 탐지 횟수 딕셔너리
        """
        feature_vector = torch.zeros(len(ALL_OBJECTS))
        object_counts = {}
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)].lower()
                    for i, obj in enumerate(ALL_OBJECTS):
                        if obj in class_name or class_name in obj:
                            feature_vector[i] += 1
                            object_counts[obj] = object_counts.get(obj, 0) + 1
        
        return feature_vector, object_counts

# ============================================================
# 이미지 분류 모델
# ============================================================
class HarmfulImageClassifier(nn.Module):
    """
    유해 이미지 분류 모델 (행동 인식 포함)
    
    YOLO + CLIP + 행동 특징을 MLP로 처리하여 이진 분류 수행
    """
    def __init__(self, yolo_dim, clip_dim, behavior_dim):
        super().__init__()
        input_dim = yolo_dim + clip_dim + behavior_dim
        
        # Model 7의 안정적인 MLP 구조 유지
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.6),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.6),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 특징 벡터
            
        Returns:
            output: 유해 확률 (0~1)
        """
        return self.mlp(x).squeeze()

# ============================================================
# 비디오 분류 모델
# ============================================================
class HarmfulVideoClassifier(nn.Module):
    """
    유해 비디오 분류 모델 (행동 인식 강화)
    
    YOLO + SlowFast + CLIP + 행동 특징을 Transformer로 시계열 분석 후 분류
    """
    def __init__(self, yolo_dim, clip_dim, slowfast_dim, behavior_dim):
        super().__init__()
        input_dim = yolo_dim + clip_dim + slowfast_dim + behavior_dim
        
        # Transformer nhead 동적 설정
        nhead = self._find_best_nhead(input_dim, max_heads=16)
        print(f"✓ Transformer nhead: {nhead} (input_dim={input_dim})")
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True,
            dropout=0.4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # MLP 분류기
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            
            nn.Linear(128, 1)
        )
    
    def _find_best_nhead(self, input_dim, max_heads=16):
        """
        input_dim을 균등하게 나눌 수 있는 최대 head 수 찾기
        
        Args:
            input_dim: Transformer 입력 차원
            max_heads: 최대 허용 head 수
            
        Returns:
            nhead: 최적의 head 수
        """
        for nhead in range(min(max_heads, input_dim), 0, -1):
            if input_dim % nhead == 0:
                return nhead
        return 1
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 시계열 특징 (batch_size, FRAME_SAMPLE, feature_dim)
            
        Returns:
            output: 유해 확률 (0~1)
        """
        # Transformer로 시계열 분석
        transformed = self.transformer(x)
        
        # 시간 차원 평균 풀링
        pooled = transformed.mean(dim=1)
        
        # MLP 분류
        logits = self.classifier(pooled)
        return torch.sigmoid(logits).squeeze()

# ============================================================
# Label Smoothing Loss
# ============================================================
class LabelSmoothingBCELoss(nn.Module):
    """
    Label Smoothing을 적용한 Binary Cross Entropy Loss
    
    과적합을 방지하기 위해 정답 라벨을 부드럽게 만듦
    예: 0 → 0.1, 1 → 0.9 (smoothing=0.2)
    """
    def __init__(self, smoothing=0.2):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        """
        손실 계산
        
        Args:
            pred: 모델 예측 (0~1)
            target: 실제 라벨 (0 또는 1)
            
        Returns:
            loss: Label Smoothing이 적용된 BCE Loss
        """
        # 라벨 스무딩 적용
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy(pred, target_smooth)

# ============================================================
# 이미지 데이터 준비
# ============================================================
def prepare_image_data():
    """
    이미지 데이터 로딩 및 전처리
    
    공개 데이터셋과 실제 수집 데이터를 통합하고,
    중복을 제거한 뒤 train/val 분할
    
    Returns:
        train_images, train_labels, val_images, val_labels
    """
    print("\n이미지 데이터 준비 중...")
    
    # HOD Dataset (유해 이미지)
    hod_path = DATA_PATH + '1_공개_데이터셋/HOD_Dataset/dataset/'
    hod_images = []
    hod_labels = []
    if os.path.exists(hod_path):
        for root, dirs, files in os.walk(hod_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    hod_images.append(os.path.join(root, file))
                    hod_labels.append(1)  # 유해
    
    # COCO Safe Dataset (안전 이미지)
    coco_path = DATA_PATH + '1_공개_데이터셋/COCO_Safe_Dataset/'
    coco_images = []
    coco_labels = []
    if os.path.exists(coco_path):
        for root, dirs, files in os.walk(coco_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    coco_images.append(os.path.join(root, file))
                    coco_labels.append(0)  # 안전
    
    # 검증된 유해 이미지
    verified_json = DATA_PATH + '3_라벨링_파일/verified_labels.json'
    verified_images = []
    verified_labels = []
    if os.path.exists(verified_json):
        with open(verified_json, 'r', encoding='utf-8') as f:
            verified_data = json.load(f)
        for img_name in verified_data.keys():
            img_path = DATA_PATH + '2_실제_수집_데이터/이미지/' + img_name
            if os.path.exists(img_path):
                verified_images.append(img_path)
                verified_labels.append(1)  # 유해
    
    # 안전 이미지
    safe_json = DATA_PATH + '3_라벨링_파일/safe_labels.json'
    safe_images = []
    safe_labels = []
    if os.path.exists(safe_json):
        with open(safe_json, 'r', encoding='utf-8') as f:
            safe_data = json.load(f)
        for img_name in safe_data.keys():
            img_path = DATA_PATH + '2_실제_수집_데이터/안전_이미지/' + img_name
            if os.path.exists(img_path):
                safe_images.append(img_path)
                safe_labels.append(0)  # 안전
    
    # 전체 데이터 통합
    X = hod_images + coco_images + verified_images + safe_images
    y = hod_labels + coco_labels + verified_labels + safe_labels
    
    print(f"✓ 중복 제거 전: {len(X)}개 이미지")
    
    # 중복 제거
    X, y = remove_duplicate_images(X, y)
    
    print(f"✓ 중복 제거 후: {len(X)}개 이미지")
    
    # Train/Val 분할 (85:15)
    from sklearn.model_selection import train_test_split
    train_images, val_images, train_labels, val_labels = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"✓ 데이터셋 통합 완료 (학습 {len(train_images)}, 검증 {len(val_images)})")
    print(f"  (유해: {sum(train_labels)}, 안전: {len(train_labels)-sum(train_labels)} / 검증 유해: {sum(val_labels)}, 안전: {len(val_labels)-sum(val_labels)})")
    
    return train_images, train_labels, val_images, val_labels

# ============================================================
# 비디오 데이터 준비
# ============================================================
def prepare_video_data():
    """
    비디오 데이터 로딩 및 전처리
    
    공개 데이터셋과 실제 수집 데이터를 통합하고,
    손상된 비디오를 필터링한 뒤 train/val 분할
    
    Returns:
        train_videos, train_labels, val_videos, val_labels
    """
    print("\n비디오 데이터 준비 중...")
    
    vpaths, vlabels = [], []
    svpaths, svlabels = [], []
    pvpaths, pvlabels = [], []
    
    # 검증된 유해 비디오
    verified_video_json = DATA_PATH + '3_라벨링_파일/verified_video_labels.json'
    if os.path.exists(verified_video_json):
        with open(verified_video_json, 'r', encoding='utf-8') as f:
            vdata = json.load(f)
        for vid in vdata.keys():
            vp = DATA_PATH + '2_실제_수집_데이터/비디오/' + vid
            if os.path.exists(vp):
                vpaths.append(vp)
                vlabels.append(1)
    
    # 안전 비디오
    safe_video_json = DATA_PATH + '3_라벨링_파일/safe_video_labels.json'
    if os.path.exists(safe_video_json):
        with open(safe_video_json, 'r', encoding='utf-8') as f:
            sdata = json.load(f)
        for vid in sdata.keys():
            vp = DATA_PATH + '2_실제_수집_데이터/안전_비디오/' + vid
            if os.path.exists(vp):
                svpaths.append(vp)
                svlabels.append(0)
    
    # 공개 비디오 데이터셋 (RWF-2000 + RLVS)
    public_video_json = DATA_PATH + '3_라벨링_파일/public_video_labels.json'
    if os.path.exists(public_video_json):
        with open(public_video_json, 'r', encoding='utf-8') as f:
            pdata = json.load(f)
        for vid, item in pdata.items():
            path = item.get("path")
            label = int(item.get("label", 0))
            if path and os.path.exists(path):
                pvpaths.append(path)
                pvlabels.append(label)
    
    # 전체 데이터 통합
    X = vpaths + svpaths + pvpaths
    y = vlabels + svlabels + pvlabels
    
    # 비디오 검증 (손상된 파일 제거)
    print(f"\n비디오 검증 중... (총 {len(X)}개)")
    X_valid = []
    y_valid = []
    filtered_count = 0
    
    for video_path, label in zip(X, y):
        if validate_video(video_path):
            X_valid.append(video_path)
            y_valid.append(label)
        else:
            filtered_count += 1
    
    print(f"✓ 검증 완료: {len(X_valid)}개 유효, {filtered_count}개 필터링됨")
    
    X, y = X_valid, y_valid
    
    # Train/Val 분할 (85:15)
    from sklearn.model_selection import train_test_split
    train_videos, val_videos, train_labels, val_labels = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"✓ 통합 완료 (학습 {len(train_videos)}, 검증 {len(val_videos)})")
    print(f"  (유해: {sum(train_labels)}, 안전: {len(train_labels)-sum(train_labels)} / 검증 유해: {sum(val_labels)}, 안전: {len(val_labels)-sum(val_labels)})")
    
    return train_videos, train_labels, val_videos, val_labels

# ============================================================
# 학습 함수
# ============================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, model_type='image'):
    """
    모델 학습 함수
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        optimizer: 최적화 함수
        scheduler: 학습률 스케줄러
        epochs: 학습 에포크 수
        model_type: 'image' 또는 'video'
    """
    best_f1 = 0
    best_threshold = 0.5
    patience, patience_count = 2, 0  # Model 7 기준
    
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            
            # 단일 샘플 처리
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 검증 모드
        model.eval()
        val_outputs = []
        val_true = []
        
        with torch.no_grad():
            for features, labels_batch in val_loader:
                features = features.to(DEVICE)
                outputs = model(features).squeeze()
                
                outputs_np = outputs.cpu().numpy()
                
                if outputs_np.ndim == 0:
                    val_outputs.append(float(outputs_np))
                else:
                    val_outputs.extend(outputs_np.tolist())
                
                val_true.extend(labels_batch.cpu().numpy())
        
        val_outputs_np = np.array(val_outputs)
        val_true_np = np.array(val_true)
        
        # Threshold 분석
        print(f"\n[Threshold 분석 @ Epoch {epoch+1}]")
        best_th = 0.5
        best_th_f1 = 0
        
        for th in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
            preds = (val_outputs_np > th).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_true_np, preds, average='binary', zero_division=0)
            print(f"  Threshold={th:.2f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            
            if f1 > best_th_f1:
                best_th_f1 = f1
                best_th = th
        
        # 최적 threshold로 최종 평가
        val_preds = (val_outputs_np > best_th).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_true_np, val_preds, average='binary', zero_division=0)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  최적 Threshold: {best_th:.2f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Learning Rate Scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(f1)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  ⚡ Learning Rate 감소: {old_lr:.6f} → {new_lr:.6f}")
        
        # Best 모델 저장
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = best_th
            patience_count = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': best_threshold
            }, f'{model_type}_model_best.pth')
            print(f"  ✓ 모델 저장 (Best F1: {best_f1:.4f}, Threshold: {best_threshold:.2f})")
        else:
            patience_count += 1
        
        # Early Stopping
        if patience_count >= patience:
            print(f"Early stopping at epoch {epoch+1}, best F1: {best_f1:.4f}")
            break
    
    return best_f1, best_threshold

# ============================================================
# Confusion Matrix 그리기
# ============================================================
def plot_confusion_matrix(model, val_loader, threshold, model_type='image'):
    """
    Confusion Matrix 시각화
    
    Args:
        model: 평가할 모델
        val_loader: 검증 데이터 로더
        threshold: 분류 임계값
        model_type: 'image' 또는 'video'
    """
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(DEVICE)
            outputs = model(features).squeeze()
            
            outputs_np = outputs.cpu().numpy()
            if outputs_np.ndim == 0:
                outputs_np = np.array([outputs_np])
            
            preds = (outputs_np > threshold).astype(int)
            all_preds.extend(preds.tolist())
            all_true.extend(labels.cpu().numpy().tolist())
    
    cm = confusion_matrix(all_true, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_type.capitalize()} Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_type}_confusion_matrix.png')
    plt.close()
    print(f"  ✓ Confusion Matrix: {model_type}_confusion_matrix.png")

# ============================================================
# 메인 실행
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("무하유 유해 콘텐츠 탐지 시스템 - 행동 인식 확장 버전 9")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Harmful Objects: {len(ALL_OBJECTS)}개 (위험물품 {len(HARMFUL_OBJECTS)}개 + 보조객체 {len(CONTEXTUAL_OBJECTS)}개)")
    print(f"Harmful Behaviors: {len(HARMFUL_BEHAVIORS)}개 (Zero-shot 감지)")
    print(f"Image Epochs: {IMAGE_EPOCHS}")
    print(f"Video Epochs: {VIDEO_EPOCHS}")
    print(f"✓ 로그 파일: {log_filename}")
    print("="*60)
    
    # ============================================================
    # 이미지 모델 학습
    # ============================================================
    print("\n" + "="*60)
    print("이미지 모델 학습")
    print("="*60)
    
    # 데이터 준비
    train_images, train_labels, val_images, val_labels = prepare_image_data()
    
    # 모델 로딩
    print("\n모델 로딩...")
    yolo = YOLO('yolov8n.pt')
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    
    yolo_dim = len(ALL_OBJECTS)
    clip_dim = 512
    behavior_dim = len(HARMFUL_BEHAVIORS)
    print(f"✓ YOLO 차원: {yolo_dim}")
    print(f"✓ CLIP 차원: {clip_dim}")
    print(f"✓ 행동 차원: {behavior_dim}")
    print(f"✓ 총 입력 차원: {yolo_dim + clip_dim + behavior_dim}")
    
    # 데이터셋 생성
    train_dataset = HarmfulImageDataset(train_images, train_labels, yolo, clip_model, clip_preprocess, augment=False)
    val_dataset = HarmfulImageDataset(val_images, val_labels, yolo, clip_model, clip_preprocess, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 생성
    image_model = HarmfulImageClassifier(yolo_dim, clip_dim, behavior_dim).to(DEVICE)
    
    # 클래스 불균형 보정
    num_harmful = sum(train_labels)
    num_safe = len(train_labels) - num_harmful
    print(f"\n✓ 클래스 분포: 유해 {num_harmful}개, 안전 {num_safe}개 (비율: {num_safe/num_harmful:.2f})")
    
    # 손실 함수 및 최적화
    criterion = LabelSmoothingBCELoss(smoothing=0.2)
    optimizer = optim.Adam(image_model.parameters(), lr=IMAGE_LR, weight_decay=IMAGE_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # 학습
    best_image_f1, best_image_threshold = train_model(
        image_model, train_loader, val_loader, criterion, optimizer, scheduler, IMAGE_EPOCHS, 'image'
    )
    
    # Confusion Matrix
    plot_confusion_matrix(image_model, val_loader, best_image_threshold, 'image')
    
    print("\n" + "="*60)
    print("✅ 이미지 모델 학습 완료!")
    print(f"   Best F1-Score: {best_image_f1:.4f}")
    if best_image_f1 >= 0.75:
        print("   ✅ 필수 목표 달성! (F1 ≥ 0.75)")
    else:
        print("   ⚠️ 목표 미달 (F1 < 0.75)")
    print(f"   ✓ Confusion Matrix: image_confusion_matrix.png")
    print("="*60)
    
    # ============================================================
    # 비디오 모델 학습
    # ============================================================
    print("\n" + "="*60)
    print("비디오 모델 학습")
    print("="*60)
    
    # 데이터 준비
    train_videos, train_vlabels, val_videos, val_vlabels = prepare_video_data()
    
    # SlowFast 모델 로딩
    print("\n모델 로딩...")
    slowfast_model = slowfast_r50(pretrained=True).to(DEVICE)
    slowfast_model.eval()
    
    # SlowFast 출력 차원 확인
    print("SlowFast 출력 차원 확인 중...")
    with torch.no_grad():
        dummy_slow = torch.randn(1, 3, 8, 256, 256).to(DEVICE)
        dummy_fast = torch.randn(1, 3, 32, 256, 256).to(DEVICE)
        slowfast_output = slowfast_model([dummy_slow, dummy_fast])
        slowfast_dim = slowfast_output.shape[-1]
        print(f"✓ SlowFast 실제 출력 차원: {slowfast_dim}")
    
    print(f"✓ YOLO 차원: {yolo_dim}")
    print(f"✓ CLIP 차원: {clip_dim}")
    print(f"✓ 행동 차원: {behavior_dim}")
    print(f"✓ 총 입력 차원: {yolo_dim + clip_dim + slowfast_dim + behavior_dim}")
    
    # 클래스 불균형 보정
    num_harmful_v = sum(train_vlabels)
    num_safe_v = len(train_vlabels) - num_harmful_v
    pos_weight = torch.tensor([num_safe_v / num_harmful_v]).to(DEVICE)
    print(f"\n✓ 클래스 불균형 보정: pos_weight={pos_weight.item():.4f}")
    print(f"  (안전: {num_safe_v}개, 유해: {num_harmful_v}개)")
    
    # 데이터셋 생성
    train_video_dataset = HarmfulVideoDataset(
        train_videos, train_vlabels, yolo, slowfast_model, clip_model, clip_preprocess, slowfast_dim
    )
    val_video_dataset = HarmfulVideoDataset(
        val_videos, val_vlabels, yolo, slowfast_model, clip_model, clip_preprocess, slowfast_dim
    )
    
    train_video_loader = DataLoader(train_video_dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=True)
    val_video_loader = DataLoader(val_video_dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=False)
    
    # 모델 생성
    video_model = HarmfulVideoClassifier(yolo_dim, clip_dim, slowfast_dim, behavior_dim).to(DEVICE)
    
    # 손실 함수 및 최적화
    criterion_video = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer_video = optim.Adam(video_model.parameters(), lr=VIDEO_LR, weight_decay=VIDEO_WEIGHT_DECAY)
    scheduler_video = optim.lr_scheduler.ReduceLROnPlateau(optimizer_video, mode='max', factor=0.5, patience=2)
    
    # 학습
    best_video_f1, best_video_threshold = train_model(
        video_model, train_video_loader, val_video_loader, criterion_video, optimizer_video, scheduler_video, VIDEO_EPOCHS, 'video'
    )
    
    # Confusion Matrix
    plot_confusion_matrix(video_model, val_video_loader, best_video_threshold, 'video')
    
    print("\n" + "="*60)
    print("✅ 비디오 모델 학습 완료!")
    print(f"   Best F1-Score: {best_video_f1:.4f}")
    if best_video_f1 >= 0.75:
        print("   ✅ 필수 목표 달성! (F1 ≥ 0.75)")
    else:
        print("   ⚠️ 목표 미달 (F1 < 0.75)")
    print(f"   ✓ Confusion Matrix: video_confusion_matrix.png")
    print("="*60)
    
    # 최종 요약
    print("\n" + "="*60)
    print("✅ 전체 학습 완료!")
    print("="*60)
    print(f"이미지 모델 Best F1: {best_image_f1:.4f} (Threshold: {best_image_threshold:.2f})")
    print(f"비디오 모델 Best F1: {best_video_f1:.4f} (Threshold: {best_video_threshold:.2f})")
    print(f"\n✓ 행동 인식 기능: {len(HARMFUL_BEHAVIORS)}가지 행동 Zero-shot 감지")
    print(f"  - {', '.join(HARMFUL_BEHAVIORS)}")
    print(f"\n✓ 로그 파일: {log_filename}")
    print("="*60)

