"""
무하유 유해 콘텐츠 탐지 시스템 - 최종 버전
이미지: YOLOv8 + CLIP + 행동 인식 + 차원 축소 MLP
비디오: YOLOv8 + SlowFast + CLIP + 행동 인식 + Transformer

이 시스템은 이미지와 비디오에서 유해한 콘텐츠와 행동을 자동으로 탐지하는 AI 모델입니다.

[학습 데이터]
- 공개 데이터셋만 사용 (독립 평가를 위해 직접 수집 데이터 제외)
- 이미지: HOD Dataset (10,631개) + COCO Safe Dataset (5,000개)
- 비디오: RWF-2000 (2,000개) + RLVS (2,000개)

[주요 특징]
- 멀티모달 특징 융합: YOLO(객체) + CLIP(맥락) + 행동(Zero-shot) 결합
- 차원 축소 레이어: 고차원 특징(538/938차원)을 256차원으로 압축하여 과적합 방지
- Focal Loss: 어려운 샘플에 집중하여 클래스 불균형 문제 해결
- CLIP 특징 정규화: L2 정규화로 코사인 유사도 계산 안정화
- Transformer Encoder: 비디오 프레임 간 시간적 관계 학습 (8-head attention)
- Early Stopping: Validation Loss 기준으로 과최적화 방지
- 재현성 보장: Seed 고정 및 결정론적 알고리즘 사용
- 안전한 데이터 처리: 손상된 비디오 프레임 자동 스킵, 중복 이미지 제거

작성자: 박상원
작성일: 2025년 2학기
"""

# ============================================================
# 라이브러리 임포트
# ============================================================

# 기본 라이브러리
import torch               # PyTorch 딥러닝 프레임워크
import torch.nn as nn      # 신경망 모듈
import torch.nn.functional as F  # 함수형 API (normalize 등)
import torch.optim as optim      # 옵티마이저 (Adam 등)
from torch.utils.data import Dataset, DataLoader  # 데이터 로딩
import torchvision.transforms as T                 # 이미지 변환

# 컴퓨터 비전 및 AI 모델
from ultralytics import YOLO  # YOLOv8 객체 탐지 모델
import clip  # CLIP 멀티모달 모델 (텍스트-이미지 이해)
from pytorchvideo.models.hub import slowfast_r50  # SlowFast 비디오 행동 인식 모델

# 이미지/비디오 처리
from PIL import Image  # 이미지 로딩 및 처리
import cv2             # OpenCV 비디오 처리

# 데이터 처리 및 분석
import json            # JSON 파일 읽기/쓰기
import os              # 파일 시스템 접근
import numpy as np     # 수치 연산
from tqdm import tqdm  # 진행 표시줄
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix  # 평가 지표

# 시각화
import matplotlib.pyplot as plt  # 그래프 그리기
import seaborn as sns            # 히트맵 등 고급 시각화

# 시스템 및 시간
import sys       # 시스템 관련 기능
import datetime  # 날짜/시간 처리

# 데이터 중복 제거
import hashlib  # 이미지 해시 계산 (MD5)

# ============================================================
# 재현성 보장 설정
# ============================================================
import random

# Seed 고정 - 모든 난수 생성을 동일하게 만들어 실험 재현 가능
SEED = 42
random.seed(SEED)              # Python random 모듈
np.random.seed(SEED)           # NumPy 난수
torch.manual_seed(SEED)        # PyTorch CPU 난수
torch.cuda.manual_seed(SEED)   # PyTorch GPU 난수 (단일 GPU)
torch.cuda.manual_seed_all(SEED)  # PyTorch GPU 난수 (멀티 GPU)

# CuDNN 설정 - 재현성을 위해 결정론적 알고리즘 사용
torch.backends.cudnn.deterministic = True  # 동일한 입력에 동일한 결과 보장
torch.backends.cudnn.benchmark = False      # 자동 최적화 비활성화 (속도 느려지지만 재현성 확보)

print("✓ 재현성 보장: Seed 고정 완료 (SEED=42)")

# ============================================================
# 로깅 시스템 설정
# ============================================================
# 현재 시간을 기반으로 로그 파일명 생성
now = datetime.datetime.now()
log_filename = now.strftime("train_log_%Y%m%d_%H%M%S.txt")

class Tee(object):
    """
    표준 출력을 콘솔과 파일에 동시에 출력하는 클래스
    
    학습 과정의 모든 출력을 파일로 저장하여 나중에 분석할 수 있도록 함.
    실험 결과를 추적하고 재현하는 데 필수적.
    """
    def __init__(self, filepath):
        """
        Args:
            filepath: 로그를 저장할 파일 경로
        """
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, data):
        """
        데이터를 콘솔과 파일에 동시 출력
        
        Args:
            data: 출력할 문자열
        """
        sys.__stdout__.write(data)  # 콘솔에 출력
        self.file.write(data)       # 파일에 저장

    def flush(self):
        """버퍼를 비워 즉시 출력"""
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

# 학습률 설정
IMAGE_LR = 0.0005       # 이미지 모델 학습률
VIDEO_LR = 0.0001       # 비디오 모델 학습률 (더 복잡해서 더 낮게 설정)

# Weight Decay 설정 (L2 정규화 강도)
IMAGE_WEIGHT_DECAY = 0.01    # 이미지 모델 가중치 감쇠
VIDEO_WEIGHT_DECAY = 0.01    # 비디오 모델 가중치 감쇠

# 비디오 처리 설정
FRAME_SAMPLE = 32       # 비디오에서 샘플링할 프레임 수 (시간적 해상도)

# ============================================================
# 유해 콘텐츠 정의
# ============================================================
# 탐지할 유해 객체 목록 (12종의 확실한 위험 물품)
HARMFUL_OBJECTS = [
    'knife', 'dagger', 'machete', 'sword', 'axe',           # 날붙이 및 도검류
    'gun', 'pistol', 'rifle', 'shotgun', 'machine_gun',     # 총기류
    'grenade', 'bomb'                                        # 폭발물
]

# 탐지할 유해 행동 목록 (7종의 위험 행동)
HARMFUL_BEHAVIORS = [
    'drug_use',        # 마약 복용/투여
    'smoking',         # 흡연 행위
    'drinking',        # 음주 행위 (과도한)
    'violent_act',     # 폭력 행위 (공격, 구타, 싸움)
    'self_harm',       # 자해 행위
    'threatening',     # 위협적 행동
    'sexual_violence'  # 성적 폭력
]

# 행동 감지 프롬프트 (Zero-shot CLIP 기반 행동 인식용)
BEHAVIOR_PROMPTS = {
    'drug_use': "person using illegal drugs",
    'smoking': "person smoking cigarette",
    'drinking': "person drinking alcohol",
    'violent_act': "people fighting violently",
    'self_harm': "person self-harming",
    'threatening': "threatening with weapon",
    'sexual_violence': "sexual assault"
}

# 맥락 기반 보조 객체 (유해성 판단에 도움)
CONTEXTUAL_OBJECTS = [
    'wine glass', 'beer',                    # 음주 관련 (bottle, cup 제거 - 일반 용도와 혼동)
    'cigarette', 'lighter',                  # 흡연 관련
    'syringe',                               # 약물 관련
]

# 전체 객체 목록 (YOLO 특징 벡터 차원)
ALL_OBJECTS = HARMFUL_OBJECTS + CONTEXTUAL_OBJECTS  # 총 19종 (12 + 7)

# ============================================================
# Focal Loss 손실 함수
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss - 어려운 샘플에 집중하는 손실 함수
    
    일반적인 Cross Entropy Loss는 쉬운 샘플과 어려운 샘플을 동등하게 취급하지만,
    Focal Loss는 어려운 샘플(낮은 확신도)에 더 큰 가중치를 부여하여
    모델이 어려운 케이스를 더 잘 학습하도록 유도합니다.
    
    수식: FL(p_t) = -α(1-p_t)^γ log(p_t)
    - p_t: 정답 클래스에 대한 예측 확률
    - γ (gamma): focusing parameter (어려운 샘플 강조 정도)
      - γ=0: 일반 Cross Entropy
      - γ↑: 어려운 샘플에 더 집중
    - α (alpha): 클래스 불균형 가중치
    
    참고 논문: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Args:
            alpha: 클래스 불균형 보정 파라미터 (기본값: 0.25)
            gamma: focusing 파라미터 (기본값: 2.0)
                   - 값이 클수록 어려운 샘플에 더 집중
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        순전파 - Focal Loss 계산
        
        Args:
            inputs: 모델 예측 확률 (0~1, sigmoid 출력)
            targets: 실제 라벨 (0 or 1)
            
        Returns:
            loss: Focal Loss 값
        """
        # 1. Binary Cross Entropy Loss 계산 (기본)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # 2. p_t 계산: 정답 클래스에 대한 예측 확률
        #    - targets=1이면 p_t = inputs (유해 확률)
        #    - targets=0이면 p_t = 1-inputs (안전 확률)
        p_t = torch.where(targets == 1, inputs, 1 - inputs)
        
        # 3. Focal term 계산: (1 - p_t)^gamma
        #    - 예측이 확실하면 (p_t 높음): focal_term 작음 → 손실 감소
        #    - 예측이 불확실하면 (p_t 낮음): focal_term 큼 → 손실 증가
        focal_term = (1 - p_t) ** self.gamma
        
        # 4. Alpha weighting (클래스 불균형 보정)
        #    - 유해(1) 샘플: alpha 가중치
        #    - 안전(0) 샘플: (1-alpha) 가중치
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # 5. 최종 Focal Loss 계산
        loss = alpha_t * focal_term * BCE_loss
        
        return loss.mean()

# ============================================================
# 데이터 처리 함수들
# ============================================================
def compute_image_hash(image_path):
    """
    이미지 파일의 해시값을 계산하여 중복 파일 감지
    
    MD5 해시를 사용하여 바이트 단위로 동일한 파일을 찾아냅니다.
    데이터 누수를 방지하고 학습 효율을 높이는 데 필수적.
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        str: 이미지의 MD5 해시값 (32자리 16진수 문자열)
        None: 파일 읽기 실패 시
    """
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def remove_duplicate_images(image_paths, labels):
    """
    중복 이미지를 제거하여 데이터 누수 방지
    
    같은 이미지가 train/val set에 모두 존재하면 데이터 누수가 발생하여
    검증 성능이 부정확하게 측정됩니다. 해시 기반 중복 제거로 이를 방지.
    
    Args:
        image_paths: 이미지 경로 리스트
        labels: 라벨 리스트
        
    Returns:
        unique_paths: 중복 제거된 이미지 경로 리스트
        unique_labels: 중복 제거된 라벨 리스트
    """
    seen_hashes = set()      # 이미 본 해시값 저장
    unique_paths = []        # 고유한 이미지 경로
    unique_labels = []       # 고유한 이미지 라벨
    duplicates = 0           # 중복 개수 카운터
    
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

def validate_video(video_path):
    """
    비디오 파일의 유효성을 검사
    
    손상되거나 읽을 수 없는 비디오를 사전에 필터링하여
    학습 중 에러를 방지합니다.
    
    Args:
        video_path: 비디오 파일 경로
        
    Returns:
        bool: 유효하면 True, 아니면 False
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames > 0
    except:
        return False

# ============================================================
# CLIP 기반 행동 감지 함수
# ============================================================
def detect_behavior_with_clip_fast(image_or_frames, clip_model, clip_preprocess):
    """
    CLIP 기반 행동 감지 (Zero-shot Learning)
    
    CLIP의 텍스트-이미지 매칭 능력을 활용하여 추가 라벨링 없이
    7가지 유해 행동을 자동으로 감지합니다.
    
    처리 전략:
    - 단일 프롬프트 사용 (행동당 1개)
    - 최대 4개 프레임 샘플링 (효율성)
    
    Args:
        image_or_frames: PIL Image 또는 프레임 리스트
        clip_model: CLIP 모델
        clip_preprocess: CLIP 전처리 함수
        
    Returns:
        behavior_scores: 행동별 점수 딕셔너리 (0~1 정규화)
                        예: {'smoking': 0.8, 'drinking': 0.3, ...}
    """
    behavior_scores = {}
    
    try:
        # 입력을 리스트로 변환
        if isinstance(image_or_frames, Image.Image):
            frames = [image_or_frames]
        else:
            frames = image_or_frames
        
        # 각 행동에 대해 CLIP 유사도 계산
        for behavior, prompt in BEHAVIOR_PROMPTS.items():
            # 단일 프롬프트로 텍스트 임베딩 생성
            text = clip.tokenize([prompt]).to(DEVICE)
            
            frame_scores = []
            # 최대 4개 프레임 샘플링
            for frame in frames[:min(len(frames), 4)]:
                image_input = clip_preprocess(frame).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    # 이미지 특징 추출
                    image_features = clip_model.encode_image(image_input)
                    # 텍스트 특징 추출
                    text_features = clip_model.encode_text(text)
                    
                    # CLIP 특징 L2 정규화 (코사인 유사도 계산 안정화)
                    image_features = F.normalize(image_features, p=2, dim=-1)
                    text_features = F.normalize(text_features, p=2, dim=-1)
                    
                    # 코사인 유사도 계산 (내적)
                    similarity = (image_features @ text_features.T).squeeze()
                    frame_scores.append(similarity.item())
            
            # 여러 프레임의 평균 점수
            behavior_scores[behavior] = np.mean(frame_scores)
        
        # 점수 정규화 (0~1 범위로)
        min_score = min(behavior_scores.values())
        max_score = max(behavior_scores.values())
        if max_score > min_score:
            for behavior in behavior_scores:
                behavior_scores[behavior] = (behavior_scores[behavior] - min_score) / (max_score - min_score)
    
    except Exception as e:
        print(f"  [행동 감지 오류] {e}")
        # 에러 시 모든 행동 점수를 0으로 설정
        behavior_scores = {behavior: 0.0 for behavior in HARMFUL_BEHAVIORS}
    
    return behavior_scores

def infer_behavior_from_objects(object_counts):
    """
    객체 기반 행동 추론 (휴리스틱 규칙)
    
    YOLO로 탐지된 객체를 바탕으로 특정 행동을 추론합니다.
    CLIP의 Zero-shot 감지를 보완하는 역할.
    
    규칙 예시:
    - cigarette 감지 → smoking 추론
    - bottle + person ≥ 2 → drinking 추론
    - knife + person → threatening 추론
    
    Args:
        object_counts: 객체별 감지 개수 딕셔너리
                      예: {'cigarette': 2, 'person': 1, ...}
        
    Returns:
        inferred_behaviors: 추론된 행동 리스트
                           예: ['smoking', 'drinking']
    """
    inferred_behaviors = []
    
    # 규칙 1: 담배 감지 → 흡연
    if object_counts.get('cigarette', 0) > 0:
        inferred_behaviors.append('smoking')
    
    # 규칙 2: 음주 관련 객체 1개 이상 → 음주 (wine glass, beer만 사용 - bottle, cup 제거)
    drinking_objects = ['wine glass', 'beer']
    if sum(object_counts.get(obj, 0) for obj in drinking_objects) >= 1:
        inferred_behaviors.append('drinking')
    
    # 규칙 3: 주사기 감지 → 약물 사용
    drug_objects = ['syringe']
    if sum(object_counts.get(obj, 0) for obj in drug_objects) > 0:
        inferred_behaviors.append('drug_use')
    
    # 규칙 4: 무기 + 사람 → 위협
    weapon_objects = ['knife', 'gun', 'pistol', 'rifle', 'sword', 'axe']
    if sum(object_counts.get(obj, 0) for obj in weapon_objects) > 0:
        if object_counts.get('person', 0) > 0:
            inferred_behaviors.append('threatening')
    
    return inferred_behaviors

# ============================================================
# 이미지 데이터셋
# ============================================================
class HarmfulImageDataset(Dataset):
    """
    유해 이미지 탐지 데이터셋 (행동 인식 포함)
    
    각 이미지에서 다음을 추출:
    1. YOLO 특징 (19차원): 객체 탐지 결과
    2. CLIP 특징 (512차원): 이미지 맥락 이해
    3. 행동 특징 (7차원): Zero-shot + 규칙 기반 행동 감지
    
    총 입력 차원: 19 + 512 + 7 = 538차원 → 차원 축소 → 256차원
    """
    def __init__(self, image_paths, labels, yolo_model, clip_model, clip_preprocess, augment=False):
        """
        Args:
            image_paths: 이미지 경로 리스트
            labels: 라벨 리스트 (0: 안전, 1: 유해)
            yolo_model: YOLOv8 모델
            clip_model: CLIP 모델
            clip_preprocess: CLIP 전처리 함수
            augment: 데이터 증강 여부 (기본값: False)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.yolo = yolo_model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.augment = augment
        
        # 데이터 증강 변환 정의 (학습 시에만 적용)
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),                    # 좌우 반전
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # 색상 변화
            T.RandomRotation(degrees=15),                      # 회전
        ])
    
    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        인덱스에 해당하는 데이터 샘플 반환
        
        Args:
            idx: 데이터 인덱스
            
        Returns:
            combined: 결합된 특징 벡터 (538차원)
            label: 라벨 (0 or 1)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 이미지 로딩
            image = Image.open(img_path).convert('RGB')
            original_image = image.copy()  # 행동 감지용 원본 보존
            
            # ------------------------------------------------------------
            # 1. YOLO 특징 추출 (19차원)
            # ------------------------------------------------------------
            yolo_results = self.yolo(img_path, verbose=False)
            yolo_features, object_counts = self._extract_yolo_features(yolo_results)
            
            # ------------------------------------------------------------
            # 2. CLIP 특징 추출 (512차원)
            # ------------------------------------------------------------
            # 데이터 증강 적용 (학습 시에만)
            if self.augment:
                try:
                    image = self.aug_transform(image)
                except:
                    pass  # 증강 실패 시 원본 사용
            
            # CLIP 전처리 및 특징 추출
            clip_image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                clip_features = self.clip_model.encode_image(clip_image).squeeze()
                # CLIP 특징 정규화
                clip_features = F.normalize(clip_features, p=2, dim=-1)
                clip_features = clip_features.cpu()
            
            # ------------------------------------------------------------
            # 3. 행동 특징 추출 (7차원)
            # ------------------------------------------------------------
            # CLIP 기반 Zero-shot 행동 감지
            behavior_scores = detect_behavior_with_clip_fast(
                original_image, self.clip_model, self.clip_preprocess
            )
            
            # 객체 기반 행동 추론
            inferred_behaviors = infer_behavior_from_objects(object_counts)
            
            # 두 방법의 결과를 결합 (CLIP 70% + 규칙 30%)
            behavior_features = torch.zeros(len(HARMFUL_BEHAVIORS))
            for i, behavior in enumerate(HARMFUL_BEHAVIORS):
                clip_score = behavior_scores.get(behavior, 0.0)
                rule_score = 1.0 if behavior in inferred_behaviors else 0.0
                behavior_features[i] = 0.7 * clip_score + 0.3 * rule_score
            
            # ------------------------------------------------------------
            # 4. 전체 특징 결합 (19 + 512 + 7 = 538차원)
            # ------------------------------------------------------------
            combined = torch.cat([yolo_features, clip_features, behavior_features])
            
            return combined, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 에러 시 zero 벡터 반환
            zero_dim = len(ALL_OBJECTS) + 512 + len(HARMFUL_BEHAVIORS)
            return torch.zeros(zero_dim), torch.tensor(label, dtype=torch.float32)
    
    def _extract_yolo_features(self, results):
        """
        YOLO 탐지 결과에서 특징 벡터 추출
        
        각 객체 카테고리별로 탐지된 개수를 세어 특징 벡터를 구성합니다.
        
        Args:
            results: YOLO 탐지 결과
            
        Returns:
            feature_vector: 객체별 탐지 개수 벡터 (19차원)
            object_counts: 객체별 탐지 개수 딕셔너리
        """
        feature_vector = torch.zeros(len(ALL_OBJECTS))
        object_counts = {}
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)].lower()
                    # 정의된 객체 목록과 매칭
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
    유해 비디오 탐지 데이터셋 (행동 인식 포함)
    
    각 비디오에서 32개 프레임을 샘플링하고, 프레임별로 다음을 추출:
    1. YOLO 특징 (19차원): 객체 탐지 결과
    2. CLIP 특징 (512차원): 프레임 맥락 이해
    3. SlowFast 특징 (400차원): 시간적 행동 패턴
    4. 행동 특징 (7차원): Zero-shot + 규칙 기반 행동 감지
    
    총 입력 차원: 19 + 512 + 400 + 7 = 938차원/프레임
    시퀀스 차원: (32, 938) → 차원 축소 → (32, 256)
    """
    def __init__(self, video_paths, labels, yolo_model, slowfast_model, clip_model, clip_preprocess, slowfast_dim):
        """
        Args:
            video_paths: 비디오 경로 리스트
            labels: 라벨 리스트 (0: 안전, 1: 유해)
            yolo_model: YOLOv8 모델
            slowfast_model: SlowFast 모델
            clip_model: CLIP 모델
            clip_preprocess: CLIP 전처리 함수
            slowfast_dim: SlowFast 출력 차원 (보통 400)
        """
        self.video_paths = video_paths
        self.labels = labels
        self.yolo = yolo_model
        self.slowfast = slowfast_model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.slowfast_dim = slowfast_dim

    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 비디오 데이터 샘플 반환
        
        Args:
            idx: 데이터 인덱스
            
        Returns:
            combined: 결합된 특징 시퀀스 (32, 938차원)
            label: 라벨 (0 or 1)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # 안전한 프레임 추출 (손상된 프레임 자동 스킵)
            frames_pil, frame_tensors = self._extract_frames_safe(video_path)
            
            if len(frames_pil) == 0:
                raise ValueError("No frames extracted")
            
            # ------------------------------------------------------------
            # 1. YOLO 특징 추출 (프레임별 19차원)
            # ------------------------------------------------------------
            yolo_features_list = []
            all_object_counts = {}  # 전체 비디오의 객체 카운트
            
            for frame_pil in frames_pil:
                frame_np = np.array(frame_pil)
                yolo_results = self.yolo(frame_np, verbose=False)
                yolo_feat, obj_counts = self._extract_yolo_features(yolo_results)
                yolo_features_list.append(yolo_feat)
                
                # 객체 카운트 누적
                for obj, count in obj_counts.items():
                    all_object_counts[obj] = all_object_counts.get(obj, 0) + count
            
            yolo_features_seq = torch.stack(yolo_features_list)  # (32, 19)
            
            # ------------------------------------------------------------
            # 2. CLIP 특징 추출 (프레임별 512차원)
            # ------------------------------------------------------------
            clip_features_list = []
            for frame_pil in frames_pil:
                clip_image = self.clip_preprocess(frame_pil).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    clip_feat = self.clip_model.encode_image(clip_image).squeeze()
                    # CLIP 특징 정규화
                    clip_feat = F.normalize(clip_feat, p=2, dim=-1)
                    clip_features_list.append(clip_feat.cpu())
            
            clip_features_seq = torch.stack(clip_features_list)  # (32, 512)
            
            # ------------------------------------------------------------
            # 3. SlowFast 특징 추출 (비디오 전체 400차원)
            # ------------------------------------------------------------
            slowfast_features = self._extract_slowfast_features(frame_tensors)
            # 모든 프레임에 동일한 SlowFast 특징 브로드캐스트
            slowfast_features_seq = slowfast_features.unsqueeze(0).expand(FRAME_SAMPLE, -1)  # (32, 400)
            
            # ------------------------------------------------------------
            # 4. 행동 특징 추출 (7차원)
            # ------------------------------------------------------------
            # CLIP 기반 Zero-shot 행동 감지 (여러 프레임 사용)
            behavior_scores = detect_behavior_with_clip_fast(
                frames_pil, self.clip_model, self.clip_preprocess
            )
            
            # 객체 기반 행동 추론
            inferred_behaviors = infer_behavior_from_objects(all_object_counts)
            
            # 두 방법의 결과를 결합 (CLIP 80% + 규칙 20%, 비디오는 CLIP 비중 더 높임)
            behavior_features = torch.zeros(len(HARMFUL_BEHAVIORS))
            for i, behavior in enumerate(HARMFUL_BEHAVIORS):
                clip_score = behavior_scores.get(behavior, 0.0)
                rule_score = 1.0 if behavior in inferred_behaviors else 0.0
                behavior_features[i] = 0.8 * clip_score + 0.2 * rule_score
            
            # 모든 프레임에 동일한 행동 특징 브로드캐스트
            behavior_features_seq = behavior_features.unsqueeze(0).expand(FRAME_SAMPLE, -1)  # (32, 7)
            
            # ------------------------------------------------------------
            # 5. 전체 특징 결합 (32, 19+512+400+7 = 32, 938차원)
            # ------------------------------------------------------------
            combined = torch.cat([
                yolo_features_seq,
                clip_features_seq,
                slowfast_features_seq,
                behavior_features_seq
            ], dim=1)
            
            return combined, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # 에러 시 zero 텐서 반환
            zero_dim = len(ALL_OBJECTS) + 512 + self.slowfast_dim + len(HARMFUL_BEHAVIORS)
            return torch.zeros(FRAME_SAMPLE, zero_dim), torch.tensor(label, dtype=torch.float32)
    
    def _extract_frames_safe(self, video_path):
        """
        안전한 프레임 추출 (손상된 프레임 자동 스킵)
        
        비디오에서 32개 프레임을 균등하게 샘플링하되,
        손상된 프레임은 자동으로 건너뛰어 에러를 방지합니다.
        
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
        
        # 균등 샘플링 인덱스 계산
        indices = np.linspace(0, total_frames - 1, FRAME_SAMPLE, dtype=int)
        
        frames_pil = []
        frame_tensors = []
        
        for idx in indices:
            try:
                # 프레임 위치 설정
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # BGR → RGB 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frames_pil.append(frame_pil)
                    
                    # 텐서 변환 (SlowFast 입력용)
                    frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                    frame_tensor = T.Resize((256, 256))(frame_tensor)
                    frame_tensors.append(frame_tensor)
            except Exception as e:
                # 에러 프레임 스킵 (출력 없이 조용히 처리)
                continue
        
        cap.release()
        
        # 프레임 부족 시 반복하여 32개 채우기
        while len(frame_tensors) < FRAME_SAMPLE:
            frame_tensors.extend(frame_tensors[:min(len(frame_tensors), FRAME_SAMPLE - len(frame_tensors))])
            frames_pil.extend(frames_pil[:min(len(frames_pil), FRAME_SAMPLE - len(frames_pil))])
        
        # 정확히 32개만 반환
        frame_tensors = frame_tensors[:FRAME_SAMPLE]
        frames_pil = frames_pil[:FRAME_SAMPLE]
        
        return frames_pil, frame_tensors
    
    def _extract_slowfast_features(self, frame_tensors):
        """
        SlowFast 모델로 비디오 행동 특징 추출
        
        SlowFast는 두 개의 pathway를 사용:
        - Slow pathway: 8 프레임, 저주파 정보 (의미론적 이해)
        - Fast pathway: 32 프레임, 고주파 정보 (빠른 움직임)
        
        Args:
            frame_tensors: 프레임 텐서 리스트 (32개)
            
        Returns:
            features: SlowFast 특징 벡터 (400차원)
        """
        try:
            # ImageNet 정규화
            mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
            frame_tensors_normalized = [(f - mean) / std for f in frame_tensors]
            
            # Fast pathway: 32 프레임 전체 사용
            fast_pathway = torch.stack(frame_tensors_normalized).unsqueeze(0).to(DEVICE)
            fast_pathway = fast_pathway.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            
            # Slow pathway: 8 프레임 균등 샘플링
            slow_indices = torch.linspace(0, 31, 8).long()
            slow_tensors = [frame_tensors_normalized[i] for i in slow_indices]
            slow_pathway = torch.stack(slow_tensors).unsqueeze(0).to(DEVICE)
            slow_pathway = slow_pathway.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            
            # SlowFast 특징 추출
            with torch.no_grad():
                features = self.slowfast([slow_pathway, fast_pathway])
                features = features.squeeze().cpu()
            
            return features
            
        except Exception as e:
            print(f"  [SlowFast 오류] {e}")
            return torch.zeros(self.slowfast_dim)
    
    def _extract_yolo_features(self, results):
        """
        YOLO 탐지 결과에서 특징 벡터 추출
        
        Args:
            results: YOLO 탐지 결과
            
        Returns:
            feature_vector: 객체별 탐지 개수 벡터 (19차원)
            object_counts: 객체별 탐지 개수 딕셔너리
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
    유해 이미지 분류 모델 (차원 축소 포함)
    
    아키텍처:
    입력 (538차원) → 차원 축소 (256차원) → MLP → 출력 (1차원, 유해 확률)
    
    차원 축소의 이점:
    1. 과적합 방지: 파라미터 수 감소로 일반화 능력 향상
    2. 학습 효율: 작은 차원에서 빠른 학습
    3. 정보 압축: 중요한 특징만 보존
    """
    def __init__(self, yolo_dim, clip_dim, behavior_dim):
        """
        Args:
            yolo_dim: YOLO 특징 차원 (19)
            clip_dim: CLIP 특징 차원 (512)
            behavior_dim: 행동 특징 차원 (7)
        """
        super().__init__()
        input_dim = yolo_dim + clip_dim + behavior_dim  # 538
        
        # 차원 축소 레이어 (538 → 256)
        # 과적합 방지 및 학습 효율 향상
        self.dimension_reduction = nn.Sequential(
            nn.Linear(input_dim, 256),   # 선형 변환
            nn.ReLU(),                    # 활성화 함수
            nn.BatchNorm1d(256),          # 배치 정규화 (안정적 학습)
            nn.Dropout(0.5),              # 드롭아웃 (과적합 방지)
        )
        
        # MLP 분류기 (3층 구조)
        self.mlp = nn.Sequential(
            # 첫 번째 은닉층
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            
            # 두 번째 은닉층
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),              # 두 번째 층은 드롭아웃 낮춤
            
            # 출력층
            nn.Linear(64, 1),
            nn.Sigmoid()                  # 이진 분류를 위한 시그모이드
        )
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 특징 (batch_size, 538)
            
        Returns:
            output: 유해 확률 (batch_size,) - 0~1 범위
        """
        # 1. 차원 축소 (538 → 256)
        x = self.dimension_reduction(x)
        
        # 2. MLP 분류 (256 → 1)
        return self.mlp(x).squeeze()

# ============================================================
# 비디오 분류 모델
# ============================================================
class HarmfulVideoClassifier(nn.Module):
    """
    유해 비디오 분류 모델 (차원 축소 + Transformer)
    
    아키텍처:
    입력 (32, 938차원) → 차원 축소 (32, 256차원) → Transformer → 시간 평균 풀링 → MLP → 출력 (1차원)
    
    Transformer의 역할:
    - 프레임 간 시간적 관계 학습 (self-attention)
    - 장거리 의존성 모델링 (long-range dependencies)
    - 비선형적 시간 패턴 포착
    
    차원 축소의 이점:
    - 8-head attention 사용 가능 (256은 8로 나누어떨어짐)
    - 과적합 방지 및 학습 효율 향상
    - Transformer 계산 복잡도 감소
    """
    def __init__(self, yolo_dim, clip_dim, slowfast_dim, behavior_dim):
        """
        Args:
            yolo_dim: YOLO 특징 차원 (19)
            clip_dim: CLIP 특징 차원 (512)
            slowfast_dim: SlowFast 특징 차원 (400)
            behavior_dim: 행동 특징 차원 (7)
        """
        super().__init__()
        input_dim = yolo_dim + clip_dim + slowfast_dim + behavior_dim  # 938
        
        # 차원 축소 레이어 (938 → 256)
        # Transformer 효율 향상 및 과적합 방지
        reduced_dim = 256
        self.dimension_reduction = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),  # 선형 변환
            nn.ReLU(),                           # 활성화 함수
            nn.Dropout(0.4),                     # 드롭아웃 (비디오는 복잡해서 과적합 위험)
        )
        
        # Transformer Encoder (8-head attention)
        # 256 차원은 8로 나누어떨어져 8개의 attention head 사용 가능
        nhead = 8
        print(f"✓ Transformer nhead: {nhead} (reduced_dim={reduced_dim})")
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=reduced_dim,           # 입력/출력 차원
            nhead=nhead,                   # attention head 개수 (8개로 다양한 관점 학습)
            dim_feedforward=512,           # feedforward 은닉층 차원
            batch_first=True,              # 배치를 첫 번째 차원으로
            dropout=0.4                    # 드롭아웃
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)  # 2층 Transformer
        
        # MLP 분류기 (2층 구조)
        self.classifier = nn.Sequential(
            # 첫 번째 은닉층
            nn.Linear(reduced_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # 출력층
            nn.Linear(128, 1)
            # Sigmoid는 forward에서 적용
        )
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 특징 시퀀스 (batch_size, 32, 938)
            
        Returns:
            output: 유해 확률 (batch_size,) - 0~1 범위
        """
        # 1. 차원 축소 (각 프레임별로)
        batch_size, seq_len, feat_dim = x.shape
        x = x.view(-1, feat_dim)  # (batch*32, 938)
        x = self.dimension_reduction(x)  # (batch*32, 256)
        x = x.view(batch_size, seq_len, -1)  # (batch, 32, 256)
        
        # 2. Transformer로 시간적 관계 학습
        transformed = self.transformer(x)  # (batch, 32, 256)
        
        # 3. 시간 평균 풀링 (32개 프레임의 평균)
        pooled = transformed.mean(dim=1)  # (batch, 256)
        
        # 4. MLP 분류
        logits = self.classifier(pooled)  # (batch, 1)
        
        # 5. Sigmoid 적용
        return torch.sigmoid(logits).squeeze()  # (batch,)

# ============================================================
# Label Smoothing Loss
# ============================================================
class LabelSmoothingBCELoss(nn.Module):
    """
    Label Smoothing을 적용한 Binary Cross Entropy Loss
    
    Label Smoothing의 효과:
    - 과신 방지: 모델이 너무 확신하는 예측을 방지
    - 일반화 향상: 경계 영역의 불확실성 학습
    - 과적합 완화: 라벨 노이즈에 대한 robustness 향상
    
    예시: smoothing=0.2 적용 시
    - 원본 라벨: 0 → 0.1, 1 → 0.9
    - 모델이 0.9 확률만 달성해도 충분히 학습된 것으로 간주
    """
    def __init__(self, smoothing=0.2):
        """
        Args:
            smoothing: smoothing 강도 (0~1)
                      0.2 = 라벨의 20%를 불확실하게 만듦
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred: 모델 예측 (0~1)
            target: 원본 라벨 (0 or 1)
            
        Returns:
            loss: Label Smoothing이 적용된 BCE Loss
        """
        # Label Smoothing 적용
        # target=1: 1 → 0.9 (1 - 0.2/2)
        # target=0: 0 → 0.1 (0 + 0.2/2)
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy(pred, target_smooth)

# ============================================================
# 데이터 준비 함수들
# ============================================================
def prepare_image_data():
    """
    이미지 데이터 준비 및 전처리 (공개 데이터만 사용)
    
    데이터 소스:
    1. HOD Dataset: 공개 유해 이미지 데이터셋
    2. COCO Safe Dataset: 안전한 일반 이미지
    
    전처리:
    - 중복 이미지 제거 (해시 기반)
    - Stratified 분할 (클래스 비율 유지)
    - Train 85% / Validation 15%
    
    Returns:
        train_images: 학습 이미지 경로 리스트
        train_labels: 학습 라벨 리스트
        val_images: 검증 이미지 경로 리스트
        val_labels: 검증 라벨 리스트
    """
    print("\n이미지 데이터 준비 중 (공개 데이터만 사용)...")
    
    # ------------------------------------------------------------
    # 1. HOD Dataset (공개 유해 이미지)
    # ------------------------------------------------------------
    hod_path = DATA_PATH + '1_공개_데이터셋/HOD_Dataset/dataset/'
    hod_images = []
    hod_labels = []
    if os.path.exists(hod_path):
        for root, dirs, files in os.walk(hod_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    hod_images.append(os.path.join(root, file))
                    hod_labels.append(1)  # 유해
    
    # ------------------------------------------------------------
    # 2. COCO Safe Dataset (안전한 일반 이미지)
    # ------------------------------------------------------------
    coco_path = DATA_PATH + '1_공개_데이터셋/COCO_Safe_Dataset/'
    coco_images = []
    coco_labels = []
    if os.path.exists(coco_path):
        for root, dirs, files in os.walk(coco_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    coco_images.append(os.path.join(root, file))
                    coco_labels.append(0)  # 안전
    
    # 전체 데이터 통합 (공개 데이터만)
    X = hod_images + coco_images
    y = hod_labels + coco_labels
    
    print(f"✓ 중복 제거 전: {len(X)}개 이미지")
    
    # 중복 이미지 제거 (데이터 누수 방지)
    X, y = remove_duplicate_images(X, y)
    print(f"✓ 중복 제거 후: {len(X)}개 이미지")
    
    # Train/Validation 분할 (Stratified - 클래스 비율 유지)
    from sklearn.model_selection import train_test_split
    train_images, val_images, train_labels, val_labels = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    
    print(f"✓ 데이터셋 통합 완료 (학습 {len(train_images)}, 검증 {len(val_images)})")
    print(f"  (유해: {sum(train_labels)}, 안전: {len(train_labels)-sum(train_labels)} / 검증 유해: {sum(val_labels)}, 안전: {len(val_labels)-sum(val_labels)})")
    
    return train_images, train_labels, val_images, val_labels

def prepare_video_data():
    """
    비디오 데이터 준비 및 전처리 (공개 데이터만 사용)
    
    데이터 소스:
    1. 공개 비디오 데이터셋 (RWF-2000, RLVS 등)
    
    전처리:
    - 비디오 유효성 검사 (손상된 파일 필터링)
    - Stratified 분할 (클래스 비율 유지)
    - Train 85% / Validation 15%
    
    Returns:
        train_videos: 학습 비디오 경로 리스트
        train_labels: 학습 라벨 리스트
        val_videos: 검증 비디오 경로 리스트
        val_labels: 검증 라벨 리스트
    """
    print("\n비디오 데이터 준비 중 (공개 데이터만 사용)...")
    
    pvpaths, pvlabels = [], [] # 공개 데이터셋
    
    # ------------------------------------------------------------
    # 1. 공개 비디오 데이터셋 (RWF-2000, RLVS 등)
    # ------------------------------------------------------------
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
    
    # 전체 데이터 통합 (공개 데이터만)
    X = pvpaths
    y = pvlabels
    
    # ------------------------------------------------------------
    # 비디오 유효성 검증 (손상된 파일 필터링)
    # ------------------------------------------------------------
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
    
    # Train/Validation 분할 (Stratified)
    from sklearn.model_selection import train_test_split
    train_videos, val_videos, train_labels, val_labels = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    
    print(f"✓ 통합 완료 (학습 {len(train_videos)}, 검증 {len(val_videos)})")
    print(f"  (유해: {sum(train_labels)}, 안전: {len(train_labels)-sum(train_labels)} / 검증 유해: {sum(val_labels)}, 안전: {len(val_labels)-sum(val_labels)})")
    
    return train_videos, train_labels, val_videos, val_labels

# ============================================================
# 모델 학습 함수
# ============================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, model_type='image'):
    """
    모델 학습 함수 (Validation Loss 기준 Early Stopping)
    
    학습 과정:
    1. 에포크마다 학습 및 검증 수행
    2. 여러 threshold에서 성능 평가하여 최적값 선택
    3. Validation Loss 기준으로 Early Stopping 적용 (과최적화 방지)
    4. Learning Rate 감소 (ReduceLROnPlateau)
    
    Early Stopping 전략:
    - Validation Loss 기준 사용 (Threshold 과최적화 방지)
    - Patience=2 (2 에포크 동안 개선 없으면 중단)
    - 일반화 능력 향상에 효과적
    
    Args:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        criterion: 손실 함수 (Focal Loss)
        optimizer: 옵티마이저 (Adam)
        scheduler: 학습률 스케줄러
        epochs: 최대 에포크 수
        model_type: 모델 타입 ('image' or 'video')
        
    Returns:
        best_f1: 최고 F1-Score
        best_threshold: 최적 threshold
    """
    best_f1 = 0
    best_threshold = 0.5
    best_val_loss = float('inf')  # ← Val Loss 기준 추가
    patience, patience_count = 2, 0  # Early Stopping 설정
    
    for epoch in range(epochs):
        # ========================================
        # 학습 모드
        # ========================================
        model.train()
        train_loss = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE).squeeze()  # labels도 squeeze 적용
            
            # 순전파
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            
            # 배치 크기 1인 경우 처리
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            # 역전파 및 가중치 업데이트
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # ========================================
        # 검증 모드
        # ========================================
        model.eval()
        val_outputs = []
        val_true = []
        val_loss = 0
        
        with torch.no_grad():
            for features, labels_batch in val_loader:
                features = features.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE).squeeze()  # labels도 squeeze 적용
                
                outputs = model(features).squeeze()
                
                # 배치 크기 1인 경우 처리
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    labels_batch = labels_batch.unsqueeze(0)
                
                # Validation Loss 계산
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                
                # 예측값 저장
                outputs_np = outputs.cpu().numpy()
                
                if outputs_np.ndim == 0:
                    val_outputs.append(float(outputs_np))
                else:
                    val_outputs.extend(outputs_np.tolist())
                
                # labels numpy 변환 (배치 크기 1 처리 포함)
                labels_np = labels_batch.cpu().numpy()
                if labels_np.ndim == 0:
                    labels_np = np.array([labels_np])
                val_true.extend(labels_np.tolist())
        
        val_loss /= len(val_loader)
        
        val_outputs_np = np.array(val_outputs)
        val_true_np = np.array(val_true)
        
        # ========================================
        # Threshold 분석 (최적 임계값 찾기)
        # ========================================
        print(f"\n[Threshold 분석 @ Epoch {epoch+1}]")
        best_th = 0.5
        best_th_f1 = 0
        
        # 여러 threshold에서 F1-Score 계산
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
        
        # ========================================
        # 에포크 결과 출력
        # ========================================
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  최적 Threshold: {best_th:.2f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # ========================================
        # Learning Rate Scheduler (F1 기준)
        # ========================================
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(f1)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  ⚡ Learning Rate 감소: {old_lr:.6f} → {new_lr:.6f}")
        
        # ========================================
        # Early Stopping (Validation Loss 기준)
        # ========================================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_f1 = f1
            best_threshold = best_th
            patience_count = 0
            
            # 모델 저장
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': best_threshold,
                'best_val_loss': best_val_loss
            }, f'{model_type}_model_best.pth')
            print(f"  ✓ 모델 저장 (Best Val Loss: {best_val_loss:.4f}, F1: {best_f1:.4f}, Threshold: {best_threshold:.2f})")
        else:
            patience_count += 1
        
        # Early Stopping 체크
        if patience_count >= patience:
            print(f"Early stopping at epoch {epoch+1} (Val Loss 기준)")
            print(f"  Best Val Loss: {best_val_loss:.4f}, Best F1: {best_f1:.4f}")
            break
    
    return best_f1, best_threshold

# ============================================================
# Confusion Matrix 시각화
# ============================================================
def plot_confusion_matrix(model, val_loader, threshold, model_type='image'):
    """
    Confusion Matrix를 생성하고 이미지로 저장
    
    Confusion Matrix는 다음 정보를 제공:
    - True Positive (TP): 유해를 유해로 정확히 예측
    - True Negative (TN): 안전을 안전으로 정확히 예측
    - False Positive (FP): 안전을 유해로 잘못 예측 (Type I 오류)
    - False Negative (FN): 유해를 안전으로 잘못 예측 (Type II 오류, 더 위험)
    
    Args:
        model: 학습된 모델
        val_loader: 검증 데이터 로더
        threshold: 분류 임계값
        model_type: 모델 타입 ('image' or 'video')
    """
    model.eval()
    all_preds = []
    all_true = []
    
    # 전체 검증 데이터에 대해 예측
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(DEVICE)
            labels = labels.squeeze()  # labels squeeze 적용
            outputs = model(features).squeeze()
            
            outputs_np = outputs.cpu().numpy()
            if outputs_np.ndim == 0:
                outputs_np = np.array([outputs_np])
            
            # Threshold 적용하여 이진 분류
            preds = (outputs_np > threshold).astype(int)
            all_preds.extend(preds.tolist())
            
            # labels numpy 변환
            labels_np = labels.cpu().numpy()
            if labels_np.ndim == 0:
                labels_np = np.array([labels_np])
            all_true.extend(labels_np.tolist())
    
    # Confusion Matrix 계산
    cm = confusion_matrix(all_true, all_preds)
    
    # 시각화
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
    print("무하유 유해 콘텐츠 탐지 시스템 - 최종 버전")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"학습 데이터: 공개 데이터셋만 사용 (독립 평가용)")
    print(f"Harmful Objects: {len(ALL_OBJECTS)}개")
    print(f"Harmful Behaviors: {len(HARMFUL_BEHAVIORS)}개")
    print(f"Image Epochs: {IMAGE_EPOCHS}")
    print(f"Video Epochs: {VIDEO_EPOCHS}")
    print(f"✓ 로그 파일: {log_filename}")
    print("="*60)
    
    # ============================================================
    # 라이브러리 버전 기록 (재현성 검증용)
    # ============================================================
    print(f"\n[라이브러리 버전]")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"Python: {sys.version.split()[0]}")
    
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
    
    # 차원 정보 출력
    yolo_dim = len(ALL_OBJECTS)
    clip_dim = 512
    behavior_dim = len(HARMFUL_BEHAVIORS)
    total_dim = yolo_dim + clip_dim + behavior_dim
    print(f"✓ YOLO 차원: {yolo_dim}")
    print(f"✓ CLIP 차원: {clip_dim}")
    print(f"✓ 행동 차원: {behavior_dim}")
    print(f"✓ 총 입력 차원: {total_dim}")
    print(f"✓ 차원 축소 후: 256")
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = HarmfulImageDataset(train_images, train_labels, yolo, clip_model, clip_preprocess, augment=False)
    val_dataset = HarmfulImageDataset(val_images, val_labels, yolo, clip_model, clip_preprocess, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 생성
    image_model = HarmfulImageClassifier(yolo_dim, clip_dim, behavior_dim).to(DEVICE)
    
    # 클래스 분포 출력
    num_harmful = sum(train_labels)
    num_safe = len(train_labels) - num_harmful
    print(f"\n✓ 클래스 분포: 유해 {num_harmful}개, 안전 {num_safe}개 (비율: {num_safe/num_harmful:.2f})")
    
    # Focal Loss 사용
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    print("✓ 손실 함수: Focal Loss (alpha=0.25, gamma=2.0)")
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = optim.Adam(image_model.parameters(), lr=IMAGE_LR, weight_decay=IMAGE_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # 학습 시작
    best_image_f1, best_image_threshold = train_model(
        image_model, train_loader, val_loader, criterion, optimizer, scheduler, IMAGE_EPOCHS, 'image'
    )
    
    # Best 모델 로드
    checkpoint = torch.load('image_model_best.pth')
    image_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Best 모델 로드 완료 (Threshold: {checkpoint['best_threshold']:.2f})")
    
    # Confusion Matrix 생성
    plot_confusion_matrix(image_model, val_loader, best_image_threshold, 'image')
    
    print("\n" + "="*60)
    print("✅ 이미지 모델 학습 완료!")
    print(f"   Best F1-Score: {best_image_f1:.4f}")
    if best_image_f1 >= 0.75:
        print("   ✅ 필수 목표 달성! (F1 ≥ 0.75)")
    print("="*60)
    
    # ============================================================
    # 비디오 모델 학습
    # ============================================================
    print("\n" + "="*60)
    print("비디오 모델 학습")
    print("="*60)
    
    # 데이터 준비
    train_videos, train_vlabels, val_videos, val_vlabels = prepare_video_data()
    
    # 모델 로딩
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
    
    # 차원 정보 출력
    total_video_dim = yolo_dim + clip_dim + slowfast_dim + behavior_dim
    print(f"✓ YOLO 차원: {yolo_dim}")
    print(f"✓ CLIP 차원: {clip_dim}")
    print(f"✓ SlowFast 차원: {slowfast_dim}")
    print(f"✓ 행동 차원: {behavior_dim}")
    print(f"✓ 총 입력 차원: {total_video_dim}")
    print(f"✓ 차원 축소 후: 256")
    
    # 클래스 불균형 보정
    num_harmful_v = sum(train_vlabels)
    num_safe_v = len(train_vlabels) - num_harmful_v
    pos_weight = torch.tensor([num_safe_v / num_harmful_v]).to(DEVICE)
    print(f"\n✓ 클래스 불균형 보정: pos_weight={pos_weight.item():.4f}")
    print(f"  (안전: {num_safe_v}개, 유해: {num_harmful_v}개)")
    
    # 데이터셋 및 데이터로더 생성
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
    
    # Focal Loss with pos_weight
    criterion_video = FocalLoss(alpha=pos_weight.item(), gamma=2.0)
    print(f"✓ 손실 함수: Focal Loss (alpha={pos_weight.item():.4f}, gamma=2.0)")
    
    # 옵티마이저 및 스케줄러 설정
    optimizer_video = optim.Adam(video_model.parameters(), lr=VIDEO_LR, weight_decay=VIDEO_WEIGHT_DECAY)
    scheduler_video = optim.lr_scheduler.ReduceLROnPlateau(optimizer_video, mode='max', factor=0.5, patience=2)
    
    # 학습 시작
    best_video_f1, best_video_threshold = train_model(
        video_model, train_video_loader, val_video_loader, criterion_video, optimizer_video, scheduler_video, VIDEO_EPOCHS, 'video'
    )
    
    # Best 모델 로드
    checkpoint = torch.load('video_model_best.pth')
    video_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Best 모델 로드 완료 (Threshold: {checkpoint['best_threshold']:.2f})")
    
    # Confusion Matrix 생성
    plot_confusion_matrix(video_model, val_video_loader, best_video_threshold, 'video')
    
    print("\n" + "="*60)
    print("✅ 비디오 모델 학습 완료!")
    print(f"   Best F1-Score: {best_video_f1:.4f}")
    if best_video_f1 >= 0.75:
        print("   ✅ 필수 목표 달성! (F1 ≥ 0.75)")
    print("="*60)
    
    # ============================================================
    # 최종 요약
    # ============================================================
    print("\n" + "="*60)
    print("✅ 전체 학습 완료!")
    print("="*60)
    print(f"이미지 모델 Best F1: {best_image_f1:.4f} (Threshold: {best_image_threshold:.2f})")
    print(f"비디오 모델 Best F1: {best_video_f1:.4f} (Threshold: {best_video_threshold:.2f})")
    print(f"\n✓ 학습 데이터:")
    print(f"  - 공개 데이터셋만 사용 (독립 평가를 위해 직접 수집 데이터 제외)")
    print(f"  - 최종 평가: 1200개 데이터 (400개 × 3명)로 독립 평가 예정")
    print(f"\n✓ 주요 특징:")
    print(f"  - 차원 축소: {total_dim} → 256 (과적합 방지)")
    print(f"  - Focal Loss: 어려운 샘플 집중 학습")
    print(f"  - CLIP 정규화: F.normalize 적용")
    print(f"  - 재현성 보장: Seed 고정 (SEED=42)")
    print(f"  - Early Stopping: Val Loss 기준")
    print(f"  - 프레임 에러 처리: 강화된 예외 처리")
    print(f"\n✓ 로그 파일: {log_filename}")
    print("="*60)
