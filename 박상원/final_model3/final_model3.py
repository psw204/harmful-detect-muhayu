"""
무하유 유해 콘텐츠 탐지 시스템 - 개선 버전 3
이미지: YOLOv8 + CLIP + MLP
비디오: YOLOv8 + SlowFast + CLIP + Transformer

이 시스템은 이미지와 비디오에서 유해한 콘텐츠를 자동으로 탐지하는 AI 모델입니다.
- 이미지: YOLO로 객체 탐지 + CLIP으로 맥락 이해 + MLP로 분류
- 비디오: YOLO + SlowFast로 행동 인식 + CLIP + Transformer로 시계열 분석
"""

# ============================================================
# [변경점] final_model(3): Import 정리 및 그룹화
# ============================================================
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

# ============================================================
# 로깅 시스템 설정
# ============================================================
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
VIDEO_BATCH_SIZE = 2    # 비디오 학습 배치 크기 (메모리 제약으로 작게 설정)

# 학습 에포크 설정
IMAGE_EPOCHS = 10       # 이미지 모델 학습 에포크
VIDEO_EPOCHS = 10       # 비디오 모델 학습 에포크

# 학습률 설정
IMAGE_LR = 0.001        # 이미지 모델 학습률
VIDEO_LR = 0.0005       # 비디오 모델 학습률 (더 작은 학습률로 안정적 학습)

# 가중치 감쇠 (정규화)
IMAGE_WEIGHT_DECAY = 0.003   # 이미지 모델 가중치 감쇠
VIDEO_WEIGHT_DECAY = 0.003   # 비디오 모델 가중치 감쇠

# 비디오 처리 설정
FRAME_SAMPLE = 32       # 비디오에서 샘플링할 프레임 수

# 탐지할 유해 객체 목록
HARMFUL_OBJECTS = [
    'knife', 'gun', 'pistol', 'rifle', 'sword', 'axe',  # 무기류
    'hammer', 'dagger', 'machete',                       # 도구류
    'beer', 'cigarette'                                  # 음주/흡연 관련
]


# ============================================================
# 이미지 데이터셋
# ============================================================
class HarmfulImageDataset(Dataset):
    """
    유해 이미지 탐지를 위한 데이터셋 클래스
    
    YOLO와 CLIP 모델을 사용하여 이미지에서 특징을 추출하고,
    데이터 증강을 통해 학습 데이터의 다양성을 증가시킴
    """
    def __init__(self, image_paths, labels, yolo_model, clip_model, clip_preprocess, augment=False):
        self.image_paths = image_paths  # 이미지 파일 경로 리스트
        self.labels = labels           # 라벨 리스트 (0: 안전, 1: 유해)
        self.yolo = yolo_model         # YOLO 객체 탐지 모델
        self.clip_model = clip_model   # CLIP 멀티모달 모델
        self.clip_preprocess = clip_preprocess  # CLIP 전처리 함수
        self.augment = augment         # 데이터 증강 여부
        
        # 데이터 증강을 위한 변환 함수들
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(),                    # 수평 뒤집기
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # 색상 조정
            T.RandomRotation(degrees=15),                 # 회전
            T.RandomResizedCrop(224, scale=(0.75, 1.0)),  # 크기 조정 및 크롭
            T.RandomPerspective(distortion_scale=0.5, p=0.5),  # 원근 변환
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        인덱스에 해당하는 이미지와 라벨을 반환
        
        Args:
            idx: 데이터셋 인덱스
            
        Returns:
            combined: YOLO + CLIP 특징 벡터
            label: 라벨 (0 또는 1)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 이미지 로드 및 전처리
            image = Image.open(img_path).convert('RGB')
            if self.augment: 
                image = self.aug_transform(image)  # 데이터 증강 적용
            
            # YOLO 특징 추출
            yolo_results = self.yolo(img_path, verbose=False)
            yolo_features = self._extract_yolo_features(yolo_results)
            
            # CLIP 특징 추출
            clip_image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                clip_features = self.clip_model.encode_image(clip_image).squeeze()
            
            # YOLO와 CLIP 특징 결합
            combined = torch.cat([yolo_features.to(DEVICE), clip_features])
            return combined, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 오류 발생 시 제로 벡터 반환
            return torch.zeros(len(HARMFUL_OBJECTS) + 512).to(DEVICE), torch.tensor(label, dtype=torch.float32)
    
    def _extract_yolo_features(self, results):
        """
        YOLO 탐지 결과에서 유해 객체 특징 벡터 추출
        
        Args:
            results: YOLO 탐지 결과
            
        Returns:
            feature_vector: 유해 객체별 탐지 횟수 벡터
        """
        feature_vector = torch.zeros(len(HARMFUL_OBJECTS))
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)].lower()
                    # 유해 객체 목록과 매칭하여 특징 벡터 업데이트
                    for i, obj in enumerate(HARMFUL_OBJECTS):
                        if obj in class_name or class_name in obj:
                            feature_vector[i] += 1
        return feature_vector


# ============================================================
# 비디오 데이터셋
# ============================================================
class HarmfulVideoDataset(Dataset):
    """
    유해 비디오 탐지를 위한 데이터셋 클래스
    
    YOLO, SlowFast, CLIP 모델을 사용하여 비디오에서 시공간 특징을 추출
    각 프레임에서 YOLO와 CLIP 특징을 추출하고, SlowFast로 전체 비디오의 행동 특징을 추출
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
            combined: 시공간 특징 벡터 (프레임별 YOLO+CLIP + SlowFast)
            label: 라벨 (0 또는 1)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # 비디오에서 프레임 샘플링
            frames = self._sample_frames(video_path)
            if len(frames) != FRAME_SAMPLE:
                print(f"Warning: {video_path} frames {len(frames)} != {FRAME_SAMPLE}")

            # 각 프레임에서 YOLO 특징 추출
            yolo_list = [self._extract_yolo_frame(frame) for frame in frames]
            yolo_seq = torch.stack(yolo_list).to(DEVICE)

            # SlowFast로 전체 비디오의 행동 특징 추출
            slowfast_feat = self._extract_slowfast_features(frames)

            # 각 프레임에서 CLIP 특징 추출
            clip_list = [self._extract_clip_frame(frame) for frame in frames]
            clip_seq = torch.stack(clip_list).to(DEVICE)

            # 프레임별 YOLO+CLIP 특징 결합
            yolo_clip_seq = torch.cat([yolo_seq, clip_seq], dim=1)
            
            # SlowFast 특징을 모든 프레임에 복제하여 결합
            slowfast_expanded = slowfast_feat.unsqueeze(0).repeat(FRAME_SAMPLE, 1)
            combined = torch.cat([yolo_clip_seq, slowfast_expanded], dim=1)

            return combined, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시 제로 벡터 반환
            default_dim = len(HARMFUL_OBJECTS) + 512 + self.slowfast_dim
            return torch.zeros(FRAME_SAMPLE, default_dim).to(DEVICE), torch.tensor(label, dtype=torch.float32)

    def _sample_frames(self, video_path):
        """
        비디오에서 균등하게 프레임을 샘플링
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            frames: 샘플링된 프레임 리스트 (FRAME_SAMPLE 개)
        """
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        if total == 0:
            # 비디오가 비어있는 경우 제로 프레임으로 채움
            cap.release()
            frames = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(FRAME_SAMPLE)]
        else:
            # 비디오 전체 길이에서 균등하게 프레임 샘플링
            indices = np.linspace(0, max(0, total-1), FRAME_SAMPLE, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # BGR을 RGB로 변환하고 크기 조정
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (256, 256))
                    frames.append(frame_resized)
                elif frames:
                    # 프레임 읽기 실패 시 이전 프레임 복사
                    frames.append(frames[-1])
                else:
                    # 첫 프레임부터 실패한 경우 제로 프레임
                    frames.append(np.zeros((256, 256, 3), dtype=np.uint8))
            
            # 부족한 프레임은 마지막 프레임으로 채움
            while len(frames) < FRAME_SAMPLE:
                frames.append(frames[-1] if frames else np.zeros((256, 256, 3), dtype=np.uint8))
        
        cap.release()
        return frames[:FRAME_SAMPLE]

    def _extract_yolo_frame(self, frame):
        """
        단일 프레임에서 YOLO 특징 추출
        
        Args:
            frame: RGB 프레임 (numpy array)
            
        Returns:
            feature: 유해 객체별 탐지 횟수 벡터
        """
        # 임시 파일로 저장 (YOLO는 파일 경로를 입력으로 받음)
        temp_path = 'temp_frame.jpg'
        cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        results = self.yolo(temp_path, verbose=False)
        
        # 유해 객체 탐지 결과를 특징 벡터로 변환
        feature = torch.zeros(len(HARMFUL_OBJECTS))
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)].lower()
                    # 유해 객체 목록과 매칭
                    for i, obj in enumerate(HARMFUL_OBJECTS):
                        if obj in class_name or class_name in obj:
                            feature[i] += 1
        
        # 임시 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return feature

    def _extract_slowfast_features(self, frames):
        """
        SlowFast 모델을 사용하여 비디오의 행동 특징 추출
        
        Args:
            frames: 프레임 리스트 (각 shape: (256, 256, 3), 길이: 32)
            
        Returns:
            features: SlowFast 특징 벡터
        """
        try:
            # 프레임을 텐서로 변환 (HWC -> CHW, 정규화)
            frame_tensors = []
            for f in frames:
                tensor = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                frame_tensors.append(tensor)

            # 32 프레임 확보 (부족하면 반복)
            while len(frame_tensors) < 32:
                frame_tensors.extend(frame_tensors[:min(len(frame_tensors), 32-len(frame_tensors))])
            frame_tensors = frame_tensors[:32]

            # SlowFast 정규화 적용
            mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
            frame_tensors = [(f - mean) / std for f in frame_tensors]

            # Fast pathway: 32 프레임 (shape: 1, 3, 32, 256, 256)
            fast_pathway = torch.stack(frame_tensors).unsqueeze(0).to(DEVICE)
            fast_pathway = fast_pathway.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

            # Slow pathway: 8 프레임 (shape: 1, 3, 8, 256, 256)
            slow_indices = torch.linspace(0, 31, 8).long()
            slow_tensors = [frame_tensors[i] for i in slow_indices]
            slow_pathway = torch.stack(slow_tensors).unsqueeze(0).to(DEVICE)
            slow_pathway = slow_pathway.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

            # SlowFast 모델로 특징 추출
            with torch.no_grad():
                features = self.slowfast([slow_pathway, fast_pathway])
            return features.squeeze().to(DEVICE)
            
        except Exception as e:
            print(f"SlowFast error: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시 제로 벡터 반환
            return torch.zeros(self.slowfast_dim).to(DEVICE)

    def _extract_clip_frame(self, frame):
        """
        단일 프레임에서 CLIP 특징 추출
        
        Args:
            frame: RGB 프레임 (numpy array, shape: (256, 256, 3))
            
        Returns:
            clip_features: CLIP 특징 벡터
        """
        # numpy array를 PIL Image로 변환
        image = Image.fromarray(frame)
        
        # CLIP 전처리 적용
        clip_image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)
        
        # CLIP 모델로 이미지 특징 추출
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(clip_image).squeeze()
        
        return clip_features


# ============================================================
# 이미지 분류 모델
# ============================================================
class ImageHarmfulClassifier(nn.Module):
    """
    이미지 유해 콘텐츠 분류를 위한 MLP 모델
    
    YOLO와 CLIP에서 추출한 특징을 입력으로 받아
    다층 퍼셉트론을 통해 유해/안전을 분류
    """
    def __init__(self, yolo_dim, clip_dim):
        super().__init__()
        input_dim = yolo_dim + clip_dim  # YOLO + CLIP 특징 차원
        
        # 다층 퍼셉트론 구조
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),      # 입력층
            nn.ReLU(),                      # 활성화 함수
            nn.BatchNorm1d(256),            # 배치 정규화
            nn.Dropout(0.5),                # 드롭아웃 (과적합 방지)
            
            nn.Linear(256, 128),            # 은닉층 1
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),             # 은닉층 2
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1),               # 출력층
            nn.Sigmoid()                    # 시그모이드 (0~1 확률)
        )
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 특징 벡터 (YOLO + CLIP)
            
        Returns:
            output: 유해 확률 (0~1)
        """
        return self.mlp(x)


# ============================================================
# 비디오 분류 모델
# ============================================================
class VideoHarmfulClassifier(nn.Module):
    """
    비디오 유해 콘텐츠 분류를 위한 Transformer 모델
    
    프레임별 YOLO+CLIP 특징과 SlowFast 특징을 결합하여
    Transformer로 시공간 관계를 학습하고 유해/안전을 분류
    """
    def __init__(self, yolo_dim, clip_dim, slowfast_dim):
        super().__init__()
        input_dim = yolo_dim + clip_dim + slowfast_dim  # 모든 특징 차원 합
        
        # Transformer 인코더 레이어 설정
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,        # 입력 차원
            nhead=2,                  # 어텐션 헤드 수
            dim_feedforward=512,      # 피드포워드 차원
            batch_first=True,         # 배치 차원이 첫 번째
            dropout=0.4               # 드롭아웃 비율
        )
        
        # Transformer 인코더 (2개 레이어)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 최종 분류기
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),      # 입력층
            nn.ReLU(),                      # 활성화 함수
            nn.BatchNorm1d(256),            # 배치 정규화
            nn.Dropout(0.5),                # 드롭아웃
            
            nn.Linear(256, 128),            # 은닉층
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            
            nn.Linear(128, 1),              # 출력층
            nn.Sigmoid()                    # 시그모이드 (0~1 확률)
        )
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 시퀀스 (프레임별 특징, shape: [batch, seq_len, features])
            
        Returns:
            output: 유해 확률 (0~1)
        """
        # Transformer로 시퀀스 특징 변환
        transformed = self.transformer(x)
        
        # 시퀀스 차원에서 평균 풀링 (전체 비디오 특징)
        pooled = transformed.mean(dim=1)
        
        # 분류기로 최종 예측
        return self.classifier(pooled)


# ============================================================
# 데이터 준비 (유해/안전/공개/실제 통합)
# ============================================================
def prepare_image_data():
    """
    이미지 데이터를 준비하고 학습/검증 세트로 분할
    
    여러 소스의 데이터를 통합:
    1. 공개 데이터셋 (HOD, COCO)
    2. 실제 수집 데이터 (검증된 라벨)
    3. 안전 이미지 데이터
    
    Returns:
        X_train, y_train: 학습 데이터 (경로, 라벨)
        X_val, y_val: 검증 데이터 (경로, 라벨)
    """
    print("\n이미지 데이터 준비 중...")
    
    # 1. HOD 데이터셋 (유해 이미지)
    hod_path = DATA_PATH + '1_공개_데이터셋/HOD_Dataset/dataset/'
    hod_images, hod_labels = [], []
    if os.path.exists(hod_path):
        for root, _, files in os.walk(hod_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    hod_images.append(os.path.join(root, file))
                    hod_labels.append(1)  # 유해 라벨
    
    # 2. COCO 안전 데이터셋
    coco_path = DATA_PATH + '1_공개_데이터셋/COCO_Safe_Dataset/'
    coco_images, coco_labels = [], []
    if os.path.exists(coco_path):
        for root, _, files in os.walk(coco_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    coco_images.append(os.path.join(root, file))
                    coco_labels.append(0)  # 안전 라벨
    
    # 3. 실제 수집 데이터 (검증된 라벨)
    verified_path = DATA_PATH + '3_라벨링_파일/verified_labels.json'
    verified_images, verified_labels = [], []
    img_dir = DATA_PATH + '2_실제_수집_데이터/이미지/'
    if os.path.exists(verified_path):
        with open(verified_path, 'r', encoding='utf-8') as f:
            vdata = json.load(f)
        for filename, detections in vdata.items():
            img_path = os.path.join(img_dir, filename)
            if os.path.exists(img_path):
                verified_images.append(img_path)
                # 탐지된 객체가 있으면 유해, 없으면 안전
                verified_labels.append(1 if len(detections) > 0 else 0)
    
    # 4. 안전 이미지 데이터
    safe_path = DATA_PATH + '2_실제_수집_데이터/안전_이미지/'
    safe_json = DATA_PATH + '3_라벨링_파일/safe_labels.json'
    safe_images, safe_labels = [], []
    if os.path.exists(safe_json):
        with open(safe_json, 'r', encoding='utf-8') as f:
            sdata = json.load(f)
        for filename in sdata.keys():
            img_path = os.path.join(safe_path, filename)
            if os.path.exists(img_path):
                safe_images.append(img_path)
                safe_labels.append(0)  # 안전 라벨
    
    # 모든 데이터 통합
    X = hod_images + coco_images + verified_images + safe_images
    y = hod_labels + coco_labels + verified_labels + safe_labels
    
    # 데이터 셔플
    from sklearn.utils import shuffle
    X, y = shuffle(X, y, random_state=42)
    
    # 학습/검증 분할 (15% 검증)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)  # stratify로 클래스 비율 유지
    
    print(f"✓ 데이터셋 통합 완료 (학습 {len(X_train)}, 검증 {len(X_val)})")
    print(f"  (유해: {sum(y_train)}, 안전: {len(y_train)-sum(y_train)} / 검증 유해: {sum(y_val)}, 안전: {len(y_val)-sum(y_val)})")
    return X_train, y_train, X_val, y_val


def prepare_video_data():
    """
    비디오 데이터를 준비하고 학습/검증 세트로 분할
    
    Returns:
        X_train, y_train: 학습 데이터 (경로, 라벨)
        X_val, y_val: 검증 데이터 (경로, 라벨)
    """
    print("\n비디오 데이터 준비 중...")
    
    # 1. 실제 수집 유해 비디오
    verified_path = DATA_PATH + '3_라벨링_파일/verified_video_labels.json'
    video_dir = DATA_PATH + '2_실제_수집_데이터/비디오/'
    vpaths, vlabels = [], []
    if os.path.exists(verified_path):
        with open(verified_path, 'r', encoding='utf-8') as f:
            vdata = json.load(f)
        for fn, info in vdata.items():
            vp = os.path.join(video_dir, fn)
            if os.path.exists(vp):
                vpaths.append(vp)
                vlabels.append(1 if info['is_harmful'] else 0)
    
    # 2. 실제 수집 안전 비디오
    safe_video_dir = DATA_PATH + '2_실제_수집_데이터/안전_비디오/'
    safe_video_json = DATA_PATH + '3_라벨링_파일/safe_video_labels.json'
    svpaths, svlabels = [], []
    if os.path.exists(safe_video_json):
        with open(safe_video_json, 'r', encoding='utf-8') as f:
            sdata = json.load(f)
        for fn in sdata.keys():
            full = os.path.join(safe_video_dir, fn)
            if os.path.exists(full):
                svpaths.append(full)
                svlabels.append(0)
    
    # 3. 공개 비디오 데이터셋 (RWF-2000, RLVS 등) 자동 통합
    public_video_json = DATA_PATH + '3_라벨링_파일/public_video_labels.json'
    pvpaths, pvlabels = [], []
    if os.path.exists(public_video_json):
        with open(public_video_json, 'r', encoding='utf-8') as f:
            pdata = json.load(f)
        for vid, item in pdata.items():
            path = item.get("path")
            if os.path.exists(path):
                label = int(item.get("label", item.get("is_harmful", 0)))
                pvpaths.append(path)
                pvlabels.append(label)
    
    # 모든 데이터 통합
    X = vpaths + svpaths + pvpaths
    y = vlabels + svlabels + pvlabels
    
    # 데이터 셔플
    from sklearn.utils import shuffle
    X, y = shuffle(X, y, random_state=42)
    
    # 학습/검증 분할 (15% 검증)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    
    print(f"✓ 통합 완료 (학습 {len(X_train)}, 검증 {len(X_val)})")
    print(f"  (유해: {sum(y_train)}, 안전: {len(y_train)-sum(y_train)} / 검증 유해: {sum(y_val)}, 안전: {len(y_val)-sum(y_val)})")
    return X_train, y_train, X_val, y_val


# ============================================================
# 이미지 모델 학습
# ============================================================
def train_image_model():
    """
    이미지 유해 콘텐츠 탐지 모델을 학습하는 함수
    
    YOLO와 CLIP 특징을 결합하여 MLP로 유해/안전을 분류
    Early Stopping과 데이터 증강을 통해 성능 향상
    """
    print("\n" + "="*60)
    print("이미지 모델 학습")
    print("="*60)
    
    # 1. 데이터 준비
    train_images, train_labels, val_images, val_labels = prepare_image_data()
    if len(train_images) == 0:
        print("✗ 학습 데이터 없음!")
        return
    
    # 2. 모델 로딩
    print("\n모델 로딩...")
    yolo = YOLO('yolov8n.pt')  # YOLO 객체 탐지 모델
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)  # CLIP 모델
    
    # CLIP 차원 자동 감지
    with torch.no_grad():
        dummy_image = torch.randn(1, 3, 224, 224).to(DEVICE)
        clip_features = clip_model.encode_image(dummy_image)
        clip_dim = clip_features.shape[-1]  # CLIP 특징 차원 (512)
    
    yolo_dim = len(HARMFUL_OBJECTS)  # YOLO 특징 차원
    print(f"✓ YOLO 차원: {yolo_dim}")
    print(f"✓ CLIP 차원: {clip_dim}")
    print(f"✓ 총 입력 차원: {yolo_dim + clip_dim}")
    
    # 3. 데이터셋 및 데이터로더 생성
    train_dataset = HarmfulImageDataset(train_images, train_labels, yolo, clip_model, clip_preprocess, augment=True)
    val_dataset = HarmfulImageDataset(val_images, val_labels, yolo, clip_model, clip_preprocess, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. 모델, 손실함수, 옵티마이저 설정
    model = ImageHarmfulClassifier(yolo_dim=yolo_dim, clip_dim=clip_dim).to(DEVICE)
    criterion = nn.BCELoss()  # 이진 분류를 위한 Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=IMAGE_LR, weight_decay=IMAGE_WEIGHT_DECAY)  # Adam 옵티마이저 + 가중치 감쇠
    
    best_f1 = 0  # 최고 F1 점수 추적
    patience, patience_count = 4, 0  # Early Stopping을 위한 patience 설정
    
    print(f"\n학습 시작 (Epochs: {IMAGE_EPOCHS})")
    
    # 5. 학습 및 검증 루프
    for epoch in range(IMAGE_EPOCHS):
        # 학습 단계
        model.train()  # 학습 모드로 설정
        train_loss = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{IMAGE_EPOCHS}"):
            features, labels = features.to(DEVICE), labels.to(DEVICE)  # GPU로 이동
            
            optimizer.zero_grad()  # 그래디언트 초기화
            outputs = model(features).squeeze()  # 모델 예측
            loss = criterion(outputs, labels)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트
            train_loss += loss.item()
        
        # 검증 단계
        model.eval()  # 평가 모드로 설정
        val_preds = []
        val_true = []
        
        with torch.no_grad():  # 그래디언트 계산 비활성화
            for features, labels in val_loader:
                features = features.to(DEVICE)
                outputs = model(features).squeeze()
                preds = (outputs > 0.5).cpu().numpy()  # 0.5 임계값으로 이진 분류
                val_preds.extend(np.atleast_1d(preds))
                val_true.extend(np.atleast_1d(labels.numpy()))
        
        # 6. 평가 지표 계산
        if len(val_true) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(val_true, val_preds, average='binary', zero_division=0)
            
            print(f"\nEpoch {epoch+1}/{IMAGE_EPOCHS}")
            print(f"  Loss: {train_loss/len(train_loader):.4f}")
            print(f"  Precision: {precision:.4f}")  # 정밀도
            print(f"  Recall: {recall:.4f}")         # 재현율
            print(f"  F1-Score: {f1:.4f}")           # F1 점수
            
            # 최고 성능 모델 저장 및 Early Stopping
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), 'image_model_best.pth')
                patience_count = 0  # 성능 향상 시 patience 리셋
            else:
                patience_count += 1  # 성능 향상 없을 시 patience 증가
            
            # Early Stopping: patience 횟수만큼 성능 향상이 없으면 학습 중단
            if patience_count >= patience:
                print(f"  ⚠️ Early stopping at epoch {epoch+1}, best F1: {best_f1:.4f}")
                break
    
    # 7. 학습 완료 요약
    print(f"\n{'='*60}")
    print(f"✅ 이미지 모델 학습 완료!")
    print(f"   Best F1-Score: {best_f1:.4f}")
    if best_f1 >= 0.75:
        print(f"   ✅ 필수 목표 달성! (F1 ≥ 0.75)")
    else:
        print(f"   ⚠️  목표 미달 (F1 < 0.75)")
    
    # 8. Confusion Matrix 생성
    if len(val_true) > 0:
        cm = confusion_matrix(val_true, val_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  # 이미지는 파란색 테마
                    xticklabels=['Safe', 'Harmful'],
                    yticklabels=['Safe', 'Harmful'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Image Model Confusion Matrix (F1={best_f1:.3f})')
        plt.savefig('image_confusion_matrix.png')
        print(f"   ✓ Confusion Matrix: image_confusion_matrix.png")
    print("="*60)


# ============================================================
# 비디오 모델 학습
# ============================================================
def train_video_model():
    """
    비디오 유해 콘텐츠 탐지 모델을 학습하는 함수
    
    YOLO, SlowFast, CLIP 특징을 결합하여 Transformer로 유해/안전을 분류
    Threshold 분석을 통해 최적의 분류 임계값을 찾음
    """
    print("\n" + "="*60)
    print("비디오 모델 학습")
    print("="*60)
    
    # 1. 비디오 데이터 준비
    train_paths, train_labels, val_paths, val_labels = prepare_video_data()
    if len(train_paths) == 0:
        print("✗ 비디오 데이터 없음!")
        return
    
    # 2. 모델 로딩
    print("\n모델 로딩...")
    yolo = YOLO('yolov8n.pt')  # YOLO 객체 탐지 모델
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)  # CLIP 모델
    
    # SlowFast 모델 로딩 (비디오 행동 인식)
    slowfast_model = slowfast_r50(pretrained=True)  # 사전 훈련된 SlowFast R50 모델
    slowfast_model = slowfast_model.to(DEVICE)
    slowfast_model.eval()  # SlowFast는 고정 (학습하지 않음)
    
    # SlowFast 실제 출력 차원 확인
    print("SlowFast 출력 차원 확인 중...")
    with torch.no_grad():
        dummy_slow = torch.randn(1, 3, 8, 256, 256).to(DEVICE)  # Slow pathway 입력
        dummy_fast = torch.randn(1, 3, 32, 256, 256).to(DEVICE)  # Fast pathway 입력
        slowfast_output = slowfast_model([dummy_slow, dummy_fast])
        slowfast_dim = slowfast_output.shape[-1]  # 실제 출력 차원
        print(f"✓ SlowFast 실제 출력 차원: {slowfast_dim}")
    
    # CLIP 차원 자동 감지
    with torch.no_grad():
        dummy_image = torch.randn(1, 3, 224, 224).to(DEVICE)
        clip_features = clip_model.encode_image(dummy_image)
        clip_dim = clip_features.shape[-1]  # CLIP 특징 차원 (512)
    
    yolo_dim = len(HARMFUL_OBJECTS)  # YOLO 특징 차원
    print(f"✓ YOLO 차원: {yolo_dim}")
    print(f"✓ CLIP 차원: {clip_dim}")
    print(f"✓ 총 입력 차원: {yolo_dim + clip_dim + slowfast_dim}")
    
    # 3. 데이터셋 및 데이터로더 생성
    train_dataset = HarmfulVideoDataset(train_paths, train_labels, yolo, slowfast_model, clip_model, clip_preprocess, slowfast_dim)
    val_dataset = HarmfulVideoDataset(val_paths, val_labels, yolo, slowfast_model, clip_model, clip_preprocess, slowfast_dim)
    train_loader = DataLoader(train_dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. 모델, 손실함수, 옵티마이저 설정
    model = VideoHarmfulClassifier(yolo_dim=yolo_dim, clip_dim=clip_dim, slowfast_dim=slowfast_dim).to(DEVICE)
    criterion = nn.BCELoss()  # 이진 분류를 위한 Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=VIDEO_LR, weight_decay=VIDEO_WEIGHT_DECAY)  # Adam 옵티마이저 + 가중치 감쇠
    
    best_f1 = 0  # 최고 F1 점수 추적
    patience, patience_count = 4, 0  # Early Stopping을 위한 patience 설정
    
    print(f"\n학습 시작 (Epochs: {VIDEO_EPOCHS})")
    
    # 5. 학습 및 검증 루프
    for epoch in range(VIDEO_EPOCHS):
        # 학습 단계
        model.train()  # 학습 모드로 설정
        train_loss = 0
        
        for features, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{VIDEO_EPOCHS}"):
            features, labels_batch = features.to(DEVICE), labels_batch.to(DEVICE)  # GPU로 이동
            
            optimizer.zero_grad()  # 그래디언트 초기화
            outputs = model(features).squeeze()  # 모델 예측
            loss = criterion(outputs, labels_batch)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트
            train_loss += loss.item()
        
        # 검증 단계
        model.eval()  # 평가 모드로 설정
        val_outputs = []
        val_true = []
        
        with torch.no_grad():  # 그래디언트 계산 비활성화
            for features, labels_batch in val_loader:
                features = features.to(DEVICE)
                outputs = model(features).squeeze()
                outputs_np = outputs.cpu().numpy()
                # 단일 값과 배열 모두 처리
                if outputs_np.ndim == 0:
                    val_outputs.append(float(outputs_np))
                else:
                    val_outputs.extend(outputs_np.tolist())
                val_true.extend(labels_batch.cpu().numpy())

        # ============================================================
        # [변경점] final_model(3): Threshold 분석 기능 추가
        # ============================================================
        # 여러 threshold 값에 대해 성능을 분석하여 최적값 찾기
        # 이유: 모든 데이터에 0.5가 최적은 아닐 수 있음
        print(f"\n[Threshold 분석 @ Epoch {epoch+1}]")
        val_outputs_np = np.array(val_outputs)
        val_true_np = np.array(val_true)
        ths = [0.3, 0.4, 0.5, 0.6, 0.7]  # 테스트할 threshold 값들
        
        for th in ths:
            preds = (val_outputs_np > th).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(val_true_np, preds, average='binary', zero_division=0)
            cm = confusion_matrix(val_true_np, preds)
            print(f"  Threshold={th:.2f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            print(f"    Confusion Matrix:\n{cm}")
        
        # 최적 threshold 선택 (여기서는 0.4 사용)
        opt_th = 0.4
        val_preds = (val_outputs_np > opt_th).astype(int)

        # 6. 평가 지표 계산
        precision, recall, f1, _ = precision_recall_fscore_support(val_true_np, val_preds, average='binary', zero_division=0)
        print(f"\nEpoch {epoch+1}/{VIDEO_EPOCHS} (최적 Threshold={opt_th})")
        print(f"  Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Precision: {precision:.4f}")  # 정밀도
        print(f"  Recall: {recall:.4f}")         # 재현율
        print(f"  F1-Score: {f1:.4f}")           # F1 점수
        
        # 최고 성능 모델 저장 및 Early Stopping
        if f1 > best_f1:
            best_f1 = f1
            patience_count = 0  # 성능 향상 시 patience 리셋
            torch.save(model.state_dict(), 'video_model_best.pth')
            print(f"  ✓ 모델 저장 (Best F1: {best_f1:.4f})")
        else:
            patience_count += 1  # 성능 향상 없을 시 patience 증가
        
        # Early Stopping: patience 횟수만큼 성능 향상이 없으면 학습 중단
        if patience_count >= patience:
            print(f"  ⚠️ Early stopping at epoch {epoch+1}, best F1: {best_f1:.4f}")
            break

    # 7. 학습 완료 요약
    print(f"\n{'='*60}")
    print(f"✅ 비디오 모델 학습 완료!")
    print(f"   Best F1-Score: {best_f1:.4f}")
    if best_f1 >= 0.75:
        print(f"   ✅ 필수 목표 달성! (F1 ≥ 0.75)")
    else:
        print(f"   ⚠️  목표 미달 (F1 < 0.75)")
    
    # 8. Confusion Matrix 생성
    if len(val_true_np) > 0:
        cm = confusion_matrix(val_true_np, val_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',  # 비디오는 빨간색 테마
                    xticklabels=['Safe', 'Harmful'],
                    yticklabels=['Safe', 'Harmful'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Video Model Confusion Matrix (F1={best_f1:.3f})')
        plt.savefig('video_confusion_matrix.png')
        print(f"   ✓ Confusion Matrix: video_confusion_matrix.png")
    print("="*60)


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    import sys
    
    # 프로그램 시작 정보 출력
    print("\n" + "="*60)
    print("무하유 유해 콘텐츠 탐지 시스템")
    print("="*60)
    print(f"Device: {DEVICE}")  # 사용할 디바이스 (CPU/GPU)
    print(f"Harmful Objects: {len(HARMFUL_OBJECTS)}개")  # 탐지할 유해 객체 개수
    print(f"Image Epochs: {IMAGE_EPOCHS}")  # 이미지 모델 학습 에포크
    print(f"Video Epochs: {VIDEO_EPOCHS}")  # 비디오 모델 학습 에포크
    print(f"✓ 로그 파일: {log_filename}")
    print("="*60)
    
    # 명령행 인수에 따른 실행 모드 결정
    if len(sys.argv) > 1:
        mode = sys.argv[1]  # 첫 번째 인수로 실행 모드 지정
        
        if mode == 'train_image':
            # 이미지 모델만 학습
            train_image_model()
        elif mode == 'train_video':
            # 비디오 모델만 학습
            train_video_model()
        else:
            # 잘못된 인수인 경우 사용법 출력
            print("\n사용법:")
            print("  python final_model.py               (전체 학습)")
            print("  python final_model.py train_image   (이미지만)")
            print("  python final_model.py train_video   (비디오만)")
    else:
        # 인수 없이 실행하면 전체 학습 수행
        train_image_model()  # 이미지 모델 학습
        train_video_model()  # 비디오 모델 학습
        
        # 학습 완료 후 결과 요약
        print("\n" + "="*60)
        print("✅ 전체 학습 완료!")
        print("="*60)
        print("생성된 파일:")
        print("  - image_model_best.pth")           # 최고 성능 이미지 모델
        print("  - video_model_best.pth")            # 최고 성능 비디오 모델
        print("  - image_confusion_matrix.png")     # 이미지 모델 혼동 행렬
        print("  - video_confusion_matrix.png")     # 비디오 모델 혼동 행렬
        print(f"  - {log_filename}")                 # 로그 파일
        print("="*60)
