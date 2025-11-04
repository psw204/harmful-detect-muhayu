"""
무하유 유해 콘텐츠 탐지 시스템 - 개선 버전 2
이미지: YOLOv8 + CLIP + MLP
비디오: YOLOv8 + SlowFast + CLIP + Transformer

이 시스템은 이미지와 비디오에서 유해한 콘텐츠를 자동으로 탐지하는 AI 모델입니다.
- 이미지: YOLO로 객체 탐지 + CLIP으로 맥락 이해 + MLP로 분류
- 비디오: YOLO + SlowFast로 행동 인식 + CLIP + Transformer로 시계열 분석
"""

# ============================================================
# 필수 라이브러리 Import
# ============================================================
# PyTorch 딥러닝 프레임워크
import torch                    # PyTorch 핵심 라이브러리
import torch.nn as nn          # 신경망 모듈들 (Linear, ReLU, Dropout 등)
import torch.optim as optim    # 최적화 알고리즘 (Adam 등)
from torch.utils.data import Dataset, DataLoader  # 데이터셋과 데이터 로더

# ============================================================
# [변경점] final_model(2): 데이터 증강을 위한 라이브러리 추가
# ============================================================
import torchvision.transforms as T  # 이미지 변환을 위한 라이브러리 (데이터 증강용)

# 컴퓨터 비전 및 AI 모델
from ultralytics import YOLO   # YOLO 객체 탐지 모델
import clip                    # OpenAI의 CLIP 모델 (이미지-텍스트 이해)
from pytorchvideo.models.hub import slowfast_r50  # SlowFast 비디오 이해 모델

# 이미지/비디오 처리
from PIL import Image          # 이미지 처리 라이브러리
import cv2                     # OpenCV (비디오 처리)

# 데이터 처리 및 분석
import json                    # JSON 파일 처리
import os                      # 파일 시스템 조작
import numpy as np             # 수치 계산 라이브러리
from tqdm import tqdm          # 진행률 표시 바
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix  # 성능 평가 지표

# 시각화
import matplotlib.pyplot as plt  # 그래프 그리기
import seaborn as sns          # 통계 시각화

# ============================================================
# [변경점] final_model(2): 로깅 시스템 추가
# ============================================================
import sys                      # 시스템 관련 기능
import datetime                 # 날짜/시간 처리

# 현재 시간을 기반으로 로그 파일명 생성
now = datetime.datetime.now()
log_filename = now.strftime("train_log_%Y%m%d_%H%M%S.txt")

class Tee(object):
    """
    출력을 화면과 파일에 동시에 기록하는 클래스
    
    이 클래스는 학습 과정의 모든 출력을 화면에 표시하면서
    동시에 로그 파일에도 저장합니다.
    
    사용 이유:
    - 학습 과정을 나중에 다시 확인할 수 있음
    - 문제 발생 시 디버깅에 유용
    - 여러 실험 결과를 비교할 때 편리
    """
    def __init__(self, filepath):
        """로그 파일을 열어서 초기화"""
        self.file = open(filepath, "w", encoding="utf-8")
    
    def write(self, data):
        """데이터를 화면과 파일에 동시에 출력"""
        sys.__stdout__.write(data)  # 화면에 출력
        self.file.write(data)       # 파일에 저장
    
    def flush(self):
        """출력 버퍼를 비우기"""
        sys.__stdout__.flush()
        self.file.flush()

# Tee 객체를 생성하여 출력을 리다이렉트
tee = Tee(log_filename)
sys.stdout = tee  # 표준 출력을 Tee로 리다이렉트
sys.stderr = tee  # 표준 에러도 Tee로 리다이렉트


# ============================================================
# 하이퍼파라미터 설정
# ============================================================
# 데이터가 저장된 경로를 설정합니다
DATA_PATH = './무하유_유해콘텐츠_데이터/'

# GPU가 있으면 GPU를 사용하고, 없으면 CPU를 사용합니다
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 배치 크기 설정 (한 번에 처리할 데이터 개수)
BATCH_SIZE = 8  # 이미지용 배치 크기
VIDEO_BATCH_SIZE = 2  # 비디오는 메모리를 많이 사용하므로 더 작게 설정

# ============================================================
# [변경점] final_model(2): 학습 에포크 수 증가 (5 → 10)
# ============================================================
# 이유: 더 많은 반복 학습을 통해 모델 성능 향상
IMAGE_EPOCHS = 10  # 이미지 모델 학습 에포크 (5에서 10으로 증가)
VIDEO_EPOCHS = 10  # 비디오 모델 학습 에포크 (5에서 10으로 증가)

# 학습률 설정
IMAGE_LR = 0.001  # 이미지 모델 학습률
VIDEO_LR = 0.0005  # 비디오 모델 학습률

# ============================================================
# [변경점] final_model(2): Weight Decay 정규화 추가
# ============================================================
# Weight Decay: 가중치가 너무 커지는 것을 방지하여 과적합 방지
IMAGE_WEIGHT_DECAY = 0.003  # 이미지 모델 정규화 계수
VIDEO_WEIGHT_DECAY = 0.003  # 비디오 모델 정규화 계수

# ============================================================
# [변경점] final_model(2): 비디오 프레임 수 증가 (16 → 32)
# ============================================================
# 이유: 더 많은 프레임을 사용하여 시간적 정보를 더 풍부하게 파악
FRAME_SAMPLE = 32  # 16에서 32로 증가

# 유해하다고 판단할 객체들의 목록
HARMFUL_OBJECTS = [
    'knife', 'gun', 'pistol', 'rifle', 'sword', 'axe',
    'hammer', 'dagger', 'machete',
    'beer', 'cigarette'
]


# ============================================================
# 이미지 데이터셋 클래스
# ============================================================
class HarmfulImageDataset(Dataset):
    """
    이미지 데이터셋 클래스: YOLO + CLIP 특징 추출 + 데이터 증강
    
    이 클래스는 이미지 파일들을 읽어서 YOLO로 객체를 탐지하고,
    CLIP으로 맥락을 이해한 후, 두 특징을 결합하여 모델에 입력할 수 있게 만듭니다.
    
    주요 역할:
    - 이미지 파일 로드 및 전처리
    - YOLO를 통한 유해 객체 탐지
    - CLIP을 통한 이미지 맥락 특징 추출
    - 데이터 증강을 통한 학습 데이터 다양화
    """
    
    # ============================================================
    # [변경점] final_model(2): 데이터 증강 파라미터 추가
    # ============================================================
    def __init__(self, image_paths, labels, yolo_model, clip_model, clip_preprocess, augment=False):
        """
        데이터셋 초기화
        
        Args:
            image_paths: 이미지 파일 경로들의 리스트
            labels: 각 이미지의 라벨 (0=안전, 1=유해)
            yolo_model: YOLO 객체 탐지 모델
            clip_model: CLIP 이미지 이해 모델
            clip_preprocess: CLIP용 이미지 전처리 함수
            augment: 데이터 증강 사용 여부 (신규 추가)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.yolo = yolo_model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        
        # ============================================================
        # [변경점] final_model(2): 데이터 증강 변환 파이프라인 추가
        # ============================================================
        self.augment = augment  # 데이터 증강 사용 여부
        
        # 데이터 증강을 위한 변환 파이프라인 정의
        # 이미지를 다양하게 변형하여 모델이 더 일반화되도록 함
        self.aug_transform = T.Compose([
            T.RandomHorizontalFlip(),  # 수평 뒤집기 (50% 확률)
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # 색상 조정
            T.RandomRotation(degrees=15),  # 랜덤 회전 (±15도)
            T.RandomResizedCrop(224, scale=(0.75, 1.0)),  # 랜덤 크롭 및 리사이즈
            T.RandomPerspective(distortion_scale=0.5, p=0.5),  # 원근 변환 (50% 확률)
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        특정 인덱스의 데이터를 가져오는 함수
        
        Args:
            idx: 가져올 데이터의 인덱스
            
        Returns:
            combined: YOLO와 CLIP 특징이 결합된 벡터
            label: 해당 이미지의 라벨
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 1. 이미지 로드
            image = Image.open(img_path).convert('RGB')
            
            # ============================================================
            # [변경점] final_model(2): 학습 시 데이터 증강 적용
            # ============================================================
            # 학습 시에만 데이터 증강을 적용하여 다양한 변형 생성
            if self.augment:
                image = self.aug_transform(image)
            
            # 2. YOLO로 객체 탐지
            yolo_results = self.yolo(img_path, verbose=False)
            yolo_features = self._extract_yolo_features(yolo_results)
            
            # 3. CLIP으로 맥락 분석
            clip_image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                clip_features = self.clip_model.encode_image(clip_image).squeeze()
            
            # 4. YOLO와 CLIP 특징 결합
            combined = torch.cat([yolo_features.to(DEVICE), clip_features])
            return combined, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(len(HARMFUL_OBJECTS) + 512).to(DEVICE), torch.tensor(label, dtype=torch.float32)
    
    def _extract_yolo_features(self, results):
        """
        YOLO 탐지 결과를 특징 벡터로 변환
        
        Args:
            results: YOLO 모델의 탐지 결과
            
        Returns:
            feature_vector: 유해 객체들의 탐지 횟수를 나타내는 벡터
        """
        feature_vector = torch.zeros(len(HARMFUL_OBJECTS))
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)].lower()
                    for i, obj in enumerate(HARMFUL_OBJECTS):
                        if obj in class_name or class_name in obj:
                            feature_vector[i] += 1
        
        return feature_vector


# ============================================================
# 비디오 데이터셋 클래스
# ============================================================
class HarmfulVideoDataset(Dataset):
    """
    비디오 데이터셋 클래스: YOLO + SlowFast + CLIP 특징 추출
    
    이 클래스는 비디오 파일들을 읽어서 여러 AI 모델을 사용하여 특징을 추출합니다.
    """
    
    # ============================================================
    # [변경점] final_model(2): slowfast_dim 파라미터 추가
    # ============================================================
    def __init__(self, video_paths, labels, yolo_model, slowfast_model, clip_model, clip_preprocess, slowfast_dim):
        """
        비디오 데이터셋 초기화
        
        Args:
            video_paths: 비디오 파일 경로들의 리스트
            labels: 각 비디오의 라벨
            yolo_model: YOLO 객체 탐지 모델
            slowfast_model: SlowFast 행동 인식 모델
            clip_model: CLIP 이미지 이해 모델
            clip_preprocess: CLIP용 이미지 전처리 함수
            slowfast_dim: SlowFast 특징의 차원 (신규 추가)
        """
        self.video_paths = video_paths
        self.labels = labels
        self.yolo = yolo_model
        self.slowfast = slowfast_model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.slowfast_dim = slowfast_dim  # SlowFast 차원 저장
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        특정 인덱스의 비디오 데이터를 가져오는 함수
        
        Args:
            idx: 가져올 데이터의 인덱스
            
        Returns:
            combined: 모든 특징이 결합된 시계열 데이터
            label: 해당 비디오의 라벨
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # 1. 비디오에서 프레임 샘플링 (32개)
            frames = self._sample_frames(video_path)
            
            # ============================================================
            # [변경점] final_model(2): 프레임 수 검증 로직 추가
            # ============================================================
            # 프레임 수가 정확한지 확인하여 문제 조기 발견
            if len(frames) != FRAME_SAMPLE:
                print(f"Warning: {video_path} frames {len(frames)} != {FRAME_SAMPLE}")
            
            # 2. YOLO로 각 프레임에서 객체 탐지
            yolo_list = [self._extract_yolo_frame(frame) for frame in frames]
            yolo_seq = torch.stack(yolo_list).to(DEVICE)
            
            # 3. SlowFast로 비디오 전체의 행동 패턴 분석
            slowfast_feat = self._extract_slowfast_features(frames)
            
            # 4. CLIP으로 각 프레임의 맥락 분석
            clip_list = [self._extract_clip_frame(frame) for frame in frames]
            clip_seq = torch.stack(clip_list).to(DEVICE)
            
            # 5. 모든 특징들을 결합
            yolo_clip_seq = torch.cat([yolo_seq, clip_seq], dim=1)
            slowfast_expanded = slowfast_feat.unsqueeze(0).repeat(FRAME_SAMPLE, 1)
            combined = torch.cat([yolo_clip_seq, slowfast_expanded], dim=1)
            
            return combined, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            import traceback
            traceback.print_exc()  # 상세한 오류 정보 출력
            default_dim = len(HARMFUL_OBJECTS) + 512 + self.slowfast_dim
            return torch.zeros(FRAME_SAMPLE, default_dim).to(DEVICE), torch.tensor(label, dtype=torch.float32)
    
    def _sample_frames(self, video_path):
        """
        비디오에서 균등 간격으로 프레임을 샘플링
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            frames: 샘플링된 프레임들의 리스트 (32개)
        """
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        if total == 0:
            cap.release()
            # ============================================================
            # [변경점] final_model(2): 프레임 크기 증가 (224 → 256)
            # ============================================================
            # SlowFast가 더 높은 해상도를 사용하도록 변경
            frames = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(FRAME_SAMPLE)]
        else:
            indices = np.linspace(0, max(0, total-1), FRAME_SAMPLE, dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (256, 256))
                    frames.append(frame_resized)
                elif frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((256, 256, 3), dtype=np.uint8))
            
            while len(frames) < FRAME_SAMPLE:
                frames.append(frames[-1] if frames else np.zeros((256, 256, 3), dtype=np.uint8))
        
        cap.release()
        return frames[:FRAME_SAMPLE]
    
    def _extract_yolo_frame(self, frame):
        """
        단일 프레임에서 YOLO 특징 추출
        
        Args:
            frame: 분석할 프레임
            
        Returns:
            feature: 유해 객체들의 탐지 횟수를 나타내는 벡터
        """
        temp_path = 'temp_frame.jpg'
        cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        results = self.yolo(temp_path, verbose=False)
        
        feature = torch.zeros(len(HARMFUL_OBJECTS))
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)].lower()
                    for i, obj in enumerate(HARMFUL_OBJECTS):
                        if obj in class_name or class_name in obj:
                            feature[i] += 1
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return feature
    
    def _extract_slowfast_features(self, frames):
        """
        SlowFast를 사용한 행동 인식 특징 추출 (개선된 버전)
        
        Args:
            frames: 분석할 프레임들의 리스트 (각 shape: (256, 256, 3), 길이: 32)
            
        Returns:
            features: SlowFast로 추출된 행동 특징 벡터
        """
        try:
            # 1. 프레임들을 텐서로 변환하고 정규화
            frame_tensors = []
            for f in frames:
                tensor = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                frame_tensors.append(tensor)
            
            # ============================================================
            # [변경점] final_model(2): 32 프레임 확보 로직 개선
            # ============================================================
            # 프레임이 부족한 경우 반복하여 32개 확보
            while len(frame_tensors) < 32:
                frame_tensors.extend(frame_tensors[:min(len(frame_tensors), 32-len(frame_tensors))])
            frame_tensors = frame_tensors[:32]
            
            # 2. SlowFast 표준 정규화 적용
            mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
            frame_tensors = [(f - mean) / std for f in frame_tensors]
            
            # 3. Fast pathway: 32 프레임 (shape: 1, 3, 32, 256, 256)
            fast_pathway = torch.stack(frame_tensors).unsqueeze(0).to(DEVICE)
            fast_pathway = fast_pathway.permute(0, 2, 1, 3, 4)
            
            # 4. Slow pathway: 8 프레임 (shape: 1, 3, 8, 256, 256)
            slow_indices = torch.linspace(0, 31, 8).long()
            slow_tensors = [frame_tensors[i] for i in slow_indices]
            slow_pathway = torch.stack(slow_tensors).unsqueeze(0).to(DEVICE)
            slow_pathway = slow_pathway.permute(0, 2, 1, 3, 4)
            
            # 5. SlowFast 모델로 특징 추출
            with torch.no_grad():
                features = self.slowfast([slow_pathway, fast_pathway])
            
            return features.squeeze().to(DEVICE)
            
        except Exception as e:
            print(f"SlowFast error: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(self.slowfast_dim).to(DEVICE)
    
    def _extract_clip_frame(self, frame):
        """
        단일 프레임에서 CLIP 특징 추출
        
        Args:
            frame: 분석할 프레임 (shape: (256, 256, 3))
            
        Returns:
            clip_features: CLIP 특징 벡터 (512차원)
        """
        image = Image.fromarray(frame)
        clip_image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(clip_image).squeeze()
        
        return clip_features


# ============================================================
# 이미지 분류 모델
# ============================================================
class ImageHarmfulClassifier(nn.Module):
    """
    이미지 유해 콘텐츠 분류를 위한 개선된 MLP 모델
    
    YOLO와 CLIP에서 추출된 특징을 입력받아 유해/안전을 분류합니다.
    """
    def __init__(self, yolo_dim, clip_dim):
        """
        이미지 분류기 초기화
        
        Args:
            yolo_dim: YOLO 특징의 차원
            clip_dim: CLIP 특징의 차원
        """
        super().__init__()
        input_dim = yolo_dim + clip_dim
        
        # ============================================================
        # [변경점] final_model(2): BatchNorm 추가로 학습 안정화
        # ============================================================
        # BatchNorm: 각 층의 출력을 정규화하여 학습 속도 향상 및 안정화
        self.mlp = nn.Sequential(
            # 첫 번째 층
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),         # 배치 정규화 추가
            nn.Dropout(0.5),              # 드롭아웃 강화 (0.4 → 0.5)
            
            # 두 번째 층
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),         # 배치 정규화 추가
            nn.Dropout(0.5),
            
            # 세 번째 층
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),          # 배치 정규화 추가
            nn.Dropout(0.3),
            
            # 출력 층
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        순전파 함수
        
        Args:
            x: 입력 특징 벡터
            
        Returns:
            유해 확률 (0~1)
        """
        return self.mlp(x)


# ============================================================
# 비디오 분류 모델
# ============================================================
class VideoHarmfulClassifier(nn.Module):
    """
    비디오 유해 콘텐츠 분류를 위한 개선된 Transformer 기반 시계열 분류기
    
    YOLO, CLIP, SlowFast 특징을 시계열로 처리하여 유해/안전을 분류합니다.
    """
    def __init__(self, yolo_dim, clip_dim, slowfast_dim):
        """
        비디오 분류기 초기화
        
        Args:
            yolo_dim: YOLO 특징의 차원
            clip_dim: CLIP 특징의 차원
            slowfast_dim: SlowFast 특징의 차원
        """
        super().__init__()
        input_dim = yolo_dim + clip_dim + slowfast_dim
        
        # Transformer 인코더 레이어 정의
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=2,
            dim_feedforward=512,
            batch_first=True,
            dropout=0.4                # 드롭아웃 증가 (0.3 → 0.4)
        )
        
        # Transformer 인코더
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # ============================================================
        # [변경점] final_model(2): 분류기에 BatchNorm 추가
        # ============================================================
        # 최종 분류기에도 BatchNorm을 추가하여 안정성 향상
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),     # 배치 정규화 추가
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),     # 배치 정규화 추가
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        순전파 함수
        
        Args:
            x: 입력 시계열 특징
            
        Returns:
            유해 확률 (0~1)
        """
        # Transformer로 시계열 특징 변환
        transformed = self.transformer(x)
        
        # 시간 차원에 대해 평균 풀링
        pooled = transformed.mean(dim=1)
        
        # 최종 분류
        return self.classifier(pooled)


# ============================================================
# 데이터 준비 함수들
# ============================================================
def prepare_image_data():
    """
    이미지 데이터를 통합하여 로드하고 학습/검증으로 분할
    
    Returns:
        X_train, y_train: 학습용 이미지 경로와 라벨
        X_val, y_val: 검증용 이미지 경로와 라벨
    """
    print("\n이미지 데이터 준비 중...")
    
    # 1. HOD Dataset (유해)
    hod_path = DATA_PATH + '1_공개_데이터셋/HOD_Dataset/dataset/'
    hod_images, hod_labels = [], []
    if os.path.exists(hod_path):
        for root, _, files in os.walk(hod_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    hod_images.append(os.path.join(root, file))
                    hod_labels.append(1)
    
    # 2. COCO Safe Dataset (안전)
    coco_path = DATA_PATH + '1_공개_데이터셋/COCO_Safe_Dataset/'
    coco_images, coco_labels = [], []
    if os.path.exists(coco_path):
        for root, _, files in os.walk(coco_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    coco_images.append(os.path.join(root, file))
                    coco_labels.append(0)
    
    # 3. 실제 수집 유해 이미지
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
                verified_labels.append(1 if len(detections) > 0 else 0)
    
    # ============================================================
    # [변경점] final_model(2): 안전 이미지 경로 수정
    # ============================================================
    # 4. 실제 수집 안전 이미지 (경로명 변경: 이미지_안전 → 안전_이미지)
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
                safe_labels.append(0)
    
    # 모든 데이터 통합
    X = hod_images + coco_images + verified_images + safe_images
    y = hod_labels + coco_labels + verified_labels + safe_labels
    
    # 데이터 셔플
    from sklearn.utils import shuffle
    X, y = shuffle(X, y, random_state=42)
    
    # 학습/검증 분할
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    
    print(f"✓ 데이터셋 통합 완료 (학습 {len(X_train)}, 검증 {len(X_val)})")
    print(f"  (유해: {sum(y_train)}, 안전: {len(y_train)-sum(y_train)} / 검증 유해: {sum(y_val)}, 안전: {len(y_val)-sum(y_val)})")
    return X_train, y_train, X_val, y_val


def prepare_video_data():
    """
    비디오 데이터를 통합하여 로드하고 학습/검증으로 분할
    
    Returns:
        X_train, y_train: 학습용 비디오 경로와 라벨
        X_val, y_val: 검증용 비디오 경로와 라벨
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
    
    # ============================================================
    # [변경점] final_model(2): 안전 비디오 경로 수정 및 공개 데이터셋 추가
    # ============================================================
    # 2. 실제 수집 안전 비디오 (경로명 변경: 비디오_안전 → 안전_비디오)
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
    
    # 3. 공개 비디오 데이터셋 추가 (RWF-2000, RLVS 등)
    # 더 많은 학습 데이터로 모델 성능 향상
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
    
    # 모든 비디오 데이터 통합
    X = vpaths + svpaths + pvpaths
    y = vlabels + svlabels + pvlabels
    
    # 데이터 셔플
    from sklearn.utils import shuffle
    X, y = shuffle(X, y, random_state=42)
    
    # 학습/검증 분할
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
    이미지 모델 학습 메인 함수
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
    yolo = YOLO('yolov8n.pt')
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    
    # CLIP 차원 자동 감지
    with torch.no_grad():
        dummy_image = torch.randn(1, 3, 224, 224).to(DEVICE)
        clip_features = clip_model.encode_image(dummy_image)
        clip_dim = clip_features.shape[-1]
    
    yolo_dim = len(HARMFUL_OBJECTS)
    print(f"✓ YOLO 차원: {yolo_dim}")
    print(f"✓ CLIP 차원: {clip_dim}")
    print(f"✓ 총 입력 차원: {yolo_dim + clip_dim}")
    
    # ============================================================
    # [변경점] final_model(2): 학습 시 데이터 증강 활성화
    # ============================================================
    # 3. 데이터셋 및 데이터로더 생성
    train_dataset = HarmfulImageDataset(train_images, train_labels, yolo, clip_model, clip_preprocess, augment=True)
    val_dataset = HarmfulImageDataset(val_images, val_labels, yolo, clip_model, clip_preprocess, augment=False)
    
    # ============================================================
    # [변경점] final_model(2): drop_last=True 추가
    # ============================================================
    # drop_last=True: 마지막 불완전한 배치를 버림 (BatchNorm 안정성을 위해)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. 모델, 손실함수, 옵티마이저 설정
    model = ImageHarmfulClassifier(yolo_dim=yolo_dim, clip_dim=clip_dim).to(DEVICE)
    criterion = nn.BCELoss()
    
    # ============================================================
    # [변경점] final_model(2): Weight Decay 추가
    # ============================================================
    # Weight Decay를 추가하여 과적합 방지
    optimizer = optim.Adam(model.parameters(), lr=IMAGE_LR, weight_decay=IMAGE_WEIGHT_DECAY)
    
    best_f1 = 0
    # ============================================================
    # [변경점] final_model(2): Early Stopping 메커니즘 추가
    # ============================================================
    # Early Stopping: 성능 향상이 없으면 학습 조기 종료
    patience, patience_count = 4, 0  # 4 에포크 동안 개선 없으면 종료
    
    print(f"\n학습 시작 (Epochs: {IMAGE_EPOCHS})")
    
    # 5. 학습 및 검증 루프
    for epoch in range(IMAGE_EPOCHS):
        # 학습 단계
        model.train()
        train_loss = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{IMAGE_EPOCHS}"):
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 검증 단계
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(DEVICE)
                outputs = model(features).squeeze()
                preds = (outputs > 0.5).cpu().numpy()
                val_preds.extend(np.atleast_1d(preds))
                val_true.extend(np.atleast_1d(labels.numpy()))
        
        # 6. 평가 지표 계산
        if len(val_true) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_true, val_preds, average='binary', zero_division=0)
            
            print(f"\nEpoch {epoch+1}/{IMAGE_EPOCHS}")
            print(f"  Loss: {train_loss/len(train_loader):.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            # 최고 성능 모델 저장 및 Early Stopping
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), 'image_model_best.pth')
                patience_count = 0
                print(f"  ✓ 모델 저장 (Best F1: {best_f1:.4f})")
            else:
                patience_count += 1
            
            # Early Stopping 체크
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
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
    비디오 모델 학습 메인 함수
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
    yolo = YOLO('yolov8n.pt')
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    
    # SlowFast 모델 로딩
    slowfast_model = slowfast_r50(pretrained=True)
    slowfast_model = slowfast_model.to(DEVICE)
    slowfast_model.eval()
    
    # ============================================================
    # [변경점] final_model(2): SlowFast 실제 출력 차원 동적 확인
    # ============================================================
    # SlowFast의 실제 출력 차원을 확인하여 정확한 모델 구성
    print("SlowFast 출력 차원 확인 중...")
    with torch.no_grad():
        dummy_slow = torch.randn(1, 3, 8, 256, 256).to(DEVICE)
        dummy_fast = torch.randn(1, 3, 32, 256, 256).to(DEVICE)
        slowfast_output = slowfast_model([dummy_slow, dummy_fast])
        slowfast_dim = slowfast_output.shape[-1]
        print(f"✓ SlowFast 실제 출력 차원: {slowfast_dim}")
    
    # CLIP 차원 자동 감지
    with torch.no_grad():
        dummy_image = torch.randn(1, 3, 224, 224).to(DEVICE)
        clip_features = clip_model.encode_image(dummy_image)
        clip_dim = clip_features.shape[-1]
    
    yolo_dim = len(HARMFUL_OBJECTS)
    print(f"✓ YOLO 차원: {yolo_dim}")
    print(f"✓ CLIP 차원: {clip_dim}")
    print(f"✓ 총 입력 차원: {yolo_dim + clip_dim + slowfast_dim}")
    
    # 3. 데이터셋 및 데이터로더 생성
    train_dataset = HarmfulVideoDataset(train_paths, train_labels, yolo, slowfast_model, clip_model, clip_preprocess, slowfast_dim)
    val_dataset = HarmfulVideoDataset(val_paths, val_labels, yolo, slowfast_model, clip_model, clip_preprocess, slowfast_dim)
    
    # ============================================================
    # [변경점] final_model(2): drop_last=True 추가
    # ============================================================
    train_loader = DataLoader(train_dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. 모델, 손실함수, 옵티마이저 설정
    model = VideoHarmfulClassifier(yolo_dim=yolo_dim, clip_dim=clip_dim, slowfast_dim=slowfast_dim).to(DEVICE)
    criterion = nn.BCELoss()
    
    # ============================================================
    # [변경점] final_model(2): Weight Decay 추가
    # ============================================================
    optimizer = optim.Adam(model.parameters(), lr=VIDEO_LR, weight_decay=VIDEO_WEIGHT_DECAY)
    
    best_f1 = 0
    # ============================================================
    # [변경점] final_model(2): Early Stopping 메커니즘 추가
    # ============================================================
    patience, patience_count = 4, 0
    
    print(f"\n학습 시작 (Epochs: {VIDEO_EPOCHS})")
    
    # 5. 학습 및 검증 루프
    for epoch in range(VIDEO_EPOCHS):
        # 학습 단계
        model.train()
        train_loss = 0
        
        for features, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{VIDEO_EPOCHS}"):
            features, labels_batch = features.to(DEVICE), labels_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 검증 단계
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for features, labels_batch in val_loader:
                features = features.to(DEVICE)
                outputs = model(features).squeeze()
                preds = (outputs > 0.5).cpu().numpy()
                val_preds.extend(np.atleast_1d(preds))
                val_true.extend(np.atleast_1d(labels_batch.numpy()))
        
        # 6. 평가 지표 계산
        if len(val_true) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_true, val_preds, average='binary', zero_division=0)
            
            print(f"\nEpoch {epoch+1}/{VIDEO_EPOCHS}")
            print(f"  Loss: {train_loss/len(train_loader):.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            # 최고 성능 모델 저장 및 Early Stopping
            if f1 > best_f1:
                best_f1 = f1
                patience_count = 0
                torch.save(model.state_dict(), 'video_model_best.pth')
                print(f"  ✓ 모델 저장 (Best F1: {best_f1:.4f})")
            else:
                patience_count += 1
            
            # Early Stopping 체크
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
    if len(val_true) > 0:
        cm = confusion_matrix(val_true, val_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
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
    
    print("\n" + "="*60)
    print("무하유 유해 콘텐츠 탐지 시스템")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Harmful Objects: {len(HARMFUL_OBJECTS)}개")
    print(f"Image Epochs: {IMAGE_EPOCHS}")
    print(f"Video Epochs: {VIDEO_EPOCHS}")
    print(f"✓ 로그 파일: {log_filename}")  # 로그 파일명 표시
    print("="*60)
    
    # 명령행 인수에 따른 실행 모드 결정
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == 'train_image':
            train_image_model()
        elif mode == 'train_video':
            train_video_model()
        else:
            print("\n사용법:")
            print("  python final_model.py               (전체 학습)")
            print("  python final_model.py train_image   (이미지만)")
            print("  python final_model.py train_video   (비디오만)")
    else:
        train_image_model()
        train_video_model()
        
        print("\n" + "="*60)
        print("✅ 전체 학습 완료!")
        print("="*60)
        print("생성된 파일:")
        print("  - image_model_best.pth")
        print("  - video_model_best.pth")
        print("  - image_confusion_matrix.png")
        print("  - video_confusion_matrix.png")
        print(f"  - {log_filename}")  # 로그 파일도 표시
        print("="*60)
