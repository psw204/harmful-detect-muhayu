"""
무하유 유해 콘텐츠 탐지 시스템 - 기본 버전
이미지: YOLOv8 + CLIP + MLP
비디오: YOLOv8 + SlowFast + CLIP + Transformer

이 시스템은 이미지와 비디오에서 유해한 콘텐츠를 자동으로 탐지하는 AI 모델입니다.
- 이미지: YOLO로 객체 탐지 + CLIP으로 맥락 이해 + MLP로 분류
- 비디오: YOLO + SlowFast로 행동 인식 + CLIP + Transformer로 시계열 분석

작성자: 박상원
작성일: 2025년 2학기
"""

# ============================================================
# 필수 라이브러리 Import
# ============================================================
# PyTorch 딥러닝 프레임워크
import torch                    # PyTorch 핵심 라이브러리
import torch.nn as nn          # 신경망 모듈들 (Linear, ReLU, Dropout 등)
import torch.optim as optim    # 최적화 알고리즘 (Adam 등)
from torch.utils.data import Dataset, DataLoader  # 데이터셋과 데이터 로더

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
# 하이퍼파라미터 설정
# ============================================================
# 데이터가 저장된 경로를 설정합니다
DATA_PATH = './무하유_유해콘텐츠_데이터/'

# GPU가 있으면 GPU를 사용하고, 없으면 CPU를 사용합니다
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 배치 크기 설정 (한 번에 처리할 데이터 개수)
BATCH_SIZE = 8  # 이미지용 배치 크기 (GPU 메모리 절약을 위해 작게 설정)
VIDEO_BATCH_SIZE = 2  # 비디오는 메모리를 많이 사용하므로 더 작게 설정

# 학습 에포크 수 (전체 데이터셋을 몇 번 반복 학습할지)
IMAGE_EPOCHS = 5  # 이미지 모델 학습 에포크 (빠른 테스트용)
VIDEO_EPOCHS = 5  # 비디오 모델 학습 에포크 (목표 달성을 위해 설정)

# 학습률 설정 (모델이 얼마나 빠르게 학습할지 결정)
IMAGE_LR = 0.001  # 이미지 모델 학습률 (0.001 = 천분의 일)
VIDEO_LR = 0.0005  # 비디오 모델 학습률 (더 안정적인 학습을 위해 더 작게)

# 비디오에서 추출할 프레임 수 (16프레임으로 시간적 정보 파악)
FRAME_SAMPLE = 16

# 유해하다고 판단할 객체들의 목록
# YOLO가 탐지할 수 있는 객체들 중에서 위험한 것들을 선별
HARMFUL_OBJECTS = [
    'knife', 'gun', 'pistol', 'rifle', 'sword', 'axe',  # 무기류
    'hammer', 'dagger', 'machete',                       # 도구류
    'beer', 'cigarette'                                  # 음주/흡연 관련
]


# ============================================================
# 이미지 데이터셋 클래스
# ============================================================
class HarmfulImageDataset(Dataset):
    """
    이미지 데이터셋 클래스: YOLO + CLIP 특징 추출
    
    이 클래스는 이미지 파일들을 읽어서 YOLO로 객체를 탐지하고,
    CLIP으로 맥락을 이해한 후, 두 특징을 결합하여 모델에 입력할 수 있게 만듭니다.
    
    주요 역할:
    - 이미지 파일 로드 및 전처리
    - YOLO를 통한 유해 객체 탐지 (객체별 개수 카운트)
    - CLIP을 통한 이미지 맥락 특징 추출
    - 두 특징을 결합하여 분류 모델에 입력
    """
    
    def __init__(self, image_paths, labels, yolo_model, clip_model, clip_preprocess):
        """
        데이터셋 초기화
        
        Args:
            image_paths: 이미지 파일 경로들의 리스트
            labels: 각 이미지의 라벨 (0=안전, 1=유해)
            yolo_model: YOLO 객체 탐지 모델
            clip_model: CLIP 이미지 이해 모델
            clip_preprocess: CLIP용 이미지 전처리 함수
        """
        self.image_paths = image_paths  # 이미지 파일 경로들
        self.labels = labels            # 각 이미지의 라벨
        self.yolo = yolo_model          # YOLO 모델
        self.clip_model = clip_model    # CLIP 모델
        self.clip_preprocess = clip_preprocess  # CLIP 전처리 함수
    
    def __len__(self):
        """데이터셋의 총 개수를 반환"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        특정 인덱스의 데이터를 가져오는 함수
        
        데이터 로더가 배치를 구성할 때 각 샘플을 가져오기 위해 호출됩니다.
        
        Args:
            idx: 가져올 데이터의 인덱스
            
        Returns:
            combined: YOLO와 CLIP 특징이 결합된 벡터
            label: 해당 이미지의 라벨 (0 또는 1)
        """
        img_path = self.image_paths[idx]  # 이미지 파일 경로
        label = self.labels[idx]          # 해당 이미지의 라벨
        
        try:
            # 1. 이미지 로드 및 RGB 변환
            image = Image.open(img_path).convert('RGB')
            
            # 2. YOLO로 객체 탐지 수행
            # YOLO는 이미지에서 어떤 물체들이 있는지 탐지합니다
            yolo_results = self.yolo(img_path, verbose=False)
            yolo_features = self._extract_yolo_features(yolo_results)
            
            # 3. CLIP으로 맥락 분석 수행
            # CLIP은 이미지의 전체적인 맥락과 의미를 이해합니다
            clip_image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():  # 그래디언트 계산 비활성화 (추론만 수행)
                clip_features = self.clip_model.encode_image(clip_image).squeeze()
            
            # 4. YOLO 특징과 CLIP 특징을 결합
            # 객체 정보(YOLO) + 맥락 정보(CLIP)를 합쳐서 최종 특징 생성
            combined = torch.cat([yolo_features.to(DEVICE), clip_features])
            return combined, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            # 오류 발생 시 기본값 반환 (모든 값이 0인 벡터)
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(len(HARMFUL_OBJECTS) + 512).to(DEVICE), torch.tensor(label, dtype=torch.float32)
    
    def _extract_yolo_features(self, results):
        """
        YOLO 탐지 결과를 특징 벡터로 변환하는 함수
        
        YOLO가 탐지한 모든 객체들 중에서 유해 객체 목록에 있는 것들만
        카운트하여 특징 벡터를 만듭니다.
        
        예시:
        - 칼(knife) 2개, 사람(person) 3명 탐지됨
        - 특징 벡터: [2, 0, 0, ..., 3, 0] (유해 객체 순서대로)
        
        Args:
            results: YOLO 모델의 탐지 결과
            
        Returns:
            feature_vector: 유해 객체들의 탐지 횟수를 나타내는 벡터
        """
        # 유해 객체 개수만큼 0으로 초기화된 벡터 생성
        feature_vector = torch.zeros(len(HARMFUL_OBJECTS))
        
        # YOLO 결과에서 탐지된 객체들을 확인
        for result in results:
            if result.boxes is not None:  # 탐지된 객체가 있는 경우
                for box in result.boxes:  # 각 탐지된 객체에 대해
                    # 객체의 클래스 이름을 소문자로 변환
                    class_name = result.names[int(box.cls)].lower()
                    
                    # 유해 객체 목록과 비교하여 매칭되는지 확인
                    for i, obj in enumerate(HARMFUL_OBJECTS):
                        if obj in class_name or class_name in obj:
                            feature_vector[i] += 1  # 해당 유해 객체의 탐지 횟수 증가
        
        return feature_vector


# ============================================================
# 비디오 데이터셋 클래스
# ============================================================
class HarmfulVideoDataset(Dataset):
    """
    비디오 데이터셋 클래스: YOLO + SlowFast + CLIP 특징 추출
    
    이 클래스는 비디오 파일들을 읽어서 여러 AI 모델을 사용하여 특징을 추출합니다:
    - YOLO: 각 프레임에서 객체 탐지 (무기, 위험 물체 등)
    - SlowFast: 비디오의 행동 패턴 분석 (시간적 움직임 이해)
    - CLIP: 각 프레임의 맥락 이해 (장면의 의미 파악)
    
    비디오는 이미지의 연속이므로, 시간적 정보가 중요합니다.
    예를 들어, 칼이 있어도 요리하는 장면이면 안전하지만,
    위협적인 동작과 함께 있으면 유해할 수 있습니다.
    """
    
    def __init__(self, video_paths, labels, yolo_model, slowfast_model, clip_model, clip_preprocess):
        """
        비디오 데이터셋 초기화
        
        Args:
            video_paths: 비디오 파일 경로들의 리스트
            labels: 각 비디오의 라벨 (0=안전, 1=유해)
            yolo_model: YOLO 객체 탐지 모델
            slowfast_model: SlowFast 행동 인식 모델
            clip_model: CLIP 이미지 이해 모델
            clip_preprocess: CLIP용 이미지 전처리 함수
        """
        self.video_paths = video_paths      # 비디오 파일 경로들
        self.labels = labels                # 각 비디오의 라벨
        self.yolo = yolo_model              # YOLO 모델
        self.slowfast = slowfast_model      # SlowFast 모델
        self.clip_model = clip_model        # CLIP 모델
        self.clip_preprocess = clip_preprocess  # CLIP 전처리 함수
    
    def __len__(self):
        """데이터셋의 총 개수를 반환"""
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        특정 인덱스의 비디오 데이터를 가져오는 함수
        
        비디오는 프레임들의 연속이므로, 시계열 데이터로 처리합니다.
        
        Args:
            idx: 가져올 데이터의 인덱스
            
        Returns:
            combined: 모든 특징이 결합된 시계열 데이터 (프레임 수 x 특징 차원)
            label: 해당 비디오의 라벨 (0 또는 1)
        """
        video_path = self.video_paths[idx]  # 비디오 파일 경로
        label = self.labels[idx]            # 해당 비디오의 라벨
        
        try:
            # 1. 비디오에서 프레임들을 샘플링 (16개 프레임 추출)
            # 비디오 전체 길이에서 균등하게 16개를 선택
            frames = self._sample_frames(video_path)
            
            # 2. YOLO로 각 프레임에서 객체 탐지
            # 각 프레임마다 유해 객체가 있는지 확인
            yolo_list = [self._extract_yolo_frame(frame) for frame in frames]
            yolo_seq = torch.stack(yolo_list).to(DEVICE)  # 시계열로 변환 (16 x 유해객체수)
            
            # 3. SlowFast로 비디오 전체의 행동 패턴 분석
            # 빠른 움직임과 느린 움직임을 동시에 분석하여 행동 이해
            slowfast_feat = self._extract_slowfast_features(frames)
            
            # 4. CLIP으로 각 프레임의 맥락 분석
            # 각 프레임이 어떤 상황인지 맥락 파악
            clip_list = [self._extract_clip_frame(frame) for frame in frames]
            clip_seq = torch.stack(clip_list).to(DEVICE)  # 시계열로 변환 (16 x 512)
            
            # 5. 모든 특징들을 결합
            yolo_clip_seq = torch.cat([yolo_seq, clip_seq], dim=1)  # YOLO + CLIP 결합
            slowfast_expanded = slowfast_feat.unsqueeze(0).repeat(FRAME_SAMPLE, 1)  # SlowFast를 모든 프레임에 복사
            combined = torch.cat([yolo_clip_seq, slowfast_expanded], dim=1)  # 최종 결합
            
            return combined, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            # 오류 발생 시 기본값 반환
            print(f"Error loading {video_path}: {e}")
            default_dim = len(HARMFUL_OBJECTS) + 512 + 2048  # YOLO + CLIP + SlowFast 차원
            return torch.zeros(FRAME_SAMPLE, default_dim).to(DEVICE), torch.tensor(label, dtype=torch.float32)
    
    def _sample_frames(self, video_path):
        """
        비디오에서 균등 간격으로 프레임을 샘플링하는 함수
        
        비디오의 전체 내용을 파악하기 위해 시작부터 끝까지
        균등하게 16개 프레임을 추출합니다.
        
        예시:
        - 100프레임 비디오 → 0, 6, 13, 19, ..., 94, 100번째 프레임 추출
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            frames: 샘플링된 프레임들의 리스트 (16개)
        """
        # 비디오 파일 열기
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수
        
        # 비디오가 비어있거나 읽을 수 없는 경우
        if total == 0:
            cap.release()
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(FRAME_SAMPLE)]
        
        # 균등하게 16개 프레임 인덱스 생성 (0부터 total-1까지)
        indices = np.linspace(0, max(0, total-1), FRAME_SAMPLE, dtype=int)
        frames = []
        
        # 각 인덱스에 해당하는 프레임 읽기
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # 해당 프레임으로 이동
            ret, frame = cap.read()  # 프레임 읽기
            if ret:
                # BGR을 RGB로 변환하여 저장 (OpenCV는 BGR 순서 사용)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                # 읽기 실패 시 검은 이미지로 대체
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()  # 비디오 파일 닫기
        
        # 프레임이 부족한 경우 마지막 프레임을 복사하거나 검은 이미지로 채움
        while len(frames) < FRAME_SAMPLE:
            if frames:
                frames.append(frames[-1])  # 마지막 프레임 복사
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))  # 검은 이미지
        
        return frames[:FRAME_SAMPLE]  # 정확히 16개만 반환
    
    def _extract_yolo_frame(self, frame):
        """
        단일 프레임에서 YOLO 특징을 추출하는 함수
        
        Args:
            frame: 분석할 프레임 (numpy 배열)
            
        Returns:
            feature: 유해 객체들의 탐지 횟수를 나타내는 벡터
        """
        # 임시 파일로 프레임 저장 (YOLO는 파일 경로를 요구)
        temp_path = 'temp_frame.jpg'
        cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # YOLO로 객체 탐지 수행
        results = self.yolo(temp_path, verbose=False)
        
        # 유해 객체 개수만큼 0으로 초기화된 벡터 생성
        feature = torch.zeros(len(HARMFUL_OBJECTS))
        
        # 탐지된 객체들을 확인하여 유해 객체 카운트
        for result in results:
            if result.boxes is not None:  # 탐지된 객체가 있는 경우
                for box in result.boxes:  # 각 탐지된 객체에 대해
                    class_name = result.names[int(box.cls)].lower()  # 클래스 이름을 소문자로
                    for i, obj in enumerate(HARMFUL_OBJECTS):
                        if obj in class_name or class_name in obj:
                            feature[i] += 1  # 해당 유해 객체의 탐지 횟수 증가
        
        # 임시 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return feature
    
    def _extract_slowfast_features(self, frames):
        """
        SlowFast를 사용한 행동 인식 특징 추출 함수
        
        SlowFast는 두 개의 경로(pathway)를 사용합니다:
        - Slow pathway: 원본 프레임으로 정적 정보 파악 (공간 정보)
        - Fast pathway: 빠르게 샘플링된 프레임으로 동적 정보 파악 (시간 정보)
        
        이 두 가지를 결합하여 행동을 정확하게 인식할 수 있습니다.
        예: 팔을 휘두르는 동작이 공격인지 인사인지 구별
        
        Args:
            frames: 분석할 프레임들의 리스트
            
        Returns:
            features: SlowFast로 추출된 행동 특징 벡터 (2048차원)
        """
        # SlowFast 모델이 없는 경우 기본값 반환
        if self.slowfast is None:
            return torch.zeros(2048).to(DEVICE)
        
        try:
            # 1. 프레임들을 텐서로 변환하고 정규화 (0-255 -> 0-1)
            frame_tensors = []
            for f in frames:
                resized = cv2.resize(f, (224, 224))  # SlowFast는 224x224 크기 요구
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0  # HWC -> CHW, 정규화
                frame_tensors.append(tensor)
            
            # 2. 프레임들을 하나의 비디오 텐서로 변환
            # [T, C, H, W] -> [C, T, H, W] (시간 차원을 두 번째로 이동)
            video_tensor = torch.stack(frame_tensors).permute(1, 0, 2, 3).unsqueeze(0)
            
            # 3. SlowFast는 두 개의 경로를 사용
            # Slow pathway: 원본 프레임들 (정적 정보 - 무엇이 있는지)
            slow_pathway = video_tensor.to(DEVICE)
            
            # Fast pathway: 시간 축에서 균등하게 샘플링 (4프레임만 사용)
            # SlowFast R50은 보통 slow:fast = 4:1 비율 사용
            # Fast는 빠른 움직임을 포착 (어떻게 움직이는지)
            fast_indices = torch.linspace(0, FRAME_SAMPLE - 1, FRAME_SAMPLE // 4).long()
            fast_pathway = video_tensor[:, :, fast_indices, :, :].to(DEVICE)
            
            # 4. SlowFast 모델로 특징 추출
            with torch.no_grad():  # 그래디언트 계산 비활성화 (추론만)
                features = self.slowfast([slow_pathway, fast_pathway])
            
            return features.squeeze().to(DEVICE)
            
        except Exception as e:
            # 오류 발생 시 기본값 반환
            print(f"SlowFast error: {e}")
            return torch.zeros(2048).to(DEVICE)
    
    def _extract_clip_frame(self, frame):
        """
        단일 프레임에서 CLIP 특징을 추출하는 함수
        
        CLIP은 이미지의 전체적인 맥락을 이해합니다.
        예: 칼이 있어도 주방 장면이면 안전, 어두운 골목이면 위험
        
        Args:
            frame: 분석할 프레임 (numpy 배열)
            
        Returns:
            clip_features: CLIP으로 추출된 맥락 특징 벡터 (512차원)
        """
        # numpy 배열을 PIL Image로 변환
        image = Image.fromarray(frame)
        
        # CLIP 전처리 수행 (크기 조정, 정규화 등)
        clip_image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)
        
        # CLIP 모델로 이미지 특징 추출
        with torch.no_grad():  # 그래디언트 계산 비활성화 (추론만)
            clip_features = self.clip_model.encode_image(clip_image).squeeze()
        
        return clip_features


# ============================================================
# 이미지 분류 모델
# ============================================================
class ImageHarmfulClassifier(nn.Module):
    """
    이미지 유해 콘텐츠 분류를 위한 3층 MLP (Multi-Layer Perceptron) 모델
    
    YOLO와 CLIP에서 추출된 특징을 입력받아 유해/안전을 분류합니다.
    
    구조:
    입력 (YOLO + CLIP) → 256 → 128 → 64 → 1 (유해 확률)
    
    각 층마다 ReLU 활성화 함수와 Dropout을 사용하여
    비선형 표현력을 높이고 과적합을 방지합니다.
    """
    def __init__(self, yolo_dim, clip_dim):
        """
        이미지 분류기 초기화
        
        Args:
            yolo_dim: YOLO 특징의 차원 (유해 객체 개수)
            clip_dim: CLIP 특징의 차원 (512)
        """
        super(ImageHarmfulClassifier, self).__init__()
        input_dim = yolo_dim + clip_dim  # 총 입력 차원
        
        # 3층 MLP 네트워크 구성
        self.mlp = nn.Sequential(
            # 첫 번째 층: 입력 -> 256
            nn.Linear(input_dim, 256),  # 완전 연결층
            nn.ReLU(),                   # 활성화 함수 (음수는 0으로, 양수는 그대로)
            nn.Dropout(0.4),             # 과적합 방지를 위한 드롭아웃 (40% 뉴런 끔)
            
            # 두 번째 층: 256 -> 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),             # 드롭아웃 (30%)
            
            # 세 번째 층: 128 -> 64
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),             # 드롭아웃 (20%)
            
            # 출력 층: 64 -> 1 (유해 확률)
            nn.Linear(64, 1),
            nn.Sigmoid()                 # 0~1 사이의 확률값 출력
        )
    
    def forward(self, x):
        """
        순전파 함수 (Forward Pass)
        
        입력 데이터를 네트워크에 통과시켜 예측값을 계산합니다.
        
        Args:
            x: 입력 특징 벡터 (YOLO + CLIP 특징)
            
        Returns:
            유해 확률 (0~1 사이의 값)
            - 0에 가까우면 안전
            - 1에 가까우면 유해
        """
        return self.mlp(x)


# ============================================================
# 비디오 분류 모델
# ============================================================
class VideoHarmfulClassifier(nn.Module):
    """
    비디오 유해 콘텐츠 분류를 위한 Transformer 기반 시계열 분류기
    
    YOLO, CLIP, SlowFast 특징을 시계열로 처리하여 유해/안전을 분류합니다.
    
    Transformer를 사용하는 이유:
    - 비디오는 프레임들의 연속이므로 시간적 관계가 중요합니다
    - Transformer의 self-attention은 프레임 간의 관계를 학습할 수 있습니다
    - 예: 1번 프레임의 칼 + 10번 프레임의 위협적 동작 = 위험
    """
    def __init__(self, yolo_dim, clip_dim, slowfast_dim=2048):
        """
        비디오 분류기 초기화
        
        Args:
            yolo_dim: YOLO 특징의 차원 (유해 객체 개수)
            clip_dim: CLIP 특징의 차원 (512)
            slowfast_dim: SlowFast 특징의 차원 (2048)
        """
        super(VideoHarmfulClassifier, self).__init__()
        input_dim = yolo_dim + clip_dim + slowfast_dim  # 총 입력 차원
        
        # Transformer 인코더 레이어 정의
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,        # 입력 차원
            nhead=2,                  # 어텐션 헤드 개수 (2개로 설정)
            dim_feedforward=512,      # 피드포워드 네트워크 차원
            batch_first=True,         # 배치 차원을 첫 번째로
            dropout=0.3               # 드롭아웃 (30%)
        )
        
        # Transformer 인코더 (2개 레이어 스택)
        # 각 레이어는 프레임 간의 관계를 학습합니다
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 최종 분류기 (MLP)
        # Transformer의 출력을 받아 최종 유해 확률을 예측
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        순전파 함수 (Forward Pass)
        
        Args:
            x: 입력 시계열 특징 (프레임 수 x 특징 차원)
               예: [16, 2576] = 16프레임, 각 프레임은 2576차원 특징
            
        Returns:
            유해 확률 (0~1 사이의 값)
        """
        # Transformer로 시계열 특징 변환
        # self-attention으로 프레임 간 관계 학습
        transformed = self.transformer(x)
        
        # 시간 차원에 대해 평균 풀링 (모든 프레임의 정보를 하나로 압축)
        # [배치, 16프레임, 특징] -> [배치, 특징]
        pooled = transformed.mean(dim=1)
        
        # 최종 분류 수행
        return self.classifier(pooled)


# ============================================================
# 데이터 준비 함수들 (유해/안전/공개/실제 통합)
# ============================================================
def prepare_image_data():
    """
    이미지 데이터를 통합하여 로드하고 학습/검증으로 분할하는 함수
    
    여러 데이터셋을 통합합니다:
    1. HOD Dataset: 공개 유해 이미지 데이터셋
    2. COCO Safe Dataset: 공개 안전 이미지 데이터셋
    3. 실제 수집 유해 이미지
    4. 실제 수집 안전 이미지
    
    Returns:
        X_train, y_train: 학습용 이미지 경로와 라벨
        X_val, y_val: 검증용 이미지 경로와 라벨
    """
    print("\n이미지 데이터 준비 중...")
    
    # 1. HOD Dataset (유해, 공개 데이터셋)
    hod_path = DATA_PATH + '1_공개_데이터셋/HOD_Dataset/dataset/'
    hod_images, hod_labels = [], []
    if os.path.exists(hod_path):
        for root, _, files in os.walk(hod_path):  # 모든 하위 폴더 탐색
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):  # 이미지 파일만
                    hod_images.append(os.path.join(root, file))
                    hod_labels.append(1)  # 유해 라벨
    
    # 2. COCO Safe Dataset (안전, 공개 데이터셋)
    coco_path = DATA_PATH + '1_공개_데이터셋/COCO_Safe_Dataset/'
    coco_images, coco_labels = [], []
    if os.path.exists(coco_path):
        for root, _, files in os.walk(coco_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    coco_images.append(os.path.join(root, file))
                    coco_labels.append(0)  # 안전 라벨
    
    # 3. 실제 수집 유해 이미지
    verified_path = DATA_PATH + '3_라벨링_파일/verified_labels.json'
    verified_images, verified_labels = [], []
    img_dir = DATA_PATH + '2_실제_수집_데이터/이미지/'
    if os.path.exists(verified_path):
        with open(verified_path, 'r', encoding='utf-8') as f:
            vdata = json.load(f)  # JSON 파일에서 라벨 정보 로드
        for filename, detections in vdata.items():
            img_path = os.path.join(img_dir, filename)
            if os.path.exists(img_path):
                verified_images.append(img_path)
                # 탐지된 객체가 있으면 유해(1), 없으면 안전(0)
                verified_labels.append(1 if len(detections) > 0 else 0)
    
    # 4. 실제 수집 안전 이미지
    safe_path = DATA_PATH + '2_실제_수집_데이터/이미지_안전/'
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
    
    # ★ 모든 데이터셋 통합 및 셔플
    X = hod_images + coco_images + verified_images + safe_images
    y = hod_labels + coco_labels + verified_labels + safe_labels
    
    # 데이터를 무작위로 섞기 (일관된 결과를 위해 random_state=42 사용)
    from sklearn.utils import shuffle
    X, y = shuffle(X, y, random_state=42)
    
    # 학습/검증 데이터로 분할 (15%는 검증용)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)  # stratify로 클래스 비율 유지
    
    print(f"✓ 데이터셋 통합 완료 (학습 {len(X_train)}, 검증 {len(X_val)})")
    print(f"  (유해: {sum(y_train)}, 안전: {len(y_train)-sum(y_train)} / 검증 유해: {sum(y_val)}, 안전: {len(y_val)-sum(y_val)})")
    return X_train, y_train, X_val, y_val


def prepare_video_data():
    """
    비디오 데이터를 통합하여 로드하고 학습/검증으로 분할하는 함수
    
    비디오 데이터셋을 통합합니다:
    1. 실제 수집 유해 비디오
    2. 실제 수집 안전 비디오
    
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
            vdata = json.load(f)  # JSON 파일에서 비디오 라벨 정보 로드
        for fn, info in vdata.items():  # 파일명과 정보
            vp = os.path.join(video_dir, fn)
            if os.path.exists(vp):
                vpaths.append(vp)
                # JSON의 is_harmful 필드에 따라 라벨 결정
                vlabels.append(1 if info['is_harmful'] else 0)
    
    # 2. 실제 수집 안전 비디오
    safe_video_dir = DATA_PATH + '2_실제_수집_데이터/비디오_안전/'
    safe_video_json = DATA_PATH + '3_라벨링_파일/safe_video_labels.json'
    svpaths, svlabels = [], []
    if os.path.exists(safe_video_json):
        with open(safe_video_json, 'r', encoding='utf-8') as f:
            sdata = json.load(f)
        for fn in sdata.keys():  # 안전 비디오 파일명들
            full = os.path.join(safe_video_dir, fn)
            if os.path.exists(full):
                svpaths.append(full)
                svlabels.append(0)  # 안전 라벨
    
    # ★ 모든 비디오 데이터 통합 및 셔플
    X = vpaths + svpaths
    y = vlabels + svlabels
    
    # 데이터를 무작위로 섞기 (일관된 결과를 위해 random_state=42 사용)
    from sklearn.utils import shuffle
    X, y = shuffle(X, y, random_state=42)
    
    # 학습/검증 데이터로 분할 (15%는 검증용)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)  # stratify로 클래스 비율 유지
    
    print(f"✓ 통합 완료 (학습 {len(X_train)}, 검증 {len(X_val)})")
    print(f"  (유해: {sum(y_train)}, 안전: {len(y_train)-sum(y_train)} / 검증 유해: {sum(y_val)}, 안전: {len(y_val)-sum(y_val)})")
    return X_train, y_train, X_val, y_val


# ============================================================
# 이미지 모델 학습 함수
# ============================================================
def train_image_model():
    """
    이미지 모델 학습을 수행하는 메인 함수
    
    이 함수는 다음 과정을 수행합니다:
    1. 데이터 준비 및 로드
    2. YOLO, CLIP 모델 로드
    3. 데이터셋 및 데이터로더 생성
    4. 모델, 손실함수, 옵티마이저 설정
    5. 학습 및 검증 루프
    6. 성능 평가 및 모델 저장
    7. Confusion Matrix 생성
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
    yolo = YOLO('yolov8n.pt')  # YOLO 객체 탐지 모델 로드 (nano 버전)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)  # CLIP 모델 로드
    
    # CLIP 차원 자동 감지 (더미 이미지로 테스트)
    with torch.no_grad():
        dummy_image = torch.randn(1, 3, 224, 224).to(DEVICE)
        clip_features = clip_model.encode_image(dummy_image)
        clip_dim = clip_features.shape[-1]  # CLIP 특징 차원 (512)
    
    yolo_dim = len(HARMFUL_OBJECTS)  # YOLO 특징 차원 (유해 객체 개수)
    print(f"✓ YOLO 차원: {yolo_dim}")
    print(f"✓ CLIP 차원: {clip_dim}")
    print(f"✓ 총 입력 차원: {yolo_dim + clip_dim}")
    
    # 3. 데이터셋 및 데이터로더 생성
    train_dataset = HarmfulImageDataset(train_images, train_labels, yolo, clip_model, clip_preprocess)
    val_dataset = HarmfulImageDataset(val_images, val_labels, yolo, clip_model, clip_preprocess)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. 모델, 손실함수, 옵티마이저 설정
    model = ImageHarmfulClassifier(yolo_dim=yolo_dim, clip_dim=clip_dim).to(DEVICE)
    criterion = nn.BCELoss()  # 이진 분류를 위한 Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=IMAGE_LR)  # Adam 옵티마이저
    
    best_f1 = 0  # 최고 F1 점수 추적
    print(f"\n학습 시작 (Epochs: {IMAGE_EPOCHS})")
    
    # 5. 학습 및 검증 루프
    for epoch in range(IMAGE_EPOCHS):
        # ====== 학습 단계 ======
        model.train()  # 학습 모드로 설정 (Dropout, BatchNorm 활성화)
        train_loss = 0
        
        # 배치별로 학습 진행
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{IMAGE_EPOCHS}"):
            features, labels = features.to(DEVICE), labels.to(DEVICE)  # GPU로 이동
            
            optimizer.zero_grad()  # 그래디언트 초기화
            outputs = model(features).squeeze()  # 모델 예측
            loss = criterion(outputs, labels)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트
            train_loss += loss.item()
        
        # ====== 검증 단계 ======
        model.eval()  # 평가 모드로 설정 (Dropout 비활성화, BatchNorm 고정)
        val_preds = []
        val_true = []
        
        with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약)
            for features, labels in val_loader:
                features = features.to(DEVICE)
                outputs = model(features).squeeze()
                preds = (outputs > 0.5).cpu().numpy()  # 0.5 임계값으로 이진 분류
                val_preds.extend(np.atleast_1d(preds))
                val_true.extend(np.atleast_1d(labels.numpy()))
        
        # 6. 평가 지표 계산
        if len(val_true) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_true, val_preds, average='binary', zero_division=0)
            
            print(f"\nEpoch {epoch+1}/{IMAGE_EPOCHS}")
            print(f"  Loss: {train_loss/len(train_loader):.4f}")
            print(f"  Precision: {precision:.4f}")  # 정밀도: 유해라고 예측한 것 중 실제 유해 비율
            print(f"  Recall: {recall:.4f}")         # 재현율: 실제 유해 중 찾아낸 비율
            print(f"  F1-Score: {f1:.4f}")           # F1 점수: Precision과 Recall의 조화평균
            
            # 최고 성능 모델 저장
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), 'image_model_best.pth')
                print(f"  ✓ 모델 저장 (Best F1: {best_f1:.4f})")
    
    # 7. 학습 완료 요약
    print(f"\n{'='*60}")
    print(f"✅ 이미지 모델 학습 완료!")
    print(f"   Best F1-Score: {best_f1:.4f}")
    if best_f1 >= 0.75:
        print(f"   ✅ 필수 목표 달성! (F1 ≥ 0.75)")
    else:
        print(f"   ⚠️  목표 미달 (F1 < 0.75) - 더 많은 epoch 필요")
    
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
# 비디오 모델 학습 함수
# ============================================================
def train_video_model():
    """
    비디오 모델 학습을 수행하는 메인 함수
    
    이 함수는 다음 과정을 수행합니다:
    1. 비디오 데이터 준비 및 로드
    2. YOLO, CLIP, SlowFast 모델 로드
    3. 데이터셋 및 데이터로더 생성
    4. 모델, 손실함수, 옵티마이저 설정
    5. 학습 및 검증 루프
    6. 성능 평가 및 모델 저장
    7. Confusion Matrix 생성
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
    slowfast_model.eval()  # SlowFast는 고정 (학습하지 않음, 특징 추출기로만 사용)
    slowfast_dim = 2048  # SlowFast 출력 차원
    print("✓ SlowFast 로드 완료")
    
    # CLIP 차원 자동 감지
    with torch.no_grad():
        dummy_image = torch.randn(1, 3, 224, 224).to(DEVICE)
        clip_features = clip_model.encode_image(dummy_image)
        clip_dim = clip_features.shape[-1]  # CLIP 특징 차원 (512)
    
    yolo_dim = len(HARMFUL_OBJECTS)  # YOLO 특징 차원
    print(f"✓ YOLO 차원: {yolo_dim}")
    print(f"✓ SlowFast 차원: {slowfast_dim}")
    print(f"✓ CLIP 차원: {clip_dim}")
    print(f"✓ 총 입력 차원: {yolo_dim + slowfast_dim + clip_dim}")
    
    # 3. 데이터셋 및 데이터로더 생성
    train_dataset = HarmfulVideoDataset(train_paths, train_labels, yolo, slowfast_model, clip_model, clip_preprocess)
    val_dataset = HarmfulVideoDataset(val_paths, val_labels, yolo, slowfast_model, clip_model, clip_preprocess)
    train_loader = DataLoader(train_dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. 모델, 손실함수, 옵티마이저 설정
    model = VideoHarmfulClassifier(yolo_dim=yolo_dim, clip_dim=clip_dim, slowfast_dim=slowfast_dim).to(DEVICE)
    criterion = nn.BCELoss()  # 이진 분류를 위한 Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=VIDEO_LR)  # Adam 옵티마이저
    
    best_f1 = 0  # 최고 F1 점수 추적
    print(f"\n학습 시작 (Epochs: {VIDEO_EPOCHS})")
    
    # 5. 학습 및 검증 루프
    for epoch in range(VIDEO_EPOCHS):
        # ====== 학습 단계 ======
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
        
        # ====== 검증 단계 ======
        model.eval()  # 평가 모드로 설정
        val_preds = []
        val_true = []
        
        with torch.no_grad():  # 그래디언트 계산 비활성화
            for features, labels_batch in val_loader:
                features = features.to(DEVICE)
                outputs = model(features).squeeze()
                preds = (outputs > 0.5).cpu().numpy()  # 0.5 임계값으로 이진 분류
                val_preds.extend(np.atleast_1d(preds))
                val_true.extend(np.atleast_1d(labels_batch.numpy()))
        
        # 6. 평가 지표 계산
        if len(val_true) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_true, val_preds, average='binary', zero_division=0
            )
            
            print(f"\nEpoch {epoch+1}/{VIDEO_EPOCHS}")
            print(f"  Loss: {train_loss/len(train_loader):.4f}")
            print(f"  Precision: {precision:.4f}")  # 정밀도
            print(f"  Recall: {recall:.4f}")         # 재현율
            print(f"  F1-Score: {f1:.4f}")           # F1 점수
            
            # 최고 성능 모델 저장
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), 'video_model_best.pth')
                print(f"  ✓ 모델 저장 (Best F1: {best_f1:.4f})")
    
    # 7. 학습 완료 요약
    print(f"\n{'='*60}")
    print(f"✅ 비디오 모델 학습 완료!")
    print(f"   Best F1-Score: {best_f1:.4f}")
    if best_f1 >= 0.75:
        print(f"   ✅ 필수 목표 달성! (F1 ≥ 0.75)")
    else:
        print(f"   ⚠️  목표 미달 (F1 < 0.75) - 더 많은 epoch 필요")
    
    # 8. Confusion Matrix 생성
    if len(val_true) > 0:
        cm = confusion_matrix(val_true, val_preds)
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
# 메인 실행 부분
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
        print("="*60)
