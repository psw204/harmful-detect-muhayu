"""
모델 클래스 정의 - Final Model 11 기반 (카테고리 구조)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

# ============================================================
# 유해 콘텐츠 정의 (카테고리 기반 구조)
# ============================================================

# 1) 최종 카테고리 (모델 출력용 9 클래스)
CATEGORIES = [
    "weapons",    # 무기
    "violence",   # 폭력
    "alcohol",    # 음주
    "smoking",    # 흡연
    "drugs",      # 약물
    "blood",      # 혈액/상처
    "threat",     # 위협
    "sexual",     # 성적 콘텐츠
    "dangerous"   # 위험행동
]

# 2) 객체 기반 탐지 (YOLO용) - 카테고리별 매핑
OBJECT_MAP = {
    "weapons": [
        "knife", "dagger", "machete", "sword", "axe",
        "gun", "pistol", "rifle", "shotgun", "machine_gun",
        "grenade", "bomb"
    ],
    "alcohol": [
        "wine glass", "beer"
    ],
    "smoking": [
        "cigarette", "lighter"
    ],
    "drugs": [
        "syringe"
    ],
    "blood": [
        "blood", "injury", "wound"
    ]
}

# 전체 객체 목록 (YOLO 특징 벡터 차원)
ALL_OBJECTS = list(set([obj for lst in OBJECT_MAP.values() for obj in lst]))

# 무기 객체 (별도 관리)
HARMFUL_OBJECTS = OBJECT_MAP["weapons"]

# 맥락 기반 보조 객체 (유해성 판단에 도움)
CONTEXTUAL_OBJECTS = [
    "wine glass", "beer",
    "cigarette", "lighter",
    "syringe"
]

# 3) 행동/맥락 기반 탐지 (CLIP/행동 인식용) - weapons 제외 (객체 기반만)
BEHAVIOR_CATEGORIES = [
    "violence",
    "alcohol",
    "smoking",
    "drugs",
    "blood",
    "threat",
    "sexual",
    "dangerous"
]

# 행동 감지 프롬프트 (Zero-shot CLIP 기반, 카테고리별 여러 프롬프트 지원)
BEHAVIOR_PROMPTS = {
    "violence": [
        "people fighting",
        "person assaulting another person",
        "people fighting violently"
    ],
    "alcohol": [
        "person drinking alcohol",
        "drunk person"
    ],
    "smoking": [
        "person smoking a cigarette",
        "person smoking cigarette"
    ],
    "drugs": [
        "person using illegal drugs",
        "person injecting drugs"
    ],
    "blood": [
        "blood",
        "injury",
        "wound",
        "person with injury"
    ],
    "threat": [
        "person threatening someone",
        "person pointing a weapon at someone",
        "threatening with weapon"
    ],
    "sexual": [
        "adult content",
        "sexualized pose",
        "revealing clothing",
        "partial nudity",
        "sexual assault"
    ],
    "dangerous": [
        "person self-harming",
        "person attempting suicide",
        "person doing a reckless stunt"
    ]
}

# 호환성을 위한 기존 변수명 유지 (BEHAVIOR_CATEGORIES와 매핑)
HARMFUL_BEHAVIORS = BEHAVIOR_CATEGORIES  # 8차원


class HarmfulImageClassifier(nn.Module):
    """
    유해 이미지 분류 모델 (차원 축소 포함)
    
    아키텍처:
    입력 (540차원) → 차원 축소 (256차원) → MLP → 출력 (1차원, 유해 확률)
    """
    def __init__(self, yolo_dim, clip_dim, behavior_dim):
        """
        Args:
            yolo_dim: YOLO 특징 차원 (20)
            clip_dim: CLIP 특징 차원 (512)
            behavior_dim: 행동 특징 차원 (8)
        """
        super().__init__()
        input_dim = yolo_dim + clip_dim + behavior_dim  # 540
        
        self.dimension_reduction = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: 입력 특징 (batch_size, 540)
            
        Returns:
            output: 유해 확률 (batch_size,)
        """
        x = self.dimension_reduction(x)
        return self.mlp(x).squeeze()


class HarmfulVideoClassifier(nn.Module):
    """
    유해 비디오 분류 모델 (차원 축소 + Transformer)
    
    아키텍처:
    입력 (32, 940차원) → 차원 축소 (32, 256차원) → Transformer → 시간 평균 풀링 → MLP → 출력 (1차원)
    """
    def __init__(self, yolo_dim, clip_dim, slowfast_dim, behavior_dim):
        """
        Args:
            yolo_dim: YOLO 특징 차원 (20)
            clip_dim: CLIP 특징 차원 (512)
            slowfast_dim: SlowFast 특징 차원 (400)
            behavior_dim: 행동 특징 차원 (8)
        """
        super().__init__()
        input_dim = yolo_dim + clip_dim + slowfast_dim + behavior_dim  # 940
        
        reduced_dim = 256
        self.dimension_reduction = nn.Sequential(
            nn.Linear(input_dim, reduced_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        
        nhead = 8
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=reduced_dim,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True,
            dropout=0.4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.classifier = nn.Sequential(
            nn.Linear(reduced_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: 입력 특징 시퀀스 (batch_size, 32, 940)
            
        Returns:
            output: 유해 확률 (batch_size,)
        """
        batch_size, seq_len, feat_dim = x.shape
        x = x.view(-1, feat_dim)
        x = self.dimension_reduction(x)
        x = x.view(batch_size, seq_len, -1)
        
        transformed = self.transformer(x)
        pooled = transformed.mean(dim=1)
        logits = self.classifier(pooled)
        
        return torch.sigmoid(logits).squeeze()
