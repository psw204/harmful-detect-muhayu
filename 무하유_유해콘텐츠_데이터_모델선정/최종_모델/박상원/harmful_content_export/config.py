"""
설정 파일 - Final Model 11 기반 (카테고리 구조)
모델 경로, 하이퍼파라미터, 디바이스 설정
"""

import torch
import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent

# 모델 경로 설정 (final_model11의 pth 파일 사용)
current = PROJECT_ROOT
final_model11_path = None
while current.parent != current:
    if current.name == "박상원":
        potential_path = current / "모델_발전_과정" / "final_model11"
        if potential_path.exists():
            final_model11_path = potential_path
            break
    current = current.parent

# 절대 경로로 시도
if final_model11_path is None:
    abs_path = Path(r"C:\Users\psw20\OneDrive\바탕 화면\PSW\한국항공대학교_3-2\무하유\Github\박상원\모델_발전_과정\final_model11")
    if abs_path.exists():
        final_model11_path = abs_path

if final_model11_path is None:
    raise FileNotFoundError("final_model11 디렉토리를 찾을 수 없습니다.")

WEIGHTS_DIR = final_model11_path
IMAGE_MODEL_PATH = WEIGHTS_DIR / "image_model_best.pth"
VIDEO_MODEL_PATH = WEIGHTS_DIR / "video_model_best.pth"
YOLO_MODEL_PATH = WEIGHTS_DIR / "yolov8n.pt"

# YOLO 모델이 없으면 harmful_content_demo에서 가져오기
if not YOLO_MODEL_PATH.exists():
    DEMO_DIR = PROJECT_ROOT.parent / "harmful_content_demo"
    DEMO_WEIGHTS_DIR = DEMO_DIR / "weights"
    if DEMO_WEIGHTS_DIR.exists():
        YOLO_MODEL_PATH = DEMO_WEIGHTS_DIR / "yolov8n.pt"

# 데이터 경로 설정
current = PROJECT_ROOT
while current.name != "무하유_유해콘텐츠_데이터_모델선정" and current.parent != current:
    current = current.parent

DATA_ROOT = current
LABELS_FILE = DATA_ROOT / "3_라벨링_파일" / "박상원" / "박상원_labels_categorized.json"
IMAGE_DIR = DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "이미지"
SAFE_IMAGE_DIR = DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "안전_이미지"
VIDEO_DIR = DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "비디오"
SAFE_VIDEO_DIR = DATA_ROOT / "2_실제_수집_데이터" / "박상원" / "안전_비디오"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Threshold는 체크포인트에서 자동 로드
IMAGE_THRESHOLD = None
VIDEO_THRESHOLD = None

FRAME_SAMPLE = 32  # SlowFast 호환을 위해 32프레임 사용

CLIP_MODEL_NAME = "ViT-B/32"

SLOWFAST_DIM = 400

# 모델 차원 설정 (카테고리 기반 구조)
YOLO_DIM = 20
CLIP_DIM = 512
BEHAVIOR_DIM = 8

IMAGE_INPUT_DIM = YOLO_DIM + CLIP_DIM + BEHAVIOR_DIM  # 540
VIDEO_INPUT_DIM = YOLO_DIM + CLIP_DIM + SLOWFAST_DIM + BEHAVIOR_DIM  # 940

def print_config():
    """설정 정보 출력"""
    print("=" * 60)
    print("설정 정보 (Final Model 11 기반)")
    print("=" * 60)
    print(f"디바이스: {DEVICE}")
    print(f"이미지 모델 경로: {IMAGE_MODEL_PATH}")
    print(f"비디오 모델 경로: {VIDEO_MODEL_PATH}")
    print(f"YOLO 모델: {YOLO_MODEL_PATH}")
    print(f"데이터 경로: {DATA_ROOT}")
    print(f"라벨 파일: {LABELS_FILE}")
    print(f"YOLO 차원: {YOLO_DIM}, CLIP 차원: {CLIP_DIM}, 행동 차원: {BEHAVIOR_DIM}")
    print(f"이미지 입력 차원: {IMAGE_INPUT_DIM}, 비디오 입력 차원: {VIDEO_INPUT_DIM}")
    print("=" * 60)
