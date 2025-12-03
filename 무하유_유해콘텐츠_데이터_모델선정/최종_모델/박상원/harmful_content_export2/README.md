# 유해 콘텐츠 탐지 모델 평가 - Export2 버전

`final_model11`의 최종 모델을 사용하여 실제 수집 데이터를 평가하는 스크립트입니다.

## 개요

이 디렉토리는 학습된 모델의 성능을 실제 수집 데이터로 평가하기 위한 평가 스크립트를 포함합니다. 모델은 `final_model11`에서 학습된 카테고리 기반 구조를 사용합니다.

## 파일 구조

```
harmful_content_export2/
├── config.py                # 설정 파일 (모델 경로, 데이터 경로 등)
├── models.py                # 모델 클래스 정의
├── inference.py             # 추론 함수들
├── evaluate_category.py     # 카테고리별 평가 스크립트 (메인)
├── export2_evaluate_category 1200개 실행결과*.txt  # 평가 로그
├── yolov8n.pt               # YOLOv8 모델 가중치
├── README.md                # 이 파일
└── 영재님 코드/              # 교차 샘플링 비교를 위한 추가 스크립트
    ├── inference_before.py
    ├── inference_after.py
    ├── fusion_scores.py
    └── ...
```

## 사용 방법

### 1. 환경 설정

필요한 패키지를 설치합니다:

```bash
pip install torch torchvision torchaudio
pip install ultralytics clip-by-openai pytorchvideo
pip install pillow opencv-python scikit-learn tqdm numpy
```

### 2. 평가 실행

```bash
python evaluate_category.py
```

## 평가 결과

스크립트는 다음 메트릭을 계산합니다:

- **Accuracy**: 전체 정확도
- **Precision**: 정밀도 (유해로 예측한 것 중 실제 유해인 비율)
- **Recall**: 재현율 (실제 유해인 것 중 유해로 예측한 비율)
- **F1-Score**: Precision과 Recall의 조화 평균
- **Confusion Matrix**: 혼동 행렬 (TN, FP, FN, TP)

## 모델 정보

- **이미지 모델**: YOLOv8 + CLIP + 행동 인식 + 차원 축소 MLP
- **비디오 모델**: YOLOv8 + SlowFast + CLIP + 행동 인식 + Transformer
- **Threshold**: 모델 체크포인트에서 자동 로드
- **FRAME_SAMPLE**: 32 (SlowFast 호환)

## 탐지 항목

**유해 객체 (20종, 카테고리 기반)**
- **무기류 (12종)**: knife, dagger, machete, sword, axe, gun, pistol, rifle, shotgun, machine_gun, grenade, bomb
- **음주 (2종)**: wine glass, beer
- **흡연 (2종)**: cigarette, lighter
- **약물 (1종)**: syringe
- **혈액/상처 (3종)**: blood, injury, wound

**유해 행동 (8종)**
- violence, alcohol, smoking, drugs, blood, threat, sexual, dangerous

## 주의사항 / 차이점

- 이 디렉토리는 `harmful_content_export`의 모델을 **임영재 샘플링 방식**으로 재평가하기 위한 비교 실험 버전입니다.
- `evaluate_category.py`는 이미지/비디오 전체 성능뿐 아니라 카테고리별 지표 및 혼동 행렬을 함께 출력합니다.
- `영재님 코드/` 폴더에는 상호 샘플링 방식을 맞추기 위한 추가 스크립트가 포함되어 있습니다.
- 평가는 학습 시와 동일한 방식으로 수행되며, GPU가 있으면 자동 사용됩니다.
- 비디오 평가는 2초 클립 단위 슬라이딩 윈도우 방식으로 진행되므로 시간이 오래 걸릴 수 있습니다 (최대 수 시간).
