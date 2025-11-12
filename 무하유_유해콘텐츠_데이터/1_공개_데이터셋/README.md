# 공개 데이터셋 다운로드 가이드

> **작성자**: 박상원  
> **작성일**: 2025년 2학기

## 📥 데이터셋 목록

다음 공개 데이터셋을 다운로드하여 각 폴더에 배치하세요.

### 1. HOD Dataset
- **용량**: 약 10GB
- **내용**: 유해 이미지 10,631개
- **다운로드**: [GitHub](https://github.com/poori-nuna/HOD-Benchmark-Dataset)
- **배치 위치**: `1_공개_데이터셋/HOD_Dataset/`

### 2. RWF-2000
- **용량**: 약 50GB
- **내용**: 폭력/비폭력 비디오 2,000개
- **다운로드**: [Zenodo](https://zenodo.org/records/15687512)
- **배치 위치**: `1_공개_데이터셋/RWF-2000/archive/`

**폴더 구조**:
- `Fight/` - 유해 비디오 (1,000개)
- `NonFight/` - 안전 비디오 (1,000개)

### 3. RLVS (Real Life Violence Situations)
- **용량**: 약 30GB
- **내용**: 실생활 폭력 비디오 2,000개
- **다운로드**: [Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
- **배치 위치**: `1_공개_데이터셋/RLVS/archive/`

**폴더 구조**:
- `Violence/` - 유해 비디오 (1,000개)
- `NonViolence/` - 안전 비디오 (1,000개)

### 4. COCO Safe Dataset
- **용량**: 약 20GB (전체), 약 5GB (5,000개 선택)
- **내용**: 안전 이미지 5,000개
- **다운로드**: [COCO](http://images.cocodataset.org/zips/val2017.zip)
- **배치 위치**: `1_공개_데이터셋/COCO_Safe_Dataset/val2017/`

**추출**: 전체 다운로드 후 이미지 5,000개만 선택하여 복사

## 📂 최종 폴더 구조

```
1_공개_데이터셋/
├── HOD_Dataset/
│   └── dataset/                    # 이미지 파일들 (*.jpg)
├── RWF-2000/
│   └── archive/
│       └── train/
│           ├── Fight/               # 유해 비디오 (*.avi)
│           └── NonFight/           # 안전 비디오 (*.avi)
├── RLVS/
│   └── archive/
│       ├── Violence/               # 유해 비디오 (*.mp4)
│       └── NonViolence/            # 안전 비디오 (*.mp4)
└── COCO_Safe_Dataset/
    └── val2017/                    # 안전 이미지 (*.jpg)
```

## ⚠️ 주의사항

1. **용량**: 총 약 110GB (압축 해제 후), 디스크 공간 확인 필요
2. **다운로드 시간**: 네트워크 속도에 따라 수 시간 소요 가능
3. **권장**: 특정 데이터셋만 먼저 다운로드하여 테스트 진행 가능

## 📊 데이터 통계

| 데이터셋 | 타입 | 개수 | 용도 |
|---------|------|------|------|
| HOD Dataset | 이미지 | 10,631개 | 유해 이미지 학습/검증 |
| RWF-2000 | 비디오 | 2,000개 | 폭력 행동 인식 |
| RLVS | 비디오 | 2,000개 | 실생활 폭력 상황 |
| COCO Safe | 이미지 | 5,000개 | 안전 이미지 학습/검증 |
| **총합** | | **19,631개** | |
