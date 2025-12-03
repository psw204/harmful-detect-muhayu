# -*- coding: utf-8 -*-
"""
무하유 유해 콘텐츠 탐지 - 파일럿 테스트 버전
이미지: YOLOv8 + CLIP + OCR + 가중 평균
비디오: YOLOv8 + VideoMAE + CLIP + 가중 평균

Original file is located at
    https://colab.research.google.com/drive/1ehCJrDV1ETtn-B_gXJy4fdNdNr_5saWz

# **유해 콘텐츠 탐지 파일럿 테스트**

**설명**  
- 이미지 및 비디오에서 잠재적인 유해 콘텐츠를 탐지하기 위한 파일럿 테스트용 구현입니다.  
- 운영 환경용 모델이 아니며, **모델 구조, 결합 방식, 점수화 로직을 실험**하고 데이터를 확보하기 위한 목적입니다.  
- 실제 서비스 적용 시에는 Fine-tuning, 데이터 증강, 모델 최적화가 필요합니다.

작성자: 박상원
작성일: 2025년 2학기

---

**지원 파일 형식**  
- 이미지: `.jpg`, `.jpeg`, `.png`, `.bmp`  
- 비디오: `.mp4`, `.avi`, `.mov`, `.mkv`  

---

**모델 구성 (파일럿용)**  

#### 1) 이미지
- **YOLOv8**: 유해 객체 탐지 (칼, 총, 무기, 술, 담배 등)
- **CLIP**: 이미지-텍스트 맥락 분석 (폭력적 상황, 위협적 제스처 등)
- **EasyOCR**: 이미지 내 텍스트 추출
- **Toxic-BERT**: 추출된 텍스트 독성 분석
- **MLP 대체**: YOLO + CLIP + OCR 점수를 단순 가중 평균으로 통합 (0.4 + 0.4 + 0.2)

#### 2) 비디오
- **YOLOv8**: 대표 프레임 유해 객체 탐지
- **VideoMAE**: 행동 인식 (I3D/SlowFast 대체, 파일럿용 경량 모델)
  - Kinetics 데이터셋으로 사전 학습된 모델 사용
  - 폭력, 공격, 싸움 등 유해 행동 감지
- **CLIP**: 대표 프레임 맥락 분석
- **Transformer 대체**: 객체, 행동, 맥락 점수를 단순 가중 평균으로 통합 (0.3 + 0.4 + 0.3)

---

**주요 기능**
- **유해 객체 탐지**: 칼, 총, 무기, 술, 담배 등 12종 탐지
- **행동 분석**: 폭력적 행위, 공격, 신체적 폭력 등 9종 행동 인식
- **텍스트 분석**: OCR 기반 텍스트 추출 + 독성 키워드 및 BERT 기반 독성 평가  
- **최종 점수화 및 판정**: 단순 가중 평균 방식으로 유해 점수 산출 (임계값 0.25)
- **출력**: 탐지 결과, 각 점수, 최종 판정(유해/안전) 및 확신도 표시  

---

**참고 사항**
- pretrained 모델 활용 → 적은 데이터로도 바로 테스트 가능  
- **임계값 및 가중치**: 파일럿용 임의 설정 (실험 기반 조정)
  - 유해 판정 임계값: 0.25 (이미지/비디오 공통)
  - 객체 탐지 신뢰도: 0.2 이상
  - 맥락 분석 신뢰도: 0.2 이상
  - 텍스트 독성 점수: 0.3 이상
- 실제 환경에서는 학습 기반 최적화 필요 (정확도 향상, 경량화, XAI 등)
- Colab 환경에 최적화되어 있음 (Google Drive 연동, 파일 업로드 인터페이스 포함)
"""

# ============================================================
# 필수 라이브러리 설치 안내
# ============================================================
# 로컬 환경에서 실행하려면 먼저 다음 라이브러리들을 설치해야 합니다:
# pip install ultralytics torch torchvision transformers pillow opencv-python
# pip install easyocr matplotlib numpy
#
# GPU 사용 시 CUDA 버전에 맞는 PyTorch 설치:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ============================================================
# 필수 라이브러리 Import
# ============================================================
import cv2                    # 비디오 처리 및 프레임 추출
import torch                  # PyTorch 딥러닝 프레임워크
import torch.nn as nn         # 신경망 모듈 (사용하지 않지만 호환성 유지)
import numpy as np            # 수치 연산
from PIL import Image         # 이미지 파일 로드 및 저장
import easyocr                # OCR 텍스트 추출 (다국어 지원)
from transformers import CLIPProcessor, CLIPModel, pipeline, VideoMAEForVideoClassification, VideoMAEImageProcessor
from ultralytics import YOLO  # YOLOv8 객체 탐지
import matplotlib.pyplot as plt  # 시각화 (현재 미사용)
from pathlib import Path      # 파일 경로 처리


# ============================================================
# 파일럿 테스트 메인 클래스
# ============================================================
class HarmfulContentPilotTest:
    """
    유해 콘텐츠 탐지 파일럿 테스트 클래스
    
    사전 학습된 모델들을 조합하여 이미지와 비디오에서 유해 콘텐츠를 탐지합니다.
    학습이 필요 없는 즉시 테스트 가능한 파일럿 시스템입니다.
    
    특징:
    - 이미지: YOLO + CLIP + OCR + 가중 평균 (MLP 대체)
    - 비디오: YOLO + VideoMAE + CLIP + 가중 평균 (Transformer 대체)
    - 사전 학습 모델만 사용 (Fine-tuning 없음)
    - 간단한 가중치 기반 점수 통합
    """
    
    def __init__(self):
        """
        모델 초기화 및 설정
        
        로드되는 모델:
        1. YOLOv8 nano: 유해 객체 탐지 (경량 모델)
        2. CLIP ViT-B/32: 이미지-텍스트 맥락 이해
        3. VideoMAE: 비디오 행동 인식 (Kinetics 데이터셋 사전 학습)
        4. EasyOCR: 다국어 텍스트 추출 (영어, 한국어)
        5. Toxic-BERT: 텍스트 독성 분류
        """
        print("파일럿 테스트 모델 초기화 중...")

        # ============================================================
        # 1. YOLOv8 모델 로드 (유해 객체 탐지)
        # ============================================================
        # YOLOv8 nano 모델 (경량, 빠른 추론)
        # COCO 데이터셋으로 사전 학습됨 (80개 클래스)
        self.yolo = YOLO('yolov8n.pt')

        # ============================================================
        # 2. CLIP 모델 로드 (이미지-텍스트 맥락 분석)
        # ============================================================
        # OpenAI CLIP ViT-B/32 모델
        # 이미지와 텍스트의 유사도를 계산하여 맥락 이해
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # ============================================================
        # 3. VideoMAE 모델 로드 (행동 인식 - SlowFast 대체)
        # ============================================================
        # VideoMAE: 비디오 행동 인식 모델 (Kinetics 데이터셋 사전 학습)
        # SlowFast 대비 경량화된 파일럿용 모델
        self.video_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

        # ============================================================
        # 4. OCR 모델 로드 (이미지 내 텍스트 추출)
        # ============================================================
        # EasyOCR: 다국어 OCR 라이브러리
        # 영어(en)와 한국어(ko) 지원
        self.ocr_reader = easyocr.Reader(['en', 'ko'])

        # ============================================================
        # 5. 독성 텍스트 분류기 로드
        # ============================================================
        # Toxic-BERT: 텍스트 독성 분류 모델
        # GPU 사용 가능 시 GPU(device=0), 아니면 CPU(device=-1)
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=0 if torch.cuda.is_available() else -1
        )

        # ============================================================
        # 탐지 대상 정의
        # ============================================================
        # 유해 객체 클래스 (12종)
        # YOLO가 탐지한 객체 중 이 목록에 포함된 것만 유해로 판단
        self.harmful_objects = [
            'knife', 'gun', 'pistol', 'rifle', 'weapon',
            'cigarette', 'blood', 'sword', 'axe'
        ]

        # CLIP용 유해 맥락 텍스트 프롬프트 (10종)
        # 이미지와 각 텍스트의 유사도를 계산하여 가장 유사한 맥락 선택
        self.harmful_contexts = [
            "person holding a knife", "weapon being used for violence",
            "threatening gesture", "aggressive fighting", "physical assault",
            "dangerous weapon", "violence and aggression", "harmful behavior",
            "person attacking someone", "hitting and punching"
        ]

        # 행동 인식을 위한 유해 행동 클래스 (9종)
        # VideoMAE가 인식한 행동이 이 목록에 포함되면 유해로 판단
        self.harmful_actions = [
            "fighting", "punching", "kicking", "hitting", "stabbing",
            "shooting", "attacking", "assault", "violence"
        ]

        print("모든 모델 초기화 완료")

    # ============================================================
    # 이미지/비디오 분석 함수들
    # ============================================================
    
    def detect_harmful_objects(self, image_path):
        """
        YOLO 기반 유해 객체 탐지
        
        이미지에서 유해한 객체(칼, 총, 술, 담배 등)를 탐지합니다.
        YOLO가 탐지한 모든 객체 중 유해 객체 목록에 포함된 것만 반환합니다.
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            list: 탐지된 유해 객체 정보 리스트
                각 항목은 다음 정보를 포함:
                - object: 객체 클래스명
                - confidence: 탐지 신뢰도 (0~1)
                - bbox: 바운딩 박스 좌표 [x1, y1, x2, y2]
        """
        # YOLO 모델로 객체 탐지 실행
        results = self.yolo(image_path)

        harmful_detections = []  # 유해 객체 리스트
        
        # 탐지 결과 순회
        for result in results:
            boxes = result.boxes  # 탐지된 모든 바운딩 박스
            if boxes is not None:
                for box in boxes:
                    # 객체 정보 추출
                    class_id = int(box.cls)  # 클래스 ID
                    class_name = result.names[class_id].lower()  # 클래스명 (소문자 변환)
                    confidence = float(box.conf)  # 신뢰도

                    # 유해 객체 목록에 포함되고 신뢰도가 0.2 이상인 경우만 저장
                    # 신뢰도 임계값 0.2: 파일럿 테스트용 낮은 임계값 (민감도 우선)
                    if any(harmful in class_name for harmful in self.harmful_objects) and confidence > 0.2:
                        harmful_detections.append({
                            'object': class_name,
                            'confidence': confidence,
                            'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2] 형식
                        })

        return harmful_detections

    def analyze_context_with_clip(self, image_path):
        """
        CLIP 기반 이미지 맥락 분석
        
        이미지와 유해 맥락 텍스트들의 유사도를 계산하여
        가장 유사한 맥락을 선택합니다.
        
        동작 방식:
        1. 이미지와 10가지 유해 맥락 텍스트를 CLIP으로 인코딩
        2. 이미지-텍스트 유사도 계산 (코사인 유사도)
        3. 가장 높은 유사도를 가진 맥락 선택
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            dict: 맥락 분석 결과
                - harmful_context: 가장 유사한 유해 맥락 텍스트
                - confidence: 유사도 (0~1, softmax 확률)
                - is_harmful: 유해 여부 (confidence > 0.2)
        """
        # 이미지 로드
        image = Image.open(image_path)

        # CLIP 전처리: 이미지와 텍스트를 모델 입력 형식으로 변환
        inputs = self.clip_processor(
            text=self.harmful_contexts,      # 10가지 유해 맥락 텍스트
            images=image,                     # 입력 이미지
            return_tensors="pt",              # PyTorch 텐서로 반환
            padding=True                      # 텍스트 길이 맞춤
        )

        # CLIP 모델로 이미지-텍스트 유사도 계산
        with torch.no_grad():  # 그래디언트 계산 불필요 (추론만)
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # 이미지-텍스트 유사도 점수
            probs = logits_per_image.softmax(dim=1)      # 확률로 변환 (합이 1)

        # 가장 높은 확률을 가진 맥락 선택
        max_prob_idx = torch.argmax(probs[0])  # 최대 확률 인덱스
        max_context = self.harmful_contexts[max_prob_idx]  # 해당 맥락 텍스트
        max_prob = float(probs[0][max_prob_idx])           # 확률 값

        return {
            'harmful_context': max_context,
            'confidence': max_prob,
            'is_harmful': max_prob > 0.2  # 임계값 0.2 (파일럿 테스트용)
        }

    def extract_and_analyze_text(self, image_path):
        """
        OCR 텍스트 추출 및 독성 분석
        
        이미지에서 텍스트를 추출하고 유해한 내용이 포함되어 있는지 분석합니다.
        
        분석 단계:
        1. EasyOCR로 이미지에서 텍스트 추출 (영어, 한국어)
        2. 독성 키워드 매칭 (8가지 욕설/폭력적 단어)
        3. Toxic-BERT로 텍스트 전체의 독성 점수 계산
        4. 키워드와 BERT 점수 중 높은 값 선택
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            dict: 텍스트 분석 결과
                - text: 추출된 텍스트
                - toxicity_score: 독성 점수 (0~1)
                - is_toxic: 독성 여부 (score > 0.3)
        """
        # EasyOCR로 텍스트 추출
        # 반환값: [(bbox, text, confidence), ...]
        ocr_results = self.ocr_reader.readtext(image_path)

        # 텍스트가 없는 경우
        if not ocr_results:
            return {'text': '', 'toxicity_score': 0.0, 'is_toxic': False}

        # 추출된 텍스트들을 하나의 문자열로 결합
        extracted_text = ' '.join([result[1] for result in ocr_results])

        # ============================================================
        # 1단계: 독성 키워드 매칭
        # ============================================================
        # 8가지 욕설/폭력적 단어 목록
        toxic_keywords = ['fuck', 'shit', 'damn', 'hate', 'kill', 'die', 'stupid', 'idiot']
        # 키워드가 포함되어 있는지 확인 (대소문자 무시)
        keyword_found = any(keyword.lower() in extracted_text.lower() for keyword in toxic_keywords)

        # ============================================================
        # 2단계: Toxic-BERT로 전체 텍스트 독성 분석
        # ============================================================
        try:
            # Toxic-BERT 모델로 텍스트 분류
            toxicity_result = self.toxicity_classifier(extracted_text)
            
            # 결과 파싱 (모델에 따라 형식이 다를 수 있음)
            if isinstance(toxicity_result, list):
                # 리스트 형식: [{'label': 'TOXIC', 'score': 0.9}, ...]
                toxic_score = next((item['score'] for item in toxicity_result if item['label'] == 'TOXIC'), 0.0)
            else:
                # 딕셔너리 형식: {'label': 'TOXIC', 'score': 0.9}
                toxic_score = toxicity_result['score'] if toxicity_result['label'] == 'TOXIC' else 0.0
        except:
            # 모델 오류 시 점수 0으로 설정
            toxic_score = 0.0

        # ============================================================
        # 3단계: 최종 점수 결정
        # ============================================================
        # 키워드 발견 시 0.7점, BERT 점수와 비교하여 높은 값 선택
        final_score = max(toxic_score, 0.7 if keyword_found else 0.0)

        return {
            'text': extracted_text,
            'toxicity_score': final_score,
            'is_toxic': final_score > 0.3  # 임계값 0.3
        }

    def extract_video_frames(self, video_path, num_frames=16):
        """
        비디오에서 균등하게 프레임 추출
        
        비디오 전체 길이에서 균등한 간격으로 프레임을 샘플링합니다.
        VideoMAE 모델의 입력으로 사용됩니다.
        
        Args:
            video_path (str): 비디오 파일 경로
            num_frames (int): 추출할 프레임 수 (기본값: 16)
                - VideoMAE는 16프레임 입력 권장
                
        Returns:
            list: RGB 프레임 리스트 (numpy array)
                각 프레임은 (H, W, 3) 형태
        """
        # 비디오 파일 열기
        cap = cv2.VideoCapture(video_path)
        
        # 비디오 메타데이터 추출
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수
        fps = int(cap.get(cv2.CAP_PROP_FPS))                   # 초당 프레임 수

        # ============================================================
        # 프레임 인덱스 계산 (균등 샘플링)
        # ============================================================
        if total_frames < num_frames:
            # 프레임 수가 부족한 경우: 모든 프레임 사용
            frame_indices = list(range(total_frames))
        else:
            # 전체 비디오에서 균등하게 num_frames개 선택
            # 예: 100 프레임 → [0, 6, 12, 18, ..., 94, 99]
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        # ============================================================
        # 프레임 추출
        # ============================================================
        frames = []
        for idx in frame_indices:
            # 특정 프레임으로 이동
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()  # 프레임 읽기
            
            if ret:
                # BGR(OpenCV 기본)을 RGB로 변환
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()  # 비디오 파일 닫기
        return frames

    def analyze_video_action(self, video_path):
        """
        VideoMAE를 사용한 비디오 행동 인식
        
        비디오에서 수행되는 행동을 인식하고 유해한 행동인지 판단합니다.
        
        동작 방식:
        1. 비디오에서 16프레임 추출 (균등 샘플링)
        2. VideoMAE 모델로 행동 분류
        3. 분류된 행동이 유해 행동 목록에 포함되는지 확인
        
        Args:
            video_path (str): 비디오 파일 경로
            
        Returns:
            dict: 행동 분석 결과
                - action: 인식된 행동 레이블 (예: "fighting", "walking")
                - confidence: 분류 신뢰도 (0~1)
                - is_harmful: 유해 행동 여부
        """
        # 비디오에서 프레임 추출 (16프레임 권장)
        frames = self.extract_video_frames(video_path)

        # 프레임 추출 실패 시
        if len(frames) == 0:
            return {'action': 'unknown', 'confidence': 0.0, 'is_harmful': False}

        # ============================================================
        # VideoMAE 입력 준비
        # ============================================================
        # 프레임 리스트를 모델 입력 형식으로 변환
        # 정규화, 리사이징, 텐서 변환 등 자동 처리
        inputs = self.video_processor(frames, return_tensors="pt")

        # ============================================================
        # VideoMAE로 행동 분류
        # ============================================================
        with torch.no_grad():  # 그래디언트 계산 불필요
            outputs = self.video_model(**inputs)
            logits = outputs.logits  # 분류 로짓 (클래스별 점수)
            
            # 가장 높은 점수를 가진 클래스 선택
            predicted_class = logits.argmax(-1).item()  # 클래스 ID
            confidence = torch.softmax(logits, dim=-1).max().item()  # 확률로 변환 후 최댓값

        # 클래스 ID를 행동 레이블로 변환
        # 예: 0 → "abseiling", 123 → "punching"
        predicted_label = self.video_model.config.id2label[predicted_class]

        # ============================================================
        # 유해 행동 판단
        # ============================================================
        # 인식된 행동이 유해 행동 목록에 포함되는지 확인
        # 예: "fighting", "punching" 등
        is_harmful_action = any(harmful in predicted_label.lower() for harmful in self.harmful_actions)

        return {
            'action': predicted_label,
            'confidence': confidence,
            'is_harmful': is_harmful_action
        }

    # ============================================================
    # 이미지/비디오 테스트 함수
    # ============================================================
    
    def test_image(self, image_path):
        """
        이미지 파일럿 테스트: YOLOv8 + CLIP + OCR + 가중 평균
        
        3가지 분석을 수행하고 가중 평균으로 최종 점수를 계산합니다.
        
        분석 파이프라인:
        1. YOLO로 유해 객체 탐지 (칼, 총, 술 등)
        2. CLIP으로 이미지 맥락 분석 (폭력 상황 등)
        3. OCR + Toxic-BERT로 텍스트 독성 분석
        4. 가중 평균으로 최종 점수 계산 (0.4 + 0.4 + 0.2)
        5. 임계값(0.25) 기반 유해/안전 판정
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            dict: 이미지 분석 결과
                - type: 'image'
                - path: 파일 경로
                - harmful_objects: 탐지된 유해 객체 리스트
                - context_analysis: CLIP 맥락 분석 결과
                - text_analysis: OCR 텍스트 분석 결과
                - scores: 각 분석의 점수
                - final_harmful_score: 최종 유해 점수 (0~1)
                - is_harmful_content: 유해 여부 (True/False)
        """
        print(f"\n[이미지 테스트] {image_path}")

        # ============================================================
        # 1단계: YOLO 유해 객체 탐지
        # ============================================================
        print("1. YOLO 유해 객체 탐지")
        objects = self.detect_harmful_objects(image_path)

        # ============================================================
        # 2단계: CLIP 맥락 분석
        # ============================================================
        print("2. CLIP 맥락 분석")
        context = self.analyze_context_with_clip(image_path)

        # ============================================================
        # 3단계: OCR 텍스트 분석
        # ============================================================
        print("3. OCR 텍스트 분석")
        text_analysis = self.extract_and_analyze_text(image_path)

        # ============================================================
        # 점수 계산
        # ============================================================
        
        # 객체 점수 계산
        # - 객체가 없으면 0점
        # - 있으면: 최고 신뢰도 + 추가 객체당 0.1점 (최대 1.0)
        object_score = 0.0
        if objects:
            max_confidence = max(obj['confidence'] for obj in objects)
            object_score = min(max_confidence + (len(objects) - 1) * 0.1, 1.0)

        # 맥락 점수: 유해 맥락으로 판정된 경우만 점수 부여
        context_score = context['confidence'] if context['is_harmful'] else 0.0
        
        # 텍스트 점수: OCR + Toxic-BERT 독성 점수
        text_score = text_analysis['toxicity_score']

        # ============================================================
        # MLP 역할: 가중 평균 (단순 MLP 대체)
        # ============================================================
        # 가중치: 객체 40%, 맥락 40%, 텍스트 20%
        # 텍스트 비중이 낮은 이유: OCR 정확도 한계, 텍스트 없는 이미지 많음
        final_score = (object_score * 0.4) + (context_score * 0.4) + (text_score * 0.2)
        
        # 유해 판정: 최종 점수가 0.25 이상이면 유해
        # 임계값 0.25: 파일럿 테스트용 낮은 값 (민감도 우선)
        is_harmful = final_score > 0.25

        return {
            'type': 'image',
            'path': image_path,
            'harmful_objects': objects,
            'context_analysis': context,
            'text_analysis': text_analysis,
            'scores': {
                'object_score': object_score,
                'context_score': context_score,
                'text_score': text_score
            },
            'final_harmful_score': final_score,
            'is_harmful_content': is_harmful
        }

    def test_video(self, video_path):
        """
        비디오 파일럿 테스트: YOLOv8 + VideoMAE + CLIP + 가중 평균
        
        3가지 분석을 수행하고 가중 평균으로 최종 점수를 계산합니다.
        
        분석 파이프라인:
        1. 프레임 추출 (8프레임, 대표 프레임 분석용)
        2. YOLO로 대표 프레임의 유해 객체 탐지
        3. VideoMAE로 전체 비디오의 행동 인식 (16프레임)
        4. CLIP으로 대표 프레임의 맥락 분석
        5. 가중 평균으로 최종 점수 계산 (0.3 + 0.4 + 0.3)
        6. 임계값(0.25) 기반 유해/안전 판정
        
        Args:
            video_path (str): 비디오 파일 경로
            
        Returns:
            dict: 비디오 분석 결과
                - type: 'video'
                - path: 파일 경로
                - harmful_objects: 탐지된 유해 객체 리스트
                - action_analysis: VideoMAE 행동 분석 결과
                - context_analysis: CLIP 맥락 분석 결과
                - scores: 각 분석의 점수
                - final_harmful_score: 최종 유해 점수 (0~1)
                - is_harmful_content: 유해 여부 (True/False)
        """
        print(f"\n[비디오 테스트] {video_path}")

        # ============================================================
        # 1단계: 프레임 추출 및 YOLO 객체 탐지
        # ============================================================
        print("1. 프레임별 YOLO 객체 탐지")
        # 8프레임 추출 (대표 프레임 분석용, 경량화)
        frames = self.extract_video_frames(video_path, num_frames=8)

        # 대표 프레임 저장 (임시 파일)
        # YOLO와 CLIP은 이미지 파일 경로를 입력으로 받으므로 임시 저장 필요
        temp_frame_path = "temp_frame.jpg"
        if frames:
            # 첫 번째 프레임을 대표 프레임으로 사용
            Image.fromarray(frames[0]).save(temp_frame_path)
            objects = self.detect_harmful_objects(temp_frame_path)
        else:
            objects = []

        # ============================================================
        # 2단계: VideoMAE 행동 인식
        # ============================================================
        print("2. VideoMAE 행동 인식")
        # VideoMAE는 16프레임을 사용하여 전체 비디오의 행동 인식
        action_analysis = self.analyze_video_action(video_path)

        # ============================================================
        # 3단계: CLIP 맥락 분석 (대표 프레임 사용)
        # ============================================================
        print("3. CLIP 맥락 분석")
        if frames:
            # 대표 프레임의 맥락 분석
            context = self.analyze_context_with_clip(temp_frame_path)
        else:
            # 프레임 추출 실패 시 기본값
            context = {'harmful_context': 'unknown', 'confidence': 0.0, 'is_harmful': False}

        # ============================================================
        # 점수 계산
        # ============================================================
        
        # 객체 점수 계산 (이미지와 동일한 방식)
        object_score = 0.0
        if objects:
            max_confidence = max(obj['confidence'] for obj in objects)
            object_score = min(max_confidence + (len(objects) - 1) * 0.1, 1.0)

        # 행동 점수: 유해 행동으로 판정된 경우만 점수 부여
        action_score = action_analysis['confidence'] if action_analysis['is_harmful'] else 0.0
        
        # 맥락 점수: 유해 맥락으로 판정된 경우만 점수 부여
        context_score = context['confidence'] if context['is_harmful'] else 0.0

        # ============================================================
        # Transformer 역할: 가중 평균 (단순 Transformer 대체)
        # ============================================================
        # 가중치: 객체 30%, 행동 40%, 맥락 30%
        # 행동 비중이 가장 높은 이유: 비디오의 핵심은 행동 분석
        final_score = (object_score * 0.3) + (action_score * 0.4) + (context_score * 0.3)
        
        # 유해 판정: 최종 점수가 0.25 이상이면 유해
        is_harmful = final_score > 0.25

        return {
            'type': 'video',
            'path': video_path,
            'harmful_objects': objects,
            'action_analysis': action_analysis,
            'context_analysis': context,
            'scores': {
                'object_score': object_score,
                'action_score': action_score,
                'context_score': context_score
            },
            'final_harmful_score': final_score,
            'is_harmful_content': is_harmful
        }

    # ============================================================
    # 결과 출력 함수
    # ============================================================
    
    def print_results(self, results):
        """
        파일럿 테스트 결과를 콘솔에 출력
        
        분석 결과를 사람이 읽기 쉬운 형식으로 포맷하여 출력합니다.
        
        Args:
            results (dict): test_image() 또는 test_video()의 반환값
                - type: 'image' 또는 'video'
                - 각 분석 결과 (객체, 맥락, 텍스트/행동)
                - 점수 및 최종 판정
        """
        print("\n" + "="*60)
        print("파일럿 테스트 결과")
        print("="*60)

        if results['type'] == 'image':
            # ========================================================
            # 이미지 결과 출력
            # ========================================================
            
            # 1. YOLO 객체 탐지 결과
            print(f"유해 객체 탐지: {len(results['harmful_objects'])}개 발견")
            for obj in results['harmful_objects']:
                print(f"   - {obj['object']}: {obj['confidence']:.3f}")

            # 2. CLIP 맥락 분석 결과
            context = results['context_analysis']
            print(f"맥락 분석: {context['harmful_context']}")
            print(f"   신뢰도: {context['confidence']:.3f} ({'유해' if context['is_harmful'] else '안전'})")

            # 3. OCR 텍스트 분석 결과
            text = results['text_analysis']
            if text['text']:
                print(f"추출된 텍스트: '{text['text']}'")
                print(f"   독성 점수: {text['toxicity_score']:.3f} ({'유해' if text['is_toxic'] else '안전'})")
            else:
                print("추출된 텍스트: 없음")

            # 4. 점수 분석 (각 모델별 점수)
            scores = results['scores']
            print(f"\n점수 분석:")
            print(f"   객체 점수: {scores['object_score']:.3f}")
            print(f"   맥락 점수: {scores['context_score']:.3f}")
            print(f"   텍스트 점수: {scores['text_score']:.3f}")

        else:  # video
            # ========================================================
            # 비디오 결과 출력
            # ========================================================
            
            # 1. YOLO 객체 탐지 결과 (대표 프레임)
            print(f"유해 객체 탐지: {len(results['harmful_objects'])}개 발견")
            for obj in results['harmful_objects']:
                print(f"   - {obj['object']}: {obj['confidence']:.3f}")

            # 2. VideoMAE 행동 인식 결과
            action = results['action_analysis']
            print(f"행동 인식: {action['action']}")
            print(f"   신뢰도: {action['confidence']:.3f} ({'유해' if action['is_harmful'] else '안전'})")

            # 3. CLIP 맥락 분석 결과 (대표 프레임)
            context = results['context_analysis']
            print(f"맥락 분석: {context['harmful_context']}")
            print(f"   신뢰도: {context['confidence']:.3f} ({'유해' if context['is_harmful'] else '안전'})")

            # 4. 점수 분석 (각 모델별 점수)
            scores = results['scores']
            print(f"\n점수 분석:")
            print(f"   객체 점수: {scores['object_score']:.3f}")
            print(f"   행동 점수: {scores['action_score']:.3f}")
            print(f"   맥락 점수: {scores['context_score']:.3f}")

        # ============================================================
        # 최종 결과 출력 (이미지/비디오 공통)
        # ============================================================
        print(f"\n최종 유해 점수: {results['final_harmful_score']:.3f}")
        
        # 판정 결과
        judgment = "유해 콘텐츠" if results['is_harmful_content'] else "안전 콘텐츠"
        
        # 확신도 레벨
        # - 높음: 0.5 초과 (강한 확신)
        # - 보통: 0.3~0.5 (중간 확신)
        # - 낮음: 0.3 미만 (약한 확신)
        confidence = "높음" if results['final_harmful_score'] > 0.5 else "보통" if results['final_harmful_score'] > 0.3 else "낮음"
        
        print(f"최종 판정: {judgment} (확신도: {confidence})")


# ============================================================
# 메인 실행 함수 (로컬 환경)
# ============================================================

def run_pilot_test():
    """
    파일럿 테스트 실행 함수 (로컬 환경)
    
    로컬 파일 시스템에서 이미지/비디오 파일을 분석합니다.
    
    실행 흐름:
    1. 명령줄 인자로 파일 경로 입력받기
    2. 파일들을 순회하며 분석
    3. 파일 확장자로 이미지/비디오 자동 판별
    4. 각 파일에 대해 유해 콘텐츠 분석 수행
    5. 결과를 콘솔에 출력
    
    지원 형식:
    - 이미지: .jpg, .jpeg, .png, .bmp
    - 비디오: .mp4, .avi, .mov, .mkv
    
    사용 예시:
    - python 파일럿_테스트.py image1.jpg video1.mp4
    - python 파일럿_테스트.py test_folder/*.jpg
    """
    import sys
    import os
    import glob

    # ============================================================
    # 명령줄 인자 처리
    # ============================================================
    if len(sys.argv) < 2:
        print("사용법: python 파일럿_테스트.py [파일경로1] [파일경로2] ...")
        print("\n예시:")
        print("  python 파일럿_테스트.py image.jpg")
        print("  python 파일럿_테스트.py image1.jpg video1.mp4")
        print("  python 파일럿_테스트.py test_data/*.jpg")
        print("\n지원 형식:")
        print("  - 이미지: .jpg, .jpeg, .png, .bmp")
        print("  - 비디오: .mp4, .avi, .mov, .mkv")
        sys.exit(1)
    
    # 파일 경로 수집 (glob 패턴 지원)
    test_files = []
    for arg in sys.argv[1:]:
        # glob 패턴인 경우 확장
        if '*' in arg or '?' in arg:
            test_files.extend(glob.glob(arg))
        else:
            # 일반 파일 경로
            if os.path.exists(arg):
                test_files.append(arg)
            else:
                print(f"경고: 파일을 찾을 수 없습니다 - {arg}")
    
    if not test_files:
        print("에러: 처리할 파일이 없습니다.")
        sys.exit(1)
    
    print(f"분석할 파일 ({len(test_files)}개):")
    for f in test_files:
        print(f"  - {f}")

    # ============================================================
    # 파일럿 테스트 객체 생성
    # ============================================================
    # 모든 모델을 초기화하고 로드
    # 시간이 다소 소요될 수 있음 (첫 실행 시 모델 다운로드)
    print("\n" + "="*60)
    pilot = HarmfulContentPilotTest()

    print("\n파일럿 테스트 시작")
    print("지원 형식: 이미지(jpg, png), 비디오(mp4, avi)")
    print("-" * 60)

    # ============================================================
    # 각 파일에 대해 테스트 실행
    # ============================================================
    for file_path in test_files:
        try:
            # 파일 확장자 추출 (소문자 변환)
            file_ext = Path(file_path).suffix.lower()

            # ========================================================
            # 이미지 또는 비디오 판별 및 분석 실행
            # ========================================================
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # 이미지 파일 분석
                results = pilot.test_image(file_path)
            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                # 비디오 파일 분석
                results = pilot.test_video(file_path)
            else:
                # 지원하지 않는 파일 형식
                print(f"지원하지 않는 파일 형식: {file_path}")
                continue

            # ========================================================
            # 결과 출력
            # ========================================================
            pilot.print_results(results)
            print("\n" + "-"*60 + "\n")

        except Exception as e:
            # 에러 발생 시 출력하고 다음 파일로 계속 진행
            print(f"에러 발생 ({file_path}): {e}")
            # 디버깅을 위해 traceback 출력
            import traceback
            traceback.print_exc()
    
    print("\n모든 파일 분석 완료!")


# ============================================================
# 스크립트 직접 실행 시 메인 함수 호출
# ============================================================
if __name__ == "__main__":
    run_pilot_test()