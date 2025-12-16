Multimodal Harmful Content Detection (Prototype)

비디오/이미지 기반 유해 콘텐츠(폭력/무기/혐오 등)를 모달별 모델로 분석하고, 결과를 Fusion Score로 합산해 결정을 내리는 프로토타입 파이프라인입니다.

구성 요소:
🎞️ Video → Frames 분할
🧠 Vision(폭력): CLIP / ViT
🏃 Action(폭력 행동): SlowFast R101
🔊 Audio(비명/총성 등): YAMNet
🧾 Text(이미지 내 텍스트): PaddleOCR + Toxicity Classifier
🧮 Fusion Scoring: 모달 점수 결합 및 임계치 기반 판정

파일 구조:
scripts/
    video_split.py          # 비디오 클립/프레임 분할
    vision_clip_violence.py 
    vision_vit.py           
    video_slowfast.py
    audio_yamnet.py
    text_toxic.py
    fusion_scores.py
    kinetics_400_labels.txt
    vit_finetuned.pth 
팀원_라벨링_모델선정/
    결과_데이터_32/ # 32프레임 균등분할 - 최종 데이터
        {사람 이름}/
            라벨결과/ # label = 0 if category == "safe" else 1, pred_label == final_label
                safe_labels.json  
                safe_video_labels.json
                verified_labels.json
                verified_video_labels.json
            video_frames/
                비디오/
                    {비디오 이름}/
                        비디오프레임 모음/
                            **.jpg
                            ...
                        audio_result.json
                        clip_result.json
                        fusion_result.json # 모든 모델의 점수 및 가중치 내포
                        slowfast_result.json
                        text_result.json
                        vit_result.json
                안전비디오/
                    위와 동
    결과_데이터_training/ # 학습 데이터 결과물 32프레임 균등분할
    evaluate_training.py            # 학습 데이터 분석
    final_model_img_training.py
    final_model_video_training.py
    evaluate.py                     # 최종 결과 분석
    final_model_img.py              # 이미지 모델 clip + vit
    final_model_video.py            # 비디오 모델 clip + vit + slowfast + ( audio + text )


데이터 검사도구

A. 비디오 검사도구 final_model_video.py

    프레임/클립
        video_split.py --clip-sec 2 (2초 단위로 프레임 저장 - 의미 x)
        CLIP/VIT/SlowFast 모두 균등 32프레임 샘플링: TARGET = 32
        SlowFast 입력 길이: frames_per_clip = 32
        SlowFast 입력 리사이즈: Resize((224,224))
        SlowFast 클래스: Kinetics-400 (num_classes=400)

    모델 실행 파라미터
        CLIP
            batch_size = 16
            temperature = 2.0
            (함수 인자로 stride=10이 있지만, 현재 코드에서는 샘플링을 균등 32로 해버려서 stride는 사실상 의미 없음)
        ViT
            batch_size = 16
        SlowFast
            topk = 5 (top5 뽑음)
        오디오/텍스트
            오디오 추출: ffmpeg -ac 1 -ar 16000
            YAMNet 실행 환경
                CUDA_VISIBLE_DEVICES = -1 (TF는 CPU 강제)
                TF_ENABLE_ONEDNN_OPTS=0, TF_CPP_MIN_LOG_LEVEL=2
            텍스트(OCR+toxic)

    최종 판정
        가중치/임계값
            W_CLIP = 0.8
            W_VIT = 0.1
            W_SLOWFAST = 0.1
            W_AUDIO = 0 (데이터셋 내 영상에 소리와 자막이 없는 특성을 고려하여 판별 기준에서 제외)
            W_TEXT = 0
            threshold = 0.63
            fused = 0.8*clip + 0.1*vit + 0.1*slowfast
            pred_label = 1 if fused >= 0.63 else 0
            기존 라벨(label)
                *_labels_categorized.json에서 label = 0 if category == "safe" else 1

B. 이미지 검사도구 final_model_img.py

    가중치/임계값
        W_CLIP = 0.8
        W_VIT = 0.2
        threshold = 0.35
        fused = 0.8*clip + 0.2*vit
        pred_label = 1 if fused >= 0.35 else 0

    모델 실행
        CLIP: --batch 16 --stride 1
        ViT : --batch 16 --stride 1
