# final_model1 — AVT 멀티모달 1차 베이스라인
**모델 조합:** MoViNet(비디오, 현재는 경량 3DConv 스텁) + YAMNet(오디오, 가중치 없으면 경량 Audio CNN) + KoBERT(텍스트) + Fusion + (옵션) CLIP-style InfoNCE

> 동영상만 있으면 됩니다. 학습 전 `scripts/prep_av_text.py`로
> 각 mp4에서 .wav(16kHz)와 텍스트(.txt; ASR 붙이면 내용 채움)를 만들고
> `../splits/train.jsonl`, `../splits/val.jsonl`에 `audio_path`, `text_path`가 채워지도록 하세요.

## 구조
final_model1/
├─ config.yaml
├─ dataset_avt.py
├─ model_avt.py
├─ losses.py
├─ train.py
├─ utils.py
├─ requirements.txt
├─ scripts/
│ └─ prep_av_text.py
└─ weights/
└─ yamnet.pt # (선택)


## 빠른 시작
```bash
# 1) 전처리 (mp4→wav(16kHz) + txt 생성/보강)
python scripts/prep_av_text.py --manifest ../splits/train.jsonl ../splits/val.jsonl

# 2) 설치
pip install -r requirements.txt

# 3) 학습
python train.py --cfg config.yaml

메모

weights/yamnet.pt가 없으면 간단한 Audio-CNN으로 대체되며 학습은 동작합니다.

KoBERT는 skt/kobert-base-v1 사용(오프라인이면 캐시 필요).

final_model2에서 MoViNet/YAMNet 실제 사전학습 가중치로 교체해 성능을 끌어올릴 예정입니다.