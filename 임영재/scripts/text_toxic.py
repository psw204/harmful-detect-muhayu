#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PaddleOCR + multilingual toxicity classifier 기반 텍스트 유해도 분석
- 이미지에서 텍스트 추출 → toxic 확률 계산
- 출력 JSON의 overall.toxic_prob 를 fusion_scores.py 에서 사용
"""

import argparse
import json

from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ⚠️ 기존: "unitary/multilingual-toxic-bert"  ->  공개 모델로 교체
MODEL_ID = "textdetox/bert-multilingual-toxicity-classifier"
# 또는
# MODEL_ID = "citizenlab/distilbert-base-multilingual-cased-toxicity"


def load_models():
    # lang는 환경에 따라 "korean", "korean+latin" 등으로 바꿀 수 있음
    ocr = PaddleOCR(use_angle_cls=True, lang="korean")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    clf = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    clf.eval()
    return ocr, tok, clf


def extract_text(ocr, img_path: str) -> str:
    res = ocr.ocr(img_path, cls=True)
    if not res or not res[0]:
        return ""
    lines = [line[1][0] for line in res[0] if line and line[1]]
    return " ".join(lines)


@torch.no_grad()
def analyze_text(tok, clf, text: str):
    """
    binary toxicity classifier 가정:
      - logits -> softmax -> [p_not_toxic, p_toxic]
      - toxic_prob = probs[0, 1]
    """
    if not text.strip():
        return {"toxic_prob": 0.0, "model": MODEL_ID}

    t = tok(text, return_tensors="pt", truncation=True, padding=True)
    out = clf(**t)
    logits = out.logits  # [1, num_labels]
    probs = torch.softmax(logits, dim=-1)

    if probs.shape[1] == 1:
        # 1-클래스 모델일 경우: sigmoid 로 해석 (거의 없겠지만 방어 코드)
        toxic_prob = float(torch.sigmoid(logits)[0, 0])
    else:
        # 일반적인 2-class: index 1 을 toxic 라고 가정
        toxic_prob = float(probs[0, -1])

    toxic_prob = max(0.0, min(1.0, toxic_prob))
    return {"toxic_prob": toxic_prob, "model": MODEL_ID}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="OCR을 수행할 이미지 경로")
    ap.add_argument("--out", required=True, help="출력 JSON 경로")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse()

    print("[TEXT] Loading OCR + Toxic model:", MODEL_ID)
    ocr, tok, clf = load_models()

    print(f"[TEXT] OCR on: {args.image}")
    text = extract_text(ocr, args.image)
    if text:
        print(f"[TEXT] Extracted text: {text[:80]}...")
    else:
        print("[TEXT] No text detected.")

    res_prob = analyze_text(tok, clf, text)

    result = {
        "model": res_prob["model"],
        "image": args.image,
        "text": text,
        "overall": {
            "toxic_prob": res_prob["toxic_prob"],
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Text saved -> {args.out} | toxic_prob={res_prob['toxic_prob']:.3f}")
