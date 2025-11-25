

"""
멀티모달 유해도 결합 스코어링 (최적화 버전)

핵심 변경:
 - R3D → SlowFast 기반 violence score 지원
 - CLIP 과탐 방지 → weight 대폭 감소
 - YOLO + SlowFast 중심 구조
 - Audio/Text optional로 존재 안해도 0점 처리
 - 폭력 행동 class 자동 매핑 후 SlowFast score 산출
"""

import argparse
import json
import os
from typing import Any, Dict


def safe_load(path: str) -> Dict[str, Any]:
    """JSON 로드 실패 또는 파일 없음 → 빈 dict"""
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except:
        return {}


# ------------------------------------------
# CLIP score
# ------------------------------------------
def get_clip_score(d: Dict[str, Any]) -> float:
    overall = d.get("overall") or {}
    for key in ("p95_violence_prob", "avg_violence_prob", "max_violence_prob"):
        if key in overall:
            return float(overall[key])
    return 0.0

def get_vit_score(d: Dict[str, Any]) -> float:
    """
    ViT violence JSON도 CLIP과 같은 overall 구조를 사용:
    - avg_violence_prob / max_violence_prob / p95_violence_prob
    """
    overall = d.get("overall") or {}
    for key in ("p95_violence_prob", "avg_violence_prob", "max_violence_prob"):
        if key in overall:
            return float(overall[key])
    return 0.0

# ------------------------------------------
# YOLO (weapon/blood/nudity)
# ------------------------------------------
def get_yolo_score(d: Dict[str, Any]) -> float:
    overall = d.get("overall") or {}
    keys = ("weapon", "blood", "explicit", "nudity")

    vals = []
    for k in keys:
        v = overall.get(k)
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, dict):
            if "prob" in v:
                vals.append(float(v["prob"]))
            elif "score" in v:
                vals.append(float(v["score"]))

    return max(vals) if vals else 0.0


# ------------------------------------------
# SlowFast violence score
# ------------------------------------------
VIOLENCE_CLASSES = {
    # AVA class index 예시 (실제 모델 JSON에 맞게 수정)
    "fighting",
    "punching",
    "hitting",
    "kicking",
    "shooting",
    "brandishing_weapon",
    "attacking",
    "chasing",
    "stabbing",
    "strangling",
    "throwing",
}

def get_slowfast_score(d: Dict[str, Any]) -> float:
    """
    기대 구조:
    {
      "clips": [
        {
          "topk": [
            { "label": "fighting", "prob": 0.84 },
            ...
          ]
        }
      ]
    }
    """
    clips = d.get("clips") or []
    if not clips:
        return 0.0

    violence_scores = []

    for c in clips:
        topk = c.get("topk") or []
        v = 0.0
        for item in topk:
            label = item.get("label", "").lower()
            prob = float(item.get("prob", 0.0))
            if any(vclass in label for vclass in VIOLENCE_CLASSES):
                v = max(v, prob)
        violence_scores.append(v)

    if not violence_scores:
        return 0.0

    return max(violence_scores)  # max가 가장 안정적


# ------------------------------------------
# Audio (optional)
# ------------------------------------------
def get_audio_score(d: Dict[str, Any]) -> float:
    """
    지원 형태:
    - {"overall": {"harm_prob": x}}
    - {"overall": {"violence_prob": x}}
    - {"harm_conf": x}  (구 버전 호환)
    """
    overall = d.get("overall") or {}
    if "harm_prob" in overall:
        return float(overall["harm_prob"])
    if "violence_prob" in overall:
        return float(overall["violence_prob"])
    if "harm_conf" in d:
        return float(d["harm_conf"])
    return 0.0


# ------------------------------------------
# Text (optional)
# ------------------------------------------
def get_text_score(d: Dict[str, Any]) -> float:
    overall = d.get("overall") or d
    keys = ("hate_prob", "sexual_prob", "toxic_prob", "violence_prob")
    vals = [overall.get(k) for k in keys if isinstance(overall.get(k), (int, float))]
    return max(vals) if vals else 0.0


# ======================================================
# Main Fusion Logic
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    # 🔻 더 이상 vision(yolo) 안 씀
    # parser.add_argument("--vision", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--vit", required=True)
    parser.add_argument("--slowfast", required=True)  # 여기에는 slowfast json이 들어올 것
    parser.add_argument("--out", required=False)
    args = parser.parse_args()

    # vision_json = safe_load(args.vision)  # 🔻 제거
    audio_json  = safe_load(args.audio)
    text_json   = safe_load(args.text)
    clip_json   = safe_load(args.clip)
    vit_json    = safe_load(args.vit)
    slow_json   = safe_load(args.slowfast)  # r3d 대신 slowfast json

    # 모달 점수
    clip_score  = get_clip_score(clip_json)
    vit_score   = get_vit_score(vit_json)
    # yolo_score  = get_yolo_score(vision_json)  # 🔻 이제 안 씀
    slow_score  = get_slowfast_score(slow_json)
    audio_score = get_audio_score(audio_json)
    text_score  = get_text_score(text_json)

    # 🔻 폭력 스트림: 지금은 SlowFast만 사용
    violence_stream = slow_score

    # 최적 튜닝된 가중치
    fusion_weights = {
        "clip": 0.15,         # clip 과탐 ↓
        "vit": 0.6, 
        "violence": 0.25,     # 지금은 slowfast 단독
        "audio": 0.0,
        "text": 0.0,
    }

    final = (
        fusion_weights["clip"]     * clip_score +
        fusion_weights["vit"]      * vit_score +
        fusion_weights["violence"] * violence_stream +
        fusion_weights["audio"]    * audio_score +
        fusion_weights["text"]     * text_score
    )

    thresholds = {
        "allow": 0.30,
        "review": 0.55,
        "block": 0.78,
    }

    if final >= thresholds["block"]:
        decision = "BLOCK"
    elif final >= thresholds["review"]:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    result = {
        "scores": {
            "clip": clip_score,
            "vit": vit_score,
            # "yolo": yolo_score,  # 🔻 지금은 사용 안 하므로 제거하거나 0.0 고정
            "slowfast": slow_score,
            "audio": audio_score,
            "text": text_score,
            "violence_stream": violence_stream,
            "final": final
        },
        "fusion_weights": fusion_weights,
        "thresholds": thresholds,
        "decision": decision
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.out:
        json.dump(result, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
