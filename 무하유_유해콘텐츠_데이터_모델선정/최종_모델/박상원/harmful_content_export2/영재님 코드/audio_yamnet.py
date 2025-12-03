#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YAMNet 기반 오디오 유해도 분석
- 오디오를 16kHz mono float32로 변환
- YAMNet 클래스 중 HARMFUL_AUDIO_KEYS 에 해당하는 것만 골라서 harm_prob 계산
- 출력 JSON은 fusion_scores.py에서 읽기 쉽게 overall.harm_prob 로 기록
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # TF는 CPU 사용

import argparse
import json
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import tensorflow as tf
import tensorflow_hub as hub

TARGET_SR = 16000

_yamnet = None
_yamnet_labels = None

# 필요에 따라 확장
HARMFUL_AUDIO_KEYS = [
    "scream", "screaming",
    "gunshot", "gun", "rifle",
    "explosion", "bang",
    "fight", "fighting",
]


def load_yamnet():
    global _yamnet, _yamnet_labels
    if _yamnet is None:
        _yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        # 라벨 로딩 (class_map_path 제공됨)
        try:
            class_map_path = _yamnet.class_map_path().numpy().decode("utf-8")
            names = []
            import csv
            with tf.io.gfile.GFile(class_map_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    names.append(row["display_name"])
            _yamnet_labels = names
        except Exception:
            _yamnet_labels = None
    return _yamnet, _yamnet_labels


def to_mono_float32(wav: np.ndarray) -> np.ndarray:
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = wav.astype(np.float32, copy=False)
    max_abs = np.max(np.abs(wav)) + 1e-9
    if max_abs > 1.0:
        wav = wav / max_abs
    return wav


def resample_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return wav
    g = math.gcd(int(sr), TARGET_SR)
    up = TARGET_SR // g
    down = sr // g
    return resample_poly(wav, up, down)


def analyze_audio(path: str):
    try:
        data, sr = sf.read(path)
    except Exception as e:
        print(f"[YAMNet] read error: {e}")
        return {
            "model": "yamnet",
            "meta": {"path": path, "sr": 0, "duration_sec": 0.0},
            "overall": {"harm_prob": 0.0, "num_harmful_classes": 0},
            "ok": False,
        }

    data = to_mono_float32(data)
    duration = len(data) / float(sr) if sr > 0 else 0.0
    data = resample_16k(data, sr)

    yamnet, labels = load_yamnet()
    scores, _, _ = yamnet(data)  # (frames, 521)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()  # (521,)

    harmful_scores = []

    if labels is not None and len(labels) == mean_scores.shape[0]:
        for i, p in enumerate(mean_scores):
            name = labels[i].lower()
            if any(k in name for k in HARMFUL_AUDIO_KEYS):
                harmful_scores.append(float(p))

    harm_conf = float(np.max(harmful_scores)) if harmful_scores else 0.0
    harm_conf = max(0.0, min(1.0, harm_conf))

    return {
        "model": "yamnet",
        "meta": {
            "path": path,
            "sr": int(TARGET_SR),
            "duration_sec": float(duration),
        },
        "overall": {
            "harm_prob": harm_conf,
            "num_harmful_classes": int(len(harmful_scores)),
        },
        "ok": True,
    }


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="입력 오디오 파일 경로")
    ap.add_argument("--out", required=True, help="출력 JSON 경로")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse()
    res = analyze_audio(args.audio)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"✅ Audio saved -> {args.out} | harm_prob={res['overall']['harm_prob']:.3f}")
