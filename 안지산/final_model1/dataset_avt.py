# -*- coding: utf-8 -*-
import json, os
import torch
from torch.utils.data import Dataset
import numpy as np
import torchaudio
import soundfile as sf

# 비디오 로더: decord 사용(없으면 제로 텐서 fallback)
try:
    import decord
    decord.bridge.set_bridge('torch')
except Exception:
    decord = None


class AVTDataset(Dataset):
    """
    매니페스트(jsonl)의 각 라인:
    {"id":"abc","video_path":".../x.mp4","audio_path":".../x.wav","text_path":".../x.txt","label":1}
    """
    def __init__(self, manifest_path, cfg, tokenizer):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.items = [json.loads(line) for line in f]
        self.v_cfg = cfg['video']
        self.a_cfg = cfg['audio']
        self.t_cfg = cfg['text']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def _load_video(self, path):
        T = self.v_cfg['num_frames']
        size = self.v_cfg['size']
        sample_rate = self.v_cfg['sample_rate']

        if (decord is None) or (not os.path.exists(path)):
            return torch.zeros(T, 3, size, size)

        vr = decord.VideoReader(path)
        idxs = list(range(0, len(vr), sample_rate))
        # T개 미만이면 반복
        if len(idxs) < T:
            mul = (T + len(idxs) - 1) // len(idxs)
            idxs = (idxs * mul)[:T]
        else:
            idxs = idxs[:T]

        frames = vr.get_batch(idxs)  # (T, H, W, C), torch
        frames = frames.permute(0,3,1,2).float()/255.0
        frames = torch.nn.functional.interpolate(frames, size=(size,size),
                                                 mode='bilinear', align_corners=False)
        return frames  # (T,C,H,W)

    def _load_audio(self, path):
        sr = self.a_cfg['sample_rate']
        clip_sec = self.a_cfg['clip_seconds']
        N = sr * clip_sec

        if (path is None) or (not os.path.exists(path)):
            return torch.zeros(1, N)

        try:
            wav, file_sr = sf.read(path)  # numpy
            wav = torch.tensor(wav).float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # (1, L)
            if file_sr != sr:
                wav = torchaudio.functional.resample(wav, file_sr, sr)
        except Exception:
            return torch.zeros(1, N)

        if wav.size(1) >= N:
            wav = wav[:, :N]
        else:
            pad = N - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad))
        return wav  # (1, N)

    def _load_text(self, path):
        text = ""
        if path and os.path.exists(path):
            try:
                text = open(path, "r", encoding="utf-8").read().strip()
            except Exception:
                text = ""
        enc = self.tokenizer(
            text if len(text) > 0 else "",
            max_length=self.t_cfg['max_len'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

    def __getitem__(self, idx):
        it = self.items[idx]
        # 멀티모달 manifest 구조 대응
        vid_path = it['video'].get('clip_path', it['video'].get('src'))
        aud_path = it['audio'].get('path')
        txt_path = it['text'].get('path')

        vid = self._load_video(vid_path)
        aud = self._load_audio(aud_path)
        ids, mask = self._load_text(txt_path)
        y = torch.tensor(it['label'], dtype=torch.float32)

        return vid, aud, ids, mask, y
