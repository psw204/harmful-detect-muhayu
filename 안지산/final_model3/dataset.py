# final_model3/dataset.py
import json
import os
from typing import Dict, Any

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_video
import torchaudio
from transformers import AutoTokenizer


class MultiModalDataset(Dataset):
    def __init__(self, manifest_path: str, config: Dict[str, Any], split="train"):
        self.config = config
        self.split = split

        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"[{split}] Loaded {len(self.samples)} samples from {manifest_path}")

        size = config["video"]["size"]
        self.video_transform = T.Compose(
            [
                T.Resize((size, size)),
                T.ConvertImageDtype(torch.float32),
                T.Normalize([0.5]*3, [0.5]*3),
            ]
        )

        self.audio_sr = config["audio"]["sample_rate"]
        self.audio_sec = config["audio"]["clip_seconds"]
        self.audio_len = self.audio_sr * self.audio_sec

        self.tokenizer = AutoTokenizer.from_pretrained(config["text"]["model_name"])
        self.text_max_len = config["text"]["max_len"]

    def __len__(self):
        return len(self.samples)

    def _load_video(self, path):
        T_num = self.config["video"]["num_frames"]
        rate = self.config["video"]["sample_rate"]
        size = self.config["video"]["size"]

        if not os.path.exists(path):
            return torch.zeros(T_num, 3, size, size)

        try:
            video, _, _ = read_video(path, pts_unit="sec")
            if video.shape[0] == 0:
                raise RuntimeError("Empty")

            idx = list(range(0, video.shape[0], rate))
            if len(idx) < T_num:
                idx = (idx * (T_num // len(idx) + 1))[:T_num]
            else:
                idx = idx[:T_num]

            video = video[idx].permute(0, 3, 1, 2)
            video = torch.stack([self.video_transform(f) for f in video])
            return video

        except:
            return torch.zeros(T_num, 3, size, size)

    def _load_audio(self, path):
        if not path or not os.path.exists(path):
            return torch.zeros(self.audio_len)

        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.audio_sr:
                wav = torchaudio.transforms.Resample(sr, self.audio_sr)(wav)

            wav = wav.squeeze()
            if wav.shape[0] > self.audio_len:
                wav = wav[:self.audio_len]
            else:
                wav = torch.nn.functional.pad(wav, (0, self.audio_len - wav.shape[0]))

            return wav

        except:
            return torch.zeros(self.audio_len)

    def _encode_text(self, text):
        if text is None:
            text = ""

        enc = self.tokenizer(
            text,
            max_length=self.text_max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, idx):
        s = self.samples[idx]

        video = self._load_video(s["video_path"])
        audio = self._load_audio(s["audio_path"])
        text = self._encode_text(s.get("text", ""))

        label = torch.tensor(float(s["label"]), dtype=torch.float32)

        return {
            "video": video,
            "audio": audio,
            "text": text,
            "label": label,
        }
