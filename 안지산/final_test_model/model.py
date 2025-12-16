# ================================
# final_model3 / model.py (복원본)
# ================================

from typing import Dict, Any
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel


# -------------------------
# Video Encoder
# -------------------------
class VideoEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()

        # torchvision old 버전 호환
        try:
            base = models.resnet18(pretrained=True)
        except:
            base = models.resnet18()

        base.fc = nn.Identity()  # remove classifier head
        self.backbone = base

        self.proj = nn.Linear(512, out_dim) if out_dim != 512 else nn.Identity()

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)              # (B*T, 512)
        feats = feats.view(b, t, 512)
        feats = feats.mean(dim=1)             # temporal avg → (B, 512)
        feats = self.proj(feats)
        return feats


# -------------------------
# Audio Encoder
# -------------------------
class AudioEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, audio):
        x = audio.unsqueeze(1)  # (B, 1, L)
        x = self.net(x)         # (B, 128, 1)
        x = x.squeeze(-1)       # (B, 128)
        x = self.fc(x)          # (B, out_dim)
        return x


# -------------------------
# Text Encoder
# -------------------------
class TextEncoder(nn.Module):
    def __init__(self, model_name="skt/kobert-base-v1", out_dim=768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.proj = nn.Linear(hidden_size, out_dim) if out_dim != hidden_size else nn.Identity()

    def forward(self, tokens):
        safe_tokens = {}

        if "input_ids" in tokens:
            safe_tokens["input_ids"] = tokens["input_ids"]

        if "attention_mask" in tokens:
            safe_tokens["attention_mask"] = tokens["attention_mask"]

        outputs = self.bert(**safe_tokens)
        cls = outputs.last_hidden_state[:, 0, :]  # CLS token
        cls = self.proj(cls)
        return cls


# -------------------------
# Multimodal Classifier
# -------------------------
class MultimodalClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.video_encoder = VideoEncoder(out_dim=512)
        self.audio_encoder = AudioEncoder(out_dim=256)
        self.text_encoder = TextEncoder(
            model_name=cfg["text"]["model_name"],
            out_dim=768
        )

        fusion_in = 512 + 256 + 768

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.cls_head = nn.Linear(128, 1)

    def forward(self, batch):
        video = batch["video"]
        audio = batch["audio"]
        text = batch["text"]

        v = self.video_encoder(video)
        a = self.audio_encoder(audio)
        t = self.text_encoder(text)

        fused = torch.cat([v, a, t], dim=1)
        fused = self.fusion(fused)
        logits = self.cls_head(fused).squeeze(-1)

        return logits
