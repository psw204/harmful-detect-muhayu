# final_model3/model.py
from typing import Dict, Any

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel


class VideoEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        # ✅ Cheetah 서버(torchvision old) 호환: pretrained=True 방식 사용
        try:
            base = models.resnet18(pretrained=True)
        except TypeError:
            # 어떤 환경에서만 weights 인자를 쓰는 경우를 대비한 fallback
            base = models.resnet18()

        base.fc = nn.Identity()
        self.backbone = base
        self.out_dim = out_dim

        if out_dim != 512:
            self.proj = nn.Linear(512, out_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)      # (B*T, 512)
        feats = feats.view(b, t, 512) # (B, T, 512)
        feats = feats.mean(dim=1)     # temporal average → (B, 512)
        feats = self.proj(feats)      # (B, out_dim)
        return feats


class AudioEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        # 1D Conv 기반 간단 오디오 인코더
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

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: (B, L)
        """
        x = audio.unsqueeze(1)  # (B, 1, L)
        x = self.net(x)         # (B, 128, 1)
        x = x.squeeze(-1)       # (B, 128)
        x = self.fc(x)          # (B, out_dim)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "skt/kobert-base-v1", out_dim: int = 768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        if out_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, out_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        tokens: dict(input_ids, attention_mask, token_type_ids, ...)
        - token_type_ids에서 index out of range 나는 문제 방지 위해
          BERT에 넘길 때 input_ids, attention_mask만 사용.
        """
        safe_tokens = {}
        if "input_ids" in tokens:
            safe_tokens["input_ids"] = tokens["input_ids"]
        if "attention_mask" in tokens:
            safe_tokens["attention_mask"] = tokens["attention_mask"]

        outputs = self.bert(**safe_tokens)
        cls = outputs.last_hidden_state[:, 0, :]  # (B, H)
        cls = self.proj(cls)                      # (B, out_dim)
        return cls


class MultimodalClassifier(nn.Module):
    """
    Video + Audio + Text 멀티모달 이진 분류기
      - 입력: batch dict (video, audio, text)
      - 출력: logits (B,)
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.video_encoder = VideoEncoder(out_dim=512)
        self.audio_encoder = AudioEncoder(out_dim=256)
        self.text_encoder = TextEncoder(
            model_name=cfg["text"]["model_name"],
            out_dim=768,
        )

        fusion_in_dim = 512 + 256 + 768

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # 이진 분류 (harmful vs safe)
        self.cls_head = nn.Linear(128, 1)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        video = batch["video"]        # (B, T, C, H, W)
        audio = batch["audio"]        # (B, L)
        text_tokens = batch["text"]   # dict(input_ids, attention_mask, ...)

        v_feat = self.video_encoder(video)
        a_feat = self.audio_encoder(audio)
        t_feat = self.text_encoder(text_tokens)

        fused = torch.cat([v_feat, a_feat, t_feat], dim=1)  # (B, 512+256+768)
        fused = self.fusion(fused)                          # (B, 128)
        logits = self.cls_head(fused).squeeze(-1)           # (B,)
        return logits
