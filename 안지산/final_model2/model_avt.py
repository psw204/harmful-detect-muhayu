# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    """
    간단한 1D Conv 기반 오디오 인코더
    입력: (B,1,N)
    출력: (B,D_a)
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=400, stride=160, padding=120),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B,128,1)
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.net(wav)          # (B,128,1)
        x = x.squeeze(-1)          # (B,128)
        x = self.proj(x)           # (B,out_dim)
        return x


class VideoEncoder(nn.Module):
    """
    매우 가벼운 3D Conv 기반 비디오 인코더
    입력: (B,T,C,H,W)
    출력: (B,D_v)
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        # (B,T,C,H,W) → (B,C,T,H,W)
        self.conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # (B,128,1,1,1)
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        if vid.dim() == 4:
            vid = vid.unsqueeze(0)  # (1,T,C,H,W)
        x = vid.permute(0, 2, 1, 3, 4).contiguous()  # (B,C,T,H,W)
        x = self.conv(x)            # (B,128,1,1,1)
        x = x.view(x.size(0), -1)   # (B,128)
        x = self.proj(x)            # (B,out_dim)
        return x


class TextProjector(nn.Module):
    """
    KoBERT CLS 임베딩(768차원)을 저차원으로 투영
    입력: (B,768)
    출력: (B,D_t)
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, out_dim),
        )

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        return self.net(cls_emb)


class AVTModel(nn.Module):
    """
    멀티모달 히든 벡터:
      - video:  D
      - audio:  D
      - text:   D
    fusion & harmful(0/1) 분류 + InfoNCE용 z_av, z_t 반환
    """

    def __init__(self, text_hidden_dim: int = 768, dim: int = 256):
        super().__init__()
        self.video_enc = VideoEncoder(out_dim=dim)
        self.audio_enc = AudioEncoder(out_dim=dim)
        self.text_proj = TextProjector(in_dim=text_hidden_dim, out_dim=dim)

        # AV fusion (for InfoNCE)
        self.fuse_av = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
        )

        # 최종 분류 헤드 (v+a+t)
        self.cls_head = nn.Sequential(
            nn.Linear(dim * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(
        self, vid: torch.Tensor, wav: torch.Tensor, txt_cls: torch.Tensor
    ):
        """
        vid:     (B,T,C,H,W)
        wav:     (B,1,N)
        txt_cls: (B,768)  # KoBERT CLS
        """
        v = self.video_enc(vid)       # (B,D)
        a = self.audio_enc(wav)       # (B,D)
        t = self.text_proj(txt_cls)   # (B,D)

        z_av = self.fuse_av(torch.cat([v, a], dim=1))   # (B,D)
        logit = self.cls_head(torch.cat([v, a, t], dim=1)).squeeze(1)  # (B,)

        return logit, z_av, t  # (B,), (B,D), (B,D)
