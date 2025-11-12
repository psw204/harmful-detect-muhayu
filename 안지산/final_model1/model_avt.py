# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioBackbone(nn.Module):
    """YAMNet 대체 경량 오디오 백본 (1D Conv → GAP).
    weights/yamnet.pt가 있으면 여기 로딩하도록 후에 교체 가능."""
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=400, stride=160, padding=120),  # 약 25ms @16kHz
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 64

    def forward(self, wav):               # (B,1,N)
        x = self.feat(wav)               # (B,64,T')
        x = self.pool(x).squeeze(-1)     # (B,64)
        return x


class VideoBackbone(nn.Module):
    """MoViNet 자리: 현재는 경량 3D Conv 스텁 → GAP → Linear"""
    def __init__(self, out_dim=512):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(32, out_dim)
        )

    def forward(self, vid):              # (B,T,C,H,W) 또는 (T,C,H,W)
        if vid.dim() == 4:
            vid = vid.unsqueeze(0)
        x = vid.permute(0, 2, 1, 3, 4).contiguous()  # (B,C,T,H,W)
        x = self.proj(x)                  # (B,out_dim)
        return x


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class AVTModel(nn.Module):
    """비디오/오디오 임베딩 → 동일 차원으로 프로젝션 → AV 융합 → (AV, T) 대조 + 분류"""
    def __init__(self, dims={'v':512, 'a':64, 't':768}, proj_dim=512):
        super().__init__()
        self.video = VideoBackbone(out_dim=dims['v'])
        self.audio = AudioBackbone()

        self.Pv = Projection(dims['v'], proj_dim)
        self.Pa = Projection(self.audio.out_dim, proj_dim)
        self.Pt = Projection(dims['t'], proj_dim)

        self.fuse = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(proj_dim * 3, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, vid, wav, t_cls):
        # vid: (B,T,C,H,W), wav: (B,1,N), t_cls: (B,768)
        v = self.video(vid)      # (B, Dv)
        a = self.audio(wav)      # (B, Da)
        v = self.Pv(v)           # (B, d)
        a = self.Pa(a)           # (B, d)
        t = self.Pt(t_cls)       # (B, d)

        z_av = self.fuse(torch.cat([v, a], dim=1))           # (B, d)
        logit = self.cls_head(torch.cat([v, a, t], dim=1)).squeeze(1)  # (B,)
        return logit, z_av, t
