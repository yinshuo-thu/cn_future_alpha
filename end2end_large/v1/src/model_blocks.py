from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFeatureMixer(nn.Module):
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.base = nn.Linear(n_features, d_model)
        self.mix = nn.Sequential(
            nn.Linear(n_features, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.gate = nn.Linear(n_features, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        return self.out_norm(self.base(xn) + torch.sigmoid(self.gate(xn)) * self.mix(xn))


class CausalConvStem(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = F.pad(h, (2, 0))
        h = self.dropout(F.gelu(self.conv3(h)))
        h = F.pad(h, (4, 0))
        h = self.dropout(F.gelu(self.conv5(h)))
        return self.norm(h.transpose(1, 2))


class PatchScaleEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int) -> None:
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.proj = nn.Linear(self.patch_len * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.patch_len:
            pad = self.patch_len - x.shape[1]
            x = F.pad(x.transpose(1, 2), (pad, 0)).transpose(1, 2)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # [B, N, D, P] -> [B, N, P*D]
        patches = patches.transpose(-1, -2).contiguous().flatten(start_dim=2)
        return self.proj(patches)


class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, d_model: int, scales: list[tuple[int, int]]) -> None:
        super().__init__()
        self.scales = [(int(p), int(s)) for p, s in scales]
        self.embedders = nn.ModuleList([PatchScaleEmbedding(d_model, p, s) for p, s in self.scales])
        self.scale_embed = nn.Parameter(torch.zeros(len(self.scales), d_model))
        nn.init.normal_(self.scale_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        outs = []
        spans = []
        start = 0
        for i, emb in enumerate(self.embedders):
            h = emb(x) + self.scale_embed[i].view(1, 1, -1)
            outs.append(h)
            end = start + h.shape[1]
            spans.append((start, end))
            start = end
        return torch.cat(outs, dim=1), spans


class PreNormTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float, attn_dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.drop1(h)
        return x + self.ffn(self.norm2(x))


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(16, d_model // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class LastByScalePooling(nn.Module):
    def forward(self, x: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
        pieces = [x[:, end - 1, :] for _, end in spans if end > 0]
        return torch.stack(pieces, dim=0).mean(dim=0)


class MultiTaskHead(nn.Module):
    def __init__(self, d_model: int, n_targets: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
