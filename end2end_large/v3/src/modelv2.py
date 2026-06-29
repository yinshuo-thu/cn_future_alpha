from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class RevINInput(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        out = (x - mu) / std
        if self.affine:
            out = out * self.gamma.view(1, 1, -1) + self.beta.view(1, 1, -1)
        return out


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class GatedFeatureMixerV2(nn.Module):
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = RMSNorm(n_features)
        self.base = nn.Linear(n_features, d_model)
        self.mix = nn.Sequential(
            nn.Linear(n_features, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.gate = nn.Linear(n_features, d_model)
        self.out_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        return self.out_norm(self.base(xn) + torch.sigmoid(self.gate(xn)) * self.mix(xn))


class TimeContextEmbedding(nn.Module):
    def __init__(self, d_model: int, minute_buckets: int = 288, dropout: float = 0.05) -> None:
        super().__init__()
        self.minute = nn.Embedding(minute_buckets, d_model)
        self.session = nn.Embedding(3, d_model)
        self.cont = nn.Sequential(nn.Linear(3, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        minute_bucket: torch.Tensor,
        session_id: torch.Tensor,
        time_cont: torch.Tensor,
    ) -> torch.Tensor:
        minute_bucket = minute_bucket.clamp_min(0).clamp_max(self.minute.num_embeddings - 1)
        session_id = session_id.clamp_min(0).clamp_max(self.session.num_embeddings - 1)
        return self.drop(self.minute(minute_bucket) + self.session(session_id) + self.cont(time_cont))


class CausalConvStemV2(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5)
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = F.pad(h, (2, 0))
        h = self.dropout(F.silu(self.conv3(h)))
        h = F.pad(h, (4, 0))
        h = self.dropout(F.silu(self.conv5(h)))
        return self.norm(h.transpose(1, 2))


class DWSepPatchEmbed(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=self.patch_len, stride=self.stride, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.patch_len:
            pad = self.patch_len - x.shape[1]
            x = F.pad(x.transpose(1, 2), (pad, 0)).transpose(1, 2)
        h = x.transpose(1, 2)
        h = self.pw(self.dw(h)).transpose(1, 2)
        return self.drop(self.norm(h))


class MultiScaleDWSepPatchEmbedding(nn.Module):
    def __init__(self, d_model: int, scales: list[tuple[int, int]], dropout: float = 0.0) -> None:
        super().__init__()
        self.scales = [(int(p), int(s)) for p, s in scales]
        self.embedders = nn.ModuleList([DWSepPatchEmbed(d_model, p, s, dropout=dropout) for p, s in self.scales])
        self.scale_embed = nn.Parameter(torch.zeros(len(self.scales), d_model))
        nn.init.normal_(self.scale_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[tuple[int, int]], torch.Tensor]:
        outs = []
        spans = []
        pos_ids = []
        start = 0
        for i, emb in enumerate(self.embedders):
            h = emb(x) + self.scale_embed[i].view(1, 1, -1)
            outs.append(h)
            n_tokens = h.shape[1]
            end = start + n_tokens
            spans.append((start, end))
            pos_ids.append(torch.arange(n_tokens, device=x.device, dtype=torch.long))
            start = end
        return torch.cat(outs, dim=1), spans, torch.cat(pos_ids)


def apply_rope(x: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
    # x: [B, H, N, Hd]
    head_dim = x.shape[-1]
    half = head_dim // 2
    freq = torch.arange(half, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (freq / half))
    angles = pos_ids.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = torch.cos(angles).view(1, 1, -1, half).to(dtype=x.dtype)
    sin = torch.sin(angles).view(1, 1, -1, half).to(dtype=x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., 2 * half :]], dim=-1)
    return out


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * hidden)
        self.w3 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.w3(F.silu(gate) * value))


class ModernTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        swiglu_hidden: int,
        dropout: float,
        attn_dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop_p = float(attn_dropout)
        self.drop = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, swiglu_hidden, dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        qkv = self.qkv(h).view(h.shape[0], h.shape[1], 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, pos_ids)
        k = apply_rope(k, pos_ids)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.d_model)
        x = x + self.drop_path1(self.drop(self.proj(attn)))
        return x + self.drop_path2(self.ffn(self.norm2(x)))


class AttentionPoolingV2(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, max(32, d_model // 2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(32, d_model // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class LastByScalePooling(nn.Module):
    def forward(self, x: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
        pieces = [x[:, end - 1, :] for _, end in spans if end > 0]
        return torch.stack(pieces, dim=0).mean(dim=0)


class MultiTaskHeadV2(nn.Module):
    def __init__(self, d_model: int, n_targets: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class E2E_GatedMSPatch_MTL_DataLimited_v2(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int = 6,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        swiglu_hidden: int = 768,
        dropout: float = 0.15,
        attn_dropout: float = 0.10,
        feature_dropout: float = 0.05,
        drop_path_max: float = 0.10,
        patch_scales: list[tuple[int, int]] | None = None,
        use_revin: bool = True,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        scales = patch_scales or [(4, 2), (8, 4), (16, 8), (32, 16)]
        self.use_checkpoint = use_checkpoint
        self.revin = RevINInput(n_features) if use_revin else nn.Identity()
        self.mixer = GatedFeatureMixerV2(n_features, d_model, feature_dropout)
        self.time_embed = TimeContextEmbedding(d_model, dropout=feature_dropout)
        self.stem = CausalConvStemV2(d_model, dropout)
        self.patch = MultiScaleDWSepPatchEmbedding(d_model, scales, dropout=feature_dropout)
        drop_paths = torch.linspace(0.0, float(drop_path_max), steps=n_layers).tolist()
        self.blocks = nn.ModuleList(
            [
                ModernTransformerBlock(d_model, n_heads, swiglu_hidden, dropout, attn_dropout, drop_path=dp)
                for dp in drop_paths
            ]
        )
        self.attn_pool = AttentionPoolingV2(d_model, dropout)
        self.last_pool = LastByScalePooling()
        self.fuse = nn.Sequential(nn.Linear(2 * d_model, d_model), RMSNorm(d_model))
        self.head = MultiTaskHeadV2(d_model, n_targets, dropout)

    def forward(
        self,
        x: torch.Tensor,
        minute_bucket: torch.Tensor,
        session_id: torch.Tensor,
        time_cont: torch.Tensor,
    ) -> torch.Tensor:
        h = self.mixer(self.revin(x))
        h = h + self.time_embed(minute_bucket, session_id, time_cont)
        h = self.stem(h)
        tokens, spans, pos_ids = self.patch(h)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                tokens = checkpoint(block, tokens, pos_ids, use_reentrant=False)
            else:
                tokens = block(tokens, pos_ids)
        pooled = torch.cat([self.attn_pool(tokens), self.last_pool(tokens, spans)], dim=-1)
        return self.head(self.fuse(pooled))
