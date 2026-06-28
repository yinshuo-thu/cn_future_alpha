from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        seq_len: int,
        patch_len: int,
        stride: int,
        dropout: float = 0.1,
        encoder: str = "mlp",
    ) -> None:
        super().__init__()
        if seq_len < patch_len:
            raise ValueError("seq_len must be >= patch_len")
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.encoder = encoder
        in_dim = patch_len * input_dim
        if encoder == "linear":
            self.proj = nn.Linear(in_dim, d_model)
        elif encoder == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        elif encoder == "conv1d":
            self.proj = nn.Conv1d(input_dim, d_model, kernel_size=patch_len, stride=stride)
        else:
            raise ValueError(f"unknown patch encoder: {encoder}")
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder == "conv1d":
            tok = self.proj(x.transpose(1, 2)).transpose(1, 2)
            return self.dropout(tok + self.pos)
        # torch unfold on dim=1 returns [B, N, C, patch_len].
        patches = x.unfold(1, self.patch_len, self.stride).contiguous()
        patches = patches.reshape(x.shape[0], self.n_patches, self.patch_len * self.input_dim)
        return self.dropout(self.proj(patches) + self.pos)


class MultiScalePatchEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        seq_len: int,
        scales: list[tuple[int, int]],
        dropout: float = 0.1,
        encoder: str = "mlp",
    ) -> None:
        super().__init__()
        if not scales:
            raise ValueError("multi-scale patch embedding requires at least one scale")
        self.embeddings = nn.ModuleList(
            [
                PatchEmbedding(
                    input_dim=input_dim,
                    d_model=d_model,
                    seq_len=seq_len,
                    patch_len=int(patch_len),
                    stride=int(stride),
                    dropout=dropout,
                    encoder=encoder,
                )
                for patch_len, stride in scales
            ]
        )
        self.scale_embed = nn.Parameter(torch.zeros(1, len(scales), d_model))
        self.dropout = nn.Dropout(dropout)
        self.n_patches = sum(e.n_patches for e in self.embeddings)
        nn.init.trunc_normal_(self.scale_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = []
        for i, emb in enumerate(self.embeddings):
            tokens.append(emb(x) + self.scale_embed[:, i : i + 1])
        return self.dropout(torch.cat(tokens, dim=1))
