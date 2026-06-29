from __future__ import annotations

import torch
import torch.nn as nn

from .model_blocks import (
    AttentionPooling,
    CausalConvStem,
    GatedFeatureMixer,
    LastByScalePooling,
    MultiScalePatchEmbedding,
    MultiTaskHead,
    PreNormTransformerBlock,
)


class E2E_GatedMSPatch_MTL_DataLimited_v1(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int = 6,
        d_model: int = 192,
        n_layers: int = 5,
        n_heads: int = 6,
        ffn_dim: int = 512,
        dropout: float = 0.15,
        attn_dropout: float = 0.10,
        feature_dropout: float = 0.05,
        patch_scales: list[tuple[int, int]] | None = None,
        causal_conv_stem: bool = True,
    ) -> None:
        super().__init__()
        self.mixer = GatedFeatureMixer(n_features, d_model, feature_dropout)
        self.stem = CausalConvStem(d_model, dropout) if causal_conv_stem else nn.Identity()
        self.patch = MultiScalePatchEmbedding(d_model, patch_scales or [(4, 2), (8, 4), (16, 8), (32, 16)])
        self.blocks = nn.ModuleList(
            [PreNormTransformerBlock(d_model, n_heads, ffn_dim, dropout, attn_dropout) for _ in range(n_layers)]
        )
        self.attn_pool = AttentionPooling(d_model, dropout)
        self.last_pool = LastByScalePooling()
        self.fuse = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.head = MultiTaskHead(d_model, n_targets, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(self.mixer(x))
        tokens, spans = self.patch(h)
        for block in self.blocks:
            tokens = block(tokens)
        pooled = torch.cat([self.attn_pool(tokens), self.last_pool(tokens, spans)], dim=-1)
        return self.head(self.fuse(pooled))


class GRUBaseline(nn.Module):
    def __init__(self, n_features: int, n_targets: int = 6, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1])


class TCNBaseline(nn.Module):
    def __init__(self, n_features: int, n_targets: int = 6, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.inp = nn.Linear(n_features, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=5, dilation=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, dilation=2)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x).transpose(1, 2)
        h = self.drop(torch.relu(self.conv1(torch.nn.functional.pad(h, (4, 0)))))
        h = self.drop(torch.relu(self.conv2(torch.nn.functional.pad(h, (8, 0)))))
        return self.head(h[:, :, -1])


class SingleScalePatchTransformerBaseline(E2E_GatedMSPatch_MTL_DataLimited_v1):
    def __init__(self, n_features: int, n_targets: int = 6, d_model: int = 128) -> None:
        super().__init__(
            n_features=n_features,
            n_targets=n_targets,
            d_model=d_model,
            n_layers=3,
            n_heads=4,
            ffn_dim=256,
            dropout=0.12,
            attn_dropout=0.08,
            patch_scales=[(15, 5)],
            causal_conv_stem=True,
        )


class AggregatedMLP(nn.Module):
    def __init__(self, n_inputs: int, n_targets: int = 6, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_inputs),
            nn.Linear(n_inputs, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
