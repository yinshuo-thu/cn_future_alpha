from __future__ import annotations

import torch
import torch.nn as nn


class LowRankFeatureInteraction(nn.Module):
    """FM-style low-rank second-order raw feature interaction block."""

    def __init__(
        self,
        n_features: int,
        out_dim: int,
        rank: int = 8,
        dropout: float = 0.1,
        gated: bool = True,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.factors = nn.Parameter(torch.empty(n_features, out_dim, rank))
        self.base = nn.Linear(n_features, out_dim)
        self.residual = nn.Linear(n_features, out_dim)
        self.gate = nn.Sequential(nn.Linear(n_features, out_dim), nn.Sigmoid()) if gated else None
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        nn.init.normal_(self.factors, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        summed = torch.einsum("btf,for->btor", x, self.factors)
        summed_square = summed.pow(2)
        square_summed = torch.einsum("btf,for->btor", x.pow(2), self.factors.pow(2))
        interaction = 0.5 * (summed_square - square_summed).sum(dim=-1)
        if self.gate is not None:
            interaction = interaction * self.gate(x)
        out = self.base(x) + self.dropout(interaction) + self.residual(x)
        return self.norm(out)

