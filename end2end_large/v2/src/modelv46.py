from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_blocks import (
    AttentionPooling,
    CausalConvStem,
    GatedFeatureMixer,
    LastByScalePooling,
    MultiTaskHead,
    PatchScaleEmbedding,
)


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


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * hidden)
        self.w3 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.w3(F.silu(gate) * value))


class MultiScalePatchEmbeddingWithCoords(nn.Module):
    def __init__(self, d_model: int, scales: list[tuple[int, int]]) -> None:
        super().__init__()
        self.scales = [(int(p), int(s)) for p, s in scales]
        self.embedders = nn.ModuleList([PatchScaleEmbedding(d_model, p, s) for p, s in self.scales])
        self.scale_embed = nn.Parameter(torch.zeros(len(self.scales), d_model))
        nn.init.normal_(self.scale_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[tuple[int, int]], torch.Tensor]:
        outs: list[torch.Tensor] = []
        spans: list[tuple[int, int]] = []
        coords: list[torch.Tensor] = []
        start = 0
        seq_len = int(x.shape[1])
        for i, (emb, (patch_len, stride)) in enumerate(zip(self.embedders, self.scales)):
            h = emb(x) + self.scale_embed[i].view(1, 1, -1)
            n_tokens = int(h.shape[1])
            end = start + n_tokens
            token_end = torch.arange(n_tokens, device=x.device, dtype=torch.float32) * stride + (patch_len - 1)
            coords.append(token_end.clamp_max(max(0, seq_len - 1)))
            outs.append(h)
            spans.append((start, end))
            start = end
        return torch.cat(outs, dim=1), spans, torch.cat(coords)


class TimeBiasSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        min_half_life: float = 4.0,
        max_half_life: float = 240.0,
        learn_decay: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout_p = float(dropout)
        half_lives = torch.logspace(
            math.log10(float(min_half_life)),
            math.log10(float(max_half_life)),
            steps=n_heads,
        )
        log_decay = torch.log(torch.log(torch.tensor(2.0)) / half_lives)
        if learn_decay:
            self.log_decay = nn.Parameter(log_decay)
        else:
            self.register_buffer("log_decay", log_decay)

    def forward(self, x: torch.Tensor, time_coords: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, _ = x.shape
        qkv = self.qkv(x).view(bsz, n_tokens, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        delta = (time_coords[:, None] - time_coords[None, :]).abs().to(device=x.device, dtype=q.dtype)
        decay = torch.exp(self.log_decay).to(device=x.device, dtype=q.dtype)
        bias = -decay.view(1, self.n_heads, 1, 1) * delta.view(1, 1, n_tokens, n_tokens)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, n_tokens, self.d_model)
        return self.proj(out)


class PlainSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout_p = float(dropout)

    def forward(self, x: torch.Tensor, time_coords: torch.Tensor | None = None) -> torch.Tensor:
        bsz, n_tokens, _ = x.shape
        qkv = self.qkv(x).view(bsz, n_tokens, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, n_tokens, self.d_model)
        return self.proj(out)


class TimeBiasTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        swiglu_hidden: int,
        dropout: float,
        attn_dropout: float,
        layerscale_init: float = 0.1,
        learn_time_decay: bool = False,
        use_time_bias: bool = True,
        use_swiglu_layerscale: bool = True,
    ) -> None:
        super().__init__()
        self.use_swiglu_layerscale = bool(use_swiglu_layerscale)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = (
            TimeBiasSelfAttention(d_model, n_heads, attn_dropout, learn_decay=learn_time_decay)
            if use_time_bias
            else PlainSelfAttention(d_model, n_heads, attn_dropout)
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        if self.use_swiglu_layerscale:
            self.gamma_attn = nn.Parameter(torch.full((d_model,), float(layerscale_init)))
            self.ffn = SwiGLU(d_model, swiglu_hidden, dropout)
            self.gamma_ffn = nn.Parameter(torch.full((d_model,), float(layerscale_init)))
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, swiglu_hidden * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(swiglu_hidden * 2, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor, time_coords: torch.Tensor) -> torch.Tensor:
        h = self.drop1(self.attn(self.norm1(x), time_coords))
        if self.use_swiglu_layerscale:
            x = x + h * self.gamma_attn.view(1, 1, -1)
            return x + self.ffn(self.norm2(x)) * self.gamma_ffn.view(1, 1, -1)
        x = x + h
        return x + self.ffn(self.norm2(x))


class LayerFusion(nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, layers: list[torch.Tensor]) -> torch.Tensor:
        w = torch.softmax(self.weights[: len(layers)], dim=0)
        fused = layers[0] * w[0]
        for i in range(1, len(layers)):
            fused = fused + layers[i] * w[i]
        return fused


def grouped_market_state(x: torch.Tensor, timestamp_ns: torch.Tensor | None) -> torch.Tensor:
    last = x[:, -1, :]
    if timestamp_ns is None:
        mean = last.mean(dim=0, keepdim=True)
        std = last.std(dim=0, unbiased=False, keepdim=True)
        return torch.cat([mean, std], dim=-1).expand(last.shape[0], -1)
    state = torch.empty(last.shape[0], last.shape[1] * 2, device=x.device, dtype=x.dtype)
    for ts in torch.unique(timestamp_ns):
        idx = torch.nonzero(timestamp_ns == ts, as_tuple=False).flatten()
        vals = last.index_select(0, idx)
        mean = vals.mean(dim=0)
        std = vals.std(dim=0, unbiased=False)
        group_state = torch.cat([mean, std], dim=0).view(1, -1).expand(idx.numel(), -1)
        state.index_copy_(0, idx, group_state)
    return state


class MarketGatedFeatureMixer(nn.Module):
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.state_norm = nn.LayerNorm(2 * n_features)
        self.base = nn.Linear(n_features, d_model)
        self.mix = nn.Sequential(
            nn.Linear(n_features, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.gate = nn.Linear(3 * n_features, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, timestamp_ns: torch.Tensor | None) -> torch.Tensor:
        xn = self.norm(x)
        state = self.state_norm(grouped_market_state(xn, timestamp_ns))
        state = state.unsqueeze(1).expand(-1, x.shape[1], -1)
        gate_in = torch.cat([xn, state], dim=-1)
        return self.out_norm(self.base(xn) + torch.sigmoid(self.gate(gate_in)) * self.mix(xn))


class FactorOperatorBank(nn.Module):
    def __init__(
        self,
        n_features: int,
        windows: list[int] | None = None,
        active_k: int = 160,
        gate_hidden: int = 80,
        output_mode: str = "project",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.windows = [int(w) for w in (windows or [3, 5, 8, 13, 21, 34, 55, 89])]
        self.pair_windows = [(3, 13), (5, 21), (8, 34), (13, 55), (21, 89)]
        self.op_dim = 19 + 53 * len(self.windows) + 8 * len(self.pair_windows)
        self.active_k = min(int(active_k), self.op_dim)
        if output_mode not in {"project", "topk", "masked"}:
            raise ValueError("factor output_mode must be one of project/topk/masked")
        self.output_mode = output_mode
        self.output_dim = self.op_dim if output_mode == "masked" else self.active_k
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * self.n_features),
            nn.Linear(3 * self.n_features, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, self.op_dim),
        )
        self.project = nn.Linear(self.op_dim, self.output_dim, bias=False) if output_mode == "project" else None
        self.out_norm = nn.LayerNorm(self.output_dim)

    def _feature(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if idx < x.shape[-1]:
            return x[:, :, idx]
        return torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)

    @staticmethod
    def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        return num / den.abs().clamp_min(eps)

    @staticmethod
    def _rolling_mean(v: torch.Tensor, window: int) -> torch.Tensor:
        h = v.unsqueeze(1)
        ones = torch.ones_like(h)
        total = F.avg_pool1d(F.pad(h, (window - 1, 0)), kernel_size=window, stride=1) * window
        count = F.avg_pool1d(F.pad(ones, (window - 1, 0)), kernel_size=window, stride=1) * window
        return (total / count.clamp_min(1.0)).squeeze(1)

    def _rolling_std(self, v: torch.Tensor, window: int) -> torch.Tensor:
        mean = self._rolling_mean(v, window)
        mean2 = self._rolling_mean(v * v, window)
        return torch.sqrt((mean2 - mean * mean).clamp_min(1e-5))

    def _rolling_corr(self, a: torch.Tensor, b: torch.Tensor, window: int) -> torch.Tensor:
        ma = self._rolling_mean(a, window)
        mb = self._rolling_mean(b, window)
        cov = self._rolling_mean(a * b, window) - ma * mb
        sa = self._rolling_std(a, window)
        sb = self._rolling_std(b, window)
        return self._safe_div(cov, sa * sb)

    def _lag(self, v: torch.Tensor, window: int) -> torch.Tensor:
        if window <= 1:
            return v
        return F.pad(v[:, : -window + 1], (window - 1, 0))

    def _bounded(self, v: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(v, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)

    def _build_ops(self, x: torch.Tensor) -> torch.Tensor:
        ret = self._feature(x, 0)
        oc = self._feature(x, 1)
        hl = self._feature(x, 2)
        upper = self._feature(x, 3)
        lower = self._feature(x, 4)
        volume = self._feature(x, 5)
        volume_chg = self._feature(x, 6)
        amount_chg = self._feature(x, 8)
        oi = self._feature(x, 9)
        oi_chg = self._feature(x, 10)
        close_pos = self._feature(x, 26)
        rel_ret = self._feature(x, 27)
        rel_oc = self._feature(x, 28)
        rel_hl = self._feature(x, 29)
        rel_volume_chg = self._feature(x, 30)
        rel_amount_chg = self._feature(x, 31)
        rel_oi_chg = self._feature(x, 32)

        ops: list[torch.Tensor] = [
            ret,
            oc,
            hl,
            upper,
            lower,
            volume_chg,
            amount_chg,
            oi_chg,
            close_pos,
            rel_ret,
            rel_oc,
            rel_hl,
            rel_volume_chg,
            rel_amount_chg,
            rel_oi_chg,
            ret * volume_chg,
            ret * amount_chg,
            ret * oi_chg,
            self._safe_div(oc.abs(), hl.abs() + upper.abs() + lower.abs() + 1e-3),
        ]

        base_vars = [ret, oc, hl, upper, lower, volume, volume_chg, amount_chg, oi, oi_chg]
        z_vars = [ret, oc, hl, volume_chg, oi_chg]
        for window in self.windows:
            window_ops: list[torch.Tensor] = []
            for var in base_vars:
                mean = self._rolling_mean(var, window)
                std = self._rolling_std(var, window)
                mom = var - self._lag(var, window)
                window_ops.extend([mean, std, mom])
            for var in z_vars:
                mean = self._rolling_mean(var, window)
                std = self._rolling_std(var, window)
                window_ops.append(self._safe_div(var - mean, std))
            rv = torch.sqrt(self._rolling_mean(ret * ret, window).clamp_min(1e-5))
            up_vol = torch.sqrt(self._rolling_mean(F.relu(ret) ** 2, window).clamp_min(1e-5))
            down_vol = torch.sqrt(self._rolling_mean(F.relu(-ret) ** 2, window).clamp_min(1e-5))
            mean_abs_ret = self._rolling_mean(ret.abs(), window)
            ret_vol = self._rolling_mean(ret * volume_chg, window)
            ret_amount = self._rolling_mean(ret * amount_chg, window)
            ret_oi = self._rolling_mean(ret * oi_chg, window)
            corr_ret_vol = self._rolling_corr(ret, volume_chg, window)
            corr_ret_amount = self._rolling_corr(ret, amount_chg, window)
            corr_ret_oi = self._rolling_corr(ret, oi_chg, window)
            vol_pressure = volume - self._rolling_mean(volume, window)
            amount_pressure = amount_chg - self._rolling_mean(amount_chg, window)
            oi_pressure = oi - self._rolling_mean(oi, window)
            stoch = self._rolling_mean(close_pos, window)
            bollinger_z = self._safe_div(close_pos - self._rolling_mean(close_pos, window), self._rolling_std(close_pos, window))
            mfi_proxy = self._rolling_mean(torch.sign(ret) * volume, window)
            trend_strength = self._safe_div(self._rolling_mean(ret, window).abs(), rv)
            range_vol_corr = self._rolling_corr(hl, volume_chg, window)
            window_ops.extend(
                [
                    rv,
                    up_vol,
                    down_vol,
                    mean_abs_ret,
                    ret_vol,
                    ret_amount,
                    ret_oi,
                    corr_ret_vol,
                    corr_ret_amount,
                    corr_ret_oi,
                    vol_pressure,
                    amount_pressure,
                    oi_pressure,
                    stoch,
                    bollinger_z,
                    mfi_proxy,
                    trend_strength,
                    range_vol_corr,
                ]
            )
            if len(window_ops) != 53:
                raise RuntimeError(f"FactorOperatorBank expected 53 window ops, got {len(window_ops)}")
            ops.extend(window_ops)

        pair_vars = [ret, oc, hl, volume_chg, amount_chg, oi_chg, close_pos, rel_ret]
        for short, long in self.pair_windows:
            for var in pair_vars:
                ops.append(self._rolling_mean(var, short) - self._rolling_mean(var, long))

        out = torch.stack([self._bounded(op) for op in ops], dim=-1)
        if out.shape[-1] != self.op_dim:
            raise RuntimeError(f"FactorOperatorBank expected {self.op_dim} ops, got {out.shape[-1]}")
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ops = self._build_ops(x)
        seq_state = torch.cat([x[:, -1, :], x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)
        scores = self.gate(seq_state)
        if self.active_k < self.op_dim:
            top_scores, top_idx = torch.topk(scores, self.active_k, dim=-1)
            if self.output_mode == "topk":
                top_idx, order = torch.sort(top_idx, dim=-1)
                top_scores = torch.gather(top_scores, 1, order)
                selected = torch.gather(ops, 2, top_idx.unsqueeze(1).expand(-1, ops.shape[1], -1))
                return self.out_norm(selected * torch.sigmoid(top_scores).unsqueeze(1))
            mask = torch.zeros_like(scores)
            mask.scatter_(1, top_idx, 1.0)
            weights = torch.sigmoid(scores) * mask
        else:
            weights = torch.sigmoid(scores)
        gated = ops * weights.unsqueeze(1)
        if self.project is not None:
            gated = self.project(gated)
        return self.out_norm(gated)


class LowRankFeatureInteraction(nn.Module):
    def __init__(self, n_features: int, d_model: int, rank: int = 8, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.linear = nn.Linear(n_features, d_model)
        self.residual = nn.Linear(n_features, d_model)
        self.factors = nn.Parameter(torch.empty(n_features, int(rank)))
        self.pair_proj = nn.Linear(int(rank), d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.factors)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        xv = torch.matmul(xn, self.factors)
        x2v2 = torch.matmul(xn * xn, self.factors * self.factors)
        pairwise = 0.5 * (xv * xv - x2v2)
        return self.out_norm(self.linear(xn) + self.pair_proj(self.dropout(pairwise)) + self.residual(xn))


class LowRankMarketFeatureMixer(nn.Module):
    def __init__(self, n_features: int, d_model: int, rank: int = 8, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.state_norm = nn.LayerNorm(2 * n_features)
        self.lowrank = LowRankFeatureInteraction(n_features, d_model, rank, dropout)
        self.mix = nn.Sequential(
            nn.Linear(n_features, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.gate = nn.Linear(3 * n_features, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, timestamp_ns: torch.Tensor | None) -> torch.Tensor:
        xn = self.norm(x)
        state = self.state_norm(grouped_market_state(xn, timestamp_ns))
        state = state.unsqueeze(1).expand(-1, x.shape[1], -1)
        gate_in = torch.cat([xn, state], dim=-1)
        return self.out_norm(self.lowrank(x) + torch.sigmoid(self.gate(gate_in)) * self.mix(xn))


class ResidualLowRankFeatureMixer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        rank: int = 8,
        dropout: float = 0.05,
        use_market_gating: bool = True,
        scale_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.use_market_gating = bool(use_market_gating)
        self.base = (
            MarketGatedFeatureMixer(n_features, d_model, dropout)
            if use_market_gating
            else GatedFeatureMixer(n_features, d_model, dropout)
        )
        self.lowrank = LowRankFeatureInteraction(n_features, d_model, rank, dropout)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, timestamp_ns: torch.Tensor | None = None) -> torch.Tensor:
        base = self.base(x, timestamp_ns) if self.use_market_gating else self.base(x)
        return self.out_norm(base + torch.sigmoid(self.scale) * self.lowrank(x))


class PatchMetaEmbedding(nn.Module):
    def __init__(
        self,
        n_symbols: int,
        d_model: int,
        seq_len: int,
        dropout: float = 0.05,
        scale_init: float = -1.5,
        use_symbol: bool = True,
        use_minute: bool = True,
        use_day: bool = True,
        use_month: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.use_symbol = bool(use_symbol)
        self.use_minute = bool(use_minute)
        self.use_day = bool(use_day)
        self.use_month = bool(use_month)
        self.symbol = nn.Embedding(max(1, int(n_symbols)), d_model)
        self.minute = nn.Embedding(1440, d_model)
        self.day = nn.Embedding(7, d_model)
        self.month = nn.Embedding(12, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(
        self,
        tokens: torch.Tensor,
        time_coords: torch.Tensor,
        symbol_id: torch.Tensor | None,
        minute_of_day: torch.Tensor | None,
        day_of_week: torch.Tensor | None,
        month: torch.Tensor | None,
    ) -> torch.Tensor:
        if symbol_id is None or minute_of_day is None or day_of_week is None or month is None:
            return tokens
        coords = time_coords.round().long().to(device=tokens.device)
        age = (self.seq_len - 1 - coords).clamp_min(0)
        token_minute = (minute_of_day.to(tokens.device).long().unsqueeze(1) - age.view(1, -1)) % 1440
        emb = torch.zeros_like(tokens)
        if self.use_minute:
            emb = emb + self.minute(token_minute)
        if self.use_symbol:
            emb = emb + self.symbol(symbol_id.to(tokens.device).long()).unsqueeze(1)
        if self.use_day:
            emb = emb + self.day(day_of_week.to(tokens.device).long().clamp(0, 6)).unsqueeze(1)
        if self.use_month:
            emb = emb + self.month(month.to(tokens.device).long().clamp(0, 11)).unsqueeze(1)
        return tokens + torch.sigmoid(self.scale) * self.dropout(emb)


class MoEPredictionHead(nn.Module):
    def __init__(self, d_model: int, n_targets: int, n_experts: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_experts = int(n_experts)
        self.router = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.n_experts))
        self.experts = nn.ModuleList([MultiTaskHead(d_model, n_targets, dropout) for _ in range(self.n_experts)])
        self.last_balance_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        weights = torch.softmax(logits, dim=-1)
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=1)
        self.last_balance_loss = ((weights.mean(dim=0) - (1.0 / self.n_experts)) ** 2).sum() * self.n_experts
        return torch.sum(expert_out * weights.unsqueeze(-1), dim=1)


class CrossSectionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ffn_mult: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.gamma_attn = nn.Parameter(torch.full((d_model,), 0.1))
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.gamma_ffn = nn.Parameter(torch.full((d_model,), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h) * self.gamma_attn.view(1, 1, -1)
        return x + self.ffn(self.norm2(x)) * self.gamma_ffn.view(1, 1, -1)


class CrossSectionStack(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float, min_group_size: int = 4) -> None:
        super().__init__()
        self.min_group_size = int(min_group_size)
        self.blocks = nn.ModuleList([CrossSectionBlock(d_model, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, timestamp_ns: torch.Tensor | None) -> torch.Tensor:
        if timestamp_ns is None:
            if x.shape[0] < self.min_group_size:
                return x
            h = x.unsqueeze(0)
            for block in self.blocks:
                h = block(h)
            return h.squeeze(0)
        out = x.clone()
        for ts in torch.unique(timestamp_ns):
            idx = torch.nonzero(timestamp_ns == ts, as_tuple=False).flatten()
            if idx.numel() < self.min_group_size:
                continue
            h = x.index_select(0, idx).unsqueeze(0)
            for block in self.blocks:
                h = block(h)
            out.index_copy_(0, idx, h.squeeze(0))
        return out


class CrossVariateBranch(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.seq_proj = nn.Linear(self.seq_len, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.normal_(self.var_embed, std=0.02)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, max(64, int(2.0 * d_model)), dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        if h.shape[-1] < self.seq_len:
            h = F.pad(h, (self.seq_len - h.shape[-1], 0))
        elif h.shape[-1] > self.seq_len:
            h = h[..., -self.seq_len :]
        h = self.seq_proj(h) + self.var_embed[:, : h.shape[1], :]
        a = self.norm1(h)
        a, _ = self.attn(a, a, a, need_weights=False)
        h = h + self.drop(a)
        h = h + self.ffn(self.norm2(h))
        return self.out_norm(h.mean(dim=1))


class E2E_GatedMSPatch_MTL_DataLimited_v46(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int = 6,
        seq_len: int = 240,
        d_model: int = 192,
        n_layers: int = 5,
        n_heads: int = 6,
        swiglu_hidden: int = 512,
        dropout: float = 0.15,
        attn_dropout: float = 0.10,
        feature_dropout: float = 0.05,
        patch_scales: list[tuple[int, int]] | None = None,
        causal_conv_stem: bool = True,
        use_time_bias: bool = True,
        use_layer_fusion: bool = True,
        use_revin: bool = False,
        use_cross_section: bool = False,
        cross_section_layers: int = 1,
        cross_section_min_group: int = 4,
        use_market_gating: bool = False,
        use_cross_variate: bool = False,
        learn_time_decay: bool = False,
        use_swiglu_layerscale: bool = True,
        use_factor_bank: bool = False,
        factor_top_k: int = 160,
        factor_gate_hidden: int = 80,
        factor_output_mode: str = "project",
        factor_scale_init: float | None = None,
        use_lowrank_input: bool = False,
        lowrank_rank: int = 8,
        lowrank_residual: bool = False,
        lowrank_scale_init: float = -2.0,
        use_meta_embedding: bool = False,
        n_symbols: int = 1,
        meta_scale_init: float = -1.5,
        meta_use_symbol: bool = True,
        meta_use_minute: bool = True,
        meta_use_day: bool = True,
        meta_use_month: bool = True,
        moe_n_experts: int = 1,
    ) -> None:
        super().__init__()
        scales = patch_scales or [(4, 2), (8, 4), (16, 8), (32, 16)]
        self.use_revin = bool(use_revin)
        self.use_layer_fusion = bool(use_layer_fusion)
        self.use_market_gating = bool(use_market_gating)
        self.use_factor_bank = bool(use_factor_bank)
        self.revin = RevINInput(n_features) if use_revin else nn.Identity()
        self.factor_bank = (
            FactorOperatorBank(
                n_features,
                active_k=factor_top_k,
                gate_hidden=factor_gate_hidden,
                output_mode=factor_output_mode,
                dropout=feature_dropout,
            )
            if use_factor_bank
            else None
        )
        self.factor_scale = (
            nn.Parameter(torch.tensor(float(factor_scale_init)))
            if self.factor_bank is not None and factor_scale_init is not None
            else None
        )
        input_features = n_features + (self.factor_bank.output_dim if self.factor_bank is not None else 0)
        if use_lowrank_input and lowrank_residual:
            self.mixer = ResidualLowRankFeatureMixer(
                input_features,
                d_model,
                lowrank_rank,
                feature_dropout,
                use_market_gating=use_market_gating,
                scale_init=lowrank_scale_init,
            )
        elif use_lowrank_input and use_market_gating:
            self.mixer = LowRankMarketFeatureMixer(input_features, d_model, lowrank_rank, feature_dropout)
        elif use_lowrank_input:
            self.mixer = LowRankFeatureInteraction(input_features, d_model, lowrank_rank, feature_dropout)
        elif use_market_gating:
            self.mixer = MarketGatedFeatureMixer(input_features, d_model, feature_dropout)
        else:
            self.mixer = GatedFeatureMixer(input_features, d_model, feature_dropout)
        self.stem = CausalConvStem(d_model, dropout) if causal_conv_stem else nn.Identity()
        self.patch = MultiScalePatchEmbeddingWithCoords(d_model, scales)
        self.meta_embedding = (
            PatchMetaEmbedding(
                n_symbols,
                d_model,
                seq_len,
                feature_dropout,
                scale_init=meta_scale_init,
                use_symbol=meta_use_symbol,
                use_minute=meta_use_minute,
                use_day=meta_use_day,
                use_month=meta_use_month,
            )
            if use_meta_embedding
            else None
        )
        self.blocks = nn.ModuleList(
            [
                TimeBiasTransformerBlock(
                    d_model,
                    n_heads,
                    swiglu_hidden,
                    dropout,
                    attn_dropout,
                    learn_time_decay=learn_time_decay,
                    use_time_bias=use_time_bias,
                    use_swiglu_layerscale=use_swiglu_layerscale,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_fusion = LayerFusion(n_layers) if use_layer_fusion else nn.Identity()
        self.attn_pool = AttentionPooling(d_model, dropout)
        self.last_pool = LastByScalePooling()
        self.fuse = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.variate_branch = (
            CrossVariateBranch(n_features, seq_len, d_model, n_heads, dropout) if use_cross_variate else None
        )
        self.variate_scale = nn.Parameter(torch.tensor(-2.0)) if use_cross_variate else None
        self.cross_section = (
            CrossSectionStack(d_model, n_heads, cross_section_layers, dropout, cross_section_min_group)
        if use_cross_section
            else None
        )
        self.head = (
            MoEPredictionHead(d_model, n_targets, moe_n_experts, dropout)
            if int(moe_n_experts) > 1
            else MultiTaskHead(d_model, n_targets, dropout)
        )

    def extra_loss(self) -> torch.Tensor | None:
        if isinstance(self.head, MoEPredictionHead):
            return self.head.last_balance_loss
        return None

    def forward(
        self,
        x: torch.Tensor,
        timestamp_ns: torch.Tensor | None = None,
        symbol_id: torch.Tensor | None = None,
        minute_of_day: torch.Tensor | None = None,
        day_of_week: torch.Tensor | None = None,
        month: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_norm = self.revin(x)
        mixer_input = x_norm
        if self.factor_bank is not None:
            factor_features = self.factor_bank(x_norm)
            if self.factor_scale is not None:
                factor_features = torch.sigmoid(self.factor_scale) * factor_features
            mixer_input = torch.cat([x_norm, factor_features], dim=-1)
        if self.use_market_gating:
            h = self.mixer(mixer_input, timestamp_ns)
        else:
            h = self.mixer(mixer_input)
        h = self.stem(h)
        tokens, spans, time_coords = self.patch(h)
        if self.meta_embedding is not None:
            tokens = self.meta_embedding(tokens, time_coords, symbol_id, minute_of_day, day_of_week, month)
        layer_outputs: list[torch.Tensor] = []
        for block in self.blocks:
            tokens = block(tokens, time_coords)
            if self.use_layer_fusion:
                layer_outputs.append(tokens)
        if self.use_layer_fusion:
            tokens = self.layer_fusion(layer_outputs)
        pooled = self.fuse(torch.cat([self.attn_pool(tokens), self.last_pool(tokens, spans)], dim=-1))
        if self.variate_branch is not None and self.variate_scale is not None:
            pooled = pooled + torch.sigmoid(self.variate_scale) * self.variate_branch(x_norm)
        if self.cross_section is not None:
            pooled = self.cross_section(pooled, timestamp_ns)
        return self.head(pooled)
