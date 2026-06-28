from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interactions import LowRankFeatureInteraction
from .patching import MultiScalePatchEmbedding, PatchEmbedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        residual_scale: float = 1.0,
        ffn_type: str = "glu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        if str(ffn_type).lower() in {"mlp", "gelu"}:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_dim * 2),
                nn.GLU(dim=-1),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model),
            )
        self.drop2 = nn.Dropout(dropout)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.residual_scale * self.drop1(attn)
        h = self.norm2(x)
        x = x + self.residual_scale * self.drop2(self.ffn(h))
        return x


class LayerOutputAggregation(nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        stack = torch.stack(hidden_states, dim=0)
        w = torch.softmax(self.weights[: len(hidden_states)], dim=0).view(-1, 1, 1, 1)
        return (stack * w).sum(dim=0)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class RegimeConditioner(nn.Module):
    """Window-state conditioning inspired by regime/time-bucket smoothing."""

    def __init__(
        self,
        n_features: int,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = int(n_features) * 3
        hidden = max(16, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, :]
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        state = torch.cat([last, mean, std], dim=-1)
        return torch.tanh(self.scale) * self.net(state)


class StaticFeatureGate(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_features))
        self.dropout = nn.Dropout(dropout)

    def scale(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x * self.scale().view(1, 1, -1))

    def l1_penalty(self) -> torch.Tensor:
        return self.scale().mean()


def _feature_base_name(col: str) -> str:
    return col[3:] if col.startswith("rz_") else col


class FactorOperatorBank(nn.Module):
    """Causal low-order operator bank inspired by the selected factor formulas.

    This layer does not consume precomputed factors. It derives differentiable
    operator responses from the raw/stable sequence channels inside the model.
    """

    def __init__(
        self,
        n_features: int,
        feature_cols: list[str] | None,
        windows: list[int] | None = None,
        dropout: float = 0.0,
        clip_value: float = 12.0,
        include_raw: bool = True,
        projection_dim: int = 0,
        gate_mode: str = "none",
        topk: int = 0,
        topk_mode: str = "hard",
        soft_topk_temperature: float = 1.0,
        gate_hidden: int = 64,
        extra_ops: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.windows = [int(w) for w in (windows or [2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 26, 34, 42, 55, 72, 89, 120])]
        self.windows = sorted({w for w in self.windows if w >= 2})
        self.dropout = nn.Dropout(dropout)
        self.clip_value = float(clip_value)
        self.include_raw = bool(include_raw)
        bases = [_feature_base_name(col) for col in (feature_cols or [])]
        self.index = {base: i for i, base in enumerate(bases)}
        self.required = [
            "open",
            "high",
            "low",
            "close",
            "log1p_volume",
            "log1p_amount",
            "log1p_oi",
            "log_ret_1",
            "range_pct",
            "close_pos",
            "oc_ret",
            "d_log1p_volume",
            "d_log1p_amount",
            "d_log1p_oi",
        ]
        self.base_output_dim = self.n_features if self.include_raw else 0
        self.extra_ops = bool(extra_ops)
        extra_global_dim = 6 if self.extra_ops else 0
        extra_per_window_dim = 20 if self.extra_ops else 0
        extra_pair_dim = 4 * len(self._valid_pairs()) if self.extra_ops else 0
        per_window_dim = 33 + extra_per_window_dim
        pair_dim = 4 * len(self._valid_pairs())
        global_dim = 13 + extra_global_dim
        self.ops_dim = global_dim + per_window_dim * len(self.windows) + pair_dim + extra_pair_dim
        self.projection_dim = max(0, int(projection_dim))
        self.gate_mode = (gate_mode or "none").lower()
        self.topk = max(0, int(topk))
        self.topk_mode = str(topk_mode or "hard").lower()
        self.soft_topk_temperature = max(1e-3, float(soft_topk_temperature or 1.0))
        self.last_gate: torch.Tensor | None = None
        self.output_dim = (
            self.base_output_dim + self.projection_dim
            if self.projection_dim > 0
            else self.base_output_dim + self.ops_dim
        )
        self.projector = (
            nn.Sequential(
                nn.LayerNorm(self.ops_dim),
                nn.Linear(self.ops_dim, self.projection_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.projection_dim * 2, self.projection_dim),
            )
            if self.projection_dim > 0
            else None
        )
        if self.gate_mode == "static":
            self.operator_logits = nn.Parameter(torch.zeros(self.ops_dim))
            self.gate_net = None
        elif self.gate_mode == "dynamic":
            hidden = max(16, int(gate_hidden))
            self.operator_logits = None
            self.gate_net = nn.Sequential(
                nn.LayerNorm(self.n_features),
                nn.Linear(self.n_features, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.ops_dim),
            )
        elif self.gate_mode in {"sequence", "seq", "sample"}:
            hidden = max(16, int(gate_hidden))
            self.operator_logits = None
            self.gate_net = nn.Sequential(
                nn.LayerNorm(self.n_features * 3),
                nn.Linear(self.n_features * 3, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.ops_dim),
            )
        elif self.gate_mode in {"none", ""}:
            self.operator_logits = None
            self.gate_net = None
        else:
            raise ValueError(f"unknown operator gate mode: {gate_mode}")

    def _apply_operator_budget(self, logits: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(logits)
        if self.topk <= 0 or self.topk >= self.ops_dim:
            return gate
        if self.topk_mode in {"budget", "sigmoid_budget", "soft_budget"}:
            scale = (float(self.topk) / (gate.sum(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)
            return gate * scale
        if self.topk_mode in {"soft", "softmax", "relaxed"}:
            selector = torch.softmax(logits / self.soft_topk_temperature, dim=-1) * float(self.topk)
            return gate * selector.clamp(max=1.0)
        idx = torch.topk(gate, k=self.topk, dim=-1, largest=True).indices
        mask = torch.zeros_like(gate)
        mask.scatter_(-1, idx, 1.0)
        hard_gate = gate * mask
        if self.topk_mode in {"straight_through", "st", "soft_hard"}:
            selector = torch.softmax(logits / self.soft_topk_temperature, dim=-1) * float(self.topk)
            soft_gate = gate * selector.clamp(max=1.0)
            return hard_gate.detach() - soft_gate.detach() + soft_gate
        return hard_gate

    def _valid_pairs(self) -> list[tuple[int, int]]:
        pairs = [(3, 13), (5, 21), (8, 34), (10, 42), (13, 55), (16, 72), (21, 89)]
        available = set(self.windows)
        return [(a, b) for a, b in pairs if a in available and b in available]

    def _col(self, x: torch.Tensor, name: str) -> torch.Tensor:
        idx = self.index.get(name)
        if idx is None or idx >= x.shape[-1]:
            return x.new_zeros(x.shape[:2])
        return x[:, :, idx]

    @staticmethod
    def _shift(s: torch.Tensor, lag: int) -> torch.Tensor:
        if lag <= 0:
            return s
        return F.pad(s[:, :-lag], (lag, 0))

    @staticmethod
    def _roll_sum(s: torch.Tensor, window: int) -> torch.Tensor:
        return F.avg_pool1d(F.pad(s.unsqueeze(1), (window - 1, 0)), kernel_size=window, stride=1).squeeze(1) * float(window)

    def _roll_mean(self, s: torch.Tensor, window: int) -> torch.Tensor:
        total = self._roll_sum(s, window)
        t = s.shape[1]
        count = torch.arange(1, t + 1, device=s.device, dtype=s.dtype).clamp(max=window).view(1, -1)
        return total / count

    def _roll_std(self, s: torch.Tensor, window: int) -> torch.Tensor:
        mean = self._roll_mean(s, window)
        mean_sq = self._roll_mean(s * s, window)
        return torch.sqrt(torch.clamp(mean_sq - mean * mean, min=0.0) + 1e-6)

    @staticmethod
    def _roll_max(s: torch.Tensor, window: int) -> torch.Tensor:
        return F.max_pool1d(F.pad(s.unsqueeze(1), (window - 1, 0), value=-1e6), kernel_size=window, stride=1).squeeze(1)

    @staticmethod
    def _roll_min(s: torch.Tensor, window: int) -> torch.Tensor:
        return -F.max_pool1d(F.pad((-s).unsqueeze(1), (window - 1, 0), value=-1e6), kernel_size=window, stride=1).squeeze(1)

    @staticmethod
    def _corr(a: torch.Tensor, b: torch.Tensor, window: int) -> torch.Tensor:
        ma = FactorOperatorBank._roll_sum(a, window)
        mb = FactorOperatorBank._roll_sum(b, window)
        t = a.shape[1]
        count = torch.arange(1, t + 1, device=a.device, dtype=a.dtype).clamp(max=window).view(1, -1)
        ma = ma / count
        mb = mb / count
        cov = FactorOperatorBank._roll_sum(a * b, window) / count - ma * mb
        va = FactorOperatorBank._roll_sum(a * a, window) / count - ma * ma
        vb = FactorOperatorBank._roll_sum(b * b, window) / count - mb * mb
        return cov / (torch.sqrt(torch.clamp(va, min=0.0) + 1e-6) * torch.sqrt(torch.clamp(vb, min=0.0) + 1e-6) + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_x = x
        open_ = self._col(x, "open")
        high = self._col(x, "high")
        low = self._col(x, "low")
        close = self._col(x, "close")
        vol = self._col(x, "log1p_volume")
        amount = self._col(x, "log1p_amount")
        oi = self._col(x, "log1p_oi")
        ret = self._col(x, "log_ret_1")
        range_pct = self._col(x, "range_pct")
        cpos = self._col(x, "close_pos")
        oc_ret = self._col(x, "oc_ret")
        dvol = self._col(x, "d_log1p_volume")
        damt = self._col(x, "d_log1p_amount")
        doi = self._col(x, "d_log1p_oi")

        body = close - open_
        spread = torch.clamp(high - low, min=1e-4)
        bop = body / (spread.abs() + 1e-4)
        upper = high - torch.maximum(open_, close)
        lower = torch.minimum(open_, close) - low
        signed_vol = torch.tanh(ret * 4.0) * vol
        signed_amt = torch.tanh(ret * 4.0) * amount
        turnover = amount - vol
        typ = (high + low + close) / 3.0
        abs_ret = ret.abs()
        ret_pos = torch.relu(ret)
        ret_neg = torch.relu(-ret)

        raw_piece = x if self.include_raw else None
        pieces = []
        pieces.extend(
            [
                body.unsqueeze(-1),
                bop.unsqueeze(-1),
                upper.unsqueeze(-1),
                lower.unsqueeze(-1),
                turnover.unsqueeze(-1),
                (ret * range_pct).unsqueeze(-1),
                (ret * cpos).unsqueeze(-1),
                (ret * dvol).unsqueeze(-1),
                (ret * doi).unsqueeze(-1),
                (range_pct * dvol).unsqueeze(-1),
                (oc_ret * vol).unsqueeze(-1),
                self._shift(ret, 1).unsqueeze(-1),
                self._shift(cpos, 1).unsqueeze(-1),
            ]
        )
        if self.extra_ops:
            pieces.extend(
                [
                    typ.unsqueeze(-1),
                    abs_ret.unsqueeze(-1),
                    (ret * ret).unsqueeze(-1),
                    (torch.sign(ret) * range_pct.abs()).unsqueeze(-1),
                    (dvol - doi).unsqueeze(-1),
                    (turnover * torch.tanh(ret * 4.0)).unsqueeze(-1),
                ]
            )

        window_stats: dict[int, dict[str, torch.Tensor]] = {}
        for w in self.windows:
            mom = self._roll_sum(ret, w)
            ret_std = self._roll_std(ret, w)
            rv = torch.sqrt(torch.clamp(self._roll_mean(ret * ret, w), min=0.0) + 1e-6)
            up = torch.sqrt(torch.clamp(self._roll_mean(torch.relu(ret).pow(2), w), min=0.0) + 1e-6)
            down = torch.sqrt(torch.clamp(self._roll_mean(torch.relu(-ret).pow(2), w), min=0.0) + 1e-6)
            volz = (vol - self._roll_mean(vol, w)) / (self._roll_std(vol, w) + 1e-6)
            amtz = (amount - self._roll_mean(amount, w)) / (self._roll_std(amount, w) + 1e-6)
            oiz = (oi - self._roll_mean(oi, w)) / (self._roll_std(oi, w) + 1e-6)
            oimom = self._roll_sum(doi, w)
            high_w = self._roll_max(high, w)
            low_w = self._roll_min(low, w)
            stoch = (close - low_w) / (high_w - low_w + 1e-4)
            bollz = (close - self._roll_mean(close, w)) / (self._roll_std(close, w) + 1e-6)
            drawup = close - low_w
            drawdn = close - high_w
            maxret = self._roll_max(ret, w)
            minret = self._roll_min(ret, w)
            upratio = self._roll_mean(torch.sigmoid(ret * 8.0), w)
            eff = mom.abs() / (self._roll_sum(ret.abs(), w) + 1e-6)
            pvcorr = self._corr(ret, vol, w)
            pacorr = self._corr(ret, amount, w)
            ret_oi_corr = self._corr(ret, doi, w)
            bop_m = self._roll_mean(bop, w)
            cpos_m = self._roll_mean(cpos, w)
            shadow_imb = self._roll_mean(lower - upper, w)
            shadow_vol_diff = self._roll_std(lower, w) - self._roll_std(upper, w)
            vol_flow = self._roll_sum(signed_vol, w)
            amt_flow = self._roll_sum(signed_amt, w)
            tr_dir = torch.sign(mom) * self._roll_mean(range_pct.abs(), w)
            sharpe = self._roll_mean(ret, w) / (ret_std + 1e-6)
            mfi_proxy = self._roll_sum(torch.relu(ret) * vol, w) / (self._roll_sum(ret.abs() * vol.abs(), w) + 1e-6)
            close_ma = self._roll_mean(close, w)
            typ_ma = self._roll_mean(typ, w)
            cci_proxy = (typ - typ_ma) / (self._roll_std(typ, w) + 1e-6)
            cmo = (self._roll_sum(ret_pos, w) - self._roll_sum(ret_neg, w)) / (
                self._roll_sum(abs_ret, w) + 1e-6
            )

            window_stats[w] = {
                "mom": mom,
                "rv": rv,
                "volz": volz,
                "oimom": oimom,
                "close_ma": close_ma,
                "stoch": stoch,
                "vol_flow": vol_flow,
                "cci": cci_proxy,
            }
            pieces.extend(
                [
                    mom.unsqueeze(-1),
                    sharpe.unsqueeze(-1),
                    upratio.unsqueeze(-1),
                    maxret.unsqueeze(-1),
                    minret.unsqueeze(-1),
                    rv.unsqueeze(-1),
                    up.unsqueeze(-1),
                    down.unsqueeze(-1),
                    ((up - down) / (up + down + 1e-6)).unsqueeze(-1),
                    volz.unsqueeze(-1),
                    amtz.unsqueeze(-1),
                    oiz.unsqueeze(-1),
                    oimom.unsqueeze(-1),
                    stoch.unsqueeze(-1),
                    (1.0 - stoch).unsqueeze(-1),
                    (stoch - self._roll_mean(stoch, min(5, w))).unsqueeze(-1),
                    bollz.unsqueeze(-1),
                    drawup.unsqueeze(-1),
                    drawdn.unsqueeze(-1),
                    eff.unsqueeze(-1),
                    pvcorr.unsqueeze(-1),
                    pacorr.unsqueeze(-1),
                    ret_oi_corr.unsqueeze(-1),
                    bop_m.unsqueeze(-1),
                    cpos_m.unsqueeze(-1),
                    shadow_imb.unsqueeze(-1),
                    vol_flow.unsqueeze(-1),
                    amt_flow.unsqueeze(-1),
                    tr_dir.unsqueeze(-1),
                    (mom * volz).unsqueeze(-1),
                    (mom * oimom).unsqueeze(-1),
                    (rv * volz).unsqueeze(-1),
                    mfi_proxy.unsqueeze(-1),
                ]
            )
            if self.extra_ops:
                close_max = self._roll_max(close, w)
                close_min = self._roll_min(close, w)
                thrust_range = torch.maximum(high_w - close_min, close_max - low_w)
                updn_vol = self._roll_sum(torch.sign(ret) * vol.abs(), w) / (
                    self._roll_sum(vol.abs(), w) + 1e-6
                )
                pieces.extend(
                    [
                        (close - close_ma).unsqueeze(-1),
                        (close - self._shift(close, w)).unsqueeze(-1),
                        torch.sqrt(torch.clamp(self._roll_mean(range_pct * range_pct, w), min=0.0) + 1e-6).unsqueeze(-1),
                        self._roll_mean(range_pct.abs(), w).unsqueeze(-1),
                        cci_proxy.unsqueeze(-1),
                        cmo.unsqueeze(-1),
                        self._corr(ret, self._shift(ret, 1), w).unsqueeze(-1),
                        self._corr(ret, self._shift(ret, 2), w).unsqueeze(-1),
                        (vol - self._shift(vol, w)).unsqueeze(-1),
                        (amount - self._shift(amount, w)).unsqueeze(-1),
                        ((turnover - self._roll_mean(turnover, w)) / (self._roll_std(turnover, w) + 1e-6)).unsqueeze(-1),
                        self._roll_mean(abs_ret / (amount.abs() + 1.0), w).unsqueeze(-1),
                        updn_vol.unsqueeze(-1),
                        (oimom * torch.sign(mom)).unsqueeze(-1),
                        shadow_vol_diff.unsqueeze(-1),
                        self._roll_mean(body.abs() * (cpos - 0.5), w).unsqueeze(-1),
                        ((close - (low_w + 0.5 * thrust_range)) / (thrust_range.abs() + 1e-6)).unsqueeze(-1),
                        self._shift(mom, 1).unsqueeze(-1),
                        self._shift(rv, 1).unsqueeze(-1),
                        self._shift(volz, 1).unsqueeze(-1),
                    ]
                )

        for short, long in self._valid_pairs():
            s = window_stats[short]
            l = window_stats[long]
            trend = s["mom"] - l["mom"]
            rv_regime = s["rv"] / (l["rv"] + 1e-6)
            pieces.extend(
                [
                    trend.unsqueeze(-1),
                    (s["volz"] - l["volz"]).unsqueeze(-1),
                    rv_regime.unsqueeze(-1),
                    (trend * l["volz"]).unsqueeze(-1),
                ]
            )
            if self.extra_ops:
                pieces.extend(
                    [
                        (s["close_ma"] - l["close_ma"]).unsqueeze(-1),
                        (s["stoch"] - l["stoch"]).unsqueeze(-1),
                        (s["vol_flow"] - l["vol_flow"]).unsqueeze(-1),
                        (s["cci"] * l["rv"]).unsqueeze(-1),
                    ]
                )

        ops = torch.cat(pieces, dim=-1)
        ops = torch.nan_to_num(ops, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)
        ops = torch.clamp(ops, -self.clip_value, self.clip_value)
        if self.gate_mode == "static":
            gate = self._apply_operator_budget(self.operator_logits).view(1, 1, -1)
            self.last_gate = gate
            ops = ops * gate
        elif self.gate_mode == "dynamic":
            gate = self._apply_operator_budget(self.gate_net(raw_x))
            self.last_gate = gate
            ops = ops * gate
        elif self.gate_mode in {"sequence", "seq", "sample"}:
            last = raw_x[:, -1, :]
            mean = raw_x.mean(dim=1)
            std = raw_x.std(dim=1, unbiased=False)
            gate = self._apply_operator_budget(self.gate_net(torch.cat([last, mean, std], dim=-1))).unsqueeze(1)
            self.last_gate = gate
            ops = ops * gate
        if self.projector is not None:
            ops = self.projector(ops)
        out = torch.cat([raw_piece, ops], dim=-1) if raw_piece is not None else ops
        return self.dropout(out)

    def regularization_loss(self) -> torch.Tensor:
        if self.gate_mode == "static" and self.operator_logits is not None:
            return torch.sigmoid(self.operator_logits).mean()
        if self.last_gate is not None:
            return self.last_gate.mean()
        return next(self.parameters()).new_tensor(0.0)

    def smoothness_loss(self) -> torch.Tensor:
        if self.last_gate is None or self.last_gate.ndim < 3 or self.last_gate.shape[1] <= 1:
            return next(self.parameters()).new_tensor(0.0)
        return (self.last_gate[:, 1:] - self.last_gate[:, :-1]).abs().mean()

    def binary_loss(self) -> torch.Tensor:
        if self.last_gate is None:
            return next(self.parameters()).new_tensor(0.0)
        gate = torch.clamp(self.last_gate, 0.0, 1.0)
        return (gate * (1.0 - gate)).mean()


class MoEPredictionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        output_dim: int,
        n_experts: int,
        hidden_dim: int,
        dropout: float,
        temperature: float = 1.0,
        expert_hidden_dim: int | None = None,
        expert_mid_dim: int | None = None,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        self.n_experts = max(1, int(n_experts))
        self.temperature = max(1e-3, float(temperature))
        expert_hidden = max(16, int(expert_hidden_dim or d_model))
        expert_mid = max(16, int(expert_mid_dim or max(16, d_model // 2)))
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(16, hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_dim), self.n_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, expert_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, expert_mid),
                    nn.GELU(),
                    nn.Linear(expert_mid, output_dim),
                )
                for _ in range(self.n_experts)
            ]
        )
        if zero_init_output:
            for expert in self.experts:
                out = expert[-1]
                if isinstance(out, nn.Linear):
                    nn.init.zeros_(out.weight)
                    nn.init.zeros_(out.bias)
        self.last_router_probs: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.router(x) / self.temperature, dim=-1)
        self.last_router_probs = probs
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.sum(expert_out * probs.unsqueeze(-1), dim=1)

    def load_balance_loss(self) -> torch.Tensor:
        if self.last_router_probs is None:
            return next(self.parameters()).new_tensor(0.0)
        load = self.last_router_probs.mean(dim=0)
        target = load.new_full(load.shape, 1.0 / float(self.n_experts))
        return ((load - target).pow(2).sum() * float(self.n_experts))


class ResidualMoEPredictionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        output_dim: int,
        base_n_experts: int,
        base_hidden_dim: int,
        residual_n_experts: int,
        residual_hidden_dim: int,
        dropout: float,
        temperature: float = 1.0,
        residual_expert_hidden_dim: int | None = None,
        residual_expert_mid_dim: int | None = None,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.base = MoEPredictionHead(
            d_model=d_model,
            output_dim=output_dim,
            n_experts=base_n_experts,
            hidden_dim=base_hidden_dim,
            dropout=dropout,
            temperature=temperature,
        )
        self.residual = MoEPredictionHead(
            d_model=d_model,
            output_dim=output_dim,
            n_experts=residual_n_experts,
            hidden_dim=residual_hidden_dim,
            dropout=dropout,
            temperature=temperature,
            expert_hidden_dim=residual_expert_hidden_dim,
            expert_mid_dim=residual_expert_mid_dim,
            zero_init_output=True,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.last_base_output: torch.Tensor | None = None
        self.last_residual_output: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        residual = self.residual(x)
        self.last_base_output = base
        self.last_residual_output = residual
        return base + self.residual_scale * residual

    def load_balance_loss(self) -> torch.Tensor:
        return self.base.load_balance_loss() + self.residual.load_balance_loss()

    def residual_output_penalty(self) -> torch.Tensor:
        if self.last_residual_output is None:
            return next(self.parameters()).new_tensor(0.0)
        return self.last_residual_output.pow(2).mean()


class AdditiveDecomposedMoEHead(nn.Module):
    """Preserve a stable base forecast while learning market/residual add-ons."""

    def __init__(
        self,
        d_model: int,
        base_n_experts: int,
        base_hidden_dim: int,
        addon_n_experts: int,
        addon_hidden_dim: int,
        dropout: float,
        temperature: float = 1.0,
        addon_expert_hidden_dim: int | None = None,
        addon_expert_mid_dim: int | None = None,
        market_scale_init: float = 1.0,
        residual_scale_init: float = 1.0,
        component_gate: bool = False,
    ) -> None:
        super().__init__()
        self.base = MoEPredictionHead(
            d_model=d_model,
            output_dim=1,
            n_experts=base_n_experts,
            hidden_dim=base_hidden_dim,
            dropout=dropout,
            temperature=temperature,
        )
        self.market = MoEPredictionHead(
            d_model=d_model,
            output_dim=1,
            n_experts=addon_n_experts,
            hidden_dim=addon_hidden_dim,
            dropout=dropout,
            temperature=temperature,
            expert_hidden_dim=addon_expert_hidden_dim,
            expert_mid_dim=addon_expert_mid_dim,
            zero_init_output=True,
        )
        self.residual = MoEPredictionHead(
            d_model=d_model,
            output_dim=1,
            n_experts=addon_n_experts,
            hidden_dim=addon_hidden_dim,
            dropout=dropout,
            temperature=temperature,
            expert_hidden_dim=addon_expert_hidden_dim,
            expert_mid_dim=addon_expert_mid_dim,
            zero_init_output=True,
        )
        self.market_scale = nn.Parameter(torch.tensor(float(market_scale_init)))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.component_gate = (
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, max(16, addon_hidden_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(16, addon_hidden_dim), 2),
            )
            if component_gate
            else None
        )
        self.last_market_output: torch.Tensor | None = None
        self.last_residual_output: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x).squeeze(-1)
        market = self.market(x).squeeze(-1) * self.market_scale
        residual = self.residual(x).squeeze(-1) * self.residual_scale
        if self.component_gate is not None:
            gate = 2.0 * torch.sigmoid(self.component_gate(x))
            market = market * gate[:, 0]
            residual = residual * gate[:, 1]
        self.last_market_output = market
        self.last_residual_output = residual
        final = base + market + residual
        return torch.stack([final, market, residual], dim=1)

    def load_balance_loss(self) -> torch.Tensor:
        return self.base.load_balance_loss() + self.market.load_balance_loss() + self.residual.load_balance_loss()

    def residual_output_penalty(self) -> torch.Tensor:
        loss = next(self.parameters()).new_tensor(0.0)
        if self.last_market_output is not None:
            loss = loss + self.last_market_output.pow(2).mean()
        if self.last_residual_output is not None:
            loss = loss + self.last_residual_output.pow(2).mean()
        return loss


class MultiPoolingFusion(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        modes: list[str] | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.modes = [str(m).lower() for m in (modes or ["attention", "mean", "last", "max", "std"])]
        if not self.modes:
            raise ValueError("MultiPoolingFusion requires at least one mode")
        self.attn_pool = AttentionPooling(d_model, dropout=dropout) if "attention" in self.modes else None
        in_dim = d_model * len(self.modes)
        hidden = max(d_model, int(hidden_dim or d_model * 2))
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        for mode in self.modes:
            if mode == "attention":
                if self.attn_pool is None:
                    raise RuntimeError("attention pooling requested but not initialized")
                pieces.append(self.attn_pool(x))
            elif mode == "mean":
                pieces.append(x.mean(dim=1))
            elif mode == "last":
                pieces.append(x[:, -1])
            elif mode == "max":
                pieces.append(x.max(dim=1).values)
            elif mode == "std":
                pieces.append(x.std(dim=1, unbiased=False))
            else:
                raise ValueError(f"unknown fusion pooling mode: {mode}")
        return self.proj(torch.cat(pieces, dim=-1))


class FeatureTokenizerBlock(nn.Module):
    """FT-Transformer-style feature tokens at each minute."""

    def __init__(
        self,
        n_features: int,
        out_dim: int,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.out_dim = int(out_dim)
        self.value_weight = nn.Parameter(torch.empty(n_features, out_dim))
        self.value_bias = nn.Parameter(torch.zeros(n_features, out_dim))
        self.feature_embed = nn.Parameter(torch.empty(n_features, out_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=n_heads,
            dim_feedforward=out_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(n_layers)))
        self.pool_score = nn.Linear(out_dim, 1)
        self.norm = nn.LayerNorm(out_dim)
        nn.init.normal_(self.value_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.feature_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_features = x.shape
        tok = x.unsqueeze(-1) * self.value_weight.view(1, 1, n_features, self.out_dim)
        tok = tok + self.value_bias.view(1, 1, n_features, self.out_dim)
        tok = tok + self.feature_embed.view(1, 1, n_features, self.out_dim)
        tok = tok.reshape(bsz * seq_len, n_features, self.out_dim)
        tok = self.encoder(tok)
        weights = torch.softmax(self.pool_score(tok).squeeze(-1), dim=-1)
        pooled = torch.sum(tok * weights.unsqueeze(-1), dim=1).reshape(bsz, seq_len, self.out_dim)
        return self.norm(pooled)


class HybridInputBlock(nn.Module):
    def __init__(
        self,
        n_features: int,
        out_dim: int,
        rank: int,
        token_heads: int,
        token_layers: int,
        dropout: float,
        mode: str,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.lowrank = (
            LowRankFeatureInteraction(
                n_features,
                out_dim,
                rank=rank,
                dropout=dropout,
                gated=mode not in {"lowrank_ungated", "hybrid_ungated"},
            )
            if mode in {"lowrank", "hybrid", "gated_hybrid", "lowrank_ungated", "hybrid_ungated"}
            else None
        )
        self.tokenizer = (
            FeatureTokenizerBlock(
                n_features=n_features,
                out_dim=out_dim,
                n_heads=token_heads,
                n_layers=token_layers,
                dropout=dropout,
            )
            if mode in {"feature_token", "hybrid", "gated_hybrid", "hybrid_ungated"}
            else None
        )
        self.gate = (
            nn.Sequential(nn.Linear(n_features, out_dim), nn.Sigmoid()) if mode == "gated_hybrid" else None
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate is not None:
            if self.lowrank is None or self.tokenizer is None:
                raise ValueError("gated_hybrid requires both lowrank and tokenizer")
            low = self.lowrank(x)
            tok = self.tokenizer(x)
            gate = self.gate(x)
            return self.norm((1.0 - gate) * low + gate * tok)
        pieces = []
        if self.lowrank is not None:
            pieces.append(self.lowrank(x))
        if self.tokenizer is not None:
            pieces.append(self.tokenizer(x))
        if not pieces:
            raise ValueError(f"unknown input block mode: {self.mode}")
        return self.norm(torch.stack(pieces, dim=0).sum(dim=0))


class StreamInputBlock(nn.Module):
    """Learn stream-specific interactions before gated fusion."""

    def __init__(
        self,
        n_features: int,
        out_dim: int,
        groups: list[list[int]],
        rank: int,
        dropout: float,
        encoder: str = "lowrank",
    ) -> None:
        super().__init__()
        if not groups:
            raise ValueError("stream input block requires at least one feature group")
        clean_groups = []
        for group in groups:
            idx = sorted({int(i) for i in group if 0 <= int(i) < n_features})
            if idx:
                clean_groups.append(idx)
        if not clean_groups:
            raise ValueError("stream input block received no valid feature indices")
        self.groups = clean_groups
        encoder = (encoder or "lowrank").lower()
        if encoder == "lowrank":
            self.streams = nn.ModuleList(
                [
                    LowRankFeatureInteraction(
                        len(group),
                        out_dim,
                        rank=max(1, min(int(rank), len(group))),
                        dropout=dropout,
                        gated=True,
                    )
                    for group in clean_groups
                ]
            )
        elif encoder == "linear":
            self.streams = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(len(group), out_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    for group in clean_groups
                ]
            )
        else:
            raise ValueError(f"unknown stream encoder: {encoder}")
        self.gate = nn.Sequential(nn.Linear(n_features, len(clean_groups)), nn.Softmax(dim=-1))
        self.residual = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = [stream(x[:, :, group]) for stream, group in zip(self.streams, self.groups)]
        stack = torch.stack(pieces, dim=2)
        weights = self.gate(x).unsqueeze(-1)
        fused = (stack * weights).sum(dim=2)
        return self.norm(self.dropout(fused) + self.residual(x))


class EndToEndTransformerBaseline(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_symbols: int,
        seq_len: int = 120,
        interaction_dim: int = 64,
        interaction_rank: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 256,
        patch_len: int = 12,
        stride: int = 6,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pooling: str = "attention",
        patch_encoder: str = "mlp",
        use_cls_token: bool = False,
        input_block: str = "lowrank",
        feature_token_heads: int = 4,
        feature_token_layers: int = 1,
        multi_patch_scales: list[tuple[int, int]] | None = None,
        aux_output_dim: int = 0,
        stream_groups: list[list[int]] | None = None,
        stream_encoder: str = "lowrank",
        feature_gate: bool = False,
        feature_gate_dropout: float = 0.0,
        latent_slots: int = 0,
        latent_mode: str = "replace",
        moe_n_experts: int = 1,
        moe_hidden_dim: int | None = None,
        moe_temperature: float = 1.0,
        feature_cols: list[str] | None = None,
        factor_operator_bank: bool = False,
        factor_operator_windows: list[int] | None = None,
        factor_operator_dropout: float = 0.0,
        factor_operator_clip: float = 12.0,
        factor_operator_projection_dim: int = 0,
        factor_operator_gate: str = "none",
        factor_operator_topk: int = 0,
        factor_operator_topk_mode: str = "hard",
        factor_operator_soft_temperature: float = 1.0,
        factor_operator_gate_hidden: int = 64,
        factor_operator_extra_ops: bool = False,
        ffn_type: str = "glu",
        layer_agg_mode: str = "learned",
        decomposed_head: bool = False,
        additive_decomposed_head: bool = False,
        moe_expert_hidden_dim: int | None = None,
        moe_expert_mid_dim: int | None = None,
        moe_base_n_experts: int = 0,
        moe_base_hidden_dim: int | None = None,
        moe_residual_scale_init: float = 1.0,
        decomp_market_scale_init: float = 1.0,
        decomp_residual_scale_init: float = 1.0,
        decomp_component_gate: bool = False,
        regime_conditioning: bool = False,
        regime_hidden_dim: int = 128,
        regime_scale_init: float = 0.1,
        regime_target: str = "both",
        time_bucket_minutes: int = 0,
        minute_encoding: str = "embedding",
        minute_harmonics: int = 4,
        use_month_embedding: bool = True,
        symbol_embedding_dim: int | None = None,
        pool_fusion_modes: list[str] | None = None,
        pool_fusion_hidden_dim: int | None = None,
        head_type: str = "default",
        detach_aux_head_input: bool = False,
        raw_xsz_scale_init: float = 0.25,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.use_cls_token = bool(use_cls_token)
        self.layer_agg_mode = str(layer_agg_mode or "learned").lower()
        self.head_type = str(head_type or "default").lower()
        self.detach_aux_head_input = bool(detach_aux_head_input)
        self.decomposed_head = bool(decomposed_head)
        self.additive_decomposed_head = bool(additive_decomposed_head)
        if self.additive_decomposed_head:
            self.decomposed_head = True
        self.output_dim = 3 if self.decomposed_head else 1 + max(0, int(aux_output_dim))
        head_output_dim = 2 if self.decomposed_head else self.output_dim
        self.raw_xsz_blend_head = self.head_type in {"raw_xsz_blend", "raw_xsz_residual"}
        self.raw_xsz_blend_scale = nn.Parameter(torch.tensor(float(raw_xsz_scale_init)))
        self.latent_slots = max(0, int(latent_slots))
        self.latent_mode = (latent_mode or "replace").lower()
        self.operator_bank = (
            FactorOperatorBank(
                n_features=n_features,
                feature_cols=feature_cols,
                windows=factor_operator_windows,
                dropout=factor_operator_dropout,
                clip_value=factor_operator_clip,
                include_raw=True,
                projection_dim=factor_operator_projection_dim,
                gate_mode=factor_operator_gate,
                topk=factor_operator_topk,
                topk_mode=factor_operator_topk_mode,
                soft_topk_temperature=factor_operator_soft_temperature,
                gate_hidden=factor_operator_gate_hidden,
                extra_ops=factor_operator_extra_ops,
            )
            if factor_operator_bank
            else None
        )
        effective_n_features = self.operator_bank.output_dim if self.operator_bank is not None else n_features
        self.feature_gate = StaticFeatureGate(effective_n_features, dropout=feature_gate_dropout) if feature_gate else None
        self.regime_target = str(regime_target or "both").lower()
        self.regime_conditioner = (
            RegimeConditioner(
                n_features=effective_n_features,
                d_model=d_model,
                hidden_dim=regime_hidden_dim,
                dropout=dropout,
                scale_init=regime_scale_init,
            )
            if regime_conditioning
            else None
        )
        self.interaction = (
            StreamInputBlock(
                n_features=effective_n_features,
                out_dim=interaction_dim,
                groups=stream_groups or [],
                rank=interaction_rank,
                dropout=dropout,
                encoder=stream_encoder,
            )
            if input_block == "stream"
            else HybridInputBlock(
                n_features=effective_n_features,
                out_dim=interaction_dim,
                mode=input_block,
                rank=interaction_rank,
                token_heads=feature_token_heads,
                token_layers=feature_token_layers,
                dropout=dropout,
            )
        )
        self.patch_embed = (
            MultiScalePatchEmbedding(
                input_dim=interaction_dim,
                d_model=d_model,
                seq_len=seq_len,
                scales=multi_patch_scales,
                dropout=dropout,
                encoder=patch_encoder,
            )
            if multi_patch_scales
            else PatchEmbedding(
                input_dim=interaction_dim,
                d_model=d_model,
                seq_len=seq_len,
                patch_len=patch_len,
                stride=stride,
                dropout=dropout,
                encoder=patch_encoder,
            )
        )
        symbol_dim = int(symbol_embedding_dim or d_model)
        self.symbol_embed = nn.Embedding(n_symbols, symbol_dim)
        self.symbol_proj = nn.Linear(symbol_dim, d_model, bias=False) if symbol_dim != d_model else None
        self.minute_encoding = str(minute_encoding or "embedding").lower()
        self.minute_harmonics = max(1, int(minute_harmonics or 1))
        self.minute_embed = nn.Embedding(24 * 60, d_model) if self.minute_encoding == "embedding" else None
        self.minute_harmonic_proj = (
            nn.Linear(2 * self.minute_harmonics, d_model, bias=False)
            if self.minute_encoding in {"harmonic", "sinusoidal", "periodic"}
            else None
        )
        self.dayofweek_embed = nn.Embedding(7, d_model)
        self.month_embed = nn.Embedding(12, d_model) if use_month_embedding else None
        self.time_bucket_minutes = max(0, int(time_bucket_minutes or 0))
        self.time_bucket_embed = (
            nn.Embedding((24 * 60 + self.time_bucket_minutes - 1) // self.time_bucket_minutes, d_model)
            if self.time_bucket_minutes > 0
            else None
        )
        self.latent_tokens = nn.Parameter(torch.zeros(1, self.latent_slots, d_model)) if self.latent_slots > 0 else None
        self.latent_norm = nn.LayerNorm(d_model) if self.latent_tokens is not None else None
        self.token_norm = nn.LayerNorm(d_model) if self.latent_tokens is not None else None
        self.latent_cross_attn = (
            nn.MultiheadAttention(d_model, n_heads, dropout=attention_dropout, batch_first=True)
            if self.latent_tokens is not None
            else None
        )
        self.latent_drop = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if self.use_cls_token else None
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    residual_scale=1.0,
                    ffn_type=ffn_type,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_agg = LayerOutputAggregation(n_layers) if self.layer_agg_mode == "learned" else None
        self.norm = nn.LayerNorm(d_model)
        self.attn_pool = AttentionPooling(d_model, dropout=dropout)
        self.fusion_pool = (
            MultiPoolingFusion(
                d_model=d_model,
                dropout=dropout,
                modes=pool_fusion_modes,
                hidden_dim=pool_fusion_hidden_dim,
            )
            if self.pooling == "fusion"
            else None
        )
        base_experts = int(moe_base_n_experts or 0)
        self.moe_head = (
            AdditiveDecomposedMoEHead(
                d_model=d_model,
                base_n_experts=max(1, base_experts),
                base_hidden_dim=int(moe_base_hidden_dim or max(16, d_model // 2)),
                addon_n_experts=int(moe_n_experts),
                addon_hidden_dim=int(moe_hidden_dim or max(16, d_model // 2)),
                dropout=dropout,
                temperature=moe_temperature,
                addon_expert_hidden_dim=moe_expert_hidden_dim,
                addon_expert_mid_dim=moe_expert_mid_dim,
                market_scale_init=decomp_market_scale_init,
                residual_scale_init=decomp_residual_scale_init,
                component_gate=decomp_component_gate,
            )
            if self.additive_decomposed_head and int(moe_n_experts) > 0
            else
            ResidualMoEPredictionHead(
                d_model=d_model,
                output_dim=head_output_dim,
                base_n_experts=base_experts,
                base_hidden_dim=int(moe_base_hidden_dim or max(16, d_model // 2)),
                residual_n_experts=int(moe_n_experts),
                residual_hidden_dim=int(moe_hidden_dim or max(16, d_model // 2)),
                dropout=dropout,
                temperature=moe_temperature,
                residual_expert_hidden_dim=moe_expert_hidden_dim,
                residual_expert_mid_dim=moe_expert_mid_dim,
                residual_scale_init=moe_residual_scale_init,
            )
            if base_experts > 0 and int(moe_n_experts) > 0
            else MoEPredictionHead(
                d_model=d_model,
                output_dim=head_output_dim,
                n_experts=int(moe_n_experts),
                hidden_dim=int(moe_hidden_dim or max(16, d_model // 2)),
                dropout=dropout,
                temperature=moe_temperature,
                expert_hidden_dim=moe_expert_hidden_dim,
                expert_mid_dim=moe_expert_mid_dim,
            )
            if int(moe_n_experts) > 1
            else None
        )
        if self.moe_head is not None:
            self.head = None
            self.aux_head = None
        elif self.raw_xsz_blend_head:
            aux_dim = max(1, int(aux_output_dim))
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1 + aux_dim),
            )
            self.aux_head = None
        elif self.head_type in {"small_mlp_separate_aux", "review_small_mlp_separate_aux"}:
            aux_dim = max(0, int(aux_output_dim))
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            self.aux_head = (
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, max(16, d_model // 2)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(16, d_model // 2), aux_dim),
                )
                if aux_dim > 0
                else None
            )
        elif self.head_type in {"small_mlp", "review_small_mlp"}:
            self.aux_head = None
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, head_output_dim),
            )
        else:
            self.aux_head = None
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, max(16, d_model // 2)),
                nn.GELU(),
                nn.Linear(max(16, d_model // 2), head_output_dim),
            )
        if self.latent_tokens is not None:
            nn.init.trunc_normal_(self.latent_tokens, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        if self.operator_bank is not None:
            x = self.operator_bank(x)
        if self.feature_gate is not None:
            x = self.feature_gate(x)
        regime = self.regime_conditioner(x) if self.regime_conditioner is not None else None
        h = self.interaction(x)
        tok = self.patch_embed(h)
        minute = time_ids[:, 0].clamp(0, 24 * 60 - 1)
        dow = time_ids[:, 1].clamp(0, 6)
        month = time_ids[:, 2].clamp(0, 11)
        symbol_meta = self.symbol_embed(symbol_id)
        if self.symbol_proj is not None:
            symbol_meta = self.symbol_proj(symbol_meta)
        if self.minute_harmonic_proj is not None:
            phase = minute.to(dtype=torch.float32) * (2.0 * torch.pi / float(24 * 60))
            freqs = torch.arange(
                1,
                self.minute_harmonics + 1,
                device=phase.device,
                dtype=phase.dtype,
            )
            angles = phase.unsqueeze(-1) * freqs.unsqueeze(0)
            minute_meta = self.minute_harmonic_proj(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1))
        elif self.minute_embed is not None:
            minute_meta = self.minute_embed(minute)
        else:
            minute_meta = torch.zeros_like(symbol_meta)
        meta = symbol_meta + minute_meta + self.dayofweek_embed(dow)
        if self.month_embed is not None:
            meta = meta + self.month_embed(month)
        if self.time_bucket_embed is not None:
            bucket = (minute // self.time_bucket_minutes).clamp(0, self.time_bucket_embed.num_embeddings - 1)
            meta = meta + self.time_bucket_embed(bucket)
        if regime is not None and self.regime_target in {"meta", "both"}:
            meta = meta + regime
        tok = tok + meta.unsqueeze(1)
        if self.latent_tokens is not None:
            lat = self.latent_tokens.expand(tok.shape[0], -1, -1) + meta.unsqueeze(1)
            query = self.latent_norm(lat)
            key_value = self.token_norm(tok)
            cross, _ = self.latent_cross_attn(query, key_value, key_value, need_weights=False)
            lat = lat + self.latent_drop(cross)
            if self.latent_mode == "append":
                tok = torch.cat([tok, lat], dim=1)
            elif self.latent_mode == "replace":
                tok = lat
            else:
                raise ValueError(f"unknown latent_mode: {self.latent_mode}")
        elif self.cls_token is not None:
            cls = self.cls_token.expand(tok.shape[0], -1, -1) + meta.unsqueeze(1)
            tok = torch.cat([tok, cls], dim=1)
        hidden_states = []
        for block in self.blocks:
            tok = block(tok)
            hidden_states.append(tok)
        if self.layer_agg is not None:
            tok = self.layer_agg(hidden_states)
        elif self.layer_agg_mode == "mean":
            tok = torch.stack(hidden_states, dim=0).mean(dim=0)
        elif self.layer_agg_mode == "last":
            tok = hidden_states[-1]
        else:
            raise ValueError(f"unknown layer_agg_mode: {self.layer_agg_mode}")
        tok = self.norm(tok)
        if self.pooling == "mean":
            pooled = tok.mean(dim=1)
        elif self.pooling == "cls":
            pooled = tok[:, -1] if self.cls_token is not None else tok[:, -1]
        elif self.pooling == "attention":
            pooled = self.attn_pool(tok)
        elif self.pooling == "fusion":
            if self.fusion_pool is None:
                raise RuntimeError("fusion pooling requested but fusion_pool was not initialized")
            pooled = self.fusion_pool(tok)
        else:
            raise ValueError(f"unknown pooling: {self.pooling}")
        if regime is not None and self.regime_target in {"pooled", "both"}:
            pooled = pooled + regime
        out = self.moe_head(pooled) if self.moe_head is not None else self.head(pooled)
        if self.raw_xsz_blend_head:
            if out.ndim != 2 or out.shape[1] < 2:
                raise RuntimeError("raw_xsz_blend head requires at least one auxiliary output")
            raw_base = out[:, 0]
            aux = out[:, 1:]
            final = raw_base + self.raw_xsz_blend_scale * aux[:, 0]
            out = torch.cat([final.unsqueeze(-1), aux], dim=-1)
        if self.aux_head is not None:
            aux_in = pooled.detach() if self.detach_aux_head_input else pooled
            out = torch.cat([out, self.aux_head(aux_in)], dim=-1)
        if self.additive_decomposed_head:
            return out
        if self.decomposed_head:
            market = out[:, 0]
            residual = out[:, 1]
            final = market + residual
            return torch.stack([final, market, residual], dim=1)
        return out.squeeze(-1) if self.output_dim == 1 else out

    def regularization_loss(self) -> torch.Tensor:
        loss = next(self.parameters()).new_tensor(0.0)
        if self.feature_gate is not None:
            loss = loss + self.feature_gate.l1_penalty()
        if self.operator_bank is not None:
            loss = loss + self.operator_bank.regularization_loss()
        return loss

    def operator_regularization_loss(self) -> torch.Tensor:
        if self.operator_bank is None:
            return next(self.parameters()).new_tensor(0.0)
        return self.operator_bank.regularization_loss()

    def operator_smoothness_loss(self) -> torch.Tensor:
        if self.operator_bank is None:
            return next(self.parameters()).new_tensor(0.0)
        return self.operator_bank.smoothness_loss()

    def operator_binary_loss(self) -> torch.Tensor:
        if self.operator_bank is None:
            return next(self.parameters()).new_tensor(0.0)
        return self.operator_bank.binary_loss()

    def moe_regularization_loss(self) -> torch.Tensor:
        if self.moe_head is None:
            return next(self.parameters()).new_tensor(0.0)
        return self.moe_head.load_balance_loss()

    def moe_residual_output_penalty(self) -> torch.Tensor:
        if self.moe_head is None or not hasattr(self.moe_head, "residual_output_penalty"):
            return next(self.parameters()).new_tensor(0.0)
        return self.moe_head.residual_output_penalty()


def ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(pred) & torch.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    pred = pred - pred.mean()
    target = target - target.mean()
    num = (pred * target).mean()
    den = torch.sqrt(pred.pow(2).mean() + 1e-8) * torch.sqrt(target.pow(2).mean() + 1e-8)
    return -num / (den + 1e-8)


def combo_loss(pred: torch.Tensor, target: torch.Tensor, mse_weight: float = 0.35) -> torch.Tensor:
    return mse_weight * F.mse_loss(pred, target) + (1.0 - mse_weight) * ic_loss(pred, target)
