"""
Plan A model zoo: neural tabular/sequence models on engineered factors.
All trained with differentiable IC loss, monthly rolling.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ic_loss(pred, target):
    pred = pred - pred.mean()
    target = target - target.mean()
    num = (pred * target).mean()
    denom = torch.sqrt(pred.pow(2).mean() + 1e-8) * torch.sqrt(target.pow(2).mean() + 1e-8) + 1e-8
    return -num / denom


# ----------------------------------------------------------------------------
# ResMLP — residual MLP, strong on tabular factors
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ResMLP(nn.Module):
    def __init__(self, n_features, dim=256, n_blocks=4, dropout=0.1, n_symbols=0, emb_dim=16):
        super().__init__()
        self.use_emb = n_symbols > 0
        in_dim = n_features + (emb_dim if self.use_emb else 0)
        if self.use_emb:
            self.emb = nn.Embedding(n_symbols, emb_dim)
        self.proj = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([ResBlock(dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x, sym=None):
        if self.use_emb and sym is not None:
            x = torch.cat([x, self.emb(sym)], dim=-1)
        h = self.proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(-1)


# ----------------------------------------------------------------------------
# FT-Transformer — each factor becomes a token, self-attention over factors
# ----------------------------------------------------------------------------
class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=64, n_layers=3, n_heads=8, dropout=0.1, n_symbols=0):
        super().__init__()
        self.n_features = n_features
        # Per-feature linear embedding: scalar -> d_token
        self.feat_embed = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.feat_bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        layer = nn.TransformerEncoderLayer(d_token, n_heads, d_token * 2, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

    def forward(self, x, sym=None):
        B = x.shape[0]
        # x: (B, n_features) -> tokens (B, n_features, d_token)
        tokens = x.unsqueeze(-1) * self.feat_embed.unsqueeze(0) + self.feat_bias.unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        h = self.encoder(tokens)
        return self.head(h[:, 0]).squeeze(-1)


# ----------------------------------------------------------------------------
# SeqTransformer — sequence of factor vectors over a time window
# ----------------------------------------------------------------------------
class SeqTransformer(nn.Module):
    def __init__(self, n_features, seq_len, d_model=128, n_layers=3, n_heads=8, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x, sym=None):
        # x: (B, seq_len, n_features)
        h = self.proj(x) + self.pos[:, :x.shape[1]]
        h = self.encoder(h)
        return self.head(h[:, -1]).squeeze(-1)


# ----------------------------------------------------------------------------
# SeqGRU — causal recurrent model over short factor windows
# ----------------------------------------------------------------------------
class SeqGRU(nn.Module):
    def __init__(self, n_features, d_model=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_features, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.gru = nn.GRU(
            d_model,
            d_model,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, sym=None):
        h = self.inp(x)
        h, _ = self.gru(h)
        return self.head(h[:, -1]).squeeze(-1)


# ----------------------------------------------------------------------------
# ALiBiCausalTransformer — advanced sequence model over factor windows
#   ALiBi linear-bias attention + causal mask + SwiGLU FFN + layer fusion
#   + last-token attention pooling + optimized residual head
# ----------------------------------------------------------------------------
def _alibi_slopes(n_heads):
    import math
    def slopes_pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]
    if math.log2(n_heads).is_integer():
        return slopes_pow2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    s = slopes_pow2(closest)
    extra = _alibi_slopes(2 * closest)[0::2][: n_heads - closest]
    return s + extra


class _SwiGLU(nn.Module):
    def __init__(self, d, mult=4, dropout=0.1):
        super().__init__()
        hid = int(d * mult)
        self.w1 = nn.Linear(d, hid); self.w2 = nn.Linear(d, hid)
        self.w3 = nn.Linear(hid, d); self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))


class _ALiBiAttention(nn.Module):
    def __init__(self, d, heads, dropout=0.1):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("slopes", torch.tensor(_alibi_slopes(heads), dtype=torch.float32))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,h,T,hd)
        scores = (q @ k.transpose(-2, -1)) / (self.hd ** 0.5)  # (B,h,T,T)
        pos = torch.arange(T, device=x.device)
        rel = pos[None, :] - pos[:, None]            # (T,T), <=0 in past
        alibi = self.slopes.view(1, self.h, 1, 1) * rel.view(1, 1, T, T)
        scores = scores + alibi
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = self.drop(scores.softmax(-1))
        o = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(o)


class _Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = _ALiBiAttention(d, heads, dropout)
        self.n2 = nn.LayerNorm(d); self.ff = _SwiGLU(d, 4, dropout)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x


class ALiBiCausalTransformer(nn.Module):
    def __init__(self, n_features, seq_len, d_model=192, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_features, d_model), nn.LayerNorm(d_model))
        self.blocks = nn.ModuleList([_Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        # Layer fusion: learned softmax weights over per-layer outputs
        self.layer_w = nn.Parameter(torch.zeros(n_layers + 1))
        self.fuse_norm = nn.LayerNorm(d_model)
        # Last-token attention pooling: last token queries the whole sequence
        self.pool_q = nn.Linear(d_model, d_model)
        self.pool_k = nn.Linear(d_model, d_model)
        self.pool_v = nn.Linear(d_model, d_model)
        # Optimized residual head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, sym=None):
        h = self.inp(x)                       # (B,T,d)
        states = [h]
        for blk in self.blocks:
            h = blk(h); states.append(h)
        w = torch.softmax(self.layer_w, 0)
        fused = sum(wi * s for wi, s in zip(w, states))  # (B,T,d)
        fused = self.fuse_norm(fused)
        # attention pooling with last token as query
        q = self.pool_q(fused[:, -1:])        # (B,1,d)
        k = self.pool_k(fused); v = self.pool_v(fused)
        a = torch.softmax((q @ k.transpose(-2, -1)) / (fused.shape[-1] ** 0.5), -1)
        pooled = (a @ v).squeeze(1)           # (B,d)
        return self.head(pooled).squeeze(-1)


def build_model(name, n_features, cfg, n_symbols=0, seq_len=None):
    if name == "alibi_transformer":
        return ALiBiCausalTransformer(
            n_features, seq_len,
            d_model=cfg.get("plan_a_d_model", 192),
            n_layers=cfg.get("plan_a_n_layers", 6),
            n_heads=8, dropout=0.1)
    if name == "resmlp":
        return ResMLP(n_features, dim=256, n_blocks=4, dropout=0.1)
    if name == "resmlp_embed":
        return ResMLP(n_features, dim=256, n_blocks=4, dropout=0.1, n_symbols=n_symbols)
    if name == "ft_transformer":
        return FTTransformer(n_features, d_token=64, n_layers=3, n_heads=8, dropout=0.1)
    if name == "seq_transformer":
        return SeqTransformer(n_features, seq_len, d_model=128, n_layers=3, n_heads=8, dropout=0.1)
    if name == "seq_gru":
        return SeqGRU(n_features, d_model=128, n_layers=2, dropout=0.1)
    raise ValueError(name)
