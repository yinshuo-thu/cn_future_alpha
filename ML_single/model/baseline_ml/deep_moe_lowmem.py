#!/usr/bin/env python3
"""Low-memory anchored deep MOE over strict OOS component predictions.

The protocol stays clean:
  - component predictions are monthly train-before-test OOS predictions;
  - the deep gate is fit on 2019-01..2019-09 only;
  - epoch/blend selection uses 2019-Q4 only;
  - 2020 labels are used only for final reporting.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")
sys.path.insert(0, "/root/feature_model")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lowmem_static_gate import WORK_DIR, collect_specs, read_component, scrub
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic
from src.plan_a.group_lgb import symbol_group_map
from strict_optimization_ablation import OUT_DIR as STRICT_OUT_DIR
from strict_optimization_ablation import PRED_START, TEST_END, TEST_START, summarize


OUT_DIR = Path(os.environ.get("DEEP_LOWMEM_OUT_DIR", "/root/autodl-tmp/quant/ML/deep_moe_lowmem_results"))
VAL_START = pd.Timestamp("2019-10-01")


@dataclass(frozen=True)
class Config:
    seed: int = 20260624
    max_train_rows: int = 800_000
    max_val_rows: int = 500_000
    batch_size: int = 8192
    epochs: int = 10
    lr: float = 1.2e-3
    weight_decay: float = 1.0e-4
    hidden: int = 128
    dropout: float = 0.18
    max_delta: float = 0.08
    delta_l2: float = 0.030
    scale_loss: float = 0.010
    blend_max: float = 1.0
    blend_steps: int = 41
    early_stop_patience: int = 3
    gate_mode: str = "full"
    delta_mode: str = "signed_residual"
    delta_weight_mode: str = "uniform"
    target_mode: str = "raw"
    anchor_weights_csv: str = ""
    anchor_missing: str = "zero"
    refit_full_2019: bool = True
    num_workers: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_from_env() -> Config:
    return Config(
        max_train_rows=int(os.environ.get("DEEP_LOWMEM_TRAIN_ROWS", "800000")),
        max_val_rows=int(os.environ.get("DEEP_LOWMEM_VAL_ROWS", "500000")),
        batch_size=int(os.environ.get("DEEP_LOWMEM_BATCH", "8192")),
        epochs=int(os.environ.get("DEEP_LOWMEM_EPOCHS", "10")),
        lr=float(os.environ.get("DEEP_LOWMEM_LR", "0.0012")),
        weight_decay=float(os.environ.get("DEEP_LOWMEM_WEIGHT_DECAY", "0.0001")),
        hidden=int(os.environ.get("DEEP_LOWMEM_HIDDEN", "128")),
        dropout=float(os.environ.get("DEEP_LOWMEM_DROPOUT", "0.18")),
        max_delta=float(os.environ.get("DEEP_LOWMEM_MAX_DELTA", "0.08")),
        delta_l2=float(os.environ.get("DEEP_LOWMEM_DELTA_L2", "0.030")),
        scale_loss=float(os.environ.get("DEEP_LOWMEM_SCALE_LOSS", "0.010")),
        blend_max=float(os.environ.get("DEEP_LOWMEM_BLEND_MAX", "1.0")),
        blend_steps=int(os.environ.get("DEEP_LOWMEM_BLEND_STEPS", "41")),
        early_stop_patience=int(os.environ.get("DEEP_LOWMEM_PATIENCE", "3")),
        gate_mode=os.environ.get("DEEP_LOWMEM_GATE_MODE", "full"),
        delta_mode=os.environ.get("DEEP_LOWMEM_DELTA_MODE", "signed_residual"),
        delta_weight_mode=os.environ.get("DEEP_LOWMEM_DELTA_WEIGHT_MODE", "uniform"),
        target_mode=os.environ.get("DEEP_LOWMEM_TARGET_MODE", "raw"),
        anchor_weights_csv=os.environ.get("DEEP_LOWMEM_ANCHOR_WEIGHTS_CSV", ""),
        anchor_missing=os.environ.get("DEEP_LOWMEM_ANCHOR_MISSING", "zero"),
        refit_full_2019=os.environ.get("DEEP_LOWMEM_REFIT_FULL_2019", "1") == "1",
        num_workers=int(os.environ.get("DEEP_LOWMEM_WORKERS", "0")),
    )


def ensure_component_memmap() -> tuple[pd.DataFrame, list[str], np.memmap]:
    os.environ.setdefault("LOWMEM_GATE_RAW_ONLY_CANDIDATES", "1")
    os.environ.setdefault("LOWMEM_GATE_MIN_BASE_2019_IC", "0.04")
    specs = collect_specs()
    if not specs:
        raise RuntimeError("no component specs")

    first = read_component(specs[0])
    n = len(first)
    expected_names = [s.name for s in specs]
    component_dir = Path(os.environ.get("DEEP_LOWMEM_COMPONENT_DIR", str(WORK_DIR)))
    component_dir.mkdir(parents=True, exist_ok=True)
    names_path = component_dir / "components.json"
    mat_path = component_dir / "components.float32.memmap"
    expected_bytes = n * len(expected_names) * np.dtype(np.float32).itemsize
    reuse = (
        names_path.exists()
        and mat_path.exists()
        and json.loads(names_path.read_text(encoding="utf-8")) == expected_names
        and mat_path.stat().st_size == expected_bytes
    )
    if not reuse:
        print("[deep-lowmem] rebuilding component memmap", flush=True)
        ref_symbol = first["symbol"].astype(str).to_numpy()
        ref_dt = first["datetime"].astype("int64").to_numpy()
        with mat_path.open("wb") as fh:
            try:
                os.posix_fallocate(fh.fileno(), 0, expected_bytes)
            except (AttributeError, OSError):
                fh.truncate(expected_bytes)
        xw = np.memmap(mat_path, mode="r+", dtype=np.float32, shape=(n, len(expected_names)))
        for j, spec in enumerate(specs):
            df = first if j == 0 else read_component(spec)
            ok = (
                len(df) == n
                and np.array_equal(df["datetime"].astype("int64").to_numpy(), ref_dt)
                and np.array_equal(df["symbol"].astype(str).to_numpy(), ref_symbol)
            )
            if not ok:
                raise RuntimeError(f"component key mismatch: {spec.name}")
            xw[:, j] = scrub(df[spec.col].to_numpy(np.float32))
            print(f"[deep-lowmem][component] {j + 1:02d}/{len(expected_names)} {spec.name}", flush=True)
            if j != 0:
                del df
        xw.flush()
        names_path.write_text(json.dumps(expected_names, indent=2), encoding="utf-8")

    base = first[["symbol", "datetime", "label"]].copy()
    x = np.memmap(mat_path, mode="r", dtype=np.float32, shape=(n, len(expected_names)))
    return base, expected_names, x


def make_label_xsz(base: pd.DataFrame) -> np.ndarray:
    g = base.groupby("datetime", sort=False)["label"]
    z = ((base["label"] - g.transform("mean")) / (g.transform("std") + 1e-9)).clip(-8, 8)
    return z.astype(np.float32).to_numpy()


def sample_indices(base: pd.DataFrame, mask: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if max_rows <= 0 or len(idx) <= max_rows:
        return idx
    rng = np.random.default_rng(seed)
    months = base.iloc[idx]["datetime"].dt.to_period("M").astype(str).to_numpy()
    ranks = base.iloc[idx].groupby("datetime", sort=False)["label"].rank(pct=True).to_numpy()
    bins = np.floor(np.clip(ranks * 6.0, 0, 5)).astype(np.int16)
    strata = np.char.add(months.astype(str), np.char.add("_", bins.astype(str)))
    pieces: list[np.ndarray] = []
    per = max(1, max_rows // max(len(np.unique(strata)), 1))
    used = 0
    for st in np.unique(strata):
        loc = idx[strata == st]
        take = min(len(loc), per)
        if take:
            pieces.append(rng.choice(loc, take, replace=False))
            used += take
    if used < max_rows:
        chosen = np.concatenate(pieces) if pieces else np.empty(0, dtype=idx.dtype)
        taken = np.zeros(len(base), dtype=bool)
        taken[chosen] = True
        rest = idx[~taken[idx]]
        fill = min(max_rows - used, len(rest))
        if fill:
            pieces.append(rng.choice(rest, fill, replace=False))
    out = np.concatenate(pieces) if pieces else idx
    if len(out) > max_rows:
        out = rng.choice(out, max_rows, replace=False)
    return np.sort(out)


def fit_static_weights(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, n_components: int) -> tuple[np.ndarray, float]:
    xt = scrub(x[train_mask]).astype(np.float64, copy=False)
    yt = y[train_mask].astype(np.float64, copy=False)
    mask = np.isfinite(yt) & np.all(np.isfinite(xt), axis=1)
    xt = xt[mask]
    yt = yt[mask]
    lower = np.full(n_components, -0.12, dtype=np.float64)
    upper = np.full(n_components, 0.85, dtype=np.float64)
    return fit_ic_weights_from_stats(xt.T @ yt, xt.T @ xt, float(yt @ yt), lower, upper)


def maybe_load_anchor_weights(
    names: list[str],
    fallback: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    if not cfg.anchor_weights_csv:
        return fallback
    path = Path(cfg.anchor_weights_csv)
    if not path.exists():
        raise FileNotFoundError(path)
    row = pd.read_csv(path).iloc[0].to_dict()
    out = np.zeros(len(names), dtype=np.float64)
    missing: list[str] = []
    for i, name in enumerate(names):
        key = f"w_{name}"
        if key in row and pd.notna(row[key]):
            out[i] = float(row[key])
        elif cfg.anchor_missing == "fallback":
            out[i] = float(fallback[i])
            missing.append(name)
        elif cfg.anchor_missing == "zero":
            out[i] = 0.0
            missing.append(name)
        else:
            raise ValueError(f"unknown DEEP_LOWMEM_ANCHOR_MISSING={cfg.anchor_missing!r}")
    print(
        f"[deep-lowmem] loaded anchor weights from {path} "
        f"matched={len(names) - len(missing)} missing={len(missing)} missing_mode={cfg.anchor_missing}",
        flush=True,
    )
    return out


def add_context(base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    symbols = sorted(base["symbol"].astype(str).unique())
    sym_map = {s: i for i, s in enumerate(symbols)}
    groups = symbol_group_map()
    group_names = sorted(base["symbol"].map(groups).fillna("other").unique())
    grp_map = {g: i for i, g in enumerate(group_names)}
    symbol_code = base["symbol"].astype(str).map(sym_map).astype(np.int64).to_numpy()
    group_code = base["symbol"].map(groups).fillna("other").map(grp_map).astype(np.int64).to_numpy()
    minute = (base["datetime"].dt.hour * 60 + base["datetime"].dt.minute).astype(np.float32)
    dow = base["datetime"].dt.dayofweek.astype(np.float32)
    month = base["datetime"].dt.month.astype(np.float32)
    cont = np.column_stack(
        [
            np.sin(2 * np.pi * minute / 1440.0),
            np.cos(2 * np.pi * minute / 1440.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
        ]
    ).astype(np.float32)
    return symbol_code, group_code, cont, sym_map, grp_map


def build_gate_features(
    x: np.ndarray,
    train_idx: np.ndarray,
    cont: np.ndarray,
    static_pred: np.ndarray,
    mode: str,
) -> np.ndarray:
    train_x = scrub(x[train_idx])
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0) + 1e-6
    if mode == "time":
        return cont.astype(np.float32)

    n = x.shape[0]
    stats = np.empty((n, 6), dtype=np.float32)
    comp_z = np.empty(x.shape, dtype=np.float32) if mode == "full" else None
    block = int(os.environ.get("DEEP_LOWMEM_FEATURE_BLOCK", "500000"))
    for start in range(0, n, block):
        end = min(n, start + block)
        cur = ((scrub(x[start:end]) - mu) / sd).clip(-8, 8).astype(np.float32)
        stats[start:end, 0] = cur.mean(axis=1)
        stats[start:end, 1] = cur.std(axis=1)
        stats[start:end, 2] = np.abs(cur).mean(axis=1)
        stats[start:end, 3] = cur.max(axis=1)
        stats[start:end, 4] = cur.min(axis=1)
        stats[start:end, 5] = static_pred[start:end].astype(np.float32)
        if comp_z is not None:
            comp_z[start:end] = cur
    if mode == "full":
        assert comp_z is not None
        return np.concatenate([comp_z, stats, cont], axis=1).astype(np.float32)
    if mode == "summary":
        return np.concatenate([stats, cont], axis=1).astype(np.float32)
    raise ValueError(f"unknown DEEP_LOWMEM_GATE_MODE={mode!r}")


def matvec_chunks(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=np.float32)
    w = weights.astype(np.float32)
    block = int(os.environ.get("DEEP_LOWMEM_MATVEC_BLOCK", "500000"))
    for start in range(0, x.shape[0], block):
        end = min(x.shape[0], start + block)
        out[start:end] = scrub(x[start:end]) @ w
    return out


class AnchoredDeepMOE(nn.Module):
    def __init__(
        self,
        n_components: int,
        gate_dim: int,
        n_symbols: int,
        n_groups: int,
        static_w: np.ndarray,
        cfg: Config,
    ) -> None:
        super().__init__()
        self.symbol_emb = nn.Embedding(n_symbols, 8)
        self.group_emb = nn.Embedding(n_groups, 4)
        inp = gate_dim + 8 + 4
        self.net = nn.Sequential(
            nn.LayerNorm(inp),
            nn.Linear(inp, cfg.hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, n_components),
        )
        self.register_buffer("static_w", torch.tensor(static_w.astype(np.float32)))
        abs_w = np.abs(static_w.astype(np.float32))
        if cfg.delta_weight_mode == "uniform":
            delta_scale = np.ones_like(abs_w, dtype=np.float32)
        elif cfg.delta_weight_mode == "static_abs":
            floor = np.quantile(abs_w[abs_w > 0], 0.25) if np.any(abs_w > 0) else 1.0
            delta_scale = np.sqrt(abs_w + floor).astype(np.float32)
            delta_scale /= float(np.mean(delta_scale) + 1e-6)
        else:
            raise ValueError(f"unknown DEEP_LOWMEM_DELTA_WEIGHT_MODE={cfg.delta_weight_mode!r}")
        self.register_buffer("delta_component_scale", torch.tensor(delta_scale))
        self.logit_delta = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.max_delta = float(cfg.max_delta)
        self.delta_mode = cfg.delta_mode

    def forward(
        self,
        mix_x: torch.Tensor,
        gate_x: torch.Tensor,
        symbol: torch.Tensor,
        group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([gate_x, self.symbol_emb(symbol), self.group_emb(group)], dim=1)
        raw = self.net(h)
        if self.delta_mode == "simplex":
            dyn = torch.softmax(raw, dim=1)
            delta = dyn - dyn.mean(dim=1, keepdim=True)
        elif self.delta_mode == "signed_residual":
            delta = torch.tanh(raw)
        else:
            raise ValueError(f"unknown DEEP_LOWMEM_DELTA_MODE={self.delta_mode!r}")
        scale = self.max_delta * torch.sigmoid(self.logit_delta)
        weights = self.static_w.unsqueeze(0) + scale * delta * self.delta_component_scale.unsqueeze(0)
        pred = (mix_x * weights).sum(dim=1)
        return pred, weights


def ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(pred) & torch.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    pred = pred - pred.mean()
    target = target - target.mean()
    den = torch.sqrt((pred.square().mean() + 1e-8) * (target.square().mean() + 1e-8))
    return -(pred * target).mean() / den


def make_loader(
    x: np.ndarray,
    gate_x: np.ndarray,
    symbol: np.ndarray,
    group: np.ndarray,
    target: np.ndarray,
    idx: np.ndarray,
    cfg: Config,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(scrub(x[idx]).astype(np.float32)),
        torch.from_numpy(gate_x[idx]),
        torch.from_numpy(symbol[idx]),
        torch.from_numpy(group[idx]),
        torch.from_numpy(target[idx]),
    )
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(), drop_last=shuffle)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    opt: torch.optim.Optimizer,
    cfg: Config,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    skipped = 0
    for mix_x, gx, sym, grp, target in train_loader:
        mix_x = mix_x.to(device, non_blocking=True)
        gx = gx.to(device, non_blocking=True)
        sym = sym.to(device, non_blocking=True)
        grp = grp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred, weights = model(mix_x, gx, sym, grp)
        loss = ic_loss(pred, target)
        loss = loss + cfg.delta_l2 * (weights - model.static_w.unsqueeze(0)).square().mean()
        loss = loss + cfg.scale_loss * (pred.std() - target.std()).square()
        if not torch.isfinite(loss):
            skipped += 1
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        if not torch.isfinite(grad_norm):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        losses.append(float(loss.detach().cpu()))
    if skipped:
        print(f"[deep-lowmem][train] skipped_nonfinite_batches={skipped}", flush=True)
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_batches(
    model: nn.Module,
    x: np.ndarray,
    gate_x: np.ndarray,
    symbol: np.ndarray,
    group: np.ndarray,
    idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    pred_out = np.empty(len(idx), dtype=np.float32)
    weight_sum = None
    weight_sq_sum = None
    seen = 0
    for start in range(0, len(idx), batch_size):
        sl = idx[start : start + batch_size]
        mix_t = torch.from_numpy(scrub(x[sl]).astype(np.float32)).to(device, non_blocking=True)
        gate_t = torch.from_numpy(gate_x[sl]).to(device, non_blocking=True)
        sym_t = torch.from_numpy(symbol[sl]).to(device, non_blocking=True)
        grp_t = torch.from_numpy(group[sl]).to(device, non_blocking=True)
        pred, weights = model(mix_t, gate_t, sym_t, grp_t)
        pred_np = pred.detach().cpu().numpy()
        w_np = weights.detach().cpu().numpy()
        pred_out[start : start + len(sl)] = pred_np
        if weight_sum is None:
            weight_sum = w_np.sum(axis=0, dtype=np.float64)
            weight_sq_sum = np.square(w_np, dtype=np.float64).sum(axis=0)
        else:
            weight_sum += w_np.sum(axis=0, dtype=np.float64)
            weight_sq_sum += np.square(w_np, dtype=np.float64).sum(axis=0)
        seen += len(sl)
    assert weight_sum is not None and weight_sq_sum is not None
    return pred_out, weight_sum / seen, np.sqrt(np.maximum(weight_sq_sum / seen - np.square(weight_sum / seen), 0.0))


def fit_blend_alpha(static_pred: np.ndarray, dyn_pred: np.ndarray, label: np.ndarray, cfg: Config) -> tuple[float, float]:
    best_alpha = 0.0
    best_ic = -np.inf
    for alpha in np.linspace(0.0, cfg.blend_max, max(2, cfg.blend_steps)):
        pred = static_pred + alpha * (dyn_pred - static_pred)
        ic = compute_ic(pred, label)
        if np.isfinite(ic) and ic > best_ic:
            best_ic = float(ic)
            best_alpha = float(alpha)
    return best_alpha, best_ic


def plot_monthly(monthly: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#2f6f8f" if x >= 0 else "#a23b3b" for x in monthly.to_numpy()]
    ax.bar(monthly.index.astype(str), monthly.to_numpy(), color=colors)
    ax.axhline(0.07, color="firebrick", linestyle="--", linewidth=1)
    ax.axhline(float(monthly.mean()), color="darkgreen", linestyle=":", linewidth=1)
    ax.set_title("Low-memory deep MOE monthly IC")
    ax.set_xlabel("month")
    ax.set_ylabel("IC")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    cfg = config_from_env()
    set_seed(cfg.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base, names, x = ensure_component_memmap()
    y = base["label"].to_numpy(np.float64)
    dt = base["datetime"]
    label_xsz = make_label_xsz(base)
    if cfg.target_mode == "xsz":
        target_all = label_xsz
    elif cfg.target_mode == "raw":
        target_all = y.astype(np.float32)
    else:
        raise ValueError(f"unknown DEEP_LOWMEM_TARGET_MODE={cfg.target_mode!r}")
    train_full_mask = ((dt >= PRED_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    fit_mask = ((dt >= PRED_START) & (dt < VAL_START) & base["label"].notna()).to_numpy()
    val_mask = ((dt >= VAL_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    test_mask = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()
    static_w_fit, static_train_ic = fit_static_weights(x, y, train_full_mask, len(names))
    static_w = maybe_load_anchor_weights(names, static_w_fit, cfg)
    static_pred_all = matvec_chunks(x, static_w)
    if cfg.anchor_weights_csv:
        static_train_ic = compute_ic(static_pred_all[train_full_mask], y[train_full_mask])
    symbol_code, group_code, cont, sym_map, grp_map = add_context(base)
    train_idx = sample_indices(base, fit_mask & np.isfinite(target_all), cfg.max_train_rows, cfg.seed)
    val_idx = sample_indices(base, val_mask & np.isfinite(target_all), cfg.max_val_rows, cfg.seed + 1)
    test_idx = np.flatnonzero(test_mask)
    gate_x = build_gate_features(x, train_idx, cont, static_pred_all, cfg.gate_mode)
    static_val_pred = static_pred_all[val_idx]
    val_label = y[val_idx]
    static_val_ic = compute_ic(static_val_pred, val_label)
    print(
        f"[deep-lowmem] components={len(names)} rows={len(base)} "
        f"train={len(train_idx)} val={len(val_idx)} static_train_ic={static_train_ic:.6f} "
        f"static_val_ic_2019q4={static_val_ic:.6f}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnchoredDeepMOE(len(names), gate_x.shape[1], len(sym_map), len(grp_map), static_w, cfg).to(device)
    train_loader = make_loader(x, gate_x, symbol_code, group_code, target_all, train_idx, cfg, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state = None
    best_val = -np.inf
    best_alpha = 0.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        loss_value = train_one_epoch(model, train_loader, opt, cfg, device)
        val_pred, _, _ = predict_batches(model, x, gate_x, symbol_code, group_code, val_idx, device, cfg.batch_size * 2)
        val_ic = compute_ic(val_pred, val_label)
        alpha, blend_ic = fit_blend_alpha(static_val_pred, val_pred, val_label, cfg)
        scale = float((cfg.max_delta * torch.sigmoid(model.logit_delta)).detach().cpu())
        row = {
            "epoch": float(epoch),
            "loss": loss_value,
            "val_ic_2019q4": float(val_ic),
            "blend_ic_2019q4": float(blend_ic),
            "blend_alpha": float(alpha),
            "delta_scale": scale,
        }
        history.append(row)
        print(
            f"[deep-lowmem][epoch {epoch:02d}] loss={row['loss']:.6f} val_ic={val_ic:.6f} "
            f"blend_ic={blend_ic:.6f} alpha={alpha:.3f} delta={scale:.4f}",
            flush=True,
        )
        if blend_ic > best_val:
            best_val = blend_ic
            best_alpha = alpha
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                print(f"[deep-lowmem] early stop after {epoch} epochs", flush=True)
                break
    if best_state is None:
        raise RuntimeError("no fitted deep MOE state")

    if cfg.refit_full_2019:
        print(
            f"[deep-lowmem] refit full 2019 for {best_epoch} epochs with alpha={best_alpha:.3f}",
            flush=True,
        )
        static_w_fit, static_train_ic = fit_static_weights(x, y, train_full_mask, len(names))
        static_w = maybe_load_anchor_weights(names, static_w_fit, cfg)
        static_pred_all = matvec_chunks(x, static_w)
        if cfg.anchor_weights_csv:
            static_train_ic = compute_ic(static_pred_all[train_full_mask], y[train_full_mask])
        train_idx = sample_indices(base, train_full_mask & np.isfinite(target_all), cfg.max_train_rows, cfg.seed + 10)
        gate_x = build_gate_features(x, train_idx, cont, static_pred_all, cfg.gate_mode)
        model = AnchoredDeepMOE(len(names), gate_x.shape[1], len(sym_map), len(grp_map), static_w, cfg).to(device)
        train_loader = make_loader(x, gate_x, symbol_code, group_code, target_all, train_idx, cfg, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for epoch in range(1, best_epoch + 1):
            loss_value = train_one_epoch(model, train_loader, opt, cfg, device)
            print(f"[deep-lowmem][refit {epoch:02d}/{best_epoch:02d}] loss={loss_value:.6f}", flush=True)
    else:
        model.load_state_dict(best_state)

    dyn_test_pred, weight_mean, weight_std = predict_batches(model, x, gate_x, symbol_code, group_code, test_idx, device, cfg.batch_size * 2)
    static_test_pred = static_pred_all[test_idx]
    test_pred = static_test_pred + best_alpha * (dyn_test_pred - static_test_pred)
    out = base.iloc[test_idx][["symbol", "datetime", "label"]].copy()
    out["pred"] = test_pred.astype(np.float32)
    out = add_cross_sectional_norms(out, "pred")
    out.to_parquet(OUT_DIR / "deep_lowmem_anchored_signed.parquet", index=False)
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    summary = pd.DataFrame([summarize(out, "deep_lowmem_anchored_signed") | {"gate_train_ic_2019": static_train_ic, "best_blend_ic_2019q4": best_val, "best_alpha_2019q4": best_alpha}])
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame({"component": names, "static_weight": static_w, "test_weight_mean": weight_mean, "test_weight_std": weight_std}).to_csv(
        OUT_DIR / "test_weight_stats.csv",
        index=False,
    )
    monthly = period_ic(out, "pred", "M")
    monthly.to_csv(OUT_DIR / "monthly_ic.csv")
    plot_monthly(monthly, OUT_DIR / "monthly_ic.png")
    meta = {
        "config": asdict(cfg),
        "components": names,
        "static_train_ic_2019": static_train_ic,
        "static_val_ic_2019q4": static_val_ic,
        "best_blend_ic_2019q4": best_val,
        "best_alpha_2019q4": best_alpha,
        "best_epoch_2019q4": best_epoch,
        "device": str(device),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary[["model", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020", "best_alpha_2019q4"]].to_string(index=False), flush=True)
    print(f"[deep-lowmem] wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
