#!/usr/bin/env python3
"""
Deep per-tick MOE over strict base-learner predictions.

Protocol:
  - base learners are already strict train-before-test OOS predictions;
  - deep gate trains on 2019 OOS predictions only;
  - epoch/model selection uses 2019 Q4 validation only;
  - 2020 labels are used only for final reporting.

The gate is intentionally anchored to the best linear signed IC weights.  The
network only learns a small zero-sum residual around those weights, which keeps
the dynamic per-tick weights from overfitting the 2019 noise too aggressively.
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

from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic
from src.plan_a.group_lgb import symbol_group_map
from strict_optimization_ablation import OUT_DIR as OPT_DIR
from strict_optimization_ablation import load_component_panel


OUT_DIR = Path(os.environ.get("DEEP_MOE_OUT_DIR", "/root/autodl-tmp/quant/ML/deep_moe_results"))
PRED_START = pd.Timestamp("2019-01-01")
VAL_START = pd.Timestamp("2019-10-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")


@dataclass(frozen=True)
class Config:
    seed: int = 20260623
    max_train_rows: int = 1_200_000
    max_val_rows: int = 700_000
    batch_size: int = 8192
    epochs: int = 18
    lr: float = 1.5e-3
    weight_decay: float = 1.0e-4
    hidden: int = 192
    dropout: float = 0.12
    max_delta: float = 0.12
    delta_l2: float = 0.010
    scale_loss: float = 0.010
    num_workers: int = 2
    early_stop_patience: int = 4
    blend_max: float = 1.25
    blend_steps: int = 51
    gate_mode: str = "full"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def make_label_xsz(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("datetime", sort=False)["label"]
    return ((df["label"] - g.transform("mean")) / (g.transform("std") + 1e-9)).clip(-8, 8).astype(np.float32)


def sample_indices(df: pd.DataFrame, mask: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if max_rows <= 0 or len(idx) <= max_rows:
        return idx
    rng = np.random.default_rng(seed)
    months = df.iloc[idx]["datetime"].dt.to_period("M").astype(str).to_numpy()
    y = df.iloc[idx]["label_xrank"].to_numpy(np.float32)
    bins = np.floor(np.clip((y + 0.5) * 6.0, 0, 5)).astype(np.int16)
    strata = np.char.add(months.astype(str), np.char.add("_", bins.astype(str)))
    pieces: list[np.ndarray] = []
    per = max(1, max_rows // max(len(np.unique(strata)), 1))
    used = 0
    for st in np.unique(strata):
        loc = idx[strata == st]
        take = min(len(loc), per)
        if take > 0:
            pieces.append(rng.choice(loc, take, replace=False))
            used += take
    if used < max_rows:
        already = np.concatenate(pieces) if pieces else np.empty(0, dtype=idx.dtype)
        taken = np.zeros(len(df), dtype=bool)
        taken[already] = True
        rest = idx[~taken[idx]]
        fill = min(max_rows - used, len(rest))
        if fill > 0:
            pieces.append(rng.choice(rest, fill, replace=False))
    out = np.concatenate(pieces) if pieces else idx
    if len(out) > max_rows:
        out = rng.choice(out, max_rows, replace=False)
    return np.sort(out)


def add_context(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], dict[str, int], list[str]]:
    out = df.copy()
    symbols = sorted(out["symbol"].unique())
    sym_map = {s: i for i, s in enumerate(symbols)}
    groups = symbol_group_map()
    out["group"] = out["symbol"].map(groups).fillna("other")
    group_names = sorted(out["group"].unique())
    grp_map = {g: i for i, g in enumerate(group_names)}
    out["symbol_code"] = out["symbol"].map(sym_map).astype(np.int16)
    out["group_code"] = out["group"].map(grp_map).astype(np.int8)
    minute = (out["datetime"].dt.hour * 60 + out["datetime"].dt.minute).astype(np.float32)
    dow = out["datetime"].dt.dayofweek.astype(np.float32)
    month = out["datetime"].dt.month.astype(np.float32)
    out["minute_sin"] = np.sin(2 * np.pi * minute / 1440.0).astype(np.float32)
    out["minute_cos"] = np.cos(2 * np.pi * minute / 1440.0).astype(np.float32)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0).astype(np.float32)
    cont_cols = ["minute_sin", "minute_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    return out, sym_map, grp_map, cont_cols


def fit_static_weights(data: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, float]:
    train = data[(data["datetime"] >= PRED_START) & (data["datetime"] < TEST_START)]
    x = scrub(train[names].to_numpy(np.float32)).astype(np.float64, copy=False)
    y = train["label"].to_numpy(np.float64)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    c = x.T @ y
    g = x.T @ x
    yy = float(y @ y)
    lower = np.full(len(names), -0.12, dtype=np.float64)
    upper = np.full(len(names), 0.85, dtype=np.float64)
    return fit_ic_weights_from_stats(c, g, yy, lower, upper)


def build_arrays(
    data: pd.DataFrame,
    names: list[str],
    cont_cols: list[str],
    train_idx: np.ndarray,
    gate_mode: str,
) -> dict[str, np.ndarray]:
    train_x = scrub(data.iloc[train_idx][names].to_numpy(np.float32))
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0) + 1e-6
    comp = scrub(data[names].to_numpy(np.float32))
    comp_z = ((comp - mu) / sd).clip(-8, 8).astype(np.float32)
    stats = np.stack(
        [
            comp_z.mean(axis=1),
            comp_z.std(axis=1),
            np.abs(comp_z).mean(axis=1),
            comp_z.max(axis=1),
            comp_z.min(axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    cont = data[cont_cols].to_numpy(np.float32)
    if gate_mode == "full":
        gate_x = np.concatenate([comp_z, stats, cont], axis=1).astype(np.float32)
    elif gate_mode == "summary":
        gate_x = np.concatenate([stats, cont], axis=1).astype(np.float32)
    elif gate_mode == "time":
        gate_x = cont.astype(np.float32)
    else:
        raise ValueError(f"unknown DEEP_MOE_GATE_MODE={gate_mode!r}")
    return {
        "mix_x": comp.astype(np.float32),
        "gate_x": gate_x,
        "symbol": data["symbol_code"].to_numpy(np.int64),
        "group": data["group_code"].to_numpy(np.int64),
        "target": data["label_xsz"].to_numpy(np.float32),
    }


class AnchoredDeepGate(nn.Module):
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
        self.logit_delta = nn.Parameter(torch.tensor(-0.75, dtype=torch.float32))
        self.max_delta = float(cfg.max_delta)

    def forward(
        self,
        mix_x: torch.Tensor,
        gate_x: torch.Tensor,
        symbol: torch.Tensor,
        group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([gate_x, self.symbol_emb(symbol), self.group_emb(group)], dim=1)
        logits = self.net(h)
        dyn = torch.softmax(logits, dim=1)
        delta = dyn - dyn.mean(dim=1, keepdim=True)
        scale = self.max_delta * torch.sigmoid(self.logit_delta)
        weights = self.static_w.unsqueeze(0) + scale * delta
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


def make_loader(arr: dict[str, np.ndarray], idx: np.ndarray, cfg: Config, shuffle: bool) -> DataLoader:
    tensors = (
        torch.from_numpy(arr["mix_x"][idx]),
        torch.from_numpy(arr["gate_x"][idx]),
        torch.from_numpy(arr["symbol"][idx]),
        torch.from_numpy(arr["group"][idx]),
        torch.from_numpy(arr["target"][idx]),
    )
    ds = TensorDataset(*tensors)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


@torch.no_grad()
def predict_batches(model: nn.Module, arr: dict[str, np.ndarray], idx: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    for start in range(0, len(idx), batch_size):
        sl = idx[start : start + batch_size]
        mix_x = torch.from_numpy(arr["mix_x"][sl]).to(device, non_blocking=True)
        gate_x = torch.from_numpy(arr["gate_x"][sl]).to(device, non_blocking=True)
        sym = torch.from_numpy(arr["symbol"][sl]).to(device, non_blocking=True)
        grp = torch.from_numpy(arr["group"][sl]).to(device, non_blocking=True)
        pred, _ = model(mix_x, gate_x, sym, grp)
        out[start : start + len(sl)] = pred.detach().cpu().numpy()
    return out


def summarize(pred: pd.DataFrame, name: str) -> dict[str, object]:
    row: dict[str, object] = {"model": name, "rows": len(pred), "label_rows": int(pred["label"].notna().sum())}
    for col in ["pred", "pred_xsz", "pred_xrank"]:
        by_m = period_ic(pred, col, "M")
        row[f"{col}_ic_2020"] = compute_ic(pred[col].to_numpy(), pred["label"].to_numpy())
        row[f"{col}_monthly_mean_2020"] = float(by_m.mean())
        row[f"{col}_monthly_ir_2020"] = float(by_m.mean() / by_m.std()) if by_m.std() > 0 else float("nan")
    return row


def plot_monthly(monthly: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#2f6f8f" if x >= 0 else "#a23b3b" for x in monthly.to_numpy()]
    ax.bar(monthly.index.astype(str), monthly.to_numpy(), color=colors)
    ax.axhline(0.07, color="firebrick", linestyle="--", linewidth=1)
    ax.axhline(float(monthly.mean()), color="darkgreen", linestyle=":", linewidth=1)
    ax.set_ylabel("IC")
    ax.set_xlabel("month")
    ax.set_title("Deep MOE monthly IC")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fit_blend_alpha(
    static_pred: np.ndarray,
    dyn_pred: np.ndarray,
    label: np.ndarray,
    max_alpha: float,
    steps: int,
) -> tuple[float, float]:
    best_alpha = 0.0
    best_ic = -np.inf
    for alpha in np.linspace(0.0, max_alpha, max(2, steps)):
        pred = static_pred + alpha * (dyn_pred - static_pred)
        ic = compute_ic(pred, label)
        if np.isfinite(ic) and ic > best_ic:
            best_ic = float(ic)
            best_alpha = float(alpha)
    return best_alpha, best_ic


def main() -> None:
    cfg = Config(
        max_train_rows=int(os.environ.get("DEEP_MOE_TRAIN_ROWS", "1200000")),
        max_val_rows=int(os.environ.get("DEEP_MOE_VAL_ROWS", "700000")),
        epochs=int(os.environ.get("DEEP_MOE_EPOCHS", "18")),
        batch_size=int(os.environ.get("DEEP_MOE_BATCH", "8192")),
        lr=float(os.environ.get("DEEP_MOE_LR", "0.0015")),
        weight_decay=float(os.environ.get("DEEP_MOE_WEIGHT_DECAY", "0.0001")),
        hidden=int(os.environ.get("DEEP_MOE_HIDDEN", "192")),
        dropout=float(os.environ.get("DEEP_MOE_DROPOUT", "0.12")),
        max_delta=float(os.environ.get("DEEP_MOE_MAX_DELTA", "0.12")),
        delta_l2=float(os.environ.get("DEEP_MOE_DELTA_L2", "0.010")),
        scale_loss=float(os.environ.get("DEEP_MOE_SCALE_LOSS", "0.010")),
        num_workers=int(os.environ.get("DEEP_MOE_WORKERS", "2")),
        early_stop_patience=int(os.environ.get("DEEP_MOE_PATIENCE", "4")),
        blend_max=float(os.environ.get("DEEP_MOE_BLEND_MAX", "1.25")),
        blend_steps=int(os.environ.get("DEEP_MOE_BLEND_STEPS", "51")),
        gate_mode=os.environ.get("DEEP_MOE_GATE_MODE", "full"),
    )
    set_seed(cfg.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, names = load_component_panel()
    data = data.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    data["label_xsz"] = make_label_xsz(data)
    data["label_xrank"] = (data.groupby("datetime", sort=False)["label"].rank(pct=True) - 0.5).astype(np.float32)
    data, sym_map, grp_map, cont_cols = add_context(data)
    static_w, static_train_ic = fit_static_weights(data, names)
    print(f"[deep-moe] components={len(names)} rows={len(data)} static_train_ic={static_train_ic:.6f}", flush=True)

    tr_mask = (
        (data["datetime"] >= PRED_START)
        & (data["datetime"] < VAL_START)
        & data["label"].notna()
        & data["label_xsz"].notna()
    ).to_numpy()
    val_mask = (
        (data["datetime"] >= VAL_START)
        & (data["datetime"] < TEST_START)
        & data["label"].notna()
        & data["label_xsz"].notna()
    ).to_numpy()
    test_mask = ((data["datetime"] >= TEST_START) & (data["datetime"] < TEST_END)).to_numpy()
    train_idx = sample_indices(data, tr_mask, cfg.max_train_rows, cfg.seed)
    val_idx = sample_indices(data, val_mask, cfg.max_val_rows, cfg.seed + 1)
    test_idx = np.flatnonzero(test_mask)
    arr = build_arrays(data, names, cont_cols, train_idx, cfg.gate_mode)
    val_label = data.iloc[val_idx]["label"].to_numpy()
    static_val_pred = scrub(data.iloc[val_idx][names].to_numpy(np.float32)) @ static_w.astype(np.float32)
    static_val_ic = compute_ic(static_val_pred, val_label)
    print(f"[deep-moe] static_val_ic_2019q4={static_val_ic:.6f}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnchoredDeepGate(
        n_components=len(names),
        gate_dim=arr["gate_x"].shape[1],
        n_symbols=len(sym_map),
        n_groups=len(grp_map),
        static_w=static_w,
        cfg=cfg,
    ).to(device)
    train_loader = make_loader(arr, train_idx, cfg, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = -np.inf
    best_state = None
    best_alpha = 0.0
    history = []
    stale_epochs = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for mix_x, gate_x, sym, grp, target in train_loader:
            mix_x = mix_x.to(device, non_blocking=True)
            gate_x = gate_x.to(device, non_blocking=True)
            sym = sym.to(device, non_blocking=True)
            grp = grp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred, weights = model(mix_x, gate_x, sym, grp)
            loss = ic_loss(pred, target)
            loss = loss + cfg.delta_l2 * (weights - model.static_w.unsqueeze(0)).square().mean()
            loss = loss + cfg.scale_loss * (pred.std() - target.std()).square()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = predict_batches(model, arr, val_idx, device, cfg.batch_size * 2)
        val_ic = compute_ic(val_pred, data.iloc[val_idx]["label"].to_numpy())
        blend_alpha, blend_ic = fit_blend_alpha(static_val_pred, val_pred, val_label, cfg.blend_max, cfg.blend_steps)
        scale = float((cfg.max_delta * torch.sigmoid(model.logit_delta)).detach().cpu())
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "val_ic_2019q4": val_ic,
                "blend_ic_2019q4": blend_ic,
                "blend_alpha": blend_alpha,
                "delta_scale": scale,
            }
        )
        print(
            f"[deep-moe][epoch {epoch:02d}] loss={np.mean(losses):.6f} "
            f"val_ic={val_ic:.6f} blend_ic={blend_ic:.6f} alpha={blend_alpha:.3f} delta={scale:.4f}",
            flush=True,
        )
        if blend_ic > best_val:
            best_val = blend_ic
            best_alpha = blend_alpha
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.early_stop_patience:
                print(f"[deep-moe] early stop after {epoch} epochs", flush=True)
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    dyn_test_pred = predict_batches(model, arr, test_idx, device, cfg.batch_size * 2)
    static_test_pred = scrub(data.iloc[test_idx][names].to_numpy(np.float32)) @ static_w.astype(np.float32)
    test_pred = static_test_pred + best_alpha * (dyn_test_pred - static_test_pred)
    out = data.iloc[test_idx][["symbol", "datetime", "label"]].copy()
    out["pred"] = test_pred.astype(np.float32)
    out = add_cross_sectional_norms(out, "pred")
    out.to_parquet(OUT_DIR / "deep_moe_anchored_signed.parquet", index=False)
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    summary = pd.DataFrame([summarize(out, "deep_moe_anchored_signed")])
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    monthly = period_ic(out, "pred", "M")
    monthly.to_csv(OUT_DIR / "monthly_ic.csv")
    plot_monthly(monthly, OUT_DIR / "monthly_ic.png")
    meta = {
        "config": asdict(cfg),
        "components": names,
        "static_train_ic_2019": static_train_ic,
        "static_val_ic_2019q4": static_val_ic,
        "best_blend_ic_2019q4": best_val,
        "best_blend_alpha_2019q4": best_alpha,
        "device": str(device),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary[["model", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020"]].to_string(index=False), flush=True)
    print(f"[deep-moe] wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
