from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .sampler import TimestampBatchSampler, WindowDataset, collate_window_batch


TARGET_WEIGHTS = torch.tensor([0.10, 0.10, 1.00, 0.05, 0.05, 0.05], dtype=torch.float32)
MAIN_TARGET_INDEX = 2


@dataclass
class TrainResult:
    model_name: str
    train_loss: float
    n_batches: int


def multitask_huber_loss(pred: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    weights = TARGET_WEIGHTS.to(pred.device).view(1, -1)
    loss = F.huber_loss(pred, targets, reduction="none", delta=1.0)
    weighted = loss * masks * weights
    denom = torch.clamp((masks * weights).sum(), min=1.0)
    return weighted.sum() / denom


def train_deep_model(
    model: torch.nn.Module,
    dataset: WindowDataset,
    device: str = "cuda",
    epochs: int = 1,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    timestamps_per_batch: int = 16,
    symbols_per_timestamp: int = 32,
    max_batches: int | None = None,
    seed: int = 11,
) -> TrainResult:
    model.to(device)
    sampler = TimestampBatchSampler(
        dataset,
        timestamps_per_batch=timestamps_per_batch,
        symbols_per_timestamp=symbols_per_timestamp,
        min_symbols_per_timestamp=10,
        shuffle=True,
        seed=seed,
        max_batches=max_batches,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_window_batch, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda") and torch.cuda.is_available()))
    losses: list[float] = []
    n_batches = 0
    model.train()
    for _ in range(int(epochs)):
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["targets"].to(device, non_blocking=True)
            m = batch["masks"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda") and torch.cuda.is_available())):
                pred = model(x)
                loss = multitask_huber_loss(pred, y, m)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            n_batches += 1
    return TrainResult(model_name=model.__class__.__name__, train_loss=float(np.mean(losses)) if losses else float("nan"), n_batches=n_batches)


@torch.no_grad()
def predict_deep_model(
    model: torch.nn.Module,
    dataset: WindowDataset,
    device: str = "cuda",
    batch_size: int = 2048,
) -> pd.DataFrame:
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_window_batch, num_workers=0)
    rows = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        pred = model(x)[:, MAIN_TARGET_INDEX].detach().float().cpu().numpy()
        rows.append(pd.DataFrame({"row_id": batch["row_id"], "pred_raw": pred}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["row_id", "pred_raw"])


def aggregate_window_tensor(x: torch.Tensor) -> torch.Tensor:
    last = x[:, -1, :]
    mean = x.mean(dim=1)
    std = x.std(dim=1, unbiased=False)
    minv = x.amin(dim=1)
    maxv = x.amax(dim=1)
    momentum = x[:, -1, :] - x[:, 0, :]
    return torch.cat([last, mean, std, minv, maxv, momentum], dim=-1)


@torch.no_grad()
def collect_aggregated_dataset(
    dataset: WindowDataset,
    max_rows: int | None = None,
    seed: int = 11,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if max_rows is not None and len(dataset) > max_rows:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(np.arange(len(dataset)), size=int(max_rows), replace=False))
        subset = torch.utils.data.Subset(dataset, chosen.tolist())
    else:
        subset = dataset
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_window_batch, num_workers=0)
    xs, ys, masks, row_ids = [], [], [], []
    for batch in loader:
        agg = aggregate_window_tensor(batch["x"]).numpy().astype("float32")
        xs.append(agg)
        ys.append(batch["targets"].numpy().astype("float32"))
        masks.append(batch["masks"].numpy().astype("float32"))
        row_ids.append(batch["row_id"])
    if not xs:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 6), dtype=np.float32),
            np.empty((0, 6), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.vstack(xs), np.vstack(ys), np.vstack(masks), np.concatenate(row_ids)


def prediction_frame_from_rows(panel: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    cols = ["datetime", "symbol", "sector", "session_type", "proxy_ret_30m", "mask_proxy_ret_30m", "ret_1m", "volume"]
    meta = panel.loc[preds["row_id"].to_numpy(dtype=np.int64), cols].reset_index(drop=True)
    out = pd.concat([meta, preds[["pred_raw"]].reset_index(drop=True)], axis=1)
    out = out.rename(columns={"proxy_ret_30m": "label"})
    out = out[out["mask_proxy_ret_30m"].astype(bool)].drop(columns=["mask_proxy_ret_30m"])
    return out
