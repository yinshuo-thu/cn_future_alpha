from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

if os.environ.get("OMP_NUM_THREADS", "1") in {"", "0"}:
    os.environ["OMP_NUM_THREADS"] = "8"

PROJECT_ROOT = Path("/root/autodl-tmp/quant/end2end_30m")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.metrics import evaluate_predictions, write_evaluation_artifacts
from src.model import E2E_GatedMSPatch_MTL_DataLimited_v1
from src.training import MAIN_TARGET_INDEX, multitask_huber_loss, prediction_frame_from_rows


@dataclass(frozen=True)
class ArrayStore:
    rows: np.ndarray
    datetime_ns: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    masks: np.ndarray


class FastWindowDataset(Dataset):
    def __init__(
        self,
        arrays: list[ArrayStore],
        seq_len: int,
        start: str,
        end: str,
    ) -> None:
        self.arrays = arrays
        self.seq_len = int(seq_len)
        start_ns = pd.Timestamp(start).value
        end_ns = pd.Timestamp(end).value
        array_ids: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        row_ids: list[np.ndarray] = []
        timestamp_ns: list[np.ndarray] = []
        for array_id, arr in enumerate(arrays):
            ok = (arr.datetime_ns >= start_ns) & (arr.datetime_ns < end_ns) & arr.masks[:, MAIN_TARGET_INDEX]
            if len(ok) < self.seq_len:
                continue
            ok[: self.seq_len - 1] = False
            pos = np.flatnonzero(ok)
            if len(pos) == 0:
                continue
            array_ids.append(np.full(len(pos), array_id, dtype=np.int16))
            positions.append(pos.astype(np.int32, copy=False))
            row_ids.append(arr.rows[pos].astype(np.int64, copy=False))
            timestamp_ns.append(arr.datetime_ns[pos].astype(np.int64, copy=False))
        if array_ids:
            self.array_ids = np.concatenate(array_ids)
            self.positions = np.concatenate(positions)
            self.row_ids = np.concatenate(row_ids)
            self.timestamp_ns = np.concatenate(timestamp_ns)
        else:
            self.array_ids = np.empty(0, dtype=np.int16)
            self.positions = np.empty(0, dtype=np.int32)
            self.row_ids = np.empty(0, dtype=np.int64)
            self.timestamp_ns = np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        array_id = int(self.array_ids[index])
        pos = int(self.positions[index])
        arr = self.arrays[array_id]
        start = pos + 1 - self.seq_len
        return {
            "x": torch.from_numpy(arr.features[start : pos + 1]),
            "targets": torch.from_numpy(arr.targets[pos]),
            "masks": torch.from_numpy(arr.masks[pos].astype(np.float32)),
            "row_id": int(self.row_ids[index]),
        }


class FastTimestampBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: FastWindowDataset,
        timestamps_per_batch: int = 16,
        symbols_per_timestamp: int = 32,
        min_symbols_per_timestamp: int = 10,
        shuffle: bool = True,
        seed: int = 11,
        max_batches: int | None = None,
    ) -> None:
        self.timestamps_per_batch = int(timestamps_per_batch)
        self.symbols_per_timestamp = int(symbols_per_timestamp)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_batches = max_batches
        self.epoch = 0
        ts = dataset.timestamp_ns
        if len(ts) == 0:
            self.groups: list[np.ndarray] = []
            return
        order = np.argsort(ts, kind="mergesort")
        sorted_ts = ts[order]
        boundaries = np.flatnonzero(sorted_ts[1:] != sorted_ts[:-1]) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(order)]])
        self.groups = [
            order[start:end].astype(np.int64, copy=False)
            for start, end in zip(starts, ends)
            if end - start >= int(min_symbols_per_timestamp)
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        group_order = np.arange(len(self.groups))
        if self.shuffle:
            rng.shuffle(group_order)
        batches = 0
        for start in range(0, len(group_order), self.timestamps_per_batch):
            batch_groups = group_order[start : start + self.timestamps_per_batch]
            batch: list[int] = []
            for group_idx in batch_groups:
                idxs = self.groups[int(group_idx)]
                if self.shuffle and len(idxs) > self.symbols_per_timestamp:
                    chosen = rng.choice(idxs, size=self.symbols_per_timestamp, replace=False)
                else:
                    chosen = idxs[: self.symbols_per_timestamp]
                batch.extend(int(i) for i in chosen)
            if batch:
                yield batch
                batches += 1
                if self.max_batches is not None and batches >= self.max_batches:
                    return

    def __len__(self) -> int:
        n = (len(self.groups) + self.timestamps_per_batch - 1) // self.timestamps_per_batch
        return min(n, int(self.max_batches)) if self.max_batches is not None else n


def fast_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "targets": torch.stack([item["targets"] for item in batch]),
        "masks": torch.stack([item["masks"] for item in batch]),
        "row_id": np.asarray([item["row_id"] for item in batch], dtype=np.int64),
    }


def build_arrays(panel: pd.DataFrame, feature_cols: list[str], target_cols: list[str], mask_cols: list[str]) -> list[ArrayStore]:
    arrays: list[ArrayStore] = []
    for _, grp in panel.groupby("symbol", sort=False):
        arrays.append(
            ArrayStore(
                rows=grp.index.to_numpy(dtype=np.int64),
                datetime_ns=grp["datetime"].to_numpy(dtype="datetime64[ns]").astype("int64"),
                features=grp[feature_cols].to_numpy(dtype=np.float32),
                targets=grp[target_cols].to_numpy(dtype=np.float32),
                masks=grp[mask_cols].to_numpy(dtype=bool),
            )
        )
    return arrays


def load_panel() -> tuple[pd.DataFrame, dict[str, object]]:
    meta_path = PROJECT_ROOT / "artifacts" / "data_limited_panel_thr12.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols = list(meta["feature_cols"])
    target_cols = list(meta["target_cols"])
    mask_cols = list(meta["mask_cols"])
    columns = list(
        dict.fromkeys(
            [
                "symbol",
                "datetime",
                "sector",
                "session_type",
                "ret_1m",
                "volume",
                "proxy_ret_30m",
                "mask_proxy_ret_30m",
            ]
            + feature_cols
            + target_cols
            + mask_cols
        )
    )
    print("[transformer-main] loading panel", flush=True)
    panel = pd.read_parquet(meta["config"]["panel_cache"], columns=columns)
    panel = panel.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    return panel, meta


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def lr_lambda(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return max(min_lr_ratio, float(step + 1) / float(max(1, warmup_steps)))
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def train_model(
    model: torch.nn.Module,
    train_ds: FastWindowDataset,
    args: argparse.Namespace,
    device: str,
    out_dir: Path,
) -> pd.DataFrame:
    sampler = FastTimestampBatchSampler(
        train_ds,
        timestamps_per_batch=args.timestamps_per_batch,
        symbols_per_timestamp=args.symbols_per_timestamp,
        min_symbols_per_timestamp=args.min_symbols_per_timestamp,
        shuffle=True,
        seed=args.seed,
        max_batches=args.max_batches_per_epoch,
    )
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=fast_collate,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = max(1, len(sampler) * args.epochs)
    min_lr_ratio = args.min_lr / args.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda step: lr_lambda(step, args.warmup_steps, total_steps, min_lr_ratio),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.startswith("cuda")))
    logs: list[dict[str, float]] = []
    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        epoch_start = time.time()
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["targets"].to(device, non_blocking=True)
            m = batch["masks"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.startswith("cuda"))):
                pred = model(x)
                loss = multitask_huber_loss(pred, y, m)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            global_step += 1
            if batch_idx == 1 or batch_idx % args.log_every == 0:
                elapsed = time.time() - epoch_start
                print(
                    f"[transformer-main] epoch={epoch + 1}/{args.epochs} "
                    f"batch={batch_idx}/{len(sampler)} step={global_step} "
                    f"loss={loss_value:.6f} lr={scheduler.get_last_lr()[0]:.6g} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "n_batches": int(len(losses)),
            "elapsed_sec": float(time.time() - epoch_start),
            "global_step": int(global_step),
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(out_dir / "train_log.csv", index=False)
    return pd.DataFrame(logs)


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    eval_ds: FastWindowDataset,
    args: argparse.Namespace,
    device: str,
) -> pd.DataFrame:
    loader = DataLoader(
        eval_ds,
        batch_size=args.predict_batch_size,
        shuffle=False,
        collate_fn=fast_collate,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    model.to(device)
    model.eval()
    rows = []
    t0 = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(args.amp and device.startswith("cuda"))):
            pred = model(x)[:, MAIN_TARGET_INDEX]
        rows.append(pd.DataFrame({"row_id": batch["row_id"], "pred_raw": pred.detach().float().cpu().numpy()}))
        if batch_idx == 1 or batch_idx % args.predict_log_every == 0:
            print(
                f"[transformer-main] predict batch={batch_idx}/{len(loader)} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["row_id", "pred_raw"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="transformer_main_config_2020_full")
    parser.add_argument("--train-start", default="2017-01-01")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--eval-start", default="2020-01-01")
    parser.add_argument("--eval-end", default="2021-01-01")
    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--timestamps-per-batch", type=int, default=16)
    parser.add_argument("--symbols-per-timestamp", type=int, default=32)
    parser.add_argument("--min-symbols-per-timestamp", type=int, default=10)
    parser.add_argument("--predict-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--predict-log-every", type=int, default=200)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = PROJECT_ROOT / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    panel, meta = load_panel()
    feature_cols = list(meta["feature_cols"])
    target_cols = list(meta["target_cols"])
    mask_cols = list(meta["mask_cols"])
    print("[transformer-main] building symbol arrays", flush=True)
    arrays = build_arrays(panel, feature_cols, target_cols, mask_cols)
    train_ds = FastWindowDataset(arrays, args.seq_len, args.train_start, args.train_end)
    eval_ds = FastWindowDataset(arrays, args.seq_len, args.eval_start, args.eval_end)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E2E_GatedMSPatch_MTL_DataLimited_v1(
        len(feature_cols),
        d_model=192,
        n_layers=5,
        n_heads=6,
        ffn_dim=512,
        dropout=0.15,
        attn_dropout=0.10,
        feature_dropout=0.05,
        patch_scales=[(4, 2), (8, 4), (16, 8), (32, 16)],
        causal_conv_stem=True,
    )
    n_params = parameter_count(model)
    train_groups = len(
        FastTimestampBatchSampler(
            train_ds,
            args.timestamps_per_batch,
            args.symbols_per_timestamp,
            args.min_symbols_per_timestamp,
            max_batches=None,
        ).groups
    )
    run_config = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "run_name": args.run_name,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "seq_len": args.seq_len,
        "n_features": len(feature_cols),
        "n_targets": len(target_cols),
        "n_train_refs": int(len(train_ds)),
        "n_eval_refs": int(len(eval_ds)),
        "n_train_timestamp_groups": int(train_groups),
        "device": device,
        "parameter_count": n_params,
        "architecture": {
            "name": "E2E_GatedMSPatch_MTL_DataLimited_v1",
            "d_model": 192,
            "n_layers": 5,
            "n_heads": 6,
            "ffn_dim": 512,
            "dropout": 0.15,
            "attn_dropout": 0.10,
            "feature_dropout": 0.05,
            "patch_scales": [[4, 2], [8, 4], [16, 8], [32, 16]],
            "causal_conv_stem": True,
        },
        "training": vars(args),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[transformer-main] train_refs={len(train_ds):,} eval_refs={len(eval_ds):,} "
        f"train_timestamps={train_groups:,} params={n_params:,} device={device}",
        flush=True,
    )
    train_model(model, train_ds, args, device, out_dir)
    torch.save(model.state_dict(), out_dir / "model.pt")
    preds = predict_model(model, eval_ds, args, device)
    pred_df = prediction_frame_from_rows(panel, preds)
    metrics, tables, pred_variants = evaluate_predictions(pred_df)
    write_evaluation_artifacts(metrics, tables, out_dir)
    pred_variants.to_parquet(out_dir / "predictions.parquet", index=False)
    print(pd.DataFrame([metrics]).to_string(index=False), flush=True)
    print(f"[transformer-main] completed {out_dir}", flush=True)


if __name__ == "__main__":
    main()
