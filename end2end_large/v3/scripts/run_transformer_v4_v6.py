from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

if os.environ.get("OMP_NUM_THREADS", "1") in {"", "0"}:
    os.environ["OMP_NUM_THREADS"] = "8"

PROJECT_ROOT = Path("/root/autodl-tmp/quant/end2end_30m")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.metrics import evaluate_predictions, write_evaluation_artifacts
from src.modelv46 import E2E_GatedMSPatch_MTL_DataLimited_v46
from src.training import MAIN_TARGET_INDEX, multitask_huber_loss, prediction_frame_from_rows


METRIC_COLUMNS = {
    "Pooled_IC": "pooled_ic",
    "SN_nonoverlap_IC": "nonoverlap_sector_neutral_cs_ic_mean",
    "raw_nonoverlap_IC": "nonoverlap_raw_cs_ic_mean",
    "dense_IC": "dense_cs_ic_mean",
    "merged_IC": "merged_proxy_ic",
    "SN_nonoverlap_RankIC": "nonoverlap_sector_neutral_cs_rankic_mean",
}


@dataclass(frozen=True)
class ArrayStoreV46:
    rows: np.ndarray
    datetime_ns: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    masks: np.ndarray


class FastWindowDatasetV46(Dataset):
    def __init__(self, arrays: list[ArrayStoreV46], seq_len: int, start: str, end: str) -> None:
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
        if len(self.timestamp_ns):
            dt = pd.DatetimeIndex(pd.to_datetime(self.timestamp_ns))
            self.minute_of_day = (dt.hour * 60 + dt.minute).astype(np.int16).to_numpy()
            self.day_of_week = dt.dayofweek.astype(np.int8).to_numpy()
            self.month = (dt.month - 1).astype(np.int8).to_numpy()
            self.symbol_id = self.array_ids.astype(np.int16, copy=False)
        else:
            self.minute_of_day = np.empty(0, dtype=np.int16)
            self.day_of_week = np.empty(0, dtype=np.int8)
            self.month = np.empty(0, dtype=np.int8)
            self.symbol_id = np.empty(0, dtype=np.int16)

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
            "timestamp_ns": int(self.timestamp_ns[index]),
            "symbol_id": int(self.symbol_id[index]),
            "minute_of_day": int(self.minute_of_day[index]),
            "day_of_week": int(self.day_of_week[index]),
            "month": int(self.month[index]),
        }


class FastTimestampBatchSamplerV46(Sampler[list[int]]):
    def __init__(
        self,
        dataset: FastWindowDatasetV46,
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


def collate_v46(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "targets": torch.stack([item["targets"] for item in batch]),
        "masks": torch.stack([item["masks"] for item in batch]),
        "row_id": np.asarray([item["row_id"] for item in batch], dtype=np.int64),
        "timestamp_ns": torch.tensor([item["timestamp_ns"] for item in batch], dtype=torch.long),
        "symbol_id": torch.tensor([item["symbol_id"] for item in batch], dtype=torch.long),
        "minute_of_day": torch.tensor([item["minute_of_day"] for item in batch], dtype=torch.long),
        "day_of_week": torch.tensor([item["day_of_week"] for item in batch], dtype=torch.long),
        "month": torch.tensor([item["month"] for item in batch], dtype=torch.long),
    }


def build_arrays(panel: pd.DataFrame, feature_cols: list[str], target_cols: list[str], mask_cols: list[str]) -> list[ArrayStoreV46]:
    arrays: list[ArrayStoreV46] = []
    for _, grp in panel.groupby("symbol", sort=False):
        arrays.append(
            ArrayStoreV46(
                rows=grp.index.to_numpy(dtype=np.int64),
                datetime_ns=grp["datetime"].to_numpy(dtype="datetime64[ns]").astype("int64"),
                features=grp[feature_cols].to_numpy(dtype=np.float32),
                targets=grp[target_cols].to_numpy(dtype=np.float32),
                masks=grp[mask_cols].to_numpy(dtype=bool),
            )
        )
    return arrays


def load_panel() -> tuple[pd.DataFrame, dict[str, object]]:
    meta = json.loads((PROJECT_ROOT / "artifacts" / "data_limited_panel_thr12.json").read_text(encoding="utf-8"))
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
    print("[v4-v6] loading panel", flush=True)
    panel = pd.read_parquet(meta["config"]["panel_cache"], columns=columns)
    panel = panel.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    return panel, meta


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def pooled_ic(df: pd.DataFrame, pred_col: str = "pred_raw", label_col: str = "label") -> float:
    pred = df[pred_col].to_numpy(dtype=float)
    label = df[label_col].to_numpy(dtype=float)
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 3:
        return float("nan")
    p = pred[mask]
    y = label[mask]
    denom = math.sqrt(float(np.mean(p * p) * np.mean(y * y)))
    return float(np.mean(p * y) / denom) if denom > 1e-18 else float("nan")


def metric_row(variant: str, metrics: dict[str, float], pooled: float) -> dict[str, Any]:
    all_metrics = {"pooled_ic": pooled, **metrics}
    row: dict[str, Any] = {"variant": variant}
    for out_col, metric_col in METRIC_COLUMNS.items():
        row[out_col] = float(all_metrics.get(metric_col, float("nan")))
    row["n_scored"] = int(metrics.get("n_scored", 0))
    return row


class EMAModel:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.n_updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            if torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value.detach())
        self.n_updates += 1

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            value.copy_(self.shadow[key].to(device=value.device, dtype=value.dtype))
        model.load_state_dict(state)


def build_model(
    args: argparse.Namespace,
    n_features: int,
    n_targets: int,
    n_symbols: int,
) -> E2E_GatedMSPatch_MTL_DataLimited_v46:
    version_rank = {"v4": 4, "v5": 5, "v6": 6}[args.version]
    use_revin = version_rank >= 5 and not args.disable_revin
    use_cross_section = version_rank >= 5 and not args.disable_cross_section
    use_market_gating = version_rank >= 6 and not args.disable_market_gating
    use_cross_variate = version_rank >= 6 and not args.disable_cross_variate
    return E2E_GatedMSPatch_MTL_DataLimited_v46(
        n_features=n_features,
        n_targets=n_targets,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        swiglu_hidden=args.swiglu_hidden,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        feature_dropout=args.feature_dropout,
        patch_scales=[(4, 2), (8, 4), (16, 8), (32, 16)],
        causal_conv_stem=True,
        use_layer_fusion=not args.disable_layer_fusion,
        use_revin=use_revin,
        use_cross_section=use_cross_section,
        cross_section_layers=args.cross_section_layers,
        cross_section_min_group=args.cross_section_min_group,
        use_market_gating=use_market_gating,
        use_cross_variate=use_cross_variate,
        learn_time_decay=args.learn_time_decay,
        use_time_bias=not args.disable_time_bias,
        use_swiglu_layerscale=not args.disable_swiglu_layerscale,
        use_factor_bank=args.use_factor_bank,
        factor_top_k=args.factor_top_k,
        factor_gate_hidden=args.factor_gate_hidden,
        factor_output_mode=args.factor_output_mode,
        factor_scale_init=args.factor_scale_init,
        use_lowrank_input=args.use_lowrank_input,
        lowrank_rank=args.lowrank_rank,
        lowrank_residual=args.lowrank_residual,
        lowrank_scale_init=args.lowrank_scale_init,
        use_meta_embedding=args.use_meta_embedding,
        n_symbols=n_symbols,
        meta_scale_init=args.meta_scale_init,
        meta_use_symbol=not args.meta_disable_symbol,
        meta_use_minute=not args.meta_disable_minute,
        meta_use_day=not args.meta_disable_day,
        meta_use_month=not args.meta_disable_month,
        moe_n_experts=args.moe_n_experts,
    )


def scheduled_lr(step: int, total_steps: int, args: argparse.Namespace) -> float:
    if step <= args.warmup_steps:
        return float(args.min_lr + (args.lr - args.min_lr) * step / max(1, args.warmup_steps))
    progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return float(args.min_lr + (args.lr - args.min_lr) * cosine)


def forward_model(model: torch.nn.Module, batch: dict[str, object], device: str) -> torch.Tensor:
    return model(
        batch["x"].to(device, non_blocking=True),
        timestamp_ns=batch["timestamp_ns"].to(device, non_blocking=True),
        symbol_id=batch["symbol_id"].to(device, non_blocking=True),
        minute_of_day=batch["minute_of_day"].to(device, non_blocking=True),
        day_of_week=batch["day_of_week"].to(device, non_blocking=True),
        month=batch["month"].to(device, non_blocking=True),
    )


def train_model(
    model: torch.nn.Module,
    train_ds: FastWindowDatasetV46,
    args: argparse.Namespace,
    device: str,
    out_dir: Path,
) -> tuple[pd.DataFrame, EMAModel]:
    sampler = FastTimestampBatchSamplerV46(
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
        collate_fn=collate_v46,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.startswith("cuda")))
    ema = EMAModel(model, decay=args.ema_decay)
    total_steps = max(1, len(sampler) * args.epochs)
    logs: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        epoch_start = time.time()
        for batch_idx, batch in enumerate(loader, start=1):
            next_step = global_step + 1
            lr = scheduled_lr(next_step, total_steps, args)
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            y = batch["targets"].to(device, non_blocking=True)
            m = batch["masks"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.startswith("cuda"))):
                pred = forward_model(model, batch, device)
                loss = multitask_huber_loss(pred, y, m)
                if args.moe_balance_weight > 0.0 and hasattr(model, "extra_loss"):
                    aux_loss = model.extra_loss()
                    if aux_loss is not None:
                        loss = loss + args.moe_balance_weight * aux_loss
            if not torch.isfinite(loss):
                print(f"[v4-v6] WARNING skip non-finite loss epoch={epoch + 1} batch={batch_idx}", flush=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            if not torch.isfinite(grad_norm):
                print(f"[v4-v6] WARNING skip non-finite grad epoch={epoch + 1} batch={batch_idx}", flush=True)
                opt.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            global_step = next_step
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            if batch_idx == 1 or batch_idx % args.log_every == 0:
                elapsed = time.time() - epoch_start
                print(
                    f"[v4-v6] {args.version} epoch={epoch + 1}/{args.epochs} "
                    f"batch={batch_idx}/{len(sampler)} step={global_step} "
                    f"loss={loss_value:.6f} lr={lr:.6g} elapsed={elapsed:.1f}s",
                    flush=True,
                )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "n_batches": int(len(losses)),
            "elapsed_sec": float(time.time() - epoch_start),
            "global_step": int(global_step),
            "ema_updates": int(ema.n_updates),
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(out_dir / "train_log.csv", index=False)
        torch.save(model.state_dict(), out_dir / f"snapshot_epoch{epoch + 1}.pt")
    return pd.DataFrame(logs), ema


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    eval_ds: FastWindowDatasetV46,
    args: argparse.Namespace,
    device: str,
    prefix: str,
) -> pd.DataFrame:
    sampler = FastTimestampBatchSamplerV46(
        eval_ds,
        timestamps_per_batch=args.eval_timestamps_per_batch,
        symbols_per_timestamp=args.eval_symbols_per_timestamp,
        min_symbols_per_timestamp=1,
        shuffle=False,
        seed=args.seed,
        max_batches=None,
    )
    loader = DataLoader(
        eval_ds,
        batch_sampler=sampler,
        collate_fn=collate_v46,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    model.to(device)
    model.eval()
    rows = []
    t0 = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        with torch.cuda.amp.autocast(enabled=(args.amp and device.startswith("cuda"))):
            pred = forward_model(model, batch, device)[:, MAIN_TARGET_INDEX]
        rows.append(pd.DataFrame({"row_id": batch["row_id"], "pred_raw": pred.detach().float().cpu().numpy()}))
        if batch_idx == 1 or batch_idx % args.predict_log_every == 0:
            print(f"[v4-v6] {prefix} predict batch={batch_idx}/{len(loader)} elapsed={time.time() - t0:.1f}s", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["row_id", "pred_raw"])


def evaluate_and_save(
    model: torch.nn.Module,
    panel: pd.DataFrame,
    eval_ds: FastWindowDatasetV46,
    args: argparse.Namespace,
    device: str,
    out_dir: Path,
    variant: str,
) -> dict[str, Any]:
    preds = predict_model(model, eval_ds, args, device, variant)
    pred_df = prediction_frame_from_rows(panel, preds)
    variant_dir = out_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(variant_dir / "predictions.parquet", index=False)
    metrics, tables, pred_variants = evaluate_predictions(pred_df)
    write_evaluation_artifacts(metrics, tables, variant_dir)
    pred_variants.to_parquet(variant_dir / "predictions_with_variants.parquet", index=False)
    pooled = pooled_ic(pred_df)
    row = metric_row(variant, metrics, pooled)
    row["prediction_path"] = str(variant_dir / "predictions.parquet")
    print(
        f"[v4-v6] {variant} pooled={row['Pooled_IC']:.6f} "
        f"sn={row['SN_nonoverlap_IC']:.6f} dense={row['dense_IC']:.6f}",
        flush=True,
    )
    return row


def write_report(args: argparse.Namespace, out_dir: Path, rows: pd.DataFrame) -> None:
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / (args.report_name or f"{args.run_name}.md")
    cols = [
        "variant",
        "Pooled_IC",
        "SN_nonoverlap_IC",
        "raw_nonoverlap_IC",
        "dense_IC",
        "merged_IC",
        "SN_nonoverlap_RankIC",
        "n_scored",
    ]
    def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = []
        for _, row in df[columns].iterrows():
            cells = []
            for col in columns:
                value = row[col]
                if isinstance(value, (float, np.floating)):
                    cells.append("nan" if not np.isfinite(value) else f"{float(value):.6f}")
                else:
                    cells.append(str(value))
            body.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, sep] + body)

    lines = [
        f"# Transformer {args.version.upper()} Controlled Run: {args.run_name}",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Train window: [{args.train_start}, {args.train_end})",
        f"- Eval window: [{args.eval_start}, {args.eval_end})",
        f"- Run directory: `{out_dir}`",
        "",
        "## Metrics",
        "",
        markdown_table(rows, cols),
        "",
        "## Artifacts",
        "",
        f"- Summary: `{out_dir / 'metrics_summary.csv'}`",
        f"- Config: `{out_dir / 'run_config.json'}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[v4-v6] report written {report_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="transformer_v4_2019_validation")
    parser.add_argument("--version", choices=["v4", "v5", "v6"], default="v4")
    parser.add_argument("--train-start", default="2017-01-01")
    parser.add_argument("--train-end", default="2019-01-01")
    parser.add_argument("--eval-start", default="2019-01-01")
    parser.add_argument("--eval-end", default="2020-01-01")
    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--timestamps-per-batch", type=int, default=16)
    parser.add_argument("--symbols-per-timestamp", type=int, default=32)
    parser.add_argument("--min-symbols-per-timestamp", type=int, default=10)
    parser.add_argument("--eval-timestamps-per-batch", type=int, default=16)
    parser.add_argument("--eval-symbols-per-timestamp", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--swiglu-hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--attn-dropout", type=float, default=0.10)
    parser.add_argument("--feature-dropout", type=float, default=0.05)
    parser.add_argument("--cross-section-layers", type=int, default=1)
    parser.add_argument("--cross-section-min-group", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--predict-log-every", type=int, default=250)
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--disable-layer-fusion", action="store_true")
    parser.add_argument("--disable-revin", action="store_true")
    parser.add_argument("--disable-cross-section", action="store_true")
    parser.add_argument("--disable-market-gating", action="store_true")
    parser.add_argument("--disable-cross-variate", action="store_true")
    parser.add_argument("--disable-time-bias", action="store_true")
    parser.add_argument("--disable-swiglu-layerscale", action="store_true")
    parser.add_argument("--learn-time-decay", action="store_true")
    parser.add_argument("--use-factor-bank", action="store_true")
    parser.add_argument("--factor-top-k", type=int, default=160)
    parser.add_argument("--factor-gate-hidden", type=int, default=80)
    parser.add_argument("--factor-output-mode", choices=["project", "topk", "masked"], default="project")
    parser.add_argument("--factor-scale-init", type=float, default=None)
    parser.add_argument("--use-lowrank-input", action="store_true")
    parser.add_argument("--lowrank-rank", type=int, default=8)
    parser.add_argument("--lowrank-residual", action="store_true")
    parser.add_argument("--lowrank-scale-init", type=float, default=-2.0)
    parser.add_argument("--use-meta-embedding", action="store_true")
    parser.add_argument("--meta-scale-init", type=float, default=-1.5)
    parser.add_argument("--meta-disable-symbol", action="store_true")
    parser.add_argument("--meta-disable-minute", action="store_true")
    parser.add_argument("--meta-disable-day", action="store_true")
    parser.add_argument("--meta-disable-month", action="store_true")
    parser.add_argument("--moe-n-experts", type=int, default=1)
    parser.add_argument("--moe-balance-weight", type=float, default=0.0)
    parser.add_argument("--eval-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = PROJECT_ROOT / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    panel, meta = load_panel()
    feature_cols = list(meta["feature_cols"])
    target_cols = list(meta["target_cols"])
    mask_cols = list(meta["mask_cols"])
    print("[v4-v6] building arrays", flush=True)
    arrays = build_arrays(panel, feature_cols, target_cols, mask_cols)
    train_ds = FastWindowDatasetV46(arrays, args.seq_len, args.train_start, args.train_end)
    eval_ds = FastWindowDatasetV46(arrays, args.seq_len, args.eval_start, args.eval_end)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args, len(feature_cols), len(target_cols), len(arrays))
    if args.version in {"v5", "v6"} and args.eval_timestamps_per_batch < 1:
        raise ValueError("--eval-timestamps-per-batch must be positive")
    sampler_full = FastTimestampBatchSamplerV46(
        train_ds,
        args.timestamps_per_batch,
        args.symbols_per_timestamp,
        args.min_symbols_per_timestamp,
        max_batches=None,
    )
    run_config = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "run_name": args.run_name,
        "version": args.version,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "seq_len": args.seq_len,
        "n_features": len(feature_cols),
        "n_targets": len(target_cols),
        "n_train_refs": int(len(train_ds)),
        "n_eval_refs": int(len(eval_ds)),
        "n_train_timestamp_groups": int(len(sampler_full.groups)),
        "device": device,
        "parameter_count": parameter_count(model),
        "architecture": {
            "name": "E2E_GatedMSPatch_MTL_DataLimited_v46",
            "version": args.version,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "swiglu_hidden": args.swiglu_hidden,
            "dropout": args.dropout,
            "attn_dropout": args.attn_dropout,
            "feature_dropout": args.feature_dropout,
            "layer_fusion": not args.disable_layer_fusion,
            "revin": args.version in {"v5", "v6"} and not args.disable_revin,
            "cross_section_attention": args.version in {"v5", "v6"} and not args.disable_cross_section,
            "market_gating": args.version == "v6" and not args.disable_market_gating,
            "cross_variate_branch": args.version == "v6" and not args.disable_cross_variate,
            "time_bias_attention": not args.disable_time_bias,
            "swiglu_layerscale": not args.disable_swiglu_layerscale,
            "learn_time_decay": args.learn_time_decay,
            "factor_bank": args.use_factor_bank,
            "factor_top_k": args.factor_top_k if args.use_factor_bank else 0,
            "factor_output_mode": args.factor_output_mode if args.use_factor_bank else "off",
            "factor_scale_init": args.factor_scale_init if args.use_factor_bank else None,
            "lowrank_input": args.use_lowrank_input,
            "lowrank_rank": args.lowrank_rank if args.use_lowrank_input else 0,
            "lowrank_residual": args.lowrank_residual if args.use_lowrank_input else False,
            "lowrank_scale_init": args.lowrank_scale_init if args.use_lowrank_input else None,
            "meta_embedding": args.use_meta_embedding,
            "meta_scale_init": args.meta_scale_init if args.use_meta_embedding else None,
            "meta_symbol": args.use_meta_embedding and not args.meta_disable_symbol,
            "meta_minute": args.use_meta_embedding and not args.meta_disable_minute,
            "meta_day": args.use_meta_embedding and not args.meta_disable_day,
            "meta_month": args.use_meta_embedding and not args.meta_disable_month,
            "n_symbols": len(arrays),
            "moe_n_experts": args.moe_n_experts,
            "moe_balance_weight": args.moe_balance_weight,
        },
        "training": vars(args),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[v4-v6] version={args.version} train_refs={len(train_ds):,} eval_refs={len(eval_ds):,} "
        f"train_timestamps={len(sampler_full.groups):,} params={parameter_count(model):,} device={device}",
        flush=True,
    )

    train_log, ema = train_model(model, train_ds, args, device, out_dir)
    torch.save(model.state_dict(), out_dir / "model_raw.pt")
    rows: list[dict[str, Any]] = []
    if args.eval_raw:
        rows.append(evaluate_and_save(model, panel, eval_ds, args, device, out_dir, "raw"))
    if args.eval_ema:
        ema_model = copy.deepcopy(model).to(device)
        ema.copy_to(ema_model)
        torch.save(ema_model.state_dict(), out_dir / "model_ema.pt")
        rows.append(evaluate_and_save(ema_model, panel, eval_ds, args, device, out_dir, "ema"))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    write_report(args, out_dir, summary)
    print(summary.to_string(index=False), flush=True)
    print(f"[v4-v6] completed {out_dir}", flush=True)


if __name__ == "__main__":
    main()
