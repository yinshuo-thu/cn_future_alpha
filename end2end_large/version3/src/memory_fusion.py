from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .data import (
    build_window_index,
    index_to_prediction_frame,
    make_quarterly_splits,
    prepare_symbol_arrays,
)
from .metrics import compute_ic, compute_rank_ic, summarize_predictions, write_metric_tables
from .rolling import load_config, make_report, prepare_frame
from .train import make_model, predict_index
from .visualize import make_standard_plots


def _feature_base_name(col: str) -> str:
    return col[3:] if col.startswith("rz_") else col


def _resolve_feature_cols(feature_cols: list[str], requested: list[str]) -> list[str]:
    out = []
    bases = {_feature_base_name(col): col for col in feature_cols}
    for name in requested:
        if name in feature_cols:
            out.append(name)
        elif name in bases:
            out.append(bases[name])
        elif f"rz_{name}" in feature_cols:
            out.append(f"rz_{name}")
        else:
            raise ValueError(f"memory feature not found: {name}")
    return out


def _source_frame(df: pd.DataFrame, arrays, index: list[tuple[int, int]], cols: list[str]) -> pd.DataFrame:
    source_idx = [int(arrays[sym_id].source_index[row]) for sym_id, row in index]
    return df.iloc[source_idx][cols].copy()


class RegimeMemory:
    def __init__(
        self,
        feature_cols: list[str],
        quantiles: list[float],
        time_bucket_minutes: int,
        min_count: int,
        shrink: float,
        decay_halflife_days: float | None = None,
    ) -> None:
        self.feature_cols = feature_cols
        self.quantiles = quantiles
        self.time_bucket_minutes = int(time_bucket_minutes)
        self.min_count = int(min_count)
        self.shrink = float(shrink)
        self.decay_halflife_days = float(decay_halflife_days) if decay_halflife_days else None
        self.edges: dict[str, np.ndarray] = {}
        self.global_mean = 0.0
        self.tables: list[tuple[list[str], pd.DataFrame]] = []

    def _add_keys(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        minute = pd.to_datetime(out["datetime"]).dt.hour * 60 + pd.to_datetime(out["datetime"]).dt.minute
        out["_tb"] = (minute // self.time_bucket_minutes).astype(np.int16)
        for col in self.feature_cols:
            vals = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            edges = self.edges[col]
            out[f"_bin_{col}"] = np.searchsorted(edges, vals.to_numpy(dtype=np.float64), side="right").astype(np.int16)
        return out

    def fit(self, train: pd.DataFrame) -> "RegimeMemory":
        train = train.copy()
        labels = pd.to_numeric(train["label"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self.decay_halflife_days is not None and self.decay_halflife_days > 0:
            dt = pd.to_datetime(train["datetime"])
            age_days = (dt.max() - dt).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
            weights = np.exp(-np.log(2.0) * age_days / self.decay_halflife_days)
            weights = np.maximum(weights, 1e-6)
        else:
            weights = np.ones(len(train), dtype=np.float64)
        train["_mem_weight"] = weights.astype(np.float64)
        train["_mem_label_weighted"] = labels.to_numpy(dtype=np.float64) * train["_mem_weight"].to_numpy(dtype=np.float64)
        weight_sum = float(train["_mem_weight"].sum())
        self.global_mean = float(train["_mem_label_weighted"].sum() / weight_sum) if weight_sum > 0 else float(labels.mean())
        for col in self.feature_cols:
            vals = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            edges = np.quantile(vals.to_numpy(dtype=np.float64), self.quantiles)
            self.edges[col] = np.unique(edges[np.isfinite(edges)])
        keyed = self._add_keys(train)
        bin_cols = [f"_bin_{col}" for col in self.feature_cols]
        key_sets = [
            ["symbol", "_tb", *bin_cols],
            ["symbol", *bin_cols],
            ["_tb", *bin_cols],
            bin_cols,
            ["symbol", "_tb"],
            ["symbol"],
            ["_tb"],
        ]
        self.tables = []
        for keys in key_sets:
            agg = (
                keyed.groupby(keys, sort=False)
                .agg(
                    label_weighted_sum=("_mem_label_weighted", "sum"),
                    weight_sum=("_mem_weight", "sum"),
                    count=("label", "count"),
                )
                .reset_index()
            )
            agg = agg[agg["count"] >= self.min_count].copy()
            if agg.empty:
                continue
            agg["mean"] = agg["label_weighted_sum"].astype(float) / agg["weight_sum"].astype(float).clip(lower=1e-8)
            agg["_mem_pred"] = (
                agg["weight_sum"].astype(float) * agg["mean"].astype(float) + self.shrink * self.global_mean
            ) / (agg["weight_sum"].astype(float) + self.shrink)
            self.tables.append((keys, agg[keys + ["_mem_pred"]]))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        keyed = self._add_keys(frame)
        pred = pd.Series(np.nan, index=keyed.index, dtype=np.float64)
        for keys, table in self.tables:
            if pred.notna().all():
                break
            missing = pred.isna()
            merged = keyed.loc[missing, keys].merge(table, on=keys, how="left")
            vals = merged["_mem_pred"].to_numpy(dtype=np.float64)
            pred.loc[missing] = vals
        return pred.fillna(self.global_mean).to_numpy(dtype=np.float32)


def _load_checkpoint_model(
    checkpoint_path: Path,
    n_features: int,
    n_symbols: int,
    feature_cols: list[str],
    device: torch.device,
):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_model(ckpt["config"], n_features, n_symbols, feature_cols=feature_cols).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model, float(ckpt["target_mean"]), float(ckpt["target_std"])


def _best_blend_weight(val_df: pd.DataFrame, grid: list[float]) -> tuple[float, float]:
    best_w = 0.0
    best_ic = -np.inf
    model = val_df["model_pred"].to_numpy(dtype=np.float64)
    mem = val_df["memory_pred"].to_numpy(dtype=np.float64)
    label = val_df["label"].to_numpy(dtype=np.float64)
    for w in grid:
        pred = (1.0 - w) * model + w * mem
        ic = compute_ic(pred, label)
        if np.isfinite(ic) and ic > best_ic:
            best_ic = float(ic)
            best_w = float(w)
    return best_w, best_ic


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    mem_cfg = config["memory"]
    base_config = load_config(mem_cfg["base_config_path"])
    base_run = mem_cfg["base_run_name"]
    seed = int(config["experiment"].get("seed", 42))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")

    run_root = Path(config["paths"]["runs_dir"]) / config["experiment"]["name"]
    fig_root = Path(config["paths"]["figures_dir"]) / config["experiment"]["name"]
    report_path = Path(config["paths"]["reports_dir"]) / config["experiment"]["report_name"]
    run_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    df, feature_cols, target_col = prepare_frame(base_config)
    df = df.dropna(subset=["label", target_col]).reset_index(drop=True)
    arrays, _ = prepare_symbol_arrays(df, feature_cols, target_col)
    resolved_memory_cols = _resolve_feature_cols(feature_cols, list(mem_cfg["feature_cols"]))
    cols = ["symbol", "datetime", "label", *resolved_memory_cols]

    data_start = pd.Timestamp(base_config["rolling"].get("data_start", df["datetime"].min()))
    data_end = pd.Timestamp(base_config["rolling"].get("data_end", df["datetime"].max())) + pd.Timedelta(nanoseconds=1)
    splits = make_quarterly_splits(
        data_start=data_start,
        data_end=data_end,
        train_start=base_config["rolling"].get("train_start", "2018-01-01"),
        first_test_start=base_config["rolling"].get("first_test_start", "2020-01-01"),
        freq_months=int(base_config["rolling"].get("freq_months", 3)),
        allow_partial_test=bool(base_config["rolling"].get("allow_partial_test", True)),
    )
    max_splits = base_config["rolling"].get("max_splits")
    if max_splits:
        splits = splits[: int(max_splits)]

    base_run_root = Path(base_config["paths"]["runs_dir"]) / base_run
    base_ckpt_root = Path(base_config["paths"]["checkpoints_dir"]) / base_run
    seq_len = int(base_config["data"]["seq_len"])
    allow_short = bool(base_config["data"].get("allow_short_windows", True))
    val_months = int(base_config["train"].get("val_months", 3))
    pred_batch_size = int(mem_cfg.get("pred_batch_size", base_config["train"].get("pred_batch_size", 1536)))
    num_workers = int(mem_cfg.get("num_workers", base_config["train"].get("num_workers", 4)))

    all_preds: list[pd.DataFrame] = []
    split_metrics: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    grid = [float(w) for w in mem_cfg.get("blend_grid", [0.0, 0.05, 0.1, 0.15, 0.2, 0.25])]

    for split in splits:
        print(f"[memory][{split.name}] building memory", flush=True)
        val_start = max(split.train_start, split.train_end - pd.DateOffset(months=val_months))
        memory_train_index = build_window_index(arrays, seq_len, split.train_start, val_start, True, allow_short)
        memory_full_index = build_window_index(arrays, seq_len, split.train_start, split.train_end, True, allow_short)
        val_index = build_window_index(arrays, seq_len, val_start, split.train_end, True, allow_short)
        test_index = build_window_index(arrays, seq_len, split.test_start, split.test_end, True, allow_short)
        train_mem = _source_frame(df, arrays, memory_train_index, cols)
        full_mem = _source_frame(df, arrays, memory_full_index, cols)
        val_source = _source_frame(df, arrays, val_index, cols)
        test_source = _source_frame(df, arrays, test_index, cols)

        val_memory = RegimeMemory(
            feature_cols=resolved_memory_cols,
            quantiles=list(mem_cfg.get("quantiles", [0.2, 0.4, 0.6, 0.8])),
            time_bucket_minutes=int(mem_cfg.get("time_bucket_minutes", 30)),
            min_count=int(mem_cfg.get("min_count", 30)),
            shrink=float(mem_cfg.get("shrink", 100.0)),
            decay_halflife_days=mem_cfg.get("decay_halflife_days"),
        ).fit(train_mem)
        val_mem_pred = val_memory.predict(val_source)

        model, y_mean, y_std = _load_checkpoint_model(
            base_ckpt_root / f"{split.name}.pt",
            len(feature_cols),
            len(arrays),
            feature_cols,
            device,
        )
        val_model_pred = predict_index(
            model,
            arrays,
            val_index,
            seq_len,
            y_mean,
            y_std,
            pred_batch_size,
            device,
            num_workers=num_workers,
            destandardize=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val_df = index_to_prediction_frame(arrays, val_index, val_model_pred)
        val_df = val_df.rename(columns={"pred": "model_pred"})
        val_df["memory_pred"] = val_mem_pred
        best_w, best_val_ic = _best_blend_weight(val_df, grid)

        test_memory = RegimeMemory(
            feature_cols=resolved_memory_cols,
            quantiles=list(mem_cfg.get("quantiles", [0.2, 0.4, 0.6, 0.8])),
            time_bucket_minutes=int(mem_cfg.get("time_bucket_minutes", 30)),
            min_count=int(mem_cfg.get("min_count", 30)),
            shrink=float(mem_cfg.get("shrink", 100.0)),
            decay_halflife_days=mem_cfg.get("decay_halflife_days"),
        ).fit(full_mem)
        test_mem_pred = test_memory.predict(test_source)
        base_test = pd.read_parquet(base_run_root / f"predictions_{split.name}.parquet")
        mem_test = index_to_prediction_frame(arrays, test_index, test_mem_pred)
        test_df = base_test.merge(
            mem_test[["symbol", "datetime", "pred"]].rename(columns={"pred": "memory_pred"}),
            on=["symbol", "datetime"],
            how="left",
            validate="one_to_one",
        )
        if test_df["memory_pred"].isna().any():
            raise RuntimeError(f"missing memory predictions for {split.name}")
        test_df["model_pred"] = test_df["pred"]
        test_df["pred"] = (1.0 - best_w) * test_df["model_pred"] + best_w * test_df["memory_pred"]
        test_df["split"] = split.name
        test_df.to_parquet(run_root / f"predictions_{split.name}.parquet", index=False)
        ic = compute_ic(test_df["pred"], test_df["label"])
        rank_ic = compute_rank_ic(test_df["pred"], test_df["label"])
        mem_ic = compute_ic(test_df["memory_pred"], test_df["label"])
        model_ic = compute_ic(test_df["model_pred"], test_df["label"])
        split_metrics.append(
            {
                "split": split.name,
                "train_start": str(split.train_start.date()),
                "train_end": str(split.train_end.date()),
                "test_start": str(split.test_start.date()),
                "test_end": str(split.test_end.date()),
                "train_windows": int(len(memory_full_index)),
                "val_windows": int(len(val_index)),
                "test_windows": int(len(test_df)),
                "ic": ic,
                "rank_ic": rank_ic,
            }
        )
        weight_rows.append(
            {
                "split": split.name,
                "blend_weight_memory": best_w,
                "val_blend_ic": best_val_ic,
                "test_model_ic": model_ic,
                "test_memory_ic": mem_ic,
                "test_blend_ic": ic,
                "test_blend_rank_ic": rank_ic,
            }
        )
        all_preds.append(test_df)
        print(
            f"[memory][{split.name}] w_mem={best_w:.3f} val_ic={best_val_ic:.4f} "
            f"test_model_ic={model_ic:.4f} mem_ic={mem_ic:.4f} blend_ic={ic:.4f}",
            flush=True,
        )

    pred_df = pd.concat(all_preds, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    pred_df.to_parquet(run_root / "predictions_with_label.parquet", index=False)
    pred_df[["symbol", "datetime", "pred"]].to_parquet(run_root / "predictions.parquet", index=False)
    pd.DataFrame(weight_rows).to_csv(run_root / "memory_weights.csv", index=False)
    table_paths = write_metric_tables(pred_df, run_root, split_metrics)
    table_paths["memory_weights"] = str(run_root / "memory_weights.csv")
    metrics = summarize_predictions(pred_df, split_metrics)
    (run_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_paths = make_standard_plots(pred_df, fig_root, config["experiment"]["name"])
    make_report(config, metrics, split_metrics, table_paths, plot_paths, report_path)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[memory] report={report_path}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
