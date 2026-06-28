from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .data import build_window_index, index_to_prediction_frame, make_quarterly_splits, prepare_symbol_arrays
from .memory_fusion import RegimeMemory, _load_checkpoint_model, _resolve_feature_cols, _source_frame
from .metrics import compute_ic, compute_rank_ic, summarize_predictions, write_metric_tables
from .rolling import load_config, make_report, prepare_frame
from .train import predict_index
from .visualize import make_standard_plots


def _best_blend_weight(primary: np.ndarray, secondary: np.ndarray, label: np.ndarray, grid: list[float]) -> tuple[float, float]:
    best_w = 0.0
    best_ic = -np.inf
    for w in grid:
        pred = (1.0 - w) * primary + w * secondary
        ic = compute_ic(pred, label)
        if np.isfinite(ic) and ic > best_ic:
            best_ic = float(ic)
            best_w = float(w)
    return best_w, best_ic


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    blend_cfg = config["blend"]
    memory_config = load_config(blend_cfg["memory_config_path"])
    memory_cfg = memory_config["memory"]
    base_config = load_config(memory_cfg["base_config_path"])
    secondary_config = load_config(blend_cfg["secondary_config_path"])
    seed = int(config["experiment"].get("seed", 42))
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
    memory_cols = _resolve_feature_cols(feature_cols, list(memory_cfg["feature_cols"]))
    cols = ["symbol", "datetime", "label", *memory_cols]

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

    memory_base_ckpt_root = Path(base_config["paths"]["checkpoints_dir"]) / memory_cfg["base_run_name"]
    memory_base_run_root = Path(base_config["paths"]["runs_dir"]) / memory_cfg["base_run_name"]
    memory_run_root = Path(memory_config["paths"]["runs_dir"]) / blend_cfg["memory_run_name"]
    secondary_ckpt_root = Path(secondary_config["paths"]["checkpoints_dir"]) / blend_cfg["secondary_run_name"]
    secondary_run_root = Path(secondary_config["paths"]["runs_dir"]) / blend_cfg["secondary_run_name"]

    seq_len = int(base_config["data"]["seq_len"])
    allow_short = bool(base_config["data"].get("allow_short_windows", True))
    val_months = int(base_config["train"].get("val_months", 3))
    pred_batch_size = int(blend_cfg.get("pred_batch_size", base_config["train"].get("pred_batch_size", 1536)))
    num_workers = int(blend_cfg.get("num_workers", base_config["train"].get("num_workers", 4)))
    grid = [float(w) for w in blend_cfg.get("secondary_weight_grid", [0.0, 0.02, 0.05, 0.08, 0.1])]

    all_preds: list[pd.DataFrame] = []
    split_metrics: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for split in splits:
        print(f"[memory_operator_blend][{split.name}] validation blend", flush=True)
        val_start = max(split.train_start, split.train_end - pd.DateOffset(months=val_months))
        memory_train_index = build_window_index(arrays, seq_len, split.train_start, val_start, True, allow_short)
        val_index = build_window_index(arrays, seq_len, val_start, split.train_end, True, allow_short)

        train_mem = _source_frame(df, arrays, memory_train_index, cols)
        val_source = _source_frame(df, arrays, val_index, cols)
        val_memory = RegimeMemory(
            feature_cols=memory_cols,
            quantiles=list(memory_cfg.get("quantiles", [0.2, 0.4, 0.6, 0.8])),
            time_bucket_minutes=int(memory_cfg.get("time_bucket_minutes", 30)),
            min_count=int(memory_cfg.get("min_count", 30)),
            shrink=float(memory_cfg.get("shrink", 100.0)),
        ).fit(train_mem)
        val_mem_pred = val_memory.predict(val_source)

        base_model, y_mean, y_std = _load_checkpoint_model(
            memory_base_ckpt_root / f"{split.name}.pt",
            len(feature_cols),
            len(arrays),
            feature_cols,
            device,
        )
        val_base_pred = predict_index(
            base_model,
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
        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        memory_weight, memory_val_ic = _best_blend_weight(
            val_base_pred.astype(np.float64),
            val_mem_pred.astype(np.float64),
            val_source["label"].to_numpy(dtype=np.float64),
            [float(w) for w in memory_cfg.get("blend_grid", [0.0, 0.05, 0.1, 0.15, 0.2])],
        )
        val_primary = (1.0 - memory_weight) * val_base_pred + memory_weight * val_mem_pred

        secondary_model, sec_mean, sec_std = _load_checkpoint_model(
            secondary_ckpt_root / f"{split.name}.pt",
            len(feature_cols),
            len(arrays),
            feature_cols,
            device,
        )
        val_secondary = predict_index(
            secondary_model,
            arrays,
            val_index,
            seq_len,
            sec_mean,
            sec_std,
            pred_batch_size,
            device,
            num_workers=num_workers,
            destandardize=True,
        )
        del secondary_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        label = val_source["label"].to_numpy(dtype=np.float64)
        secondary_weight, blend_val_ic = _best_blend_weight(
            val_primary.astype(np.float64),
            val_secondary.astype(np.float64),
            label,
            grid,
        )

        primary_test = pd.read_parquet(memory_run_root / f"predictions_{split.name}.parquet")
        secondary_test = pd.read_parquet(secondary_run_root / f"predictions_{split.name}.parquet")
        test_df = primary_test.merge(
            secondary_test[["symbol", "datetime", "pred"]].rename(columns={"pred": "secondary_pred"}),
            on=["symbol", "datetime"],
            how="left",
            validate="one_to_one",
        )
        if test_df["secondary_pred"].isna().any():
            raise RuntimeError(f"missing secondary predictions for {split.name}")
        test_df["primary_pred"] = test_df["pred"]
        test_df["pred"] = (1.0 - secondary_weight) * test_df["primary_pred"] + secondary_weight * test_df["secondary_pred"]
        test_df["split"] = split.name
        test_df.to_parquet(run_root / f"predictions_{split.name}.parquet", index=False)

        ic = compute_ic(test_df["pred"], test_df["label"])
        rank_ic = compute_rank_ic(test_df["pred"], test_df["label"])
        primary_ic = compute_ic(test_df["primary_pred"], test_df["label"])
        secondary_ic = compute_ic(test_df["secondary_pred"], test_df["label"])
        split_metrics.append(
            {
                "split": split.name,
                "train_start": str(split.train_start.date()),
                "train_end": str(split.train_end.date()),
                "test_start": str(split.test_start.date()),
                "test_end": str(split.test_end.date()),
                "train_windows": int(len(build_window_index(arrays, seq_len, split.train_start, split.train_end, True, allow_short))),
                "val_windows": int(len(val_index)),
                "test_windows": int(len(test_df)),
                "ic": ic,
                "rank_ic": rank_ic,
            }
        )
        weight_rows.append(
            {
                "split": split.name,
                "memory_weight": memory_weight,
                "memory_val_ic": memory_val_ic,
                "secondary_weight": secondary_weight,
                "blend_val_ic": blend_val_ic,
                "test_primary_ic": primary_ic,
                "test_secondary_ic": secondary_ic,
                "test_blend_ic": ic,
                "test_blend_rank_ic": rank_ic,
            }
        )
        all_preds.append(test_df)
        print(
            f"[memory_operator_blend][{split.name}] w_secondary={secondary_weight:.3f} "
            f"val_ic={blend_val_ic:.4f} primary_ic={primary_ic:.4f} secondary_ic={secondary_ic:.4f} blend_ic={ic:.4f}",
            flush=True,
        )

    pred_df = pd.concat(all_preds, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    pred_df.to_parquet(run_root / "predictions_with_label.parquet", index=False)
    pred_df[["symbol", "datetime", "pred"]].to_parquet(run_root / "predictions.parquet", index=False)
    pd.DataFrame(weight_rows).to_csv(run_root / "blend_weights.csv", index=False)
    table_paths = write_metric_tables(pred_df, run_root, split_metrics)
    table_paths["blend_weights"] = str(run_root / "blend_weights.csv")
    metrics = summarize_predictions(pred_df, split_metrics)
    (run_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_paths = make_standard_plots(pred_df, fig_root, config["experiment"]["name"])
    make_report(config, metrics, split_metrics, table_paths, plot_paths, report_path)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[memory_operator_blend] report={report_path}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
