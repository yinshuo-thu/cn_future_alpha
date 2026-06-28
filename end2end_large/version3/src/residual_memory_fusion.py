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
from .memory_fusion import RegimeMemory, _load_checkpoint_model, _resolve_feature_cols, _source_frame
from .metrics import compute_ic, compute_rank_ic, summarize_predictions, write_metric_tables
from .rolling import load_config, make_report, prepare_frame
from .train import predict_index
from .visualize import make_standard_plots


def _sample_index(
    index: list[tuple[int, int]],
    max_windows: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if max_windows <= 0 or len(index) <= max_windows:
        return index
    chosen = rng.choice(len(index), size=int(max_windows), replace=False)
    chosen.sort()
    return [index[int(i)] for i in chosen]


def _fit_residual_memory(
    frame: pd.DataFrame,
    model_pred: np.ndarray,
    cfg: dict[str, Any],
    feature_cols: list[str],
    residual_clip: float | None,
) -> RegimeMemory:
    fit_frame = frame.copy()
    residual = fit_frame["label"].to_numpy(dtype=np.float64) - model_pred.astype(np.float64)
    if residual_clip is not None and residual_clip > 0:
        residual = np.clip(residual, -float(residual_clip), float(residual_clip))
    fit_frame["label"] = residual.astype(np.float32)
    return RegimeMemory(
        feature_cols=feature_cols,
        quantiles=list(cfg.get("quantiles", [0.2, 0.4, 0.6, 0.8])),
        time_bucket_minutes=int(cfg.get("time_bucket_minutes", 30)),
        min_count=int(cfg.get("min_count", 30)),
        shrink=float(cfg.get("shrink", 100.0)),
        decay_halflife_days=cfg.get("decay_halflife_days"),
    ).fit(fit_frame)


def _best_residual_weight(
    model_pred: np.ndarray,
    residual_pred: np.ndarray,
    label: np.ndarray,
    grid: list[float],
) -> tuple[float, float]:
    best_w = 0.0
    best_ic = compute_ic(model_pred, label)
    for weight in grid:
        pred = model_pred + float(weight) * residual_pred
        ic = compute_ic(pred, label)
        if np.isfinite(ic) and ic > best_ic:
            best_ic = float(ic)
            best_w = float(weight)
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
    max_fit = int(mem_cfg.get("max_fit_windows", 750_000))
    max_full_fit = int(mem_cfg.get("max_full_fit_windows", max_fit))
    residual_clip = mem_cfg.get("residual_clip")
    residual_clip = float(residual_clip) if residual_clip is not None else None
    grid = [float(w) for w in mem_cfg.get("residual_weight_grid", [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])]

    all_preds: list[pd.DataFrame] = []
    split_metrics: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for split in splits:
        print(f"[residual_memory][{split.name}] building sampled residual memories", flush=True)
        val_start = max(split.train_start, split.train_end - pd.DateOffset(months=val_months))
        memory_train_index = build_window_index(arrays, seq_len, split.train_start, val_start, True, allow_short)
        memory_full_index = build_window_index(arrays, seq_len, split.train_start, split.train_end, True, allow_short)
        val_index = build_window_index(arrays, seq_len, val_start, split.train_end, True, allow_short)
        test_index = build_window_index(arrays, seq_len, split.test_start, split.test_end, True, allow_short)
        fit_index = _sample_index(memory_train_index, max_fit, rng)
        full_fit_index = _sample_index(memory_full_index, max_full_fit, rng)

        fit_source = _source_frame(df, arrays, fit_index, cols)
        full_fit_source = _source_frame(df, arrays, full_fit_index, cols)
        val_source = _source_frame(df, arrays, val_index, cols)
        test_source = _source_frame(df, arrays, test_index, cols)

        model, y_mean, y_std = _load_checkpoint_model(
            base_ckpt_root / f"{split.name}.pt",
            len(feature_cols),
            len(arrays),
            feature_cols,
            device,
        )
        fit_model_pred = predict_index(
            model,
            arrays,
            fit_index,
            seq_len,
            y_mean,
            y_std,
            pred_batch_size,
            device,
            num_workers=num_workers,
            destandardize=True,
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
        full_fit_model_pred = predict_index(
            model,
            arrays,
            full_fit_index,
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

        val_memory = _fit_residual_memory(
            fit_source,
            fit_model_pred,
            mem_cfg,
            resolved_memory_cols,
            residual_clip=residual_clip,
        )
        val_resid_pred = val_memory.predict(val_source).astype(np.float64)
        label = val_source["label"].to_numpy(dtype=np.float64)
        best_w, best_val_ic = _best_residual_weight(val_model_pred.astype(np.float64), val_resid_pred, label, grid)

        test_memory = _fit_residual_memory(
            full_fit_source,
            full_fit_model_pred,
            mem_cfg,
            resolved_memory_cols,
            residual_clip=residual_clip,
        )
        test_resid_pred = test_memory.predict(test_source).astype(np.float64)
        base_test = pd.read_parquet(base_run_root / f"predictions_{split.name}.parquet")
        resid_test = index_to_prediction_frame(arrays, test_index, test_resid_pred)
        test_df = base_test.merge(
            resid_test[["symbol", "datetime", "pred"]].rename(columns={"pred": "residual_memory_pred"}),
            on=["symbol", "datetime"],
            how="left",
            validate="one_to_one",
        )
        if test_df["residual_memory_pred"].isna().any():
            raise RuntimeError(f"missing residual memory predictions for {split.name}")
        test_df["model_pred"] = test_df["pred"]
        test_df["pred"] = test_df["model_pred"] + best_w * test_df["residual_memory_pred"]
        test_df["split"] = split.name
        test_df.to_parquet(run_root / f"predictions_{split.name}.parquet", index=False)

        ic = compute_ic(test_df["pred"], test_df["label"])
        rank_ic = compute_rank_ic(test_df["pred"], test_df["label"])
        model_ic = compute_ic(test_df["model_pred"], test_df["label"])
        resid_ic = compute_ic(test_df["residual_memory_pred"], test_df["label"])
        split_metrics.append(
            {
                "split": split.name,
                "train_start": str(split.train_start.date()),
                "train_end": str(split.train_end.date()),
                "test_start": str(split.test_start.date()),
                "test_end": str(split.test_end.date()),
                "train_windows": int(len(memory_full_index)),
                "fit_windows": int(len(fit_index)),
                "full_fit_windows": int(len(full_fit_index)),
                "val_windows": int(len(val_index)),
                "test_windows": int(len(test_df)),
                "ic": ic,
                "rank_ic": rank_ic,
            }
        )
        weight_rows.append(
            {
                "split": split.name,
                "residual_weight": best_w,
                "val_blend_ic": best_val_ic,
                "test_model_ic": model_ic,
                "test_residual_ic": resid_ic,
                "test_blend_ic": ic,
                "test_blend_rank_ic": rank_ic,
                "fit_windows": int(len(fit_index)),
                "full_fit_windows": int(len(full_fit_index)),
            }
        )
        all_preds.append(test_df)
        print(
            f"[residual_memory][{split.name}] w_resid={best_w:.3f} val_ic={best_val_ic:.4f} "
            f"test_model_ic={model_ic:.4f} resid_ic={resid_ic:.4f} blend_ic={ic:.4f}",
            flush=True,
        )

    pred_df = pd.concat(all_preds, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    pred_df.to_parquet(run_root / "predictions_with_label.parquet", index=False)
    pred_df[["symbol", "datetime", "pred"]].to_parquet(run_root / "predictions.parquet", index=False)
    pd.DataFrame(weight_rows).to_csv(run_root / "residual_memory_weights.csv", index=False)
    table_paths = write_metric_tables(pred_df, run_root, split_metrics)
    table_paths["residual_memory_weights"] = str(run_root / "residual_memory_weights.csv")
    metrics = summarize_predictions(pred_df, split_metrics)
    (run_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_paths = make_standard_plots(pred_df, fig_root, config["experiment"]["name"])
    make_report(config, metrics, split_metrics, table_paths, plot_paths, report_path)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[residual_memory] report={report_path}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
