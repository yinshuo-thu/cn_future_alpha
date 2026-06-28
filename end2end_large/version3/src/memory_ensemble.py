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


def _iter_simplex_weights(n_weights: int, step: float, max_total: float):
    values = np.arange(0.0, max_total + step * 0.5, step, dtype=np.float64)
    values = [float(round(v, 10)) for v in values]

    def rec(prefix: list[float], remaining: int, used: float):
        if remaining == 0:
            yield tuple(prefix)
            return
        for value in values:
            if used + value <= max_total + 1e-12:
                prefix.append(value)
                yield from rec(prefix, remaining - 1, used + value)
                prefix.pop()

    yield from rec([], n_weights, 0.0)


def _best_ensemble_weights(
    val_df: pd.DataFrame,
    memory_cols: list[str],
    step: float,
    max_total: float,
) -> tuple[list[float], float]:
    best_weights = [0.0 for _ in memory_cols]
    best_ic = -np.inf
    model = val_df["model_pred"].to_numpy(dtype=np.float64)
    memories = [val_df[col].to_numpy(dtype=np.float64) for col in memory_cols]
    label = val_df["label"].to_numpy(dtype=np.float64)
    for weights in _iter_simplex_weights(len(memory_cols), step, max_total):
        pred = (1.0 - sum(weights)) * model
        for weight, mem in zip(weights, memories):
            pred = pred + weight * mem
        ic = compute_ic(pred, label)
        if np.isfinite(ic) and ic > best_ic:
            best_ic = float(ic)
            best_weights = [float(w) for w in weights]
    return best_weights, best_ic


def _fit_memory(frame: pd.DataFrame, cfg: dict[str, Any], feature_cols: list[str]) -> RegimeMemory:
    return RegimeMemory(
        feature_cols=feature_cols,
        quantiles=list(cfg.get("quantiles", [0.2, 0.4, 0.6, 0.8])),
        time_bucket_minutes=int(cfg.get("time_bucket_minutes", 30)),
        min_count=int(cfg.get("min_count", 30)),
        shrink=float(cfg.get("shrink", 100.0)),
        decay_halflife_days=cfg.get("decay_halflife_days"),
    ).fit(frame)


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    mem_cfg = config["memory"]
    base_config = load_config(mem_cfg["base_config_path"])
    base_run = mem_cfg["base_run_name"]
    variants = list(mem_cfg["variants"])
    seed = int(config["experiment"].get("seed", 42))
    np.random.default_rng(seed)
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
    variant_feature_cols = {
        str(variant["name"]): _resolve_feature_cols(feature_cols, list(variant["feature_cols"]))
        for variant in variants
    }
    cols_by_variant = {
        name: ["symbol", "datetime", "label", *cols]
        for name, cols in variant_feature_cols.items()
    }

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
    step = float(mem_cfg.get("weight_grid_step", 0.05))
    max_total = float(mem_cfg.get("max_total_memory_weight", 0.6))

    all_preds: list[pd.DataFrame] = []
    split_metrics: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    memory_names = [str(variant["name"]) for variant in variants]
    memory_cols = [f"memory_pred_{name}" for name in memory_names]

    for split in splits:
        print(f"[memory_ensemble][{split.name}] building memories", flush=True)
        val_start = max(split.train_start, split.train_end - pd.DateOffset(months=val_months))
        memory_train_index = build_window_index(arrays, seq_len, split.train_start, val_start, True, allow_short)
        memory_full_index = build_window_index(arrays, seq_len, split.train_start, split.train_end, True, allow_short)
        val_index = build_window_index(arrays, seq_len, val_start, split.train_end, True, allow_short)
        test_index = build_window_index(arrays, seq_len, split.test_start, split.test_end, True, allow_short)

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
        test_mem_preds: dict[str, np.ndarray] = {}
        for variant in variants:
            name = str(variant["name"])
            cols = cols_by_variant[name]
            train_mem = _source_frame(df, arrays, memory_train_index, cols)
            full_mem = _source_frame(df, arrays, memory_full_index, cols)
            val_source = _source_frame(df, arrays, val_index, cols)
            test_source = _source_frame(df, arrays, test_index, cols)
            resolved_cols = variant_feature_cols[name]
            val_memory = _fit_memory(train_mem, variant, resolved_cols)
            val_df[f"memory_pred_{name}"] = val_memory.predict(val_source)
            test_memory = _fit_memory(full_mem, variant, resolved_cols)
            test_mem_preds[name] = test_memory.predict(test_source)

        best_weights, best_val_ic = _best_ensemble_weights(val_df, memory_cols, step, max_total)
        base_test = pd.read_parquet(base_run_root / f"predictions_{split.name}.parquet")
        test_df = base_test.copy()
        test_df["model_pred"] = test_df["pred"]
        test_df["pred"] = (1.0 - sum(best_weights)) * test_df["model_pred"]
        row: dict[str, Any] = {
            "split": split.name,
            "blend_weight_model": 1.0 - sum(best_weights),
            "val_blend_ic": best_val_ic,
        }
        mem_ics: dict[str, float] = {}
        for name, weight in zip(memory_names, best_weights):
            mem_test = index_to_prediction_frame(arrays, test_index, test_mem_preds[name])
            test_df = test_df.merge(
                mem_test[["symbol", "datetime", "pred"]].rename(columns={"pred": f"memory_pred_{name}"}),
                on=["symbol", "datetime"],
                how="left",
                validate="one_to_one",
            )
            col = f"memory_pred_{name}"
            if test_df[col].isna().any():
                raise RuntimeError(f"missing memory predictions for {split.name} / {name}")
            test_df["pred"] = test_df["pred"] + weight * test_df[col]
            row[f"blend_weight_{name}"] = weight
            mem_ics[name] = compute_ic(test_df[col], test_df["label"])
            row[f"test_memory_ic_{name}"] = mem_ics[name]

        test_df["split"] = split.name
        test_df.to_parquet(run_root / f"predictions_{split.name}.parquet", index=False)
        ic = compute_ic(test_df["pred"], test_df["label"])
        rank_ic = compute_rank_ic(test_df["pred"], test_df["label"])
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
        row.update(
            {
                "test_model_ic": model_ic,
                "test_blend_ic": ic,
                "test_blend_rank_ic": rank_ic,
            }
        )
        weight_rows.append(row)
        all_preds.append(test_df)
        weight_str = " ".join(f"{name}={weight:.2f}" for name, weight in zip(memory_names, best_weights))
        print(
            f"[memory_ensemble][{split.name}] w_model={1.0 - sum(best_weights):.2f} {weight_str} "
            f"val_ic={best_val_ic:.4f} test_model_ic={model_ic:.4f} blend_ic={ic:.4f}",
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
    print(f"[memory_ensemble] report={report_path}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
