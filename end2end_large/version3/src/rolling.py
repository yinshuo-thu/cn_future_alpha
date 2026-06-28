from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

if os.environ.get("OMP_NUM_THREADS", "1") in {"", "0"}:
    os.environ["OMP_NUM_THREADS"] = "8"
if os.environ.get("MKL_NUM_THREADS", "1") in {"", "0"}:
    os.environ["MKL_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
import torch
import yaml

from .data import (
    add_auxiliary_targets,
    add_stable_raw_columns,
    add_target_transform,
    build_labeled_frame,
    build_window_index,
    index_to_prediction_frame,
    make_quarterly_splits,
    prepare_symbol_arrays,
)
from .metrics import compute_ic, compute_rank_ic, summarize_predictions, write_metric_tables
from .normalization import add_rolling_zscore
from .train import predict_index, train_split
from .visualize import make_standard_plots


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    parent = config.get("extends") if isinstance(config, dict) else None
    if not parent:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    base = load_config(parent_path)

    def merge_dict(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
        for key, value in src.items():
            if key == "extends":
                continue
            if isinstance(value, list) and isinstance(dst.get(key), list):
                dst_items = dst[key]
                if all(isinstance(item, dict) and "name" in item for item in dst_items + value):
                    by_name = {str(item["name"]): item for item in dst_items}
                    merged = []
                    seen = set()
                    for item in dst_items:
                        name = str(item["name"])
                        if name in {str(src_item["name"]) for src_item in value}:
                            merged_item = merge_dict(dict(item), by_name[name])
                            for src_item in value:
                                if str(src_item["name"]) == name:
                                    merged_item = merge_dict(merged_item, src_item)
                                    break
                            merged.append(merged_item)
                        else:
                            merged.append(item)
                        seen.add(name)
                    for src_item in value:
                        if str(src_item["name"]) not in seen:
                            merged.append(src_item)
                    dst[key] = merged
                    continue
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge_dict(dst[key], value)
            else:
                dst[key] = value
        return dst

    return merge_dict(base, config)


def _maybe_add_auxiliary_targets(config: dict[str, Any], df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data_cfg = config["data"]
    train_cfg = config.get("train", {})
    aux_cols = list(train_cfg.get("aux_targets") or data_cfg.get("aux_targets") or [])
    if not aux_cols:
        data_cfg["aux_target_cols"] = []
        return df, []
    df, aux_cols = add_auxiliary_targets(
        df,
        aux_cols,
        horizon=int(data_cfg.get("label_horizon", 30)),
    )
    data_cfg["aux_target_cols"] = aux_cols
    train_cfg["aux_targets"] = aux_cols
    config["train"] = train_cfg
    return df, aux_cols


def prepare_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str], str]:
    data_cfg = config["data"]
    cache_path = Path(data_cfg.get("cache_path", ""))
    if cache_path and cache_path.exists() and data_cfg.get("use_cache", True):
        print(f"[baseline] loading cache {cache_path}", flush=True)
        df = pd.read_parquet(cache_path)
        meta_path = cache_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        feature_cols = list(data_cfg.get("feature_cols") or meta.get("feature_cols") or [])
        target_col = data_cfg.get("target_col") or meta.get("target_col") or "target"
        if feature_cols and target_col in df.columns:
            requested_target_mode = data_cfg.get("target_mode", meta.get("target_mode", "raw"))
            cached_target_mode = meta.get("target_mode", "raw")
            requested_min_xs = int(data_cfg.get("target_min_xs_count", meta.get("target_min_xs_count", 8)))
            cached_min_xs = int(meta.get("target_min_xs_count", requested_min_xs))
            if requested_target_mode != cached_target_mode or requested_min_xs != cached_min_xs:
                print(
                    f"[baseline] cache features reused; recomputing target mode "
                    f"{cached_target_mode}/{cached_min_xs} -> {requested_target_mode}/{requested_min_xs}",
                    flush=True,
                )
                df, target_col = add_target_transform(
                    df,
                    requested_target_mode,
                    min_xs_count=requested_min_xs,
                )
            df, aux_cols = _maybe_add_auxiliary_targets(config, df)
            print(f"[baseline] cache hit rows={len(df)} features={len(feature_cols)}", flush=True)
            if aux_cols:
                print(f"[baseline] auxiliary targets={aux_cols}", flush=True)
            return df, feature_cols, target_col
        print("[baseline] cache metadata missing or stale; rebuilding normalized frame", flush=True)
    print("[baseline] building labeled raw frame", flush=True)
    df = build_labeled_frame(
        data_dir=data_cfg["data_dir"],
        symbols=data_cfg.get("symbols"),
        excluded_symbols=data_cfg.get("excluded_symbols", ["T", "TF", "TS", "IF", "IC", "IH"]),
        max_symbols=data_cfg.get("max_symbols"),
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
        label_horizon=int(data_cfg.get("label_horizon", 30)),
    )
    print(f"[baseline] labeled rows={len(df)} symbols={df['symbol'].nunique()}", flush=True)
    df, raw_cols = add_stable_raw_columns(df, feature_set=data_cfg.get("raw_feature_set", "base"))
    norm_cfg = config.get("normalization", {})
    if norm_cfg.get("enabled", True) is False or str(norm_cfg.get("mode", "")).lower() == "none":
        feature_cols = raw_cols
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(
                "float32"
            )
        print(f"[baseline] using precomputed feature columns without rolling z-score features={len(feature_cols)}", flush=True)
    else:
        print(
            f"[baseline] applying causal rolling z-score window={norm_cfg.get('window', 240)} "
            f"min_periods={norm_cfg.get('min_periods', 20)}",
            flush=True,
        )
        df, feature_cols = add_rolling_zscore(
            df,
            raw_cols,
            window=int(norm_cfg.get("window", 240)),
            min_periods=int(norm_cfg.get("min_periods", 20)),
            clip_value=float(norm_cfg.get("clip_value", 8.0)),
            group_cols=tuple(norm_cfg.get("group_cols", ["symbol", "session_id"])),
        )
    df, target_col = add_target_transform(
        df,
        data_cfg.get("target_mode", "raw"),
        min_xs_count=int(data_cfg.get("target_min_xs_count", 8)),
    )
    df, aux_cols = _maybe_add_auxiliary_targets(config, df)
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        meta = {
            "feature_cols": feature_cols,
            "target_col": target_col,
            "target_mode": data_cfg.get("target_mode", "raw"),
            "target_min_xs_count": int(data_cfg.get("target_min_xs_count", 8)),
        }
        cache_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[baseline] wrote cache {cache_path}", flush=True)
    data_cfg["feature_cols"] = feature_cols
    data_cfg["target_col"] = target_col
    if aux_cols:
        print(f"[baseline] auxiliary targets={aux_cols}", flush=True)
    return df, feature_cols, target_col


def make_report(
    config: dict[str, Any],
    metrics: dict[str, Any],
    split_metrics: list[dict[str, Any]],
    table_paths: dict[str, str],
    plot_paths: list[str],
    report_path: str | Path,
) -> None:
    def simple_markdown_table(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No splits were scored."
        cols = list(rows[0].keys())
        out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in rows:
            vals = []
            for col in cols:
                val = row.get(col, "")
                if isinstance(val, float):
                    vals.append(f"{val:.6f}" if np.isfinite(val) else "nan")
                else:
                    vals.append(str(val))
            out.append("| " + " | ".join(vals) + " |")
        return "\n".join(out)

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    aux_cols = list(config.get("train", {}).get("aux_targets") or [])
    aux_note = (
        f" Auxiliary heads trained on: {', '.join(aux_cols)}."
        if aux_cols
        else ""
    )
    lines = [
        "# Stage 1 Baseline Report",
        "",
        f"Experiment: `{config['experiment']['name']}`",
        "",
        "## Architecture",
        "",
        "Raw 1-minute OHLCV/amount/OI -> strictly causal rolling z-score -> low-rank FM-style interaction block -> PatchTST-style patch embedding -> pre-norm Transformer -> learnable multi-layer output aggregation -> attention/mean/CLS pooling -> MLP prediction head."
        + aux_note,
        "",
        "## Key Metrics",
        "",
        f"- Merged test IC: `{metrics['merged_ic']:.6f}`",
        f"- Merged test RankIC: `{metrics['merged_rank_ic']:.6f}`",
        f"- Monthly ICIR: `{metrics['icir_monthly']:.6f}`",
        f"- Scored rows: `{metrics['n_scored']}`",
        "",
        "## Rolling Splits",
        "",
    ]
    if split_metrics:
        lines.append(simple_markdown_table(split_metrics))
    else:
        lines.append("No splits were scored.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, path in table_paths.items():
        lines.append(f"- {name}: `{path}`")
    for path in plot_paths:
        lines.append(f"- figure: `{path}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Rolling normalization is history-only via shifted rolling windows within symbol/session groups.",
            "- The baseline is intentionally compact for first-pass reliability; larger widths, more windows, and additional ideas should be tested through the ablation harness next.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _calibrate_split_predictions(
    config: dict[str, Any],
    model: torch.nn.Module,
    arrays,
    val_index: list[tuple[int, int]],
    pred_index: list[tuple[int, int]] | None,
    pred: np.ndarray,
    seq_len: int,
    y_mean: float,
    y_std: float,
    device: torch.device,
    prediction_options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    calib_cfg = config.get("prediction_calibration") or config.get("calibration") or {}
    if not bool(calib_cfg.get("enabled", False)) or not val_index:
        return pred, {}
    mode = str(calib_cfg.get("mode", "global_mean")).lower()
    if mode not in {"global_mean", "global_median", "global_ic_grid", "linear_blend_aux"}:
        raise ValueError(f"unknown prediction calibration mode: {mode}")
    if mode == "linear_blend_aux":
        if pred_index is None:
            return pred, {}
        base_options = dict(prediction_options or {})
        main_options = dict(base_options)
        main_options["output_mode"] = "main"
        val_main = predict_index(
            model,
            arrays,
            val_index,
            seq_len,
            y_mean,
            y_std,
            int(config["train"].get("pred_batch_size", config["train"].get("batch_size", 1024))),
            device,
            num_workers=int(config["train"].get("pred_num_workers", config["train"].get("num_workers", 2))),
            destandardize=bool(config["train"].get("destandardize_pred", True)),
            **main_options,
        )
        aux_indices = calib_cfg.get("aux_indices", calib_cfg.get("aux_index", 0))
        if not isinstance(aux_indices, (list, tuple)):
            aux_indices = [aux_indices]
        aux_indices = [int(i) for i in aux_indices]
        val_features = [np.asarray(val_main, dtype=np.float64)]
        test_features = [np.asarray(pred, dtype=np.float64)]
        feature_names = ["main"]
        for aux_idx in aux_indices:
            aux_options = dict(base_options)
            aux_options["output_mode"] = "aux_value"
            aux_options["scale_aux_indices"] = [aux_idx]
            val_aux = predict_index(
                model,
                arrays,
                val_index,
                seq_len,
                y_mean,
                y_std,
                int(config["train"].get("pred_batch_size", config["train"].get("batch_size", 1024))),
                device,
                num_workers=int(config["train"].get("pred_num_workers", config["train"].get("num_workers", 2))),
                destandardize=False,
                **aux_options,
            )
            test_aux = predict_index(
                model,
                arrays,
                pred_index,
                seq_len,
                y_mean,
                y_std,
                int(config["train"].get("pred_batch_size", config["train"].get("batch_size", 1024))),
                device,
                num_workers=int(config["train"].get("pred_num_workers", config["train"].get("num_workers", 2))),
                destandardize=False,
                **aux_options,
            )
            val_features.append(np.asarray(val_aux, dtype=np.float64))
            test_features.append(np.asarray(test_aux, dtype=np.float64))
            feature_names.append(f"aux{aux_idx}")
        val_df = index_to_prediction_frame(arrays, val_index, val_main)
        y = val_df["label"].to_numpy(dtype=np.float64)
        X = np.column_stack(val_features)
        X_test = np.column_stack(test_features)
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if int(mask.sum()) < max(20, X.shape[1] + 2):
            return pred, {"calibration_mode": mode, "calibration_status": "too_few_val_rows"}
        X_fit = X[mask]
        y_fit = y[mask]
        standardize = bool(calib_cfg.get("standardize_features", True))
        if standardize:
            mu = np.nanmean(X_fit, axis=0)
            sd = np.nanstd(X_fit, axis=0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            X_fit = (X_fit - mu) / sd
            X_test_use = (X_test - mu) / sd
        else:
            X_test_use = X_test
        fit_intercept = bool(calib_cfg.get("fit_intercept", True))
        if fit_intercept:
            X_fit = np.column_stack([X_fit, np.ones(len(X_fit), dtype=np.float64)])
            X_test_use = np.column_stack([X_test_use, np.ones(len(X_test_use), dtype=np.float64)])
        ridge = float(calib_cfg.get("ridge", 1e-4) or 0.0)
        reg = np.eye(X_fit.shape[1], dtype=np.float64) * ridge
        if fit_intercept:
            reg[-1, -1] = 0.0
        try:
            coef = np.linalg.solve(X_fit.T @ X_fit + reg, X_fit.T @ y_fit)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]
        coef_clip = float(calib_cfg.get("coef_clip", 0.0) or 0.0)
        if coef_clip > 0:
            last = coef[-1:] if fit_intercept else np.asarray([], dtype=np.float64)
            body = np.clip(coef[:-1] if fit_intercept else coef, -coef_clip, coef_clip)
            coef = np.concatenate([body, last]) if fit_intercept else body
        out = (X_test_use @ coef).astype(np.float32)
        val_pred = (X_fit @ coef).astype(np.float32)
        best_val_ic = compute_ic(val_pred, y_fit)
        info = {
            "calibration_mode": mode,
            "calibration_features": ",".join(feature_names),
            "calibration_val_ic": best_val_ic,
        }
        for i, value in enumerate(coef):
            key = f"calibration_coef_{feature_names[i]}" if i < len(feature_names) else "calibration_intercept"
            info[key] = float(value)
        print(
            f"[baseline] prediction_calibration mode={mode} features={feature_names} "
            f"val_ic={best_val_ic:.6f} coef={[float(v) for v in coef]}",
            flush=True,
        )
        return out, info
    val_pred = predict_index(
        model,
        arrays,
        val_index,
        seq_len,
        y_mean,
        y_std,
        int(config["train"].get("pred_batch_size", config["train"].get("batch_size", 1024))),
        device,
        num_workers=int(config["train"].get("pred_num_workers", config["train"].get("num_workers", 2))),
        destandardize=bool(config["train"].get("destandardize_pred", True)),
        **(prediction_options or {}),
    )
    center = float(np.nanmedian(val_pred) if mode == "global_median" else np.nanmean(val_pred))
    shrink = float(calib_cfg.get("shrinkage", 1.0))
    best_val_ic = float("nan")
    if mode == "global_ic_grid":
        val_df = index_to_prediction_frame(arrays, val_index, val_pred)
        basis = abs(center)
        if not np.isfinite(basis) or basis < 1e-12:
            basis = float(np.nanstd(val_pred))
        grid_min = float(calib_cfg.get("grid_min", -3.0))
        grid_max = float(calib_cfg.get("grid_max", 3.0))
        grid_steps = max(3, int(calib_cfg.get("grid_steps", 49)))
        candidates = np.linspace(grid_min, grid_max, grid_steps)
        best_score = -np.inf
        best_shrink = 0.0
        best_shift = 0.0
        labels = val_df["label"].to_numpy(dtype=float)
        for candidate in candidates:
            candidate_shift = float(candidate * basis)
            score = compute_ic(val_pred - candidate_shift, labels)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_shrink = float(candidate)
                best_shift = candidate_shift
        shrink = best_shrink
        shift = best_shift
        best_val_ic = float(best_score)
    else:
        shift = shrink * center
    out = np.asarray(pred, dtype=np.float32) - np.float32(shift)
    info = {
        "calibration_mode": mode,
        "calibration_shrinkage": shrink,
        "calibration_center": center,
        "calibration_shift": shift,
        "calibration_val_ic": best_val_ic,
    }
    print(
        f"[baseline] prediction_calibration mode={mode} center={center:.6e} "
        f"shrinkage={shrink:.3f} shift={shift:.6e} val_ic={best_val_ic:.6f}",
        flush=True,
    )
    return out, info


def _prediction_output_options(config: dict[str, Any], train_info: dict[str, Any]) -> dict[str, Any]:
    pred_cfg = config.get("prediction_output") or {}
    train_cfg = config.get("train", {})
    mode = str(pred_cfg.get("mode", train_cfg.get("prediction_output_mode", "main")) or "main")
    indices = pred_cfg.get("scale_aux_indices", train_cfg.get("scale_aux_indices"))
    if indices is not None:
        indices = [int(i) for i in indices]
    return {
        "output_mode": mode,
        "aux_mean": np.asarray(train_info.get("aux_mean", []), dtype=np.float32),
        "aux_std": np.asarray(train_info.get("aux_std", []), dtype=np.float32),
        "scale_aux_indices": indices,
        "scale_power": float(pred_cfg.get("scale_power", train_cfg.get("scale_power", 1.0)) or 1.0),
        "scale_min": float(pred_cfg.get("scale_min", train_cfg.get("scale_min", 0.25)) or 0.25),
        "scale_max": float(pred_cfg.get("scale_max", train_cfg.get("scale_max", 4.0)) or 4.0),
    }


def _postprocess_prediction_frame(config: dict[str, Any], split_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    post_cfg = config.get("prediction_postprocess") or {}
    mode = str(post_cfg.get("mode", "none")).lower()
    if mode in {"", "none"}:
        return split_df, {}
    out = split_df.copy()
    if bool(post_cfg.get("keep_raw_pred", False)) and "pred_raw" not in out.columns:
        out["pred_raw"] = out["pred"]
    if mode in {"datetime_demean", "cross_section_demean", "cs_demean"}:
        out["pred"] = out["pred"] - out.groupby("datetime")["pred"].transform("mean")
    elif mode in {"datetime_zscore", "cross_section_zscore", "cs_zscore"}:
        grouped = out.groupby("datetime")["pred"]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0.0, np.nan)
        out["pred"] = ((out["pred"] - mean) / (std + 1e-12)).fillna(0.0)
    elif mode == "symbol_demean":
        out["pred"] = out["pred"] - out.groupby("symbol")["pred"].transform("mean")
    else:
        raise ValueError(f"unknown prediction postprocess mode: {mode}")
    info = {"postprocess_mode": mode}
    print(f"[baseline] prediction_postprocess mode={mode}", flush=True)
    return out, info


def _cross_section_metric_summary(split_df: pd.DataFrame, min_count: int = 8) -> dict[str, Any]:
    ic_vals: list[float] = []
    rank_vals: list[float] = []
    for _dt, g in split_df.groupby("datetime", sort=False):
        if len(g) < int(min_count):
            continue
        pred = g["pred"].to_numpy(dtype=float)
        label = g["label"].to_numpy(dtype=float)
        ic = compute_ic(pred, label)
        rank_ic = compute_rank_ic(pred, label)
        if np.isfinite(ic):
            ic_vals.append(float(ic))
        if np.isfinite(rank_ic):
            rank_vals.append(float(rank_ic))

    def stats(vals: list[float], prefix: str) -> dict[str, Any]:
        if not vals:
            return {
                f"{prefix}_mean": float("nan"),
                f"{prefix}_std": float("nan"),
                f"{prefix}_ir": float("nan"),
                f"{prefix}_positive_frac": float("nan"),
                f"{prefix}_n": 0,
            }
        arr = np.asarray(vals, dtype=float)
        std = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else float("nan")
        mean = float(np.nanmean(arr))
        return {
            f"{prefix}_mean": mean,
            f"{prefix}_std": std,
            f"{prefix}_ir": mean / std if np.isfinite(std) and std > 0 else float("nan"),
            f"{prefix}_positive_frac": float(np.nanmean(arr > 0)),
            f"{prefix}_n": int(len(arr)),
        }

    out: dict[str, Any] = {}
    out.update(stats(ic_vals, "cs_ic"))
    out.update(stats(rank_vals, "cs_rank_ic"))
    return out


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config["experiment"].get("seed", 42))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and config["train"].get("device", "auto") != "cpu" else "cpu")
    run_root = Path(config["paths"]["runs_dir"]) / config["experiment"]["name"]
    ckpt_root = Path(config["paths"]["checkpoints_dir"]) / config["experiment"]["name"]
    fig_root = Path(config["paths"]["figures_dir"]) / config["experiment"]["name"]
    report_path = Path(config["paths"]["reports_dir"]) / config["experiment"].get("report_name", "01_baseline_report.md")
    run_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    print(f"[baseline] config={config_path} device={device}", flush=True)
    df, feature_cols, target_col = prepare_frame(config)
    df = df.dropna(subset=["label", target_col]).reset_index(drop=True)
    aux_cols = list(config["data"].get("aux_target_cols") or config.get("train", {}).get("aux_targets") or [])
    arrays, symbol_to_id = prepare_symbol_arrays(df, feature_cols, target_col, aux_cols=aux_cols)
    data_start = pd.Timestamp(config["rolling"].get("data_start", df["datetime"].min()))
    data_end = pd.Timestamp(config["rolling"].get("data_end", df["datetime"].max())) + pd.Timedelta(nanoseconds=1)
    splits = make_quarterly_splits(
        data_start=data_start,
        data_end=data_end,
        train_start=config["rolling"].get("train_start", "2018-01-01"),
        first_test_start=config["rolling"].get("first_test_start", "2020-01-01"),
        freq_months=int(config["rolling"].get("freq_months", 3)),
        allow_partial_test=bool(config["rolling"].get("allow_partial_test", True)),
    )
    max_splits = config["rolling"].get("max_splits")
    if max_splits:
        splits = splits[: int(max_splits)]
    print(
        f"[baseline] rows={len(df)} symbols={len(arrays)} features={len(feature_cols)} "
        f"aux_targets={len(aux_cols)} splits={len(splits)}",
        flush=True,
    )

    all_preds = []
    split_metrics = []
    val_months = int(config["train"].get("val_months", 3))
    allow_short = bool(config["data"].get("allow_short_windows", True))
    embargo_days = int(config["train"].get("embargo_days", config["rolling"].get("embargo_days", 0)) or 0)
    train_anchor_stride = int(config["train"].get("train_anchor_stride", config["train"].get("anchor_stride", 1)) or 1)
    val_anchor_stride = int(config["train"].get("val_anchor_stride", train_anchor_stride) or 1)
    pred_anchor_stride = int(config["train"].get("pred_anchor_stride", 1) or 1)
    for split in splits:
        val_start = max(split.train_start, split.train_end - pd.DateOffset(months=val_months))
        train_end = val_start - pd.DateOffset(days=embargo_days) if embargo_days > 0 else val_start
        val_end = split.train_end - pd.DateOffset(days=embargo_days) if embargo_days > 0 else split.train_end
        train_index = build_window_index(
            arrays,
            config["data"]["seq_len"],
            split.train_start,
            train_end,
            require_target=True,
            allow_short=allow_short,
            anchor_stride=train_anchor_stride,
        )
        val_index = build_window_index(
            arrays,
            config["data"]["seq_len"],
            val_start,
            val_end,
            require_target=True,
            allow_short=allow_short,
            anchor_stride=val_anchor_stride,
        )
        pred_index = build_window_index(
            arrays,
            config["data"]["seq_len"],
            split.test_start,
            split.test_end,
            require_target=True,
            allow_short=allow_short,
            anchor_stride=pred_anchor_stride,
        )
        max_pred = config["train"].get("max_pred_windows")
        if max_pred and len(pred_index) > int(max_pred):
            chosen = rng.choice(len(pred_index), size=int(max_pred), replace=False)
            pred_index = [pred_index[int(i)] for i in chosen]
        if len(train_index) < int(config["train"].get("min_train_windows", 1000)) or not pred_index:
            print(f"[baseline][{split.name}] skipped train={len(train_index)} pred={len(pred_index)}", flush=True)
            continue
        model, y_mean, y_std, train_info = train_split(
            config,
            arrays,
            feature_cols,
            split,
            train_index,
            val_index,
            ckpt_root / f"{split.name}.pt",
            rng,
            device,
        )
        prediction_options = _prediction_output_options(config, train_info)
        prediction_mode = str(prediction_options.get("output_mode", "main")).lower()
        timestamp_scale = prediction_mode in {"rank_times_datetime_aux_scale", "score_times_datetime_aux_scale"}
        main_prediction_options = dict(prediction_options)
        if timestamp_scale:
            main_prediction_options["output_mode"] = "main"
        pred = predict_index(
            model,
            arrays,
            pred_index,
            config["data"]["seq_len"],
            y_mean,
            y_std,
            int(config["train"].get("pred_batch_size", 1024)),
            device,
            num_workers=int(config["train"].get("pred_num_workers", config["train"].get("num_workers", 2))),
            destandardize=bool(config["train"].get("destandardize_pred", True)),
            **main_prediction_options,
        )
        pred, calib_info = _calibrate_split_predictions(
            config,
            model,
            arrays,
            val_index,
            pred_index,
            pred,
            config["data"]["seq_len"],
            y_mean,
            y_std,
            device,
            prediction_options=main_prediction_options,
        )
        split_df = index_to_prediction_frame(arrays, pred_index, pred)
        split_df["split"] = split.name
        split_df = split_df.sort_values(["datetime", "symbol"]).reset_index(drop=True)
        if timestamp_scale:
            scale_options = dict(prediction_options)
            scale_options["output_mode"] = "aux_scale"
            aux_scale = predict_index(
                model,
                arrays,
                pred_index,
                config["data"]["seq_len"],
                y_mean,
                y_std,
                int(config["train"].get("pred_batch_size", 1024)),
                device,
                num_workers=int(config["train"].get("pred_num_workers", config["train"].get("num_workers", 2))),
                destandardize=False,
                **scale_options,
            )
            split_df["_aux_scale"] = aux_scale.astype(np.float32)
            dt_scale = split_df.groupby("datetime")["_aux_scale"].transform("mean").clip(
                float(prediction_options.get("scale_min", 0.25)),
                float(prediction_options.get("scale_max", 4.0)),
            )
            split_df["pred"] = split_df["pred"] * dt_scale
        split_df, post_info = _postprocess_prediction_frame(config, split_df)
        split_df.to_parquet(run_root / f"predictions_{split.name}.parquet", index=False)
        ic = compute_ic(split_df["pred"], split_df["label"])
        rank_ic = compute_rank_ic(split_df["pred"], split_df["label"])
        cs_info = _cross_section_metric_summary(
            split_df,
            min_count=int(config.get("evaluation", {}).get("min_xs_count", config["data"].get("target_min_xs_count", 8))),
        )
        sm = {
            "split": split.name,
            "train_start": str(split.train_start.date()),
            "train_end": str(split.train_end.date()),
            "effective_train_end": str(pd.Timestamp(train_end).date()),
            "effective_val_end": str(pd.Timestamp(val_end).date()),
            "test_start": str(split.test_start.date()),
            "test_end": str(split.test_end.date()),
            "train_windows": int(train_info["train_windows"]),
            "val_windows": int(train_info["val_windows"]),
            "test_windows": int(len(split_df)),
            "ic": ic,
            "rank_ic": rank_ic,
        }
        sm.update(calib_info)
        sm.update(post_info)
        sm.update(cs_info)
        split_metrics.append(sm)
        all_preds.append(split_df)
        print(f"[baseline][{split.name}] test_rows={len(split_df)} IC={ic:.4f} RankIC={rank_ic:.4f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_preds:
        raise RuntimeError("no predictions generated")
    pred_df = pd.concat(all_preds, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    pred_df.to_parquet(run_root / "predictions_with_label.parquet", index=False)
    pred_df[["symbol", "datetime", "pred"]].to_parquet(run_root / "predictions.parquet", index=False)
    table_paths = write_metric_tables(pred_df, run_root, split_metrics)
    metrics = summarize_predictions(pred_df, split_metrics)
    plot_paths = make_standard_plots(pred_df, fig_root, config["experiment"]["name"])
    make_report(config, metrics, split_metrics, table_paths, plot_paths, report_path)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[baseline] report={report_path}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
