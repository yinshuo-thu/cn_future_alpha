from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .data import build_window_index, make_quarterly_splits, prepare_symbol_arrays
from .memory_fusion import RegimeMemory, _load_checkpoint_model, _resolve_feature_cols, _source_frame
from .metrics import compute_ic, compute_rank_ic, summarize_predictions, write_metric_tables
from .residual_memory_fusion import _fit_residual_memory, _sample_index
from .rolling import load_config, make_report, prepare_frame
from .train import predict_index
from .visualize import make_standard_plots


def _best_pair_weight(
    primary: np.ndarray,
    secondary: np.ndarray,
    label: np.ndarray,
    grid: list[float],
) -> tuple[float, float]:
    best_w = 0.0
    best_ic = compute_ic(primary, label)
    for w in grid:
        pred = (1.0 - w) * primary + w * secondary
        ic = compute_ic(pred, label)
        if np.isfinite(ic) and ic > best_ic:
            best_ic = float(ic)
            best_w = float(w)
    return best_w, best_ic


def _scale_to_reference(
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    ref_val: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    stats = {
        "val_mean": float(np.mean(val_pred)) if len(val_pred) else 0.0,
        "val_std": float(np.std(val_pred)) if len(val_pred) else 0.0,
        "ref_mean": float(np.mean(ref_val)) if len(ref_val) else 0.0,
        "ref_std": float(np.std(ref_val)) if len(ref_val) else 0.0,
    }
    if not enabled:
        return val_pred.astype(np.float64), test_pred.astype(np.float64), stats
    cand_std = max(stats["val_std"], 1e-8)
    ref_std = max(stats["ref_std"], 1e-8)
    val_scaled = (val_pred - stats["val_mean"]) / cand_std * ref_std + stats["ref_mean"]
    test_scaled = (test_pred - stats["val_mean"]) / cand_std * ref_std + stats["ref_mean"]
    return val_scaled.astype(np.float64), test_scaled.astype(np.float64), stats


def _stability_stats(
    current: np.ndarray,
    candidate: np.ndarray,
    label: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    deltas: list[float] = []
    for group in pd.unique(groups):
        mask = groups == group
        if int(mask.sum()) < 2:
            continue
        cur_ic = compute_ic(current[mask], label[mask])
        cand_ic = compute_ic(candidate[mask], label[mask])
        if np.isfinite(cur_ic) and np.isfinite(cand_ic):
            deltas.append(float(cand_ic - cur_ic))
    if not deltas:
        return {
            "positive_groups": 0.0,
            "n_groups": 0.0,
            "min_group_delta": 0.0,
            "last_group_delta": 0.0,
        }
    return {
        "positive_groups": float(sum(delta > 0.0 for delta in deltas)),
        "n_groups": float(len(deltas)),
        "min_group_delta": float(min(deltas)),
        "last_group_delta": float(deltas[-1]),
    }


def _passes_stability(stats: dict[str, float], cfg: dict[str, Any]) -> bool:
    if not cfg.get("enabled", False):
        return True
    n_groups = int(stats.get("n_groups", 0.0))
    if n_groups <= 0:
        return False
    min_positive = min(int(cfg.get("min_positive_groups", 1)), n_groups)
    if stats.get("positive_groups", 0.0) < float(min_positive):
        return False
    if stats.get("min_group_delta", 0.0) < float(cfg.get("max_group_decline", -1e9)):
        return False
    if stats.get("last_group_delta", 0.0) < float(cfg.get("min_last_group_delta", -1e9)):
        return False
    return True


def _apply_symbol_guard(
    current_val: np.ndarray,
    current_test: np.ndarray,
    label: np.ndarray,
    val_symbols: np.ndarray,
    test_symbols: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not cfg.get("enabled", False):
        return current_val, current_test, []
    min_count = int(cfg.get("min_val_count", 5000))
    ic_lte_raw = cfg.get("shrink_ic_lte", -0.01)
    ic_gte_raw = cfg.get("shrink_ic_gte")
    ic_lte = float(ic_lte_raw) if ic_lte_raw is not None else None
    ic_gte = float(ic_gte_raw) if ic_gte_raw is not None else None
    shrink_weight = float(cfg.get("shrink_weight", 0.0))
    max_symbols = int(cfg.get("max_symbols", 0) or 0)
    default_min_weight = -1.0 if bool(cfg.get("allow_negative_weight", False)) else 0.0
    min_weight = float(cfg.get("min_shrink_weight", default_min_weight))
    max_weight = float(cfg.get("max_shrink_weight", 1.0))
    if not bool(cfg.get("allow_negative_weight", False)):
        min_weight = max(0.0, min_weight)
    shrink_weight = min(max_weight, max(min_weight, shrink_weight))
    raw_weight_grid = cfg.get("weight_grid")
    weight_grid = None
    if raw_weight_grid:
        weight_grid = [min(max_weight, max(min_weight, float(weight))) for weight in raw_weight_grid]
        weight_grid = sorted(set(weight_grid))
    guarded_val = current_val.copy()
    guarded_test = current_test.copy()
    rows: list[dict[str, Any]] = []
    eligible: list[tuple[float, str, np.ndarray, np.ndarray]] = []
    for symbol in sorted(pd.unique(val_symbols)):
        val_mask = val_symbols == symbol
        if int(val_mask.sum()) < min_count:
            continue
        test_mask = test_symbols == symbol
        if int(test_mask.sum()) <= 0:
            continue
        sym_ic = compute_ic(current_val[val_mask], label[val_mask])
        if not np.isfinite(sym_ic):
            continue
        eligible_by_lte = ic_lte is not None and sym_ic <= ic_lte
        eligible_by_gte = ic_gte is not None and sym_ic >= ic_gte
        if not eligible_by_lte and not eligible_by_gte:
            continue
        eligible.append((float(sym_ic), str(symbol), val_mask, test_mask))
    sort_desc = ic_lte is None and ic_gte is not None
    eligible = sorted(eligible, key=lambda item: item[0], reverse=sort_desc)
    if max_symbols > 0:
        eligible = eligible[:max_symbols]
    for sym_ic, symbol, val_mask, test_mask in eligible:
        chosen_weight = shrink_weight
        if weight_grid:
            best_ic = -np.inf
            best_weight = shrink_weight
            for weight in weight_grid:
                trial = guarded_val.copy()
                trial[val_mask] = float(weight) * guarded_val[val_mask]
                ic = compute_ic(trial, label)
                if np.isfinite(ic) and ic > best_ic:
                    best_ic = float(ic)
                    best_weight = float(weight)
            chosen_weight = best_weight
        guarded_val[val_mask] = chosen_weight * guarded_val[val_mask]
        guarded_test[test_mask] = chosen_weight * guarded_test[test_mask]
        rows.append(
            {
                "symbol_guard": True,
                "guard_symbol": symbol,
                "guard_symbol_weight": float(chosen_weight),
                "guard_symbol_val_ic": float(sym_ic),
                "guard_symbol_val_count": int(val_mask.sum()),
                "guard_symbol_test_count": int(test_mask.sum()),
            }
        )
    return guarded_val, guarded_test, rows


def _rescale_subset_to_reference(
    cand_val: np.ndarray,
    cand_test: np.ndarray,
    ref_val: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return cand_val, cand_test
    cand_mean = float(np.mean(cand_val)) if len(cand_val) else 0.0
    cand_std = max(float(np.std(cand_val)) if len(cand_val) else 0.0, 1e-8)
    ref_mean = float(np.mean(ref_val)) if len(ref_val) else 0.0
    ref_std = max(float(np.std(ref_val)) if len(ref_val) else 0.0, 1e-8)
    return (
        (cand_val - cand_mean) / cand_std * ref_std + ref_mean,
        (cand_test - cand_mean) / cand_std * ref_std + ref_mean,
    )


def _apply_candidate_symbol_guard(
    current_val: np.ndarray,
    current_test: np.ndarray,
    label: np.ndarray,
    val_symbols: np.ndarray,
    test_symbols: np.ndarray,
    cand_payload: dict[str, tuple[np.ndarray, np.ndarray, dict[str, float]]],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not cfg.get("enabled", False) or not cand_payload:
        return current_val, current_test, []
    min_count = int(cfg.get("min_val_count", 5000))
    ic_lte_raw = cfg.get("trigger_ic_lte", cfg.get("shrink_ic_lte", -0.01))
    ic_gte_raw = cfg.get("trigger_ic_gte", cfg.get("shrink_ic_gte"))
    ic_lte = float(ic_lte_raw) if ic_lte_raw is not None else None
    ic_gte = float(ic_gte_raw) if ic_gte_raw is not None else None
    max_symbols = int(cfg.get("max_symbols", 0) or 0)
    candidate_names = cfg.get("candidates")
    allowed_names = {str(name) for name in candidate_names} if candidate_names else set(cand_payload)
    allowed_names = sorted(allowed_names & set(cand_payload))
    if not allowed_names:
        return current_val, current_test, []
    weight_grid = [float(weight) for weight in cfg.get("weight_grid", [0.05, 0.1, 0.15, 0.2])]
    if bool(cfg.get("include_zero_weight", True)):
        weight_grid.append(0.0)
    weight_grid = sorted(set(weight_grid))
    min_full_delta = float(cfg.get("min_full_val_delta", 0.0))
    min_group_delta = float(cfg.get("min_group_val_delta", 0.0))
    max_full_drop = float(cfg.get("max_full_val_ic_drop", 0.0))
    rescale = bool(cfg.get("rescale_to_current_symbol", False))

    guarded_val = current_val.copy()
    guarded_test = current_test.copy()
    rows: list[dict[str, Any]] = []
    eligible: list[tuple[float, str, np.ndarray, np.ndarray]] = []
    for symbol in sorted(pd.unique(val_symbols)):
        val_mask = val_symbols == symbol
        if int(val_mask.sum()) < min_count:
            continue
        test_mask = test_symbols == symbol
        if int(test_mask.sum()) <= 0:
            continue
        sym_ic = compute_ic(guarded_val[val_mask], label[val_mask])
        if not np.isfinite(sym_ic):
            continue
        eligible_by_lte = ic_lte is not None and sym_ic <= ic_lte
        eligible_by_gte = ic_gte is not None and sym_ic >= ic_gte
        if not eligible_by_lte and not eligible_by_gte:
            continue
        eligible.append((float(sym_ic), str(symbol), val_mask, test_mask))
    sort_desc = ic_lte is None and ic_gte is not None
    eligible = sorted(eligible, key=lambda item: item[0], reverse=sort_desc)
    if max_symbols > 0:
        eligible = eligible[:max_symbols]

    for sym_ic, symbol, val_mask, test_mask in eligible:
        base_full_ic = compute_ic(guarded_val, label)
        base_group_ic = compute_ic(guarded_val[val_mask], label[val_mask])
        best: tuple[float, float, str | None, float, np.ndarray, np.ndarray, float] = (
            base_full_ic,
            base_group_ic,
            None,
            0.0,
            guarded_val[val_mask],
            guarded_test[test_mask],
            base_group_ic,
        )
        for name in allowed_names:
            cand_val, cand_test, _stats = cand_payload[name]
            cand_val_sub, cand_test_sub = _rescale_subset_to_reference(
                cand_val[val_mask],
                cand_test[test_mask],
                guarded_val[val_mask],
                rescale,
            )
            for weight in weight_grid:
                blended_val_sub = (1.0 - weight) * guarded_val[val_mask] + weight * cand_val_sub
                group_ic = compute_ic(blended_val_sub, label[val_mask])
                if not np.isfinite(group_ic) or group_ic < base_group_ic + min_group_delta:
                    continue
                trial = guarded_val.copy()
                trial[val_mask] = blended_val_sub
                full_ic = compute_ic(trial, label)
                if not np.isfinite(full_ic):
                    continue
                if full_ic < base_full_ic - max_full_drop:
                    continue
                if full_ic < base_full_ic + min_full_delta:
                    continue
                if full_ic > best[0] or (full_ic == best[0] and group_ic > best[1]):
                    blended_test_sub = (1.0 - weight) * guarded_test[test_mask] + weight * cand_test_sub
                    best = (
                        float(full_ic),
                        float(group_ic),
                        name,
                        float(weight),
                        blended_val_sub,
                        blended_test_sub,
                        float(group_ic),
                    )
        if best[2] is None or best[3] == 0.0:
            continue
        guarded_val[val_mask] = best[4]
        guarded_test[test_mask] = best[5]
        rows.append(
            {
                "candidate_symbol_guard": True,
                "guard_symbol": symbol,
                "guard_candidate": best[2],
                "guard_symbol_weight": best[3],
                "guard_symbol_val_ic": float(sym_ic),
                "guard_symbol_val_ic_after": float(best[6]),
                "guard_full_val_ic_before": float(base_full_ic),
                "guard_full_val_ic_after": float(best[0]),
                "guard_symbol_val_count": int(val_mask.sum()),
                "guard_symbol_test_count": int(test_mask.sum()),
            }
        )
    return guarded_val, guarded_test, rows


def _symbol_guard_allowed(
    cfg: dict[str, Any],
    freeze: str | None,
    val_ic: float,
    memory_weight: float,
    split_name: str,
) -> bool:
    if not cfg.get("enabled", False):
        return False
    allowed_reasons = cfg.get("apply_freeze_reasons")
    if allowed_reasons and (freeze is None or str(freeze) not in {str(reason) for reason in allowed_reasons}):
        return False
    allowed_splits = cfg.get("apply_splits")
    if allowed_splits and split_name not in {str(name) for name in allowed_splits}:
        return False
    min_ic = cfg.get("current_val_ic_gte")
    max_ic = cfg.get("current_val_ic_lte")
    min_mem = cfg.get("memory_weight_gte")
    max_mem = cfg.get("memory_weight_lte")
    if min_ic is not None and val_ic < float(min_ic):
        return False
    if max_ic is not None and val_ic > float(max_ic):
        return False
    if min_mem is not None and memory_weight < float(min_mem):
        return False
    if max_mem is not None and memory_weight > float(max_mem):
        return False
    if not allowed_reasons and cfg.get("require_freeze", False) and freeze is None:
        return False
    return True


def _month_offsets(values: pd.Series | np.ndarray) -> np.ndarray:
    periods = pd.to_datetime(values).dt.to_period("M").astype(str) if isinstance(values, pd.Series) else pd.to_datetime(pd.Series(values)).dt.to_period("M").astype(str)
    unique = {period: idx + 1 for idx, period in enumerate(pd.unique(periods))}
    return periods.map(unique).to_numpy(dtype=np.int16)


def _cross_section_transform(pred: np.ndarray, datetimes: pd.Series | np.ndarray, mode: str) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(datetimes),
            "pred": np.asarray(pred, dtype=np.float64),
        }
    )
    mode = (mode or "zscore").lower()
    if mode in {"zscore", "z"}:
        group = frame.groupby("datetime", sort=False)["pred"]
        mean = group.transform("mean")
        std = group.transform("std").replace(0.0, np.nan)
        out = ((frame["pred"] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out.to_numpy(dtype=np.float64)
    if mode == "rank":
        group = frame.groupby("datetime", sort=False)["pred"]
        rank = group.rank(method="average")
        count = group.transform("count").astype(float)
        out = ((rank - 1.0) / (count - 1.0) - 0.5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out.to_numpy(dtype=np.float64)
    raise ValueError(f"unknown cross-section postprocess mode: {mode}")


def _apply_cross_section_postprocess(
    current_val: np.ndarray,
    current_test: np.ndarray,
    label: np.ndarray,
    val_datetimes: pd.Series | np.ndarray,
    test_datetimes: pd.Series | np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    if not cfg.get("enabled", False):
        return current_val, current_test, None
    mode = str(cfg.get("mode", "zscore"))
    val_cs = _cross_section_transform(current_val, val_datetimes, mode)
    test_cs = _cross_section_transform(current_test, test_datetimes, mode)
    if bool(cfg.get("rescale_to_current", True)):
        val_cs, test_cs = _rescale_subset_to_reference(val_cs, test_cs, current_val, True)
    base_ic = compute_ic(current_val, label)
    best_ic = base_ic
    best_weight = 0.0
    best_val = current_val
    best_test = current_test
    for weight in [float(w) for w in cfg.get("weight_grid", [-0.1, -0.05, 0.0, 0.05, 0.1])]:
        trial_val = (1.0 - weight) * current_val + weight * val_cs
        ic = compute_ic(trial_val, label)
        if np.isfinite(ic) and ic > best_ic:
            best_ic = float(ic)
            best_weight = float(weight)
            best_val = trial_val
            best_test = (1.0 - weight) * current_test + weight * test_cs
    min_delta = float(cfg.get("min_val_delta", 0.0))
    if best_weight == 0.0 or best_ic < base_ic + min_delta:
        return current_val, current_test, None
    return (
        best_val,
        best_test,
        {
            "cross_section_postprocess": True,
            "cs_mode": mode,
            "cs_weight": best_weight,
            "cs_val_ic_before": float(base_ic),
            "cs_val_ic_after": float(best_ic),
        },
    )


def _freeze_reason(primary_val_ic: float, memory_weight: float, rules: list[dict[str, Any]]) -> str | None:
    for rule in rules:
        name = str(rule.get("name", "freeze_rule"))
        min_ic = rule.get("primary_val_ic_gte")
        max_ic = rule.get("primary_val_ic_lte")
        min_mem = rule.get("memory_weight_gte")
        max_mem = rule.get("memory_weight_lte")
        if min_ic is not None and primary_val_ic < float(min_ic):
            continue
        if max_ic is not None and primary_val_ic > float(max_ic):
            continue
        if min_mem is not None and memory_weight < float(min_mem):
            continue
        if max_mem is not None and memory_weight > float(max_mem):
            continue
        return name
    return None


def _freeze_rule(primary_val_ic: float, memory_weight: float, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    reason = _freeze_reason(primary_val_ic, memory_weight, rules)
    if reason is None:
        return None
    for rule in rules:
        if str(rule.get("name", "freeze_rule")) == reason:
            return rule
    return {"name": reason}


def _candidate_val_pred(
    cfg: dict[str, Any],
    run_name: str,
    split_name: str,
    arrays,
    val_index: list[tuple[int, int]],
    seq_len: int,
    feature_cols: list[str],
    device: torch.device,
    pred_batch_size: int,
    num_workers: int,
    cache_path: Path,
    use_cache: bool,
) -> np.ndarray:
    if use_cache and cache_path.exists():
        return np.load(cache_path).astype(np.float32)
    ckpt_root = Path(cfg["paths"]["checkpoints_dir"]) / run_name
    model, y_mean, y_std = _load_checkpoint_model(
        ckpt_root / f"{split_name}.pt",
        len(feature_cols),
        len(arrays),
        feature_cols,
        device,
    )
    pred = predict_index(
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
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, pred.astype(np.float32))
    return pred.astype(np.float32)


def _candidate_memory_val_pred(
    memory_config: dict[str, Any],
    split_name: str,
    arrays,
    memory_train_index: list[tuple[int, int]],
    val_index: list[tuple[int, int]],
    seq_len: int,
    feature_cols: list[str],
    df: pd.DataFrame,
    device: torch.device,
    pred_batch_size: int,
    num_workers: int,
    cache_path: Path,
    use_cache: bool,
) -> np.ndarray:
    if use_cache and cache_path.exists():
        return np.load(cache_path).astype(np.float32)
    mem_cfg = memory_config["memory"]
    memory_cols = _resolve_feature_cols(feature_cols, list(mem_cfg["feature_cols"]))
    cols = ["symbol", "datetime", "label", *memory_cols]
    train_mem = _source_frame(df, arrays, memory_train_index, cols)
    val_source = _source_frame(df, arrays, val_index, cols)
    memory = RegimeMemory(
        feature_cols=memory_cols,
        quantiles=list(mem_cfg.get("quantiles", [0.2, 0.4, 0.6, 0.8])),
        time_bucket_minutes=int(mem_cfg.get("time_bucket_minutes", 30)),
        min_count=int(mem_cfg.get("min_count", 30)),
        shrink=float(mem_cfg.get("shrink", 100.0)),
        decay_halflife_days=mem_cfg.get("decay_halflife_days"),
    ).fit(train_mem)
    val_mem_pred = memory.predict(val_source)
    base_config = load_config(mem_cfg["base_config_path"])
    base_ckpt_root = Path(base_config["paths"]["checkpoints_dir"]) / mem_cfg["base_run_name"]
    base_model, y_mean, y_std = _load_checkpoint_model(
        base_ckpt_root / f"{split_name}.pt",
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
    label = val_source["label"].to_numpy(dtype=np.float64)
    memory_weight, _memory_val_ic = _best_pair_weight(
        val_base_pred.astype(np.float64),
        val_mem_pred.astype(np.float64),
        label,
        [float(w) for w in mem_cfg.get("blend_grid", [0.0, 0.05, 0.1, 0.15, 0.2])],
    )
    pred = (1.0 - memory_weight) * val_base_pred + memory_weight * val_mem_pred
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, pred.astype(np.float32))
    return pred.astype(np.float32)


def _stable_split_seed(seed: int, split_name: str) -> int:
    offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(split_name))
    return int(seed + offset) % (2**32 - 1)


def _candidate_residual_memory_val_pred(
    residual_config: dict[str, Any],
    split_name: str,
    arrays,
    memory_train_index: list[tuple[int, int]],
    val_index: list[tuple[int, int]],
    seq_len: int,
    feature_cols: list[str],
    df: pd.DataFrame,
    device: torch.device,
    pred_batch_size: int,
    num_workers: int,
    cache_path: Path,
    use_cache: bool,
) -> np.ndarray:
    if use_cache and cache_path.exists():
        return np.load(cache_path).astype(np.float32)
    mem_cfg = residual_config["memory"]
    base_config = load_config(mem_cfg["base_config_path"])
    base_run = str(mem_cfg["base_run_name"])
    memory_cols = _resolve_feature_cols(feature_cols, list(mem_cfg["feature_cols"]))
    cols = ["symbol", "datetime", "label", *memory_cols]
    seed = _stable_split_seed(int(residual_config["experiment"].get("seed", 42)), split_name)
    rng = np.random.default_rng(seed)
    max_fit = int(mem_cfg.get("max_fit_windows", 750_000))
    fit_index = _sample_index(memory_train_index, max_fit, rng)
    fit_source = _source_frame(df, arrays, fit_index, cols)
    val_source = _source_frame(df, arrays, val_index, cols)
    base_ckpt_root = Path(base_config["paths"]["checkpoints_dir"]) / base_run
    model, y_mean, y_std = _load_checkpoint_model(
        base_ckpt_root / f"{split_name}.pt",
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
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    residual_clip = mem_cfg.get("residual_clip")
    residual_memory = _fit_residual_memory(
        fit_source,
        fit_model_pred,
        mem_cfg,
        memory_cols,
        residual_clip=float(residual_clip) if residual_clip is not None else None,
    )
    val_resid_pred = residual_memory.predict(val_source).astype(np.float64)
    label = val_source["label"].to_numpy(dtype=np.float64)
    grid = [float(w) for w in mem_cfg.get("residual_weight_grid", [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])]
    best_w = 0.0
    best_ic = compute_ic(val_model_pred, label)
    for weight in grid:
        pred = val_model_pred.astype(np.float64) + float(weight) * val_resid_pred
        ic = compute_ic(pred, label)
        if np.isfinite(ic) and ic > best_ic:
            best_ic = float(ic)
            best_w = float(weight)
    pred = val_model_pred.astype(np.float64) + best_w * val_resid_pred
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, pred.astype(np.float32))
    return pred.astype(np.float32)


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    selector_cfg = config["selector"]
    memory_config = load_config(selector_cfg["primary_memory_config_path"])
    memory_cfg = memory_config["memory"]
    base_config = load_config(memory_cfg["base_config_path"])
    seed = int(config["experiment"].get("seed", 42))
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")

    run_root = Path(config["paths"]["runs_dir"]) / config["experiment"]["name"]
    fig_root = Path(config["paths"]["figures_dir"]) / config["experiment"]["name"]
    report_path = Path(config["paths"]["reports_dir"]) / config["experiment"]["report_name"]
    cache_root = Path(selector_cfg.get("val_cache_dir") or (run_root / "val_cache"))
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

    primary_memory_run = selector_cfg.get("primary_memory_run_name") or memory_config["experiment"]["name"]
    primary_run_root = Path(memory_config["paths"]["runs_dir"]) / primary_memory_run
    base_run_name = memory_cfg["base_run_name"]
    base_ckpt_root = Path(base_config["paths"]["checkpoints_dir"]) / base_run_name
    seq_len = int(base_config["data"]["seq_len"])
    allow_short = bool(base_config["data"].get("allow_short_windows", True))
    val_months = int(base_config["train"].get("val_months", 3))
    pred_batch_size = int(selector_cfg.get("pred_batch_size", base_config["train"].get("pred_batch_size", 1536)))
    num_workers = int(selector_cfg.get("num_workers", base_config["train"].get("num_workers", 4)))
    weight_grid = [float(w) for w in selector_cfg.get("weight_grid", [0.02, 0.05, 0.08, 0.1, 0.15])]
    max_steps = int(selector_cfg.get("max_steps", 3))
    min_delta = float(selector_cfg.get("min_delta", 0.0))
    scale_candidates = bool(selector_cfg.get("scale_candidates_to_primary", True))
    use_cache = bool(selector_cfg.get("use_val_cache", True))
    candidates = list(selector_cfg.get("candidates", []))
    candidate_weight_grids = {
        str(cand["name"]): [float(weight) for weight in cand.get("weight_grid", weight_grid)]
        for cand in candidates
    }
    candidate_array_cache: dict[str, tuple[pd.DataFrame, list, list[str], int, bool]] = {}
    stability_cfg = dict(selector_cfg.get("stability", {}))
    freeze_rules = list(selector_cfg.get("freeze_rules", []))

    def _uses_primary_feature_schema(candidate_config: dict[str, Any]) -> bool:
        cand_data = candidate_config.get("data", {})
        base_data = base_config.get("data", {})
        keys = ("cache_path", "raw_feature_set", "target_mode")
        return all(str(cand_data.get(key)) == str(base_data.get(key)) for key in keys)

    def _model_candidate_val_pred(
        cand_config: dict[str, Any],
        run_name: str,
        name: str,
        split_name: str,
        val_start: pd.Timestamp,
        val_end: pd.Timestamp,
        val_source_ref: pd.DataFrame,
    ) -> np.ndarray:
        if _uses_primary_feature_schema(cand_config):
            return _candidate_val_pred(
                cand_config,
                run_name,
                split_name,
                arrays,
                val_index,
                seq_len,
                feature_cols,
                device,
                pred_batch_size,
                num_workers,
                cache_root / f"{split_name}__{name}.npy",
                use_cache,
            )
        if name not in candidate_array_cache:
            cand_df, cand_feature_cols, cand_target_col = prepare_frame(cand_config)
            cand_df = cand_df.dropna(subset=["label", cand_target_col]).reset_index(drop=True)
            cand_arrays, _ = prepare_symbol_arrays(cand_df, cand_feature_cols, cand_target_col)
            cand_seq_len = int(cand_config["data"]["seq_len"])
            cand_allow_short = bool(cand_config["data"].get("allow_short_windows", True))
            candidate_array_cache[name] = (cand_df, cand_arrays, cand_feature_cols, cand_seq_len, cand_allow_short)
        cand_df, cand_arrays, cand_feature_cols, cand_seq_len, cand_allow_short = candidate_array_cache[name]
        cand_val_index = build_window_index(
            cand_arrays,
            cand_seq_len,
            val_start,
            val_end,
            True,
            cand_allow_short,
        )
        raw_pred = _candidate_val_pred(
            cand_config,
            run_name,
            split_name,
            cand_arrays,
            cand_val_index,
            cand_seq_len,
            cand_feature_cols,
            device,
            pred_batch_size,
            num_workers,
            cache_root / f"{split_name}__{name}.npy",
            use_cache,
        )
        cand_val_frame = _source_frame(cand_df, cand_arrays, cand_val_index, ["symbol", "datetime"])
        aligned = val_source_ref[["symbol", "datetime"]].merge(
            cand_val_frame.assign(_candidate_pred=raw_pred.astype(np.float32)),
            on=["symbol", "datetime"],
            how="left",
            validate="one_to_one",
        )
        if aligned["_candidate_pred"].isna().any():
            missing = int(aligned["_candidate_pred"].isna().sum())
            raise RuntimeError(f"missing aligned validation predictions for {name} / {split_name}: {missing}")
        return aligned["_candidate_pred"].to_numpy(dtype=np.float32)

    all_preds: list[pd.DataFrame] = []
    split_metrics: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for split in splits:
        print(f"[validation_branch_selector][{split.name}] preparing validation signals", flush=True)
        val_start = max(split.train_start, split.train_end - pd.DateOffset(months=val_months))
        memory_train_index = build_window_index(arrays, seq_len, split.train_start, val_start, True, allow_short)
        val_index = build_window_index(arrays, seq_len, val_start, split.train_end, True, allow_short)

        train_mem = _source_frame(df, arrays, memory_train_index, cols)
        val_source = _source_frame(df, arrays, val_index, cols)
        label = val_source["label"].to_numpy(dtype=np.float64)
        val_groups = pd.to_datetime(val_source["datetime"]).dt.to_period("M").astype(str).to_numpy()

        memory = RegimeMemory(
            feature_cols=memory_cols,
            quantiles=list(memory_cfg.get("quantiles", [0.2, 0.4, 0.6, 0.8])),
            time_bucket_minutes=int(memory_cfg.get("time_bucket_minutes", 30)),
            min_count=int(memory_cfg.get("min_count", 30)),
            shrink=float(memory_cfg.get("shrink", 100.0)),
            decay_halflife_days=memory_cfg.get("decay_halflife_days"),
        ).fit(train_mem)
        val_mem_pred = memory.predict(val_source)
        base_model, y_mean, y_std = _load_checkpoint_model(
            base_ckpt_root / f"{split.name}.pt",
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

        memory_weight, memory_val_ic = _best_pair_weight(
            val_base_pred.astype(np.float64),
            val_mem_pred.astype(np.float64),
            label,
            [float(w) for w in memory_cfg.get("blend_grid", [0.0, 0.05, 0.1, 0.15, 0.2])],
        )
        current_val = (1.0 - memory_weight) * val_base_pred + memory_weight * val_mem_pred
        current_val = current_val.astype(np.float64)
        current_ic = compute_ic(current_val, label)

        primary_test = pd.read_parquet(primary_run_root / f"predictions_{split.name}.parquet")
        current_test = primary_test["pred"].to_numpy(dtype=np.float64)
        test_df = primary_test.copy()
        test_df["primary_pred"] = current_test

        cand_payload: dict[str, tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}
        fallback_only_names: set[str] = set()
        for cand in candidates:
            name = str(cand["name"])
            allowed_splits = cand.get("apply_splits")
            if allowed_splits and split.name not in {str(item) for item in allowed_splits}:
                continue
            if bool(cand.get("fallback_only", False)):
                fallback_only_names.add(name)
            cand_config = load_config(cand["config_path"])
            run_name = str(cand.get("run_name") or cand_config["experiment"]["name"])
            cand_type = str(cand.get("type", "model")).lower()
            if cand_type == "memory":
                val_pred = _candidate_memory_val_pred(
                    cand_config,
                    split.name,
                    arrays,
                    memory_train_index,
                    val_index,
                    seq_len,
                    feature_cols,
                    df,
                    device,
                    pred_batch_size,
                    num_workers,
                    cache_root / f"{split.name}__{name}.npy",
                    use_cache,
                )
            elif cand_type == "residual_memory":
                val_pred = _candidate_residual_memory_val_pred(
                    cand_config,
                    split.name,
                    arrays,
                    memory_train_index,
                    val_index,
                    seq_len,
                    feature_cols,
                    df,
                    device,
                    pred_batch_size,
                    num_workers,
                    cache_root / f"{split.name}__{name}.npy",
                    use_cache,
                )
            else:
                val_pred = _model_candidate_val_pred(
                    cand_config,
                    run_name,
                    name,
                    split.name,
                    val_start,
                    split.train_end,
                    val_source,
                )
            cand_test = pd.read_parquet(Path(cand_config["paths"]["runs_dir"]) / run_name / f"predictions_{split.name}.parquet")
            merged = test_df[["symbol", "datetime"]].merge(
                cand_test[["symbol", "datetime", "pred"]].rename(columns={"pred": name}),
                on=["symbol", "datetime"],
                how="left",
                validate="one_to_one",
            )
            if merged[name].isna().any():
                raise RuntimeError(f"missing candidate predictions for {name} / {split.name}")
            val_scaled, test_scaled, stats = _scale_to_reference(
                val_pred.astype(np.float64),
                merged[name].to_numpy(dtype=np.float64),
                current_val,
                enabled=scale_candidates,
            )
            cand_payload[name] = (val_scaled, test_scaled, stats)
            test_df[f"{name}_pred"] = merged[name].to_numpy(dtype=np.float64)

        selected: list[dict[str, Any]] = []
        remaining = set(cand_payload) - fallback_only_names
        freeze_rule = _freeze_rule(float(current_ic), float(memory_weight), freeze_rules)
        freeze = str(freeze_rule.get("name")) if freeze_rule is not None else None
        if freeze_rule is not None and freeze_rule.get("fallback_candidates"):
            allowed = {str(name) for name in freeze_rule.get("fallback_candidates", [])}
            fallback_grid = [float(w) for w in freeze_rule.get("fallback_weight_grid", weight_grid)]
            max_drop = float(freeze_rule.get("fallback_max_val_ic_drop", 0.0))
            prefer_high_weight = bool(freeze_rule.get("fallback_prefer_high_weight", True))
            fallback_best: tuple[float, float, str | None, np.ndarray, np.ndarray] = (
                -np.inf,
                -np.inf,
                None,
                current_val,
                current_test,
            )
            for name in sorted(allowed & set(cand_payload)):
                val_pred, test_pred, _stats = cand_payload[name]
                for weight in fallback_grid:
                    blended_val = (1.0 - weight) * current_val + weight * val_pred
                    ic = compute_ic(blended_val, label)
                    if not np.isfinite(ic) or ic < current_ic - max_drop:
                        continue
                    score = weight if prefer_high_weight else ic
                    tie = ic if prefer_high_weight else weight
                    if score > fallback_best[0] or (score == fallback_best[0] and tie > fallback_best[1]):
                        blended_test = (1.0 - weight) * current_test + weight * test_pred
                        fallback_best = (float(score), float(tie), name, blended_val, blended_test)
            if fallback_best[2] is not None:
                _score, _tie, name, current_val, current_test = fallback_best
                current_ic = compute_ic(current_val, label)
                selected.append(
                    {
                        "step": 0,
                        "candidate": name,
                        "weight": _score if prefer_high_weight else _tie,
                        "val_ic_after": current_ic,
                        "freeze_reason": freeze,
                        "fallback": True,
                    }
                )
            else:
                selected.append(
                    {
                        "step": 0,
                        "candidate": "none",
                        "weight": 0.0,
                        "val_ic_after": current_ic,
                        "freeze_reason": freeze,
                        "fallback": False,
                    }
                )
            remaining.clear()
        elif freeze is not None:
            remaining.clear()
            selected.append(
                {
                    "step": 0,
                    "candidate": "none",
                    "weight": 0.0,
                    "val_ic_after": current_ic,
                    "freeze_reason": freeze,
                    "fallback": False,
                }
            )
        for step in range(max_steps):
            best: tuple[float, str | None, float, np.ndarray, np.ndarray, dict[str, float]] = (
                current_ic,
                None,
                0.0,
                current_val,
                current_test,
                {
                    "positive_groups": 0.0,
                    "n_groups": 0.0,
                    "min_group_delta": 0.0,
                    "last_group_delta": 0.0,
                },
            )
            for name in sorted(remaining):
                val_pred, test_pred, _stats = cand_payload[name]
                for weight in candidate_weight_grids.get(name, weight_grid):
                    blended_val = (1.0 - weight) * current_val + weight * val_pred
                    ic = compute_ic(blended_val, label)
                    stable = _stability_stats(current_val, blended_val, label, val_groups)
                    if not _passes_stability(stable, stability_cfg):
                        continue
                    if np.isfinite(ic) and ic > best[0]:
                        blended_test = (1.0 - weight) * current_test + weight * test_pred
                        best = (float(ic), name, float(weight), blended_val, blended_test, stable)
            if best[1] is None or best[0] < current_ic + min_delta:
                break
            current_ic, name, weight, current_val, current_test, stable = best
            remaining.remove(name)
            selected.append(
                {
                    "step": step + 1,
                    "candidate": name,
                    "weight": weight,
                    "val_ic_after": current_ic,
                    "stability_positive_groups": stable["positive_groups"],
                    "stability_n_groups": stable["n_groups"],
                    "stability_min_group_delta": stable["min_group_delta"],
                    "stability_last_group_delta": stable["last_group_delta"],
                    **{f"{name}_{k}": v for k, v in cand_payload[name][2].items()},
                }
            )

        if freeze_rule is not None and freeze_rule.get("fallback_month_overrides"):
            val_month_offsets = _month_offsets(val_source["datetime"])
            test_month_offsets = _month_offsets(test_df["datetime"])
            for override in freeze_rule.get("fallback_month_overrides", []):
                name = str(override["candidate"])
                if name not in cand_payload:
                    continue
                month_offset = int(override.get("month_offset", 1))
                val_mask = val_month_offsets == month_offset
                test_mask = test_month_offsets == month_offset
                if int(val_mask.sum()) < 2 or int(test_mask.sum()) < 1:
                    continue
                val_pred, test_pred, _stats = cand_payload[name]
                val_pred_month = val_pred[val_mask]
                test_pred_month = test_pred[test_mask]
                if bool(override.get("rescale_to_current_month", False)):
                    cand_mean = float(np.mean(val_pred_month))
                    cand_std = max(float(np.std(val_pred_month)), 1e-8)
                    ref_mean = float(np.mean(current_val[val_mask]))
                    ref_std = max(float(np.std(current_val[val_mask])), 1e-8)
                    val_pred_month = (val_pred_month - cand_mean) / cand_std * ref_std + ref_mean
                    test_pred_month = (test_pred_month - cand_mean) / cand_std * ref_std + ref_mean
                month_base_ic = compute_ic(current_val[val_mask], label[val_mask])
                if not np.isfinite(month_base_ic):
                    continue
                month_grid = [float(w) for w in override.get("weight_grid", weight_grid)]
                min_delta_month = float(override.get("min_val_delta", 0.0))
                max_drop_month = float(override.get("max_val_ic_drop", 0.0))
                prefer_high_weight = bool(override.get("prefer_high_weight", True))
                month_best: tuple[float, float, float, float, np.ndarray, np.ndarray] | None = None
                for weight in month_grid:
                    blended_val_month = (1.0 - weight) * current_val[val_mask] + weight * val_pred_month
                    month_ic = compute_ic(blended_val_month, label[val_mask])
                    if not np.isfinite(month_ic):
                        continue
                    if month_ic < month_base_ic + min_delta_month:
                        continue
                    if month_ic < month_base_ic - max_drop_month:
                        continue
                    score = weight if prefer_high_weight else month_ic
                    tie = month_ic if prefer_high_weight else weight
                    if month_best is None or score > month_best[0] or (score == month_best[0] and tie > month_best[1]):
                        blended_test_month = (1.0 - weight) * current_test[test_mask] + weight * test_pred_month
                        month_best = (
                            float(score),
                            float(tie),
                            float(weight),
                            float(month_ic),
                            blended_val_month,
                            blended_test_month,
                        )
                if month_best is None:
                    continue
                _score, _tie, weight, month_ic, blended_val_month, blended_test_month = month_best
                current_val = current_val.copy()
                current_test = current_test.copy()
                current_val[val_mask] = blended_val_month
                current_test[test_mask] = blended_test_month
                current_ic = compute_ic(current_val, label)
                selected.append(
                    {
                        "step": len(selected) + 1,
                        "candidate": name,
                        "weight": weight,
                        "val_ic_after": current_ic,
                        "freeze_reason": freeze,
                        "fallback": True,
                        "month_override": str(override.get("name", f"month_{month_offset}")),
                        "month_offset": month_offset,
                        "month_val_ic_after": month_ic,
                        "month_val_ic_before": month_base_ic,
                    }
                )

        raw_candidate_symbol_guard_cfg = dict(selector_cfg.get("candidate_symbol_guard", {}))
        candidate_guard_cfgs = [dict(rule) for rule in raw_candidate_symbol_guard_cfg.get("rules", [])] or [
            raw_candidate_symbol_guard_cfg
        ]
        for guard_cfg in candidate_guard_cfgs:
            if not _symbol_guard_allowed(guard_cfg, freeze, float(current_ic), float(memory_weight), split.name):
                continue
            if guard_cfg.get("by_month_offset", False):
                guard_val = current_val.copy()
                guard_test = current_test.copy()
                guard_rows: list[dict[str, Any]] = []
                val_guard_offsets = _month_offsets(val_source["datetime"])
                test_guard_offsets = _month_offsets(test_df["datetime"])
                allowed_guard_offsets = guard_cfg.get("apply_month_offsets")
                allowed_guard_offsets = (
                    {int(offset) for offset in allowed_guard_offsets}
                    if allowed_guard_offsets
                    else None
                )
                for guard_offset in sorted(set(val_guard_offsets).intersection(set(test_guard_offsets))):
                    if allowed_guard_offsets is not None and int(guard_offset) not in allowed_guard_offsets:
                        continue
                    val_mask = val_guard_offsets == guard_offset
                    test_mask = test_guard_offsets == guard_offset
                    sub_payload = {
                        name: (payload[0][val_mask], payload[1][test_mask], payload[2])
                        for name, payload in cand_payload.items()
                    }
                    sub_val, sub_test, sub_rows = _apply_candidate_symbol_guard(
                        current_val[val_mask],
                        current_test[test_mask],
                        label[val_mask],
                        val_source.loc[val_mask, "symbol"].to_numpy(),
                        test_df.loc[test_mask, "symbol"].to_numpy(),
                        sub_payload,
                        guard_cfg,
                    )
                    if not sub_rows:
                        continue
                    guard_val[val_mask] = sub_val
                    guard_test[test_mask] = sub_test
                    for row in sub_rows:
                        row["guard_month_offset"] = int(guard_offset)
                    guard_rows.extend(sub_rows)
            else:
                guard_val, guard_test, guard_rows = _apply_candidate_symbol_guard(
                    current_val,
                    current_test,
                    label,
                    val_source["symbol"].to_numpy(),
                    test_df["symbol"].to_numpy(),
                    cand_payload,
                    guard_cfg,
                )
            if guard_rows:
                current_val = guard_val
                current_test = guard_test
                current_ic = compute_ic(current_val, label)
                for guard_row in guard_rows:
                    selected.append(
                        {
                            "step": len(selected) + 1,
                            "candidate": "candidate_symbol_guard",
                            "weight": float(guard_row.get("guard_symbol_weight", 0.0)),
                            "val_ic_after": current_ic,
                            **guard_row,
                        }
                    )

        raw_symbol_guard_cfg = dict(selector_cfg.get("symbol_guard", {}))
        guard_cfgs = [dict(rule) for rule in raw_symbol_guard_cfg.get("rules", [])] or [raw_symbol_guard_cfg]
        for guard_cfg in guard_cfgs:
            if not _symbol_guard_allowed(guard_cfg, freeze, float(current_ic), float(memory_weight), split.name):
                continue
            if guard_cfg.get("by_month_offset", False):
                guard_val = current_val.copy()
                guard_test = current_test.copy()
                guard_rows: list[dict[str, Any]] = []
                val_guard_offsets = _month_offsets(val_source["datetime"])
                test_guard_offsets = _month_offsets(test_df["datetime"])
                allowed_guard_offsets = guard_cfg.get("apply_month_offsets")
                allowed_guard_offsets = (
                    {int(offset) for offset in allowed_guard_offsets}
                    if allowed_guard_offsets
                    else None
                )
                for guard_offset in sorted(set(val_guard_offsets).intersection(set(test_guard_offsets))):
                    if allowed_guard_offsets is not None and int(guard_offset) not in allowed_guard_offsets:
                        continue
                    val_mask = val_guard_offsets == guard_offset
                    test_mask = test_guard_offsets == guard_offset
                    sub_val, sub_test, sub_rows = _apply_symbol_guard(
                        current_val[val_mask],
                        current_test[test_mask],
                        label[val_mask],
                        val_source.loc[val_mask, "symbol"].to_numpy(),
                        test_df.loc[test_mask, "symbol"].to_numpy(),
                        guard_cfg,
                    )
                    if not sub_rows:
                        continue
                    guard_val[val_mask] = sub_val
                    guard_test[test_mask] = sub_test
                    for row in sub_rows:
                        row["guard_month_offset"] = int(guard_offset)
                    guard_rows.extend(sub_rows)
            else:
                guard_val, guard_test, guard_rows = _apply_symbol_guard(
                    current_val,
                    current_test,
                    label,
                    val_source["symbol"].to_numpy(),
                    test_df["symbol"].to_numpy(),
                    guard_cfg,
                )
            if guard_rows:
                current_val = guard_val
                current_test = guard_test
                current_ic = compute_ic(current_val, label)
                for guard_row in guard_rows:
                    selected.append(
                        {
                            "step": len(selected) + 1,
                            "candidate": "symbol_guard",
                            "weight": float(guard_row.get("guard_symbol_weight", guard_cfg.get("shrink_weight", 0.0))),
                            "val_ic_after": current_ic,
                            **guard_row,
                        }
                    )

        raw_cs_cfg = dict(selector_cfg.get("cross_section_postprocess", {}))
        cs_cfgs = [dict(rule) for rule in raw_cs_cfg.get("rules", [])] or [raw_cs_cfg]
        for cs_cfg in cs_cfgs:
            cs_apply_splits = cs_cfg.get("apply_splits")
            if cs_apply_splits and split.name not in {str(item) for item in cs_apply_splits}:
                continue
            if not _symbol_guard_allowed(
                cs_cfg,
                freeze,
                float(current_ic),
                float(memory_weight),
                split.name,
            ):
                continue
            current_val, current_test, cs_row = _apply_cross_section_postprocess(
                current_val,
                current_test,
                label,
                val_source["datetime"],
                test_df["datetime"],
                cs_cfg,
            )
            if cs_row is None:
                continue
            current_ic = compute_ic(current_val, label)
            selected.append(
                {
                    "step": len(selected) + 1,
                    "candidate": "cross_section_postprocess",
                    "weight": float(cs_row.get("cs_weight", 0.0)),
                    "val_ic_after": current_ic,
                    "cs_rule": str(cs_cfg.get("name", "cross_section_postprocess")),
                    **cs_row,
                }
            )

        for scale_rule in [dict(rule) for rule in dict(selector_cfg.get("split_scale", {})).get("rules", [])]:
            if not _symbol_guard_allowed(scale_rule, freeze, float(current_ic), float(memory_weight), split.name):
                continue
            scale = float(scale_rule.get("scale", 1.0))
            current_val = current_val * scale
            current_test = current_test * scale
            selected.append(
                {
                    "step": len(selected) + 1,
                    "candidate": "split_scale",
                    "weight": scale,
                    "val_ic_after": current_ic,
                    "split_scale": scale,
                    "split_scale_rule": str(scale_rule.get("name", "split_scale")),
                }
            )
            if bool(scale_rule.get("stop_after", True)):
                break

        test_df["pred"] = current_test
        test_df["split"] = split.name
        test_df.to_parquet(run_root / f"predictions_{split.name}.parquet", index=False)
        ic = compute_ic(test_df["pred"], test_df["label"])
        rank_ic = compute_rank_ic(test_df["pred"], test_df["label"])
        primary_ic = compute_ic(test_df["primary_pred"], test_df["label"])
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
        if not selected:
            selected = [{"step": 0, "candidate": "none", "weight": 0.0, "val_ic_after": current_ic, "freeze_reason": ""}]
        for row in selected:
            weight_rows.append(
                {
                    "split": split.name,
                    "memory_weight": memory_weight,
                    "memory_val_ic": memory_val_ic,
                    "primary_test_ic": primary_ic,
                    "final_test_ic": ic,
                    "final_test_rank_ic": rank_ic,
                    **row,
                }
            )
        all_preds.append(test_df)
        pick_msg = ", ".join(f"{r['candidate']}@{r['weight']:.2f}" for r in selected if r["candidate"] != "none")
        print(
            f"[validation_branch_selector][{split.name}] val_ic={current_ic:.4f} "
            f"primary_ic={primary_ic:.4f} final_ic={ic:.4f} picks={pick_msg or 'none'}",
            flush=True,
        )

    pred_df = pd.concat(all_preds, ignore_index=True).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    pred_df.to_parquet(run_root / "predictions_with_label.parquet", index=False)
    pred_df[["symbol", "datetime", "pred"]].to_parquet(run_root / "predictions.parquet", index=False)
    pd.DataFrame(weight_rows).to_csv(run_root / "branch_weights.csv", index=False)
    table_paths = write_metric_tables(pred_df, run_root, split_metrics)
    table_paths["branch_weights"] = str(run_root / "branch_weights.csv")
    metrics = summarize_predictions(pred_df, split_metrics)
    (run_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_paths = make_standard_plots(pred_df, fig_root, config["experiment"]["name"])
    make_report(config, metrics, split_metrics, table_paths, plot_paths, report_path)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[validation_branch_selector] report={report_path}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
