#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import ndtri

ROOT = Path("/root/autodl-tmp/quant/ML")
OUT_DIR = ROOT / "agent_runs" / "current_three_model_ensemble_20260629"

MLP_CAL_PATH = ROOT / "agent_runs" / "mlp_parallel_20260628" / "mlp_2019only_calibration_experiments.py"
MLP_SOURCE = ROOT / "effective_rolling_results" / "mlp_overlap333_xsz_hl12_n1200k" / "month_parts"
MLP_SELECTED = ROOT / "agent_runs" / "mlp_parallel_20260628" / "time120_slope_a025_strong" / "selected_config.json"

RIDGE_POST_PATH = ROOT / "agent_runs" / "ridge_parallel_20260628" / "postprocess_view_scan.py"

LGB_SEQ_DIR = ROOT / "agent_runs" / "lgb_reg8_shape_seqcal_20260628"
LGB_PASS_CONFIG = LGB_SEQ_DIR / "recent_weak_selector_pass_config.json"
LGB_2020_PASS = LGB_SEQ_DIR / "ref_time90_a1_then_signed_abs12_a0.8_recent_weak_selector_winner_audit2020_predictions.parquet"

BEST_ARTIFACT = Path("/root/autodl-tmp/quant/artifacts/predictions_best_ic0716.parquet")
CORE_ARTIFACT = Path("/root/autodl-tmp/quant/artifacts/predictions_core_moe_noDL_ic0617.parquet")
EXPANDED_CLEAN_SUMMARY = ROOT / "strict_opt_results" / "expanded_history_gate_clean" / "stack_summary.csv"


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mlpcal = import_from_path("mlp_calibration_experiments_current_ensemble", MLP_CAL_PATH)
ridgepost = import_from_path("ridge_postprocess_current_ensemble", RIDGE_POST_PATH)
if str(LGB_SEQ_DIR) not in sys.path:
    sys.path.insert(0, str(LGB_SEQ_DIR))
import lgb_shape_seqcal_reg8 as lgbseq  # noqa: E402


def compute_ic(pred: np.ndarray | pd.Series, label: np.ndarray | pd.Series) -> float:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(p) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    p = p[mask]
    y = y[mask]
    den = math.sqrt(float(np.mean(p * p) * np.mean(y * y)))
    if den <= 1e-18:
        return float("nan")
    return float(np.mean(p * y) / den)


def monthly_ic(df: pd.DataFrame, col: str) -> pd.Series:
    tmp = df.loc[df[col].notna() & df["label"].notna(), ["datetime", "label", col]].copy()
    if tmp.empty:
        return pd.Series(dtype=np.float64)
    tmp["_month"] = pd.to_datetime(tmp["datetime"]).dt.strftime("%Y-%m")
    return tmp.groupby("_month", sort=True).apply(
        lambda g: compute_ic(g[col].to_numpy(), g["label"].to_numpy()),
        include_groups=False,
    )


def summarize_pred(df: pd.DataFrame, col: str, prefix: str = "") -> dict[str, float | int]:
    mic = monthly_ic(df, col)
    std = float(mic.std(ddof=1)) if len(mic) > 1 else float("nan")
    key = f"{prefix}_" if prefix else ""
    return {
        f"{key}rows": int(len(df)),
        f"{key}label_rows": int(df["label"].notna().sum()),
        f"{key}pooled_ic": compute_ic(df[col].to_numpy(), df["label"].to_numpy()),
        f"{key}monthly_mean": float(mic.mean()) if len(mic) else float("nan"),
        f"{key}monthly_std": std,
        f"{key}monthly_ir": float(mic.mean() / std) if np.isfinite(std) and std > 0 else float("nan"),
    }


def month_strings(start: str, end: str) -> list[str]:
    # Inclusive month endpoints.
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def period_mask(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    dt = pd.to_datetime(df["datetime"])
    return (dt >= pd.Timestamp(start)) & (dt < pd.Timestamp(end))


def mlp_candidate():
    return mlpcal.candidate_raw_center_z()


def fit_mlp(train_months: list[str]):
    candidate = mlp_candidate()
    stats = mlpcal.base.stats_for_months(MLP_SOURCE, train_months)
    weights, fit_ic = mlpcal.base.fit_candidate(stats, candidate, train_months)
    cfg = json.loads(MLP_SELECTED.read_text())["calibration"]
    cal_cfg = mlpcal.CalConfig(
        name=str(cfg["name"]),
        kind=str(cfg["kind"]),
        alpha=float(cfg["alpha"]),
        bucket_minutes=int(cfg["bucket_minutes"]),
        min_count=int(cfg["min_count"]),
        k=float(cfg["k"]),
        clip_low=float(cfg["clip_low"]),
        clip_high=float(cfg["clip_high"]),
    )
    calibrator = mlpcal.fit_multiplier_calibrator(MLP_SOURCE, train_months, candidate, weights, cal_cfg)
    return candidate, weights, calibrator, float(fit_ic)


def predict_mlp(months: list[str], fitted) -> pd.DataFrame:
    candidate, weights, calibrator, _ = fitted
    parts = []
    for month in months:
        frame, pred = mlpcal.load_month_prediction(MLP_SOURCE, month, candidate, weights)
        pred = calibrator.apply(frame, pred)
        out = frame[["symbol", "datetime", "label"]].copy()
        out["mlp"] = pred.astype(np.float32)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def lgb_candidate_from_pass_config() -> lgbseq.Candidate:
    cfg = json.loads(LGB_PASS_CONFIG.read_text())
    return lgbseq.Candidate(str(cfg["candidate"]), tuple(lgbseq.StageSpec(**x) for x in cfg["stages"]))


def fit_predict_lgb(train: pd.DataFrame, target: pd.DataFrame, fit_json: str, cand: lgbseq.Candidate) -> np.ndarray:
    shape_fit = lgbseq.fit_shape_transform(train, fit_json)
    pred_train_shape = lgbseq.transform_shape(train, shape_fit)
    pred_target_shape = lgbseq.transform_shape(target, shape_fit)
    pred_target, _ = lgbseq.fit_apply_stages(train, pred_train_shape, target, pred_target_shape, cand.stages)
    return pred_target.astype(np.float32, copy=False)


def lgb_frame(train_start: str, train_end: str, target_start: str, target_end: str, d19: pd.DataFrame, d20: pd.DataFrame) -> pd.DataFrame:
    selected = pd.read_csv(lgbseq.SHAPE_SELECTED).iloc[0]
    fit_json = str(selected["fit_json"])
    cand = lgb_candidate_from_pass_config()
    train = lgbseq.subset(d19, train_start, train_end)
    if target_start >= "2020-01-01":
        target = lgbseq.subset(d20, target_start, target_end)
        # Use the already materialized best 2020 pass if the window is full 2020.
        if target_start == "2020-01-01" and target_end == "2021-01-01":
            pass20 = pd.read_parquet(
                LGB_2020_PASS,
                columns=["symbol", "datetime", "label", "pred_lgb_recent_weak_selector"],
            )
            pass20["datetime"] = pd.to_datetime(pass20["datetime"])
            pass20 = pass20.rename(columns={"pred_lgb_recent_weak_selector": "lgb"})
            return pass20
    else:
        target = lgbseq.subset(d19, target_start, target_end)
    pred = fit_predict_lgb(train, target, fit_json, cand)
    out = target[["symbol", "datetime", "label"]].copy()
    out["lgb"] = pred
    return out


RIDGE_COLS = ["pred", "pred_xcenter", "pred_xsz", "pred_xrank"]


def load_ridge_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = ridgepost.load_pred(ridgepost.FIT_PATH, "2019-01-01", "2020-01-01")
    apply = ridgepost.load_pred(ridgepost.APPLY_PATH, "2020-01-01", "2021-01-01")
    ridgepost.add_views(fit)
    ridgepost.add_views(apply)
    return fit, apply


def ridge_frame(train_start: str, train_end: str, target_start: str, target_end: str, fit_src: pd.DataFrame, apply_src: pd.DataFrame) -> pd.DataFrame:
    train = fit_src.loc[period_mask(fit_src, train_start, train_end)]
    if target_start >= "2020-01-01":
        target = apply_src.loc[period_mask(apply_src, target_start, target_end)].copy()
    else:
        target = fit_src.loc[period_mask(fit_src, target_start, target_end)].copy()
    w, _ = ridgepost.simplex_fit(
        train[RIDGE_COLS].to_numpy(np.float64, copy=False),
        train["label"].to_numpy(np.float64, copy=False),
    )
    out = target[["symbol", "datetime", "label"]].copy()
    out["ridge"] = (target[RIDGE_COLS].to_numpy(np.float64, copy=False) @ w).astype(np.float32)
    return out


def merge_components(mlp: pd.DataFrame, lgb: pd.DataFrame, ridge: pd.DataFrame) -> pd.DataFrame:
    base = mlp[["symbol", "datetime", "label", "mlp"]].copy()
    base["datetime"] = pd.to_datetime(base["datetime"])
    for name, df in [("lgb", lgb), ("ridge", ridge)]:
        part = df[["symbol", "datetime", name]].copy()
        part["datetime"] = pd.to_datetime(part["datetime"])
        base = base.merge(part, on=["symbol", "datetime"], how="inner")
    return base


def add_component_views(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["mlp", "lgb", "ridge"]:
        g = out.groupby("datetime", sort=False)[col]
        mean = g.transform("mean").astype(np.float64)
        std = g.transform("std").astype(np.float64)
        z = ((out[col].astype(np.float64) - mean) / (std + 1e-9)).astype(np.float32)
        rank = g.rank(pct=True).astype(np.float64).clip(0.001, 0.999)
        out[f"{col}_xcenter"] = (out[col].astype(np.float64) - mean).astype(np.float32)
        out[f"{col}_xsz"] = z
        out[f"{col}_xrank"] = (rank - 0.5).astype(np.float32)
        out[f"{col}_rankgauss"] = ndtri(rank).clip(-3.1, 3.1).astype(np.float32)
        out[f"{col}_tanh"] = np.tanh(z.astype(np.float64) / 2.0).astype(np.float32)
    return out


def fit_equal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    del x, y
    return np.full(n, 1.0 / n, dtype=np.float64)


def fit_top1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ics = np.asarray([compute_ic(x[:, i], y) for i in range(x.shape[1])], dtype=np.float64)
    idx = int(np.nanargmax(ics))
    w = np.zeros(x.shape[1], dtype=np.float64)
    w[idx] = 1.0
    return w


def fit_ic_weight(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ics = np.asarray([max(0.0, compute_ic(x[:, i], y)) for i in range(x.shape[1])], dtype=np.float64)
    if not np.isfinite(ics).any() or float(np.nansum(ics)) <= 0:
        return fit_equal(x, y)
    ics = np.nan_to_num(ics, nan=0.0)
    return ics / ics.sum()


def fit_simplex(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, _ = ridgepost.simplex_fit(x, y)
    return np.asarray(w, dtype=np.float64)


def fit_signed_ridge(alpha: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def _fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        xs = x[mask]
        ys = y[mask]
        mean = xs.mean(axis=0)
        scale = xs.std(axis=0) + 1e-9
        z = (xs - mean) / scale
        gram = z.T @ z / len(z)
        cov = z.T @ ys / len(z)
        coef = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), cov)
        # Store transformed-space coefficients converted to raw-space linear weights.
        return coef / scale

    return _fit


def fit_capped_grid(cap: float, step: float = 0.02) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def _fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = x.shape[1]
        if n > 3:
            return fit_simplex(x, y)
        best_w = fit_equal(x, y)
        best_ic = compute_ic(x @ best_w, y)
        if n == 2:
            vals = np.arange(0.0, 1.0 + 1e-12, step)
            for a in vals:
                w = np.asarray([a, 1.0 - a], dtype=np.float64)
                if np.max(w) > cap + 1e-12:
                    continue
                ic = compute_ic(x @ w, y)
                if np.isfinite(ic) and ic > best_ic:
                    best_ic, best_w = ic, w
        elif n == 3:
            vals = np.arange(0.0, 1.0 + 1e-12, step)
            for a in vals:
                for b in vals:
                    c = 1.0 - a - b
                    if c < -1e-12:
                        continue
                    w = np.asarray([a, b, max(0.0, c)], dtype=np.float64)
                    if abs(w.sum() - 1.0) > 1e-9 or np.max(w) > cap + 1e-12:
                        continue
                    ic = compute_ic(x @ w, y)
                    if np.isfinite(ic) and ic > best_ic:
                        best_ic, best_w = ic, w
        return best_w

    return _fit


@dataclass(frozen=True)
class EnsembleSpec:
    name: str
    cols: tuple[str, ...]
    method: str
    fit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    postcal: tuple[int, float] | None = None


def build_specs() -> list[EnsembleSpec]:
    comps = ["mlp", "lgb", "ridge"]
    sets: dict[str, list[str]] = {
        "raw3": comps,
        "xcenter3": [f"{c}_xcenter" for c in comps],
        "xsz3": [f"{c}_xsz" for c in comps],
        "xrank3": [f"{c}_xrank" for c in comps],
        "rankgauss3": [f"{c}_rankgauss" for c in comps],
        "tanh3": [f"{c}_tanh" for c in comps],
        "raw_xsz6": comps + [f"{c}_xsz" for c in comps],
        "xsz_rank6": [f"{c}_xsz" for c in comps] + [f"{c}_xrank" for c in comps],
        "mlp_lgb_raw2": ["mlp", "lgb"],
        "mlp_lgb_xsz2": ["mlp_xsz", "lgb_xsz"],
        "mlp_lgb_rank2": ["mlp_rankgauss", "lgb_rankgauss"],
    }
    methods: list[tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = [
        ("equal", fit_equal),
        ("top1", fit_top1),
        ("icpos", fit_ic_weight),
        ("simplex", fit_simplex),
        ("signed_ridge_a001", fit_signed_ridge(0.01)),
        ("signed_ridge_a01", fit_signed_ridge(0.1)),
        ("signed_ridge_a1", fit_signed_ridge(1.0)),
    ]
    out: list[EnsembleSpec] = []
    for set_name, cols in sets.items():
        for method, fit_fn in methods:
            out.append(EnsembleSpec(f"{set_name}__{method}", tuple(cols), method, fit_fn))
    cal_sets = {
        "raw3": sets["raw3"],
        "raw_xsz6": sets["raw_xsz6"],
        "mlp_lgb_raw2": sets["mlp_lgb_raw2"],
    }
    cal_methods = {
        "equal": fit_equal,
        "simplex": fit_simplex,
        "signed_ridge_a001": fit_signed_ridge(0.01),
        "signed_ridge_a01": fit_signed_ridge(0.1),
        "signed_ridge_a1": fit_signed_ridge(1.0),
    }
    for set_name, cols in cal_sets.items():
        for method, fit_fn in cal_methods.items():
            for minutes, alpha in [(60, 0.25), (90, 0.25), (120, 0.25), (120, 0.50)]:
                out.append(
                    EnsembleSpec(
                        f"{set_name}__{method}__time{minutes}_a{alpha:g}",
                        tuple(cols),
                        f"{method}_timecal",
                        fit_fn,
                        (minutes, alpha),
                    )
                )
    return out


def fit_time_multiplier(frame: pd.DataFrame, pred: np.ndarray, bucket_minutes: int, alpha: float) -> dict[str, object]:
    dt = pd.to_datetime(frame["datetime"])
    key = (dt.dt.hour.to_numpy(np.int32) * 60 + dt.dt.minute.to_numpy(np.int32)) // int(bucket_minutes)
    p = np.nan_to_num(pred.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(frame["label"].to_numpy(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    total_xy = float(np.dot(p, y))
    total_xx = float(np.dot(p, p))
    global_slope = total_xy / max(total_xx, 1e-30)
    if not np.isfinite(global_slope) or abs(global_slope) < 1e-30:
        return {"bucket_minutes": bucket_minutes, "alpha": alpha, "multipliers": {}}
    stat = pd.DataFrame({"key": key, "xy": p * y, "xx": p * p, "n": 1.0})
    agg = stat.groupby("key", sort=True).sum(numeric_only=True)
    slope = agg["xy"].to_numpy(np.float64) / np.maximum(agg["xx"].to_numpy(np.float64), 1e-30)
    raw = np.clip(slope / global_slope, 0.80, 1.20)
    n = agg["n"].to_numpy(np.float64)
    reliability = np.where(n >= 5000.0, n / (n + 20000.0), 0.0)
    mult = 1.0 + float(alpha) * reliability * (raw - 1.0)
    return {
        "bucket_minutes": int(bucket_minutes),
        "alpha": float(alpha),
        "multipliers": {str(int(k)): float(v) for k, v in zip(agg.index.tolist(), mult)},
    }


def apply_time_multiplier(frame: pd.DataFrame, pred: np.ndarray, fit: dict[str, object]) -> np.ndarray:
    if not fit.get("multipliers"):
        return pred
    dt = pd.to_datetime(frame["datetime"])
    key = (dt.dt.hour.to_numpy(np.int32) * 60 + dt.dt.minute.to_numpy(np.int32)) // int(fit["bucket_minutes"])
    mult_map = {int(k): float(v) for k, v in dict(fit["multipliers"]).items()}
    mult = pd.Series(key).map(mult_map).fillna(1.0).to_numpy(np.float64, copy=False)
    return (pred.astype(np.float64, copy=False) * mult).astype(np.float32)


def fit_apply_spec(spec: EnsembleSpec, train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_train = train.loc[:, list(spec.cols)].to_numpy(np.float64, copy=False)
    y_train = train["label"].to_numpy(np.float64, copy=False)
    x_target = target.loc[:, list(spec.cols)].to_numpy(np.float64, copy=False)
    w = spec.fit_fn(x_train, y_train)
    pred_train = x_train @ w
    pred = x_target @ w
    if compute_ic(pred_train, y_train) < 0:
        w = -w
        pred_train = -pred_train
        pred = -pred
    if spec.postcal is not None:
        minutes, alpha = spec.postcal
        cal = fit_time_multiplier(train, pred_train.astype(np.float32), minutes, alpha)
        pred = apply_time_multiplier(target, pred.astype(np.float32), cal)
    return pred.astype(np.float32), w.astype(np.float64)


def build_outer_frames() -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame, pd.DataFrame]:
    print("[load] loading LGB and Ridge sources", flush=True)
    d19_lgb = lgbseq.add_known_cols(pd.read_parquet(lgbseq.PRED_2019))
    d20_lgb = lgbseq.add_known_cols(pd.read_parquet(lgbseq.PRED_2020))
    ridge_fit, ridge_apply = load_ridge_sources()

    folds = {
        "q2": ("2019-01-01", "2019-04-01", "2019-04-01", "2019-07-01", month_strings("2019-01", "2019-03"), month_strings("2019-04", "2019-06")),
        "q3": ("2019-01-01", "2019-07-01", "2019-07-01", "2019-10-01", month_strings("2019-01", "2019-06"), month_strings("2019-07", "2019-09")),
        "q4": ("2019-01-01", "2019-10-01", "2019-10-01", "2020-01-01", month_strings("2019-01", "2019-09"), month_strings("2019-10", "2019-12")),
        "h2": ("2019-01-01", "2019-07-01", "2019-07-01", "2020-01-01", month_strings("2019-01", "2019-06"), month_strings("2019-07", "2019-12")),
    }
    fold_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, (tr_s, tr_e, val_s, val_e, tr_months, val_months) in folds.items():
        print(f"[fold] building {name} components", flush=True)
        mlp_fit = fit_mlp(tr_months)
        tr = merge_components(
            predict_mlp(tr_months, mlp_fit),
            lgb_frame(tr_s, tr_e, tr_s, tr_e, d19_lgb, d20_lgb),
            ridge_frame(tr_s, tr_e, tr_s, tr_e, ridge_fit, ridge_apply),
        )
        val = merge_components(
            predict_mlp(val_months, mlp_fit),
            lgb_frame(tr_s, tr_e, val_s, val_e, d19_lgb, d20_lgb),
            ridge_frame(tr_s, tr_e, val_s, val_e, ridge_fit, ridge_apply),
        )
        fold_frames[name] = (add_component_views(tr), add_component_views(val))
        print(
            f"[fold] {name} rows train={len(tr)} val={len(val)} "
            f"single_val_ic mlp={compute_ic(val['mlp'], val['label']):.6f} "
            f"lgb={compute_ic(val['lgb'], val['label']):.6f} ridge={compute_ic(val['ridge'], val['label']):.6f}",
            flush=True,
        )

    print("[final] building full2019 and 2020 components", flush=True)
    full_mlp_fit = fit_mlp(month_strings("2019-01", "2019-12"))
    full2019 = merge_components(
        predict_mlp(month_strings("2019-01", "2019-12"), full_mlp_fit),
        lgb_frame("2019-01-01", "2020-01-01", "2019-01-01", "2020-01-01", d19_lgb, d20_lgb),
        ridge_frame("2019-01-01", "2020-01-01", "2019-01-01", "2020-01-01", ridge_fit, ridge_apply),
    )
    test2020 = merge_components(
        predict_mlp(month_strings("2020-01", "2020-12"), full_mlp_fit),
        lgb_frame("2019-01-01", "2020-01-01", "2020-01-01", "2021-01-01", d19_lgb, d20_lgb),
        ridge_frame("2019-01-01", "2020-01-01", "2020-01-01", "2021-01-01", ridge_fit, ridge_apply),
    )
    return fold_frames, add_component_views(full2019), add_component_views(test2020)


def previous_benchmarks() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if BEST_ARTIFACT.exists():
        df = pd.read_parquet(
            BEST_ARTIFACT,
            columns=["symbol", "datetime", "label", "pred", "pred_xsz", "pred_xrank"],
            filters=[("datetime", ">=", pd.Timestamp("2020-01-01")), ("datetime", "<", pd.Timestamp("2021-01-01"))],
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        rows.append({"benchmark": "historical_best_ic0716_artifact_2020_pred", **summarize_pred(df, "pred")})
        rows.append({"benchmark": "historical_best_ic0716_artifact_2020_xsz", **summarize_pred(df, "pred_xsz")})
    if CORE_ARTIFACT.exists():
        df = pd.read_parquet(
            CORE_ARTIFACT,
            columns=["symbol", "datetime", "label", "pred", "pred_xsz", "pred_xrank"],
            filters=[("datetime", ">=", pd.Timestamp("2020-01-01")), ("datetime", "<", pd.Timestamp("2021-01-01"))],
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        rows.append({"benchmark": "core_moe_noDL_artifact_2020_pred", **summarize_pred(df, "pred")})
        rows.append({"benchmark": "core_moe_noDL_artifact_2020_xsz", **summarize_pred(df, "pred_xsz")})
    if EXPANDED_CLEAN_SUMMARY.exists():
        s = pd.read_csv(EXPANDED_CLEAN_SUMMARY).iloc[0]
        rows.append(
            {
                "benchmark": "expanded_history_gate_clean_stack_summary",
                "pooled_ic": float(s["pred_ic_2020"]),
                "monthly_mean": float(s["monthly_mean"]),
                "monthly_ir": float(s["monthly_ir"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_frames, full2019, test2020 = build_outer_frames()
    specs = build_specs()

    single_rows = []
    for col in ["mlp", "lgb", "ridge"]:
        single_rows.append({"model": col, **summarize_pred(test2020, col)})
    pd.DataFrame(single_rows).to_csv(OUT_DIR / "single_component_2020_check.csv", index=False)
    print("[check] final single components")
    print(pd.DataFrame(single_rows).to_string(index=False), flush=True)

    cv_rows = []
    for i, spec in enumerate(specs, 1):
        fold_ics = []
        fold_weights: dict[str, list[float]] = {}
        for fold, (train, val) in fold_frames.items():
            pred, w = fit_apply_spec(spec, train, val)
            ic = compute_ic(pred, val["label"].to_numpy())
            fold_ics.append(float(ic))
            fold_weights[fold] = [float(x) for x in w]
        q = np.asarray(fold_ics[:3], dtype=np.float64)
        allv = np.asarray(fold_ics, dtype=np.float64)
        cv_rows.append(
            {
                "candidate": spec.name,
                "method": spec.method,
                "cols_json": json.dumps(list(spec.cols)),
                "q2_ic": fold_ics[0],
                "q3_ic": fold_ics[1],
                "q4_ic": fold_ics[2],
                "h2_ic": fold_ics[3],
                "cv_mean_q2q4": float(np.nanmean(q)),
                "cv_std_q2q4": float(np.nanstd(q, ddof=1)),
                "cv_min_q2q4": float(np.nanmin(q)),
                "score_mean_m025std": float(np.nanmean(q) - 0.25 * np.nanstd(q, ddof=1)),
                "score_mean_m050std": float(np.nanmean(q) - 0.50 * np.nanstd(q, ddof=1)),
                "score_q3_h2": float(fold_ics[1] + fold_ics[3]),
                "score_q3_q4_h2": float(fold_ics[1] + fold_ics[2] + fold_ics[3]),
                "score_min_mean": float(np.nanmin(q) + 0.25 * np.nanmean(q)),
                "weights_by_fold_json": json.dumps(fold_weights, sort_keys=True),
            }
        )
        if i % 10 == 0 or i == len(specs):
            print(f"[cv] {i}/{len(specs)} last={spec.name}", flush=True)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUT_DIR / "ensemble_2019_cv_screen.csv", index=False)

    selector_cols = [
        "score_mean_m025std",
        "score_mean_m050std",
        "score_q3_h2",
        "score_q3_q4_h2",
        "score_min_mean",
        "h2_ic",
        "q4_ic",
    ]
    audit_indices: set[int] = set()
    selector_winners = []
    for sel in selector_cols:
        ranked = cv.sort_values(sel, ascending=False).reset_index()
        top = ranked.iloc[0]
        selector_winners.append({"selector": sel, "candidate": str(top["candidate"])})
        audit_indices.update(int(x) for x in ranked.head(8)["index"].tolist())
    # Also audit equal/raw family baselines even if not selected.
    for name in ["raw3__equal", "xsz3__equal", "rankgauss3__equal", "mlp_lgb_raw2__equal", "mlp_lgb_xsz2__equal"]:
        hit = cv.index[cv["candidate"].eq(name)].tolist()
        audit_indices.update(hit)

    audit_rows = []
    monthly_rows = []
    final_weight_rows = []
    spec_by_name = {s.name: s for s in specs}
    for idx in sorted(audit_indices):
        row = cv.loc[idx]
        spec = spec_by_name[str(row["candidate"])]
        pred20, w = fit_apply_spec(spec, full2019, test2020)
        col = "pred_ens"
        eval_df = test2020[["symbol", "datetime", "label"]].copy()
        eval_df[col] = pred20
        winners = [x["selector"] for x in selector_winners if x["candidate"] == spec.name]
        summary = {
            **row.to_dict(),
            **summarize_pred(eval_df, col, prefix="audit2020"),
            "selector_winner": bool(winners),
            "winner_selectors": "|".join(winners),
        }
        audit_rows.append(summary)
        for c, ww in zip(spec.cols, w):
            final_weight_rows.append({"candidate": spec.name, "feature": c, "weight": float(ww)})
        mic = monthly_ic(eval_df, col)
        for month, ic in mic.items():
            monthly_rows.append({"candidate": spec.name, "month": month, "ic": float(ic)})
        print(
            f"[audit] {spec.name} winner={bool(winners)} 2020={summary['audit2020_pooled_ic']:.6f} "
            f"monthly={summary['audit2020_monthly_mean']:.6f}",
            flush=True,
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_DIR / "ensemble_2020_audit.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(OUT_DIR / "ensemble_2020_monthly_ic.csv", index=False)
    pd.DataFrame(final_weight_rows).to_csv(OUT_DIR / "ensemble_final_weights.csv", index=False)

    winners = audit[audit["selector_winner"]].copy().sort_values("audit2020_pooled_ic", ascending=False)
    winners.to_csv(OUT_DIR / "ensemble_selector_winners_2020_audit.csv", index=False)

    best_strict = winners.iloc[0].to_dict() if not winners.empty else {}
    best_diag = audit.sort_values("audit2020_pooled_ic", ascending=False).iloc[0].to_dict()
    bench = previous_benchmarks()
    bench.to_csv(OUT_DIR / "previous_ensemble_benchmarks.csv", index=False)

    report = {
        "selected_strict_by_2019_selectors": best_strict,
        "best_2020_diagnostic": best_diag,
        "selectors": selector_winners,
        "previous_benchmarks": bench.to_dict(orient="records"),
        "no_future_leakage_note": (
            "Selector winners are chosen only from 2019 component-level outer folds. "
            "The 2020 audit is computed after candidate selectors are fixed. "
            "Rows marked diagnostic are not strict selection results."
        ),
    }
    (OUT_DIR / "ensemble_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\nSelector winners")
    show = [
        "candidate",
        "winner_selectors",
        "q2_ic",
        "q3_ic",
        "q4_ic",
        "h2_ic",
        "audit2020_pooled_ic",
        "audit2020_monthly_mean",
    ]
    print(winners[show].to_string(index=False), flush=True)
    print("\nTop audited diagnostics")
    print(audit.sort_values("audit2020_pooled_ic", ascending=False)[show + ["selector_winner"]].head(20).to_string(index=False), flush=True)
    print("\nPrevious benchmarks")
    print(bench.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
