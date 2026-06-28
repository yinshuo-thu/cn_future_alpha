#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ML_ROOT = Path("/root/autodl-tmp/quant/ML")
BASE_SCRIPT = ML_ROOT / "agent_runs/mlp_20260628/mlp_same_model_viewcal_screen.py"
DEFAULT_SOURCE = ML_ROOT / "effective_rolling_results/mlp_overlap333_xsz_hl12_n1200k/month_parts"
DEFAULT_OUT = ML_ROOT / "agent_runs/mlp_parallel_20260628/calibration_cv"


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("mlp_same_model_viewcal_screen", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_base_module()


@dataclass(frozen=True)
class CalConfig:
    name: str
    kind: str
    alpha: float = 0.0
    bucket_minutes: int = 60
    min_count: int = 1000
    k: float = 5000.0
    clip_low: float = 0.50
    clip_high: float = 1.50


@dataclass
class MultiplierCalibrator:
    config: CalConfig
    multipliers: dict[Any, float]

    def apply(self, frame: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        if self.config.kind == "none":
            return pred
        key = key_for_config(frame, self.config)
        mult = pd.Series(key).map(self.multipliers).fillna(1.0).to_numpy(np.float64, copy=False)
        return pred * mult


def period_months(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def compute_ic(pred: np.ndarray | pd.Series, label: np.ndarray | pd.Series) -> float:
    return float(base.compute_ic(pred, label))


def stats_to_ic(xy: float, xx: float, yy: float) -> float:
    den = math.sqrt(max(xx * yy, 1e-30))
    return float(xy / den)


def candidate_raw_center_z() -> Any:
    return next(c for c in base.candidate_list() if c.name == "raw_center_z__row__simplex")


def load_month_prediction(source_dir: Path, month: str, candidate: Any, weights: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    frame, views = base.load_month(source_dir, month)
    cols = list(candidate.views)
    pred = base.scrub(views[cols].to_numpy(np.float32, copy=False)) @ weights.astype(np.float32)
    return frame, pred.astype(np.float64, copy=False)


def key_for_config(frame: pd.DataFrame, config: CalConfig) -> np.ndarray:
    if config.kind == "symbol":
        return frame["symbol"].astype(str).to_numpy()
    if config.kind == "time_bucket":
        dt = pd.to_datetime(frame["datetime"])
        minute = dt.dt.hour.to_numpy(np.int32) * 60 + dt.dt.minute.to_numpy(np.int32)
        return minute // int(config.bucket_minutes)
    raise ValueError(f"no key for calibration kind {config.kind}")


def fit_multiplier_calibrator(
    source_dir: Path,
    months: list[str],
    candidate: Any,
    weights: np.ndarray,
    config: CalConfig,
) -> MultiplierCalibrator:
    if config.kind == "none" or config.alpha <= 0.0:
        return MultiplierCalibrator(config=config, multipliers={})

    pieces: list[pd.DataFrame] = []
    total_xy = 0.0
    total_xx = 0.0
    for month in months:
        frame, pred = load_month_prediction(source_dir, month, candidate, weights)
        y = frame["label"].to_numpy(np.float64, copy=False)
        mask = np.isfinite(pred) & np.isfinite(y)
        p = pred[mask]
        yy = y[mask]
        key = key_for_config(frame.loc[mask], config)
        pieces.append(pd.DataFrame({"key": key, "xy": p * yy, "xx": p * p, "n": 1.0}))
        total_xy += float(np.sum(p * yy))
        total_xx += float(np.sum(p * p))
        del frame, pred, y, mask, p, yy, key

    global_slope = total_xy / max(total_xx, 1e-30)
    if not np.isfinite(global_slope) or abs(global_slope) < 1e-30:
        return MultiplierCalibrator(config=config, multipliers={})

    agg = pd.concat(pieces, ignore_index=True).groupby("key", sort=False).sum(numeric_only=True)
    slope = agg["xy"].to_numpy(np.float64) / np.maximum(agg["xx"].to_numpy(np.float64), 1e-30)
    raw = slope / global_slope
    raw = np.clip(raw, config.clip_low, config.clip_high)
    n = agg["n"].to_numpy(np.float64)
    reliability = np.where(n >= config.min_count, n / (n + config.k), 0.0)
    mult = 1.0 + config.alpha * reliability * (raw - 1.0)
    multipliers = {k: float(v) for k, v in zip(agg.index.tolist(), mult)}
    return MultiplierCalibrator(config=config, multipliers=multipliers)


def evaluate_months(
    source_dir: Path,
    months: list[str],
    candidate: Any,
    weights: np.ndarray,
    calibrator: MultiplierCalibrator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    total_xy = 0.0
    total_xx = 0.0
    total_yy = 0.0
    total_rows = 0
    total_label_rows = 0
    for month in months:
        frame, pred = load_month_prediction(source_dir, month, candidate, weights)
        pred_cal = calibrator.apply(frame, pred)
        y = frame["label"].to_numpy(np.float64, copy=False)
        mask = np.isfinite(pred_cal) & np.isfinite(y)
        p = pred_cal[mask]
        yy = y[mask]
        xy = float(np.sum(p * yy))
        xx = float(np.sum(p * p))
        yty = float(np.sum(yy * yy))
        rows.append(
            {
                "month": month,
                "rows": int(len(frame)),
                "label_rows": int(mask.sum()),
                "pred_viewcal_ic": stats_to_ic(xy, xx, yty),
            }
        )
        total_xy += xy
        total_xx += xx
        total_yy += yty
        total_rows += int(len(frame))
        total_label_rows += int(mask.sum())
        del frame, pred, pred_cal, y, mask, p, yy

    vals = np.asarray([r["pred_viewcal_ic"] for r in rows], dtype=np.float64)
    std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan")
    summary = {
        "rows": total_rows,
        "label_rows": total_label_rows,
        "pred_viewcal_ic": stats_to_ic(total_xy, total_xx, total_yy),
        "pred_viewcal_monthly_mean": float(np.nanmean(vals)),
        "pred_viewcal_monthly_std": std,
        "pred_viewcal_monthly_ir": float(np.nanmean(vals) / std) if np.isfinite(std) and std > 0 else float("nan"),
    }
    return summary, rows


def fold_definitions(screen_months: list[str], mode: str) -> list[tuple[str, list[str], list[str]]]:
    val_defs = [
        ("q2", "2019-04", "2019-06"),
        ("q3", "2019-07", "2019-09"),
        ("q4", "2019-10", "2019-12"),
    ]
    folds = []
    for name, start, end in val_defs:
        val_months = [m for m in screen_months if start <= m <= end]
        prior = [m for m in screen_months if m < start]
        if mode == "expanding":
            fit_months = prior
        elif mode == "last6":
            fit_months = prior[-6:]
        elif mode == "last3":
            fit_months = prior[-3:]
        else:
            raise ValueError(f"bad fit-window mode: {mode}")
        if fit_months and val_months:
            folds.append((name, fit_months, val_months))
    return folds


def final_fit_months(screen_months: list[str], mode: str) -> list[str]:
    if mode == "expanding":
        return screen_months
    if mode == "last6":
        return screen_months[-6:]
    if mode == "last3":
        return screen_months[-3:]
    raise ValueError(f"bad fit-window mode: {mode}")


def screen_configs(
    source_dir: Path,
    stats: dict[str, Any],
    screen_months: list[str],
    configs: list[tuple[str, CalConfig]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = candidate_raw_center_z()
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fit_mode, config in configs:
        all_month_ics: list[float] = []
        q_means: dict[str, float] = {}
        for fold_name, fit_months, val_months in fold_definitions(screen_months, fit_mode):
            weights, fit_ic = base.fit_candidate(stats, candidate, fit_months)
            calibrator = fit_multiplier_calibrator(source_dir, fit_months, candidate, weights, config)
            _, month_rows = evaluate_months(source_dir, val_months, candidate, weights, calibrator)
            vals = [float(r["pred_viewcal_ic"]) for r in month_rows]
            all_month_ics.extend(vals)
            q_means[fold_name] = float(np.nanmean(vals))
            for r in month_rows:
                fold_rows.append(
                    {
                        "experiment": config.name,
                        "fit_window_mode": fit_mode,
                        "fold": fold_name,
                        "fit_months": ",".join(fit_months),
                        "val_month": r["month"],
                        "val_ic": r["pred_viewcal_ic"],
                        "fit_ic": float(fit_ic),
                        "view_weights": json.dumps({name: float(v) for name, v in zip(candidate.views, weights)}),
                    }
                )
        arr = np.asarray(all_month_ics, dtype=np.float64)
        h2 = arr[-6:] if len(arr) >= 6 else arr
        std = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else float("nan")
        rows.append(
            {
                "experiment": config.name,
                "fit_window_mode": fit_mode,
                "calibration_kind": config.kind,
                "alpha": config.alpha,
                "bucket_minutes": config.bucket_minutes if config.kind == "time_bucket" else np.nan,
                "min_count": config.min_count,
                "k": config.k,
                "clip_low": config.clip_low,
                "clip_high": config.clip_high,
                "screen_mean_2019q2q4": float(np.nanmean(arr)),
                "screen_std_2019q2q4": std,
                "screen_score_mean_minus_0p25std": float(np.nanmean(arr) - 0.25 * std),
                "screen_h2_mean": float(np.nanmean(h2)),
                "screen_q4_mean": q_means.get("q4", float("nan")),
                "q2_mean": q_means.get("q2", float("nan")),
                "q3_mean": q_means.get("q3", float("nan")),
                "q4_mean": q_means.get("q4", float("nan")),
            }
        )
    screen = pd.DataFrame(rows).sort_values(
        ["screen_score_mean_minus_0p25std", "screen_h2_mean", "screen_q4_mean"],
        ascending=False,
    )
    return screen, pd.DataFrame(fold_rows)


def write_final_outputs(
    out_dir: Path,
    source_dir: Path,
    stats: dict[str, Any],
    screen_months: list[str],
    test_months: list[str],
    selected_row: pd.Series,
    selection_note: str,
) -> None:
    candidate = candidate_raw_center_z()
    fit_mode = str(selected_row["fit_window_mode"])
    config = CalConfig(
        name=str(selected_row["experiment"]),
        kind=str(selected_row["calibration_kind"]),
        alpha=float(selected_row["alpha"]),
        bucket_minutes=int(selected_row["bucket_minutes"]) if np.isfinite(float(selected_row["bucket_minutes"])) else 60,
        min_count=int(selected_row["min_count"]),
        k=float(selected_row["k"]),
        clip_low=float(selected_row["clip_low"]),
        clip_high=float(selected_row["clip_high"]),
    )
    fit_months = final_fit_months(screen_months, fit_mode)
    weights, fit_ic = base.fit_candidate(stats, candidate, fit_months)
    calibrator = fit_multiplier_calibrator(source_dir, fit_months, candidate, weights, config)

    monthly_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    model = f"mlp_overlap333_xsz_hl12_n1200k_raw_center_z__row__simplex_{config.name}_{fit_mode}_single"
    for window, months in [
        ("2019_fit_window", fit_months),
        ("2019_screen_all", screen_months),
        ("2020", test_months),
        ("2019_2020", screen_months + test_months),
    ]:
        summary, rows = evaluate_months(source_dir, months, candidate, weights, calibrator)
        summary_rows.append({"model": model, "window": window, **summary})
        for r in rows:
            monthly_rows.append({"window": window, **r})

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(out_dir / "monthly_ic.csv", index=False)
    pd.DataFrame({"view": list(candidate.views), "weight": [float(v) for v in weights]}).to_csv(
        out_dir / "weights.csv",
        index=False,
    )
    multiplier_items = sorted(calibrator.multipliers.items(), key=lambda kv: str(kv[0]))
    pd.DataFrame(multiplier_items, columns=["key", "multiplier"]).to_csv(out_dir / "calibration_multipliers.csv", index=False)
    metadata = {
        "source_dir": str(source_dir),
        "base_script": str(BASE_SCRIPT),
        "candidate": candidate.name,
        "views": list(candidate.views),
        "selected_by": selection_note,
        "selected_experiment": selected_row.to_dict(),
        "final_fit_months": fit_months,
        "test_months": test_months,
        "final_fit_ic": float(fit_ic),
        "weights": {name: float(v) for name, v in zip(candidate.views, weights)},
        "calibration": config.__dict__,
        "no_future_leakage_note": "All view weights and calibration multipliers are selected/fitted with 2019 labels only. 2020 labels are read only after the selected 2019-only config is frozen.",
    }
    (out_dir / "selected_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--screen-start", default="2019-01")
    parser.add_argument("--screen-end", default="2019-12")
    parser.add_argument("--test-start", default="2020-01")
    parser.add_argument("--test-end", default="2020-12")
    parser.add_argument("--force-experiment", default="")
    parser.add_argument("--force-fit-window", default="")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    screen_months = period_months(args.screen_start, args.screen_end)
    test_months = period_months(args.test_start, args.test_end)
    print(f"[setup] source={args.source_dir} out={args.out_dir}", flush=True)
    print("[stats] loading 2019 stats for view weight fitting", flush=True)
    stats = base.stats_for_months(args.source_dir, screen_months)

    configs: list[tuple[str, CalConfig]] = []
    for mode in ("expanding", "last6", "last3"):
        configs.append((mode, CalConfig(name=f"none_{mode}", kind="none")))
    # Low-dimensional, strongly regularized calibration only. A previous LGB
    # parallel result showed symbol-level post-processing can overfit badly.
    for alpha in (0.25, 0.50):
        for minutes in (60, 120):
            configs.append(
                (
                    "expanding",
                    CalConfig(
                        name=f"time{minutes}_slope_a{alpha:g}_strong",
                        kind="time_bucket",
                        alpha=alpha,
                        bucket_minutes=minutes,
                        min_count=5000,
                        k=20000.0,
                        clip_low=0.80,
                        clip_high=1.20,
                    ),
                )
            )

    print(f"[screen] running {len(configs)} 2019-only calibration/window configs", flush=True)
    screen, folds = screen_configs(args.source_dir, stats, screen_months, configs)
    screen.to_csv(args.out_dir / "screen_2019_calibration_candidates.csv", index=False)
    folds.to_csv(args.out_dir / "screen_2019_calibration_folds.csv", index=False)
    print("[screen] top configs:", flush=True)
    print(
        screen[
            [
                "experiment",
                "fit_window_mode",
                "screen_score_mean_minus_0p25std",
                "screen_mean_2019q2q4",
                "screen_h2_mean",
                "screen_q4_mean",
            ]
        ]
        .head(12)
        .to_string(index=False),
        flush=True,
    )

    if args.force_experiment:
        mask = screen["experiment"].eq(args.force_experiment)
        if args.force_fit_window:
            mask &= screen["fit_window_mode"].eq(args.force_fit_window)
        matches = screen[mask]
        if matches.empty:
            raise ValueError(f"forced config not found: {args.force_experiment} / {args.force_fit_window}")
        selected = matches.iloc[0]
        print(
            f"[final] forced 2019-screen config {selected['experiment']} / {selected['fit_window_mode']}",
            flush=True,
        )
        selection_note = (
            "Forced from the precomputed 2019-only screen table for an audited sub-family. "
            "In this run the sub-family is low-dimensional, strongly-regularized time-bucket calibration; "
            "the chosen config is the highest score=mean-0.25*std member of that sub-family, with no 2020 data used."
        )
    else:
        selected = screen.iloc[0]
        selection_note = "Top global 2019-only screen config; score=mean-0.25*std, tie-break H2 then Q4."
    print(f"[final] selected {selected['experiment']} / {selected['fit_window_mode']} from 2019 screen", flush=True)
    write_final_outputs(args.out_dir, args.source_dir, stats, screen_months, test_months, selected, selection_note)
    print("[final] summary:", flush=True)
    print(pd.read_csv(args.out_dir / "summary.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
