#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ML_DIR = Path("/root/autodl-tmp/quant/ML")
LGB_PARALLEL = ML_DIR / "agent_runs" / "lgb_parallel_20260628"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(LGB_PARALLEL) not in sys.path:
    sys.path.insert(0, str(LGB_PARALLEL))

from rolling_factor_model_eval import compute_ic, period_ic  # noqa: E402
import shape_refine_2019_search as shape  # noqa: E402


RUN_DIR = ML_DIR / "agent_runs" / "lgb_reg8_shape_seqcal_20260628"
PRED_2019 = ML_DIR / "agent_runs" / "lgb_reg8_viewcal_20260628" / "opt_lgb_worker_t500_xsz_random_lb18_reg8_seed140_2019full.parquet"
PRED_2020 = ML_DIR / "strict_opt_results" / "opt_lgb_worker_t500_xsz_random_lb18_reg8_seed140_audit2020.parquet"
SHAPE_SELECTED = LGB_PARALLEL / "shape_refine_selected_2019_only.csv"


@dataclass(frozen=True)
class StageSpec:
    kind: str
    bucket_minutes: int = 90
    n_bins: int = 5
    alpha: float = 0.5
    k: float = 20_000.0
    min_count: int = 5_000
    clip_low: float = 0.8
    clip_high: float = 1.2


@dataclass(frozen=True)
class Candidate:
    name: str
    stages: tuple[StageSpec, ...]


def add_known_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    minute = out["datetime"].dt.hour.astype(np.int16) * 60 + out["datetime"].dt.minute.astype(np.int16)
    out["_minute"] = minute.astype(np.int16)
    return out


def subset(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df["datetime"] >= pd.Timestamp(start)) & (df["datetime"] < pd.Timestamp(end))]


def fit_shape_transform(train: pd.DataFrame, fit_json: str) -> dict[str, Any]:
    selected_fit = json.loads(fit_json)
    spec = {key: selected_fit[key] for key in ["kind", "qlo", "qhi", "gamma", "scale_q"] if key in selected_fit}
    cand = shape.Candidate("selected_shape", spec, len(spec))
    return shape.fit_candidate(train, cand)


def transform_shape(df: pd.DataFrame, fit: dict[str, Any]) -> np.ndarray:
    return shape.transform(df, fit).astype(np.float32, copy=False)


def make_time_key(df: pd.DataFrame, stage: StageSpec, pred: np.ndarray, fit_state: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    del pred
    return (df["_minute"].to_numpy(np.int32) // int(stage.bucket_minutes)).astype(np.int16), {}


def make_abs_key(df: pd.DataFrame, stage: StageSpec, pred: np.ndarray, fit_state: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    del df
    abs_pred = np.abs(np.nan_to_num(pred.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0))
    if fit_state is None:
        qs = np.linspace(0.0, 1.0, stage.n_bins + 1)[1:-1]
        edges = np.quantile(abs_pred, qs).astype(np.float64)
        edges = np.unique(edges)
        fit_state = {"edges": edges.tolist()}
    edges = np.asarray(fit_state["edges"], dtype=np.float64)
    key = np.searchsorted(edges, abs_pred, side="right").astype(np.int16)
    return key, fit_state


def make_signed_abs_key(df: pd.DataFrame, stage: StageSpec, pred: np.ndarray, fit_state: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    abs_key, fit_state = make_abs_key(df, stage, pred, fit_state)
    sign_key = (pred >= 0).astype(np.int16)
    return (abs_key * 2 + sign_key).astype(np.int16), fit_state


def make_xrank_strength_key(df: pd.DataFrame, stage: StageSpec, pred: np.ndarray, fit_state: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    del pred, fit_state
    rank = np.nan_to_num(df["pred_xrank"].to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    abs_rank = np.abs(rank)
    key = np.minimum((abs_rank * stage.n_bins).astype(np.int16), stage.n_bins - 1)
    return key.astype(np.int16), {}


def make_key(df: pd.DataFrame, stage: StageSpec, pred: np.ndarray, fit_state: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if stage.kind == "time":
        return make_time_key(df, stage, pred, fit_state)
    if stage.kind == "abs":
        return make_abs_key(df, stage, pred, fit_state)
    if stage.kind == "signed_abs":
        return make_signed_abs_key(df, stage, pred, fit_state)
    if stage.kind == "xrank_strength":
        return make_xrank_strength_key(df, stage, pred, fit_state)
    raise ValueError(f"bad stage kind {stage.kind!r}")


def fit_multiplier(df: pd.DataFrame, pred: np.ndarray, stage: StageSpec) -> dict[str, Any]:
    p = np.nan_to_num(pred.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(df["label"].to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    key, fit_state = make_key(df, stage, p, None)
    total_xy = float(np.dot(p, y))
    total_xx = float(np.dot(p, p))
    global_slope = total_xy / max(total_xx, 1e-30)
    if not np.isfinite(global_slope) or abs(global_slope) < 1e-30:
        return {"stage": asdict(stage), "state": fit_state, "multipliers": {}}
    stat = pd.DataFrame({"key": key, "xy": p * y, "xx": p * p, "n": 1.0})
    agg = stat.groupby("key", sort=True).sum(numeric_only=True)
    slope = agg["xy"].to_numpy(np.float64) / np.maximum(agg["xx"].to_numpy(np.float64), 1e-30)
    raw = np.clip(slope / global_slope, stage.clip_low, stage.clip_high)
    n = agg["n"].to_numpy(np.float64)
    reliability = np.where(n >= stage.min_count, n / (n + stage.k), 0.0)
    mult = 1.0 + stage.alpha * reliability * (raw - 1.0)
    return {
        "stage": asdict(stage),
        "state": fit_state,
        "multipliers": {str(int(k)): float(v) for k, v in zip(agg.index.tolist(), mult)},
    }


def apply_multiplier(df: pd.DataFrame, pred: np.ndarray, stage_fit: dict[str, Any]) -> np.ndarray:
    stage = StageSpec(**stage_fit["stage"])
    key, _ = make_key(df, stage, pred, stage_fit.get("state") or {})
    mult_map = {int(k): float(v) for k, v in stage_fit["multipliers"].items()}
    mult = pd.Series(key).map(mult_map).fillna(1.0).to_numpy(np.float64, copy=False)
    return (pred.astype(np.float64, copy=False) * mult).astype(np.float32)


def fit_apply_stages(train: pd.DataFrame, pred_train: np.ndarray, test: pd.DataFrame, pred_test: np.ndarray, stages: tuple[StageSpec, ...]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    p_train = pred_train.astype(np.float32, copy=True)
    p_test = pred_test.astype(np.float32, copy=True)
    fits: list[dict[str, Any]] = []
    for stage in stages:
        fit = fit_multiplier(train, p_train, stage)
        p_train = apply_multiplier(train, p_train, fit)
        p_test = apply_multiplier(test, p_test, fit)
        fits.append(fit)
    return p_test, fits


def pooled(df: pd.DataFrame, pred: np.ndarray) -> float:
    return float(compute_ic(pred, df["label"].to_numpy()))


def monthly(df: pd.DataFrame, pred: np.ndarray) -> tuple[float, float, float]:
    tmp = df[["datetime", "label"]].copy()
    tmp["pred_cal"] = pred
    by_m = pd.to_numeric(period_ic(tmp, "pred_cal", "M"), errors="coerce")
    return float(by_m.mean()), float(by_m.min()), float(by_m.std(ddof=1))


def screen_one(d19: pd.DataFrame, fit_json: str, cand: Candidate) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = [
        ("q2", "2019-01-01", "2019-04-01", "2019-04-01", "2019-07-01"),
        ("q3", "2019-01-01", "2019-07-01", "2019-07-01", "2019-10-01"),
        ("q4", "2019-01-01", "2019-10-01", "2019-10-01", "2020-01-01"),
        ("h2", "2019-01-01", "2019-07-01", "2019-07-01", "2020-01-01"),
    ]
    fold_rows = []
    scores = []
    for fold, tr_s, tr_e, val_s, val_e in folds:
        tr = subset(d19, tr_s, tr_e)
        val = subset(d19, val_s, val_e)
        shape_fit = fit_shape_transform(tr, fit_json)
        pred_tr = transform_shape(tr, shape_fit)
        pred_val_shape = transform_shape(val, shape_fit)
        pred_val, stage_fits = fit_apply_stages(tr, pred_tr, val, pred_val_shape, cand.stages)
        ic = pooled(val, pred_val)
        scores.append(ic)
        fold_rows.append(
            {
                "candidate": cand.name,
                "fold": fold,
                "val_ic": ic,
                "shape_fit": json.dumps(shape_fit, sort_keys=True),
                "stage_fits": json.dumps(stage_fits, sort_keys=True),
            }
        )
    q_scores = np.asarray(scores[:3], dtype=np.float64)
    all_scores = np.asarray(scores, dtype=np.float64)
    row = {
        "candidate": cand.name,
        "stages_json": json.dumps([asdict(s) for s in cand.stages], sort_keys=True),
        "params": len(cand.stages),
        "screen_mean_q2q4": float(np.nanmean(q_scores)),
        "screen_std_q2q4": float(np.nanstd(q_scores, ddof=1)),
        "screen_score_mean_minus_0p25std": float(np.nanmean(q_scores) - 0.25 * np.nanstd(q_scores, ddof=1)),
        "screen_score_robust": float(np.nanmean(q_scores) - 0.50 * np.nanstd(q_scores, ddof=1) + 0.25 * np.nanmin(q_scores)),
        "screen_mean_allfolds": float(np.nanmean(all_scores)),
        "screen_h2_ic": float(scores[3]),
    }
    return row, fold_rows


def audit_one(d19: pd.DataFrame, d20: pd.DataFrame, fit_json: str, cand: Candidate) -> tuple[dict[str, Any], np.ndarray]:
    shape_fit = fit_shape_transform(d19, fit_json)
    pred19_shape = transform_shape(d19, shape_fit)
    pred20_shape = transform_shape(d20, shape_fit)
    pred20, stage_fits = fit_apply_stages(d19, pred19_shape, d20, pred20_shape, cand.stages)
    pred19, _ = fit_apply_stages(d19, pred19_shape, d19, pred19_shape, cand.stages)
    m19_mean, m19_min, m19_std = monthly(d19, pred19)
    m20_mean, m20_min, m20_std = monthly(d20, pred20)
    return {
        "shape_fit": json.dumps(shape_fit, sort_keys=True),
        "stage_fits": json.dumps(stage_fits, sort_keys=True),
        "fit2019_ic": pooled(d19, pred19),
        "fit2019_monthly_mean": m19_mean,
        "fit2019_monthly_min": m19_min,
        "fit2019_monthly_std": m19_std,
        "audit2020_ic": pooled(d20, pred20),
        "audit2020_monthly_mean": m20_mean,
        "audit2020_monthly_min": m20_min,
        "audit2020_monthly_std": m20_std,
    }, pred20


def candidate_grid() -> list[Candidate]:
    out: list[Candidate] = []
    for bm in [60, 90, 120, 180]:
        for alpha in [0.25, 0.5, 0.75, 1.0]:
            for k in [5_000.0, 20_000.0, 80_000.0]:
                out.append(Candidate(f"time{bm}_a{alpha:g}_k{int(k)}", (StageSpec("time", bucket_minutes=bm, alpha=alpha, k=k),)))
    for kind in ["abs", "signed_abs", "xrank_strength"]:
        for bins in [4, 5, 8]:
            for alpha in [0.25, 0.5, 0.75]:
                for k in [20_000.0, 80_000.0, 200_000.0]:
                    out.append(Candidate(f"{kind}{bins}_a{alpha:g}_k{int(k)}", (StageSpec(kind, n_bins=bins, alpha=alpha, k=k),)))
    for bm in [90, 120, 180]:
        for ta in [0.5, 0.75, 1.0]:
            for kind in ["abs", "signed_abs"]:
                for bins in [4, 5, 8]:
                    for ba in [0.25, 0.5, 0.75]:
                        out.append(
                            Candidate(
                                f"time{bm}_a{ta:g}_then_{kind}{bins}_a{ba:g}",
                                (
                                    StageSpec("time", bucket_minutes=bm, alpha=ta, k=5_000.0),
                                    StageSpec(kind, n_bins=bins, alpha=ba, k=80_000.0),
                                ),
                            )
                        )
    return out


def candidate_from_row(row: pd.Series) -> Candidate:
    return Candidate(str(row["candidate"]), tuple(StageSpec(**x) for x in json.loads(row["stages_json"])))


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SHAPE_SELECTED).iloc[0]
    fit_json = str(selected["fit_json"])
    d19 = add_known_cols(pd.read_parquet(PRED_2019))
    d20 = add_known_cols(pd.read_parquet(PRED_2020))

    rows = []
    fold_rows = []
    for cand in candidate_grid():
        row, folds = screen_one(d19, fit_json, cand)
        rows.append(row)
        fold_rows.extend(folds)
        print(
            f"[seq-screen] {cand.name} score={row['screen_score_mean_minus_0p25std']:.6f} "
            f"robust={row['screen_score_robust']:.6f} h2={row['screen_h2_ic']:.6f}",
            flush=True,
        )

    screen = pd.DataFrame(rows)
    screen.to_csv(RUN_DIR / "screen_2019_candidates.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RUN_DIR / "screen_2019_folds.csv", index=False)

    audit_indices = set(screen.sort_values("screen_score_mean_minus_0p25std", ascending=False).head(20).index)
    audit_indices |= set(screen.sort_values("screen_score_robust", ascending=False).head(20).index)
    audit_indices |= set(screen.sort_values("screen_h2_ic", ascending=False).head(10).index)
    audit_rows = []
    selected_pred: np.ndarray | None = None
    primary_idx = screen.sort_values("screen_score_mean_minus_0p25std", ascending=False).index[0]
    for idx in sorted(audit_indices):
        cand = candidate_from_row(screen.loc[idx])
        audit, pred20 = audit_one(d19, d20, fit_json, cand)
        combined = {**screen.loc[idx].to_dict(), **audit}
        audit_rows.append(combined)
        if idx == primary_idx:
            selected_pred = pred20
        print(f"[seq-audit] {cand.name} 2020={audit['audit2020_ic']:.6f}", flush=True)

    audited = pd.DataFrame(audit_rows).sort_values("screen_score_mean_minus_0p25std", ascending=False)
    audited.to_csv(RUN_DIR / "audited_selected_by_2019.csv", index=False)
    audited.sort_values("audit2020_ic", ascending=False).to_csv(RUN_DIR / "top2020_diagnostics.csv", index=False)
    selected_row = audited.iloc[0].to_dict()
    selected_config = {
        "shape_source": str(SHAPE_SELECTED),
        "shape_candidate": str(selected["candidate"]),
        "shape_fit_json_from_2019_selection": fit_json,
        "candidate": selected_row["candidate"],
        "stages": json.loads(selected_row["stages_json"]),
        "selected_by": "highest 2019 screen_score_mean_minus_0p25std inside shape-primary sequential multiplier family",
        "no_future_leakage_note": "shape candidate was selected on 2019; sequential multiplier config selected on 2019 folds; final shape thresholds and multipliers fit on full 2019; 2020 labels used only for audit.",
    }
    (RUN_DIR / "selected_config.json").write_text(json.dumps(selected_config, indent=2, sort_keys=True), encoding="utf-8")
    if selected_pred is not None:
        pred_out = d20[["symbol", "datetime", "label", "pred", "pred_xsz", "pred_xrank"]].copy()
        pred_out["pred_lgb_shape_seqcal"] = selected_pred.astype(np.float32)
        pred_out.to_parquet(RUN_DIR / "selected_by_2019_audit2020_predictions.parquet", index=False)
    show = ["candidate", "screen_score_mean_minus_0p25std", "screen_score_robust", "screen_h2_ic", "audit2020_ic", "audit2020_monthly_mean", "audit2020_monthly_min"]
    print("\nSelected by 2019 score")
    print(audited[show].head(20).to_string(index=False))
    print("\nTop 2020 diagnostics")
    print(audited.sort_values("audit2020_ic", ascending=False)[show].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
