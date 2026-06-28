#!/usr/bin/env python3
"""Residual low-correlation learner anchored on the new rolling LGB signal."""

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import lightgbm as lgb
import numpy as np
import pandas as pd

from lowcorr_assist_models import build_meta_features, feature_columns, first_bar_by_month, sample_indices, scrub_array, write_parquet_atomic
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, period_ic


ROOT = Path("/root/autodl-tmp/quant/ML")
FU_DIR = ROOT / "strict_opt_results" / "fu_newfactor_three_model"
OUT_DIR = FU_DIR / "lowcorr_residual"
LGB_PARTS = FU_DIR / "prediction_parts" / "rolling_lgb"

TRAIN_START = pd.Timestamp("2019-01-01")
PRED_START = pd.Timestamp("2019-01-01")
FIT_START = pd.Timestamp("2019-03-01")
TEST_END = pd.Timestamp("2021-01-01")


@dataclass(frozen=True)
class ResidualConfig:
    name: str = "new_lgb_resid_lgb_meta_chain_xsz"
    max_rows: int = 280_000
    seed: int = 9401
    n_estimators: int = 120
    learning_rate: float = 0.035
    num_leaves: int = 31
    min_child_samples: int = 180
    reg_lambda: float = 12.0
    colsample_bytree: float = 0.86
    subsample: float = 0.85
    embargo_bars: int = 30


def month_range(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def load_new_lgb_signal() -> pd.DataFrame:
    pieces = []
    for month in month_range("2019-01", "2020-12"):
        path = LGB_PARTS / f"{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_parquet(path, columns=["symbol", "datetime", "pred_xsz"])
        pieces.append(part)
    pred = pd.concat(pieces, ignore_index=True)
    pred["datetime"] = pd.to_datetime(pred["datetime"])
    pred = pred.rename(columns={"pred_xsz": "new_lgb_pred_xsz"})
    return pred


def attach_signal(data: pd.DataFrame) -> pd.DataFrame:
    pred = load_new_lgb_signal()
    out = data.merge(pred, on=["symbol", "datetime"], how="left")
    out["new_lgb_pred_xsz"] = out["new_lgb_pred_xsz"].astype(np.float32)
    return out


def train_mask_for_month(data: pd.DataFrame, ms: pd.Timestamp, first_pos: dict[pd.Timestamp, pd.Series], cfg: ResidualConfig) -> np.ndarray:
    mask = (
        (data["datetime"] >= TRAIN_START)
        & (data["datetime"] < ms)
        & data["label_xsz"].notna()
        & data["new_lgb_pred_xsz"].notna()
    )
    starts = first_pos.get(ms)
    if starts is not None:
        cutoff = data["symbol"].map(starts).astype("float64")
        keep = cutoff.isna() | (data["_bar_no"].astype("float64") < cutoff - cfg.embargo_bars)
        mask &= keep
    return mask.to_numpy()


def residual_target(data: pd.DataFrame, idx: np.ndarray) -> tuple[np.ndarray, float]:
    y = data.iloc[idx]["label_xsz"].to_numpy(np.float64)
    p = data.iloc[idx]["new_lgb_pred_xsz"].to_numpy(np.float64)
    good = np.isfinite(y) & np.isfinite(p)
    yy = y[good]
    pp = p[good]
    beta = float((pp @ yy) / max(pp @ pp, 1e-12))
    resid = y - beta * p
    return np.clip(resid, -8, 8).astype(np.float32), beta


def fit_predict_month(data: pd.DataFrame, cfg: ResidualConfig, feat_cols: list[str], first_pos: dict[pd.Timestamp, pd.Series], ms: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, object]]:
    train_mask = train_mask_for_month(data, ms, first_pos, cfg)
    tr_idx = sample_indices(data, train_mask, cfg.max_rows, cfg.seed + int(ms.year * 12 + ms.month))
    test_mask = (data["datetime"] >= ms) & (data["datetime"] < ms + pd.DateOffset(months=1))
    te_idx = np.flatnonzero(test_mask.to_numpy())
    if len(tr_idx) < 50_000:
        out = data.iloc[te_idx][["symbol", "datetime", "label"]].copy()
        out["pred"] = 0.0
        return out, {"month": f"{ms:%Y-%m}", "train_rows": int(len(tr_idx)), "beta": np.nan, "month_ic": compute_ic(out["pred"], out["label"])}

    xtr = scrub_array(data.iloc[tr_idx][feat_cols].to_numpy(np.float32))
    ytr, beta = residual_target(data, tr_idx)
    xte = scrub_array(data.iloc[te_idx][feat_cols].to_numpy(np.float32))
    model = lgb.LGBMRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_samples=cfg.min_child_samples,
        reg_lambda=cfg.reg_lambda,
        n_jobs=int(os.environ.get("FU_LOWCORR_N_JOBS", "4")),
        random_state=cfg.seed,
        verbose=-1,
        force_col_wise=True,
    )
    model.fit(xtr, ytr)
    pred = model.predict(xte).astype(np.float32)
    out = data.iloc[te_idx][["symbol", "datetime", "label"]].copy()
    out["pred"] = pred
    info = {
        "month": f"{ms:%Y-%m}",
        "train_rows": int(len(tr_idx)),
        "test_rows": int(len(te_idx)),
        "beta": beta,
        "month_ic": compute_ic(out["pred"].to_numpy(), out["label"].to_numpy()),
    }
    print(f"[new-lowcorr][{ms:%Y-%m}] tr={len(tr_idx):7d} te={len(te_idx):6d} beta={beta:.4f} IC={info['month_ic']:.5f}", flush=True)
    del xtr, xte, ytr, model
    gc.collect()
    return out, info


def summarize(pred: pd.DataFrame, cfg: ResidualConfig) -> dict[str, object]:
    row: dict[str, object] = {"model": cfg.name, "rows": len(pred), "label_rows": int(pred["label"].notna().sum())}
    for window, start, end in [
        ("2019", pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01")),
        ("2020", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")),
    ]:
        part = pred[(pred["datetime"] >= start) & (pred["datetime"] < end)]
        for col in ["pred", "pred_xsz", "pred_xrank"]:
            mic = period_ic(part, col, "M")
            row[f"{col}_ic_{window}"] = compute_ic(part[col].to_numpy(), part["label"].to_numpy())
            row[f"{col}_monthly_mean_{window}"] = float(mic.mean())
            row[f"{col}_monthly_ir_{window}"] = float(mic.mean() / mic.std(ddof=1)) if mic.std(ddof=1) > 0 else float("nan")
    return row


def main() -> None:
    cfg = ResidualConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUT_DIR / f"{cfg.name}.parquet"
    summary_path = OUT_DIR / "summary.csv"
    if pred_path.exists() and summary_path.exists() and os.environ.get("FU_LOWCORR_FORCE", "0") != "1":
        print(pd.read_csv(summary_path).to_string(index=False), flush=True)
        return

    data = build_meta_features()
    data = attach_signal(data)
    data = data[(data["datetime"] >= PRED_START) & (data["datetime"] < TEST_END)].copy()
    data = data.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    first_pos = first_bar_by_month(data)
    feat_cols = feature_columns("all")

    pieces = []
    info_rows = []
    for ms in pd.date_range(PRED_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS"):
        part, info = fit_predict_month(data, cfg, feat_cols, first_pos, ms)
        pieces.append(part)
        info_rows.append(info)
    pred = pd.concat(pieces, ignore_index=True).sort_values(["datetime", "symbol"], kind="mergesort").reset_index(drop=True)
    pred = add_cross_sectional_norms(pred, "pred")
    write_parquet_atomic(pred, pred_path)
    pd.DataFrame(info_rows).to_csv(OUT_DIR / f"{cfg.name}_monthly_train_log.csv", index=False)
    row = summarize(pred, cfg)
    pd.DataFrame([row]).to_csv(summary_path, index=False)
    (OUT_DIR / f"{cfg.name}.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    print(pd.DataFrame([row]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
