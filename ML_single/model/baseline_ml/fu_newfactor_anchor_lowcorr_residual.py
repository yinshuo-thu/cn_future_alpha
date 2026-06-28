#!/usr/bin/env python3
"""Low-correlation residual LGB on top of the FU top-K rolling ensemble.

The anchor is a 2019-only weighted Ridge/LGB/MLP rolling ensemble.  This script
uses only meta/chain features to predict the remaining cross-sectional residual
month by month.
"""

from __future__ import annotations

import argparse
import gc
import json
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
DEFAULT_TOPK_DIR = ROOT / "strict_opt_results" / "fu_newfactor_topk_best_2019q4"
DEFAULT_OUT = DEFAULT_TOPK_DIR / "lowcorr_anchor_residual"


@dataclass(frozen=True)
class ResidualConfig:
    name: str = "topk_anchor_lowcorr_lgb_meta_chain_xsz"
    max_rows: int = 240_000
    seed: int = 9501
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


def read_component(topk_dir: Path, part: str, month: str, view: str) -> pd.DataFrame:
    path = topk_dir / "prediction_parts" / part / f"{month}.parquet"
    cur = pd.read_parquet(path, columns=["symbol", "datetime", "label", view])
    cur["datetime"] = pd.to_datetime(cur["datetime"])
    return cur.rename(columns={view: part})


def build_anchor(topk_dir: Path, model_name: str) -> pd.DataFrame:
    summary = pd.read_csv(topk_dir / "rolling_three_2019full_weight_summary.csv")
    row = summary[summary["model"] == model_name]
    if row.empty:
        row = summary.sort_values(["pred_xsz_ic_2020", "monthly_mean"], ascending=False).head(1)
    row = row.iloc[0]
    view = str(row["view"])
    weights = {
        "rolling_ridge": float(row["w_ridge"]),
        "rolling_lgb": float(row["w_lgb"]),
        "rolling_mlp": float(row["w_mlp"]),
    }
    pieces = []
    for month in month_range("2019-01", "2020-12"):
        base = None
        for part in ["rolling_ridge", "rolling_lgb", "rolling_mlp"]:
            cur = read_component(topk_dir, part, month, view)
            if base is None:
                base = cur
            else:
                base = base.merge(cur[["symbol", "datetime", part]], on=["symbol", "datetime"], how="inner")
        assert base is not None
        pred = np.zeros(len(base), dtype=np.float32)
        for part, weight in weights.items():
            pred += base[part].to_numpy(np.float32) * np.float32(weight)
        out = base[["symbol", "datetime", "label"]].copy()
        out["pred"] = pred
        out = add_cross_sectional_norms(out, "pred")
        out = out.rename(columns={"pred_xsz": "anchor_pred_xsz", "pred": "anchor_pred"})
        pieces.append(out[["symbol", "datetime", "label", "anchor_pred", "anchor_pred_xsz"]])
    anchor = pd.concat(pieces, ignore_index=True)
    anchor["datetime"] = pd.to_datetime(anchor["datetime"])
    return anchor


def attach_anchor(data: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    out = data.merge(anchor[["symbol", "datetime", "anchor_pred_xsz"]], on=["symbol", "datetime"], how="left")
    out["anchor_pred_xsz"] = out["anchor_pred_xsz"].astype(np.float32)
    return out


def train_mask_for_month(data: pd.DataFrame, ms: pd.Timestamp, first_pos: dict[pd.Timestamp, pd.Series], cfg: ResidualConfig) -> np.ndarray:
    mask = (
        (data["datetime"] >= pd.Timestamp("2019-01-01"))
        & (data["datetime"] < ms)
        & data["label_xsz"].notna()
        & data["anchor_pred_xsz"].notna()
    )
    starts = first_pos.get(ms)
    if starts is not None:
        cutoff = data["symbol"].map(starts).astype("float64")
        keep = cutoff.isna() | (data["_bar_no"].astype("float64") < cutoff - cfg.embargo_bars)
        mask &= keep
    return mask.to_numpy()


def residual_target(data: pd.DataFrame, idx: np.ndarray) -> tuple[np.ndarray, float]:
    y = data.iloc[idx]["label_xsz"].to_numpy(np.float64)
    p = data.iloc[idx]["anchor_pred_xsz"].to_numpy(np.float64)
    good = np.isfinite(y) & np.isfinite(p)
    yy = y[good]
    pp = p[good]
    beta = float((pp @ yy) / max(pp @ pp, 1e-12))
    resid = y - beta * p
    return np.clip(resid, -8, 8).astype(np.float32), beta


def fit_predict_month(data: pd.DataFrame, cfg: ResidualConfig, feat_cols: list[str], first_pos: dict[pd.Timestamp, pd.Series], ms: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, object]]:
    test_mask = (data["datetime"] >= ms) & (data["datetime"] < ms + pd.DateOffset(months=1))
    te_idx = np.flatnonzero(test_mask.to_numpy())
    out = data.iloc[te_idx][["symbol", "datetime", "label", "anchor_pred_xsz"]].copy()
    train_mask = train_mask_for_month(data, ms, first_pos, cfg)
    tr_idx = sample_indices(data, train_mask, cfg.max_rows, cfg.seed + int(ms.year * 12 + ms.month))
    if len(tr_idx) < 50_000 or len(te_idx) == 0:
        out["resid_pred"] = 0.0
        out["combined_pred"] = out["anchor_pred_xsz"].fillna(0.0).astype(np.float32)
        return out, {"month": f"{ms:%Y-%m}", "train_rows": int(len(tr_idx)), "beta": np.nan, "resid_ic": compute_ic(out["resid_pred"], out["label"])}

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
        n_jobs=4,
        random_state=cfg.seed,
        verbose=-1,
        force_col_wise=True,
    )
    model.fit(xtr, ytr)
    resid_pred = model.predict(xte).astype(np.float32)
    out["resid_pred"] = resid_pred
    out["combined_pred"] = (np.float32(beta) * out["anchor_pred_xsz"].fillna(0.0).to_numpy(np.float32) + resid_pred).astype(np.float32)
    info = {
        "month": f"{ms:%Y-%m}",
        "train_rows": int(len(tr_idx)),
        "test_rows": int(len(te_idx)),
        "beta": beta,
        "resid_ic": compute_ic(out["resid_pred"].to_numpy(), out["label"].to_numpy()),
        "combined_ic": compute_ic(out["combined_pred"].to_numpy(), out["label"].to_numpy()),
    }
    print(
        f"[anchor-lowcorr][{ms:%Y-%m}] tr={len(tr_idx):7d} te={len(te_idx):6d} "
        f"beta={beta:.4f} residIC={info['resid_ic']:.5f} combIC={info['combined_ic']:.5f}",
        flush=True,
    )
    del xtr, xte, ytr, model
    gc.collect()
    return out, info


def summarize_signal(pred: pd.DataFrame, pred_col: str, model: str) -> dict[str, object]:
    out = pred[["symbol", "datetime", "label", pred_col]].rename(columns={pred_col: "pred"}).copy()
    out = add_cross_sectional_norms(out, "pred")
    rows: dict[str, object] = {"model": model, "rows": len(out), "label_rows": int(out["label"].notna().sum())}
    for tag, start, end in [
        ("2019", pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01")),
        ("2020", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")),
    ]:
        sub = out[(out["datetime"] >= start) & (out["datetime"] < end)]
        for col in ["pred", "pred_xsz", "pred_xrank"]:
            monthly = period_ic(sub, col, "M")
            rows[f"{col}_ic_{tag}"] = compute_ic(sub[col].to_numpy(), sub["label"].to_numpy())
            rows[f"{col}_monthly_mean_{tag}"] = float(monthly.mean())
            rows[f"{col}_monthly_ir_{tag}"] = float(monthly.mean() / monthly.std(ddof=1)) if monthly.std(ddof=1) > 0 else float("nan")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-dir", type=Path, default=DEFAULT_TOPK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--anchor-model", type=str, default="rolling_three_2019full_pred_nonneg_meq")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ResidualConfig()
    pred_path = args.out_dir / f"{cfg.name}.parquet"
    summary_path = args.out_dir / "summary.csv"
    if pred_path.exists() and summary_path.exists() and not args.force:
        print(pd.read_csv(summary_path).to_string(index=False), flush=True)
        return

    anchor = build_anchor(args.topk_dir, args.anchor_model)
    data = build_meta_features()
    data = attach_anchor(data, anchor)
    data = data[(data["datetime"] >= pd.Timestamp("2019-01-01")) & (data["datetime"] < pd.Timestamp("2021-01-01"))].copy()
    data = data.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    first_pos = first_bar_by_month(data)
    feat_cols = feature_columns("all")

    pieces = []
    info_rows = []
    for ms in pd.date_range("2019-01-01", "2020-12-01", freq="MS"):
        part, info = fit_predict_month(data, cfg, feat_cols, first_pos, ms)
        pieces.append(part)
        info_rows.append(info)
    pred = pd.concat(pieces, ignore_index=True).sort_values(["datetime", "symbol"], kind="mergesort").reset_index(drop=True)
    write_parquet_atomic(pred, pred_path)
    pd.DataFrame(info_rows).to_csv(args.out_dir / f"{cfg.name}_monthly_train_log.csv", index=False)
    rows = [
        summarize_signal(pred, "anchor_pred_xsz", "topk_anchor_xsz"),
        summarize_signal(pred, "resid_pred", cfg.name + "_resid"),
        summarize_signal(pred, "combined_pred", cfg.name + "_combined"),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    (args.out_dir / f"{cfg.name}.json").write_text(json.dumps(asdict(cfg) | {"anchor_model": args.anchor_model}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
