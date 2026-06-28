#!/usr/bin/env python3
"""
Strict train-before-test monthly LightGBM rebuild from the materialized factors.

Prediction months:
  - 2019-01..2019-12: OOS history used only to train the final gate
  - 2020-01..2020-12: final test

For each predicted month, the LightGBM model is fit only on rows with
datetime < month_start and datetime >= 2018-01-01.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic
from src.plan_a.group_lgb import symbol_group_map


FACTOR_PATH = Path("/root/shared-nvme/feature_model/data_factors_big.parquet")
SELECTED_PATH = Path("/root/quant/work/outputs/selected_factors.txt")
OUT_DIR = Path("/root/autodl-tmp/quant/ML/strict_lgb_results")

TRAIN_START = pd.Timestamp("2018-01-01")
PRED_START = pd.Timestamp("2019-01-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def load_factor_data(top_n: int) -> tuple[pd.DataFrame, list[str]]:
    selected = [x.strip() for x in SELECTED_PATH.read_text().splitlines() if x.strip()]
    feat_cols = selected[:top_n]
    cols = ["symbol", "datetime", "label"] + feat_cols
    print(f"[load] reading {len(cols)} cols from {FACTOR_PATH}", flush=True)
    data = pd.read_parquet(
        FACTOR_PATH,
        columns=cols,
        filters=[
            ("datetime", ">=", TRAIN_START),
            ("datetime", "<", TEST_END),
        ],
    )
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    print(
        f"[load] rows={len(data)} date={data.datetime.min()}..{data.datetime.max()} "
        f"features={len(feat_cols)}",
        flush=True,
    )
    return data, feat_cols


def add_context(data: pd.DataFrame) -> pd.DataFrame:
    groups = symbol_group_map()
    out = data
    symbols = sorted(out["symbol"].unique())
    sym_map = {s: i for i, s in enumerate(symbols)}
    group_names = sorted(set(groups.values()) | {"other"})
    grp_map = {g: i for i, g in enumerate(group_names)}
    out["symbol_code"] = out["symbol"].map(sym_map).astype(np.int16)
    out["group_code"] = out["symbol"].map(groups).fillna("other").map(grp_map).astype(np.int8)
    minute = (out["datetime"].dt.hour * 60 + out["datetime"].dt.minute).astype(np.float32)
    out["minute_sin"] = np.sin(2 * np.pi * minute / 1440.0).astype(np.float32)
    out["minute_cos"] = np.cos(2 * np.pi * minute / 1440.0).astype(np.float32)
    dow = out["datetime"].dt.dayofweek.astype(np.float32)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    month = out["datetime"].dt.month.astype(np.float32)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0).astype(np.float32)
    return out


def add_target_transforms(data: pd.DataFrame) -> pd.DataFrame:
    g = data.groupby("datetime", sort=False)["label"]
    mu = g.transform("mean")
    sd = g.transform("std")
    data["label_xsz"] = ((data["label"] - mu) / (sd + 1e-9)).astype(np.float32)
    data["label_xrank"] = (g.rank(pct=True) - 0.5).astype(np.float32)
    return data


def make_x(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(scrub(df[feat_cols].to_numpy(np.float32)), columns=feat_cols, index=df.index)
    for col in ["symbol_code", "group_code"]:
        x[col] = df[col].to_numpy()
    for col in ["minute_sin", "minute_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        x[col] = df[col].to_numpy(np.float32)
    return x


def summarize(pred: pd.DataFrame, model: str) -> dict:
    test = pred[(pred.datetime >= TEST_START) & (pred.datetime < TEST_END)].copy()
    row: dict[str, object] = {"model": model, "rows": len(test), "label_rows": int(test.label.notna().sum())}
    for col in ["pred", "pred_xsz", "pred_xrank"]:
        by_m = period_ic(test, col, "M")
        row[f"{col}_ic_2020"] = compute_ic(test[col].to_numpy(), test["label"].to_numpy())
        row[f"{col}_monthly_mean"] = float(by_m.mean())
        row[f"{col}_monthly_ir"] = float(by_m.mean() / by_m.std()) if by_m.std() > 0 else float("nan")
    return row


def train_monthly_lgb(
    data: pd.DataFrame,
    feat_cols: list[str],
    *,
    target_mode: str,
    max_rows: int,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    target_col = {"raw": "label", "xsz": "label_xsz", "xrank": "label_xrank"}[target_mode]
    params = dict(
        n_estimators=260,
        learning_rate=0.04,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.65,
        min_child_samples=120,
        reg_lambda=4.0,
        n_jobs=n_jobs,
        random_state=seed,
        verbose=-1,
        force_col_wise=True,
    )
    out_mask = (data["datetime"] >= PRED_START) & (data["datetime"] < TEST_END)
    out = data.loc[out_mask, ["symbol", "datetime", "label"]].copy()
    out["pred"] = np.nan
    months = pd.date_range(PRED_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS")
    rng = np.random.default_rng(seed)
    for mi, ms in enumerate(months):
        next_ms = ms + pd.DateOffset(months=1)
        tr_mask = (
            (data["datetime"] >= TRAIN_START)
            & (data["datetime"] < ms)
            & data[target_col].notna()
            & data["label"].notna()
        )
        pr_mask = (data["datetime"] >= ms) & (data["datetime"] < next_ms)
        tr_idx = np.flatnonzero(tr_mask.to_numpy())
        pr_idx = np.flatnonzero(pr_mask.to_numpy())
        if len(tr_idx) < 5000 or len(pr_idx) == 0:
            print(f"  [lgb-{target_mode}][{ms:%Y-%m}] skip tr={len(tr_idx)} pr={len(pr_idx)}", flush=True)
            continue
        if max_rows and len(tr_idx) > max_rows:
            tr_idx = rng.choice(tr_idx, max_rows, replace=False)
        tr = data.iloc[tr_idx]
        pr = data.iloc[pr_idx]
        xtr = make_x(tr, feat_cols)
        ytr = tr[target_col].to_numpy(np.float32)
        xpr = make_x(pr, feat_cols)
        model = lgb.LGBMRegressor(**params)
        model.fit(xtr, ytr, categorical_feature=["symbol_code", "group_code"])
        pred = model.predict(xpr)
        out.loc[pr.index, "pred"] = pred
        mic = compute_ic(pred, pr["label"].to_numpy())
        print(f"  [lgb-{target_mode}][{ms:%Y-%m}] tr={len(tr_idx):7d} pr={len(pr_idx):6d} IC={mic:.5f}", flush=True)
        del xtr, xpr, tr, pr, model
    pred = add_cross_sectional_norms(out, "pred")
    return pred


def component_matrix(preds: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    base = None
    names = []
    for name, df in preds.items():
        cur = df[["symbol", "datetime", "label", "pred", "pred_xsz", "pred_xrank"]].copy()
        cur = cur.rename(
            columns={
                "pred": f"{name}_raw",
                "pred_xsz": f"{name}_xsz",
                "pred_xrank": f"{name}_xrank",
            }
        )
        cur_names = [f"{name}_raw", f"{name}_xsz", f"{name}_xrank"]
        if base is None:
            base = cur[["symbol", "datetime", "label"] + cur_names]
        else:
            base = base.merge(cur[["symbol", "datetime"] + cur_names], on=["symbol", "datetime"], how="inner")
        names.extend(cur_names)
    assert base is not None
    return base, names


def fit_gate(base: pd.DataFrame, names: list[str], *, signed: bool) -> tuple[pd.DataFrame, dict]:
    train = base[(base.datetime >= PRED_START) & (base.datetime < TEST_START)]
    test = base[(base.datetime >= TEST_START) & (base.datetime < TEST_END)].copy()
    x = train[names].to_numpy(np.float32)
    y = train["label"].to_numpy(np.float64)
    x = scrub(x).astype(np.float64, copy=False)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    c = x.T @ y
    g = x.T @ x
    yy = float(y @ y)
    lower = np.full(len(names), -0.15 if signed else 0.0, dtype=np.float64)
    upper = np.full(len(names), 1.10 if signed else 0.90, dtype=np.float64)
    w, train_ic = fit_ic_weights_from_stats(c, g, yy, lower, upper)
    xt = scrub(test[names].to_numpy(np.float32))
    pred = xt @ w.astype(np.float32)
    out = test[["symbol", "datetime", "label"]].copy()
    out["pred"] = pred
    out = add_cross_sectional_norms(out, "pred")
    meta = {
        "model": "strict_lgb_gate_signed" if signed else "strict_lgb_gate_nonneg",
        "gate_train_2019_ic": train_ic,
        "pred_ic_2020": compute_ic(out["pred"].to_numpy(), out["label"].to_numpy()),
        **{f"w_{n}": float(v) for n, v in zip(names, w)},
    }
    return out, meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    top_n = int(__import__("os").environ.get("STRICT_LGB_TOP_N", "650"))
    max_rows = int(__import__("os").environ.get("STRICT_LGB_MAX_ROWS", "600000"))
    n_jobs = int(__import__("os").environ.get("STRICT_LGB_N_JOBS", "16"))
    seed = int(__import__("os").environ.get("STRICT_LGB_SEED", "42"))
    data, feat_cols = load_factor_data(top_n)
    data = add_context(data)
    data = add_target_transforms(data)
    preds = {}
    summary_rows = []
    for mode in ["raw", "xsz", "xrank"]:
        pred = train_monthly_lgb(data, feat_cols, target_mode=mode, max_rows=max_rows, seed=seed + len(preds), n_jobs=n_jobs)
        name = f"strict_lgb_{mode}_top{top_n}_n{max_rows}"
        pred.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        preds[mode] = pred
        summary_rows.append(summarize(pred, name))
    base, names = component_matrix(preds)
    base.to_parquet(OUT_DIR / f"strict_lgb_components_top{top_n}_n{max_rows}.parquet", index=False)
    gate_rows = []
    for signed in [False, True]:
        gated, meta = fit_gate(base, names, signed=signed)
        gated.to_parquet(OUT_DIR / f"{meta['model']}_top{top_n}_n{max_rows}.parquet", index=False)
        row = summarize(gated, meta["model"])
        row.update(meta)
        gate_rows.append(row)
        by_m = period_ic(gated, "pred", "M")
        by_m.to_csv(OUT_DIR / f"{meta['model']}_monthly_ic.csv")
    summary = pd.DataFrame(summary_rows + gate_rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    metadata = {"top_n": top_n, "max_rows": max_rows, "n_jobs": n_jobs, "features": feat_cols}
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[["model", "pred_ic_2020", "pred_monthly_mean", "pred_monthly_ir"]].to_string(index=False), flush=True)
    print(f"[done] wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
