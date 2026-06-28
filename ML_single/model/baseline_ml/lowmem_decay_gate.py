#!/usr/bin/env python3
"""Validation-selected recency-decayed static gate.

Protocol:
  - use 2019-01..2019-09 to select a window/decay on 2019-Q4;
  - refit the selected recipe using only 2019 data;
  - report 2020 once.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import numpy as np
import pandas as pd

from lowmem_rolling_gate import ensure_memmap
from lowmem_static_gate import scrub
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic
from strict_optimization_ablation import OUT_DIR, PRED_START, TEST_END, TEST_START, summarize


OUT_SUBDIR = OUT_DIR / "decay_gate"
OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
VAL_START = pd.Timestamp("2019-10-01")


@dataclass(frozen=True)
class Recipe:
    target: str
    balance: str
    window_months: int
    half_life_months: float
    signed: bool = True


def label_xsz(base: pd.DataFrame) -> np.ndarray:
    g = base.groupby("datetime", sort=False)["label"]
    z = ((base["label"] - g.transform("mean")) / (g.transform("std") + 1e-9)).clip(-8, 8)
    return z.astype(np.float64).to_numpy()


def month_weight(month_start: pd.Timestamp, end: pd.Timestamp, half_life: float) -> float:
    if half_life <= 0:
        return 1.0
    age = (end.year - month_start.year) * 12 + (end.month - month_start.month)
    return float(0.5 ** (max(age, 0) / half_life))


def fit_recipe(
    base: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    end: pd.Timestamp,
    recipe: Recipe,
) -> tuple[np.ndarray, float]:
    start = max(PRED_START, end - pd.DateOffset(months=recipe.window_months))
    dt = base["datetime"]
    lower = np.full(x.shape[1], -0.12 if recipe.signed else 0.0, dtype=np.float64)
    upper = np.full(x.shape[1], 0.85 if recipe.signed else 0.75, dtype=np.float64)
    c = np.zeros(x.shape[1], dtype=np.float64)
    g = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    yy = 0.0
    denom = 0.0
    for ms in pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS"):
        me = ms + pd.DateOffset(months=1)
        mask = ((dt >= ms) & (dt < me) & base["label"].notna()).to_numpy()
        if int(mask.sum()) < 10_000:
            continue
        xt = scrub(x[mask]).astype(np.float64, copy=False)
        yt = y[mask].astype(np.float64, copy=False)
        ok = np.isfinite(yt) & np.all(np.isfinite(xt), axis=1)
        if int(ok.sum()) < 10_000:
            continue
        xt = xt[ok]
        yt = yt[ok]
        w = month_weight(ms, end, recipe.half_life_months)
        if recipe.balance == "month":
            w /= float(len(yt))
        c += (xt.T @ yt) * w
        g += (xt.T @ xt) * w
        yy += float(yt @ yt) * w
        denom += w if recipe.balance == "row" else month_weight(ms, end, recipe.half_life_months)
    if denom <= 0:
        raise RuntimeError(f"no data for recipe {recipe}")
    weights, _ = fit_ic_weights_from_stats(c, g, yy, lower, upper)
    train_mask = ((dt >= start) & (dt < end) & base["label"].notna()).to_numpy()
    train_pred = scrub(x[train_mask]) @ weights.astype(np.float32)
    train_ic = compute_ic(train_pred, base.loc[train_mask, "label"].to_numpy())
    return weights, train_ic


def predict(base: pd.DataFrame, x: np.ndarray, weights: np.ndarray, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dt = base["datetime"]
    mask = ((dt >= start) & (dt < end)).to_numpy()
    out = base.loc[mask, ["symbol", "datetime", "label"]].copy()
    out["pred"] = (scrub(x[mask]) @ weights.astype(np.float32)).astype(np.float32)
    return add_cross_sectional_norms(out, "pred")


def main() -> None:
    base, names, x = ensure_memmap()
    raw_y = base["label"].to_numpy(np.float64)
    xsz_y = label_xsz(base)
    y_by_name = {"raw": raw_y, "xsz": xsz_y}
    recipes = [
        Recipe(target=target, balance=balance, window_months=window, half_life_months=half_life)
        for target in ["raw", "xsz"]
        for balance in ["row", "month"]
        for window in [3, 6, 9]
        for half_life in [0.0, 1.5, 3.0, 6.0]
    ]
    grid_rows = []
    for recipe in recipes:
        weights, train_ic = fit_recipe(base, x, y_by_name[recipe.target], VAL_START, recipe)
        val = predict(base, x, weights, VAL_START, TEST_START)
        val_ic = compute_ic(val["pred"].to_numpy(), val["label"].to_numpy())
        row = asdict(recipe) | {"train_ic_2019q1q3": train_ic, "val_ic_2019q4": val_ic}
        grid_rows.append(row)
        print(f"[decay-gate][val] {row}", flush=True)
    grid = pd.DataFrame(grid_rows).sort_values("val_ic_2019q4", ascending=False)
    grid.to_csv(OUT_SUBDIR / "validation_grid.csv", index=False)
    best = Recipe(
        target=str(grid.iloc[0]["target"]),
        balance=str(grid.iloc[0]["balance"]),
        window_months=int(grid.iloc[0]["window_months"]),
        half_life_months=float(grid.iloc[0]["half_life_months"]),
        signed=bool(grid.iloc[0]["signed"]),
    )
    weights, train_ic = fit_recipe(base, x, y_by_name[best.target], TEST_START, best)
    pred = predict(base, x, weights, TEST_START, TEST_END)
    pred.to_parquet(OUT_SUBDIR / "decay_gate_selected.parquet", index=False)
    monthly = period_ic(pred, "pred", "M")
    monthly.to_csv(OUT_SUBDIR / "monthly_ic.csv")
    summary = pd.DataFrame([summarize(pred, "decay_gate_selected") | {"gate_train_ic_2019": train_ic, "selected_val_ic_2019q4": float(grid.iloc[0]["val_ic_2019q4"])} | asdict(best)])
    summary.to_csv(OUT_SUBDIR / "summary.csv", index=False)
    pd.DataFrame([{"component": n, "weight": float(w)} for n, w in zip(names, weights)]).to_csv(OUT_SUBDIR / "weights.csv", index=False)
    (OUT_SUBDIR / "metadata.json").write_text(json.dumps({"components": names, "selected_recipe": asdict(best)}, indent=2), encoding="utf-8")
    print(summary[["model", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020", "selected_val_ic_2019q4", "target", "balance", "window_months", "half_life_months"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
