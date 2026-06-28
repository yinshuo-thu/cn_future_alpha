#!/usr/bin/env python3
"""Clean 2019Q4-selected market-direction overlay for the view-stack model.

The view-stack is mostly cross-sectional.  This script adds a date-level
prediction of the raw label mean, selected only on 2019Q4, then reports 2020.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

from expanded_gate_view_stack import (  # noqa: E402
    StackSpec,
    apply_stack,
    choose_columns,
    fit_stack,
    rolling_predict_period,
    selected_config_names,
)
from expanded_history_gate_clean import (  # noqa: E402
    OUT_DIR,
    TEST_END,
    TEST_START,
    TRAIN_START,
    VAL_START,
    configs,
    finalize_matrix,
    fit_weights,
    mask_between,
    predict_frame,
    summarize,
)
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, period_ic  # noqa: E402


OUT_PREFIX = OUT_DIR / "market_overlay"
ALPHAS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0]
GAMMAS = [0.0, 0.15, 0.30, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


def selected_stack_spec() -> StackSpec:
    summary_path = OUT_DIR / "view_stack_summary.csv"
    if summary_path.exists():
        row = pd.read_csv(summary_path).query("selected_by_2019q4 == True").iloc[0]
        return StackSpec(
            name=str(row["model"]).replace("expanded_gate_view_stack__", ""),
            modes=tuple(str(row["modes"]).split("+")),
            views=tuple(str(row["views"]).split("+")),
            top_n=int(row["top_n"]),
            target=str(row["target"]),
            standardize=bool(row["standardize"]),
            upper=float(row["upper"]),
        )
    return StackSpec(
        name="fixed+rolling__pred__top10__xsz__std1__u1",
        modes=("fixed", "rolling"),
        views=("pred",),
        top_n=10,
        target="xsz",
        standardize=True,
        upper=1.0,
    )


def build_stack_design(base: pd.DataFrame, x: np.ndarray, names: list[str], families: list[str], spec: StackSpec):
    name_to_idx = {n: i for i, n in enumerate(names)}
    cfg_by_name = {cfg.name: cfg for cfg in configs(names, families)}
    base_order = selected_config_names()[: spec.top_n]

    dt = base["datetime"]
    train_mask = mask_between(dt, TRAIN_START, VAL_START, base["label"])
    full_train_mask = mask_between(dt, TRAIN_START, TEST_START, base["label"])
    val_mask = ((dt >= VAL_START) & (dt < TEST_START)).to_numpy()
    test_mask = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()

    val_base = base.loc[val_mask, ["symbol", "datetime", "label", "label_xsz_fit"]].copy().reset_index(drop=True)
    test_base = base.loc[test_mask, ["symbol", "datetime", "label", "label_xsz_fit"]].copy().reset_index(drop=True)

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feature_names: list[str] = []
    rows = []
    for pos, cfg_name in enumerate(base_order, start=1):
        cfg = cfg_by_name[cfg_name]
        cols = [name_to_idx[c] for c in cfg.components if c in name_to_idx]
        if not cols:
            continue

        w_val, train_ic = fit_weights(base, x, cols, train_mask, cfg)
        val_fixed = predict_frame(base, x, cols, w_val, val_mask).reset_index(drop=True)
        w_final, final_train_ic = fit_weights(base, x, cols, full_train_mask, cfg)
        test_fixed = predict_frame(base, x, cols, w_final, test_mask).reset_index(drop=True)

        val_roll = rolling_predict_period(base, x, cols, cfg, VAL_START, TEST_START).reset_index(drop=True)
        test_roll = rolling_predict_period(base, x, cols, cfg, TEST_START, TEST_END).reset_index(drop=True)

        for view in ("pred", "pred_xsz", "pred_xrank"):
            for mode, vp, tp in [("fixed", val_fixed, test_fixed), ("rolling", val_roll, test_roll)]:
                fname = f"{mode}::{cfg_name}::{view}"
                arrays[fname] = (
                    vp[view].to_numpy(np.float32, copy=False),
                    tp[view].to_numpy(np.float32, copy=False),
                )
                feature_names.append(fname)
        rows.append(
            {
                "rank": pos,
                "config": cfg_name,
                "k": len(cols),
                "fixed_train_ic_q1q3": float(train_ic),
                "fixed_train_ic_2019": float(final_train_ic),
            }
        )
        print(f"[market-overlay][stack-base {pos:02d}/{len(base_order):02d}] {cfg_name}", flush=True)

    xv = np.column_stack([arrays[n][0] for n in feature_names]).astype(np.float32)
    xt = np.column_stack([arrays[n][1] for n in feature_names]).astype(np.float32)
    cols = choose_columns(feature_names, spec, base_order)
    y = (
        val_base["label_xsz_fit"].to_numpy(np.float64, copy=False)
        if spec.target == "xsz"
        else val_base["label"].to_numpy(np.float64, copy=False)
    )
    w, val_ic, mean, scale = fit_stack(xv, y, cols, spec)
    val_pred = apply_stack(xv, cols, w, mean, scale, spec.standardize)
    test_pred = apply_stack(xt, cols, w, mean, scale, spec.standardize)
    return val_base, test_base, val_pred, test_pred, pd.DataFrame(rows), feature_names, cols


def date_feature_frame(base: pd.DataFrame, x: np.ndarray, component_cols: list[int], mask: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame(x[mask][:, component_cols], dtype=np.float32)
    dt = pd.to_datetime(base.loc[mask, "datetime"].to_numpy())
    frame["datetime"] = dt
    grouped = frame.groupby("datetime", sort=True)
    mean = grouped.mean()
    std = grouped.std().fillna(0.0)
    mean.columns = [f"c{c}_mean" for c in component_cols]
    std.columns = [f"c{c}_std" for c in component_cols]
    feat = mean.join(std, how="left").fillna(0.0)

    idx = pd.DatetimeIndex(feat.index)
    minute = (idx.hour * 60 + idx.minute).astype(np.float64)
    feat["minute_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    feat["minute_cos"] = np.cos(2 * np.pi * minute / 1440.0)
    dow = idx.dayofweek.astype(np.float64)
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    month = idx.month.astype(np.float64)
    feat["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    feat["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    y = base.loc[mask, ["datetime", "label"]].groupby("datetime", sort=True)["label"].mean().reindex(feat.index)
    return feat.astype(np.float32), y.astype(np.float64)


def fit_ridge(x: pd.DataFrame, y: pd.Series, alpha: float):
    arr = x.to_numpy(np.float64, copy=True)
    yy = y.to_numpy(np.float64, copy=True)
    good = np.isfinite(yy) & np.all(np.isfinite(arr), axis=1)
    arr = arr[good]
    yy = yy[good]
    mean = arr.mean(axis=0)
    scale = np.maximum(arr.std(axis=0), 1e-9)
    z = (arr - mean) / scale
    y_mean = float(yy.mean())
    y0 = yy - y_mean
    gram = z.T @ z / max(len(z), 1)
    cov = z.T @ y0 / max(len(z), 1)
    w = np.linalg.solve(gram + alpha * np.eye(z.shape[1]), cov)
    return {"mean": mean, "scale": scale, "weight": w, "y_mean": y_mean, "columns": list(x.columns), "alpha": alpha}


def predict_ridge(model: dict, x: pd.DataFrame) -> pd.Series:
    arr = x[model["columns"]].to_numpy(np.float64, copy=True)
    z = (np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) - model["mean"]) / model["scale"]
    pred = z @ model["weight"] + float(model["y_mean"])
    return pd.Series(pred.astype(np.float32), index=x.index)


def summarize_overlay(pred: pd.DataFrame, model: str, val_ic: float, market_val_ic: float, alpha: float, gamma: float) -> dict[str, object]:
    row = summarize(pred, model)
    row.update({"stack_val_ic_2019q4": val_ic, "market_val_ic_2019q4": market_val_ic, "alpha": alpha, "gamma": gamma})
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = selected_stack_spec()
    print(f"[market-overlay] stack_spec={asdict(spec)}", flush=True)

    base, x, names, families = finalize_matrix()
    val_base, test_base, stack_val, stack_test, base_configs, feature_names, stack_cols = build_stack_design(base, x, names, families, spec)
    base_configs.to_csv(f"{OUT_PREFIX}_stack_base_configs.csv", index=False)

    cfg_by_name = {cfg.name: cfg for cfg in configs(names, families)}
    component_names = []
    for cfg_name in selected_config_names()[: spec.top_n]:
        component_names.extend(cfg_by_name[cfg_name].components)
    component_cols = sorted({names.index(c) for c in component_names if c in names})

    dt = base["datetime"]
    train_mask = ((dt >= TRAIN_START) & (dt < VAL_START) & base["label"].notna()).to_numpy()
    full_train_mask = ((dt >= TRAIN_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    val_mask = ((dt >= VAL_START) & (dt < TEST_START)).to_numpy()
    test_mask = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()

    train_feat, train_y = date_feature_frame(base, x, component_cols, train_mask)
    val_feat, val_y = date_feature_frame(base, x, component_cols, val_mask)
    full_feat, full_y = date_feature_frame(base, x, component_cols, full_train_mask)
    test_feat, _ = date_feature_frame(base, x, component_cols, test_mask)

    val_dt = pd.to_datetime(val_base["datetime"])
    test_dt = pd.to_datetime(test_base["datetime"])
    val_rows = []
    best = None
    for alpha in ALPHAS:
        model = fit_ridge(train_feat, train_y, alpha)
        market_val_by_dt = predict_ridge(model, val_feat)
        market_val = market_val_by_dt.reindex(val_dt).to_numpy(np.float32)
        market_ic = compute_ic(market_val, val_base["label"].to_numpy())
        for gamma in GAMMAS:
            pred = stack_val.astype(np.float32) + gamma * market_val
            ic = compute_ic(pred, val_base["label"].to_numpy())
            row = {"alpha": alpha, "gamma": gamma, "val_ic_2019q4": ic, "market_val_ic_2019q4": market_ic}
            val_rows.append(row)
            key = (ic, market_ic, -alpha, -gamma)
            if best is None or key > best[0]:
                best = (key, row)
    assert best is not None
    selected = best[1]
    pd.DataFrame(val_rows).sort_values("val_ic_2019q4", ascending=False).to_csv(f"{OUT_PREFIX}_validation_grid.csv", index=False)
    print(f"[market-overlay] selected={selected}", flush=True)

    final_model = fit_ridge(full_feat, full_y, float(selected["alpha"]))
    market_test_by_dt = predict_ridge(final_model, test_feat)
    market_test = market_test_by_dt.reindex(test_dt).to_numpy(np.float32)
    out = test_base[["symbol", "datetime", "label"]].copy()
    out["stack_pred"] = stack_test.astype(np.float32)
    out["market_pred"] = market_test.astype(np.float32)
    out["pred"] = (out["stack_pred"].to_numpy(np.float32) + float(selected["gamma"]) * market_test).astype(np.float32)
    out = add_cross_sectional_norms(out, "pred")
    out.to_parquet(f"{OUT_PREFIX}_selected.parquet", index=False)

    row = summarize_overlay(
        out,
        "market_overlay_view_stack_2019q4",
        float(selected["val_ic_2019q4"]),
        float(selected["market_val_ic_2019q4"]),
        float(selected["alpha"]),
        float(selected["gamma"]),
    )
    pd.DataFrame([row]).to_csv(f"{OUT_PREFIX}_summary.csv", index=False)
    monthly = out.assign(month=out["datetime"].dt.to_period("M").astype(str)).groupby("month").apply(
        lambda g: pd.Series(
            {
                "pred_ic": compute_ic(g["pred"], g["label"]),
                "pred_xsz_ic": compute_ic(g["pred_xsz"], g["label"]),
                "pred_xrank_ic": compute_ic(g["pred_xrank"], g["label"]),
                "market_ic": compute_ic(g["market_pred"], g["label"]),
                "stack_ic": compute_ic(g["stack_pred"], g["label"]),
            }
        ),
        include_groups=False,
    )
    monthly.reset_index().to_csv(f"{OUT_PREFIX}_monthly_ic.csv", index=False)
    (OUT_DIR / "market_overlay_model.json").write_text(
        json.dumps(
            {
                "stack_spec": asdict(spec),
                "component_cols": len(component_cols),
                "selected": selected,
                "ridge_columns": len(final_model["columns"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame([row]).to_string(index=False), flush=True)
    print(monthly.to_string(), flush=True)


if __name__ == "__main__":
    main()
