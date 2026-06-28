#!/usr/bin/env python3
"""2019Q4 view-stack over fixed and rolling expanded gates.

All first-level predictions are generated train-before-test:
  - fixed gates use 2019Q1-Q3 for 2019Q4 validation predictions and all 2019
    for 2020 test predictions;
  - rolling gates fit only on rows before each predicted month.

The second-level stack is fit only on 2019Q4 labels and then applied to 2020.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

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
from rolling_factor_model_eval import add_cross_sectional_norms, fit_ic_weights_from_stats


OUT_PREFIX = OUT_DIR / "view_stack"


@dataclass(frozen=True)
class StackSpec:
    name: str
    modes: tuple[str, ...]
    views: tuple[str, ...]
    top_n: int
    target: str
    standardize: bool
    upper: float


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS"))


def rolling_predict_period(base: pd.DataFrame, x: np.ndarray, cols: list[int], cfg, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dt = base["datetime"]
    pieces = []
    for ms in month_starts(start, end):
        train_mask = ((dt >= TRAIN_START) & (dt < ms) & base["label"].notna()).to_numpy()
        pred_mask = ((dt >= ms) & (dt < ms + pd.DateOffset(months=1))).to_numpy()
        w, train_ic = fit_weights(base, x, cols, train_mask, cfg)
        pred = predict_frame(base, x, cols, w, pred_mask)
        print(f"[view-stack][rolling] {cfg.name} {ms:%Y-%m} train_ic={train_ic:.6f}", flush=True)
        pieces.append(pred)
    return pd.concat(pieces, ignore_index=True)


def selected_config_names() -> list[str]:
    grid = pd.read_csv(OUT_DIR / "validation_grid.csv")
    grid = grid[grid["signed"] == False].sort_values("val_ic_2019q4", ascending=False)  # noqa: E712
    selected = list(dict.fromkeys(grid.head(10)["model"].astype(str).tolist()))
    for must in [
        "old_family_selected_raw_month_equal_u090",
        "old_family_selected_raw_row_u090",
        "old_family_selected_xsz_month_equal_u090",
        "old_family_selected_xsz_row_u090",
        "old_old9_family_selected_xsz_month_equal_u090",
        "old_old9_family_selected_xsz_row_u090",
    ]:
        if must not in selected:
            selected.append(must)
    return selected


def add_design_columns(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    names: list[str],
    mode: str,
    cfg_name: str,
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
) -> None:
    for view in ["pred", "pred_xsz", "pred_xrank"]:
        name = f"{mode}::{cfg_name}::{view}"
        arrays[name] = (
            val_pred[view].to_numpy(np.float32, copy=False),
            test_pred[view].to_numpy(np.float32, copy=False),
        )
        names.append(name)


def build_base_predictions() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    base, x, names, families = finalize_matrix()
    name_to_idx = {n: i for i, n in enumerate(names)}
    cfg_by_name = {cfg.name: cfg for cfg in configs(names, families)}

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
    for pos, cfg_name in enumerate(selected_config_names(), start=1):
        cfg = cfg_by_name[cfg_name]
        cols = [name_to_idx[c] for c in cfg.components if c in name_to_idx]
        if not cols:
            continue

        w_val, train_ic = fit_weights(base, x, cols, train_mask, cfg)
        val_fixed = predict_frame(base, x, cols, w_val, val_mask).reset_index(drop=True)
        w_final, final_train_ic = fit_weights(base, x, cols, full_train_mask, cfg)
        test_fixed = predict_frame(base, x, cols, w_final, test_mask).reset_index(drop=True)
        add_design_columns(arrays, feature_names, "fixed", cfg_name, val_fixed, test_fixed)

        val_roll = rolling_predict_period(base, x, cols, cfg, VAL_START, TEST_START).reset_index(drop=True)
        test_roll = rolling_predict_period(base, x, cols, cfg, TEST_START, TEST_END).reset_index(drop=True)
        add_design_columns(arrays, feature_names, "rolling", cfg_name, val_roll, test_roll)

        rows.append(
            {
                "rank": pos,
                "config": cfg_name,
                "k": len(cols),
                "fixed_train_ic_q1q3": float(train_ic),
                "fixed_train_ic_2019": float(final_train_ic),
            }
        )
        print(f"[view-stack][base {pos:02d}] {cfg_name}", flush=True)

    pd.DataFrame(rows).to_csv(f"{OUT_PREFIX}_base_configs.csv", index=False)
    pd.DataFrame({"feature": feature_names}).to_csv(f"{OUT_PREFIX}_features.csv", index=False)
    return val_base, test_base, arrays, pd.DataFrame(rows)


def stack_specs(n_configs: int) -> list[StackSpec]:
    specs: list[StackSpec] = []
    for modes in [("fixed",), ("rolling",), ("fixed", "rolling")]:
        mode_name = "+".join(modes)
        for views in [("pred",), ("pred_xsz",), ("pred", "pred_xsz"), ("pred", "pred_xsz", "pred_xrank")]:
            view_name = "+".join(views)
            for top_n in [4, 6, 8, n_configs]:
                if top_n > n_configs:
                    continue
                for target in ["raw", "xsz"]:
                    for standardize in [False, True]:
                        for upper in [0.35, 0.60, 1.00]:
                            specs.append(
                                StackSpec(
                                    name=f"{mode_name}__{view_name}__top{top_n}__{target}__std{int(standardize)}__u{upper:g}",
                                    modes=modes,
                                    views=views,
                                    top_n=top_n,
                                    target=target,
                                    standardize=standardize,
                                    upper=upper,
                                )
                            )
    return specs


def choose_columns(all_names: list[str], spec: StackSpec, base_order: list[str]) -> list[int]:
    keep_configs = set(base_order[: spec.top_n])
    cols = []
    for i, name in enumerate(all_names):
        mode, cfg, view = name.split("::", 2)
        if mode in spec.modes and cfg in keep_configs and view in spec.views:
            cols.append(i)
    return cols


def fit_stack(xv: np.ndarray, y: np.ndarray, cols: list[int], spec: StackSpec) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    x = np.nan_to_num(xv[:, cols].astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.zeros(x.shape[1], dtype=np.float64)
    scale = np.ones(x.shape[1], dtype=np.float64)
    if spec.standardize:
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-9)
        x = (x - mean) / scale
    good = np.isfinite(y)
    x = x[good]
    yy = y[good]
    gram = x.T @ x
    cov = x.T @ yy
    yty = float(yy @ yy)
    lower = np.zeros(len(cols), dtype=np.float64)
    upper = np.full(len(cols), spec.upper, dtype=np.float64)
    w, ic = fit_ic_weights_from_stats(cov, gram, yty, lower, upper)
    return w, float(ic), mean, scale


def apply_stack(xt: np.ndarray, cols: list[int], w: np.ndarray, mean: np.ndarray, scale: np.ndarray, standardize: bool) -> np.ndarray:
    x = np.nan_to_num(xt[:, cols].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if standardize:
        x = ((x.astype(np.float64) - mean) / scale).astype(np.float32)
    return x @ w.astype(np.float32)


def main() -> None:
    val_base, test_base, arrays, base_configs = build_base_predictions()
    feature_names = list(arrays)
    xv = np.column_stack([arrays[n][0] for n in feature_names]).astype(np.float32)
    xt = np.column_stack([arrays[n][1] for n in feature_names]).astype(np.float32)
    y_raw = val_base["label"].to_numpy(np.float64, copy=False)
    y_xsz = val_base["label_xsz_fit"].to_numpy(np.float64, copy=False)
    base_order = base_configs["config"].astype(str).tolist()

    rows = []
    weight_rows = []
    pred_rows = []
    for spec in stack_specs(len(base_order)):
        cols = choose_columns(feature_names, spec, base_order)
        if not cols:
            continue
        y = y_xsz if spec.target == "xsz" else y_raw
        w, val_ic, mean, scale = fit_stack(xv, y, cols, spec)
        pred = apply_stack(xt, cols, w, mean, scale, spec.standardize)
        out = test_base[["symbol", "datetime", "label"]].copy()
        out["pred"] = pred.astype(np.float32)
        out = add_cross_sectional_norms(out, "pred")
        row = summarize(out, f"expanded_gate_view_stack__{spec.name}")
        row.update(
            {
                "stack_val_ic_2019q4": val_ic,
                "n_features": len(cols),
                "modes": "+".join(spec.modes),
                "views": "+".join(spec.views),
                "top_n": spec.top_n,
                "target": spec.target,
                "standardize": spec.standardize,
                "upper": spec.upper,
            }
        )
        rows.append(row)
        pred_rows.append((val_ic, row["model"], out))
        for feature, weight in zip([feature_names[i] for i in cols], w):
            if abs(float(weight)) > 1e-8:
                weight_rows.append({"model": row["model"], "feature": feature, "weight": float(weight)})
        print(f"[view-stack][eval] {row['model']} val={val_ic:.6f} test={row['pred_ic_2020']:.6f}", flush=True)

    summary = pd.DataFrame(rows).sort_values("stack_val_ic_2019q4", ascending=False)
    summary["selected_by_2019q4"] = summary["model"] == summary.iloc[0]["model"]
    summary.to_csv(f"{OUT_PREFIX}_summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(f"{OUT_PREFIX}_weights.csv", index=False)
    best_model = str(summary.iloc[0]["model"])
    for _, model, pred in pred_rows:
        if model == best_model:
            pred.to_parquet(f"{OUT_PREFIX}_selected.parquet", index=False)
            pred.assign(month=pred["datetime"].dt.to_period("M").astype(str)).groupby("month").apply(
                lambda g: pd.Series(
                    {
                        "pred_ic": g["pred"].corr(g["label"]),
                        "pred_xsz_ic": g["pred_xsz"].corr(g["label"]),
                        "pred_xrank_ic": g["pred_xrank"].corr(g["label"]),
                    }
                ),
                include_groups=False,
            ).reset_index().to_csv(f"{OUT_PREFIX}_selected_monthly_ic.csv", index=False)
            break
    print(summary.head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
