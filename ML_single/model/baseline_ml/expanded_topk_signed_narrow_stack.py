#!/usr/bin/env python3
"""Narrow clean stack over top signed/nonnegative 2019Q4 topK gates.

This is a follow-up to ``expanded_topk_view_stack_clean.py``.  It reuses the
same train-before-test component matrix and the already-computed 2019Q4
validation grid, but focuses only on the best signed and nonnegative first-level
gates.  2020 labels are used only for final reporting/diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import expanded_topk_view_stack_clean as topk  # noqa: E402
import expanded_history_gate_clean as eh  # noqa: E402
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, period_ic  # noqa: E402


ROOT = Path("/root/autodl-tmp/quant/ML")
STRICT_OUT = ROOT / "strict_opt_results"
SOURCE_DIR = STRICT_OUT / "expanded_topk_view_stack_clean"
OUT_DIR = STRICT_OUT / "expanded_topk_signed_narrow_stack"
OUT_PREFIX = OUT_DIR / "view_stack"
BASE_PRED_DIR = OUT_DIR / "base_predictions"


def configure_module_paths() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BASE_PRED_DIR.mkdir(parents=True, exist_ok=True)
    topk.OUT_DIR = OUT_DIR
    topk.OUT_PREFIX = OUT_PREFIX


def select_configs(grid: pd.DataFrame) -> list[str]:
    valid = set(grid["model"].astype(str))
    selected = [
        "old_old9_topk_all_xsz_month_equal_u090",
        "old_old9_topk_all_xsz_month_decay6_u090",
        "old_old9_topk_all_xsz_row_u090",
        "old_topk_val12_xsz_month_equal_u090",
        "old_old9_topk_core_xsz_month_equal_u090",
        "old_family_selected_raw_row_signed05_u090",
        "old_family_selected_raw_month_equal_signed05_u090",
        "old_family_top24_xsz_month_equal_signed05_u090",
        "old_old9_top_val24_xsz_month_equal_signed03_u090",
        "old_old9_topk_val12_xsz_month_equal_signed03_u090",
        "old_family_selected_xsz_month_equal_signed05_u090",
    ]
    out: list[str] = []
    for name in selected:
        if name in valid and name not in out:
            out.append(name)
    return out


def narrow_stack_specs(n_configs: int) -> list[topk.StackSpec]:
    specs: list[topk.StackSpec] = []
    for modes in [("rolling",), ("fixed", "rolling")]:
        mode_name = "+".join(modes)
        for views in [("pred",), ("pred", "pred_xsz")]:
            view_name = "+".join(views)
            for top_n in [3, 4, 5, 6, 8, 10, n_configs]:
                if top_n > n_configs:
                    continue
                for target in ["raw", "xsz"]:
                    for standardize in [False, True]:
                        for upper in [0.15, 0.25, 0.35, 0.60]:
                            specs.append(
                                topk.StackSpec(
                                    name=f"{mode_name}__{view_name}__top{top_n}__{target}__std{int(standardize)}__u{upper:g}",
                                    modes=modes,
                                    views=views,
                                    top_n=top_n,
                                    target=target,
                                    standardize=standardize,
                                    upper=upper,
                                )
                            )
    dedup: dict[str, topk.StackSpec] = {}
    for spec in specs:
        dedup[spec.name] = spec
    return list(dedup.values())


def cached_or_build_predictions(
    base: pd.DataFrame,
    x: np.ndarray,
    names: list[str],
    cfg_by_name: dict[str, eh.GateConfig],
    selected: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    name_to_idx = {n: i for i, n in enumerate(names)}
    dt = base["datetime"]
    train_mask = eh.mask_between(dt, topk.TRAIN_START, topk.VAL_START, base["label"])
    full_train_mask = eh.mask_between(dt, topk.TRAIN_START, topk.TEST_START, base["label"])
    val_mask = ((dt >= topk.VAL_START) & (dt < topk.TEST_START)).to_numpy()
    test_mask = ((dt >= topk.TEST_START) & (dt < topk.TEST_END)).to_numpy()

    val_base = base.loc[val_mask, ["symbol", "datetime", "label", "label_xsz_fit"]].copy().reset_index(drop=True)
    test_base = base.loc[test_mask, ["symbol", "datetime", "label", "label_xsz_fit"]].copy().reset_index(drop=True)

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feature_names: list[str] = []
    rows: list[dict[str, object]] = []

    for pos, cfg_name in enumerate(selected, start=1):
        cfg = cfg_by_name[cfg_name]
        cols = [name_to_idx[c] for c in cfg.components if c in name_to_idx]
        if not cols:
            continue

        safe = cfg_name.replace("/", "_")
        val_fixed_path = BASE_PRED_DIR / f"{safe}__val_fixed.parquet"
        test_fixed_path = BASE_PRED_DIR / f"{safe}__test_fixed.parquet"
        val_roll_path = BASE_PRED_DIR / f"{safe}__val_roll.parquet"
        test_roll_path = BASE_PRED_DIR / f"{safe}__test_roll.parquet"

        if val_fixed_path.exists() and test_fixed_path.exists():
            val_fixed = pd.read_parquet(val_fixed_path)
            test_fixed = pd.read_parquet(test_fixed_path)
            w_val, train_ic = eh.fit_weights(base, x, cols, train_mask, cfg)
            w_final, final_train_ic = eh.fit_weights(base, x, cols, full_train_mask, cfg)
        else:
            w_val, train_ic = eh.fit_weights(base, x, cols, train_mask, cfg)
            val_fixed = eh.predict_frame(base, x, cols, w_val, val_mask).reset_index(drop=True)
            w_final, final_train_ic = eh.fit_weights(base, x, cols, full_train_mask, cfg)
            test_fixed = eh.predict_frame(base, x, cols, w_final, test_mask).reset_index(drop=True)
            val_fixed.to_parquet(val_fixed_path, index=False)
            test_fixed.to_parquet(test_fixed_path, index=False)

        if val_roll_path.exists() and test_roll_path.exists():
            val_roll = pd.read_parquet(val_roll_path)
            test_roll = pd.read_parquet(test_roll_path)
        else:
            val_roll = topk.rolling_predict_period(base, x, cols, cfg, topk.VAL_START, topk.TEST_START).reset_index(drop=True)
            test_roll = topk.rolling_predict_period(base, x, cols, cfg, topk.TEST_START, topk.TEST_END).reset_index(drop=True)
            val_roll.to_parquet(val_roll_path, index=False)
            test_roll.to_parquet(test_roll_path, index=False)

        topk.add_design_columns(arrays, feature_names, "fixed", cfg_name, val_fixed, test_fixed)
        topk.add_design_columns(arrays, feature_names, "rolling", cfg_name, val_roll, test_roll)
        rows.append(
            {
                "rank": pos,
                "config": cfg_name,
                "k": len(cols),
                "fixed_train_ic_q1q3": float(train_ic),
                "fixed_train_ic_2019": float(final_train_ic),
                "val_fixed_ic": compute_ic(val_fixed["pred"], val_fixed["label"]),
                "test_fixed_ic": compute_ic(test_fixed["pred"], test_fixed["label"]),
                "val_roll_ic": compute_ic(val_roll["pred"], val_roll["label"]),
                "test_roll_ic": compute_ic(test_roll["pred"], test_roll["label"]),
            }
        )
        print(f"[signed-narrow][base {pos:02d}/{len(selected):02d}] {cfg_name}", flush=True)

    pd.DataFrame(rows).to_csv(f"{OUT_PREFIX}_base_configs.csv", index=False)
    pd.DataFrame({"feature": feature_names}).to_csv(f"{OUT_PREFIX}_features.csv", index=False)
    return val_base, test_base, arrays, pd.DataFrame(rows)


def write_selected_monthly(pred_path: Path, out_csv: Path) -> None:
    if not pred_path.exists():
        return
    pred = pd.read_parquet(pred_path)
    monthly = pred.assign(month=pred["datetime"].dt.to_period("M").astype(str)).groupby("month").apply(
        lambda g: pd.Series(
            {
                "pred_ic": compute_ic(g["pred"], g["label"]),
                "pred_xsz_ic": compute_ic(g["pred_xsz"], g["label"]),
                "pred_xrank_ic": compute_ic(g["pred_xrank"], g["label"]),
            }
        ),
        include_groups=False,
    )
    monthly.reset_index().to_csv(out_csv, index=False)


def main() -> None:
    configure_module_paths()
    base, x, names, families = topk.finalize_matrix_topk()
    comp_ic_path = SOURCE_DIR / "component_ic_2019q4.csv"
    comp_ic = pd.read_csv(comp_ic_path) if comp_ic_path.exists() else topk.component_ic_table(base, x, names, families)
    all_configs = topk.configs_topk(names, families, comp_ic)
    cfg_by_name = {cfg.name: cfg for cfg in all_configs}

    grid = pd.read_csv(SOURCE_DIR / "validation_grid.csv")
    selected = [name for name in select_configs(grid) if name in cfg_by_name]
    pd.DataFrame({"rank": range(1, len(selected) + 1), "config": selected}).to_csv(
        OUT_DIR / "selected_base_config_names.csv",
        index=False,
    )
    print("[signed-narrow] selected configs:", ", ".join(selected), flush=True)

    val_base, test_base, arrays, base_configs = cached_or_build_predictions(base, x, names, cfg_by_name, selected)

    old_stack_specs = topk.stack_specs
    try:
        topk.stack_specs = narrow_stack_specs
        summary = topk.run_stack(val_base, test_base, arrays, base_configs)
    finally:
        topk.stack_specs = old_stack_specs

    selected_path = Path(f"{OUT_PREFIX}_selected_by_val_raw.parquet")
    write_selected_monthly(selected_path, Path(f"{OUT_PREFIX}_selected_by_val_raw_monthly_ic.csv"))
    best = summary.sort_values("stack_val_raw_ic_2019q4", ascending=False).head(20)
    best.to_csv(f"{OUT_PREFIX}_top20_by_val.csv", index=False)
    diag = summary.sort_values("pred_ic_2020", ascending=False).head(20)
    diag.to_csv(f"{OUT_PREFIX}_top20_by_2020_diagnostic.csv", index=False)
    print(best.to_string(index=False), flush=True)
    print("[signed-narrow] diagnostic top by 2020", flush=True)
    print(diag.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
