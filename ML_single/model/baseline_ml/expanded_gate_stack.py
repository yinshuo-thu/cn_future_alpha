#!/usr/bin/env python3
"""Second-level 2019Q4 stack over nonnegative expanded gate configs."""

from __future__ import annotations

import sys

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


def main() -> None:
    base, x, names, families = finalize_matrix()
    name_to_idx = {n: i for i, n in enumerate(names)}
    cfg_by_name = {cfg.name: cfg for cfg in configs(names, families)}
    grid = pd.read_csv(OUT_DIR / "validation_grid.csv")
    grid = grid[grid["signed"] == False].sort_values("val_ic_2019q4", ascending=False)  # noqa: E712
    selected = list(dict.fromkeys(grid.head(12)["model"].astype(str).tolist()))
    for must in [
        "old_family_selected_raw_month_equal_u090",
        "old_family_selected_raw_row_u090",
        "old_family_selected_xsz_month_equal_u090",
        "old_family_selected_xsz_row_u090",
    ]:
        if must in cfg_by_name and must not in selected:
            selected.append(must)

    dt = base["datetime"]
    train_mask = mask_between(dt, TRAIN_START, VAL_START, base["label"])
    val_mask = mask_between(dt, VAL_START, TEST_START, base["label"])
    final_train_mask = mask_between(dt, TRAIN_START, TEST_START, base["label"])
    test_mask = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()

    val_base = base.loc[val_mask, ["symbol", "datetime", "label"]].copy().reset_index(drop=True)
    test_base = base.loc[test_mask, ["symbol", "datetime", "label"]].copy().reset_index(drop=True)
    val_cols = []
    test_cols = []
    kept = []
    rows = []
    for name in selected:
        cfg = cfg_by_name[name]
        cols = [name_to_idx[c] for c in cfg.components if c in name_to_idx]
        w_val, train_ic = fit_weights(base, x, cols, train_mask, cfg)
        val_pred = predict_frame(base, x, cols, w_val, val_mask).reset_index(drop=True)
        w_final, final_ic = fit_weights(base, x, cols, final_train_mask, cfg)
        test_pred = predict_frame(base, x, cols, w_final, test_mask).reset_index(drop=True)
        val_cols.append(val_pred["pred"].to_numpy(np.float32))
        test_cols.append(test_pred["pred"].to_numpy(np.float32))
        kept.append(name)
        rows.append(
            {
                "config": name,
                "base_val_ic": float(grid.loc[grid["model"] == name, "val_ic_2019q4"].iloc[0]) if name in set(grid["model"]) else np.nan,
                "train_ic_q1q3": float(train_ic),
                "final_train_ic_2019": float(final_ic),
            }
        )
        print(f"[stack-load] {name}", flush=True)

    xv = np.column_stack(val_cols).astype(np.float64)
    yv = val_base["label"].to_numpy(np.float64)
    mask = np.isfinite(yv)
    xv = np.nan_to_num(xv[mask], nan=0.0, posinf=0.0, neginf=0.0)
    yv = yv[mask]
    cov = xv.T @ yv
    gram = xv.T @ xv
    yty = float(yv @ yv)
    lower = np.zeros(len(kept), dtype=np.float64)
    upper = np.full(len(kept), 1.0, dtype=np.float64)
    w, stack_val_ic = fit_ic_weights_from_stats(cov, gram, yty, lower, upper)

    xt = np.column_stack(test_cols).astype(np.float32)
    pred = np.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0) @ w.astype(np.float32)
    out = test_base.copy()
    out["pred"] = pred.astype(np.float32)
    out = add_cross_sectional_norms(out, "pred")
    summary = summarize(out, "expanded_gate_stack_2019q4_nonneg")
    summary.update({"stack_val_ic_2019q4": float(stack_val_ic), "k": len(kept), "gate_mode": "stack_fit_on_2019q4"})
    pd.DataFrame([summary]).to_csv(OUT_DIR / "stack_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT_DIR / "stack_base_configs.csv", index=False)
    pd.DataFrame({"config": kept, "weight": w}).to_csv(OUT_DIR / "stack_weights.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)
    print(pd.DataFrame({"config": kept, "weight": w}).query("abs(weight) > 1e-8").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
