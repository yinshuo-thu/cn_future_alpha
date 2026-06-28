#!/usr/bin/env python3
"""Rolling train-before-test refit for expanded history gate configs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

from expanded_history_gate_clean import (  # noqa: E402
    OUT_DIR,
    TEST_END,
    TEST_START,
    TRAIN_START,
    configs,
    finalize_matrix,
    fit_weights,
    predict_frame,
    summarize,
)


def month_starts() -> list[pd.Timestamp]:
    return list(pd.date_range(TEST_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS"))


def rolling_predict(base: pd.DataFrame, x: np.ndarray, cols: list[int], cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    dt = base["datetime"]
    pieces = []
    rows = []
    for ms in month_starts():
        train_mask = ((dt >= TRAIN_START) & (dt < ms) & base["label"].notna()).to_numpy()
        test_mask = ((dt >= ms) & (dt < ms + pd.DateOffset(months=1))).to_numpy()
        w, train_ic = fit_weights(base, x, cols, train_mask, cfg)
        pred = predict_frame(base, x, cols, w, test_mask)
        rows.append(
            {
                "model": cfg.name,
                "month": f"{ms:%Y-%m}",
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_ic": float(train_ic),
                "month_ic": float(pred["pred"].corr(pred["label"])),
                **{f"w_{i}": float(v) for i, v in zip(cols, w)},
            }
        )
        pieces.append(pred)
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(rows)


def main() -> None:
    base, x, names, families = finalize_matrix()
    name_to_idx = {n: i for i, n in enumerate(names)}
    cfg_by_name = {cfg.name: cfg for cfg in configs(names, families)}

    grid_path = OUT_DIR / "validation_grid.csv"
    if not grid_path.exists():
        raise FileNotFoundError(grid_path)
    grid = pd.read_csv(grid_path)
    grid = grid[grid["signed"] == False].sort_values("val_ic_2019q4", ascending=False)  # noqa: E712
    selected = list(dict.fromkeys(grid.head(8)["model"].astype(str).tolist()))
    for must in [
        "old_family_selected_raw_month_equal_u090",
        "old_family_selected_raw_row_u090",
        "old_family_selected_xsz_month_equal_u090",
        "old_family_selected_xsz_row_u090",
    ]:
        if must in cfg_by_name and must not in selected:
            selected.append(must)

    summaries = []
    weight_parts = []
    for pos, name in enumerate(selected, start=1):
        cfg = cfg_by_name[name]
        cols = [name_to_idx[c] for c in cfg.components if c in name_to_idx]
        pred, weights = rolling_predict(base, x, cols, cfg)
        row = summarize(pred, f"{name}__rolling_refit")
        val_ic = float(grid.loc[grid["model"] == name, "val_ic_2019q4"].iloc[0]) if name in set(grid["model"]) else np.nan
        row.update(
            {
                "base_config": name,
                "val_ic_2019q4": val_ic,
                "k": len(cols),
                "target": cfg.target,
                "scheme": cfg.scheme,
                "signed": cfg.signed,
                "upper": cfg.upper,
                "gate_mode": "rolling_train_before_test",
            }
        )
        summaries.append(row)
        weight_parts.append(weights)
        print(
            f"[rolling-refit {pos:02d}/{len(selected):02d}] {name} "
            f"val={val_ic:.6f} test={row['pred_ic_2020']:.6f}",
            flush=True,
        )

    summary = pd.DataFrame(summaries).sort_values("val_ic_2019q4", ascending=False)
    summary["selected_by_2019q4_nonneg"] = summary["base_config"] == summary.iloc[0]["base_config"]
    summary.to_csv(OUT_DIR / "rolling_refit_summary.csv", index=False)
    pd.concat(weight_parts, ignore_index=True).to_csv(OUT_DIR / "rolling_refit_weights.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
