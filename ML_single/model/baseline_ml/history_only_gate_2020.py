#!/usr/bin/env python3
"""
Train a fixed clean gate on 2018-2019 only and evaluate it on 2020.

This intentionally excludes predictions_best_ic0716 because that artifact
contains full-window post-processing selection.  It is a quick no-2020-label
gate over the readable clean/base component artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rolling_factor_model_eval import (
    OUT,
    TEST_END,
    TEST_START,
    add_cross_sectional_norms,
    build_candidate_panel,
    compute_ic,
    fit_ic_weights_from_stats,
    period_ic,
    summarize_predictions,
)


RESULT_DIR = Path("/root/autodl-tmp/quant/ML/history_gate_results")
TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2020-01-01")


def fit_fixed_gate(data: pd.DataFrame, names: list[str], candidates: list[str], *, signed: bool) -> tuple[pd.DataFrame, dict]:
    active = [c for c in candidates if c in names]
    idx = [names.index(c) for c in active]
    train = data[(data["datetime"] >= TRAIN_START) & (data["datetime"] < TRAIN_END)]
    y = train["label"].to_numpy(np.float64)
    x = train[names].to_numpy(np.float32)[:, idx]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    mask = np.isfinite(y)
    x = x[mask]
    y = y[mask]
    c = x.T @ y
    g = x.T @ x
    yy = float(y @ y)
    lower = np.full(len(active), -0.15 if signed else 0.0, dtype=np.float64)
    upper = np.full(len(active), 1.10 if signed else 0.90, dtype=np.float64)
    w, train_ic = fit_ic_weights_from_stats(c, g, yy, lower, upper)

    test = data[(data["datetime"] >= TEST_START) & (data["datetime"] < TEST_END)].copy()
    xt = test[active].to_numpy(np.float32)
    xt = np.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0)
    pred = xt @ w.astype(np.float32)
    out = test[["symbol", "datetime", "label"]].copy()
    out["pred"] = pred
    out = add_cross_sectional_norms(out, "pred")
    meta = {
        "signed": signed,
        "train_ic": train_ic,
        "test_ic": compute_ic(out["pred"].to_numpy(), out["label"].to_numpy()),
        "active_components": len(active),
        **{f"w_{name}": float(val) for name, val in zip(active, w)},
    }
    return out, meta


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data, names, logs = build_candidate_panel()
    clean_candidates = [
        "core_raw",
        "core_xsz",
        "core_xrank",
        "anchor_raw",
        "anchor_xsz",
        "anchor_xrank",
        "flgb1819_z",
        "flgb2021_z",
        "group650_z",
        "group650_xsz",
        "group650_xrank",
    ]
    rows = []
    monthly = []
    for signed in [False, True]:
        model = "history_gate_clean_signed" if signed else "history_gate_clean_nonneg"
        pred, meta = fit_fixed_gate(data, names, clean_candidates, signed=signed)
        pred.to_parquet(RESULT_DIR / f"{model}_predictions.parquet", index=False)
        row = summarize_predictions(pred, model, TEST_START, TEST_END)
        row.update(meta)
        rows.append(row)
        by_m = period_ic(pred, "pred", "M").rename("month_ic").reset_index().rename(columns={"_period": "month"})
        by_m.insert(0, "model", model)
        monthly.append(by_m)
    summary = pd.DataFrame(rows).sort_values("pred_ic", ascending=False)
    monthly_df = pd.concat(monthly, ignore_index=True)
    summary.to_csv(RESULT_DIR / "summary.csv", index=False)
    monthly_df.to_csv(RESULT_DIR / "monthly_ic.csv", index=False)
    (RESULT_DIR / "candidate_load_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    print(summary[["model", "pred_ic", "pred_monthly_mean", "pred_monthly_ir", "train_ic", "active_components"]].to_string(index=False))
    print(monthly_df.to_string(index=False))
    print(f"wrote {RESULT_DIR}")


if __name__ == "__main__":
    main()
