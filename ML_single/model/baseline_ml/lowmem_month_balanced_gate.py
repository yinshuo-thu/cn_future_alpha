#!/usr/bin/env python3
"""Month-balanced static IC gate over strict component predictions.

The standard static gate fits one covariance system over all 2019 rows.  This
variant gives each 2019 month equal weight, which is a small clean robustness
check against high-row months dominating the OOS gate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import numpy as np
import pandas as pd

from lowmem_rolling_gate import ensure_memmap
from lowmem_static_gate import scrub
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic
from strict_optimization_ablation import OUT_DIR, PRED_START, TEST_END, TEST_START, summarize


OUT_SUBDIR = Path(os.environ.get("MONTH_BALANCED_OUT_DIR", str(OUT_DIR / "month_balanced_gate")))
OUT_SUBDIR.mkdir(parents=True, exist_ok=True)


def label_xsz(base: pd.DataFrame) -> np.ndarray:
    g = base.groupby("datetime", sort=False)["label"]
    z = ((base["label"] - g.transform("mean")) / (g.transform("std") + 1e-9)).clip(-8, 8)
    return z.astype(np.float64).to_numpy()


def fit_month_balanced(
    base: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    signed: bool,
    balance: str,
) -> tuple[np.ndarray, float]:
    dt = base["datetime"]
    train_mask = ((dt >= PRED_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    lower = np.full(x.shape[1], -0.12 if signed else 0.0, dtype=np.float64)
    upper = np.full(x.shape[1], 0.85 if signed else 0.75, dtype=np.float64)

    if balance == "row":
        xt = scrub(x[train_mask]).astype(np.float64, copy=False)
        yt = y[train_mask].astype(np.float64, copy=False)
        m = np.isfinite(yt) & np.all(np.isfinite(xt), axis=1)
        return fit_ic_weights_from_stats(xt[m].T @ yt[m], xt[m].T @ xt[m], float(yt[m] @ yt[m]), lower, upper)

    months = pd.period_range(PRED_START, TEST_START - pd.offsets.MonthBegin(1), freq="M")
    c = np.zeros(x.shape[1], dtype=np.float64)
    g = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    yy = 0.0
    used = 0
    for month in months:
        start = month.to_timestamp()
        end = start + pd.DateOffset(months=1)
        mask = ((dt >= start) & (dt < end) & base["label"].notna()).to_numpy()
        if int(mask.sum()) < 10_000:
            continue
        xt = scrub(x[mask]).astype(np.float64, copy=False)
        yt = y[mask].astype(np.float64, copy=False)
        m = np.isfinite(yt) & np.all(np.isfinite(xt), axis=1)
        if int(m.sum()) < 10_000:
            continue
        xt = xt[m]
        yt = yt[m]
        scale = 1.0 / float(len(yt))
        c += (xt.T @ yt) * scale
        g += (xt.T @ xt) * scale
        yy += float(yt @ yt) * scale
        used += 1
    if used == 0:
        raise RuntimeError("no usable 2019 months")
    return fit_ic_weights_from_stats(c / used, g / used, yy / used, lower, upper)


def predict_with_weights(base: pd.DataFrame, x: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    dt = base["datetime"]
    mask = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()
    pred = scrub(x[mask]) @ weights.astype(np.float32)
    out = base.loc[mask, ["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype(np.float32)
    return add_cross_sectional_norms(out, "pred")


def main() -> None:
    base, names, x = ensure_memmap()
    raw_y = base["label"].to_numpy(np.float64)
    xsz_y = label_xsz(base)
    rows: list[dict[str, object]] = []
    for target_name, y in [("raw", raw_y), ("xsz", xsz_y)]:
        for balance in ["row", "month"]:
            for signed in [False, True]:
                tag = f"lowmem_moe_{balance}_{target_name}_{'signed' if signed else 'nonneg'}"
                weights, train_ic = fit_month_balanced(base, x, y, signed=signed, balance=balance)
                pred = predict_with_weights(base, x, weights)
                pred.to_parquet(OUT_SUBDIR / f"{tag}.parquet", index=False)
                pd.DataFrame(
                    [{"model": tag, "gate_train_ic_2019": train_ic} | {f"w_{n}": float(v) for n, v in zip(names, weights)}]
                ).to_csv(OUT_SUBDIR / f"{tag}_weights.csv", index=False)
                row = summarize(pred, tag) | {"gate_train_ic_2019": train_ic}
                rows.append(row)
                monthly = period_ic(pred, "pred", "M")
                monthly.to_csv(OUT_SUBDIR / f"{tag}_monthly_ic.csv")
                print(
                    f"[month-balanced] {tag} train_ic={train_ic:.6f} "
                    f"pred_ic_2020={row['pred_ic_2020']:.6f} monthly_mean={row['pred_monthly_mean_2020']:.6f}",
                    flush=True,
                )
    summary = pd.DataFrame(rows).sort_values("pred_ic_2020", ascending=False)
    summary.to_csv(OUT_SUBDIR / "summary.csv", index=False)
    print(summary[["model", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020", "gate_train_ic_2019"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
