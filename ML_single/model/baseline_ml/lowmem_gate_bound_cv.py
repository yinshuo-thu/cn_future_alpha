#!/usr/bin/env python3
"""Clean 2019-Q4 selection of static gate weight bounds.

For each component pool selected by the LOWMEM_GATE_* environment variables:
  1. Fit signed static IC weights on 2019-Q1..Q3 for a small bound grid.
  2. Select bounds by 2019-Q4 IC only.
  3. Refit on full 2019 with the selected bounds.
  4. Report 2020 IC without using 2020 for selection.
"""

from __future__ import annotations

import json
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


VAL_START = pd.Timestamp(os.environ.get("GATE_BOUND_VAL_START", "2019-10-01"))
OUT_SUBDIR = Path(os.environ.get("GATE_BOUND_OUT_DIR", str(OUT_DIR / "gate_bound_cv")))
OUT_SUBDIR.mkdir(parents=True, exist_ok=True)


def parse_grid() -> list[tuple[float, float]]:
    raw = os.environ.get(
        "GATE_BOUND_GRID",
        "-0.08:0.75,-0.12:0.85,-0.16:1.00,-0.20:1.10,-0.25:1.25,-0.30:1.50",
    )
    out: list[tuple[float, float]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        lo, hi = item.split(":")
        out.append((float(lo), float(hi)))
    return out


def fit_weights(x: np.ndarray, y: np.ndarray, mask: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, float]:
    xt = scrub(x[mask]).astype(np.float64, copy=False)
    yt = y[mask].astype(np.float64, copy=False)
    ok = np.isfinite(yt) & np.all(np.isfinite(xt), axis=1)
    xt = xt[ok]
    yt = yt[ok]
    lower = np.full(x.shape[1], lo, dtype=np.float64)
    upper = np.full(x.shape[1], hi, dtype=np.float64)
    return fit_ic_weights_from_stats(xt.T @ yt, xt.T @ xt, float(yt @ yt), lower, upper)


def predict(base: pd.DataFrame, x: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    pred = scrub(x[mask]) @ weights.astype(np.float32)
    out = base.loc[mask, ["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype(np.float32)
    return add_cross_sectional_norms(out, "pred")


def main() -> None:
    base, names, x = ensure_memmap()
    dt = base["datetime"]
    y = base["label"].to_numpy(np.float64)
    fit_mask = ((dt >= PRED_START) & (dt < VAL_START) & base["label"].notna()).to_numpy()
    val_mask = ((dt >= VAL_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    full_train_mask = ((dt >= PRED_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    test_mask = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()

    records: list[dict[str, object]] = []
    for lo, hi in parse_grid():
        weights, train_ic = fit_weights(x, y, fit_mask, lo, hi)
        val_pred = scrub(x[val_mask]) @ weights.astype(np.float32)
        val_ic = compute_ic(val_pred, y[val_mask])
        records.append(
            {
                "lo": lo,
                "hi": hi,
                "fit_train_ic_2019q1q3": train_ic,
                "val_ic_2019q4": val_ic,
                "base_raw_raw_w": float(weights[names.index("base_raw_raw")]) if "base_raw_raw" in names else np.nan,
            }
        )
        print(
            f"[gate-bound-cv] lo={lo:.3f} hi={hi:.3f} train_ic={train_ic:.6f} "
            f"val_ic_2019q4={val_ic:.6f}",
            flush=True,
        )
    grid = pd.DataFrame(records).sort_values(["val_ic_2019q4", "fit_train_ic_2019q1q3"], ascending=False)
    best = grid.iloc[0]
    lo = float(best["lo"])
    hi = float(best["hi"])
    weights, full_train_ic = fit_weights(x, y, full_train_mask, lo, hi)
    out = predict(base, x, weights, test_mask)
    tag = f"gate_bound_cv_signed_lo{abs(lo):.3f}_hi{hi:.3f}".replace(".", "p")
    out.to_parquet(OUT_SUBDIR / f"{tag}.parquet", index=False)
    monthly = period_ic(out, "pred", "M")
    monthly.to_csv(OUT_SUBDIR / f"{tag}_monthly_ic.csv")
    row = summarize(out, tag) | {
        "gate_train_ic_2019": full_train_ic,
        "selected_lo": lo,
        "selected_hi": hi,
        "selected_val_ic_2019q4": float(best["val_ic_2019q4"]),
    }
    pd.DataFrame([row]).to_csv(OUT_SUBDIR / "summary.csv", index=False)
    pd.DataFrame(
        [{"model": tag, "train_ic_2019": full_train_ic, "selected_val_ic_2019q4": float(best["val_ic_2019q4"])}
         | {f"w_{n}": float(v) for n, v in zip(names, weights)}]
    ).to_csv(OUT_SUBDIR / f"{tag}_weights.csv", index=False)
    grid.to_csv(OUT_SUBDIR / "bound_grid_2019q4.csv", index=False)
    (OUT_SUBDIR / "components.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print("[gate-bound-cv] selected", grid.head(3).to_string(index=False), flush=True)
    print(
        pd.DataFrame([row])[
            ["model", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020", "selected_val_ic_2019q4"]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
