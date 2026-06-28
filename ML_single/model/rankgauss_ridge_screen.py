#!/usr/bin/env python3
"""Rolling Ridge with train-distribution rank-gauss feature normalization.

This is a small exploratory runner. For each test month it:
  - loads only sample-cache months strictly before the test month;
  - fits per-feature quantile landmarks on that training sample;
  - maps train/test features through the frozen train CDF -> Gaussian scores;
  - fits a ridge model on label_xsz and evaluates the held-out month.

The feature transform is history-only. Test-month labels are used only for
reporting IC after predictions are written.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri

from factor_count_ablation import CACHE_DIR, SELECTED_PATH, read_month_full
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, period_ic
from strict_optimization_ablation import TEST_END, TRAIN_START, summarize


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS"))


def read_factor_list(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def scrub(x: np.ndarray) -> np.ndarray:
    arr = np.array(x, copy=True)
    return np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def load_train_sample(month: pd.Timestamp, factors: list[str], max_rows: int) -> pd.DataFrame:
    pieces = []
    for tr_ms in month_starts(TRAIN_START, month):
        path = CACHE_DIR / f"{tr_ms:%Y-%m}.parquet"
        if path.exists():
            pieces.append(pd.read_parquet(path, columns=["datetime", "label", "label_xsz"] + factors))
    if not pieces:
        raise RuntimeError(f"no training cache before {month:%Y-%m}")
    train = pd.concat(pieces, ignore_index=True)
    train = train[train["label"].notna()].copy()
    if len(train) > max_rows:
        rng = np.random.default_rng(20260628 + month.year * 12 + month.month)
        idx = rng.choice(len(train), max_rows, replace=False)
        train = train.iloc[np.sort(idx)].reset_index(drop=True)
    return train


def fit_quantiles(x: np.ndarray, n_quantiles: int) -> tuple[np.ndarray, np.ndarray]:
    probs = np.linspace(0.001, 0.999, n_quantiles, dtype=np.float32)
    qs = np.nanquantile(x, probs, axis=0).astype(np.float32)
    return probs, qs


def transform_rankgauss(x: np.ndarray, probs: np.ndarray, qs: np.ndarray) -> np.ndarray:
    x = scrub(x.astype(np.float32, copy=False))
    out = np.empty_like(x, dtype=np.float32)
    # Column-wise search keeps memory bounded and handles duplicate quantiles robustly.
    for j in range(x.shape[1]):
        q = qs[:, j]
        idx = np.searchsorted(q, x[:, j], side="left")
        idx = np.clip(idx, 0, len(probs) - 1)
        out[:, j] = ndtri(probs[idx]).astype(np.float32)
    return np.nan_to_num(out, copy=False, nan=0.0, posinf=3.1, neginf=-3.1)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    y = np.nan_to_num(y.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    x64 = x.astype(np.float64, copy=False)
    y_mean = float(y.mean())
    yc = y - y_mean
    xtx = x64.T @ x64
    xtx.flat[:: xtx.shape[0] + 1] += alpha
    coef = np.linalg.solve(xtx, x64.T @ yc).astype(np.float32)
    return {"coef": coef, "intercept": y_mean}


def predict_ridge(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    return (x @ model["coef"] + float(model["intercept"])).astype(np.float32)  # type: ignore[operator]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rankgauss_ridge_selected_top500")
    ap.add_argument("--out-dir", type=Path, default=Path("/root/autodl-tmp/quant/ML/strict_opt_results/rankgauss_ridge_screen"))
    ap.add_argument("--test-start", default="2019-07")
    ap.add_argument("--test-end", default="2019-12")
    ap.add_argument("--n-factors", type=int, default=500)
    ap.add_argument("--max-train-rows", type=int, default=240_000)
    ap.add_argument("--n-quantiles", type=int, default=1001)
    ap.add_argument("--alpha", type=float, default=80.0)
    args = ap.parse_args()

    factors = read_factor_list(SELECTED_PATH)[: args.n_factors]
    model_dir = args.out_dir / args.name
    parts_dir = model_dir / "month_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    test_months = month_starts(pd.Timestamp(args.test_start), pd.Timestamp(args.test_end) + pd.DateOffset(months=1))

    for ms in test_months:
        part_path = parts_dir / f"{ms:%Y-%m}.parquet"
        if part_path.exists():
            continue
        train = load_train_sample(ms, factors, args.max_train_rows)
        x_train_raw = train[factors].to_numpy(np.float32)
        probs, qs = fit_quantiles(x_train_raw, args.n_quantiles)
        x_train = transform_rankgauss(x_train_raw, probs, qs)
        model = fit_ridge(x_train, train["label_xsz"].to_numpy(np.float32), args.alpha)
        del x_train_raw, x_train

        test = read_month_full(ms, factors)
        preds = np.empty(len(test), dtype=np.float32)
        chunk = 200_000
        for start in range(0, len(test), chunk):
            end = min(len(test), start + chunk)
            x_test = transform_rankgauss(test.iloc[start:end][factors].to_numpy(np.float32), probs, qs)
            preds[start:end] = predict_ridge(model, x_test)
        out = test[["symbol", "datetime", "label"]].copy()
        out["pred"] = preds
        out.to_parquet(part_path, index=False)
        print(
            f"[rankgauss-ridge][{args.name}][{ms:%Y-%m}] train={len(train)} "
            f"test={len(test)} ic={compute_ic(out['pred'], out['label']):.6f}",
            flush=True,
        )
        del train, test, out, preds, probs, qs, model
        gc.collect()

    pred = pd.concat([pd.read_parquet(parts_dir / f"{ms:%Y-%m}.parquet") for ms in test_months], ignore_index=True)
    pred = add_cross_sectional_norms(pred, "pred")
    pred_path = model_dir / f"{args.name}.parquet"
    pred.to_parquet(pred_path, index=False)
    row = summarize(pred, args.name)
    row.update(
        {
            "n_factors": len(factors),
            "max_train_rows": args.max_train_rows,
            "n_quantiles": args.n_quantiles,
            "alpha": args.alpha,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "feature_transform": "train_distribution_rankgauss",
        }
    )
    pd.DataFrame([row]).to_csv(model_dir / "summary.csv", index=False)
    monthly = []
    for month, ic in period_ic(pred, "pred", "M").items():
        monthly.append({"model": args.name, "month": month, "pred_ic": float(ic)})
    pd.DataFrame(monthly).to_csv(model_dir / "monthly_ic.csv", index=False)
    (model_dir / "metadata.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(pd.DataFrame([row]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
