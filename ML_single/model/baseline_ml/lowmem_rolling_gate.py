#!/usr/bin/env python3
"""Low-memory rolling IC gate over strict component predictions.

Each 2020 test month is predicted by a gate fit only on historical OOS
component predictions available before that month.  This is a clean realtime
variant of the static gate: no 2020 future labels are used for a given month.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import numpy as np
import pandas as pd

from lowmem_static_gate import WORK_DIR, collect_specs, fit_static, read_component, scrub
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic
from strict_optimization_ablation import OUT_DIR, PRED_START, TEST_END, TEST_START, summarize


def month_starts() -> list[pd.Timestamp]:
    return list(pd.date_range(TEST_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS"))


def ensure_memmap() -> tuple[pd.DataFrame, list[str], np.memmap]:
    specs = collect_specs()
    if not specs:
        raise RuntimeError("no component specs")
    first = read_component(specs[0])
    n = len(first)
    names = [s.name for s in specs]
    mat_path = WORK_DIR / "rolling_components.float32.memmap"
    names_path = OUT_DIR / "lowmem_rolling_components.json"
    expected_bytes = n * len(names) * np.dtype(np.float32).itemsize
    reuse = (
        mat_path.exists()
        and names_path.exists()
        and mat_path.stat().st_size == expected_bytes
        and json.loads(names_path.read_text(encoding="utf-8")) == names
    )
    if not reuse:
        print("[rolling-gate] rebuilding component memmap", flush=True)
        ref_symbol = first["symbol"].astype(str).to_numpy()
        ref_dt = first["datetime"].astype("int64").to_numpy()
        xw = np.memmap(mat_path, mode="w+", dtype=np.float32, shape=(n, len(names)))
        for j, spec in enumerate(specs):
            df = first if j == 0 else read_component(spec)
            ok = (
                len(df) == n
                and np.array_equal(df["datetime"].astype("int64").to_numpy(), ref_dt)
                and np.array_equal(df["symbol"].astype(str).to_numpy(), ref_symbol)
            )
            if not ok:
                raise RuntimeError(f"component key mismatch: {spec.name}")
            xw[:, j] = scrub(df[spec.col].to_numpy(np.float32))
            print(f"[rolling-gate][component] {j + 1:02d}/{len(names)} {spec.name}", flush=True)
            if j != 0:
                del df
        xw.flush()
        names_path.write_text(json.dumps(names, indent=2), encoding="utf-8")
    base = first[["symbol", "datetime", "label"]].copy()
    x = np.memmap(mat_path, mode="r", dtype=np.float32, shape=(n, len(names)))
    return base, names, x


def fit_monthly_predictions(
    base: pd.DataFrame,
    names: list[str],
    x: np.ndarray,
    signed: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dt = base["datetime"]
    y = base["label"].to_numpy(np.float64)
    lookback = int(os.environ.get("LOWMEM_ROLLING_LOOKBACK_MONTHS", "0"))
    rows: list[pd.DataFrame] = []
    weight_rows: list[dict[str, object]] = []
    for ms in month_starts():
        train_start = PRED_START if lookback <= 0 else max(PRED_START, ms - pd.DateOffset(months=lookback))
        train_mask = ((dt >= train_start) & (dt < ms) & base["label"].notna()).to_numpy()
        test_mask = ((dt >= ms) & (dt < ms + pd.DateOffset(months=1))).to_numpy()
        weights, train_ic = fit_static(x, y, train_mask, signed=signed)
        pred = scrub(x[test_mask]) @ weights.astype(np.float32)
        part = base.loc[test_mask, ["symbol", "datetime", "label"]].copy()
        part["pred"] = pred.astype(np.float32)
        rows.append(part)
        row = {
            "month": f"{ms:%Y-%m}",
            "train_start": f"{train_start:%Y-%m-%d}",
            "train_end": f"{ms:%Y-%m-%d}",
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_ic": float(train_ic),
            "month_ic": compute_ic(part["pred"].to_numpy(), part["label"].to_numpy()),
        }
        row.update({f"w_{n}": float(w) for n, w in zip(names, weights)})
        weight_rows.append(row)
        print(
            f"[rolling-gate][{ms:%Y-%m}][{'signed' if signed else 'nonneg'}] "
            f"train={train_mask.sum()} test={test_mask.sum()} "
            f"train_ic={train_ic:.6f} month_ic={row['month_ic']:.6f}",
            flush=True,
        )
    out = pd.concat(rows, ignore_index=True)
    out = add_cross_sectional_norms(out, "pred")
    return out, pd.DataFrame(weight_rows)


def main() -> None:
    base, names, x = ensure_memmap()
    rows = []
    for signed in [False, True]:
        tag = "lowmem_moe_rolling_signed" if signed else "lowmem_moe_rolling_nonneg"
        pred, weights = fit_monthly_predictions(base, names, x, signed=signed)
        pred.to_parquet(OUT_DIR / f"{tag}.parquet", index=False)
        weights.to_csv(OUT_DIR / f"{tag}_weights.csv", index=False)
        row = summarize(pred, tag) | {"gate_train_ic_2019": float(weights["train_ic"].iloc[0])}
        rows.append(row)
        print(f"[rolling-gate] {tag} pred_ic_2020={row['pred_ic_2020']:.6f}", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "lowmem_rolling_moe_summary.csv", index=False)
    print(summary[["model", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020", "gate_train_ic_2019"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
