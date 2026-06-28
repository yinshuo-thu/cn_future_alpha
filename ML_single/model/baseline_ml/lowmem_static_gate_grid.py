#!/usr/bin/env python3
"""2019-only grid search for low-memory static gates.

The grid uses only 2019 OOS component predictions for model selection:
  - blocked 2019 validation folds choose subset/bounds/ridge;
  - final weights are fit on all 2019;
  - 2020 is used only for reporting the chosen and all grid rows.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import numpy as np
import pandas as pd

from lowmem_static_gate import WORK_DIR, collect_specs, fit_static, read_component, scrub
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats
from strict_optimization_ablation import OUT_DIR, PRED_START, TEST_END, TEST_START, summarize


VAL_FOLDS = [
    (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-07-01"), pd.Timestamp("2019-09-01")),
    (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-09-01"), pd.Timestamp("2019-11-01")),
    (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-11-01"), pd.Timestamp("2020-01-01")),
]


@dataclass(frozen=True)
class GridConfig:
    subset: str
    signed: bool
    lower: float
    upper: float
    ridge: float


def ensure_memmap() -> tuple[pd.DataFrame, list[str], np.memmap]:
    specs = collect_specs()
    first = read_component(specs[0])
    n = len(first)
    names = [s.name for s in specs]
    names_path = OUT_DIR / "lowmem_gate_grid_components.json"
    mat_path = WORK_DIR / "grid_components.float32.memmap"
    expected_bytes = n * len(names) * np.dtype(np.float32).itemsize
    reuse = (
        mat_path.exists()
        and names_path.exists()
        and mat_path.stat().st_size == expected_bytes
        and json.loads(names_path.read_text(encoding="utf-8")) == names
    )
    if not reuse:
        print("[gate-grid] rebuilding grid memmap", flush=True)
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
            print(f"[gate-grid][component] {j + 1:02d}/{len(names)} {spec.name}", flush=True)
            if j != 0:
                del df
        xw.flush()
        names_path.write_text(json.dumps(names, indent=2), encoding="utf-8")
    base = first[["symbol", "datetime", "label"]].copy()
    x = np.memmap(mat_path, mode="r", dtype=np.float32, shape=(n, len(names)))
    return base, names, x


def subset_index(names: list[str], subset: str) -> np.ndarray:
    if subset == "all":
        keep = np.ones(len(names), dtype=bool)
    elif subset == "raw_only":
        keep = np.array([n.endswith("_raw") for n in names])
    elif subset == "raw_xsz":
        keep = np.array([n.endswith("_raw") or n.endswith("_xsz") for n in names])
    elif subset == "no_lowcorr":
        keep = np.array(["lowcorr_" not in n for n in names])
    elif subset == "no_xrank":
        keep = np.array([not n.endswith("_xrank") for n in names])
    elif subset == "base_opt_chunk_raw":
        keep = np.array([n.endswith("_raw") and ("lowcorr_" not in n) for n in names])
    else:
        raise ValueError(subset)
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        raise ValueError(f"empty subset {subset}")
    return idx


def fit_from_stats(x: np.ndarray, y: np.ndarray, mask: np.ndarray, idx: np.ndarray, cfg: GridConfig) -> tuple[np.ndarray, float]:
    xt = scrub(x[mask][:, idx]).astype(np.float64, copy=False)
    yt = y[mask].astype(np.float64, copy=False)
    ok = np.isfinite(yt) & np.all(np.isfinite(xt), axis=1)
    xt = xt[ok]
    yt = yt[ok]
    g = xt.T @ xt
    if cfg.ridge > 0:
        g = g + np.eye(len(idx), dtype=np.float64) * (cfg.ridge * float(np.mean(np.diag(g))))
    lower = np.full(len(idx), cfg.lower if cfg.signed else 0.0, dtype=np.float64)
    upper = np.full(len(idx), cfg.upper, dtype=np.float64)
    return fit_ic_weights_from_stats(xt.T @ yt, g, float(yt @ yt), lower, upper)


def predict_subset(x: np.ndarray, idx: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return scrub(x[mask][:, idx]) @ weights.astype(np.float32)


def make_grid() -> list[GridConfig]:
    rows: list[GridConfig] = []
    subsets = ["all", "raw_xsz", "no_xrank", "raw_only", "base_opt_chunk_raw", "no_lowcorr"]
    for subset in subsets:
        rows.extend(
            [
                GridConfig(subset, True, -0.12, 0.85, 0.0),
                GridConfig(subset, True, -0.08, 0.65, 0.0),
                GridConfig(subset, True, -0.05, 0.50, 0.0),
                GridConfig(subset, True, -0.08, 0.65, 0.02),
                GridConfig(subset, True, -0.05, 0.50, 0.05),
                GridConfig(subset, False, 0.0, 0.75, 0.0),
                GridConfig(subset, False, 0.0, 0.55, 0.02),
            ]
        )
    return rows


def main() -> None:
    base, names, x = ensure_memmap()
    y = base["label"].to_numpy(np.float64)
    dt = base["datetime"]
    grid = make_grid()
    rows: list[dict[str, object]] = []
    idx_cache = {subset: subset_index(names, subset) for subset in sorted({g.subset for g in grid})}
    for cfg in grid:
        idx = idx_cache[cfg.subset]
        fold_ics = []
        train_ics = []
        for train_start, val_start, val_end in VAL_FOLDS:
            train_mask = ((dt >= train_start) & (dt < val_start) & base["label"].notna()).to_numpy()
            val_mask = ((dt >= val_start) & (dt < val_end) & base["label"].notna()).to_numpy()
            weights, train_ic = fit_from_stats(x, y, train_mask, idx, cfg)
            pred = predict_subset(x, idx, weights, val_mask)
            fold_ics.append(compute_ic(pred, y[val_mask]))
            train_ics.append(train_ic)
        row = {
            "subset": cfg.subset,
            "signed": cfg.signed,
            "lower": cfg.lower,
            "upper": cfg.upper,
            "ridge": cfg.ridge,
            "n_components": int(len(idx)),
            "cv_ic_mean_2019": float(np.nanmean(fold_ics)),
            "cv_ic_min_2019": float(np.nanmin(fold_ics)),
            "cv_train_ic_mean_2019": float(np.nanmean(train_ics)),
        }
        print(
            f"[gate-grid] {cfg.subset} signed={cfg.signed} lower={cfg.lower} "
            f"upper={cfg.upper} ridge={cfg.ridge} cv={row['cv_ic_mean_2019']:.6f}",
            flush=True,
        )
        rows.append(row)
    grid_df = pd.DataFrame(rows).sort_values(["cv_ic_mean_2019", "cv_ic_min_2019"], ascending=False)
    grid_df.to_csv(OUT_DIR / "lowmem_gate_grid_2019cv.csv", index=False)

    best = grid_df.iloc[0]
    cfg = GridConfig(str(best["subset"]), bool(best["signed"]), float(best["lower"]), float(best["upper"]), float(best["ridge"]))
    idx = idx_cache[cfg.subset]
    train_mask = ((dt >= PRED_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    weights, train_ic = fit_from_stats(x, y, train_mask, idx, cfg)
    pred = scrub(x[:, idx]) @ weights.astype(np.float32)
    out = base.copy()
    out["pred"] = pred.astype(np.float32)
    out = add_cross_sectional_norms(out, "pred")
    tag = "lowmem_moe_static_grid2019"
    out.to_parquet(OUT_DIR / f"{tag}.parquet", index=False)
    summary = pd.DataFrame([summarize(out, tag) | best.to_dict() | {"gate_train_ic_2019": train_ic}])
    summary.to_csv(OUT_DIR / f"{tag}_summary.csv", index=False)
    wdf = pd.DataFrame({"component": np.array(names)[idx], "weight": weights})
    wdf.to_csv(OUT_DIR / f"{tag}_weights.csv", index=False)
    print(summary[["model", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020", "cv_ic_mean_2019", "subset", "signed", "upper", "ridge"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
