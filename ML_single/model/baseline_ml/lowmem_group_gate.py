#!/usr/bin/env python3
"""2019-selected group-specific static gate over strict components."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")
sys.path.insert(0, "/root/feature_model")

import numpy as np
import pandas as pd

from lowmem_static_gate import WORK_DIR, collect_specs, fit_static, read_component, scrub
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats
from src.plan_a.group_lgb import symbol_group_map
from strict_optimization_ablation import OUT_DIR, PRED_START, TEST_END, TEST_START, summarize


FOLDS = [
    (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-07-01"), pd.Timestamp("2019-09-01")),
    (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-09-01"), pd.Timestamp("2019-11-01")),
    (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-11-01"), pd.Timestamp("2020-01-01")),
]
ALPHAS = [0.0, 0.15, 0.30, 0.50, 0.75, 1.0]


def ensure_memmap() -> tuple[pd.DataFrame, list[str], np.memmap]:
    specs = collect_specs()
    first = read_component(specs[0])
    n = len(first)
    names = [s.name for s in specs]
    names_path = OUT_DIR / "lowmem_group_gate_components.json"
    mat_path = WORK_DIR / "group_components.float32.memmap"
    expected_bytes = n * len(names) * np.dtype(np.float32).itemsize
    reuse = (
        mat_path.exists()
        and names_path.exists()
        and mat_path.stat().st_size == expected_bytes
        and json.loads(names_path.read_text(encoding="utf-8")) == names
    )
    if not reuse:
        print("[group-gate] rebuilding component memmap", flush=True)
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
            print(f"[group-gate][component] {j + 1:02d}/{len(names)} {spec.name}", flush=True)
            if j != 0:
                del df
        xw.flush()
        names_path.write_text(json.dumps(names, indent=2), encoding="utf-8")
    base = first[["symbol", "datetime", "label"]].copy()
    x = np.memmap(mat_path, mode="r", dtype=np.float32, shape=(n, len(names)))
    return base, names, x


def fit_weights(x: np.ndarray, y: np.ndarray, mask: np.ndarray, signed: bool) -> tuple[np.ndarray, float]:
    return fit_static(x, y, mask, signed=signed)


def fit_group_weights(
    x: np.ndarray,
    y: np.ndarray,
    base: pd.DataFrame,
    groups: np.ndarray,
    group_names: list[str],
    train_mask: np.ndarray,
    signed: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    global_w, global_ic = fit_weights(x, y, train_mask, signed=signed)
    out: dict[str, np.ndarray] = {}
    for g in group_names:
        m = train_mask & (groups == g)
        if int(m.sum()) < 80_000:
            out[g] = global_w
            continue
        w, ic = fit_weights(x, y, m, signed=signed)
        out[g] = w
        print(f"[group-gate][fit] signed={signed} group={g} rows={int(m.sum())} train_ic={ic:.6f}", flush=True)
    return global_w, out, global_ic


def predict_group(
    x: np.ndarray,
    mask: np.ndarray,
    groups: np.ndarray,
    group_names: list[str],
    global_w: np.ndarray,
    group_w: dict[str, np.ndarray],
    alpha: float,
) -> np.ndarray:
    idx_all = np.flatnonzero(mask)
    pred = np.empty(len(idx_all), dtype=np.float32)
    for g in group_names:
        loc = np.flatnonzero(mask & (groups == g))
        if len(loc) == 0:
            continue
        w = (1.0 - alpha) * global_w + alpha * group_w[g]
        pos = np.searchsorted(idx_all, loc)
        pred[pos] = scrub(x[loc]) @ w.astype(np.float32)
    return pred


def main() -> None:
    base, names, x = ensure_memmap()
    y = base["label"].to_numpy(np.float64)
    dt = base["datetime"]
    gmap = symbol_group_map()
    groups = base["symbol"].map(gmap).fillna("other").astype(str).to_numpy()
    group_names = sorted(pd.unique(groups))
    rows = []
    for signed in [False, True]:
        alpha_scores = {a: [] for a in ALPHAS}
        train_scores = []
        for train_start, val_start, val_end in FOLDS:
            train_mask = ((dt >= train_start) & (dt < val_start) & base["label"].notna()).to_numpy()
            val_mask = ((dt >= val_start) & (dt < val_end) & base["label"].notna()).to_numpy()
            global_w, group_w, train_ic = fit_group_weights(x, y, base, groups, group_names, train_mask, signed=signed)
            train_scores.append(train_ic)
            for alpha in ALPHAS:
                pred = predict_group(x, val_mask, groups, group_names, global_w, group_w, alpha)
                alpha_scores[alpha].append(compute_ic(pred, y[val_mask]))
        alpha_mean = {a: float(np.nanmean(v)) for a, v in alpha_scores.items()}
        best_alpha = max(alpha_mean, key=alpha_mean.get)
        print(f"[group-gate] signed={signed} alpha_scores={alpha_mean} best_alpha={best_alpha}", flush=True)

        train_mask = ((dt >= PRED_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
        global_w, group_w, train_ic = fit_group_weights(x, y, base, groups, group_names, train_mask, signed=signed)
        pred = predict_group(x, np.ones(len(base), dtype=bool), groups, group_names, global_w, group_w, float(best_alpha))
        out = base.copy()
        tag = "lowmem_group_gate_signed" if signed else "lowmem_group_gate_nonneg"
        out["pred"] = pred.astype(np.float32)
        out = add_cross_sectional_norms(out, "pred")
        out.to_parquet(OUT_DIR / f"{tag}.parquet", index=False)
        rows.append(
            summarize(out, tag)
            | {
                "gate_train_ic_2019": train_ic,
                "best_alpha_2019cv": float(best_alpha),
                "cv_ic_mean_2019": alpha_mean[best_alpha],
                "cv_train_ic_mean_2019": float(np.nanmean(train_scores)),
            }
        )
        wrows = []
        for g in group_names:
            w = (1.0 - float(best_alpha)) * global_w + float(best_alpha) * group_w[g]
            for name, weight in zip(names, w):
                wrows.append({"group": g, "component": name, "weight": float(weight)})
        pd.DataFrame(wrows).to_csv(OUT_DIR / f"{tag}_weights.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "lowmem_group_gate_summary.csv", index=False)
    print(summary[["model", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020", "best_alpha_2019cv", "cv_ic_mean_2019"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
