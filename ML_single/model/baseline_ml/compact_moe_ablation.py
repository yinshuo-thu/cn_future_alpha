#!/usr/bin/env python3
"""Compact clean MOE ablation over strict OOS component predictions.

The goal is to find the smallest component set that keeps validation
performance close to the larger MOE, then evaluate it with a realtime monthly
rolling gate on 2020.  Model selection uses 2019Q4 only; 2020 labels are used
only for final reporting.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lowmem_static_gate import Spec, collect_specs, fit_static, read_component, scrub
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, period_ic
from strict_optimization_ablation import OUT_DIR as STRICT_OUT_DIR
from strict_optimization_ablation import PRED_START, TEST_END, TEST_START, summarize


OUT_DIR = STRICT_OUT_DIR / "compact_moe"
ROLLING_NAMES_PATH = STRICT_OUT_DIR / "lowmem_rolling_components.json"
ROLLING_MAT_PATH = STRICT_OUT_DIR / "lowmem_gate" / "rolling_components.float32.memmap"
VAL_START = pd.Timestamp("2019-10-01")


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS"))


def spec_registry() -> dict[str, Spec]:
    specs = collect_specs()
    return {s.name: s for s in specs}


def load_component_stack() -> tuple[pd.DataFrame, list[str], np.memmap]:
    if not ROLLING_NAMES_PATH.exists() or not ROLLING_MAT_PATH.exists():
        raise FileNotFoundError(
            "missing rolling component memmap; run lowmem_rolling_gate.py once before compact ablation"
        )
    names = json.loads(ROLLING_NAMES_PATH.read_text(encoding="utf-8"))
    registry = spec_registry()
    missing = [n for n in names if n not in registry]
    if missing:
        raise RuntimeError(f"missing component specs for {missing[:5]}")
    first = read_component(registry[names[0]])
    n = len(first)
    expected_bytes = n * len(names) * np.dtype(np.float32).itemsize
    if ROLLING_MAT_PATH.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"rolling memmap size mismatch: got {ROLLING_MAT_PATH.stat().st_size}, expected {expected_bytes}"
        )
    x = np.memmap(ROLLING_MAT_PATH, mode="r", dtype=np.float32, shape=(n, len(names)))
    base = first[["symbol", "datetime", "label"]].copy()
    return base, names, x


def fit_pred(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, pred_mask: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    weights, train_ic = fit_static(x[:, cols], y, train_mask, signed=True)
    pred = scrub(x[pred_mask][:, cols]) @ weights.astype(np.float32)
    return pred.astype(np.float32), weights.astype(np.float64), float(train_ic)


def rolling_predictions(
    base: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    cols: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dt = base["datetime"]
    rows: list[pd.DataFrame] = []
    weight_rows: list[dict[str, object]] = []
    for ms in month_starts(start, end):
        train_mask = ((dt >= PRED_START) & (dt < ms) & base["label"].notna()).to_numpy()
        pred_mask = ((dt >= ms) & (dt < ms + pd.DateOffset(months=1))).to_numpy()
        pred, weights, train_ic = fit_pred(x, y, train_mask, pred_mask, cols)
        part = base.loc[pred_mask, ["symbol", "datetime", "label"]].copy()
        part["pred"] = pred
        rows.append(part)
        row = {
            "month": f"{ms:%Y-%m}",
            "train_rows": int(train_mask.sum()),
            "test_rows": int(pred_mask.sum()),
            "train_ic": train_ic,
            "month_ic": compute_ic(part["pred"].to_numpy(), part["label"].to_numpy()),
        }
        for local_i, w in enumerate(weights):
            row[f"w_{int(cols[local_i])}"] = float(w)
        weight_rows.append(row)
    out = pd.concat(rows, ignore_index=True)
    return out, pd.DataFrame(weight_rows)


def choose_alpha(static_pred: np.ndarray, rolling_pred: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    best_alpha = 0.0
    best_ic = -np.inf
    for alpha in np.linspace(0.0, 1.0, 41):
        pred = static_pred + alpha * (rolling_pred - static_pred)
        ic = compute_ic(pred, label)
        if ic > best_ic:
            best_ic = float(ic)
            best_alpha = float(alpha)
    return best_alpha, best_ic


def component_rank(names: list[str], weights: np.ndarray, x: np.ndarray, y: np.ndarray, train_mask: np.ndarray) -> pd.DataFrame:
    xt = scrub(x[train_mask]).astype(np.float64, copy=False)
    yt = y[train_mask].astype(np.float64, copy=False)
    standalone = []
    for j in range(len(names)):
        standalone.append(compute_ic(xt[:, j], yt))
    rank = pd.DataFrame(
        {
            "component_index": np.arange(len(names), dtype=np.int32),
            "component": names,
            "full_train_weight": weights,
            "abs_weight": np.abs(weights),
            "standalone_train_ic": standalone,
            "standalone_abs_train_ic": np.abs(standalone),
        }
    )
    rank["score"] = rank["abs_weight"] + 0.08 * rank["standalone_abs_train_ic"]
    return rank.sort_values("score", ascending=False).reset_index(drop=True)


def evaluate_k(
    base: pd.DataFrame,
    names: list[str],
    x: np.ndarray,
    y: np.ndarray,
    ranked_idx: np.ndarray,
    k: int,
) -> dict[str, object]:
    cols = ranked_idx[:k]
    dt = base["datetime"]
    train_core = ((dt >= PRED_START) & (dt < VAL_START) & base["label"].notna()).to_numpy()
    val_mask = ((dt >= VAL_START) & (dt < TEST_START)).to_numpy()

    static_val_pred, static_w, static_train_ic = fit_pred(x, y, train_core, val_mask, cols)
    val_roll, val_weights = rolling_predictions(base, x, y, cols, VAL_START, TEST_START)
    val_roll_pred = val_roll["pred"].to_numpy(np.float32)
    val_label = val_roll["label"].to_numpy(np.float64)
    static_ic = compute_ic(static_val_pred, val_label)
    rolling_ic = compute_ic(val_roll_pred, val_label)
    alpha, blend_ic = choose_alpha(static_val_pred, val_roll_pred, val_label)

    return {
        "k": k,
        "components": [names[int(i)] for i in cols],
        "component_indices": [int(i) for i in cols],
        "val_static_ic_2019q4": float(static_ic),
        "val_rolling_ic_2019q4": float(rolling_ic),
        "val_blend_ic_2019q4": float(blend_ic),
        "val_alpha_rolling": float(alpha),
        "static_train_ic_2019q1q3": float(static_train_ic),
        "val_weight_rows": val_weights,
        "static_weights_q1q3": static_w,
    }


def finalize_selected(
    base: pd.DataFrame,
    names: list[str],
    x: np.ndarray,
    y: np.ndarray,
    selected: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = np.array(selected["component_indices"], dtype=np.int32)
    alpha = float(selected["val_alpha_rolling"])
    dt = base["datetime"]
    train_2019 = ((dt >= PRED_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    test_2020 = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()
    static_test_pred, static_w, static_train_ic = fit_pred(x, y, train_2019, test_2020, cols)
    rolling_test, rolling_weights = rolling_predictions(base, x, y, cols, TEST_START, TEST_END)
    pred = static_test_pred + alpha * (rolling_test["pred"].to_numpy(np.float32) - static_test_pred)
    out = rolling_test[["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype(np.float32)
    out = add_cross_sectional_norms(out, "pred")
    component_rows = []
    for local_i, idx in enumerate(cols):
        component_rows.append(
            {
                "rank": local_i + 1,
                "component_index": int(idx),
                "component": names[int(idx)],
                "static_weight_full_2019": float(static_w[local_i]),
            }
        )
    component_df = pd.DataFrame(component_rows)
    meta_df = pd.DataFrame(
        [
            {
                "model": "compact_moe_selected",
                "k": int(selected["k"]),
                "val_static_ic_2019q4": selected["val_static_ic_2019q4"],
                "val_rolling_ic_2019q4": selected["val_rolling_ic_2019q4"],
                "val_blend_ic_2019q4": selected["val_blend_ic_2019q4"],
                "val_alpha_rolling": alpha,
                "static_train_ic_full_2019": static_train_ic,
            }
        ]
    )
    return out, rolling_weights, pd.concat([meta_df, component_df], axis=1)


def plot_monthly(monthly: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(kind="bar", ax=ax, color="#3A7CA5")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Compact MOE 2020 Monthly IC")
    ax.set_xlabel("Month")
    ax.set_ylabel("IC")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base, names, x = load_component_stack()
    y = base["label"].to_numpy(np.float64)
    dt = base["datetime"]
    train_core = ((dt >= PRED_START) & (dt < VAL_START) & base["label"].notna()).to_numpy()
    full_weights, full_train_ic = fit_static(x, y, train_core, signed=True)
    rank = component_rank(names, full_weights, x, y, train_core)
    rank.to_csv(OUT_DIR / "component_rank.csv", index=False)
    ranked_idx = rank["component_index"].to_numpy(np.int32)

    k_grid = [int(x) for x in os.environ.get("COMPACT_MOE_K_GRID", "1,2,3,5,8,12,16,24,32,51").split(",")]
    candidates = []
    for k in k_grid:
        result = evaluate_k(base, names, x, y, ranked_idx, k)
        candidates.append(result)
        print(
            f"[compact-moe][k={k:02d}] val_static={result['val_static_ic_2019q4']:.6f} "
            f"val_roll={result['val_rolling_ic_2019q4']:.6f} "
            f"val_blend={result['val_blend_ic_2019q4']:.6f} alpha={result['val_alpha_rolling']:.3f}",
            flush=True,
        )

    grid_rows = []
    for c in candidates:
        row = {k: v for k, v in c.items() if k not in {"components", "component_indices", "val_weight_rows", "static_weights_q1q3"}}
        row["components"] = "|".join(c["components"])
        grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT_DIR / "validation_grid.csv", index=False)

    best_val = float(grid["val_blend_ic_2019q4"].max())
    tol = float(os.environ.get("COMPACT_MOE_VAL_TOL", "0.0010"))
    eligible = grid[grid["val_blend_ic_2019q4"] >= best_val - tol].sort_values(["k", "val_blend_ic_2019q4"], ascending=[True, False])
    selected_k = int(eligible.iloc[0]["k"])
    selected = next(c for c in candidates if int(c["k"]) == selected_k)
    pred, rolling_weights, selected_components = finalize_selected(base, names, x, y, selected)
    pred.to_parquet(OUT_DIR / "compact_moe_selected.parquet", index=False)
    rolling_weights.to_csv(OUT_DIR / "compact_moe_selected_rolling_weights.csv", index=False)
    selected_components.to_csv(OUT_DIR / "compact_moe_selected_components.csv", index=False)

    summary = summarize(pred, "compact_moe_selected")
    summary.update(
        {
            "k": selected_k,
            "selection_best_val_ic_2019q4": best_val,
            "selection_tol": tol,
            "val_blend_ic_2019q4": selected["val_blend_ic_2019q4"],
            "val_alpha_rolling": selected["val_alpha_rolling"],
            "full_51_train_ic_2019q1q3": full_train_ic,
        }
    )
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    monthly = period_ic(pred, "pred", "M")
    monthly.to_csv(OUT_DIR / "monthly_ic.csv")
    plot_monthly(monthly, OUT_DIR / "monthly_ic.png")
    print(summary_df[["model", "k", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020", "val_blend_ic_2019q4", "val_alpha_rolling"]].to_string(index=False), flush=True)
    print(f"[compact-moe] selected components: {', '.join(selected['components'])}", flush=True)


if __name__ == "__main__":
    main()
