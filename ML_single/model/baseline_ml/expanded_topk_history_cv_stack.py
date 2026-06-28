#!/usr/bin/env python3
"""History-CV stack for FU topK/old clean rolling gate predictions.

The previous stack variants fit stack weights directly on 2019Q4, then also
selected hyperparameters on 2019Q4.  This script uses an earlier 2019 history
window to fit the stack:

  - first-level gate predictions are rolling, train-before-test;
  - stack train: 2019-04..2019-09;
  - stack validation/selection: 2019-10..2019-12;
  - final stack fit: 2019-04..2019-12;
  - final test: 2020 only.

No 2020 labels are used for feature/config/weight selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import expanded_history_gate_clean as eh  # noqa: E402
import expanded_topk_signed_narrow_stack as narrow  # noqa: E402
import expanded_topk_view_stack_clean as topk  # noqa: E402
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, period_ic  # noqa: E402


ROOT = Path("/root/autodl-tmp/quant/ML")
STRICT_OUT = ROOT / "strict_opt_results"
OUT_DIR = STRICT_OUT / "expanded_topk_history_cv_stack"
SOURCE_CACHE = STRICT_OUT / "expanded_topk_signed_narrow_stack" / "base_predictions"
BASE_PRED_DIR = OUT_DIR / "base_predictions"
OUT_PREFIX = OUT_DIR / "history_cv_stack"

HIST_START = pd.Timestamp("2019-04-01")
VAL_START = pd.Timestamp("2019-10-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")


def configure_paths() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BASE_PRED_DIR.mkdir(parents=True, exist_ok=True)
    topk.OUT_DIR = OUT_DIR
    topk.OUT_PREFIX = OUT_PREFIX


def selected_configs(grid: pd.DataFrame) -> list[str]:
    return narrow.select_configs(grid)


def stack_specs(n_configs: int) -> list[topk.StackSpec]:
    specs: list[topk.StackSpec] = []
    for views in [("pred",), ("pred", "pred_xsz")]:
        view_name = "+".join(views)
        for top_n in [3, 4, 5, 6, 8, 10, n_configs]:
            if top_n > n_configs:
                continue
            for target in ["raw", "xsz"]:
                for standardize in [False, True]:
                    for upper in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.60]:
                        specs.append(
                            topk.StackSpec(
                                name=f"rolling__{view_name}__top{top_n}__{target}__std{int(standardize)}__u{upper:g}",
                                modes=("rolling",),
                                views=views,
                                top_n=top_n,
                                target=target,
                                standardize=standardize,
                                upper=upper,
                            )
                        )
    dedup: dict[str, topk.StackSpec] = {}
    for spec in specs:
        dedup[spec.name] = spec
    return list(dedup.values())


def source_cache_path(cfg_name: str, suffix: str) -> Path:
    return SOURCE_CACHE / f"{cfg_name.replace('/', '_')}__{suffix}.parquet"


def build_or_read_roll(
    base: pd.DataFrame,
    x: np.ndarray,
    names: list[str],
    cfg: eh.GateConfig,
    cfg_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    safe = cfg_name.replace("/", "_")
    hist_path = BASE_PRED_DIR / f"{safe}__hist_roll.parquet"
    test_path = BASE_PRED_DIR / f"{safe}__test_roll.parquet"
    name_to_idx = {n: i for i, n in enumerate(names)}
    cols = [name_to_idx[c] for c in cfg.components if c in name_to_idx]
    if not cols:
        raise ValueError(f"no columns for {cfg_name}")

    if hist_path.exists():
        hist = pd.read_parquet(hist_path)
    else:
        early = topk.rolling_predict_period(base, x, cols, cfg, HIST_START, VAL_START).reset_index(drop=True)
        cached_q4 = source_cache_path(cfg_name, "val_roll")
        if cached_q4.exists():
            q4 = pd.read_parquet(cached_q4)
        else:
            q4 = topk.rolling_predict_period(base, x, cols, cfg, VAL_START, TEST_START).reset_index(drop=True)
        hist = pd.concat([early, q4], ignore_index=True)
        hist.to_parquet(hist_path, index=False)

    if test_path.exists():
        test = pd.read_parquet(test_path)
    else:
        cached_test = source_cache_path(cfg_name, "test_roll")
        if cached_test.exists():
            test = pd.read_parquet(cached_test)
        else:
            test = topk.rolling_predict_period(base, x, cols, cfg, TEST_START, TEST_END).reset_index(drop=True)
        test.to_parquet(test_path, index=False)

    return hist, test


def build_design(
    base: pd.DataFrame,
    x: np.ndarray,
    names: list[str],
    cfg_by_name: dict[str, eh.GateConfig],
    selected: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feature_names: list[str] = []
    hist_base: pd.DataFrame | None = None
    test_base: pd.DataFrame | None = None
    rows: list[dict[str, object]] = []

    for pos, cfg_name in enumerate(selected, start=1):
        cfg = cfg_by_name[cfg_name]
        hist, test = build_or_read_roll(base, x, names, cfg, cfg_name)
        if hist_base is None:
            hist_base = hist[["symbol", "datetime", "label"]].copy().reset_index(drop=True)
            hist_base = eh.add_xsz_label(hist_base)
            test_base = test[["symbol", "datetime", "label"]].copy().reset_index(drop=True)
            test_base = eh.add_xsz_label(test_base)
        else:
            if len(hist) != len(hist_base) or not np.array_equal(hist["datetime"].to_numpy(), hist_base["datetime"].to_numpy()):
                raise ValueError(f"history alignment mismatch: {cfg_name}")
            if test_base is None or len(test) != len(test_base) or not np.array_equal(test["datetime"].to_numpy(), test_base["datetime"].to_numpy()):
                raise ValueError(f"test alignment mismatch: {cfg_name}")

        for view in ["pred", "pred_xsz", "pred_xrank"]:
            feature = f"rolling::{cfg_name}::{view}"
            arrays[feature] = (
                hist[view].to_numpy(np.float32, copy=False),
                test[view].to_numpy(np.float32, copy=False),
            )
            feature_names.append(feature)

        rows.append(
            {
                "rank": pos,
                "config": cfg_name,
                "hist_ic": compute_ic(hist["pred"], hist["label"]),
                "hist_xsz_ic": compute_ic(hist["pred_xsz"], hist["label"]),
                "q4_ic": compute_ic(hist.loc[hist["datetime"] >= VAL_START, "pred"], hist.loc[hist["datetime"] >= VAL_START, "label"]),
                "test_ic": compute_ic(test["pred"], test["label"]),
                "test_xsz_ic": compute_ic(test["pred_xsz"], test["label"]),
            }
        )
        print(f"[history-cv][base {pos:02d}/{len(selected):02d}] {cfg_name}", flush=True)

    assert hist_base is not None and test_base is not None
    pd.DataFrame(rows).to_csv(f"{OUT_PREFIX}_base_configs.csv", index=False)
    pd.DataFrame({"feature": feature_names}).to_csv(f"{OUT_PREFIX}_features.csv", index=False)
    return hist_base, test_base, arrays, pd.DataFrame(rows)


def choose_columns(all_names: list[str], spec: topk.StackSpec, base_order: list[str]) -> list[int]:
    keep_configs = set(base_order[: spec.top_n])
    cols = []
    for i, name in enumerate(all_names):
        mode, cfg, view = name.split("::", 2)
        if mode in spec.modes and cfg in keep_configs and view in spec.views:
            cols.append(i)
    return cols


def fit_apply_final(
    xh: np.ndarray,
    xt: np.ndarray,
    hist_base: pd.DataFrame,
    test_base: pd.DataFrame,
    cols: list[int],
    spec: topk.StackSpec,
) -> tuple[pd.DataFrame, float]:
    y = hist_base["label_xsz_fit"].to_numpy(np.float64) if spec.target == "xsz" else hist_base["label"].to_numpy(np.float64)
    w, fit_ic, mean, scale = topk.fit_stack(xh, y, cols, spec)
    pred = topk.apply_stack(xt, cols, w, mean, scale, spec.standardize)
    out = test_base[["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype(np.float32)
    return add_cross_sectional_norms(out, "pred"), fit_ic


def summarize_pred(pred: pd.DataFrame, model: str) -> dict[str, object]:
    monthly = period_ic(pred, "pred", "M")
    return {
        "model": model,
        "rows": int(len(pred)),
        "label_rows": int(pred["label"].notna().sum()),
        "pred_ic_2020": compute_ic(pred["pred"], pred["label"]),
        "pred_xsz_ic_2020": compute_ic(pred["pred_xsz"], pred["label"]),
        "pred_xrank_ic_2020": compute_ic(pred["pred_xrank"], pred["label"]),
        "monthly_mean": float(monthly.mean()),
        "monthly_ir": float(monthly.mean() / monthly.std()) if monthly.std() > 0 else float("nan"),
    }


def run_stack(
    hist_base: pd.DataFrame,
    test_base: pd.DataFrame,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    base_configs: pd.DataFrame,
) -> pd.DataFrame:
    feature_names = list(arrays)
    xh_all = np.column_stack([arrays[n][0] for n in feature_names]).astype(np.float32)
    xt = np.column_stack([arrays[n][1] for n in feature_names]).astype(np.float32)
    train_mask = (hist_base["datetime"] < VAL_START).to_numpy()
    val_mask = (hist_base["datetime"] >= VAL_START).to_numpy()
    x_train = xh_all[train_mask]
    x_val = xh_all[val_mask]
    y_train_raw = hist_base.loc[train_mask, "label"].to_numpy(np.float64)
    y_train_xsz = hist_base.loc[train_mask, "label_xsz_fit"].to_numpy(np.float64)
    y_val_raw = hist_base.loc[val_mask, "label"].to_numpy(np.float64)
    y_val_xsz = hist_base.loc[val_mask, "label_xsz_fit"].to_numpy(np.float64)
    base_order = base_configs["config"].astype(str).tolist()

    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    best_val: tuple[float, str, pd.DataFrame] | None = None
    best_test: tuple[float, str, pd.DataFrame] | None = None

    for spec in stack_specs(len(base_order)):
        cols = choose_columns(feature_names, spec, base_order)
        if not cols:
            continue
        y_train = y_train_xsz if spec.target == "xsz" else y_train_raw
        w, train_ic, mean, scale = topk.fit_stack(x_train, y_train, cols, spec)
        val_pred = topk.apply_stack(x_val, cols, w, mean, scale, spec.standardize)
        val_raw = compute_ic(val_pred, y_val_raw)
        val_xsz = compute_ic(val_pred, y_val_xsz)
        final_pred, final_fit_ic = fit_apply_final(xh_all, xt, hist_base, test_base, cols, spec)
        row = summarize_pred(final_pred, f"history_cv_stack__{spec.name}")
        row.update(
            {
                "stack_train_ic_2019apr_sep": train_ic,
                "stack_val_raw_ic_2019q4": val_raw,
                "stack_val_xsz_ic_2019q4": val_xsz,
                "stack_final_fit_ic_2019apr_dec": final_fit_ic,
                "n_features": len(cols),
                "views": "+".join(spec.views),
                "top_n": spec.top_n,
                "target": spec.target,
                "standardize": spec.standardize,
                "upper": spec.upper,
            }
        )
        rows.append(row)
        if best_val is None or float(val_raw) > best_val[0]:
            best_val = (float(val_raw), str(row["model"]), final_pred)
        if best_test is None or float(row["pred_ic_2020"]) > best_test[0]:
            best_test = (float(row["pred_ic_2020"]), str(row["model"]), final_pred)
        for feature, weight in zip([feature_names[i] for i in cols], w):
            if abs(float(weight)) > 1e-8:
                weight_rows.append({"model": row["model"], "feature": feature, "weight": float(weight)})
        print(
            f"[history-cv][stack] {row['model']} "
            f"train={train_ic:.6f} val={val_raw:.6f} test={row['pred_ic_2020']:.6f}",
            flush=True,
        )

    summary = pd.DataFrame(rows).sort_values("stack_val_raw_ic_2019q4", ascending=False)
    summary["selected_by_history_val_raw"] = summary["model"] == summary.iloc[0]["model"]
    if best_test is not None:
        summary["diagnostic_best_2020_raw"] = summary["model"] == best_test[1]
    summary.to_csv(f"{OUT_PREFIX}_summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(f"{OUT_PREFIX}_weights.csv", index=False)
    if best_val is not None:
        _, _, pred = best_val
        pred.to_parquet(f"{OUT_PREFIX}_selected_by_val_raw.parquet", index=False)
    if best_test is not None:
        _, _, pred = best_test
        pred.to_parquet(f"{OUT_PREFIX}_diagnostic_best_2020_raw.parquet", index=False)
    summary.sort_values("pred_ic_2020", ascending=False).head(30).to_csv(f"{OUT_PREFIX}_top30_by_2020_diagnostic.csv", index=False)
    summary.head(30).to_csv(f"{OUT_PREFIX}_top30_by_val.csv", index=False)
    return summary


def main() -> None:
    configure_paths()
    base, x, names, families = topk.finalize_matrix_topk()
    grid = pd.read_csv(STRICT_OUT / "expanded_topk_view_stack_clean" / "validation_grid.csv")
    selected = selected_configs(grid)
    all_configs = topk.configs_topk(names, families, pd.read_csv(STRICT_OUT / "expanded_topk_view_stack_clean" / "component_ic_2019q4.csv"))
    cfg_by_name = {cfg.name: cfg for cfg in all_configs}
    selected = [name for name in selected if name in cfg_by_name]
    pd.DataFrame({"rank": range(1, len(selected) + 1), "config": selected}).to_csv(OUT_DIR / "selected_base_config_names.csv", index=False)
    print("[history-cv] selected configs:", ", ".join(selected), flush=True)
    hist_base, test_base, arrays, base_configs = build_design(base, x, names, cfg_by_name, selected)
    summary = run_stack(hist_base, test_base, arrays, base_configs)
    print(summary.head(20).to_string(index=False), flush=True)
    print("[history-cv] diagnostic top", flush=True)
    print(summary.sort_values("pred_ic_2020", ascending=False).head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
