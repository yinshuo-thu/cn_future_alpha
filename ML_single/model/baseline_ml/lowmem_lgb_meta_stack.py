#!/usr/bin/env python3
"""LightGBM meta stacker over strict OOS component predictions.

Selection is historical only:
  - train on 2019-01..2019-09 sampled rows;
  - early-stop on 2019-Q4 sampled rows;
  - refit on all 2019 sampled rows using the selected iteration count;
  - report 2020 once.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")
sys.path.insert(0, "/root/feature_model")

import lightgbm as lgb
import numpy as np
import pandas as pd

from deep_moe_lowmem import add_context, ensure_component_memmap, make_label_xsz, sample_indices
from lowmem_static_gate import scrub
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, period_ic
from strict_optimization_ablation import OUT_DIR as STRICT_OUT_DIR
from strict_optimization_ablation import PRED_START, TEST_END, TEST_START, summarize


OUT_DIR = Path(os.environ.get("LGB_META_OUT_DIR", "/root/autodl-tmp/quant/ML/lgb_meta_stack_results"))
VAL_START = pd.Timestamp("2019-10-01")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_features(
    x: np.ndarray,
    idx: np.ndarray,
    cont: np.ndarray,
    symbol_code: np.ndarray,
    group_code: np.ndarray,
    component_mean: np.ndarray,
    component_std: np.ndarray,
) -> np.ndarray:
    comp = ((scrub(x[idx]) - component_mean) / component_std).clip(-8, 8).astype(np.float32)
    stats = np.column_stack(
        [
            comp.mean(axis=1),
            comp.std(axis=1),
            np.abs(comp).mean(axis=1),
            comp.max(axis=1),
            comp.min(axis=1),
        ]
    ).astype(np.float32)
    codes = np.column_stack([symbol_code[idx], group_code[idx]]).astype(np.float32)
    return np.concatenate([comp, stats, cont[idx], codes], axis=1).astype(np.float32)


def predict_chunks(
    model: lgb.LGBMRegressor,
    x: np.ndarray,
    idx: np.ndarray,
    cont: np.ndarray,
    symbol_code: np.ndarray,
    group_code: np.ndarray,
    component_mean: np.ndarray,
    component_std: np.ndarray,
) -> np.ndarray:
    out = np.empty(len(idx), dtype=np.float32)
    block = int(os.environ.get("LGB_META_PRED_BLOCK", "300000"))
    for start in range(0, len(idx), block):
        sl = idx[start : start + block]
        feat = make_features(x, sl, cont, symbol_code, group_code, component_mean, component_std)
        out[start : start + len(sl)] = model.predict(feat).astype(np.float32)
    return out


def train_model(
    x: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    cont: np.ndarray,
    symbol_code: np.ndarray,
    group_code: np.ndarray,
    component_mean: np.ndarray,
    component_std: np.ndarray,
    n_estimators: int,
) -> tuple[lgb.LGBMRegressor, int, float | None]:
    params = dict(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=float(os.environ.get("LGB_META_LR", "0.035")),
        num_leaves=int(os.environ.get("LGB_META_LEAVES", "31")),
        min_child_samples=int(os.environ.get("LGB_META_MIN_CHILD", "600")),
        subsample=float(os.environ.get("LGB_META_SUBSAMPLE", "0.82")),
        colsample_bytree=float(os.environ.get("LGB_META_COLSAMPLE", "0.72")),
        reg_lambda=float(os.environ.get("LGB_META_L2", "18.0")),
        reg_alpha=float(os.environ.get("LGB_META_L1", "0.0")),
        random_state=int(os.environ.get("LGB_META_SEED", "20260624")),
        n_jobs=int(os.environ.get("LGB_META_N_JOBS", "1")),
        verbosity=-1,
    )
    model = lgb.LGBMRegressor(**params)
    x_train = make_features(x, train_idx, cont, symbol_code, group_code, component_mean, component_std)
    y_train = target[train_idx]
    def eval_ic(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        return "ic", compute_ic(y_pred, y_true), True

    if val_idx is None or len(val_idx) == 0:
        model.fit(x_train, y_train)
        return model, int(n_estimators), None
    x_val = make_features(x, val_idx, cont, symbol_code, group_code, component_mean, component_std)
    y_val = target[val_idx]
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric=eval_ic,
        callbacks=[lgb.early_stopping(int(os.environ.get("LGB_META_EARLY_STOP", "60")), verbose=False)],
    )
    pred_val = model.predict(x_val)
    val_ic = compute_ic(pred_val, y_val)
    best_iter = int(model.best_iteration_ or n_estimators)
    return model, best_iter, val_ic


def main() -> None:
    seed = int(os.environ.get("LGB_META_SEED", "20260624"))
    set_seed(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base, names, x = ensure_component_memmap()
    y = base["label"].to_numpy(np.float64)
    target_mode = os.environ.get("LGB_META_TARGET", "xsz")
    if target_mode == "xsz":
        target = make_label_xsz(base)
    elif target_mode == "raw":
        target = y.astype(np.float32)
    else:
        raise ValueError(target_mode)
    symbol_code, group_code, cont, sym_map, grp_map = add_context(base)
    dt = base["datetime"]
    fit_mask = ((dt >= PRED_START) & (dt < VAL_START) & np.isfinite(target)).to_numpy()
    val_mask = ((dt >= VAL_START) & (dt < TEST_START) & np.isfinite(target)).to_numpy()
    train_full_mask = ((dt >= PRED_START) & (dt < TEST_START) & np.isfinite(target)).to_numpy()
    test_mask = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()
    train_idx = sample_indices(base, fit_mask, int(os.environ.get("LGB_META_TRAIN_ROWS", "1200000")), seed)
    val_idx = sample_indices(base, val_mask, int(os.environ.get("LGB_META_VAL_ROWS", "500000")), seed + 1)
    full_idx = sample_indices(base, train_full_mask, int(os.environ.get("LGB_META_FULL_ROWS", "1500000")), seed + 2)
    comp_train = scrub(x[train_idx])
    component_mean = comp_train.mean(axis=0).astype(np.float32)
    component_std = (comp_train.std(axis=0) + 1e-6).astype(np.float32)

    model, best_iter, val_ic = train_model(
        x,
        target,
        train_idx,
        val_idx,
        cont,
        symbol_code,
        group_code,
        component_mean,
        component_std,
        int(os.environ.get("LGB_META_ESTIMATORS", "900")),
    )
    print(f"[lgb-meta] val_ic_2019q4={val_ic:.6f} best_iter={best_iter}", flush=True)

    comp_full = scrub(x[full_idx])
    component_mean = comp_full.mean(axis=0).astype(np.float32)
    component_std = (comp_full.std(axis=0) + 1e-6).astype(np.float32)
    model, _, _ = train_model(
        x,
        target,
        full_idx,
        None,
        cont,
        symbol_code,
        group_code,
        component_mean,
        component_std,
        max(20, best_iter),
    )
    test_idx = np.flatnonzero(test_mask)
    pred = predict_chunks(model, x, test_idx, cont, symbol_code, group_code, component_mean, component_std)
    out = base.iloc[test_idx][["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype(np.float32)
    out = add_cross_sectional_norms(out, "pred")
    out.to_parquet(OUT_DIR / "lgb_meta_stack.parquet", index=False)
    summary = pd.DataFrame(
        [
            summarize(out, "lgb_meta_stack")
            | {
                "val_ic_2019q4": float(val_ic),
                "best_iter_2019q4": int(best_iter),
                "target_mode": target_mode,
                "components": len(names),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    period_ic(out, "pred", "M").to_csv(OUT_DIR / "monthly_ic.csv")
    meta = {
        "components": names,
        "symbols": len(sym_map),
        "groups": len(grp_map),
        "target_mode": target_mode,
        "best_iter_2019q4": int(best_iter),
        "val_ic_2019q4": float(val_ic),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary[["model", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020", "val_ic_2019q4", "best_iter_2019q4"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
