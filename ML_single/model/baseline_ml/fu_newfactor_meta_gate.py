#!/usr/bin/env python3
"""2019-only meta gates over clean new-factor family component predictions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fu_newfactor_family_ensemble import (
    OUT_DIR as FAMILY_OUT_DIR,
    TEST_END,
    TEST_START,
    TRAIN_START,
    VAL_START,
    build_matrix,
    collect_candidates,
    compute_ic,
    corr_matrix,
    period_ic,
    scrub,
    standalone_table,
    stats,
)
from rolling_factor_model_eval import add_cross_sectional_norms


OUT_DIR = FAMILY_OUT_DIR.parent / "meta_gate_clean"
OLD_MINIMAL_RAW_IC = 0.05549757798302793


@dataclass(frozen=True)
class MetaConfig:
    name: str
    top_overall: int
    top_per_family: int
    max_train_rows: int
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    reg_lambda: float
    colsample_bytree: float = 0.82
    subsample: float = 0.86
    seed: int = 20260625


CONFIGS = [
    MetaConfig("meta_lgb_small_top24", 18, 4, 800_000, 180, 0.035, 15, 1200, 18.0),
    MetaConfig("meta_lgb_mid_top36", 24, 5, 1_000_000, 220, 0.030, 31, 900, 14.0),
    MetaConfig("meta_lgb_wide_top48", 32, 6, 1_200_000, 260, 0.026, 31, 700, 12.0, colsample_bytree=0.72),
]


def add_label_xsz(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    g = out.groupby("datetime", sort=False)["label"]
    out["label_xsz"] = ((out["label"] - g.transform("mean")) / (g.transform("std") + 1e-9)).clip(-8, 8).astype(np.float32)
    return out


def select_components(standalone: pd.DataFrame, cfg: MetaConfig) -> list[int]:
    selected = set(standalone.head(cfg.top_overall)["idx"].astype(int).tolist())
    for family in sorted(standalone["family"].unique()):
        selected.update(
            standalone[standalone["family"] == family]
            .sort_values("val_ic_2019q4", ascending=False)
            .head(cfg.top_per_family)["idx"]
            .astype(int)
            .tolist()
        )
    ordered = standalone[standalone["idx"].isin(selected)]["idx"].astype(int).tolist()
    return ordered


def sample_train(mask: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask & np.isfinite(y))
    if len(idx) <= max_rows:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, max_rows, replace=False))


def fit_lgb(x: np.ndarray, y: np.ndarray, idx: np.ndarray, cols: list[int], cfg: MetaConfig) -> lgb.LGBMRegressor:
    xtr = scrub(x[idx][:, cols]).astype(np.float32, copy=False)
    ytr = y[idx].astype(np.float32, copy=False)
    model = lgb.LGBMRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        reg_lambda=cfg.reg_lambda,
        colsample_bytree=cfg.colsample_bytree,
        subsample=cfg.subsample,
        random_state=cfg.seed,
        n_jobs=4,
        verbose=-1,
        force_col_wise=True,
    )
    model.fit(xtr, ytr)
    return model


def predict(model: lgb.LGBMRegressor, x: np.ndarray, mask: np.ndarray, cols: list[int]) -> np.ndarray:
    idx = np.flatnonzero(mask)
    pred = np.empty(len(idx), dtype=np.float32)
    for start in range(0, len(idx), 400_000):
        loc = idx[start : start + 400_000]
        pred[start : start + 400_000] = model.predict(scrub(x[loc][:, cols]).astype(np.float32, copy=False)).astype(np.float32)
    return pred


def summarize(pred: pd.DataFrame, model_name: str, val_ic: float, cols: list[int], names: list[str], families: list[str], cfg: MetaConfig) -> dict[str, object]:
    row: dict[str, object] = {
        "model": model_name,
        "val_ic_2019q4": val_ic,
        "components": len(cols),
        "beats_old_minimal_raw": False,
        **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
    }
    for col in ["pred", "pred_xsz", "pred_xrank"]:
        mic = period_ic(pred, col, "M")
        row[f"{col}_ic_2020"] = compute_ic(pred[col].to_numpy(), pred["label"].to_numpy())
        row[f"{col}_monthly_mean_2020"] = float(mic.mean())
        row[f"{col}_monthly_ir_2020"] = float(mic.mean() / mic.std(ddof=1)) if mic.std(ddof=1) > 0 else float("nan")
    row["beats_old_minimal_raw"] = bool(row["pred_ic_2020"] > OLD_MINIMAL_RAW_IC)
    row["component_list"] = "|".join(names[i] for i in cols)
    row["family_list"] = "|".join(families[i] for i in cols)
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates()
    base, x, names, families, logs = build_matrix(candidates)
    base = add_label_xsz(base)
    (OUT_DIR / "candidate_load_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")

    dt = base["datetime"]
    y_raw = base["label"].to_numpy(np.float64)
    y = base["label_xsz"].to_numpy(np.float32)
    train_mask = ((dt >= TRAIN_START) & (dt < VAL_START) & base["label"].notna()).to_numpy()
    val_mask = ((dt >= VAL_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    all_2019_mask = ((dt >= TRAIN_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    test_mask = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()
    test_label_mask = test_mask & base["label"].notna().to_numpy()

    train_gram, train_cov, train_yty = stats(x, y_raw, train_mask)
    val_gram, val_cov, val_yty = stats(x, y_raw, val_mask)
    test_gram, test_cov, test_yty = stats(x, y_raw, test_label_mask)
    standalone = standalone_table(names, families, train_gram, train_cov, train_yty, val_gram, val_cov, val_yty, test_gram, test_cov, test_yty)
    standalone.to_csv(OUT_DIR / "component_ic.csv", index=False)
    corr = corr_matrix(x, train_mask)
    pd.DataFrame(corr, index=names, columns=names).to_csv(OUT_DIR / "component_corr_2019q1q3.csv")

    val_rows = []
    selected_cache: dict[str, list[int]] = {}
    for cfg in CONFIGS:
        cols = select_components(standalone, cfg)
        selected_cache[cfg.name] = cols
        tr_idx = sample_train(train_mask, y, cfg.max_train_rows, cfg.seed)
        model = fit_lgb(x, y, tr_idx, cols, cfg)
        pred_val = predict(model, x, val_mask, cols)
        val_df = base.loc[val_mask, ["symbol", "datetime", "label"]].copy()
        val_df["pred"] = pred_val
        val_ic = compute_ic(add_cross_sectional_norms(val_df, "pred")["pred_xsz"].to_numpy(), val_df["label"].to_numpy())
        val_rows.append({"config": cfg.name, "val_ic_2019q4": val_ic, "components": len(cols), "component_list": "|".join(names[i] for i in cols)})
        print(f"[meta-val][{cfg.name}] k={len(cols)} val_xsz_ic={val_ic:.6f}", flush=True)
    val_table = pd.DataFrame(val_rows).sort_values("val_ic_2019q4", ascending=False)
    val_table.to_csv(OUT_DIR / "validation_grid.csv", index=False)
    best_name = str(val_table.iloc[0]["config"])
    best_cfg = next(cfg for cfg in CONFIGS if cfg.name == best_name)
    best_cols = selected_cache[best_name]

    tr_idx = sample_train(all_2019_mask, y, best_cfg.max_train_rows, best_cfg.seed + 17)
    model = fit_lgb(x, y, tr_idx, best_cols, best_cfg)
    pred_test = predict(model, x, test_mask, best_cols)
    out = base.loc[test_mask, ["symbol", "datetime", "label"]].copy()
    out["pred"] = pred_test
    out = add_cross_sectional_norms(out, "pred")
    out_path = OUT_DIR / f"{best_name}_2020.parquet"
    out.to_parquet(out_path, index=False)
    period_ic(out, "pred", "M").to_csv(OUT_DIR / f"{best_name}_monthly_ic.csv")
    summary = summarize(out, best_name, float(val_table.iloc[0]["val_ic_2019q4"]), best_cols, names, families, best_cfg)
    pd.DataFrame([summary]).to_csv(OUT_DIR / "summary.csv", index=False)
    (OUT_DIR / "selected_components.json").write_text(json.dumps({"config": best_name, "components": [names[i] for i in best_cols]}, indent=2), encoding="utf-8")
    print(pd.DataFrame([summary])[["model", "val_ic_2019q4", "pred_ic_2020", "pred_xsz_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020", "components", "beats_old_minimal_raw"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
