#!/usr/bin/env python3
"""Factor-count ablation for the 1144 materialized factors.

This script keeps the protocol uniform across linear and tree models:
  - monthly train-before-test;
  - training samples are drawn only from months before the test month;
  - 2020 is used only for final reporting;
  - factor libraries are written as reusable txt files.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, period_ic
from strict_optimization_ablation import FACTOR_PATH, META_COLS, OUT_DIR as STRICT_OUT_DIR
from strict_optimization_ablation import TEST_END, TEST_START, TRAIN_START, summarize


SELECTED_PATH = Path("/root/quant/work/outputs/selected_factors.txt")
IC2018_PATH = STRICT_OUT_DIR / "selected_factors_2018_ic.txt"
OUT_DIR = STRICT_OUT_DIR / "factor_count"
LIB_DIR = STRICT_OUT_DIR / "factor_library"
CACHE_DIR = OUT_DIR / "month_sample_cache_all1144"


@dataclass(frozen=True)
class FactorLib:
    name: str
    factors: list[str]
    source: str


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def read_factor_list(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS"))


def add_label_xsz(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("datetime", sort=False)["label"]
    df["label_xsz"] = ((df["label"] - g.transform("mean")) / (g.transform("std") + 1e-9)).clip(-8, 8).astype(np.float32)
    return df


def sample_month(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    valid = np.flatnonzero(df["label"].notna().to_numpy())
    if len(valid) <= cap:
        return df.iloc[valid].copy()
    rng = np.random.default_rng(seed)
    ranks = df.iloc[valid].groupby("datetime", sort=False)["label"].rank(pct=True).to_numpy()
    bins = np.floor(np.clip(ranks * 6.0, 0, 5)).astype(np.int16)
    pieces = []
    per = max(1, cap // max(1, len(np.unique(bins))))
    for b in np.unique(bins):
        loc = valid[bins == b]
        take = min(len(loc), per)
        if take:
            pieces.append(rng.choice(loc, take, replace=False))
    picked = np.concatenate(pieces) if pieces else np.array([], dtype=np.int64)
    if len(picked) < cap:
        rest = np.setdiff1d(valid, picked, assume_unique=False)
        if len(rest):
            extra = rng.choice(rest, min(len(rest), cap - len(picked)), replace=False)
            picked = np.concatenate([picked, extra])
    return df.iloc[np.sort(picked[:cap])].copy()


def read_month_full(ms: pd.Timestamp, factors: list[str]) -> pd.DataFrame:
    next_ms = ms + pd.DateOffset(months=1)
    cols = list(dict.fromkeys(["symbol", "datetime", "label"] + factors))
    df = pd.read_parquet(
        FACTOR_PATH,
        columns=cols,
        filters=[("datetime", ">=", ms), ("datetime", "<", next_ms)],
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["datetime", "symbol"], kind="mergesort").reset_index(drop=True)
    return add_label_xsz(df)


def ensure_month_sample_cache(all_factors: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cap = int(os.environ.get("FACTOR_COUNT_MONTH_SAMPLE_ROWS", "10000"))
    for i, ms in enumerate(month_starts(TRAIN_START, TEST_END)):
        path = CACHE_DIR / f"{ms:%Y-%m}.parquet"
        if path.exists():
            continue
        df = read_month_full(ms, all_factors)
        sample = sample_month(df, cap, seed=20260624 + i)
        sample.to_parquet(path, index=False)
        print(f"[factor-cache][{ms:%Y-%m}] rows={len(sample)}", flush=True)
        del df, sample


def load_train_sample(ms: pd.Timestamp, factors: list[str]) -> pd.DataFrame:
    pieces = []
    for tr_ms in month_starts(TRAIN_START, ms):
        path = CACHE_DIR / f"{tr_ms:%Y-%m}.parquet"
        if path.exists():
            pieces.append(pd.read_parquet(path, columns=["datetime", "label", "label_xsz"] + factors))
    if not pieces:
        raise RuntimeError(f"no train samples before {ms:%Y-%m}")
    train = pd.concat(pieces, ignore_index=True)
    train = train[train["label"].notna()].copy()
    cap = int(os.environ.get("FACTOR_COUNT_TRAIN_ROWS", "240000"))
    if len(train) > cap:
        rng = np.random.default_rng(91011 + int(ms.year * 12 + ms.month))
        idx = rng.choice(len(train), cap, replace=False)
        train = train.iloc[np.sort(idx)].reset_index(drop=True)
    return train


def fit_linear(train: pd.DataFrame, factors: list[str]) -> dict[str, np.ndarray | float]:
    x = scrub(train[factors].to_numpy(np.float32)).astype(np.float64, copy=False)
    y = train["label_xsz"].to_numpy(np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    x = (x - mean) / std
    y_mean = float(y.mean())
    yc = y - y_mean
    alpha = float(os.environ.get("FACTOR_COUNT_RIDGE_ALPHA", "80.0"))
    xtx = x.T @ x
    xtx.flat[:: xtx.shape[0] + 1] += alpha
    xty = x.T @ yc
    coef = np.linalg.solve(xtx, xty).astype(np.float32)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "coef": coef, "intercept": y_mean}


def predict_linear(model: dict[str, np.ndarray | float], test: pd.DataFrame, factors: list[str], chunk: int = 250_000) -> np.ndarray:
    out = np.empty(len(test), dtype=np.float32)
    mean = model["mean"]  # type: ignore[assignment]
    std = model["std"]  # type: ignore[assignment]
    coef = model["coef"]  # type: ignore[assignment]
    intercept = float(model["intercept"])
    for start in range(0, len(test), chunk):
        end = min(len(test), start + chunk)
        x = scrub(test.iloc[start:end][factors].to_numpy(np.float32))
        x = (x - mean) / std
        out[start:end] = (x @ coef + intercept).astype(np.float32)
    return out


def fit_tree(train: pd.DataFrame, factors: list[str]) -> lgb.LGBMRegressor:
    params = dict(
        objective="regression",
        n_estimators=int(os.environ.get("FACTOR_COUNT_LGB_ESTIMATORS", "90")),
        learning_rate=float(os.environ.get("FACTOR_COUNT_LGB_LR", "0.055")),
        num_leaves=int(os.environ.get("FACTOR_COUNT_LGB_LEAVES", "31")),
        min_child_samples=int(os.environ.get("FACTOR_COUNT_LGB_MIN_CHILD", "180")),
        subsample=float(os.environ.get("FACTOR_COUNT_LGB_SUBSAMPLE", "0.82")),
        colsample_bytree=float(os.environ.get("FACTOR_COUNT_LGB_COLSAMPLE", "0.72")),
        reg_lambda=float(os.environ.get("FACTOR_COUNT_LGB_L2", "12.0")),
        n_jobs=int(os.environ.get("FACTOR_COUNT_LGB_JOBS", "4")),
        random_state=20260624,
        verbose=-1,
        force_col_wise=True,
    )
    model = lgb.LGBMRegressor(**params)
    x = pd.DataFrame(scrub(train[factors].to_numpy(np.float32)), columns=factors)
    y = train["label_xsz"].to_numpy(np.float32)
    model.fit(x, y)
    return model


def predict_tree(model: lgb.LGBMRegressor, test: pd.DataFrame, factors: list[str]) -> np.ndarray:
    x = pd.DataFrame(scrub(test[factors].to_numpy(np.float32)), columns=factors)
    return model.predict(x).astype(np.float32)


def run_variant(model_type: str, lib: FactorLib) -> pd.DataFrame:
    name = f"{model_type}_{lib.name}"
    parts_dir = OUT_DIR / f"{name}_month_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for ms in month_starts(TEST_START, TEST_END):
        part_path = parts_dir / f"{ms:%Y-%m}.parquet"
        if part_path.exists():
            continue
        train = load_train_sample(ms, lib.factors)
        test = read_month_full(ms, lib.factors)
        if model_type == "linear":
            model = fit_linear(train, lib.factors)
            pred = predict_linear(model, test, lib.factors)
        elif model_type == "tree":
            model = fit_tree(train, lib.factors)
            pred = predict_tree(model, test, lib.factors)
        else:
            raise ValueError(f"unknown model_type={model_type}")
        out = test[["symbol", "datetime", "label"]].copy()
        out["pred"] = pred
        out.to_parquet(part_path, index=False)
        mic = compute_ic(out["pred"].to_numpy(), out["label"].to_numpy())
        print(
            f"[factor-count][{name}][{ms:%Y-%m}] train={len(train)} test={len(test)} ic={mic:.6f}",
            flush=True,
        )
        del train, test, out, model
    pred = pd.concat([pd.read_parquet(parts_dir / f"{ms:%Y-%m}.parquet") for ms in month_starts(TEST_START, TEST_END)], ignore_index=True)
    pred = add_cross_sectional_norms(pred, "pred")
    pred.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
    return pred


def build_decorr_library(all_factors: list[str], ranked: list[str], n: int) -> list[str]:
    out_path = LIB_DIR / f"ic2018_decorr_top{n}.txt"
    if out_path.exists():
        return read_factor_list(out_path)
    rows = []
    max_rows = int(os.environ.get("FACTOR_COUNT_DECORR_ROWS", "90000"))
    for ms in month_starts(TRAIN_START, pd.Timestamp("2019-01-01")):
        path = CACHE_DIR / f"{ms:%Y-%m}.parquet"
        if path.exists():
            rows.append(pd.read_parquet(path, columns=all_factors))
        if sum(len(x) for x in rows) >= max_rows:
            break
    sample = pd.concat(rows, ignore_index=True)
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=20260624)
    ordered = [f for f in ranked if f in sample.columns]
    x = scrub(sample[ordered].to_numpy(np.float32)).astype(np.float64, copy=False)
    x -= x.mean(axis=0)
    x /= x.std(axis=0) + 1e-6
    corr = np.corrcoef(x, rowvar=False)
    selected: list[int] = []
    threshold = float(os.environ.get("FACTOR_COUNT_DECORR_MAX_ABS_CORR", "0.92"))
    for j in range(len(ordered)):
        if not selected or float(np.max(np.abs(corr[j, selected]))) < threshold:
            selected.append(j)
        if len(selected) >= n:
            break
    factors = [ordered[i] for i in selected]
    out_path.write_text("\n".join(factors) + "\n", encoding="utf-8")
    return factors


def make_libraries(requested: set[str] | None = None) -> list[FactorLib]:
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    selected = read_factor_list(SELECTED_PATH)
    ic2018 = read_factor_list(IC2018_PATH) if IC2018_PATH.exists() else selected
    all_factors = selected
    libs: list[FactorLib] = []

    def wants(name: str) -> bool:
        return not requested or name in requested

    base_defs = [
        ("selected_top300", selected[:300], "selected_factors_head"),
        ("selected_top500", selected[:500], "selected_factors_head"),
        ("all1144", all_factors, "selected_factors_all"),
        ("ic2018_top300", ic2018[:300], "2018_abs_ic_head"),
        ("ic2018_top500", ic2018[:500], "2018_abs_ic_head"),
    ]
    for name, factors, source in base_defs:
        if wants(name):
            libs.append(FactorLib(name, factors, source))
    ensure_month_sample_cache(all_factors)
    if wants("ic2018_decorr_top300"):
        libs.append(FactorLib("ic2018_decorr_top300", build_decorr_library(all_factors, ic2018, 300), "2018_abs_ic_decorr"))
    if wants("ic2018_decorr_top500"):
        libs.append(FactorLib("ic2018_decorr_top500", build_decorr_library(all_factors, ic2018, 500), "2018_abs_ic_decorr"))
    for lib in libs:
        path = LIB_DIR / f"{lib.name}.txt"
        path.write_text("\n".join(lib.factors) + "\n", encoding="utf-8")
    manifest = [
        {"name": lib.name, "source": lib.source, "n_factors": len(lib.factors), "path": str(LIB_DIR / f"{lib.name}.txt")}
        for lib in libs
    ]
    (LIB_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return libs


def plot_summary(monthly_rows: pd.DataFrame) -> None:
    if monthly_rows.empty:
        return
    pivot = monthly_rows.pivot(index="month", columns="model", values="ic")
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(ax=ax, linewidth=1.4)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Factor Count Ablation 2020 Monthly IC")
    ax.set_xlabel("Month")
    ax.set_ylabel("IC")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "monthly_ic.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    requested_libs = {x.strip() for x in os.environ.get("FACTOR_COUNT_LIBS", "").split(",") if x.strip()}
    libs = make_libraries(requested_libs or None)
    model_types = [x.strip() for x in os.environ.get("FACTOR_COUNT_MODELS", "linear,tree").split(",") if x.strip()]
    rows = []
    monthly_rows = []
    for model_type in model_types:
        for lib in libs:
            pred = run_variant(model_type, lib)
            model_name = f"{model_type}_{lib.name}"
            row = summarize(pred, model_name)
            row.update({"model_type": model_type, "factor_library": lib.name, "n_factors": len(lib.factors), "library_source": lib.source})
            rows.append(row)
            by_m = period_ic(pred, "pred", "M")
            for month, ic in by_m.items():
                monthly_rows.append({"model": model_name, "month": month, "ic": float(ic)})
            print(
                pd.DataFrame([row])[
                    ["model", "n_factors", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020"]
                ].to_string(index=False),
                flush=True,
            )
    summary = pd.DataFrame(rows).sort_values("pred_ic_2020", ascending=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(OUT_DIR / "monthly_ic.csv", index=False)
    plot_summary(monthly)
    print(summary[["model", "model_type", "factor_library", "n_factors", "pred_ic_2020", "pred_monthly_mean_2020"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
