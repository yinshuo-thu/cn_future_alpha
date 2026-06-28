#!/usr/bin/env python3
"""Generate lightweight audit metrics and dashboards for migrated ML models.

The script reads existing prediction/model artifacts from /root/autodl-tmp, but
only writes small CSV/PNG audit files under /root/jump_model. It does not copy
raw data, factor matrices, or large prediction parquet files.
"""

from __future__ import annotations

import gc
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


JUMP = Path("/root/jump_model")
ML_ROOT = Path("/root/autodl-tmp/quant/ML")
SINGLE_METRICS = JUMP / "ML_single" / "metrics"
SINGLE_FIGURES = JUMP / "ML_single" / "figures"
ENSEMBLE_METRICS = JUMP / "ML_ensemble" / "metrics"
ENSEMBLE_FIGURES = JUMP / "ML_ensemble" / "figures"

CURRENT_THREE = ML_ROOT / "agent_runs" / "current_three_model_ensemble_20260629" / "ensemble_current_three.py"
EXPANDED_STACK = ML_ROOT / "expanded_gate_stack.py"
STACK_DIR = ML_ROOT / "strict_opt_results" / "expanded_history_gate_clean"

PRIOR_GROUPS = {
    "precious": ["AU", "AG"],
    "nonferrous": ["CU", "AL", "ZN", "PB", "NI", "SN", "SS"],
    "ferrous": ["RB", "HC", "I", "J", "JM", "ZC", "SF", "SM"],
    "energy_chem": ["BU", "RU", "NR", "TA", "MA", "EG", "PP", "L", "V", "FU", "SC", "FG", "EB", "SA", "UR", "PG"],
    "agri": ["A", "B", "C", "CS", "M", "Y", "P", "OI", "RM", "CF", "CY", "SR", "AP", "CJ", "JD", "RR", "SP", "FB"],
}
SYM2SEC = {s: group for group, symbols in PRIOR_GROUPS.items() for s in symbols}
STRIDE = 30


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def compute_ic(pred: np.ndarray | pd.Series, label: np.ndarray | pd.Series) -> float:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(p) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    p = p[mask]
    y = y[mask]
    den = math.sqrt(float(np.mean(p * p) * np.mean(y * y)))
    if den <= 1e-18:
        return float("nan")
    return float(np.mean(p * y) / den)


def _mean_cs_ic(df: pd.DataFrame, pcol: str, subset_times: pd.Index | None = None) -> tuple[float, int]:
    d = df if subset_times is None else df[df["datetime"].isin(subset_times)]
    d = d[np.isfinite(d[pcol]) & np.isfinite(d["label"])]
    if d.empty:
        return float("nan"), 0
    grouped = d.groupby("datetime", sort=False)
    pbar = grouped[pcol].transform("mean")
    ybar = grouped["label"].transform("mean")
    dp = d[pcol].to_numpy(np.float64, copy=False) - pbar.to_numpy(np.float64, copy=False)
    dy = d["label"].to_numpy(np.float64, copy=False) - ybar.to_numpy(np.float64, copy=False)
    tmp = pd.DataFrame({"dt": d["datetime"].to_numpy(), "xy": dp * dy, "xx": dp * dp, "yy": dy * dy})
    agg = tmp.groupby("dt", sort=False)[["xy", "xx", "yy"]].sum()
    denom = np.sqrt(agg["xx"] * agg["yy"])
    ic = (agg["xy"] / denom).replace([np.inf, -np.inf], np.nan).dropna()
    return float(ic.mean()), int(len(ic))


def four_ics(df: pd.DataFrame, pred_col: str) -> dict[str, float | int]:
    d = df[["symbol", "datetime", "label", pred_col]].copy()
    d = d.rename(columns={pred_col: "pred"})
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d[np.isfinite(d["pred"]) & np.isfinite(d["label"])].copy()
    d["sec"] = d["symbol"].map(SYM2SEC).fillna("other")
    d["pred_sn"] = d["pred"] - d.groupby(["datetime", "sec"], sort=False)["pred"].transform("mean")
    times = pd.Index(np.sort(d["datetime"].unique())[::STRIDE])
    sn, _ = _mean_cs_ic(d, "pred_sn", times)
    raw_no, n_no = _mean_cs_ic(d, "pred", times)
    dense, n_dense = _mean_cs_ic(d, "pred")
    return {
        "pooled_ic": compute_ic(d["pred"], d["label"]),
        "SN_nonoverlap_ic": sn,
        "raw_nonoverlap_ic": raw_no,
        "dense_cs_ic": dense,
        "merged_pearson_ic": float(np.corrcoef(d["pred"].to_numpy(), d["label"].to_numpy())[0, 1]),
        "n_nonoverlap_ts": n_no,
        "n_dense_ts": n_dense,
        "rows": int(len(df)),
        "label_rows": int(np.isfinite(df["label"]).sum()),
    }


def monthly_ic(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    d = df[["datetime", "label", pred_col]].copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d[np.isfinite(d[pred_col]) & np.isfinite(d["label"])]
    d["month"] = d["datetime"].dt.strftime("%Y-%m")
    rows = []
    for month, group in d.groupby("month", sort=True):
        rows.append({"month": month, "ic": compute_ic(group[pred_col], group["label"]), "rows": int(len(group))})
    return pd.DataFrame(rows)


def bin_returns(df: pd.DataFrame, pred_col: str, n_bins: int = 20) -> pd.DataFrame:
    d = df[["datetime", "label", pred_col]].copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d[np.isfinite(d[pred_col]) & np.isfinite(d["label"])].copy()
    rank = d.groupby("datetime", sort=False)[pred_col].rank(pct=True, method="first")
    d["bin"] = np.minimum((rank.to_numpy(np.float64) * n_bins).astype(np.int16), n_bins - 1) + 1
    out = d.groupby("bin", sort=True)["label"].agg(["mean", "count"]).reset_index()
    out = out.rename(columns={"mean": "mean_return", "count": "rows"})
    return out


def plot_dashboard(name: str, monthly: pd.DataFrame, bins: pd.DataFrame, metrics: dict[str, float | int], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), dpi=150)
    fig.suptitle(f"{name} 2020 audit", fontsize=13, fontweight="bold")

    axes[0].plot(monthly["month"], monthly["ic"], marker="o", lw=1.8, color="#22577a")
    axes[0].axhline(0, color="#888888", lw=0.8)
    axes[0].set_title("Monthly pooled IC")
    axes[0].set_ylabel("IC")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(alpha=0.25)

    colors = np.where(bins["mean_return"].to_numpy() >= 0, "#2a9d8f", "#d95d39")
    axes[1].bar(bins["bin"], bins["mean_return"], color=colors, width=0.82)
    axes[1].axhline(0, color="#888888", lw=0.8)
    axes[1].set_title("20-bin mean forward return")
    axes[1].set_xlabel("Prediction rank bin")
    axes[1].set_ylabel("Mean label")
    axes[1].grid(axis="y", alpha=0.25)

    text = f"Pooled IC {metrics['pooled_ic']:.6f}\nSN non-overlap {metrics['SN_nonoverlap_ic']:.6f}"
    axes[1].text(0.02, 0.98, text, transform=axes[1].transAxes, ha="left", va="top", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


def save_assets(df: pd.DataFrame, pred_col: str, name: str, metrics_dir: Path, figures_dir: Path) -> dict[str, float | int | str]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = four_ics(df, pred_col)
    monthly = monthly_ic(df, pred_col)
    bins = bin_returns(df, pred_col)
    safe = name.replace("/", "_")
    monthly.to_csv(metrics_dir / f"{safe}_monthly_ic.csv", index=False)
    bins.to_csv(metrics_dir / f"{safe}_20bin_return.csv", index=False)
    plot_dashboard(name, monthly, bins, metrics, figures_dir / f"{safe}_dashboard.png")
    return {"model": name, **metrics}


def build_mlp_2020(current) -> pd.DataFrame:
    print("[single] building MLP 2020", flush=True)
    fit = current.fit_mlp(current.month_strings("2019-01", "2019-12"))
    df = current.predict_mlp(current.month_strings("2020-01", "2020-12"), fit)
    return df.rename(columns={"mlp": "pred"})


def build_lgb_2020(current) -> pd.DataFrame:
    print("[single] loading LGB 2020", flush=True)
    df = pd.read_parquet(
        current.LGB_2020_PASS,
        columns=["symbol", "datetime", "label", "pred_lgb_recent_weak_selector"],
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.rename(columns={"pred_lgb_recent_weak_selector": "pred"})


def build_ridge_2020(current) -> pd.DataFrame:
    print("[single] building Ridge 2020", flush=True)
    ridge_fit, ridge_apply = current.load_ridge_sources()
    df = current.ridge_frame("2019-01-01", "2020-01-01", "2020-01-01", "2021-01-01", ridge_fit, ridge_apply)
    return df.rename(columns={"ridge": "pred"})


def build_stack_2020() -> pd.DataFrame:
    print("[ensemble] reconstructing expanded clean stack", flush=True)
    stack = import_from_path("migrated_expanded_gate_stack", EXPANDED_STACK)
    base, x, names, families = stack.finalize_matrix()
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    cfg_by_name = {cfg.name: cfg for cfg in stack.configs(names, families)}
    weights = pd.read_csv(STACK_DIR / "stack_weights.csv")
    dt = base["datetime"]
    final_train_mask = stack.mask_between(dt, stack.TRAIN_START, stack.TEST_START, base["label"])
    test_mask = ((dt >= stack.TEST_START) & (dt < stack.TEST_END)).to_numpy()
    test_base = base.loc[test_mask, ["symbol", "datetime", "label"]].copy().reset_index(drop=True)

    cols = []
    used_rows = []
    for _, row in weights.iterrows():
        config = str(row["config"])
        stack_weight = float(row["weight"])
        cfg = cfg_by_name[config]
        comp_cols = [name_to_idx[c] for c in cfg.components if c in name_to_idx]
        fitted, final_ic = stack.fit_weights(base, x, comp_cols, final_train_mask, cfg)
        pred = stack.predict_frame(base, x, comp_cols, fitted, test_mask).reset_index(drop=True)["pred"].to_numpy(np.float32)
        cols.append(pred)
        used_rows.append({"config": config, "stack_weight": stack_weight, "final_train_ic_2019": float(final_ic)})
        print(f"[ensemble] loaded {config} stack_weight={stack_weight:.6g}", flush=True)

    mat = np.column_stack(cols).astype(np.float32)
    w = weights["weight"].to_numpy(np.float32)
    out = test_base
    out["pred"] = (mat @ w).astype(np.float32)
    pd.DataFrame(used_rows).to_csv(ENSEMBLE_METRICS / "expanded_gate_stack_reconstructed_components.csv", index=False)
    return out


def main() -> None:
    for path in [SINGLE_METRICS, SINGLE_FIGURES, ENSEMBLE_METRICS, ENSEMBLE_FIGURES]:
        path.mkdir(parents=True, exist_ok=True)

    current = import_from_path("migrated_current_three_ensemble", CURRENT_THREE)
    single_rows = []
    builders = [
        ("mlp_time120_slope_a025_strong", build_mlp_2020),
        ("lgb_ref_time90_a1_signed_abs12_a08", build_lgb_2020),
        ("ridge_simplex_basic_full2019", build_ridge_2020),
    ]
    for name, builder in builders:
        df = builder(current)
        row = save_assets(df, "pred", name, SINGLE_METRICS, SINGLE_FIGURES)
        single_rows.append(row)
        print(f"[single] {name} pooled={row['pooled_ic']:.6f} sn={row['SN_nonoverlap_ic']:.6f}", flush=True)
        del df
        gc.collect()
    pd.DataFrame(single_rows).to_csv(SINGLE_METRICS / "single_model_audit_metrics.csv", index=False)

    stack_df = build_stack_2020()
    ensemble_row = save_assets(stack_df, "pred", "expanded_gate_stack_2019q4_nonneg", ENSEMBLE_METRICS, ENSEMBLE_FIGURES)
    pd.DataFrame([ensemble_row]).to_csv(ENSEMBLE_METRICS / "best_ensemble_audit_metrics.csv", index=False)
    print(f"[ensemble] pooled={ensemble_row['pooled_ic']:.6f} sn={ensemble_row['SN_nonoverlap_ic']:.6f}", flush=True)


if __name__ == "__main__":
    main()
