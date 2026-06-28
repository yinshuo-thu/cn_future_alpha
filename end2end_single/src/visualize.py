from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import compute_ic, monthly_ic_table


def plot_distributions(df: pd.DataFrame, out_dir: str | Path, prefix: str) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for col in ["pred", "label"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(vals, bins=100, color="#3b82f6" if col == "pred" else "#64748b", alpha=0.75)
        ax.set_title(f"{prefix} {col} distribution")
        ax.grid(alpha=0.2)
        path = out_dir / f"{prefix}_{col}_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_binned_pred_label(df: pd.DataFrame, out_dir: str | Path, prefix: str, bins: int = 30) -> str:
    out_dir = Path(out_dir)
    clean = df.dropna(subset=["pred", "label"]).copy()
    clean["bin"] = pd.qcut(clean["pred"], q=min(bins, max(2, len(clean) // 100)), duplicates="drop")
    grouped = clean.groupby("bin", observed=True).agg(pred=("pred", "mean"), label=("label", "mean"), n=("label", "size"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grouped["pred"], grouped["label"], marker="o", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("mean prediction")
    ax.set_ylabel("mean label")
    ax.set_title(f"{prefix} binned prediction vs label")
    ax.grid(alpha=0.2)
    path = out_dir / f"{prefix}_binned_pred_label.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_monthly_ic(df: pd.DataFrame, out_dir: str | Path, prefix: str) -> str:
    out_dir = Path(out_dir)
    tbl = monthly_ic_table(df)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(tbl["month"], tbl["ic"], color="#0f766e", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    if len(tbl):
        ax.axhline(tbl["ic"].mean(), color="#dc2626", linestyle="--", linewidth=1.0)
    ax.set_title(f"{prefix} monthly IC")
    ax.tick_params(axis="x", rotation=70)
    ax.grid(axis="y", alpha=0.2)
    path = out_dir / f"{prefix}_monthly_ic.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_cumulative_ic(df: pd.DataFrame, out_dir: str | Path, prefix: str) -> str:
    out_dir = Path(out_dir)
    clean = df.dropna(subset=["pred", "label"]).sort_values("datetime").copy()
    chunks = []
    for dt, grp in clean.groupby(pd.to_datetime(clean["datetime"]).dt.to_period("W")):
        chunks.append({"period": str(dt), "ic": compute_ic(grp["pred"], grp["label"])})
    tbl = pd.DataFrame(chunks).dropna()
    tbl["cum_ic"] = tbl["ic"].fillna(0.0).cumsum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tbl["period"], tbl["cum_ic"], color="#7c3aed", linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{prefix} cumulative weekly IC")
    ax.tick_params(axis="x", rotation=70)
    ax.grid(alpha=0.2)
    path = out_dir / f"{prefix}_cumulative_ic.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def make_standard_plots(df: pd.DataFrame, out_dir: str | Path, prefix: str) -> list[str]:
    paths = []
    paths.extend(plot_distributions(df, out_dir, prefix))
    paths.append(plot_binned_pred_label(df, out_dir, prefix))
    paths.append(plot_monthly_ic(df, out_dir, prefix))
    paths.append(plot_cumulative_ic(df, out_dir, prefix))
    return paths

