#!/usr/bin/env python3
"""Clean ensemble over rolling FU new-factor Ridge/LGB/MLP predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic


def month_range(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def load_parts(out_dir: Path, part_name: str, months: list[str], view: str) -> pd.DataFrame:
    rows = []
    for month in months:
        path = out_dir / "prediction_parts" / part_name / f"{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        cur = pd.read_parquet(path, columns=["symbol", "datetime", "label", view])
        cur["datetime"] = pd.to_datetime(cur["datetime"])
        cur = cur.rename(columns={view: part_name})
        rows.append(cur)
    return pd.concat(rows, ignore_index=True).sort_values(["datetime", "symbol"], kind="mergesort").reset_index(drop=True)


def merge_components(out_dir: Path, months: list[str], view: str) -> pd.DataFrame:
    mapping = {
        "ridge": "rolling_ridge",
        "lgb": "rolling_lgb",
        "mlp": "rolling_mlp",
    }
    base = None
    for short, part_name in mapping.items():
        cur = load_parts(out_dir, part_name, months, view).rename(columns={part_name: short})
        if base is None:
            base = cur
        else:
            base = base.merge(cur[["symbol", "datetime", short]], on=["symbol", "datetime"], how="inner")
    assert base is not None
    return base


def fit_weights(val: pd.DataFrame, names: list[str], signed: bool) -> tuple[np.ndarray, float]:
    x = val[names].to_numpy(np.float64)
    y = val["label"].to_numpy(np.float64)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = np.nan_to_num(x[mask], nan=0.0, posinf=0.0, neginf=0.0)
    y = y[mask]
    c = x.T @ y
    g = x.T @ x
    yy = float(y @ y)
    lower = np.full(len(names), -0.15 if signed else 0.0)
    upper = np.ones(len(names))
    return fit_ic_weights_from_stats(c, g, yy, lower, upper)


def evaluate(base: pd.DataFrame, names: list[str], w: np.ndarray) -> pd.DataFrame:
    out = base[["symbol", "datetime", "label"]].copy()
    x = np.nan_to_num(base[names].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out["pred"] = x @ w.astype(np.float32)
    return add_cross_sectional_norms(out, "pred")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--view", choices=["pred", "pred_xsz", "pred_xrank"], default="pred_xsz")
    parser.add_argument("--signed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = ["ridge", "lgb", "mlp"]
    val = merge_components(args.out_dir, month_range("2019-10", "2019-12"), args.view)
    test = merge_components(args.out_dir, month_range("2020-01", "2020-12"), args.view)
    w, val_fit_ic = fit_weights(val, names, args.signed)
    pred = evaluate(test, names, w)
    monthly = period_ic(pred, "pred_xsz", "M")
    corr = test[names].corr()

    tag = f"rolling_three_{args.view}_{'signed' if args.signed else 'nonneg'}"
    pred.to_parquet(args.out_dir / f"{tag}.parquet", index=False)
    monthly.to_csv(args.out_dir / f"{tag}_monthly_ic.csv")
    corr.to_csv(args.out_dir / f"{tag}_component_corr_2020.csv")
    row = {
        "model": tag,
        "view": args.view,
        "signed": bool(args.signed),
        "val_fit_ic_2019q4": float(val_fit_ic),
        "pred_ic_2020": compute_ic(pred["pred"].to_numpy(), pred["label"].to_numpy()),
        "pred_xsz_ic_2020": compute_ic(pred["pred_xsz"].to_numpy(), pred["label"].to_numpy()),
        "pred_xrank_ic_2020": compute_ic(pred["pred_xrank"].to_numpy(), pred["label"].to_numpy()),
        "monthly_mean": float(monthly.mean()),
        "monthly_ir": float(monthly.mean() / monthly.std()) if monthly.std() > 0 else float("nan"),
        "uses_2020_for_weights": False,
        **{f"w_{name}": float(weight) for name, weight in zip(names, w)},
    }
    summary_path = args.out_dir / "rolling_three_ensemble_summary.csv"
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        old = old[old["model"] != tag]
        out = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])
    out.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Rolling three-model ensemble monthly IC ({args.view})")
    ax.set_ylabel("IC")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"{tag}_monthly_ic.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame([row]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
