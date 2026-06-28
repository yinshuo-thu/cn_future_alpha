from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_ic(pred, label) -> float:
    pred = np.asarray(pred, dtype=float)
    label = np.asarray(label, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 2:
        return float("nan")
    x = pred[mask]
    y = label[mask]
    den = np.sqrt(np.mean(x * x) * np.mean(y * y))
    if den < 1e-12:
        return float("nan")
    return float(np.mean(x * y) / den)


def compute_rank_ic(pred, label) -> float:
    s = pd.DataFrame({"pred": pred, "label": label}).dropna()
    if len(s) < 2:
        return float("nan")
    rp = s["pred"].rank(method="average").to_numpy(dtype=float)
    ry = s["label"].rank(method="average").to_numpy(dtype=float)
    rp = rp - rp.mean()
    ry = ry - ry.mean()
    den = np.sqrt(np.mean(rp * rp) * np.mean(ry * ry))
    return float(np.mean(rp * ry) / den) if den > 1e-12 else float("nan")


def metric_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, grp in df.dropna(subset=["pred", "label"]).groupby(group_col):
        rows.append(
            {
                group_col: key,
                "n": int(len(grp)),
                "ic": compute_ic(grp["pred"], grp["label"]),
                "rank_ic": compute_rank_ic(grp["pred"], grp["label"]),
            }
        )
    return pd.DataFrame(rows)


def add_period_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["year"] = out["datetime"].dt.year
    out["month"] = out["datetime"].dt.to_period("M").astype(str)
    return out


def monthly_ic_table(df: pd.DataFrame) -> pd.DataFrame:
    return metric_by_group(add_period_columns(df), "month")


def summarize_predictions(df: pd.DataFrame, split_metrics: list[dict] | None = None) -> dict:
    clean = df.dropna(subset=["pred", "label"]).copy()
    by_month = monthly_ic_table(clean)
    icir = float(by_month["ic"].mean() / by_month["ic"].std()) if len(by_month) > 1 and by_month["ic"].std() > 0 else float("nan")
    return {
        "n_rows": int(len(df)),
        "n_scored": int(len(clean)),
        "merged_ic": compute_ic(clean["pred"], clean["label"]),
        "merged_rank_ic": compute_rank_ic(clean["pred"], clean["label"]),
        "icir_monthly": icir,
        "pred_mean": float(clean["pred"].mean()) if len(clean) else float("nan"),
        "pred_std": float(clean["pred"].std()) if len(clean) else float("nan"),
        "label_mean": float(clean["label"].mean()) if len(clean) else float("nan"),
        "label_std": float(clean["label"].std()) if len(clean) else float("nan"),
        "split_metrics": split_metrics or [],
    }


def write_metric_tables(df: pd.DataFrame, out_dir: str | Path, split_metrics: list[dict]) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    pd.DataFrame(split_metrics).to_csv(out_dir / "split_metrics.csv", index=False)
    paths["split_metrics"] = str(out_dir / "split_metrics.csv")
    by_year = metric_by_group(add_period_columns(df), "year")
    by_year.to_csv(out_dir / "yearly_metrics.csv", index=False)
    paths["yearly_metrics"] = str(out_dir / "yearly_metrics.csv")
    by_month = monthly_ic_table(df)
    by_month.to_csv(out_dir / "monthly_metrics.csv", index=False)
    paths["monthly_metrics"] = str(out_dir / "monthly_metrics.csv")
    by_symbol = metric_by_group(df, "symbol")
    by_symbol.to_csv(out_dir / "symbol_metrics.csv", index=False)
    paths["symbol_metrics"] = str(out_dir / "symbol_metrics.csv")
    summary = summarize_predictions(df, split_metrics)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["metrics"] = str(out_dir / "metrics.json")
    return paths

