from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    a = x[mask] - np.nanmean(x[mask])
    b = y[mask] - np.nanmean(y[mask])
    den = math.sqrt(float(np.mean(a * a) * np.mean(b * b)))
    return float(np.mean(a * b) / den) if den > 1e-12 else float("nan")


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 3:
        return float("nan")
    return pearson_corr(s["x"].rank(method="average").to_numpy(), s["y"].rank(method="average").to_numpy())


def merged_ic(pred: np.ndarray, label: np.ndarray, rank: bool = False) -> float:
    return rank_corr(pred, label) if rank else pearson_corr(pred, label)


def zscore_by_timestamp(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("datetime", sort=False)[col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0.0, np.nan)
    out[out_col] = ((out[col] - mean) / (std + 1e-8)).replace([np.inf, -np.inf], np.nan)
    return out


def add_prediction_variants(df: pd.DataFrame, pred_col: str = "pred_raw") -> pd.DataFrame:
    out = zscore_by_timestamp(df, pred_col, "pred_cs_zscore")
    sector_mean = out.groupby(["datetime", "sector"], sort=False)[pred_col].transform("mean")
    out["pred_sector_neutral"] = out[pred_col] - sector_mean
    out = zscore_by_timestamp(out, "pred_sector_neutral", "pred_sector_neutral_zscore")
    return out


def timestamp_ic_series(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str = "label",
    min_symbols: int = 10,
    rank: bool = False,
    stride: int = 1,
) -> pd.DataFrame:
    rows = []
    for ts, grp in df.dropna(subset=[pred_col, label_col]).groupby("datetime", sort=True):
        if len(grp) < min_symbols:
            continue
        rows.append(
            {
                "datetime": pd.Timestamp(ts),
                "n": int(len(grp)),
                "ic": rank_corr(grp[pred_col].to_numpy(), grp[label_col].to_numpy())
                if rank
                else pearson_corr(grp[pred_col].to_numpy(), grp[label_col].to_numpy()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if stride > 1:
        out = out.iloc[:: int(stride)].reset_index(drop=True)
    out["month"] = out["datetime"].dt.to_period("M").astype(str)
    out["session_type"] = np.where((out["datetime"].dt.hour >= 20) | (out["datetime"].dt.hour < 8), "night", "day")
    return out


def newey_west_tstat(values: np.ndarray, lag: int = 5) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    xc = x - x.mean()
    gamma0 = float(np.mean(xc * xc))
    var = gamma0
    max_lag = min(int(lag), n - 1)
    for k in range(1, max_lag + 1):
        gamma = float(np.mean(xc[k:] * xc[:-k]))
        weight = 1.0 - k / (max_lag + 1.0)
        var += 2.0 * weight * gamma
    se = math.sqrt(max(var, 1e-18) / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def block_bootstrap_ci(values: np.ndarray, block: int = 20, n_boot: int = 1000, seed: int = 11) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < block or n == 0:
        return float("nan"), float("nan")
    if n > 20000:
        # Dense stride=1 diagnostics can contain hundreds of thousands of timestamps.
        # Use a fast normal approximation there; block bootstrap remains for smaller
        # non-overlap primary series.
        se = float(np.std(x, ddof=1) / np.sqrt(n))
        mean = float(np.mean(x))
        return mean - 1.96 * se, mean + 1.96 * se
    rng = np.random.default_rng(seed)
    means = []
    starts = np.arange(0, n)
    for _ in range(int(n_boot)):
        parts = []
        while sum(len(p) for p in parts) < n:
            s = int(rng.choice(starts))
            e = min(n, s + block)
            parts.append(x[s:e])
        sample = np.concatenate(parts)[:n]
        means.append(float(np.mean(sample)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_ic_series(series: pd.DataFrame, prefix: str) -> dict[str, float]:
    vals = series["ic"].to_numpy(dtype=float) if not series.empty else np.asarray([], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {
            f"{prefix}_n_timestamps": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_icir": float("nan"),
            f"{prefix}_nw_tstat": float("nan"),
            f"{prefix}_boot_ci_low": float("nan"),
            f"{prefix}_boot_ci_high": float("nan"),
        }
    ci_low, ci_high = block_bootstrap_ci(vals)
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    return {
        f"{prefix}_n_timestamps": int(len(vals)),
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_std": std,
        f"{prefix}_icir": float(np.mean(vals) / std) if np.isfinite(std) and std > 0 else float("nan"),
        f"{prefix}_nw_tstat": newey_west_tstat(vals),
        f"{prefix}_boot_ci_low": ci_low,
        f"{prefix}_boot_ci_high": ci_high,
    }


def evaluate_predictions(
    pred_df: pd.DataFrame,
    pred_col: str = "pred_raw",
    label_col: str = "label",
    min_symbols: int = 10,
    eval_stride: int = 30,
) -> tuple[dict[str, float], dict[str, pd.DataFrame], pd.DataFrame]:
    df = add_prediction_variants(pred_df, pred_col)
    dense_raw = timestamp_ic_series(df, "pred_cs_zscore", label_col, min_symbols, rank=False, stride=1)
    dense_rank = timestamp_ic_series(df, "pred_cs_zscore", label_col, min_symbols, rank=True, stride=1)
    non_raw = timestamp_ic_series(df, "pred_sector_neutral_zscore", label_col, min_symbols, rank=False, stride=eval_stride)
    non_rank = timestamp_ic_series(df, "pred_sector_neutral_zscore", label_col, min_symbols, rank=True, stride=eval_stride)
    raw_non = timestamp_ic_series(df, "pred_cs_zscore", label_col, min_symbols, rank=False, stride=eval_stride)
    metrics: dict[str, float] = {}
    metrics.update(summarize_ic_series(dense_raw, "dense_cs_ic"))
    metrics.update(summarize_ic_series(dense_rank, "dense_cs_rankic"))
    metrics.update(summarize_ic_series(raw_non, "nonoverlap_raw_cs_ic"))
    metrics.update(summarize_ic_series(non_raw, "nonoverlap_sector_neutral_cs_ic"))
    metrics.update(summarize_ic_series(non_rank, "nonoverlap_sector_neutral_cs_rankic"))
    clean = df.dropna(subset=[pred_col, label_col])
    metrics["merged_proxy_ic"] = merged_ic(clean[pred_col].to_numpy(), clean[label_col].to_numpy(), rank=False)
    metrics["merged_proxy_rankic"] = merged_ic(clean[pred_col].to_numpy(), clean[label_col].to_numpy(), rank=True)
    metrics["n_predictions"] = int(len(df))
    metrics["n_scored"] = int(len(clean))

    tables: dict[str, pd.DataFrame] = {
        "dense_cs_ic": dense_raw,
        "dense_cs_rankic": dense_rank,
        "nonoverlap_sector_neutral_cs_ic": non_raw,
        "nonoverlap_sector_neutral_cs_rankic": non_rank,
        "nonoverlap_raw_cs_ic": raw_non,
    }
    if not non_raw.empty:
        tables["monthly_nonoverlap_sector_neutral_cs_ic"] = non_raw.groupby("month")["ic"].agg(["count", "mean", "std"]).reset_index()
        tables["session_nonoverlap_sector_neutral_cs_ic"] = non_raw.groupby("session_type")["ic"].agg(["count", "mean", "std"]).reset_index()
    sector_rows = []
    for sector, grp in df.dropna(subset=["pred_sector_neutral_zscore", label_col]).groupby("sector"):
        sector_rows.append(
            {
                "sector": sector,
                "n": len(grp),
                "merged_ic": merged_ic(grp["pred_sector_neutral_zscore"].to_numpy(), grp[label_col].to_numpy()),
                "merged_rankic": merged_ic(grp["pred_sector_neutral_zscore"].to_numpy(), grp[label_col].to_numpy(), rank=True),
            }
        )
    tables["sector_merged_metrics"] = pd.DataFrame(sector_rows)
    return metrics, tables, df


def write_evaluation_artifacts(metrics: dict[str, float], tables: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out / "metrics_summary.csv", index=False)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
