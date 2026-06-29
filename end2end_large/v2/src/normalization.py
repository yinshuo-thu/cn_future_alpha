from __future__ import annotations

import numpy as np
import pandas as pd


def _causal_rolling_z(values: np.ndarray, window: int, min_periods: int, clip: float) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(clean, dtype=np.float64)])
    csum2 = np.concatenate([[0.0], np.cumsum(clean * clean, dtype=np.float64)])
    ccnt = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    n = len(values)
    end = np.arange(0, n)  # exclusive prior endpoint for each current row
    start = np.maximum(0, end - int(window))
    counts = ccnt[end] - ccnt[start]
    sums = csum[end] - csum[start]
    sums2 = csum2[end] - csum2[start]
    mean = np.zeros(n, dtype=np.float64)
    std = np.ones(n, dtype=np.float64)
    ok = counts >= int(min_periods)
    mean[ok] = sums[ok] / counts[ok]
    var = np.maximum(sums2[ok] / counts[ok] - mean[ok] * mean[ok], 1e-12)
    std[ok] = np.sqrt(var)
    z = (values - mean) / (std + 1e-8)
    z[~ok | ~finite] = 0.0
    return np.nan_to_num(np.clip(z, -clip, clip), nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def add_causal_rolling_zscores(
    df: pd.DataFrame,
    columns: list[str],
    window: int = 240,
    min_periods: int = 60,
    clip: float = 8.0,
    prefix: str = "rz_",
) -> tuple[pd.DataFrame, list[str]]:
    out = df.sort_values(["symbol", "datetime"]).reset_index(drop=True).copy()
    feature_cols: list[str] = []
    groups = [np.asarray(idx, dtype=np.int64) for idx in out.groupby("symbol", sort=False).indices.values()]
    for col in columns:
        zcol = f"{prefix}{col}"
        result = np.zeros(len(out), dtype=np.float32)
        values_all = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        for pos in groups:
            result[pos] = _causal_rolling_z(values_all[pos], window, min_periods, clip)
        out[zcol] = result
        feature_cols.append(zcol)
    return out, feature_cols


def add_clip_only_features(
    df: pd.DataFrame,
    columns: list[str],
    clip: float = 8.0,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    feature_cols: list[str] = []
    for col in columns:
        fcol = f"cl_{col}"
        out[fcol] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-clip, clip)
            .astype("float32")
        )
        feature_cols.append(fcol)
    return out, feature_cols


def build_normalized_features(
    df: pd.DataFrame,
    zscore_columns: list[str],
    clip_columns: list[str],
    window: int = 240,
    min_periods: int = 60,
    clip: float = 8.0,
) -> tuple[pd.DataFrame, list[str]]:
    out, zcols = add_causal_rolling_zscores(df, zscore_columns, window, min_periods, clip)
    out, ccols = add_clip_only_features(out, clip_columns, clip)
    return out, zcols + ccols
