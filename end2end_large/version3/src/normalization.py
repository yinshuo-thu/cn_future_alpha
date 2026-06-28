from __future__ import annotations

import numpy as np
import pandas as pd


def add_rolling_zscore(
    df: pd.DataFrame,
    value_cols: list[str],
    window: int = 240,
    min_periods: int = 20,
    clip_value: float = 8.0,
    group_cols: tuple[str, ...] = ("symbol", "session_id"),
    prefix: str = "rz_",
) -> tuple[pd.DataFrame, list[str]]:
    """Add strictly causal rolling z-score columns.

    The mean/std at row t are computed from rows strictly before t within the
    same group. This keeps normalization history-only and avoids session-tail
    leakage.
    """
    out = df.copy()
    n = len(out)
    new_cols = [f"{prefix}{col}" for col in value_cols]
    source = {col: pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float32) for col in value_cols}
    result = {col: np.zeros(n, dtype=np.float32) for col in new_cols}
    group_indices = [np.asarray(idx, dtype=np.int64) for idx in out.groupby(list(group_cols), sort=False).indices.values()]
    for col, new_col in zip(value_cols, new_cols):
        values = source[col].astype(np.float64, copy=False)
        dest = result[new_col]
        for idx_arr in group_indices:
            vals = values[idx_arr]
            finite = np.isfinite(vals)
            clean = np.where(finite, vals, 0.0)
            count = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
            csum = np.concatenate([[0.0], np.cumsum(clean, dtype=np.float64)])
            csum2 = np.concatenate([[0.0], np.cumsum(clean * clean, dtype=np.float64)])
            end = np.arange(len(vals), dtype=np.int64)
            start = np.maximum(0, end - int(window))
            nobs = count[end] - count[start]
            sums = csum[end] - csum[start]
            sumsq = csum2[end] - csum2[start]
            valid = nobs >= int(min_periods)
            mean = np.zeros(len(vals), dtype=np.float64)
            var = np.zeros(len(vals), dtype=np.float64)
            mean[valid] = sums[valid] / np.maximum(nobs[valid], 1.0)
            var[valid] = sumsq[valid] / np.maximum(nobs[valid], 1.0) - mean[valid] * mean[valid]
            std = np.sqrt(np.maximum(var, 0.0))
            z = np.zeros(len(vals), dtype=np.float64)
            z[valid] = (vals[valid] - mean[valid]) / (std[valid] + 1e-8)
            z = np.nan_to_num(np.clip(z, -clip_value, clip_value), nan=0.0, posinf=0.0, neginf=0.0)
            dest[idx_arr] = z.astype(np.float32)
    for col in new_cols:
        out[col] = result[col]
    return out, new_cols
