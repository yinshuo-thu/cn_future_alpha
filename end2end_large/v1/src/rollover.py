from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RolloverDetectionConfig:
    max_horizon_bars: int = 60
    min_abs_log_factor_jump: float = 0.015
    mad_multiplier: float = 10.0


def infer_proxy_adjustment_factor(
    adjusted_close: pd.Series,
    raw_vwap_proxy: pd.Series,
) -> pd.Series:
    close = pd.to_numeric(adjusted_close, errors="coerce").replace(0.0, np.nan)
    proxy = pd.to_numeric(raw_vwap_proxy, errors="coerce").replace(0.0, np.nan)
    factor = proxy / close
    return factor.replace([np.inf, -np.inf], np.nan)


def detect_rollover_candidates(
    df: pd.DataFrame,
    adjusted_close_col: str = "close",
    raw_vwap_proxy_col: str = "raw_vwap_proxy",
    config: RolloverDetectionConfig = RolloverDetectionConfig(),
) -> pd.DataFrame:
    factor = infer_proxy_adjustment_factor(df[adjusted_close_col], df[raw_vwap_proxy_col])
    log_factor = np.log(factor.to_numpy(dtype=np.float64))
    jump = np.diff(log_factor, prepend=np.nan)
    finite = jump[np.isfinite(jump)]
    if len(finite) == 0:
        threshold = np.nan
        mask = np.zeros(len(df), dtype=bool)
    else:
        med = np.nanmedian(finite)
        mad = np.nanmedian(np.abs(finite - med)) * 1.4826
        threshold = max(config.min_abs_log_factor_jump, config.mad_multiplier * float(mad))
        mask = np.isfinite(jump) & (np.abs(jump) > threshold)
    out = df.loc[mask, ["symbol", "datetime"]].copy()
    if "gap_min" in df.columns:
        out["gap_min"] = df.loc[mask, "gap_min"].to_numpy()
    out["log_factor_jump"] = jump[mask]
    out["abs_log_factor_jump"] = np.abs(jump[mask])
    out["roll_threshold"] = threshold
    return out.sort_values("abs_log_factor_jump", ascending=False)


def invalid_window_mask(n_rows: int, roll_indices: np.ndarray, bars: int = 60) -> np.ndarray:
    mask = np.zeros(n_rows, dtype=bool)
    for idx in roll_indices:
        left = max(0, int(idx) - bars)
        right = min(n_rows, int(idx) + bars + 1)
        mask[left:right] = True
    return mask
