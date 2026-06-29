from __future__ import annotations

import numpy as np
import pandas as pd


RETURN_HORIZONS = (5, 10, 30, 60)


def _forward_log_return(close: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) <= horizon:
        return out
    cur = close[:-horizon]
    fut = close[horizon:]
    ok = np.isfinite(cur) & np.isfinite(fut) & (cur > 0) & (fut > 0)
    vals = np.full(len(cur), np.nan, dtype=np.float64)
    vals[ok] = np.log(fut[ok] / cur[ok])
    out[:-horizon] = vals
    return out


def _forward_rv(ret: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(len(ret), np.nan, dtype=np.float64)
    if len(ret) <= horizon:
        return out
    vals = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0) ** 2
    csum = np.concatenate([[0.0], np.cumsum(vals, dtype=np.float64)])
    # future window excludes t and includes t+1...t+h
    out[: len(ret) - horizon] = csum[horizon + 1 :] - csum[1 : len(ret) - horizon + 1]
    return out


def _forward_range(high: np.ndarray, low: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(len(high), np.nan, dtype=np.float64)
    if len(high) <= horizon:
        return out
    hi_win = np.lib.stride_tricks.sliding_window_view(high, horizon + 1)[:, 1:]
    lo_win = np.lib.stride_tricks.sliding_window_view(low, horizon + 1)[:, 1:]
    hi = np.nanmax(hi_win, axis=1)
    lo = np.nanmin(lo_win, axis=1)
    ok = np.isfinite(hi) & np.isfinite(lo) & (hi > 0) & (lo > 0)
    vals = np.full(len(hi), np.nan, dtype=np.float64)
    vals[ok] = np.log(hi[ok] / lo[ok])
    out[: len(vals)] = vals
    return out


def add_proxy_labels(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = RETURN_HORIZONS,
    jump_threshold: int | None = 12,
) -> pd.DataFrame:
    out = df.sort_values(["symbol", "datetime"]).reset_index(drop=True).copy()
    n = len(out)
    label_arrays: dict[int, np.ndarray] = {h: np.full(n, np.nan, dtype=np.float32) for h in horizons}
    mask_arrays: dict[int, np.ndarray] = {h: np.zeros(n, dtype=bool) for h in horizons}
    rv_array = np.full(n, np.nan, dtype=np.float32)
    range_array = np.full(n, np.nan, dtype=np.float32)
    rv_mask = np.zeros(n, dtype=bool)
    range_mask = np.zeros(n, dtype=bool)
    close_all = out["close"].astype("float64").to_numpy()
    high_all = out["high"].astype("float64").to_numpy()
    low_all = out["low"].astype("float64").to_numpy()
    ret_all = out["ret_1m"].astype("float64").to_numpy() if "ret_1m" in out.columns else np.full(n, np.nan)
    jump_ok: dict[int, np.ndarray] = {}
    if jump_threshold is not None:
        for horizon in horizons:
            col = f"jump_ok_h{horizon}_thr{jump_threshold}"
            if col in out.columns:
                jump_ok[horizon] = out[col].to_numpy(dtype=bool)
    for horizon in horizons:
        out[f"proxy_ret_{horizon}m"] = np.nan
        out[f"mask_proxy_ret_{horizon}m"] = False
    out["log_proxy_rv_30m"] = np.nan
    out["log_proxy_range_30m"] = np.nan
    out["mask_log_proxy_rv_30m"] = False
    out["mask_log_proxy_range_30m"] = False

    for pos in [np.asarray(idx, dtype=np.int64) for idx in out.groupby(["symbol", "session_id"], sort=False).indices.values()]:
        close = close_all[pos]
        high = high_all[pos]
        low = low_all[pos]
        ret1 = ret_all[pos]
        for horizon in horizons:
            label = _forward_log_return(close, horizon)
            label_arrays[horizon][pos] = label.astype("float32")
            valid = np.isfinite(label)
            if horizon in jump_ok:
                valid &= jump_ok[horizon][pos]
            mask_arrays[horizon][pos] = valid
        rv = _forward_rv(ret1, 30)
        rng = _forward_range(high, low, 30)
        rv_array[pos] = np.log(rv + 1e-12).astype("float32")
        range_array[pos] = np.log(rng + 1e-12).astype("float32")
        rv_valid = np.isfinite(rv) & (rv >= 0)
        rng_valid = np.isfinite(rng) & (rng >= 0)
        if 30 in jump_ok:
            rv_valid &= jump_ok[30][pos]
            rng_valid &= jump_ok[30][pos]
        rv_mask[pos] = rv_valid
        range_mask[pos] = rng_valid
    for horizon in horizons:
        out[f"proxy_ret_{horizon}m"] = label_arrays[horizon]
        out[f"mask_proxy_ret_{horizon}m"] = mask_arrays[horizon]
    out["log_proxy_rv_30m"] = rv_array
    out["log_proxy_range_30m"] = range_array
    out["mask_log_proxy_rv_30m"] = rv_mask
    out["mask_log_proxy_range_30m"] = range_mask
    return out


def add_cross_sectional_targets(df: pd.DataFrame, horizons: tuple[int, ...] = RETURN_HORIZONS) -> pd.DataFrame:
    out = df.copy()
    for horizon in horizons:
        raw_col = f"proxy_ret_{horizon}m"
        mask_col = f"mask_proxy_ret_{horizon}m"
        target_col = f"proxy_ret_{horizon}m_cs_zscore"
        rank_col = f"proxy_ret_{horizon}m_rank_gauss"
        out[target_col] = np.nan
        out[rank_col] = np.nan
        valid_values = out[raw_col].where(out[mask_col].astype(bool))
        g = valid_values.groupby(out["datetime"], sort=False)
        mean = g.transform("mean")
        std = g.transform("std").replace(0.0, np.nan)
        out[target_col] = ((valid_values - mean) / (std + 1e-8)).clip(-5.0, 5.0).astype("float32")
        # Rank-gaussian approximation without scipy: map rank pct to [-1, 1].
        ranks = valid_values.groupby(out["datetime"], sort=False).rank(method="average", pct=True)
        out[rank_col] = (2.0 * ranks - 1.0).clip(-1.0, 1.0).astype("float32")
    return out


def add_aux_target_normalization(
    df: pd.DataFrame,
    aux_cols: tuple[str, ...] = ("log_proxy_rv_30m", "log_proxy_range_30m"),
    train_end: str = "2019-01-01",
) -> pd.DataFrame:
    out = df.copy()
    train_mask = out["datetime"] < pd.Timestamp(train_end)
    for col in aux_cols:
        mask_col = f"mask_{col}"
        norm_col = f"{col}_norm"
        valid = train_mask & out.get(mask_col, pd.Series(True, index=out.index)).astype(bool) & np.isfinite(out[col])
        mean = float(out.loc[valid, col].mean()) if valid.any() else 0.0
        std = float(out.loc[valid, col].std()) if valid.any() else 1.0
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        out[norm_col] = ((out[col] - mean) / (std + 1e-8)).clip(-8.0, 8.0).astype("float32")
    return out
