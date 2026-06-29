from __future__ import annotations

import numpy as np
import pandas as pd


def add_jump_risk_columns(
    df: pd.DataFrame,
    window: int = 240,
    min_periods: int = 60,
    thresholds: tuple[int, ...] = (8, 12, 16),
) -> pd.DataFrame:
    """Add robust return z-score jump flags using only historical one-minute returns.

    The rolling median/MAD are shifted by one bar, so the current return is compared with
    strictly prior returns for the same symbol.
    """
    out = df.sort_values(["symbol", "datetime"]).reset_index(drop=True).copy()
    close = pd.to_numeric(out["close"], errors="coerce").replace(0.0, np.nan)
    out["ret_1m_raw"] = out.groupby("symbol", sort=False)[close.name].transform(lambda s: np.log(s / s.shift(1)))
    out.loc[out["is_long_break_before"].fillna(False), "ret_1m_raw"] = np.nan
    robust = np.full(len(out), np.nan, dtype=np.float32)
    for _, idx in out.groupby("symbol", sort=False).indices.items():
        pos = np.asarray(idx, dtype=np.int64)
        r = out.loc[pos, "ret_1m_raw"].astype("float64")
        hist = r.shift(1)
        med = hist.rolling(window, min_periods=min_periods).median()
        mad = (hist - med).abs().rolling(window, min_periods=min_periods).median() * 1.4826
        z = (r - med) / (mad + 1e-8)
        robust[pos] = z.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    out["ret_1m_robust_z"] = robust
    for threshold in thresholds:
        out[f"jump_risk_{threshold}"] = (out["ret_1m_robust_z"].abs() > float(threshold)).fillna(False)
    return out


def add_horizon_jump_masks(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 10, 30, 60),
    thresholds: tuple[int, ...] = (8, 12, 16),
) -> pd.DataFrame:
    """For each horizon, mark labels invalid if [t+1, t+h] contains a jump-risk bar."""
    out = df.sort_values(["symbol", "datetime"]).reset_index(drop=True).copy()
    group_positions = [np.asarray(idx, dtype=np.int64) for idx in out.groupby(["symbol", "session_id"], sort=False).indices.values()]
    for threshold in thresholds:
        jump_col = f"jump_risk_{threshold}"
        jumps = out[jump_col].astype("int8").to_numpy(dtype=np.int8)
        for horizon in horizons:
            mask = np.zeros(len(out), dtype=bool)
            for pos in group_positions:
                vals = jumps[pos]
                if len(vals) <= horizon:
                    continue
                csum = np.concatenate([[0], np.cumsum(vals, dtype=np.int32)])
                # forward window excludes t and includes t+1...t+h
                counts = csum[horizon + 1 :] - csum[1 : len(vals) - horizon + 1]
                valid = counts == 0
                mask[pos[: len(valid)]] = valid
            out[f"jump_ok_h{horizon}_thr{threshold}"] = mask
    return out


def jump_filter_summary(df: pd.DataFrame, thresholds: tuple[int, ...] = (8, 12, 16)) -> pd.DataFrame:
    rows = [{"threshold": "no_filter", "jump_bars": 0, "jump_bar_pct": 0.0}]
    n = max(len(df), 1)
    for threshold in thresholds:
        col = f"jump_risk_{threshold}"
        count = int(df[col].sum()) if col in df.columns else 0
        rows.append({"threshold": threshold, "jump_bars": count, "jump_bar_pct": count / n})
    return pd.DataFrame(rows)
