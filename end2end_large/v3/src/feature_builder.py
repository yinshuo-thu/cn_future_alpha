from __future__ import annotations

import numpy as np
import pandas as pd


BASE_ZSCORE_FEATURES = [
    "ret_1m",
    "oc_ret",
    "hl_range",
    "upper_shadow",
    "lower_shadow",
    "log1p_volume",
    "volume_change",
    "log1p_amount",
    "amount_change",
    "log1p_oi",
    "oi_change",
]

CLIP_ONLY_FEATURES = [
    "close_pos",
    "rel_ret_1m",
    "rel_oc_ret",
    "rel_hl_range",
    "rel_volume_change",
    "rel_amount_change",
    "rel_oi_change",
]

OPERATOR_FEATURES = []


def _safe_log_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce").replace(0.0, np.nan)
    den = pd.to_numeric(den, errors="coerce").replace(0.0, np.nan)
    return np.log(num / den).replace([np.inf, -np.inf], np.nan)


def add_base_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.sort_values(["symbol", "datetime"]).reset_index(drop=True).copy()
    eps = 1e-8
    open_ = out["open"].astype("float64").clip(lower=eps)
    high = out["high"].astype("float64").clip(lower=eps)
    low = out["low"].astype("float64").clip(lower=eps)
    close = out["close"].astype("float64").clip(lower=eps)
    log_close = np.log(close)
    out["ret_1m"] = out.groupby("symbol", sort=False)["close"].transform(lambda s: np.log(s.clip(lower=eps) / s.shift(1).clip(lower=eps)))
    out.loc[out["is_long_break_before"].fillna(False), "ret_1m"] = np.nan
    out["oc_ret"] = np.log(close / open_)
    out["hl_range"] = np.log(high / low)
    spread = (high - low).replace(0.0, np.nan)
    out["close_pos"] = ((close - low) / (spread + eps)).clip(0.0, 1.0)
    out["upper_shadow"] = ((high - np.maximum(open_, close)) / (spread + eps)).clip(lower=0.0)
    out["lower_shadow"] = ((np.minimum(open_, close) - low) / (spread + eps)).clip(lower=0.0)
    for src, dst, diff_dst in [
        ("volume", "log1p_volume", "volume_change"),
        ("amount", "log1p_amount", "amount_change"),
        ("oi", "log1p_oi", "oi_change"),
    ]:
        out[dst] = np.log1p(pd.to_numeric(out[src], errors="coerce").clip(lower=0.0))
        out[diff_dst] = out.groupby("symbol", sort=False)[dst].diff()
        out.loc[out["is_long_break_before"].fillna(False), diff_dst] = np.nan
    # remove accidental infinities before normalization
    for col in BASE_ZSCORE_FEATURES + ["close_pos"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    z_features = list(BASE_ZSCORE_FEATURES)
    clip_features = ["close_pos"]
    return out, z_features, clip_features


def add_cross_sectional_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    rel_specs = [
        ("ret_1m", "rel_ret_1m"),
        ("oc_ret", "rel_oc_ret"),
        ("hl_range", "rel_hl_range"),
        ("volume_change", "rel_volume_change"),
        ("amount_change", "rel_amount_change"),
        ("oi_change", "rel_oi_change"),
    ]
    for src, dst in rel_specs:
        g = out.groupby("datetime", sort=False)[src]
        mean = g.transform("mean")
        std = g.transform("std").replace(0.0, np.nan)
        out[dst] = ((out[src] - mean) / (std + 1e-8)).clip(-8.0, 8.0).astype("float32")
    return out, [dst for _, dst in rel_specs]


def _rolling_mean(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(clean, dtype=np.float64)])
    ccnt = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    n = len(values)
    end = np.arange(1, n + 1)
    start = np.maximum(0, end - int(window))
    sums = csum[end] - csum[start]
    counts = ccnt[end] - ccnt[start]
    out = np.full(n, np.nan, dtype=np.float64)
    ok = counts >= int(min_periods)
    out[ok] = sums[ok] / counts[ok]
    return out


def _rolling_sum(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(clean, dtype=np.float64)])
    ccnt = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    n = len(values)
    end = np.arange(1, n + 1)
    start = np.maximum(0, end - int(window))
    sums = csum[end] - csum[start]
    counts = ccnt[end] - ccnt[start]
    out = np.full(n, np.nan, dtype=np.float64)
    out[counts >= int(min_periods)] = sums[counts >= int(min_periods)]
    return out


def add_lite_operator_features(df: pd.DataFrame, windows: tuple[int, ...] = (5, 20, 60)) -> tuple[pd.DataFrame, list[str]]:
    out = df.sort_values(["symbol", "datetime"]).reset_index(drop=True).copy()
    op_cols: list[str] = []
    for window in windows:
        for name in [
            f"op_ret_zscore_{window}",
            f"op_momentum_{window}",
            f"op_abs_ret_mean_{window}",
            f"op_volume_zscore_{window}",
            f"op_ret_volume_corr_{window}",
        ]:
            out[name] = np.nan
            op_cols.append(name)
    for _, idx in out.groupby("symbol", sort=False).indices.items():
        pos = np.asarray(idx, dtype=np.int64)
        ret = out.loc[pos, "ret_1m"].to_numpy(dtype=np.float64)
        vol_chg = out.loc[pos, "volume_change"].to_numpy(dtype=np.float64)
        log_close = np.log(out.loc[pos, "close"].astype("float64").clip(lower=1e-8).to_numpy())
        for window in windows:
            mean = _rolling_mean(ret, window, max(3, window // 3))
            mean_sq = _rolling_mean(ret * ret, window, max(3, window // 3))
            std = np.sqrt(np.maximum(mean_sq - mean * mean, 1e-12))
            out.loc[pos, f"op_ret_zscore_{window}"] = np.clip((ret - mean) / (std + 1e-8), -12, 12).astype("float32")
            mom = np.full(len(pos), np.nan, dtype=np.float64)
            if len(pos) > window:
                mom[window:] = log_close[window:] - log_close[:-window]
            out.loc[pos, f"op_momentum_{window}"] = mom.astype("float32")
            out.loc[pos, f"op_abs_ret_mean_{window}"] = _rolling_mean(np.abs(ret), window, max(3, window // 3)).astype("float32")
            vmean = _rolling_mean(vol_chg, window, max(3, window // 3))
            vstd = np.sqrt(np.maximum(_rolling_mean(vol_chg * vol_chg, window, max(3, window // 3)) - vmean * vmean, 1e-12))
            out.loc[pos, f"op_volume_zscore_{window}"] = np.clip((vol_chg - vmean) / (vstd + 1e-8), -12, 12).astype("float32")
            r = ret - mean
            v = vol_chg - vmean
            cov = _rolling_mean(r * v, window, max(3, window // 3))
            out.loc[pos, f"op_ret_volume_corr_{window}"] = np.clip(cov / (std * vstd + 1e-8), -1, 1).astype("float32")
    for col in op_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    return out, op_cols


def build_features(df: pd.DataFrame, use_lite_operators: bool = True) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    out, z_cols, clip_cols = add_base_features(df)
    out, rel_cols = add_cross_sectional_features(out)
    clip_cols = clip_cols + rel_cols
    op_cols: list[str] = []
    if use_lite_operators:
        out, op_cols = add_lite_operator_features(out)
        z_cols = z_cols + op_cols
    return out, z_cols, clip_cols, op_cols
