from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import ndtri


RAW_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "oi"]
DEFAULT_EXCLUDED_SYMBOLS = ["T", "TF", "TS", "IF", "IC", "IH"]
LONG_BREAK_THRESH_MIN = 60


@dataclass(frozen=True)
class RollingSplit:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class SymbolArrays:
    symbol: str
    features: np.ndarray
    labels: np.ndarray
    targets: np.ndarray
    aux_targets: np.ndarray | None
    datetimes: np.ndarray
    minute_id: np.ndarray
    dayofweek_id: np.ndarray
    month_id: np.ndarray
    source_index: np.ndarray
    session_pos: np.ndarray


def discover_symbols(
    data_dir: str | Path,
    symbols: Iterable[str] | None = None,
    excluded_symbols: Iterable[str] = DEFAULT_EXCLUDED_SYMBOLS,
    max_symbols: int | None = None,
) -> list[str]:
    data_dir = Path(data_dir)
    excluded = set(excluded_symbols or [])
    if symbols:
        out = [s.strip().upper() for s in symbols if s and s.strip().upper() not in excluded]
    else:
        out = sorted(p.stem for p in data_dir.glob("*.csv") if p.stem not in excluded)
    if max_symbols:
        out = out[: int(max_symbols)]
    return out


def load_symbol_csv(data_dir: str | Path, symbol: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{symbol}.csv"
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    dt_cols = [c for c in df.columns if "time" in c or "date" in c or c in ("datetime", "dt")]
    if not dt_cols:
        dt_cols = [df.columns[0]]
    df = df.rename(columns={dt_cols[0]: "datetime"})
    df["datetime"] = pd.to_datetime(
        df["datetime"].astype(str).str.slice(0, 16),
        format="%Y-%m-%d %H:%M",
    )
    df["symbol"] = symbol
    df = df.rename(columns={"open interest": "oi", "openinterest": "oi", "open_interest": "oi"})
    keep = ["symbol", "datetime"] + [c for c in RAW_COLUMNS if c in df.columns]
    df = df[keep].drop_duplicates(subset=["symbol", "datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["symbol", "datetime"] + RAW_COLUMNS]


def detect_sessions(df_sym: pd.DataFrame) -> pd.DataFrame:
    df = df_sym.copy().sort_values("datetime").reset_index(drop=True)
    n = len(df)
    gaps = df["datetime"].diff().dt.total_seconds().div(60).fillna(0.0).to_numpy()
    session_id = np.zeros(n, dtype=np.int32)
    is_long_break_before = np.zeros(n, dtype=bool)
    sid = 0
    for i in range(1, n):
        if gaps[i] >= LONG_BREAK_THRESH_MIN:
            sid += 1
            is_long_break_before[i] = True
        session_id[i] = sid
    df["session_id"] = session_id
    df["is_long_break_before"] = is_long_break_before
    df["gap_min"] = gaps.astype(np.float32)
    bars_to_next = np.zeros(n, dtype=np.int32)
    remaining = 0
    for i in range(n - 1, -1, -1):
        if i < n - 1 and is_long_break_before[i + 1]:
            remaining = 0
        bars_to_next[i] = remaining
        remaining += 1
    df["bars_to_next_long_break"] = bars_to_next
    return df


def build_labels(df_sym: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    df = df_sym.copy().sort_values("datetime").reset_index(drop=True)
    n = len(df)
    labels = np.full(n, np.nan, dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    session_id = df["session_id"].to_numpy()
    for _, idx in pd.Series(np.arange(n)).groupby(session_id, sort=False):
        pos = idx.to_numpy()
        if len(pos) <= horizon:
            continue
        cur = close[pos[:-horizon]]
        fut = close[pos[horizon:]]
        ok = (cur != 0) & np.isfinite(cur) & np.isfinite(fut)
        vals = np.full(len(cur), np.nan, dtype=np.float64)
        vals[ok] = fut[ok] / cur[ok] - 1.0
        labels[pos[:-horizon]] = vals
    is_lb = df["is_long_break_before"].to_numpy()
    for lb_idx in np.where(is_lb)[0]:
        labels[max(0, lb_idx - horizon) : lb_idx] = np.nan
    df["label"] = labels.astype(np.float32)
    return df


def build_labeled_frame(
    data_dir: str | Path,
    symbols: Iterable[str] | None = None,
    excluded_symbols: Iterable[str] = DEFAULT_EXCLUDED_SYMBOLS,
    max_symbols: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    label_horizon: int = 30,
) -> pd.DataFrame:
    frames = []
    for symbol in discover_symbols(data_dir, symbols, excluded_symbols, max_symbols):
        raw = load_symbol_csv(data_dir, symbol)
        if start_date:
            raw = raw[raw["datetime"] >= pd.Timestamp(start_date) - pd.Timedelta(days=7)]
        if end_date:
            raw = raw[raw["datetime"] < pd.Timestamp(end_date) + pd.Timedelta(days=7)]
        labeled = build_labels(detect_sessions(raw), horizon=label_horizon)
        frames.append(labeled)
    if not frames:
        raise ValueError("no symbols loaded")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    if start_date:
        df = df[df["datetime"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["datetime"] < pd.Timestamp(end_date)]
    return df.reset_index(drop=True)


def _safe_divide(num: np.ndarray, den: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return num / (den + eps)


def _group_shifted_rolling_z(
    out: pd.DataFrame,
    group_cols: list[str],
    col: str,
    window: int,
    min_periods: int,
) -> pd.Series:
    values = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(
        dtype=np.float64
    )
    result = np.zeros(len(out), dtype=np.float32)
    for idx in out.groupby(group_cols, sort=False).indices.values():
        idx_arr = np.asarray(idx, dtype=np.int64)
        prior = pd.Series(values[idx_arr]).shift(1)
        mean = prior.rolling(window, min_periods=min_periods).mean().to_numpy(dtype=np.float64)
        std = prior.rolling(window, min_periods=min_periods).std(ddof=0).to_numpy(dtype=np.float64)
        z = (values[idx_arr] - mean) / (std + 1e-8)
        result[idx_arr] = np.nan_to_num(np.clip(z, -20.0, 20.0), nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return pd.Series(result, index=out.index)


def _add_ohlcv_factor_columns(out: pd.DataFrame, feature_cols: list[str], group_cols: list[str]) -> list[str]:
    print("[data] adding ohlcv factor columns: bar shape", flush=True)
    close = np.clip(out["close"].to_numpy(dtype=np.float64), 1e-8, None)
    open_ = np.clip(out["open"].to_numpy(dtype=np.float64), 1e-8, None)
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    spread = np.maximum(high - low, 1e-8)
    mid = 0.5 * (high + low)
    body = close - open_

    factor_cols: list[str] = []
    shape_values = {
        "hl_log_range": np.log(np.clip(high, 1e-8, None) / np.clip(low, 1e-8, None)),
        "body_pct": _safe_divide(body, spread),
        "upper_wick_pct": _safe_divide(high - np.maximum(open_, close), spread),
        "lower_wick_pct": _safe_divide(np.minimum(open_, close) - low, spread),
        "close_to_mid": _safe_divide(close - mid, spread),
        "body_to_close": _safe_divide(body, np.abs(close)),
    }
    for name, values in shape_values.items():
        out[name] = values.astype("float32")
        factor_cols.append(name)

    print("[data] adding ohlcv factor columns: temporal momentum/volatility", flush=True)
    g = out.groupby(group_cols, sort=False)
    group_indices = [np.asarray(idx, dtype=np.int64) for idx in g.indices.values()]

    def _rolling_mean_by_group(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
        result = np.zeros(len(out), dtype=np.float32)
        clean_all = np.nan_to_num(values.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        for pos in group_indices:
            vals = clean_all[pos]
            n = len(vals)
            if n == 0:
                continue
            csum = np.concatenate([[0.0], np.cumsum(vals, dtype=np.float64)])
            end = np.arange(1, n + 1)
            start = np.maximum(0, end - window)
            counts = end - start
            mean = (csum[end] - csum[start]) / np.maximum(counts, 1)
            mean[counts < min_periods] = 0.0
            result[pos] = mean.astype(np.float32)
        return result

    log_close = pd.Series(np.log(close), index=out.index)
    out["_tmp_log_close"] = log_close.astype("float32")
    log_close_arr = log_close.to_numpy(dtype=np.float64)
    for window in [2, 5, 15, 30]:
        col = f"log_ret_{window}"
        vals = np.zeros(len(out), dtype=np.float32)
        for pos in group_indices:
            if len(pos) > window:
                vals[pos[window:]] = (log_close_arr[pos[window:]] - log_close_arr[pos[:-window]]).astype(np.float32)
        out[col] = vals
        factor_cols.append(col)
    ret_arr = out["log_ret_1"].to_numpy(dtype=np.float64)
    range_arr = out["range_pct"].to_numpy(dtype=np.float64)
    for window in [5, 15, 30]:
        minp = max(3, window // 3)
        rv_col = f"rv_{window}"
        out[rv_col] = np.sqrt(np.maximum(_rolling_mean_by_group(ret_arr * ret_arr, window, minp), 0.0)).astype("float32")
        range_col = f"range_mean_{window}"
        out[range_col] = _rolling_mean_by_group(range_arr, window, minp).astype("float32")
        factor_cols.extend([rv_col, range_col])

    print("[data] adding ohlcv factor columns: past-only activity surprises", flush=True)
    for base in ["log1p_volume", "log1p_amount", "log1p_oi"]:
        for window in [30, 120]:
            col = f"{base}_surprise_{window}"
            out[col] = _group_shifted_rolling_z(out, group_cols, base, window=window, min_periods=max(10, window // 4)).astype(
                "float32"
            )
            factor_cols.append(col)

    print("[data] adding ohlcv factor columns: price-volume interactions", flush=True)
    interaction_values = {
        "ret_x_range": out["log_ret_1"].to_numpy(dtype=np.float64) * out["range_pct"].to_numpy(dtype=np.float64),
        "ret_x_close_pos": out["log_ret_1"].to_numpy(dtype=np.float64) * out["close_pos"].to_numpy(dtype=np.float64),
        "close_pos_x_range": out["close_pos"].to_numpy(dtype=np.float64) * out["range_pct"].to_numpy(dtype=np.float64),
        "ret_x_volume_surprise": out["log_ret_1"].to_numpy(dtype=np.float64)
        * out["log1p_volume_surprise_30"].to_numpy(dtype=np.float64),
        "body_x_volume_surprise": out["body_pct"].to_numpy(dtype=np.float64)
        * out["log1p_volume_surprise_30"].to_numpy(dtype=np.float64),
        "ret_x_oi_change": out["log_ret_1"].to_numpy(dtype=np.float64) * out["d_log1p_oi"].to_numpy(dtype=np.float64),
        "absret_per_amount": _safe_divide(
            np.abs(out["log_ret_1"].to_numpy(dtype=np.float64)),
            1.0 + np.abs(out["log1p_amount"].to_numpy(dtype=np.float64)),
        ),
    }
    for name, values in interaction_values.items():
        out[name] = values.astype("float32")
        factor_cols.append(name)

    print("[data] adding ohlcv factor columns: lagged state", flush=True)
    lag_map = {
        "log_ret_1": [1, 2, 5],
        "range_pct": [1],
        "close_pos": [1],
        "d_log1p_volume": [1],
        "log1p_volume_surprise_30": [1],
    }
    for base, lags in lag_map.items():
        for lag in lags:
            col = f"{base}_lag{lag}"
            out[col] = (
                g[base].shift(lag).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
            )
            factor_cols.append(col)

    print("[data] adding ohlcv factor columns: same-timestamp cross-symbol context", flush=True)
    xsec_cols = ["log_ret_1", "range_pct", "d_log1p_volume", "rv_30", "log1p_volume_surprise_30"]
    for col in xsec_cols:
        by_time = out.groupby("datetime", sort=False)[col]
        mean = by_time.transform("mean")
        std = by_time.transform("std").replace(0.0, np.nan)
        rel_col = f"xrel_{col}"
        rank_col = f"xrank_{col}"
        out[rel_col] = ((out[col] - mean) / (std + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(
            "float32"
        )
        out[rank_col] = (by_time.rank(pct=True) - 0.5).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
        factor_cols.extend([rel_col, rank_col])

    out.drop(columns=["_tmp_log_close"], inplace=True)
    print(f"[data] ohlcv factor columns added={len(factor_cols)} total_raw_features={len(feature_cols) + len(factor_cols)}", flush=True)
    return feature_cols + factor_cols


def _load_selected_factor_names() -> list[str]:
    path = Path("/root/autodl-tmp/quant/artifacts/selected_factors.txt")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"no selected factors found in {path}")
    return names


def _selected_factor_base(name: str) -> tuple[str, str]:
    for view in ("tsz", "csz", "csr"):
        prefix = f"{view}_"
        if name.startswith(prefix):
            return view, name[len(prefix) :]
    return "raw", name


def _selected_csz(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = panel.groupby("datetime", sort=False)
    out = {}
    for col in cols:
        mu = g[col].transform("mean")
        sd = g[col].transform("std").replace(0.0, np.nan)
        out[col] = ((panel[col] - mu) / (sd + 1e-8)).astype("float32")
    return pd.DataFrame(out, index=panel.index)


def _selected_csr(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return (panel.groupby("datetime", sort=False)[cols].rank(pct=True) - 0.5).astype("float32")


def _selected_tsz(panel: pd.DataFrame, cols: list[str], window: int = 120, min_periods: int = 30) -> pd.DataFrame:
    out = np.full((len(panel), len(cols)), np.nan, dtype=np.float32)
    for idx in panel.groupby("symbol", sort=False).indices.values():
        pos = np.asarray(idx, dtype=np.int64)
        vals = panel.iloc[pos][cols].to_numpy(dtype=np.float32, copy=True)
        finite = np.isfinite(vals)
        clean = np.where(finite, vals, 0.0).astype(np.float64, copy=False)
        cnt = np.cumsum(finite.astype(np.float64), axis=0)
        s1 = np.cumsum(clean, axis=0)
        s2 = np.cumsum(clean * clean, axis=0)
        cnt = np.vstack([np.zeros((1, len(cols))), cnt])
        s1 = np.vstack([np.zeros((1, len(cols))), s1])
        s2 = np.vstack([np.zeros((1, len(cols))), s2])
        end = np.arange(1, len(pos) + 1)
        start = np.maximum(0, end - window)
        nobs = cnt[end] - cnt[start]
        sums = s1[end] - s1[start]
        sumsq = s2[end] - s2[start]
        mean = sums / np.maximum(nobs, 1.0)
        var = (sumsq - sums * sums / np.maximum(nobs, 1.0)) / np.maximum(nobs - 1.0, 1.0)
        z = (clean - mean) / (np.sqrt(np.maximum(var, 0.0)) + 1e-8)
        z[(nobs < min_periods) | ~finite] = np.nan
        out[pos] = z.astype(np.float32)
    return pd.DataFrame(out, index=panel.index, columns=cols)


def _add_selected_factor_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    selected = _load_selected_factor_names()
    by_view: dict[str, list[str]] = {"raw": [], "tsz": [], "csz": [], "csr": []}
    for name in selected:
        view, base = _selected_factor_base(name)
        by_view[view].append(base)
    base_needed = sorted({base for bases in by_view.values() for base in bases})

    feature_model_root = Path("/root/autodl-tmp/quant/code/feature_model")
    if str(feature_model_root) not in sys.path:
        sys.path.insert(0, str(feature_model_root))
    from src.features.factor_lib import compute_symbol_factors

    print(
        f"[data] selected factor catalog: selected={len(selected)} raw_bases={len(base_needed)}",
        flush=True,
    )
    parts = []
    for i, (sym, g) in enumerate(df.sort_values(["symbol", "datetime"]).groupby("symbol", sort=False), 1):
        ff = compute_symbol_factors(g)
        keep = ["symbol", "datetime", "label"] + [c for c in base_needed if c in ff.columns]
        missing = [c for c in base_needed if c not in ff.columns]
        if missing:
            raise ValueError(f"selected factor bases missing for {sym}: {missing[:10]}")
        ff = ff[keep].copy()
        for col in keep:
            if col not in ("symbol", "datetime"):
                ff[col] = pd.to_numeric(ff[col], errors="coerce").astype("float32")
        parts.append(ff)
        if i % 5 == 0 or i == 1:
            print(f"[data] selected factors computed symbols={i} rows~{sum(len(p) for p in parts)}", flush=True)
    panel = pd.concat(parts, ignore_index=True)
    del parts
    panel = panel.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    print(f"[data] selected factor raw panel rows={len(panel)} bases={len(base_needed)}", flush=True)

    feature_frames: dict[str, pd.Series] = {}
    for base in by_view["raw"]:
        feature_frames[base] = panel[base].astype("float32")
    for view, fn in (("tsz", _selected_tsz), ("csz", _selected_csz), ("csr", _selected_csr)):
        bases = sorted(set(by_view[view]))
        if not bases:
            continue
        print(f"[data] selected factor view={view} cols={len(bases)}", flush=True)
        view_df = fn(panel[["symbol", "datetime"] + bases] if view == "tsz" else panel[["datetime"] + bases], bases)
        for base in by_view[view]:
            feature_frames[f"{view}_{base}"] = view_df[base].astype("float32")
        del view_df

    features = pd.DataFrame({name: feature_frames[name] for name in selected}, index=panel.index)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-20.0, 20.0).astype("float32")
    meta_cols = [
        "symbol",
        "datetime",
        "session_id",
        "is_long_break_before",
        "gap_min",
        "bars_to_next_long_break",
        "label",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "oi",
    ]
    out = df[meta_cols].merge(
        pd.concat([panel[["symbol", "datetime"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1),
        on=["symbol", "datetime"],
        how="left",
        validate="one_to_one",
    )
    out[selected] = out[selected].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    print(f"[data] selected factor features materialized rows={len(out)} features={len(selected)}", flush=True)
    return out, selected


def add_stable_raw_columns(df: pd.DataFrame, feature_set: str = "base") -> tuple[pd.DataFrame, list[str]]:
    if feature_set == "selected_factors":
        return _add_selected_factor_columns(df)
    out = df.copy()
    for col in ["volume", "amount", "oi"]:
        out[f"log1p_{col}"] = np.log1p(np.clip(out[col].fillna(0.0).to_numpy(dtype=np.float64), 0.0, None)).astype(
            np.float32
        )
    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    feature_cols = ["open", "high", "low", "close", "log1p_volume", "log1p_amount", "log1p_oi"]
    if feature_set in {"extended", "extended_market", "extended_market_lite", "ohlcv_factors"}:
        keys = ["symbol", "session_id"]
        close = np.clip(out["close"].to_numpy(dtype=np.float64), 1e-8, None)
        open_ = np.clip(out["open"].to_numpy(dtype=np.float64), 1e-8, None)
        high = out["high"].to_numpy(dtype=np.float64)
        low = out["low"].to_numpy(dtype=np.float64)
        log_close = pd.Series(np.log(close), index=out.index)
        out["log_ret_1"] = log_close.groupby([out[k] for k in keys], sort=False).diff().fillna(0.0).astype("float32")
        out["range_pct"] = ((high - low) / (np.abs(close) + 1e-8)).astype("float32")
        out["close_pos"] = ((close - low) / (high - low + 1e-8) - 0.5).astype("float32")
        out["oc_ret"] = (close / open_ - 1.0).astype("float32")
        for col in ["log1p_volume", "log1p_amount", "log1p_oi"]:
            out[f"d_{col}"] = (
                out.groupby(keys, sort=False)[col]
                .diff()
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype("float32")
            )
        feature_cols = feature_cols + [
            "log_ret_1",
            "range_pct",
            "close_pos",
            "oc_ret",
            "d_log1p_volume",
            "d_log1p_amount",
            "d_log1p_oi",
        ]
        if feature_set == "ohlcv_factors":
            feature_cols = _add_ohlcv_factor_columns(out, feature_cols, keys)
        if feature_set in {"extended_market", "extended_market_lite"}:
            context_cols = (
                [
                    "log_ret_1",
                    "range_pct",
                    "close_pos",
                    "oc_ret",
                    "d_log1p_volume",
                    "d_log1p_amount",
                    "d_log1p_oi",
                ]
                if feature_set == "extended_market"
                else ["log_ret_1", "range_pct", "d_log1p_volume"]
            )
            for col in context_cols:
                g = out.groupby("datetime", sort=False)[col]
                mean = g.transform("mean").astype("float32")
                std = g.transform("std").replace(0.0, np.nan).fillna(0.0).astype("float32")
                rel = ((out[col] - mean) / (std + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                out[f"mkt_mean_{col}"] = mean
                out[f"mkt_std_{col}"] = std
                out[f"rel_{col}"] = rel.astype("float32")
                feature_cols.extend([f"mkt_mean_{col}", f"mkt_std_{col}", f"rel_{col}"])
    elif feature_set != "base":
        raise ValueError(f"unknown raw feature set: {feature_set}")
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    return out, feature_cols


def add_target_transform(df: pd.DataFrame, mode: str = "raw", min_xs_count: int = 8) -> tuple[pd.DataFrame, str]:
    mode = (mode or "raw").lower()
    out = df.copy()
    if mode == "raw":
        out["target"] = out["label"].astype("float32")
    elif mode == "xsz":
        g = out.groupby("datetime")["label"]
        count = g.transform("count")
        mu = g.transform("mean")
        sd = g.transform("std").replace(0.0, np.nan)
        target = ((out["label"] - mu) / (sd + 1e-8)).clip(-4.0, 4.0)
        target[count < int(min_xs_count)] = np.nan
        out["target"] = target.astype("float32")
    elif mode == "xrank":
        g = out.groupby("datetime")["label"]
        count = g.transform("count")
        target = g.rank(pct=True) - 0.5
        target[count < int(min_xs_count)] = np.nan
        out["target"] = target.astype("float32")
    elif mode in {"xrank_gauss", "rank_gaussian", "rank_gauss"}:
        g = out.groupby("datetime")["label"]
        count = g.transform("count")
        rank = g.rank(method="average")
        prob = ((rank - 0.5) / count).clip(1e-6, 1.0 - 1e-6)
        target = pd.Series(ndtri(prob.to_numpy(dtype=np.float64)), index=out.index, dtype="float64")
        target[count < int(min_xs_count)] = np.nan
        out["target"] = target.astype("float32")
    else:
        raise ValueError(f"unknown target mode: {mode}")
    return out, "target"


def add_auxiliary_targets(
    df: pd.DataFrame,
    aux_names: Iterable[str] | None,
    horizon: int = 30,
) -> tuple[pd.DataFrame, list[str]]:
    names = [str(name).strip() for name in (aux_names or []) if str(name).strip()]
    if not names:
        return df, []
    out = df.copy().sort_values(["symbol", "datetime"]).reset_index(drop=True)
    requested = []
    for name in names:
        if name not in requested:
            requested.append(name)
    label = pd.to_numeric(out["label"], errors="coerce").astype("float32")
    if "aux_raw_label" in requested:
        out["aux_raw_label"] = label.astype("float32")
    if "aux_abs_label" in requested:
        out["aux_abs_label"] = np.abs(label.to_numpy(dtype=np.float32)).astype("float32")
    if "aux_label_sq" in requested:
        vals = label.to_numpy(dtype=np.float32)
        out["aux_label_sq"] = (vals * vals).astype("float32")
    if "aux_xs_label_std" in requested:
        out["aux_xs_label_std"] = (
            out.groupby("datetime", sort=False)["label"].transform("std").replace([np.inf, -np.inf], np.nan).astype("float32")
        )
    if "aux_xs_abs_label_mean" in requested:
        out["aux_xs_abs_label_mean"] = (
            label.abs().groupby(out["datetime"], sort=False).transform("mean").replace([np.inf, -np.inf], np.nan).astype("float32")
        )
    if "aux_xsz_label" in requested:
        g = out.groupby("datetime", sort=False)["label"]
        count = g.transform("count")
        mu = g.transform("mean")
        sd = g.transform("std").replace(0.0, np.nan)
        vals = ((out["label"] - mu) / (sd + 1e-8)).clip(-4.0, 4.0)
        vals[count < 8] = np.nan
        out["aux_xsz_label"] = vals.astype("float32")
    if "aux_xrank_label" in requested:
        g = out.groupby("datetime", sort=False)["label"]
        count = g.transform("count")
        vals = g.rank(pct=True) - 0.5
        vals[count < 8] = np.nan
        out["aux_xrank_label"] = vals.astype("float32")
    if "aux_xrank_gauss_label" in requested:
        g = out.groupby("datetime", sort=False)["label"]
        count = g.transform("count")
        rank = g.rank(method="average")
        prob = ((rank - 0.5) / count).clip(1e-6, 1.0 - 1e-6)
        vals = pd.Series(ndtri(prob.to_numpy(dtype=np.float64)), index=out.index, dtype="float64")
        vals[count < 8] = np.nan
        out["aux_xrank_gauss_label"] = vals.astype("float32")
    needs_future_rv = "aux_future_rv" in requested
    needs_future_range = "aux_future_range" in requested
    if needs_future_rv or needs_future_range:
        if "log_ret_1" in out.columns:
            ret = pd.to_numeric(out["log_ret_1"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            close = np.clip(pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=np.float64), 1e-8, None)
            ret = pd.Series(np.log(close), index=out.index).groupby([out["symbol"], out["session_id"]], sort=False).diff()
            ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "range_pct" in out.columns:
            range_pct = pd.to_numeric(out["range_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            close = np.clip(pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=np.float64), 1e-8, None)
            high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=np.float64)
            low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=np.float64)
            range_pct = pd.Series((high - low) / (np.abs(close) + 1e-8), index=out.index).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        rv = np.full(len(out), np.nan, dtype=np.float32) if needs_future_rv else None
        future_range = np.full(len(out), np.nan, dtype=np.float32) if needs_future_range else None
        session_key = out["symbol"].astype(str) + "#" + out["session_id"].astype(str)
        for _, idx in pd.Series(np.arange(len(out))).groupby(session_key, sort=False):
            pos = idx.to_numpy()
            if len(pos) <= horizon:
                continue
            if needs_future_rv:
                vals = ret.iloc[pos].to_numpy(dtype=np.float64)
                sq = vals * vals
                csum = np.concatenate([[0.0], np.cumsum(sq)])
                sums = csum[horizon + 1 :] - csum[1 : len(pos) - horizon + 1]
                rv[pos[: len(sums)]] = np.sqrt(np.maximum(sums, 0.0)).astype(np.float32)
            if needs_future_range:
                vals = range_pct.iloc[pos].to_numpy(dtype=np.float64)
                csum = np.concatenate([[0.0], np.cumsum(vals)])
                sums = csum[horizon + 1 :] - csum[1 : len(pos) - horizon + 1]
                future_range[pos[: len(sums)]] = (sums / float(horizon)).astype(np.float32)
        if needs_future_rv:
            out["aux_future_rv"] = rv
        if needs_future_range:
            out["aux_future_range"] = future_range
    missing = [name for name in requested if name not in out.columns]
    if missing:
        raise ValueError(f"unknown auxiliary targets: {missing}")
    for name in requested:
        out[name] = pd.to_numeric(out[name], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    return out, requested


def prepare_symbol_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    aux_cols: list[str] | None = None,
) -> tuple[list[SymbolArrays], dict[str, int]]:
    arrays: list[SymbolArrays] = []
    symbols = sorted(df["symbol"].dropna().unique())
    symbol_to_id = {sym: i for i, sym in enumerate(symbols)}
    base = df.reset_index(drop=False).rename(columns={"index": "_source_index"})
    aux_cols = list(aux_cols or [])
    for sym in symbols:
        sdf = base[base["symbol"] == sym].sort_values("datetime").reset_index(drop=True)
        dt = pd.to_datetime(sdf["datetime"])
        aux_targets = sdf[aux_cols].to_numpy(dtype=np.float32) if aux_cols else None
        arrays.append(
            SymbolArrays(
                symbol=sym,
                features=sdf[feature_cols].to_numpy(dtype=np.float32),
                labels=sdf["label"].to_numpy(dtype=np.float32),
                targets=sdf[target_col].to_numpy(dtype=np.float32),
                aux_targets=aux_targets,
                datetimes=dt.to_numpy(dtype="datetime64[ns]"),
                minute_id=(dt.dt.hour * 60 + dt.dt.minute).to_numpy(dtype=np.int16),
                dayofweek_id=dt.dt.dayofweek.to_numpy(dtype=np.int8),
                month_id=(dt.dt.month - 1).to_numpy(dtype=np.int8),
                source_index=sdf["_source_index"].to_numpy(dtype=np.int64),
                session_pos=sdf.groupby("session_id", sort=False).cumcount().to_numpy(dtype=np.int32),
            )
        )
    return arrays, symbol_to_id


def build_window_index(
    arrays: list[SymbolArrays],
    seq_len: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    require_target: bool = True,
    allow_short: bool = False,
    anchor_stride: int = 1,
) -> list[tuple[int, int]]:
    lo = np.datetime64(pd.Timestamp(start))
    hi = np.datetime64(pd.Timestamp(end))
    min_pos = 0 if allow_short else seq_len - 1
    anchor_stride = max(1, int(anchor_stride))
    out: list[tuple[int, int]] = []
    for sym_id, arr in enumerate(arrays):
        mask = (arr.datetimes >= lo) & (arr.datetimes < hi) & (arr.session_pos >= min_pos)
        if anchor_stride > 1:
            mask &= (arr.session_pos % anchor_stride) == 0
        if require_target:
            mask &= np.isfinite(arr.targets)
        rows = np.flatnonzero(mask)
        out.extend((sym_id, int(row)) for row in rows)
    return out


def make_quarterly_splits(
    data_start: str | pd.Timestamp,
    data_end: str | pd.Timestamp,
    train_start: str | pd.Timestamp = "2018-01-01",
    first_test_start: str | pd.Timestamp = "2020-01-01",
    freq_months: int = 3,
    allow_partial_test: bool = True,
) -> list[RollingSplit]:
    data_start = pd.Timestamp(data_start)
    data_end = pd.Timestamp(data_end)
    train_start = max(pd.Timestamp(train_start), data_start)
    test_start = pd.Timestamp(first_test_start)
    splits: list[RollingSplit] = []
    q = 1
    while test_start < data_end:
        test_end = test_start + pd.DateOffset(months=freq_months)
        actual_end = min(test_end, data_end) if allow_partial_test else test_end
        if actual_end > test_start and train_start < test_start:
            year = test_start.year
            suffix = "partial" if actual_end < test_end else ""
            if freq_months == 3:
                quarter = ((test_start.month - 1) // 3) + 1
                name = f"{year}Q{quarter}{('_' + suffix) if suffix else ''}"
            else:
                name = f"{year}M{test_start.month:02d}{('_' + suffix) if suffix else ''}"
            splits.append(RollingSplit(name, train_start, test_start, test_start, actual_end))
        q += 1
        test_start = pd.Timestamp(first_test_start) + pd.DateOffset(months=freq_months * (q - 1))
    return splits


class WindowDataset:
    def __init__(
        self,
        arrays: list[SymbolArrays],
        index: list[tuple[int, int]],
        seq_len: int,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        allow_short: bool = False,
        target_clip: tuple[float | None, float | None] = (None, None),
        aux_mean: np.ndarray | None = None,
        aux_std: np.ndarray | None = None,
        aux_clip: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
    ) -> None:
        self.arrays = arrays
        self.index = index
        self.seq_len = int(seq_len)
        self.target_mean = float(target_mean)
        self.target_std = float(max(target_std, 1e-8))
        self.allow_short = bool(allow_short)
        self.target_clip = target_clip
        inferred_aux_dim = 0
        if aux_mean is None or aux_std is None:
            for arr in arrays:
                if arr.aux_targets is not None:
                    inferred_aux_dim = int(arr.aux_targets.shape[1])
                    break
        self.aux_mean = np.asarray(
            aux_mean if aux_mean is not None else np.zeros(inferred_aux_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.aux_std = np.maximum(
            np.asarray(aux_std if aux_std is not None else np.ones(inferred_aux_dim, dtype=np.float32), dtype=np.float32),
            1e-8,
        )
        self.aux_clip = aux_clip

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        import torch

        sym_id, row = self.index[idx]
        arr = self.arrays[sym_id]
        hist = min(int(arr.session_pos[row]), self.seq_len - 1)
        start = row - hist
        x = arr.features[start : row + 1].copy()
        if len(x) < self.seq_len:
            pad_len = self.seq_len - len(x)
            pad = np.zeros((pad_len, x.shape[1]), dtype=np.float32)
            x = np.concatenate([pad, x], axis=0)
        y_raw = float(arr.targets[row])
        lo, hi = self.target_clip
        if lo is not None and y_raw < lo:
            y_raw = float(lo)
        if hi is not None and y_raw > hi:
            y_raw = float(hi)
        y = (y_raw - self.target_mean) / self.target_std
        if arr.aux_targets is not None and arr.aux_targets.shape[1] > 0:
            aux_raw = arr.aux_targets[row].astype(np.float32, copy=True)
            aux_lo, aux_hi = self.aux_clip
            if aux_lo is not None:
                aux_raw = np.maximum(aux_raw, aux_lo.astype(np.float32))
            if aux_hi is not None:
                aux_raw = np.minimum(aux_raw, aux_hi.astype(np.float32))
            aux = (aux_raw - self.aux_mean) / self.aux_std
            aux = np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            aux = np.zeros((0,), dtype=np.float32)
        time_ids = np.array([arr.minute_id[row], arr.dayofweek_id[row], arr.month_id[row]], dtype=np.int64)
        group_id = np.asarray(arr.datetimes[row], dtype="datetime64[ns]").astype(np.int64).item()
        return (
            torch.from_numpy(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)),
            torch.tensor(sym_id, dtype=torch.long),
            torch.from_numpy(time_ids),
            torch.tensor(y, dtype=torch.float32),
            torch.from_numpy(aux),
            torch.tensor(group_id, dtype=torch.long),
        )


def collect_targets(arrays: list[SymbolArrays], index: list[tuple[int, int]]) -> np.ndarray:
    return np.array([arrays[s].targets[row] for s, row in index], dtype=np.float32)


def collect_aux_targets(arrays: list[SymbolArrays], index: list[tuple[int, int]]) -> np.ndarray:
    first = next((arr.aux_targets for arr in arrays if arr.aux_targets is not None), None)
    if first is None:
        return np.zeros((len(index), 0), dtype=np.float32)
    return np.stack([arrays[s].aux_targets[row] for s, row in index], axis=0).astype(np.float32)


def index_to_prediction_frame(
    arrays: list[SymbolArrays],
    index: list[tuple[int, int]],
    pred: np.ndarray,
) -> pd.DataFrame:
    records = []
    for (sym_id, row), pv in zip(index, pred):
        arr = arrays[sym_id]
        records.append(
            {
                "symbol": arr.symbol,
                "datetime": pd.Timestamp(arr.datetimes[row]),
                "label": float(arr.labels[row]),
                "target": float(arr.targets[row]),
                "pred": float(pv),
            }
        )
    return pd.DataFrame.from_records(records)
