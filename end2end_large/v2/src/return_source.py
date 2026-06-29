from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataLimitedReturnPolicy:
    feature_price_source: str
    label_price_source: str
    strict_tradable_label_available: bool
    label_status: str
    limitation: str


DATA_LIMITED_POLICY = DataLimitedReturnPolicy(
    feature_price_source="连续主力 synthetic continuous normalized OHLCV",
    label_price_source="continuous normalized close proxy return",
    strict_tradable_label_available=False,
    label_status="proxy_alpha_research_only",
    limitation=(
        "当前 CSV 缺少 contract_id、逐合约 raw close 和官方换月表；"
        "所有收益标签均为连续主力归一化序列上的 proxy return，不能解释为严格可交易 raw contract return。"
    ),
)


def raw_vwap_proxy(df: pd.DataFrame) -> pd.Series:
    """Return Amount / Volume as an unadjusted turnover-implied price proxy.

    The contract multiplier cancels in returns if it is constant for the product, but this proxy is
    VWAP-like rather than close-like and still lacks contract identity.
    """
    amount = pd.to_numeric(df["amount"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    out = amount / volume.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def proxy_log_return_from_close(
    df: pd.DataFrame,
    horizon: int = 30,
    price_col: str = "close",
    session_col: str = "session_id",
) -> pd.Series:
    """Continuous-normalized proxy forward log return inside each detected session."""
    return forward_log_return_within_session(df, pd.to_numeric(df[price_col], errors="coerce"), horizon, session_col)


def forward_log_return_within_session(
    df: pd.DataFrame,
    price: pd.Series,
    horizon: int = 30,
    session_col: str = "session_id",
) -> pd.Series:
    values = price.to_numpy(dtype=np.float64)
    labels = np.full(len(df), np.nan, dtype=np.float64)
    for _, idx in pd.Series(np.arange(len(df))).groupby(df[session_col].to_numpy(), sort=False):
        pos = idx.to_numpy(dtype=np.int64)
        if len(pos) <= horizon:
            continue
        cur = values[pos[:-horizon]]
        fut = values[pos[horizon:]]
        ok = np.isfinite(cur) & np.isfinite(fut) & (cur > 0) & (fut > 0)
        vals = np.full(len(cur), np.nan, dtype=np.float64)
        vals[ok] = np.log(fut[ok] / cur[ok])
        labels[pos[:-horizon]] = vals
    return pd.Series(labels, index=df.index, name=f"proxy_log_ret_{horizon}m")
