from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RAW_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "oi"]
DEFAULT_EXCLUDED_SYMBOLS = ["T", "TF", "TS", "IF", "IC", "IH"]
LONG_BREAK_MIN = 60


def discover_symbols(
    data_dir: str | Path,
    excluded_symbols: Iterable[str] = DEFAULT_EXCLUDED_SYMBOLS,
    symbols: Iterable[str] | None = None,
    max_symbols: int | None = None,
) -> list[str]:
    data_dir = Path(data_dir)
    excluded = {s.upper() for s in excluded_symbols}
    if symbols:
        out = [s.strip().upper() for s in symbols if s.strip().upper() not in excluded]
    else:
        out = sorted(p.stem.upper() for p in data_dir.glob("*.csv") if p.stem.upper() not in excluded)
    return out[: int(max_symbols)] if max_symbols else out


def load_sector_map(path: str | Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current = line[:-1].strip().upper()
            mapping[current] = {}
            continue
        if current and ":" in line:
            key, value = line.strip().split(":", 1)
            mapping[current][key.strip()] = value.strip()
    return mapping


def load_symbol_csv(data_dir: str | Path, symbol: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{symbol}.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    time_col = "time" if "time" in df.columns else df.columns[0]
    df = df.rename(columns={time_col: "datetime", "open_interest": "oi", "openinterest": "oi"})
    df["datetime"] = pd.to_datetime(
        df["datetime"].astype(str).str.slice(0, 16),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["datetime"] + RAW_COLUMNS].dropna(subset=["datetime"])
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["symbol"] = symbol.upper()
    return df[["symbol", "datetime"] + RAW_COLUMNS]


def add_session_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "datetime"]).reset_index(drop=True).copy()
    out["gap_min"] = out.groupby("symbol", sort=False)["datetime"].diff().dt.total_seconds().div(60).fillna(0.0)
    out["is_long_break_before"] = out["gap_min"] >= LONG_BREAK_MIN
    out["session_id"] = out.groupby("symbol", sort=False)["is_long_break_before"].cumsum().astype("int32")
    out["session_pos"] = out.groupby(["symbol", "session_id"], sort=False).cumcount().astype("int32")
    hours = out["datetime"].dt.hour
    out["session_type"] = np.where((hours >= 20) | (hours < 8), "night", "day")
    out["minute_of_day"] = (out["datetime"].dt.hour * 60 + out["datetime"].dt.minute).astype("int16")
    out["month"] = out["datetime"].dt.month.astype("int8")
    return out


def load_all_symbols(
    data_dir: str | Path,
    sector_map_path: str | Path,
    start_date: str = "2017-01-01",
    end_date: str = "2021-01-01",
    symbols: Iterable[str] | None = None,
    excluded_symbols: Iterable[str] = DEFAULT_EXCLUDED_SYMBOLS,
    max_symbols: int | None = None,
    warmup_days: int = 7,
) -> pd.DataFrame:
    sector_map = load_sector_map(sector_map_path)
    start = pd.Timestamp(start_date) - pd.Timedelta(days=int(warmup_days))
    end = pd.Timestamp(end_date)
    frames = []
    for symbol in discover_symbols(data_dir, excluded_symbols, symbols, max_symbols):
        raw = load_symbol_csv(data_dir, symbol)
        raw = raw[(raw["datetime"] >= start) & (raw["datetime"] < end)].copy()
        if raw.empty:
            continue
        meta = sector_map.get(symbol, {})
        raw["sector"] = meta.get("sector", "unknown")
        raw["exchange"] = meta.get("exchange", "unknown")
        frames.append(raw)
    if not frames:
        raise ValueError("no symbols loaded")
    out = pd.concat(frames, ignore_index=True)
    out = add_session_metadata(out)
    out = out[out["datetime"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    return out
