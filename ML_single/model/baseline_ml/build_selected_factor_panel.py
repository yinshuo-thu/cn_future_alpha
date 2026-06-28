#!/usr/bin/env python3
"""
Low-memory rebuild of the archived 1,144 selected factor panel.

The original full builder re-discovers factors and holds a very large TS panel
in memory.  In this environment that concat is killed, so this script rebuilds
the already archived selected_factors.txt exactly:
  1. compute per-symbol base factors from raw bars with official sessions/labels
  2. write month-partitioned chunks containing only needed raw bases and tsz views
  3. for each month, compute cross-sectional zscore/rank views and append the
     final selected panel

Final output is compatible with feature_model plan_a loaders:
  /root/shared-nvme/feature_model/data_factors_big.parquet
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "/root/feature_model")

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.labels import build_labels
from src.data.loader import get_symbols, load_config, load_symbol
from src.data.sessions import detect_sessions
from src.features.factor_lib import compute_symbol_factors

warnings.simplefilter("ignore", PerformanceWarning)

ROOT = Path("/root/autodl-tmp/quant")
OUT_DIR = Path("/root/shared-nvme/feature_model")
TMP_DIR = OUT_DIR / "selected_month_parts"
FINAL_PATH = OUT_DIR / "data_factors_big.parquet"
BUILDING_PATH = OUT_DIR / "data_factors_big.building.parquet"
SELECTED_SRC = ROOT / "artifacts" / "selected_factors.txt"
SELECTED_OUT = ROOT / "work" / "outputs" / "selected_factors.txt"

FACTOR_START = pd.Timestamp("2017-01-01")
PANEL_END_EXCL = pd.Timestamp("2021-01-01")
TSZ_WINDOW = 120
TSZ_MINP = 30

META_COLS = [
    "symbol",
    "datetime",
    "label",
    "is_long_break_before",
    "session_id",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "amount",
    "oi",
]


def parse_selected() -> tuple[list[str], dict[str, list[str]], list[str]]:
    selected = [x.strip() for x in SELECTED_SRC.read_text().splitlines() if x.strip()]
    by_view = {"raw": [], "tsz": [], "csz": [], "csr": []}
    needed_base: set[str] = set()
    for name in selected:
        for view in ("tsz", "csz", "csr"):
            prefix = view + "_"
            if name.startswith(prefix):
                base = name[len(prefix) :]
                by_view[view].append(base)
                needed_base.add(base)
                break
        else:
            by_view["raw"].append(name)
            needed_base.add(name)
    return selected, by_view, sorted(needed_base)


def tsz_frame(values: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=values.index)
    vals = values[cols].to_numpy(dtype=np.float32, copy=True)
    finite = np.isfinite(vals)
    clean = np.where(finite, vals, 0.0).astype(np.float64, copy=False)
    cnt = np.cumsum(finite.astype(np.float64), axis=0)
    s1 = np.cumsum(clean, axis=0)
    s2 = np.cumsum(clean * clean, axis=0)
    zeros = np.zeros((1, len(cols)), dtype=np.float64)
    cnt = np.vstack([zeros, cnt])
    s1 = np.vstack([zeros, s1])
    s2 = np.vstack([zeros, s2])
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - TSZ_WINDOW)
    nobs = cnt[end] - cnt[start]
    sums = s1[end] - s1[start]
    sumsq = s2[end] - s2[start]
    mean = sums / np.maximum(nobs, 1.0)
    var = (sumsq - sums * sums / np.maximum(nobs, 1.0)) / np.maximum(nobs - 1.0, 1.0)
    z = (clean - mean) / (np.sqrt(np.maximum(var, 0.0)) + 1e-8)
    z[(nobs < TSZ_MINP) | ~finite] = np.nan
    out = pd.DataFrame(z.astype(np.float32), index=values.index, columns=[f"tsz_{c}" for c in cols])
    return out


def make_symbol_frame(sym: str, cfg: dict, by_view: dict[str, list[str]], needed_base: list[str]) -> pd.DataFrame:
    raw = load_symbol(sym, cfg)
    raw = detect_sessions(raw)
    raw = build_labels(raw)
    factors = compute_symbol_factors(raw)
    keep = (factors["datetime"] >= FACTOR_START) & (factors["datetime"] < PANEL_END_EXCL)
    factors = factors.loc[keep].reset_index(drop=True)
    meta = raw.loc[(raw["datetime"] >= FACTOR_START) & (raw["datetime"] < PANEL_END_EXCL), META_COLS].copy()
    meta = meta.reset_index(drop=True)
    if len(meta) != len(factors):
        meta = factors[["symbol", "datetime", "label"]].merge(
            raw[META_COLS].drop(columns=["label"]),
            on=["symbol", "datetime"],
            how="left",
        )
    for col in needed_base:
        if col not in factors.columns:
            factors[col] = np.nan
        factors[col] = factors[col].astype(np.float32)

    raw_cols = list(dict.fromkeys(by_view["raw"]))
    cs_bases = sorted(set(by_view["csz"]) | set(by_view["csr"]))
    base_cols = list(dict.fromkeys(raw_cols + cs_bases))
    base_df = pd.DataFrame(
        {col: factors[col].to_numpy(np.float32) for col in base_cols},
        index=meta.index,
    )
    frames = [meta.copy(), base_df]
    if by_view["tsz"]:
        tsz = tsz_frame(factors, by_view["tsz"])
        frames.append(tsz.reset_index(drop=True))
    out = pd.concat(frames, axis=1).copy()
    for col in out.columns:
        if col not in ("symbol", "datetime", "is_long_break_before"):
            if pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].astype(np.float32)
    return out


def write_symbol_month_parts(sym_frame: pd.DataFrame, sym: str) -> None:
    sym_frame["_month"] = sym_frame["datetime"].dt.to_period("M").astype(str)
    for month, chunk in sym_frame.groupby("_month", sort=True):
        month_dir = TMP_DIR / f"month={month}"
        month_dir.mkdir(parents=True, exist_ok=True)
        path = month_dir / f"{sym}.parquet"
        chunk = chunk.drop(columns=["_month"]).copy()
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_table(table, path, compression="zstd")


def build_month_chunk(month_dir: Path, selected: list[str], by_view: dict[str, list[str]]) -> pd.DataFrame:
    panel = pd.read_parquet(month_dir)
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    frames = [panel[[c for c in META_COLS if c in panel.columns]].copy()]
    if by_view["raw"]:
        frames.append(panel[by_view["raw"]].astype(np.float32).reset_index(drop=True))
    if by_view["tsz"]:
        tsz_cols = [f"tsz_{base}" for base in by_view["tsz"]]
        frames.append(panel[tsz_cols].astype(np.float32).reset_index(drop=True))
    if by_view["csz"]:
        g = panel.groupby("datetime", sort=False)
        csz_data = {}
        for base in by_view["csz"]:
            mu = g[base].transform("mean")
            sd = g[base].transform("std")
            csz_data[f"csz_{base}"] = ((panel[base] - mu) / (sd + 1e-8)).astype(np.float32).to_numpy()
        frames.append(pd.DataFrame(csz_data, index=panel.index).reset_index(drop=True))
    if by_view["csr"]:
        csr_data = {}
        for base in by_view["csr"]:
            csr_data[f"csr_{base}"] = (
                panel.groupby("datetime", sort=False)[base].rank(pct=True) - 0.5
            ).astype(np.float32).to_numpy()
        frames.append(pd.DataFrame(csr_data, index=panel.index).reset_index(drop=True))
    out = pd.concat(frames, axis=1).copy()
    final_cols = [c for c in META_COLS if c in out.columns] + selected
    out = out[final_cols].sort_values(["symbol", "datetime"]).reset_index(drop=True)
    for col in selected:
        out[col] = out[col].astype(np.float32)
    return out


def main() -> None:
    cfg = load_config()
    selected, by_view, needed_base = parse_selected()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    if FINAL_PATH.exists():
        FINAL_PATH.unlink()
    if BUILDING_PATH.exists():
        BUILDING_PATH.unlink()
    SELECTED_OUT.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_OUT.write_text("\n".join(selected) + "\n", encoding="utf-8")

    print(
        f"[selected-build] selected={len(selected)} needed_base={len(needed_base)} "
        f"raw={len(by_view['raw'])} tsz={len(by_view['tsz'])} "
        f"csz={len(by_view['csz'])} csr={len(by_view['csr'])}",
        flush=True,
    )
    symbols = get_symbols(cfg)
    for i, sym in enumerate(symbols, 1):
        frame = make_symbol_frame(sym, cfg, by_view, needed_base)
        write_symbol_month_parts(frame, sym)
        print(f"  [symbol {i:02d}/{len(symbols)}] {sym} rows={len(frame)}", flush=True)
        del frame
        gc.collect()

    writer = None
    rows_written = 0
    month_dirs = sorted([p for p in TMP_DIR.iterdir() if p.is_dir()])
    for i, month_dir in enumerate(month_dirs, 1):
        chunk = build_month_chunk(month_dir, selected, by_view)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(BUILDING_PATH, table.schema, compression="zstd")
        writer.write_table(table)
        rows_written += len(chunk)
        print(f"  [month {i:02d}/{len(month_dirs)}] {month_dir.name} rows={len(chunk)} total={rows_written}", flush=True)
        del chunk, table
        shutil.rmtree(month_dir)
        gc.collect()
    if writer is not None:
        writer.close()
    if not BUILDING_PATH.exists() or BUILDING_PATH.stat().st_size <= 0:
        raise RuntimeError(f"building parquet missing after writer close: {BUILDING_PATH}")
    os.replace(BUILDING_PATH, FINAL_PATH)
    if not FINAL_PATH.exists() or FINAL_PATH.stat().st_size <= 0:
        raise RuntimeError(f"final parquet missing after rename: {FINAL_PATH}")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    if not FINAL_PATH.exists() or FINAL_PATH.stat().st_size <= 0:
        raise RuntimeError(f"final parquet disappeared after temp cleanup: {FINAL_PATH}")
    print(f"[selected-build] wrote {FINAL_PATH} rows={rows_written} factors={len(selected)}", flush=True)


if __name__ == "__main__":
    main()
