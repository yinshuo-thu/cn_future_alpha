#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.labels import build_labels
from src.data.loader import get_symbols, load_config, load_symbol
from src.data.sessions import LONG_BREAK_THRESH_MIN, detect_sessions
from src.evaluation.metrics import compute_ic, sanity_check


AUDIT_SYMBOLS = ["A", "AG", "AP", "SC", "UR", "ZN"]


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def synthetic_label_check():
    # 09:00-10:14, short break, 10:30-11:30. The 10:00 label should land on
    # 10:45 when the 10:15-10:29 missing minutes are skipped by row counting.
    dts = list(pd.date_range("2020-01-02 09:00", "2020-01-02 10:14", freq="min"))
    dts += list(pd.date_range("2020-01-02 10:30", "2020-01-02 11:30", freq="min"))
    df = pd.DataFrame(
        {
            "symbol": "X",
            "datetime": dts,
            "open": np.arange(1, len(dts) + 1, dtype=float),
            "high": np.arange(1, len(dts) + 1, dtype=float),
            "low": np.arange(1, len(dts) + 1, dtype=float),
            "close": np.arange(1, len(dts) + 1, dtype=float),
            "volume": 1.0,
            "amount": 1.0,
            "oi": 1.0,
        }
    )
    out = build_labels(detect_sessions(df), horizon=30)
    i = out.index[out["datetime"].eq(pd.Timestamp("2020-01-02 10:00"))][0]
    j = i + 30
    _assert(out.loc[j, "datetime"] == pd.Timestamp("2020-01-02 10:45"), "short break target did not chain to 10:45")
    expected = out.loc[j, "close"] / out.loc[i, "close"] - 1.0
    _assert(abs(out.loc[i, "label"] - expected) < 1e-12, "short break chained label mismatch")

    # Add a long break. Last 30 rows before the long break must be NaN.
    dts2 = list(pd.date_range("2020-01-03 09:00", "2020-01-03 10:59", freq="min"))
    dts2 += list(pd.date_range("2020-01-03 21:00", "2020-01-03 21:40", freq="min"))
    df2 = pd.DataFrame(
        {
            "symbol": "Y",
            "datetime": dts2,
            "open": np.arange(1, len(dts2) + 1, dtype=float),
            "high": np.arange(1, len(dts2) + 1, dtype=float),
            "low": np.arange(1, len(dts2) + 1, dtype=float),
            "close": np.arange(1, len(dts2) + 1, dtype=float),
            "volume": 1.0,
            "amount": 1.0,
            "oi": 1.0,
        }
    )
    out2 = build_labels(detect_sessions(df2), horizon=30)
    lb_idx = int(np.where(out2["is_long_break_before"].to_numpy())[0][0])
    _assert(out2.loc[lb_idx - 30 : lb_idx - 1, "label"].isna().all(), "long-break pre-window labels are not all NaN")
    _assert(math.isnan(out2.loc[lb_idx - 1, "label"]), "last bar before long break has non-NaN label")


def inspect_symbol(symbol, cfg, deep=False):
    df = load_symbol(symbol, cfg)
    required = {"symbol", "datetime", "open", "high", "low", "close", "volume", "amount", "oi"}
    _assert(required.issubset(df.columns), f"{symbol}: missing required columns {required - set(df.columns)}")
    _assert(df["datetime"].is_monotonic_increasing, f"{symbol}: datetime not monotonic")
    _assert(not df.duplicated(["symbol", "datetime"]).any(), f"{symbol}: duplicate symbol/datetime after load")
    _assert((df["datetime"].dt.second == 0).all(), f"{symbol}: seconds not floored to minute")
    _assert((df["datetime"].dt.microsecond == 0).all(), f"{symbol}: microseconds not floored to minute")
    _assert(df["close"].notna().mean() > 0.5, f"{symbol}: close coverage too low")
    _assert(df["amount"].notna().any(), f"{symbol}: amount is entirely missing")
    _assert(df["oi"].notna().any(), f"{symbol}: oi is entirely missing")

    stats = {
        "rows": int(len(df)),
        "start": str(df["datetime"].min()),
        "end": str(df["datetime"].max()),
        "amount_non_null": float(df["amount"].notna().mean()),
        "oi_non_null": float(df["oi"].notna().mean()),
        "close_non_null": float(df["close"].notna().mean()),
    }
    if deep:
        sess = detect_sessions(df)
        labels = build_labels(sess)
        gaps = sess["gap_min"]
        long_gaps = gaps[gaps >= LONG_BREAK_THRESH_MIN]
        short_gaps = gaps[(gaps > 1) & (gaps < LONG_BREAK_THRESH_MIN)]
        pre_long = labels["bars_to_next_long_break"].lt(30)
        _assert(labels.loc[pre_long, "label"].isna().all(), f"{symbol}: labels within 30 bars of long break are not all NaN")
        if len(labels) >= 30:
            _assert(labels.tail(30)["label"].isna().all(), f"{symbol}: final 30 labels are not all NaN")
        rnd = np.random.default_rng(42).standard_normal(len(labels))
        random_ic = compute_ic(rnd, labels["label"].to_numpy())
        _assert(abs(random_ic) < 0.02, f"{symbol}: random-label IC too large ({random_ic})")
        stats.update(
            {
                "sessions": int(sess["session_id"].nunique()),
                "long_breaks": int(len(long_gaps)),
                "short_breaks": int(len(short_gaps)),
                "max_gap_min": float(gaps.max()),
                "label_coverage": float(labels["label"].notna().mean()),
                "random_pred_ic": float(random_ic),
            }
        )
    return stats


def main():
    cfg = load_config()
    symbols = get_symbols(cfg)
    _assert(len(symbols) == 51, f"expected 51 tradable symbols, got {len(symbols)}")
    _assert(not (set(symbols) & set(cfg["excluded_symbols"])), "excluded symbols leaked into universe")

    sanity_check()
    synthetic_label_check()

    all_stats = {}
    for symbol in symbols:
        all_stats[symbol] = inspect_symbol(symbol, cfg, deep=symbol in AUDIT_SYMBOLS)

    audit = {s: all_stats[s] for s in AUDIT_SYMBOLS if s in all_stats}
    result = {
        "symbols": symbols,
        "row_count": int(sum(v["rows"] for v in all_stats.values())),
        "date_start": min(v["start"] for v in all_stats.values()),
        "date_end": max(v["end"] for v in all_stats.values()),
        "audit_symbols": audit,
    }

    out_dir = Path(cfg["reports_dir"])
    journal_dir = out_dir / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Phase 1-5 Data Foundation Check",
        "",
        "Status: PASS",
        "",
        "## Scope",
        f"- Tradable symbols: {len(symbols)}",
        f"- Total loaded rows after exclusion: {result['row_count']:,}",
        f"- Date range: {result['date_start']} to {result['date_end']}",
        f"- Excluded symbols: {', '.join(cfg['excluded_symbols'])}",
        "",
        "## Checks",
        "- Loader floors timestamps to integer minute, normalizes amount/OI, drops duplicate symbol-minute rows.",
        "- Universe is 51 symbols and excludes T/TF/TS/IF/IC/IH.",
        "- Evaluator sanity passed: perfect IC ~1, random IC ~0, flipped IC ~-1.",
        "- Synthetic label test passed: short breaks are chained by row count and long-break pre-windows are NaN.",
        "- Real symbol audits passed for session detection, long-break NaN labels, final horizon NaNs, random-pred IC near zero.",
        "",
        "## Audited Symbols",
        "",
        "| symbol | rows | close cov | sessions | long breaks | short breaks | max gap min | label coverage | random IC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for symbol, st in audit.items():
        lines.append(
            f"| {symbol} | {st['rows']:,} | {st['close_non_null']:.4f} | {st['sessions']:,} | {st['long_breaks']:,} | "
            f"{st['short_breaks']:,} | {st['max_gap_min']:.0f} | {st['label_coverage']:.4f} | {st['random_pred_ic']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- `/root/feature_model` currently has no `.git` directory in this runtime, so commit/push cannot be performed until the repository metadata or remote checkout is restored.",
            "- Stage 2 should build the large factor parquet under `/root/shared-nvme/feature_model/` with float32 and per-symbol filtering.",
            "",
            "```json",
            json.dumps(result, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    (out_dir / "phase1-5_check.md").write_text("\n".join(lines), encoding="utf-8")

    journal = [
        "# Journal - Phase 1-5",
        "",
        "- Read Jump official requirements, integrated plan, and factor reference.",
        "- Confirmed raw data has 57 csv files and configured tradable universe has 51 symbols.",
        "- Ran loader/session/label/evaluator checks via `python3 run/phase1_5_check.py`.",
        "- Phase 1-5 data foundation gate passed; next priority is large factor build and LightGBM baseline.",
        "",
    ]
    (journal_dir / "phase1_5.md").write_text("\n".join(journal), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
