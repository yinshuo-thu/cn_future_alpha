#!/usr/bin/env python3
"""
Low-memory strict monthly LightGBM ablations.

This runner avoids loading the whole wide factor panel at once.  It reads one
calendar month at a time, caches event-driven training samples, then trains each
rolling model only from historical cached samples and predicts the next month.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")
sys.path.insert(0, "/root/feature_model")

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import ndtri

from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, period_ic
from src.plan_a.group_lgb import symbol_group_map
from strict_optimization_ablation import (
    FACTOR_PATH,
    FIG_DIR,
    META_COLS,
    OUT_DIR,
    PRED_START,
    REGIME_COLS,
    TEST_END,
    TEST_START,
    TRAIN_START,
    selected_features,
    summarize,
)


CACHE_ROOT = OUT_DIR / "chunked_cache"
SYMBOL_MAP_PATH = OUT_DIR / "chunked_symbol_maps.json"
FACTOR_CATALOG_PATH = Path("/root/autodl-tmp/quant/artifacts/factor_catalog.csv")
SELECTED_2018_IC_PATH = OUT_DIR / "selected_factors_2018_ic.txt"


@dataclass(frozen=True)
class ChunkVariant:
    name: str
    cache_name: str = ""
    top_n: int = 300
    feature_mode: str = "top"
    model_type: str = "regressor"
    target_mode: str = "xsz"
    sample_mode: str = "soft_event"
    max_rows: int = 430_000
    month_sample_rows: int = 24_000
    seed: int = 2027
    n_estimators: int = 280
    learning_rate: float = 0.035
    num_leaves: int = 63
    min_child_samples: int = 120
    reg_lambda: float = 6.0
    colsample_bytree: float = 0.68
    subsample: float = 0.82
    lookback_months: int = 0
    embargo_bars: int = 0
    group_expert: bool = False


VARIANTS = [
    ChunkVariant(
        name="chunk_t80_xsz_softevent_micro",
        top_n=80,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=100_000,
        month_sample_rows=5_000,
        seed=381,
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=180,
        colsample_bytree=0.86,
        reg_lambda=14.0,
    ),
    ChunkVariant(
        name="chunk_t150_xsz_softevent_light",
        top_n=150,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=240_000,
        month_sample_rows=12_000,
        seed=401,
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=220,
        colsample_bytree=0.78,
        reg_lambda=12.0,
    ),
    ChunkVariant(
        name="chunk_t220_xsz_softevent_mid",
        top_n=220,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=260_000,
        month_sample_rows=14_500,
        seed=451,
        n_estimators=210,
        learning_rate=0.04,
        num_leaves=39,
        min_child_samples=200,
        colsample_bytree=0.72,
        subsample=0.82,
        reg_lambda=10.0,
    ),
    ChunkVariant(
        name="chunk_t220_xsz_softevent_fast_lb12",
        cache_name="chunk_t220_xsz_softevent_mid",
        top_n=220,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=154_000,
        month_sample_rows=14_500,
        seed=455,
        n_estimators=120,
        learning_rate=0.055,
        num_leaves=31,
        min_child_samples=250,
        colsample_bytree=0.74,
        subsample=0.82,
        reg_lambda=13.0,
        lookback_months=12,
    ),
    ChunkVariant(
        name="chunk_t180_xsz_softevent_midlight",
        top_n=180,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=210_000,
        month_sample_rows=12_000,
        seed=431,
        n_estimators=190,
        learning_rate=0.04,
        num_leaves=35,
        min_child_samples=210,
        colsample_bytree=0.76,
        subsample=0.82,
        reg_lambda=11.0,
    ),
    ChunkVariant(
        name="chunk_t180_xsz_softevent_fast",
        cache_name="chunk_t180_xsz_softevent_midlight",
        top_n=180,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=168_000,
        month_sample_rows=12_000,
        seed=431,
        n_estimators=115,
        learning_rate=0.055,
        num_leaves=31,
        min_child_samples=240,
        colsample_bytree=0.78,
        subsample=0.82,
        reg_lambda=13.0,
    ),
    ChunkVariant(
        name="chunk_t180_xsz_softevent_fast_lb12",
        cache_name="chunk_t180_xsz_softevent_midlight",
        top_n=180,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=144_000,
        month_sample_rows=12_000,
        seed=433,
        n_estimators=115,
        learning_rate=0.055,
        num_leaves=31,
        min_child_samples=240,
        colsample_bytree=0.78,
        subsample=0.82,
        reg_lambda=13.0,
        lookback_months=12,
    ),
    ChunkVariant(
        name="chunk_t180_xsz_softevent_fast_lb12_divoi",
        top_n=180,
        feature_mode="top120_oi_vol_tail",
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=132_000,
        month_sample_rows=10_000,
        seed=447,
        n_estimators=105,
        learning_rate=0.058,
        num_leaves=31,
        min_child_samples=260,
        colsample_bytree=0.78,
        subsample=0.82,
        reg_lambda=14.0,
        lookback_months=12,
    ),
    ChunkVariant(
        name="chunk_t180_xsz_event50_lb12",
        top_n=180,
        target_mode="xsz",
        sample_mode="event50",
        max_rows=144_000,
        month_sample_rows=12_000,
        seed=449,
        n_estimators=115,
        learning_rate=0.055,
        num_leaves=31,
        min_child_samples=240,
        colsample_bytree=0.78,
        subsample=0.82,
        reg_lambda=13.0,
        lookback_months=12,
    ),
    ChunkVariant(
        name="chunk_t180_ic2018_xsz_event50_lb12",
        top_n=180,
        feature_mode="ic2018_top",
        target_mode="xsz",
        sample_mode="event50",
        max_rows=144_000,
        month_sample_rows=12_000,
        seed=461,
        n_estimators=115,
        learning_rate=0.055,
        num_leaves=31,
        min_child_samples=240,
        colsample_bytree=0.78,
        subsample=0.82,
        reg_lambda=13.0,
        lookback_months=12,
    ),
    ChunkVariant(
        name="chunk_t500_xsz_random_shallow_lb18",
        top_n=500,
        target_mode="xsz",
        sample_mode="random",
        max_rows=330_000,
        month_sample_rows=22_000,
        seed=471,
        n_estimators=220,
        learning_rate=0.040,
        num_leaves=31,
        min_child_samples=280,
        colsample_bytree=0.55,
        subsample=0.82,
        reg_lambda=14.0,
        lookback_months=18,
    ),
    ChunkVariant(
        name="chunk_t500_ic2018_xsz_random_shallow_lb18",
        top_n=500,
        feature_mode="ic2018_top",
        target_mode="xsz",
        sample_mode="random",
        max_rows=330_000,
        month_sample_rows=22_000,
        seed=473,
        n_estimators=220,
        learning_rate=0.040,
        num_leaves=31,
        min_child_samples=280,
        colsample_bytree=0.55,
        subsample=0.82,
        reg_lambda=14.0,
        lookback_months=18,
    ),
    ChunkVariant(
        name="chunk_t300_xsz_eventtail_lb18",
        top_n=300,
        target_mode="xsz",
        sample_mode="event_tail",
        max_rows=260_000,
        month_sample_rows=18_000,
        seed=481,
        n_estimators=180,
        learning_rate=0.045,
        num_leaves=47,
        min_child_samples=180,
        colsample_bytree=0.68,
        subsample=0.82,
        reg_lambda=10.0,
        lookback_months=18,
    ),
    ChunkVariant(
        name="chunk_t300_ic2018_xsz_eventtail_lb18",
        top_n=300,
        feature_mode="ic2018_top",
        target_mode="xsz",
        sample_mode="event_tail",
        max_rows=260_000,
        month_sample_rows=18_000,
        seed=483,
        n_estimators=180,
        learning_rate=0.045,
        num_leaves=47,
        min_child_samples=180,
        colsample_bytree=0.68,
        subsample=0.82,
        reg_lambda=10.0,
        lookback_months=18,
    ),
    ChunkVariant(
        name="chunk_t300_lambdarank_event50_lb18",
        top_n=300,
        model_type="ranker",
        target_mode="xrank",
        sample_mode="event50",
        max_rows=240_000,
        month_sample_rows=16_000,
        seed=491,
        n_estimators=120,
        learning_rate=0.050,
        num_leaves=31,
        min_child_samples=160,
        colsample_bytree=0.70,
        subsample=0.82,
        reg_lambda=8.0,
        lookback_months=18,
    ),
    ChunkVariant(
        name="chunk_t150_lambdarank_event50_lb18",
        top_n=150,
        model_type="ranker",
        target_mode="xrank",
        sample_mode="event50",
        max_rows=120_000,
        month_sample_rows=9_000,
        seed=493,
        n_estimators=90,
        learning_rate=0.060,
        num_leaves=23,
        min_child_samples=220,
        colsample_bytree=0.78,
        subsample=0.82,
        reg_lambda=12.0,
        lookback_months=18,
    ),
    ChunkVariant(
        name="chunk_group_t300_xsz_softevent_lb18",
        top_n=300,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=360_000,
        month_sample_rows=24_000,
        seed=495,
        n_estimators=180,
        learning_rate=0.045,
        num_leaves=39,
        min_child_samples=90,
        colsample_bytree=0.70,
        subsample=0.84,
        reg_lambda=8.0,
        lookback_months=18,
        group_expert=True,
    ),
    ChunkVariant(
        name="chunk_group_t300_xsz_event50_lb18",
        top_n=300,
        target_mode="xsz",
        sample_mode="event50",
        max_rows=360_000,
        month_sample_rows=24_000,
        seed=497,
        n_estimators=180,
        learning_rate=0.045,
        num_leaves=39,
        min_child_samples=90,
        colsample_bytree=0.70,
        subsample=0.84,
        reg_lambda=8.0,
        lookback_months=18,
        group_expert=True,
    ),
    ChunkVariant(
        name="chunk_group_t220_xsz_softevent_fast_lb18",
        top_n=220,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=180_000,
        month_sample_rows=12_000,
        seed=499,
        n_estimators=95,
        learning_rate=0.060,
        num_leaves=31,
        min_child_samples=140,
        colsample_bytree=0.74,
        subsample=0.84,
        reg_lambda=12.0,
        lookback_months=18,
        group_expert=True,
    ),
    ChunkVariant(
        name="chunk_t180_xsz_softevent_fast_lb12_s2",
        cache_name="chunk_t180_xsz_softevent_midlight",
        top_n=180,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=144_000,
        month_sample_rows=12_000,
        seed=439,
        n_estimators=145,
        learning_rate=0.045,
        num_leaves=39,
        min_child_samples=220,
        colsample_bytree=0.72,
        subsample=0.82,
        reg_lambda=11.0,
        lookback_months=12,
    ),
    ChunkVariant(
        name="chunk_t180_xsz_softevent_fast_lb12_s3",
        cache_name="chunk_t180_xsz_softevent_midlight",
        top_n=180,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=132_000,
        month_sample_rows=12_000,
        seed=443,
        n_estimators=95,
        learning_rate=0.065,
        num_leaves=31,
        min_child_samples=280,
        colsample_bytree=0.84,
        subsample=0.82,
        reg_lambda=16.0,
        lookback_months=12,
    ),
    ChunkVariant(
        name="chunk_t180_xsz_softevent_fast_lb6",
        cache_name="chunk_t180_xsz_softevent_midlight",
        top_n=180,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=96_000,
        month_sample_rows=12_000,
        seed=435,
        n_estimators=105,
        learning_rate=0.06,
        num_leaves=31,
        min_child_samples=260,
        colsample_bytree=0.80,
        subsample=0.82,
        reg_lambda=15.0,
        lookback_months=6,
    ),
    ChunkVariant(
        name="chunk_t180_xsz_softevent_fast_lb18",
        cache_name="chunk_t180_xsz_softevent_midlight",
        top_n=180,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=168_000,
        month_sample_rows=12_000,
        seed=437,
        n_estimators=115,
        learning_rate=0.055,
        num_leaves=31,
        min_child_samples=240,
        colsample_bytree=0.78,
        subsample=0.82,
        reg_lambda=13.0,
        lookback_months=18,
    ),
    ChunkVariant(
        name="chunk_t300_xsz_softevent_assist",
        top_n=300,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=430_000,
        month_sample_rows=24_000,
        seed=501,
        colsample_bytree=0.64,
        reg_lambda=7.0,
    ),
    ChunkVariant(
        name="chunk_t300_xsz_softevent_assist_emb30",
        top_n=300,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=430_000,
        month_sample_rows=24_000,
        seed=501,
        colsample_bytree=0.64,
        reg_lambda=7.0,
        embargo_bars=30,
    ),
    ChunkVariant(
        name="chunk_t500_xsz_softevent_assist",
        top_n=500,
        target_mode="xsz",
        sample_mode="soft_event",
        max_rows=430_000,
        month_sample_rows=22_000,
        seed=503,
        colsample_bytree=0.54,
        reg_lambda=8.0,
    ),
    ChunkVariant(
        name="chunk_t300_ranknorm_softevent_assist",
        top_n=300,
        target_mode="ranknorm",
        sample_mode="soft_event",
        max_rows=430_000,
        month_sample_rows=24_000,
        seed=509,
        colsample_bytree=0.64,
        reg_lambda=9.0,
    ),
]


def parse_variants() -> list[ChunkVariant]:
    wanted = os.environ.get("CHUNKED_VARIANTS", "").strip()
    if not wanted:
        return VARIANTS
    names = {x.strip() for x in wanted.split(",") if x.strip()}
    out = [v for v in VARIANTS if v.name in names]
    missing = names - {v.name for v in out}
    if missing:
        raise ValueError(f"unknown chunked variants: {sorted(missing)}")
    return out


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def write_parquet_atomic(data: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        data.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def cache_dir_for(variant: ChunkVariant) -> Path:
    return CACHE_ROOT / (variant.cache_name or variant.name) / "samples"


def feature_cols_for(variant: ChunkVariant) -> list[str]:
    if variant.feature_mode == "top":
        return selected_features(variant.top_n)
    if variant.feature_mode == "ic2018_top":
        if not SELECTED_2018_IC_PATH.exists():
            raise FileNotFoundError(f"missing {SELECTED_2018_IC_PATH}; run select_factors_2018_ic.py first")
        selected = [x.strip() for x in SELECTED_2018_IC_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
        return selected[: variant.top_n]
    selected_all = selected_features(10_000)
    if variant.feature_mode == "top120_oi_vol_tail":
        catalog = pd.read_csv(FACTOR_CATALOG_PATH).set_index("factor") if FACTOR_CATALOG_PATH.exists() else pd.DataFrame()
        keep_families = {"open-interest", "volume/price", "volatility", "volatility/range", "range"}
        head = selected_all[: min(120, variant.top_n)]
        out = list(dict.fromkeys(head))
        for feat in selected_all:
            if feat in out or feat not in catalog.index:
                continue
            if str(catalog.loc[feat, "family"]) in keep_families:
                out.append(feat)
                if len(out) >= variant.top_n:
                    break
        for feat in selected_all:
            if len(out) >= variant.top_n:
                break
            if feat not in out:
                out.append(feat)
        return out[: variant.top_n]
    raise ValueError(f"unknown feature_mode={variant.feature_mode!r}")


def month_code(dt: pd.Series) -> np.ndarray:
    return (dt.dt.year.to_numpy(dtype=np.int16) * 12 + dt.dt.month.to_numpy(dtype=np.int16)).astype(np.int32)


def xsec_z(data: pd.DataFrame, col: str) -> pd.Series:
    g = data.groupby("datetime", sort=False)[col]
    return ((data[col] - g.transform("mean")) / (g.transform("std") + 1e-9)).astype(np.float32)


def all_months() -> list[pd.Timestamp]:
    return list(pd.date_range(TRAIN_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS"))


def prediction_months() -> list[pd.Timestamp]:
    return list(pd.date_range(PRED_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS"))


def symbol_maps() -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    if SYMBOL_MAP_PATH.exists():
        payload = json.loads(SYMBOL_MAP_PATH.read_text(encoding="utf-8"))
        return payload["sym_map"], payload["grp_map"], payload["groups"]
    pieces = []
    for ms in all_months():
        next_ms = ms + pd.DateOffset(months=1)
        symbols = pd.read_parquet(
            FACTOR_PATH,
            columns=["symbol"],
            filters=[("datetime", ">=", ms), ("datetime", "<", next_ms)],
        )["symbol"].drop_duplicates()
        pieces.append(symbols.astype(str))
    sym_names = sorted(pd.concat(pieces, ignore_index=True).drop_duplicates().tolist())
    groups = symbol_group_map()
    group_names = sorted(set(groups.values()) | {"other"})
    sym_map = {s: i for i, s in enumerate(sym_names)}
    grp_map = {g: i for i, g in enumerate(group_names)}
    SYMBOL_MAP_PATH.write_text(
        json.dumps({"sym_map": sym_map, "grp_map": grp_map, "groups": groups}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return sym_map, grp_map, groups


def read_month(ms: pd.Timestamp, feat_cols: list[str]) -> pd.DataFrame:
    next_ms = ms + pd.DateOffset(months=1)
    cols = list(dict.fromkeys(META_COLS + feat_cols))
    data = pd.read_parquet(
        FACTOR_PATH,
        columns=cols,
        filters=[("datetime", ">=", ms), ("datetime", "<", next_ms)],
    )
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    data["_month_code"] = month_code(data["datetime"])
    return data


def add_context_targets(
    data: pd.DataFrame,
    sym_map: dict[str, int],
    grp_map: dict[str, int],
    groups: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    data["group"] = data["symbol"].map(groups).fillna("other")
    data["_bars_to_month_end"] = data.groupby("symbol", sort=False).cumcount(ascending=False).astype(np.int16)
    data["symbol_code"] = data["symbol"].map(sym_map).fillna(-1).astype(np.int16)
    data["group_code"] = data["group"].map(grp_map).fillna(grp_map.get("other", 0)).astype(np.int8)

    minute = (data["datetime"].dt.hour * 60 + data["datetime"].dt.minute).astype(np.float32)
    data["minute_sin"] = np.sin(2 * np.pi * minute / 1440.0).astype(np.float32)
    data["minute_cos"] = np.cos(2 * np.pi * minute / 1440.0).astype(np.float32)
    dow = data["datetime"].dt.dayofweek.astype(np.float32)
    data["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    data["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    month = data["datetime"].dt.month.astype(np.float32)
    data["month_sin"] = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
    data["month_cos"] = np.cos(2 * np.pi * month / 12.0).astype(np.float32)
    data["session_pos"] = data.groupby(["symbol", "session_id"], sort=False).cumcount().astype(np.float32)
    data["session_pos"] = (data["session_pos"] / 48.0).clip(0, 8).astype(np.float32)

    close = data["close"].astype(np.float64)
    open_ = data["open"].astype(np.float64)
    high = data["high"].astype(np.float64)
    low = data["low"].astype(np.float64)
    ret1 = np.log(close / close.groupby(data["symbol"], sort=False).shift(1).clip(lower=1e-12))
    ret1 = ret1.mask(data["is_long_break_before"].fillna(False), 0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data["ret1"] = ret1.astype(np.float32)
    data["intrabar_ret"] = np.log(close.clip(lower=1e-12) / open_.clip(lower=1e-12)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0).astype(np.float32)
    data["range_rel"] = ((high - low) / close.abs().clip(lower=1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    data["body_rel"] = ((close - open_) / open_.abs().clip(lower=1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    data["log_volume"] = np.log1p(data["volume"].clip(lower=0)).astype(np.float32)
    data["log_amount"] = np.log1p(data["amount"].clip(lower=0)).astype(np.float32)
    oi_log = np.log1p(data["oi"].clip(lower=0)).astype(np.float64)
    oi_chg = oi_log.groupby(data["symbol"], sort=False).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    oi_chg = oi_chg.mask(data["is_long_break_before"].fillna(False), 0.0)
    data["oi_chg"] = oi_chg.astype(np.float32)
    for col, out_col in [("ret1", "xret1"), ("range_rel", "xrange"), ("log_amount", "xamount"), ("oi_chg", "xoi_chg")]:
        data[out_col] = xsec_z(data, col)
    data["event_score"] = (
        data["xret1"].abs().clip(0, 8)
        + data["xrange"].abs().clip(0, 8)
        + 0.65 * data["xamount"].abs().clip(0, 8)
        + 0.35 * data["xoi_chg"].abs().clip(0, 8)
    ).astype(np.float32)

    g = data.groupby("datetime", sort=False)["label"]
    mu = g.transform("mean")
    sd = g.transform("std")
    rank = g.rank(pct=True).astype(np.float32)
    data["label_xsz"] = ((data["label"] - mu) / (sd + 1e-9)).clip(-8, 8).astype(np.float32)
    data["label_xrank"] = (rank - 0.5).astype(np.float32)
    data["label_ranknorm"] = ndtri(rank.clip(0.01, 0.99)).astype(np.float32)

    data["grp_ret_mean"] = data.groupby(["datetime", "group"], sort=False)["ret1"].transform("mean").astype(np.float32)
    data["grp_amt_mean"] = data.groupby(["datetime", "group"], sort=False)["log_amount"].transform("mean").astype(np.float32)
    data["grp_range_mean"] = data.groupby(["datetime", "group"], sort=False)["range_rel"].transform("mean").astype(np.float32)
    data["rel_ret_group"] = (data["ret1"] - data["grp_ret_mean"]).astype(np.float32)
    data["rel_amt_group"] = (data["log_amount"] - data["grp_amt_mean"]).astype(np.float32)
    data["market_ret_mean"] = data.groupby("datetime", sort=False)["ret1"].transform("mean").astype(np.float32)
    data["market_amt_mean"] = data.groupby("datetime", sort=False)["log_amount"].transform("mean").astype(np.float32)

    extra = [
        "symbol_code",
        "group_code",
        "minute_sin",
        "minute_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "session_pos",
    ] + REGIME_COLS + [
        "grp_ret_mean",
        "grp_amt_mean",
        "grp_range_mean",
        "rel_ret_group",
        "rel_amt_group",
        "market_ret_mean",
        "market_amt_mean",
    ]
    for col in extra:
        if col not in {"symbol_code", "group_code"}:
            data[col] = data[col].astype(np.float32)
    return data, extra


def stratified_sample(data: pd.DataFrame, pool: np.ndarray, need: int, rng: np.random.Generator) -> np.ndarray:
    if need <= 0 or len(pool) == 0:
        return np.empty(0, dtype=pool.dtype)
    if len(pool) <= need:
        return pool
    ranks = data["label_xrank"].to_numpy(np.float32)[pool]
    bins = np.floor(np.clip((ranks + 0.5) * 6.0, 0, 5)).astype(np.int16)
    pieces = []
    per = max(1, need // 6)
    for b in range(6):
        loc = pool[bins == b]
        if len(loc):
            pieces.append(rng.choice(loc, min(len(loc), per), replace=False))
    used = sum(len(x) for x in pieces)
    if used < need:
        already = np.concatenate(pieces) if pieces else np.empty(0, dtype=pool.dtype)
        taken = np.zeros(len(data), dtype=bool)
        taken[already] = True
        rest = pool[~taken[pool]]
        if len(rest):
            pieces.append(rng.choice(rest, min(need - used, len(rest)), replace=False))
    out = np.concatenate(pieces) if pieces else pool
    if len(out) > need:
        out = rng.choice(out, need, replace=False)
    return np.sort(out)


def sample_rows(data: pd.DataFrame, cap: int, mode: str, seed: int) -> pd.DataFrame:
    pool = np.flatnonzero(data["label"].notna().to_numpy() & data["label_xrank"].notna().to_numpy())
    if len(pool) <= cap:
        return data.iloc[pool].copy()
    rng = np.random.default_rng(seed)
    if mode == "random":
        idx = rng.choice(pool, cap, replace=False)
    elif mode == "event_tail":
        scores = np.nan_to_num(data["event_score"].to_numpy(np.float32)[pool], nan=0.0, posinf=0.0, neginf=0.0)
        cut = float(np.quantile(scores, 0.72))
        event_pool = pool[scores >= cut]
        event_scores = scores[scores >= cut]
        event_cap = min(len(event_pool), int(cap * 0.68))
        if len(event_pool) > event_cap:
            take = np.argpartition(event_scores, -event_cap)[-event_cap:]
            event_pick = event_pool[take]
        else:
            event_pick = event_pool
        picked = np.zeros(len(data), dtype=bool)
        picked[event_pick] = True
        rest = pool[~picked[pool]]
        rest_pick = stratified_sample(data, rest, cap - len(event_pick), rng)
        idx = np.concatenate([event_pick, rest_pick])
    else:
        scores = np.nan_to_num(data["event_score"].to_numpy(np.float32)[pool], nan=0.0, posinf=0.0, neginf=0.0)
        event_frac = {"soft_event": 0.25, "event": 0.35, "event50": 0.50}.get(mode)
        if event_frac is None:
            raise ValueError(f"bad sample_mode={mode}")
        event_need = min(len(pool), int(cap * event_frac))
        weights = np.sqrt(np.maximum(scores, 0.0) + 0.05)
        weights = weights / weights.sum()
        event_pick = rng.choice(pool, event_need, replace=False, p=weights)
        picked = np.zeros(len(data), dtype=bool)
        picked[event_pick] = True
        rest = pool[~picked[pool]]
        rest_pick = stratified_sample(data, rest, cap - len(event_pick), rng)
        idx = np.concatenate([event_pick, rest_pick])
    return data.iloc[np.sort(idx)].copy()


def make_x(df: pd.DataFrame, feat_cols: list[str], extra_cols: list[str]) -> pd.DataFrame:
    float_cols = feat_cols + [c for c in extra_cols if c not in {"symbol_code", "group_code"}]
    x = pd.DataFrame(scrub(df[float_cols].to_numpy(np.float32)), columns=float_cols, index=df.index)
    x["symbol_code"] = df["symbol_code"].to_numpy()
    x["group_code"] = df["group_code"].to_numpy()
    return x


def ensure_month_cache(variant: ChunkVariant, feat_cols: list[str], extra_cols_ref: list[str] | None) -> list[str]:
    sym_map, grp_map, groups = symbol_maps()
    cache_dir = cache_dir_for(variant)
    cache_dir.mkdir(parents=True, exist_ok=True)
    extra_cols = extra_cols_ref
    for i, ms in enumerate(all_months()):
        path = cache_dir / f"{ms:%Y-%m}.parquet"
        if path.exists():
            continue
        data = read_month(ms, feat_cols)
        data, extra_cols = add_context_targets(data, sym_map, grp_map, groups)
        keep_cols = list(
            dict.fromkeys(
                META_COLS
                + feat_cols
                + extra_cols
                + ["label_xsz", "label_xrank", "label_ranknorm", "event_score", "_bars_to_month_end"]
            )
        )
        sample = sample_rows(data[keep_cols], variant.month_sample_rows, variant.sample_mode, variant.seed + i)
        write_parquet_atomic(sample, path)
        print(f"[cache][{variant.name}][{ms:%Y-%m}] rows={len(sample)}", flush=True)
        del data, sample
        gc.collect()
    if extra_cols is None:
        probe = pd.read_parquet(sorted(cache_dir.glob("*.parquet"))[0])
        extra_cols = [c for c in probe.columns if c not in set(META_COLS + feat_cols + ["label_xsz", "label_xrank", "label_ranknorm", "event_score"])]
    return extra_cols


def load_train_samples(variant: ChunkVariant, ms: pd.Timestamp) -> pd.DataFrame:
    cache_dir = cache_dir_for(variant)
    tr_start = TRAIN_START if variant.lookback_months <= 0 else max(TRAIN_START, ms - pd.DateOffset(months=variant.lookback_months))
    pieces = []
    for m in all_months():
        if tr_start <= m < ms:
            path = cache_dir / f"{m:%Y-%m}.parquet"
            if path.exists():
                piece = pd.read_parquet(path)
                if variant.embargo_bars > 0 and m == ms - pd.offsets.MonthBegin(1):
                    if "_bars_to_month_end" not in piece.columns:
                        raise RuntimeError(
                            f"{path} lacks _bars_to_month_end; rebuild cache before using embargo"
                        )
                    piece = piece[piece["_bars_to_month_end"] >= variant.embargo_bars].copy()
                pieces.append(piece)
    if not pieces:
        return pd.DataFrame()
    train = pd.concat(pieces, ignore_index=True)
    if len(train) > variant.max_rows:
        train = sample_rows(train, variant.max_rows, variant.sample_mode, variant.seed + int(ms.year * 12 + ms.month))
    return train


def predict_month(
    variant: ChunkVariant,
    ms: pd.Timestamp,
    feat_cols: list[str],
    extra_cols: list[str],
    sym_map: dict[str, int],
    grp_map: dict[str, int],
    groups: dict[str, str],
) -> pd.DataFrame:
    target_col = {"raw": "label", "xsz": "label_xsz", "xrank": "label_xrank", "ranknorm": "label_ranknorm"}[variant.target_mode]
    train = load_train_samples(variant, ms)
    if len(train) < 5000:
        raise RuntimeError(f"not enough train rows for {ms:%Y-%m}: {len(train)}")
    test = read_month(ms, feat_cols)
    test, _ = add_context_targets(test, sym_map, grp_map, groups)
    params = dict(
        n_estimators=variant.n_estimators,
        learning_rate=variant.learning_rate,
        num_leaves=variant.num_leaves,
        subsample=variant.subsample,
        colsample_bytree=variant.colsample_bytree,
        min_child_samples=variant.min_child_samples,
        reg_lambda=variant.reg_lambda,
        n_jobs=int(os.environ.get("CHUNKED_N_JOBS", "4")),
        random_state=variant.seed,
        verbose=-1,
        force_col_wise=True,
    )
    if variant.group_expert:
        if variant.model_type != "regressor":
            raise ValueError("group_expert currently supports only regressor variants")
        ytr = train[target_col].to_numpy(np.float32)
        pred = np.full(len(test), np.nan, dtype=np.float32)
        fitted_groups = 0
        fallback_model: lgb.LGBMRegressor | None = None
        fallback_xtr: pd.DataFrame | None = None
        train_group = train["group_code"].to_numpy()
        test_group = test["group_code"].to_numpy()
        for grp in sorted(pd.unique(test_group)):
            pr_mask = test_group == grp
            if int(pr_mask.sum()) == 0:
                continue
            tr_mask = train_group == grp
            if int(tr_mask.sum()) >= 8_000:
                gxtr = make_x(train.loc[tr_mask], feat_cols, extra_cols)
                gytr = train.loc[tr_mask, target_col].to_numpy(np.float32)
                gxte = make_x(test.loc[pr_mask], feat_cols, extra_cols)
                model = lgb.LGBMRegressor(**params)
                model.fit(gxtr, gytr, categorical_feature=["symbol_code", "group_code"])
                pred[pr_mask] = model.predict(gxte).astype(np.float32)
                fitted_groups += 1
                del gxtr, gytr, gxte, model
            else:
                if fallback_model is None:
                    fallback_xtr = make_x(train, feat_cols, extra_cols)
                    fallback_model = lgb.LGBMRegressor(**params)
                    fallback_model.fit(fallback_xtr, ytr, categorical_feature=["symbol_code", "group_code"])
                gxte = make_x(test.loc[pr_mask], feat_cols, extra_cols)
                pred[pr_mask] = fallback_model.predict(gxte).astype(np.float32)
                del gxte
        if fallback_xtr is not None:
            del fallback_xtr
        print(f"[predict][{variant.name}][{ms:%Y-%m}] group_models={fitted_groups}/{len(pd.unique(test_group))}", flush=True)
    elif variant.model_type == "ranker":
        train_fit = train.sort_values(["datetime", "symbol"], kind="mergesort").copy()
        xtr = make_x(train_fit, feat_cols, extra_cols)
        rel = np.floor(np.clip((train_fit["label_xrank"].to_numpy(np.float32) + 0.5) * 10.0, 0, 10)).astype(np.int32)
        group_sizes = train_fit.groupby("datetime", sort=False).size().to_numpy(np.int32)
        model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", label_gain=list(range(11)), **params)
        model.fit(xtr, rel, group=group_sizes, categorical_feature=["symbol_code", "group_code"])
        xte = make_x(test, feat_cols, extra_cols)
        pred = model.predict(xte)
        del train_fit
    elif variant.model_type == "regressor":
        xtr = make_x(train, feat_cols, extra_cols)
        ytr = train[target_col].to_numpy(np.float32)
        model = lgb.LGBMRegressor(**params)
        model.fit(xtr, ytr, categorical_feature=["symbol_code", "group_code"])
        xte = make_x(test, feat_cols, extra_cols)
        pred = model.predict(xte)
    else:
        raise ValueError(f"bad model_type={variant.model_type!r}")
    out = test[["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype(np.float32)
    mic = compute_ic(out["pred"].to_numpy(), out["label"].to_numpy())
    print(f"[predict][{variant.name}][{ms:%Y-%m}] tr={len(train):7d} pr={len(out):6d} IC={mic:.5f}", flush=True)
    for name in ["xtr", "xte", "model", "fallback_model"]:
        if name in locals():
            del locals()[name]
    del train, test
    gc.collect()
    return out


def run_variant(variant: ChunkVariant) -> pd.DataFrame:
    feat_cols = feature_cols_for(variant)
    print(f"[features][{variant.name}] mode={variant.feature_mode} n={len(feat_cols)}", flush=True)
    pred_path = OUT_DIR / f"{variant.name}.parquet"
    parts_dir = OUT_DIR / f"{variant.name}_month_parts"
    if pred_path.exists():
        return pd.read_parquet(pred_path)
    parts_dir.mkdir(parents=True, exist_ok=True)
    one_month = os.environ.get("CHUNKED_ONE_MONTH", "0") == "1"
    if one_month:
        missing = next(
            (ms for ms in prediction_months() if not (parts_dir / f"{ms:%Y-%m}.parquet").exists()),
            None,
        )
        if missing is not None:
            extra_cols = ensure_month_cache(variant, feat_cols, None)
            print(f"[run][{variant.name}] cache_ready extra={len(extra_cols)}", flush=True)
            sym_map, grp_map, groups = symbol_maps()
            print(f"[run][{variant.name}] symbol_maps symbols={len(sym_map)} groups={len(grp_map)}", flush=True)
            part = predict_month(variant, missing, feat_cols, extra_cols, sym_map, grp_map, groups)
            write_parquet_atomic(part, parts_dir / f"{missing:%Y-%m}.parquet")
            print(f"[one-month][{variant.name}] wrote {missing:%Y-%m}", flush=True)
            return pd.DataFrame()
        pieces = [pd.read_parquet(parts_dir / f"{ms:%Y-%m}.parquet") for ms in prediction_months()]
        pred = pd.concat(pieces, ignore_index=True)
        pred = add_cross_sectional_norms(pred, "pred")
        write_parquet_atomic(pred, pred_path)
        (OUT_DIR / f"{variant.name}.json").write_text(json.dumps(asdict(variant), indent=2), encoding="utf-8")
        return pred
    extra_cols = ensure_month_cache(variant, feat_cols, None)
    print(f"[run][{variant.name}] cache_ready extra={len(extra_cols)}", flush=True)
    sym_map, grp_map, groups = symbol_maps()
    print(f"[run][{variant.name}] symbol_maps symbols={len(sym_map)} groups={len(grp_map)}", flush=True)
    pieces = []
    for ms in prediction_months():
        part_path = parts_dir / f"{ms:%Y-%m}.parquet"
        if part_path.exists():
            part = pd.read_parquet(part_path)
            print(f"[predict][{variant.name}][{ms:%Y-%m}] ckpt rows={len(part)}", flush=True)
        else:
            part = predict_month(variant, ms, feat_cols, extra_cols, sym_map, grp_map, groups)
            write_parquet_atomic(part, part_path)
        pieces.append(part)
    pred = pd.concat(pieces, ignore_index=True)
    pred = add_cross_sectional_norms(pred, "pred")
    write_parquet_atomic(pred, pred_path)
    (OUT_DIR / f"{variant.name}.json").write_text(json.dumps(asdict(variant), indent=2), encoding="utf-8")
    return pred


def write_summaries(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    cur = pd.DataFrame(rows)
    path = OUT_DIR / "base_ablation_summary.csv"
    if path.exists():
        old = pd.read_csv(path)
        cur = pd.concat([old[~old["model"].isin(cur["model"])], cur], ignore_index=True)
    cur.to_csv(path, index=False)
    monthly_rows = []
    for model in cur["model"].tail(len(rows)):
        pred_path = OUT_DIR / f"{model}.parquet"
        if not pred_path.exists():
            continue
        pred = pd.read_parquet(pred_path)
        by_m = period_ic(pred[(pred["datetime"] >= TEST_START) & (pred["datetime"] < TEST_END)], "pred", "M")
        for month, ic in by_m.items():
            monthly_rows.append({"model": model, "month": month, "ic": float(ic)})
    if monthly_rows:
        pd.DataFrame(monthly_rows).to_csv(OUT_DIR / "chunked_base_monthly_ic.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in parse_variants():
        print(f"[variant] {variant.name} top={variant.top_n} feature_mode={variant.feature_mode}", flush=True)
        pred = run_variant(variant)
        if pred.empty:
            continue
        row = summarize(pred, variant.name) | {"ablation_type": "chunked_base_learner"}
        rows.append(row)
        print(pd.DataFrame([row])[["model", "pred_ic_2019", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020"]].to_string(index=False), flush=True)
    write_summaries(rows)


if __name__ == "__main__":
    main()
