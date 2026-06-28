#!/usr/bin/env python3
"""
Strict 2020 optimization ablations for the archived factor route.

The script keeps the validation protocol fixed:
  - every base learner prediction for 2019/2020 is produced by monthly
    train-before-test LightGBM models;
  - model/gate fitting uses 2019 OOS predictions only;
  - 2020 is read once as the final test window, never for selecting weights.

New ablation dimensions:
  - event-driven and label-balanced training row selection;
  - IC-oriented target transforms (cross-sectional z/rank-normal labels);
  - extra regime and sector-assist features from current bar information;
  - prior sector expert LightGBM;
  - static and bucketed dynamic MOE weight allocators trained on 2019 OOS.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, "/root/feature_model")

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtri

from rolling_factor_model_eval import (
    add_cross_sectional_norms,
    compute_ic,
    fit_ic_weights_from_stats,
    monthly_stats,
    period_ic,
    sum_stats,
)
from src.plan_a.group_lgb import symbol_group_map


FACTOR_PATH = Path("/root/shared-nvme/feature_model/data_factors_big.parquet")
SELECTED_PATH = Path("/root/quant/work/outputs/selected_factors.txt")
BASE_STRICT_DIR = Path("/root/autodl-tmp/quant/ML/strict_lgb_results")
OUT_DIR = Path("/root/autodl-tmp/quant/ML/strict_opt_results")
FIG_DIR = OUT_DIR / "figures"

TRAIN_START = pd.Timestamp("2018-01-01")
PRED_START = pd.Timestamp("2019-01-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")

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

CONTEXT_COLS = [
    "symbol_code",
    "group_code",
    "minute_sin",
    "minute_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "session_pos",
]

REGIME_COLS = [
    "ret1",
    "intrabar_ret",
    "range_rel",
    "body_rel",
    "log_volume",
    "log_amount",
    "oi_chg",
    "xret1",
    "xrange",
    "xamount",
    "xoi_chg",
    "event_score",
]


@dataclass(frozen=True)
class Variant:
    name: str
    top_n: int = 300
    target_mode: str = "xsz"
    sample_mode: str = "event"
    lookback_months: int = 0
    max_rows: int = 500_000
    use_assist: bool = True
    group_expert: bool = False
    seed: int = 42
    n_estimators: int = 280
    learning_rate: float = 0.035
    num_leaves: int = 63
    min_child_samples: int = 120
    reg_lambda: float = 5.0
    colsample_bytree: float = 0.68
    subsample: float = 0.82


DEFAULT_VARIANTS = [
    Variant(
        name="opt_t300_xsz_random_assist_exp",
        top_n=300,
        target_mode="xsz",
        sample_mode="random",
        lookback_months=0,
        max_rows=500_000,
        use_assist=True,
        group_expert=False,
        seed=97,
    ),
    Variant(
        name="opt_t300_xsz_softevent_assist_exp",
        top_n=300,
        target_mode="xsz",
        sample_mode="soft_event",
        lookback_months=0,
        max_rows=500_000,
        use_assist=True,
        group_expert=False,
        seed=98,
    ),
    Variant(
        name="opt_t220_xsz_softevent_assist_exp",
        top_n=220,
        target_mode="xsz",
        sample_mode="soft_event",
        lookback_months=0,
        max_rows=460_000,
        use_assist=True,
        group_expert=False,
        seed=198,
        colsample_bytree=0.70,
        reg_lambda=6.0,
    ),
    Variant(
        name="opt_t150_xsz_softevent_assist_exp",
        top_n=150,
        target_mode="xsz",
        sample_mode="soft_event",
        lookback_months=0,
        max_rows=420_000,
        use_assist=True,
        group_expert=False,
        seed=298,
        colsample_bytree=0.78,
        reg_lambda=6.0,
    ),
    Variant(
        name="opt_t220_ranknorm_softevent_assist_exp",
        top_n=220,
        target_mode="ranknorm",
        sample_mode="soft_event",
        lookback_months=0,
        max_rows=460_000,
        use_assist=True,
        group_expert=False,
        seed=199,
        colsample_bytree=0.70,
        reg_lambda=8.0,
    ),
    Variant(
        name="opt_t150_ranknorm_softevent_assist_exp",
        top_n=150,
        target_mode="ranknorm",
        sample_mode="soft_event",
        lookback_months=0,
        max_rows=420_000,
        use_assist=True,
        group_expert=False,
        seed=299,
        colsample_bytree=0.78,
        reg_lambda=8.0,
    ),
    Variant(
        name="opt_t300_ranknorm_random_assist_exp",
        top_n=300,
        target_mode="ranknorm",
        sample_mode="random",
        lookback_months=0,
        max_rows=500_000,
        use_assist=True,
        group_expert=False,
        seed=102,
        reg_lambda=7.0,
    ),
    Variant(
        name="opt_t500_xsz_random_assist_exp",
        top_n=500,
        target_mode="xsz",
        sample_mode="random",
        lookback_months=0,
        max_rows=550_000,
        use_assist=True,
        group_expert=False,
        seed=103,
        colsample_bytree=0.58,
        reg_lambda=6.0,
    ),
    Variant(
        name="opt_t500_xsz_softevent_assist_exp",
        top_n=500,
        target_mode="xsz",
        sample_mode="soft_event",
        lookback_months=0,
        max_rows=550_000,
        use_assist=True,
        group_expert=False,
        seed=104,
        colsample_bytree=0.58,
        reg_lambda=6.0,
    ),
    Variant(
        name="opt_t650_xsz_random_assist_exp",
        top_n=650,
        target_mode="xsz",
        sample_mode="random",
        lookback_months=0,
        max_rows=600_000,
        use_assist=True,
        group_expert=False,
        seed=107,
        colsample_bytree=0.50,
        reg_lambda=7.0,
    ),
    Variant(
        name="opt_t500_xsz_random_assist_lb18",
        top_n=500,
        target_mode="xsz",
        sample_mode="random",
        lookback_months=18,
        max_rows=550_000,
        use_assist=True,
        group_expert=False,
        seed=109,
        colsample_bytree=0.58,
        reg_lambda=6.0,
    ),
    Variant(
        name="opt_t500_xsz_random_assist_lb12",
        top_n=500,
        target_mode="xsz",
        sample_mode="random",
        lookback_months=12,
        max_rows=520_000,
        use_assist=True,
        group_expert=False,
        seed=111,
        colsample_bytree=0.58,
        reg_lambda=6.0,
    ),
    Variant(
        name="opt_t500_xsz_random_assist_shallow",
        top_n=500,
        target_mode="xsz",
        sample_mode="random",
        lookback_months=0,
        max_rows=550_000,
        use_assist=True,
        group_expert=False,
        seed=117,
        n_estimators=340,
        num_leaves=31,
        min_child_samples=260,
        colsample_bytree=0.55,
        reg_lambda=12.0,
    ),
    Variant(
        name="opt_t220_xsz_random_assist_shallow",
        top_n=220,
        target_mode="xsz",
        sample_mode="random",
        lookback_months=0,
        max_rows=460_000,
        use_assist=True,
        group_expert=False,
        seed=217,
        n_estimators=340,
        num_leaves=31,
        min_child_samples=260,
        colsample_bytree=0.72,
        reg_lambda=12.0,
    ),
    Variant(
        name="opt_t150_xsz_random_assist_shallow",
        top_n=150,
        target_mode="xsz",
        sample_mode="random",
        lookback_months=0,
        max_rows=420_000,
        use_assist=True,
        group_expert=False,
        seed=317,
        n_estimators=340,
        num_leaves=31,
        min_child_samples=260,
        colsample_bytree=0.78,
        reg_lambda=12.0,
    ),
    Variant(
        name="opt_t300_xsz_event_plain_exp",
        top_n=300,
        target_mode="xsz",
        sample_mode="event",
        lookback_months=0,
        max_rows=500_000,
        use_assist=False,
        group_expert=False,
        seed=99,
    ),
    Variant(
        name="opt_t300_xsz_event_assist_exp",
        top_n=300,
        target_mode="xsz",
        sample_mode="event",
        lookback_months=0,
        max_rows=500_000,
        use_assist=True,
        group_expert=False,
        seed=101,
    ),
    Variant(
        name="opt_t300_ranknorm_event_assist_exp",
        top_n=300,
        target_mode="ranknorm",
        sample_mode="event",
        lookback_months=0,
        max_rows=500_000,
        use_assist=True,
        group_expert=False,
        seed=113,
        reg_lambda=7.0,
    ),
    Variant(
        name="opt_t500_xsz_event_assist_exp",
        top_n=500,
        target_mode="xsz",
        sample_mode="event",
        lookback_months=0,
        max_rows=550_000,
        use_assist=True,
        group_expert=False,
        seed=127,
        colsample_bytree=0.58,
        reg_lambda=6.0,
    ),
    Variant(
        name="opt_t300_xsz_event_assist_lb18",
        top_n=300,
        target_mode="xsz",
        sample_mode="event",
        lookback_months=18,
        max_rows=450_000,
        use_assist=True,
        group_expert=False,
        seed=131,
    ),
    Variant(
        name="opt_group_t260_xsz_event_assist_exp",
        top_n=260,
        target_mode="xsz",
        sample_mode="event",
        lookback_months=0,
        max_rows=120_000,
        use_assist=True,
        group_expert=True,
        seed=151,
        n_estimators=240,
        min_child_samples=70,
        colsample_bytree=0.72,
    ),
    Variant(
        name="opt_group_t220_xsz_softevent_assist_exp",
        top_n=220,
        target_mode="xsz",
        sample_mode="soft_event",
        lookback_months=0,
        max_rows=110_000,
        use_assist=True,
        group_expert=True,
        seed=251,
        n_estimators=260,
        min_child_samples=80,
        colsample_bytree=0.78,
        reg_lambda=7.0,
    ),
    Variant(
        name="opt_group_t150_xsz_softevent_assist_exp",
        top_n=150,
        target_mode="xsz",
        sample_mode="soft_event",
        lookback_months=0,
        max_rows=95_000,
        use_assist=True,
        group_expert=True,
        seed=351,
        n_estimators=260,
        min_child_samples=80,
        colsample_bytree=0.82,
        reg_lambda=7.0,
    ),
    Variant(
        name="opt_group_t260_xsz_random_assist_exp",
        top_n=260,
        target_mode="xsz",
        sample_mode="random",
        lookback_months=0,
        max_rows=120_000,
        use_assist=True,
        group_expert=True,
        seed=157,
        n_estimators=240,
        min_child_samples=70,
        colsample_bytree=0.72,
    ),
]


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def selected_features(top_n: int) -> list[str]:
    selected = [x.strip() for x in SELECTED_PATH.read_text().splitlines() if x.strip()]
    return selected[:top_n]


def month_code(dt: pd.Series) -> np.ndarray:
    return (dt.dt.year.to_numpy(dtype=np.int16) * 12 + dt.dt.month.to_numpy(dtype=np.int16)).astype(np.int32)


def xsec_z(data: pd.DataFrame, col: str) -> pd.Series:
    g = data.groupby("datetime", sort=False)[col]
    return ((data[col] - g.transform("mean")) / (g.transform("std") + 1e-9)).astype(np.float32)


def load_factor_data(top_n: int) -> tuple[pd.DataFrame, list[str]]:
    feat_cols = selected_features(top_n)
    cols = list(dict.fromkeys(META_COLS + feat_cols))
    print(f"[load] top_n={top_n} cols={len(cols)} path={FACTOR_PATH}", flush=True)
    data = pd.read_parquet(
        FACTOR_PATH,
        columns=cols,
        filters=[
            ("datetime", ">=", TRAIN_START),
            ("datetime", "<", TEST_END),
        ],
    )
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    data["_month_code"] = month_code(data["datetime"])
    print(
        f"[load] rows={len(data)} dates={data.datetime.min()}..{data.datetime.max()} features={len(feat_cols)}",
        flush=True,
    )
    return data, feat_cols


def add_context_regime(data: pd.DataFrame, *, use_assist: bool) -> tuple[pd.DataFrame, list[str]]:
    groups = symbol_group_map()
    group_names = sorted(set(groups.values()) | {"other"})
    sym_names = sorted(data["symbol"].unique())
    sym_map = {s: i for i, s in enumerate(sym_names)}
    grp_map = {g: i for i, g in enumerate(group_names)}

    data["group"] = data["symbol"].map(groups).fillna("other")
    data["symbol_code"] = data["symbol"].map(sym_map).astype(np.int16)
    data["group_code"] = data["group"].map(grp_map).astype(np.int8)

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
    data["range_rel"] = ((high - low) / close.abs().clip(lower=1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(
        np.float32
    )
    data["body_rel"] = ((close - open_) / open_.abs().clip(lower=1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(
        np.float32
    )
    data["log_volume"] = np.log1p(data["volume"].clip(lower=0)).astype(np.float32)
    data["log_amount"] = np.log1p(data["amount"].clip(lower=0)).astype(np.float32)
    oi_log = np.log1p(data["oi"].clip(lower=0)).astype(np.float64)
    oi_chg = oi_log.groupby(data["symbol"], sort=False).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    oi_chg = oi_chg.mask(data["is_long_break_before"].fillna(False), 0.0)
    data["oi_chg"] = oi_chg.astype(np.float32)

    for col, out_col in [
        ("ret1", "xret1"),
        ("range_rel", "xrange"),
        ("log_amount", "xamount"),
        ("oi_chg", "xoi_chg"),
    ]:
        data[out_col] = xsec_z(data, col)
    data["event_score"] = (
        data["xret1"].abs().clip(0, 8)
        + data["xrange"].abs().clip(0, 8)
        + 0.65 * data["xamount"].abs().clip(0, 8)
        + 0.35 * data["xoi_chg"].abs().clip(0, 8)
    ).astype(np.float32)

    extra = CONTEXT_COLS + REGIME_COLS
    if use_assist:
        data["grp_ret_mean"] = data.groupby(["datetime", "group"], sort=False)["ret1"].transform("mean").astype(np.float32)
        data["grp_amt_mean"] = data.groupby(["datetime", "group"], sort=False)["log_amount"].transform("mean").astype(
            np.float32
        )
        data["grp_range_mean"] = data.groupby(["datetime", "group"], sort=False)["range_rel"].transform("mean").astype(
            np.float32
        )
        data["rel_ret_group"] = (data["ret1"] - data["grp_ret_mean"]).astype(np.float32)
        data["rel_amt_group"] = (data["log_amount"] - data["grp_amt_mean"]).astype(np.float32)
        data["market_ret_mean"] = data.groupby("datetime", sort=False)["ret1"].transform("mean").astype(np.float32)
        data["market_amt_mean"] = data.groupby("datetime", sort=False)["log_amount"].transform("mean").astype(np.float32)
        extra += [
            "grp_ret_mean",
            "grp_amt_mean",
            "grp_range_mean",
            "rel_ret_group",
            "rel_amt_group",
            "market_ret_mean",
            "market_amt_mean",
        ]

        if os.environ.get("STRICT_OPT_SKIP_CHAIN_ASSIST", "0") != "1":
            group_ret = data.pivot_table(index="datetime", columns="group", values="ret1", aggfunc="mean")
            group_ret = group_ret.reindex(columns=group_names).fillna(0.0).astype(np.float32)
            group_ret.columns = [f"chain_ret_{c}" for c in group_ret.columns]
            data = data.merge(group_ret.reset_index(), on="datetime", how="left")
            extra += list(group_ret.columns)

    for col in extra:
        if col not in {"symbol_code", "group_code"}:
            data[col] = data[col].astype(np.float32)
    return data, extra


def add_target_transforms(data: pd.DataFrame) -> pd.DataFrame:
    g = data.groupby("datetime", sort=False)["label"]
    mu = g.transform("mean")
    sd = g.transform("std")
    data["label_xsz"] = ((data["label"] - mu) / (sd + 1e-9)).clip(-8, 8).astype(np.float32)
    rank = g.rank(pct=True).astype(np.float32)
    data["label_xrank"] = (rank - 0.5).astype(np.float32)
    data["label_ranknorm"] = ndtri(rank.clip(0.01, 0.99)).astype(np.float32)
    return data


def make_x(df: pd.DataFrame, feat_cols: list[str], extra_cols: list[str]) -> pd.DataFrame:
    all_float = feat_cols + [c for c in extra_cols if c not in {"symbol_code", "group_code"}]
    x = pd.DataFrame(scrub(df[all_float].to_numpy(np.float32)), columns=all_float, index=df.index)
    for col in ["symbol_code", "group_code"]:
        if col in extra_cols:
            x[col] = df[col].to_numpy()
    return x


def stratified_month_label_sample(
    data: pd.DataFrame,
    pool: np.ndarray,
    need: int,
    rng: np.random.Generator,
    *,
    weighted_fill: bool,
) -> np.ndarray:
    if need <= 0 or len(pool) == 0:
        return np.empty(0, dtype=pool.dtype)
    if len(pool) <= need:
        return pool
    months = data["_month_code"].to_numpy(np.int32)[pool]
    ranks = data["label_xrank"].to_numpy(np.float32)[pool]
    bins = np.floor(np.clip((ranks + 0.5) * 6.0, 0, 5)).astype(np.int16)
    strata = months.astype(np.int64) * 10 + bins.astype(np.int64)
    unique = np.unique(strata)
    per = max(1, need // max(len(unique), 1))
    pieces = []
    used = 0
    for st in unique:
        loc = np.flatnonzero(strata == st)
        take = min(len(loc), per)
        if take > 0:
            pieces.append(pool[rng.choice(loc, take, replace=False)])
            used += take
    if used < need:
        already = np.concatenate(pieces) if pieces else np.empty(0, dtype=pool.dtype)
        taken = np.zeros(len(data), dtype=bool)
        taken[already] = True
        rest = pool[~taken[pool]]
        fill = min(need - used, len(rest))
        if fill > 0:
            if weighted_fill:
                weights = np.nan_to_num(data["event_score"].to_numpy(np.float32)[rest], nan=0.0)
                weights = np.sqrt(np.maximum(weights, 0.0) + 0.05)
                weights = weights / weights.sum()
                pieces.append(rng.choice(rest, fill, replace=False, p=weights))
            else:
                pieces.append(rng.choice(rest, fill, replace=False))
    out = np.concatenate(pieces) if pieces else np.empty(0, dtype=pool.dtype)
    if len(out) > need:
        out = rng.choice(out, need, replace=False)
    return out


def sample_indices(data: pd.DataFrame, tr_idx: np.ndarray, variant: Variant, rng: np.random.Generator) -> np.ndarray:
    max_rows = variant.max_rows
    if max_rows <= 0 or len(tr_idx) <= max_rows:
        return tr_idx
    if variant.sample_mode == "random":
        return np.sort(rng.choice(tr_idx, max_rows, replace=False))
    if variant.sample_mode == "soft_event":
        scores = np.nan_to_num(data["event_score"].to_numpy(np.float32)[tr_idx], nan=0.0, posinf=0.0, neginf=0.0)
        event_need = min(len(tr_idx), int(max_rows * 0.25))
        weights = np.sqrt(np.maximum(scores, 0.0) + 0.05)
        weights = weights / weights.sum()
        event_pick = rng.choice(tr_idx, event_need, replace=False, p=weights)
        picked = np.zeros(len(data), dtype=bool)
        picked[event_pick] = True
        rest_pool = tr_idx[~picked[tr_idx]]
        rest_pick = stratified_month_label_sample(
            data,
            rest_pool,
            max_rows - len(event_pick),
            rng,
            weighted_fill=False,
        )
        return np.sort(np.concatenate([event_pick, rest_pick]))
    if variant.sample_mode != "event":
        raise ValueError(f"bad sample_mode={variant.sample_mode}")

    scores = data["event_score"].to_numpy(np.float32)[tr_idx]
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    cut = float(np.quantile(scores, 0.72))
    event_pool = tr_idx[scores >= cut]
    event_scores = scores[scores >= cut]
    event_cap = min(len(event_pool), int(max_rows * 0.70))
    if len(event_pool) > event_cap:
        order = np.argpartition(event_scores, -event_cap)[-event_cap:]
        event_pick = event_pool[order]
    else:
        event_pick = event_pool

    need = max_rows - len(event_pick)
    if need <= 0:
        return np.sort(event_pick)

    picked = np.zeros(len(data), dtype=bool)
    picked[event_pick] = True
    rest = tr_idx[~picked[tr_idx]]
    if len(rest) <= need:
        return np.sort(np.concatenate([event_pick, rest]))
    rest_pick = stratified_month_label_sample(data, rest, need, rng, weighted_fill=True)
    out = np.concatenate([event_pick, rest_pick])
    if len(out) > max_rows:
        out = rng.choice(out, max_rows, replace=False)
    return np.sort(out)


def summarize(pred: pd.DataFrame, model: str) -> dict[str, object]:
    row: dict[str, object] = {"model": model, "rows": int(len(pred)), "label_rows": int(pred["label"].notna().sum())}
    for start, end, tag in [(PRED_START, TEST_START, "2019"), (TEST_START, TEST_END, "2020")]:
        sub = pred[(pred["datetime"] >= start) & (pred["datetime"] < end)].copy()
        for col in ["pred", "pred_xsz", "pred_xrank"]:
            if col not in sub.columns:
                continue
            valid = sub[col].notna() & sub["label"].notna()
            if sub.empty or int(valid.sum()) < 2:
                row[f"{col}_ic_{tag}"] = float("nan")
                row[f"{col}_monthly_mean_{tag}"] = float("nan")
                row[f"{col}_monthly_ir_{tag}"] = float("nan")
                continue
            by_m = period_ic(sub, col, "M")
            if isinstance(by_m, pd.DataFrame):
                by_m = by_m.iloc[:, 0] if by_m.shape[1] else pd.Series(dtype=float)
            by_m = pd.to_numeric(by_m, errors="coerce")
            row[f"{col}_ic_{tag}"] = compute_ic(sub[col].to_numpy(), sub["label"].to_numpy())
            row[f"{col}_monthly_mean_{tag}"] = float(by_m.mean())
            row[f"{col}_monthly_ir_{tag}"] = float(by_m.mean() / by_m.std()) if by_m.std() > 0 else float("nan")
    return row


def train_monthly_global(data: pd.DataFrame, feat_cols: list[str], extra_cols: list[str], variant: Variant) -> pd.DataFrame:
    target_col = {"raw": "label", "xsz": "label_xsz", "xrank": "label_xrank", "ranknorm": "label_ranknorm"}[
        variant.target_mode
    ]
    params = dict(
        n_estimators=variant.n_estimators,
        learning_rate=variant.learning_rate,
        num_leaves=variant.num_leaves,
        subsample=variant.subsample,
        colsample_bytree=variant.colsample_bytree,
        min_child_samples=variant.min_child_samples,
        reg_lambda=variant.reg_lambda,
        n_jobs=int(os.environ.get("STRICT_OPT_N_JOBS", "16")),
        random_state=variant.seed,
        verbose=-1,
        force_col_wise=True,
    )
    out_mask = (data["datetime"] >= PRED_START) & (data["datetime"] < TEST_END)
    out = data.loc[out_mask, ["symbol", "datetime", "label"]].copy()
    out["pred"] = np.nan
    months = pd.date_range(PRED_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS")
    rng = np.random.default_rng(variant.seed)
    ckpt_dir = OUT_DIR / f"{variant.name}_month_parts"
    use_ckpt = os.environ.get("STRICT_OPT_MONTH_CKPT", "0") == "1"
    if use_ckpt:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    for ms in months:
        next_ms = ms + pd.DateOffset(months=1)
        ckpt_path = ckpt_dir / f"{ms:%Y-%m}.parquet"
        if use_ckpt and ckpt_path.exists():
            part = pd.read_parquet(ckpt_path)
            loc = out.index[(out["datetime"] >= ms) & (out["datetime"] < next_ms)]
            if len(part) == len(loc):
                out.loc[loc, "pred"] = part["pred"].to_numpy(np.float32)
                mic = compute_ic(part["pred"].to_numpy(), part["label"].to_numpy())
                print(f"  [{variant.name}][{ms:%Y-%m}] ckpt pr={len(part):6d} IC={mic:.5f}", flush=True)
                continue
        tr_start = TRAIN_START if variant.lookback_months <= 0 else max(TRAIN_START, ms - pd.DateOffset(months=variant.lookback_months))
        tr_mask = (
            (data["datetime"] >= tr_start)
            & (data["datetime"] < ms)
            & data[target_col].notna()
            & data["label"].notna()
        )
        pr_mask = (data["datetime"] >= ms) & (data["datetime"] < next_ms)
        tr_idx = np.flatnonzero(tr_mask.to_numpy())
        pr_idx = np.flatnonzero(pr_mask.to_numpy())
        if len(tr_idx) < 5000 or len(pr_idx) == 0:
            print(f"  [{variant.name}][{ms:%Y-%m}] skip tr={len(tr_idx)} pr={len(pr_idx)}", flush=True)
            continue
        tr_idx = sample_indices(data, tr_idx, variant, rng)
        tr = data.iloc[tr_idx]
        pr = data.iloc[pr_idx]
        xtr = make_x(tr, feat_cols, extra_cols)
        ytr = tr[target_col].to_numpy(np.float32)
        xpr = make_x(pr, feat_cols, extra_cols)
        model = lgb.LGBMRegressor(**params)
        model.fit(xtr, ytr, categorical_feature=[c for c in ["symbol_code", "group_code"] if c in xtr.columns])
        pred = model.predict(xpr)
        out.loc[pr.index, "pred"] = pred
        if use_ckpt:
            part = pr[["symbol", "datetime", "label"]].copy()
            part["pred"] = pred.astype(np.float32)
            part.to_parquet(ckpt_path, index=False)
        mic = compute_ic(pred, pr["label"].to_numpy())
        print(f"  [{variant.name}][{ms:%Y-%m}] tr={len(tr_idx):7d} pr={len(pr_idx):6d} IC={mic:.5f}", flush=True)
        del tr, pr, xtr, xpr, model
        gc.collect()
    return add_cross_sectional_norms(out, "pred")


def train_monthly_group(data: pd.DataFrame, feat_cols: list[str], extra_cols: list[str], variant: Variant) -> pd.DataFrame:
    target_col = {"raw": "label", "xsz": "label_xsz", "xrank": "label_xrank", "ranknorm": "label_ranknorm"}[
        variant.target_mode
    ]
    params = dict(
        n_estimators=variant.n_estimators,
        learning_rate=variant.learning_rate,
        num_leaves=variant.num_leaves,
        subsample=variant.subsample,
        colsample_bytree=variant.colsample_bytree,
        min_child_samples=variant.min_child_samples,
        reg_lambda=variant.reg_lambda,
        n_jobs=int(os.environ.get("STRICT_OPT_N_JOBS", "16")),
        random_state=variant.seed,
        verbose=-1,
        force_col_wise=True,
    )
    out_mask = (data["datetime"] >= PRED_START) & (data["datetime"] < TEST_END)
    out = data.loc[out_mask, ["symbol", "datetime", "label", "group"]].copy()
    out["pred"] = np.nan
    months = pd.date_range(PRED_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS")
    groups = sorted(data["group"].dropna().unique())
    rng = np.random.default_rng(variant.seed)
    for ms in months:
        next_ms = ms + pd.DateOffset(months=1)
        tr_start = TRAIN_START if variant.lookback_months <= 0 else max(TRAIN_START, ms - pd.DateOffset(months=variant.lookback_months))
        month_ics = []
        for grp in groups:
            tr_mask = (
                (data["group"] == grp)
                & (data["datetime"] >= tr_start)
                & (data["datetime"] < ms)
                & data[target_col].notna()
                & data["label"].notna()
            )
            pr_mask = (data["group"] == grp) & (data["datetime"] >= ms) & (data["datetime"] < next_ms)
            tr_idx = np.flatnonzero(tr_mask.to_numpy())
            pr_idx = np.flatnonzero(pr_mask.to_numpy())
            if len(tr_idx) < 2500 or len(pr_idx) == 0:
                continue
            tr_idx = sample_indices(data, tr_idx, variant, rng)
            tr = data.iloc[tr_idx]
            pr = data.iloc[pr_idx]
            xtr = make_x(tr, feat_cols, extra_cols)
            ytr = tr[target_col].to_numpy(np.float32)
            xpr = make_x(pr, feat_cols, extra_cols)
            model = lgb.LGBMRegressor(**params)
            model.fit(xtr, ytr, categorical_feature=[c for c in ["symbol_code", "group_code"] if c in xtr.columns])
            pred = model.predict(xpr)
            out.loc[pr.index, "pred"] = pred
            month_ics.append(compute_ic(pred, pr["label"].to_numpy()))
            del tr, pr, xtr, xpr, model
            gc.collect()
        cur_mask = (out["datetime"] >= ms) & (out["datetime"] < next_ms)
        mic = compute_ic(out.loc[cur_mask, "pred"].to_numpy(), out.loc[cur_mask, "label"].to_numpy())
        print(f"  [{variant.name}][{ms:%Y-%m}] groups={len(month_ics)} IC={mic:.5f}", flush=True)
    out = out.drop(columns=["group"])
    return add_cross_sectional_norms(out, "pred")


def run_variant(variant: Variant) -> pd.DataFrame:
    pred_path = OUT_DIR / f"{variant.name}.parquet"
    if pred_path.exists() and os.environ.get("STRICT_OPT_OVERWRITE", "0") != "1":
        print(f"[skip] {variant.name} exists", flush=True)
        return pd.read_parquet(pred_path)
    data, feat_cols = load_factor_data(variant.top_n)
    data, extra_cols = add_context_regime(data, use_assist=variant.use_assist)
    data = add_target_transforms(data)
    print(f"[variant] {variant.name} extra={len(extra_cols)} group={variant.group_expert}", flush=True)
    if variant.group_expert:
        pred = train_monthly_group(data, feat_cols, extra_cols, variant)
    else:
        pred = train_monthly_global(data, feat_cols, extra_cols, variant)
    pred.to_parquet(pred_path, index=False)
    (OUT_DIR / f"{variant.name}.json").write_text(json.dumps(asdict(variant), indent=2), encoding="utf-8")
    print(f"[write] {pred_path}", flush=True)
    del data
    gc.collect()
    return pred


def prediction_column(df: pd.DataFrame, pred_col: str, name: str) -> pd.DataFrame:
    out = df[["symbol", "datetime", "label", pred_col]].copy().rename(columns={pred_col: name})
    out[name] = out[name].astype(np.float32)
    return out


def load_component_panel() -> tuple[pd.DataFrame, list[str]]:
    specs: list[tuple[str, Path, str]] = []
    strict_files = [
        ("base_raw", BASE_STRICT_DIR / "strict_lgb_raw_top300_n500000.parquet"),
        ("base_xsz", BASE_STRICT_DIR / "strict_lgb_xsz_top300_n500000.parquet"),
        ("base_xrank", BASE_STRICT_DIR / "strict_lgb_xrank_top300_n500000.parquet"),
    ]
    for prefix, path in strict_files:
        if path.exists():
            specs.extend([(f"{prefix}_raw", path, "pred"), (f"{prefix}_xsz", path, "pred_xsz"), (f"{prefix}_xrank", path, "pred_xrank")])
    min_opt_ic = float(os.environ.get("STRICT_OPT_MIN_BASE_2019_IC", "0.04"))
    eligible_opt: set[str] | None = None
    summary_path = OUT_DIR / "base_ablation_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "pred_ic_2019" in summary.columns:
            eligible_opt = set(
                summary.loc[summary["pred_ic_2019"].fillna(-1.0) >= min_opt_ic, "model"].astype(str).tolist()
            )
    candidate_files = (
        list(OUT_DIR.glob("opt_*.parquet"))
        + list(OUT_DIR.glob("chunk_*.parquet"))
        + list(OUT_DIR.glob("lowcorr_*.parquet"))
    )
    candidate_files = [p for p in candidate_files if not p.name.endswith("_meta_features.parquet")]
    for path in sorted(candidate_files):
        prefix = path.stem
        if eligible_opt is not None and prefix not in eligible_opt:
            print(f"[component-filter] skip {prefix}: 2019 IC below {min_opt_ic:.4f}", flush=True)
            continue
        specs.extend([(f"{prefix}_raw", path, "pred"), (f"{prefix}_xsz", path, "pred_xsz"), (f"{prefix}_xrank", path, "pred_xrank")])
    base = None
    names: list[str] = []
    for name, path, col in specs:
        try:
            df = pd.read_parquet(path, columns=["symbol", "datetime", "label", col])
        except Exception as exc:  # noqa: BLE001
            print(f"[component-skip] {path}: {exc}", flush=True)
            continue
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[(df["datetime"] >= PRED_START) & (df["datetime"] < TEST_END)].copy()
        cur = prediction_column(df, col, name)
        if base is None:
            base = cur
        else:
            base = base.merge(cur[["symbol", "datetime", name]], on=["symbol", "datetime"], how="inner")
        names.append(name)
    if base is None:
        raise RuntimeError("no component predictions available")
    base["_month"] = base["datetime"].dt.to_period("M").astype(str)
    print(f"[components] rows={len(base)} names={len(names)}", flush=True)
    return base, names


def embargo_train_slice(base: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    train = base[(base["datetime"] >= start) & (base["datetime"] < end)].copy()
    embargo_bars = int(os.environ.get("STRICT_OPT_GATE_EMBARGO_BARS", "30"))
    if embargo_bars <= 0 or train.empty:
        return train
    drop_idx = train.sort_values(["symbol", "datetime"]).groupby("symbol", sort=False).tail(embargo_bars).index
    if len(drop_idx):
        train = train.drop(index=drop_idx)
    return train


def fit_static_gate(base: pd.DataFrame, names: list[str], *, signed: bool, tag: str) -> tuple[pd.DataFrame, dict[str, object]]:
    train = embargo_train_slice(base, PRED_START, TEST_START)
    test = base[(base["datetime"] >= TEST_START) & (base["datetime"] < TEST_END)].copy()
    x = scrub(train[names].to_numpy(np.float32)).astype(np.float64, copy=False)
    y = train["label"].to_numpy(np.float64)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    c = x.T @ y
    g = x.T @ x
    yy = float(y @ y)
    lower = np.full(len(names), -0.12 if signed else 0.0, dtype=np.float64)
    upper = np.full(len(names), 0.85 if signed else 0.75, dtype=np.float64)
    w, train_ic = fit_ic_weights_from_stats(c, g, yy, lower, upper)
    pred = scrub(test[names].to_numpy(np.float32)) @ w.astype(np.float32)
    out = test[["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype(np.float32)
    out = add_cross_sectional_norms(out, "pred")
    meta = {"model": tag, "train_ic_2019": train_ic, "pred_ic_2020": compute_ic(out["pred"], out["label"])}
    meta.update({f"w_{n}": float(v) for n, v in zip(names, w)})
    return out, meta


def rolling_2020_gate(base: pd.DataFrame, names: list[str], *, signed: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats, months = monthly_stats(base, names)
    name_idx = np.arange(len(names), dtype=int)
    lower = np.full(len(names), -0.12 if signed else 0.0, dtype=np.float64)
    upper = np.full(len(names), 0.85 if signed else 0.75, dtype=np.float64)
    out = base[["symbol", "datetime", "label", "_month"]].copy()
    out["pred"] = np.nan
    records = []
    prev_w: np.ndarray | None = None
    test_months = [m for m in months if TEST_START <= pd.Period(m).to_timestamp() < TEST_END]
    for month in test_months:
        i = months.index(month)
        hist_months = months[:i]
        c, g, yy, n = sum_stats(stats, hist_months, name_idx)
        w, train_ic = fit_ic_weights_from_stats(c, g, yy, lower, upper, prev_w)
        prev_w = w
        cur = out["_month"].to_numpy() == month
        x = scrub(base.loc[cur, names].to_numpy(np.float32))
        pred = x @ w.astype(np.float32)
        out.loc[cur, "pred"] = pred
        rec: dict[str, object] = {
            "model": "moe_rolling_signed" if signed else "moe_rolling_nonneg",
            "month": month,
            "train_months": len(hist_months),
            "train_rows": int(n),
            "train_ic": train_ic,
            "month_ic": compute_ic(pred, out.loc[cur, "label"].to_numpy()),
        }
        for name, val in zip(names, w):
            rec[f"w_{name}"] = float(val)
        records.append(rec)
    pred_df = out[(out["datetime"] >= TEST_START) & (out["datetime"] < TEST_END)].drop(columns=["_month"]).copy()
    pred_df = add_cross_sectional_norms(pred_df, "pred")
    return pred_df, pd.DataFrame(records)


def apply_static_weights(base: pd.DataFrame, names: list[str], meta: dict[str, object]) -> pd.DataFrame:
    w = np.array([float(meta.get(f"w_{n}", 0.0)) for n in names], dtype=np.float32)
    out = base[["symbol", "datetime", "label"]].copy()
    out["pred"] = scrub(base[names].to_numpy(np.float32)) @ w
    return out


def add_moe_regime(base: pd.DataFrame) -> pd.DataFrame:
    cols = ["symbol", "datetime", "label", "event_score", "group_code", "minute_sin", "minute_cos", "month_sin", "month_cos"]
    meta = pd.read_parquet(
        FACTOR_PATH,
        columns=list(dict.fromkeys(META_COLS)),
        filters=[
            ("datetime", ">=", PRED_START),
            ("datetime", "<", TEST_END),
        ],
    )
    meta["datetime"] = pd.to_datetime(meta["datetime"])
    meta = meta.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    meta, _ = add_context_regime(meta, use_assist=False)
    out = base.merge(meta[cols].drop(columns=["label"]), on=["symbol", "datetime"], how="left")
    out["_month_code"] = month_code(out["datetime"])
    return out


def add_meta_targets(data: pd.DataFrame) -> pd.DataFrame:
    g = data.groupby("datetime", sort=False)["label"]
    mu = g.transform("mean")
    sd = g.transform("std")
    data["label_xsz"] = ((data["label"] - mu) / (sd + 1e-9)).clip(-8, 8).astype(np.float32)
    data["label_xrank"] = (g.rank(pct=True) - 0.5).astype(np.float32)
    return data


def affine_calibrate_static(static_all: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_moe_regime(static_all)
    train = data[(data["datetime"] >= PRED_START) & (data["datetime"] < TEST_START)].copy()
    test = data[(data["datetime"] >= TEST_START) & (data["datetime"] < TEST_END)].copy()

    train_event = train["event_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float64)
    q = np.quantile(train_event, [0.50])
    for df in [train, test]:
        minute = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
        df["_time_bucket"] = pd.cut(
            minute,
            bins=[-1, 615, 765, 900, 10_000],
            labels=[0, 1, 2, 3],
        ).astype(np.int8)
        df["_event_bucket"] = np.searchsorted(q, df["event_score"].fillna(0.0).to_numpy(), side="right").astype(np.int8)
        df["_bucket"] = df["group_code"].astype(np.int16).astype(str) + "_" + df["_time_bucket"].astype(str)

    def fit_xy(df: pd.DataFrame) -> tuple[float, float, int]:
        x = df["pred"].to_numpy(np.float64)
        y = df["label"].to_numpy(np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 1000:
            return 1.0, 0.0, int(m.sum())
        x = x[m]
        y = y[m]
        vx = float(np.mean((x - x.mean()) ** 2))
        if vx < 1e-18:
            return 1.0, 0.0, int(m.sum())
        slope = float(np.mean((x - x.mean()) * (y - y.mean())) / vx)
        intercept = float(y.mean() - slope * x.mean())
        return slope, intercept, int(m.sum())

    gs, gi, gn = fit_xy(train)
    records = [{"bucket": "__global__", "rows": gn, "slope": gs, "intercept": gi, "alpha": 1.0}]
    params: dict[str, tuple[float, float]] = {}
    for bucket, hist in train.groupby("_bucket", sort=True):
        slope, intercept, n = fit_xy(hist)
        if n < 25_000:
            continue
        alpha = float(n / (n + 100_000.0))
        slope = (1.0 - alpha) * gs + alpha * np.clip(slope, 0.35, 1.85)
        intercept = alpha * 0.35 * intercept
        params[str(bucket)] = (float(slope), float(intercept))
        records.append({"bucket": bucket, "rows": n, "slope": slope, "intercept": intercept, "alpha": alpha})

    out = test[["symbol", "datetime", "label", "pred", "_bucket"]].copy()
    pred = out["pred"].to_numpy(np.float64)
    buckets = out["_bucket"].astype(str).to_numpy()
    adj = np.empty(len(out), dtype=np.float32)
    for bucket in np.unique(buckets):
        loc = buckets == bucket
        slope, intercept = params.get(str(bucket), (gs, 0.0))
        adj[loc] = (slope * pred[loc] + intercept).astype(np.float32)
    result = out[["symbol", "datetime", "label"]].copy()
    result["pred"] = adj
    result = add_cross_sectional_norms(result, "pred")
    rec = pd.DataFrame(records)
    rec.attrs["global_train_ic"] = compute_ic(train["pred"].to_numpy(), train["label"].to_numpy())
    rec.attrs["global_slope"] = gs
    return result, rec


def lgb_moe_stacker(base: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    data = add_moe_regime(base)
    data = add_meta_targets(data)
    train = data[(data["datetime"] >= PRED_START) & (data["datetime"] < TEST_START) & data["label"].notna()].copy()
    test = data[(data["datetime"] >= TEST_START) & (data["datetime"] < TEST_END)].copy()
    feature_cols = names + ["event_score", "group_code", "minute_sin", "minute_cos", "month_sin", "month_cos"]
    max_rows = int(os.environ.get("STRICT_OPT_LGB_MOE_ROWS", "900000"))
    rng = np.random.default_rng(int(os.environ.get("STRICT_OPT_LGB_MOE_SEED", "2026")))
    tr_idx = train.index.to_numpy()
    if len(tr_idx) > max_rows:
        tr_pos = stratified_month_label_sample(train, np.arange(len(train)), max_rows, rng, weighted_fill=True)
        train_fit = train.iloc[tr_pos]
    else:
        train_fit = train
    xtr = pd.DataFrame(scrub(train_fit[feature_cols].to_numpy(np.float32)), columns=feature_cols, index=train_fit.index)
    xtr["group_code"] = train_fit["group_code"].to_numpy()
    ytr = train_fit["label_xsz"].to_numpy(np.float32)
    xte = pd.DataFrame(scrub(test[feature_cols].to_numpy(np.float32)), columns=feature_cols, index=test.index)
    xte["group_code"] = test["group_code"].to_numpy()
    params = dict(
        n_estimators=int(os.environ.get("STRICT_OPT_LGB_MOE_TREES", "360")),
        learning_rate=0.025,
        num_leaves=31,
        subsample=0.82,
        colsample_bytree=0.78,
        min_child_samples=450,
        reg_lambda=18.0,
        n_jobs=int(os.environ.get("STRICT_OPT_N_JOBS", "16")),
        random_state=2026,
        verbose=-1,
        force_col_wise=True,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(xtr, ytr, categorical_feature=["group_code"])
    out = test[["symbol", "datetime", "label"]].copy()
    out["pred"] = model.predict(xte).astype(np.float32)
    out = add_cross_sectional_norms(out, "pred")
    print(f"[lgb-moe] train_rows={len(train_fit)} features={len(feature_cols)}", flush=True)
    return out


def bucketed_moe(base: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_moe_regime(base)
    train = data[(data["datetime"] >= PRED_START) & (data["datetime"] < TEST_START)].copy()
    test = data[(data["datetime"] >= TEST_START) & (data["datetime"] < TEST_END)].copy()

    train_event = train["event_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    q = np.quantile(train_event.to_numpy(np.float64), [0.33, 0.66])
    for df in [train, test]:
        minute = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
        df["_time_bucket"] = pd.cut(
            minute,
            bins=[-1, 615, 765, 900, 10_000],
            labels=[0, 1, 2, 3],
        ).astype(np.int8)
        df["_event_bucket"] = np.searchsorted(q, df["event_score"].fillna(0.0).to_numpy(), side="right").astype(np.int8)
        df["_bucket"] = (
            df["group_code"].astype(np.int16).astype(str)
            + "_"
            + df["_time_bucket"].astype(str)
            + "_"
            + df["_event_bucket"].astype(str)
        )

    xg = scrub(train[names].to_numpy(np.float32)).astype(np.float64, copy=False)
    yg = train["label"].to_numpy(np.float64)
    mask = np.isfinite(yg) & np.all(np.isfinite(xg), axis=1)
    lower = np.full(len(names), -0.12, dtype=np.float64)
    upper = np.full(len(names), 0.85, dtype=np.float64)
    wg, global_ic = fit_ic_weights_from_stats(xg[mask].T @ yg[mask], xg[mask].T @ xg[mask], float(yg[mask] @ yg[mask]), lower, upper)

    records = []
    bucket_weights: dict[str, np.ndarray] = {}
    for bucket, hist in train.groupby("_bucket", sort=True):
        if hist["label"].notna().sum() < 15_000:
            continue
        xb = scrub(hist[names].to_numpy(np.float32)).astype(np.float64, copy=False)
        yb = hist["label"].to_numpy(np.float64)
        m = np.isfinite(yb) & np.all(np.isfinite(xb), axis=1)
        if m.sum() < 15_000:
            continue
        wb, bic = fit_ic_weights_from_stats(xb[m].T @ yb[m], xb[m].T @ xb[m], float(yb[m] @ yb[m]), lower, upper, wg)
        alpha = float(m.sum() / (m.sum() + 75_000.0))
        ws = alpha * wb + (1.0 - alpha) * wg
        ws = ws / max(ws.sum(), 1e-12)
        bucket_weights[str(bucket)] = ws
        records.append({"bucket": bucket, "rows": int(m.sum()), "train_ic": bic, "alpha": alpha})

    out = test[["symbol", "datetime", "label", "_bucket"]].copy()
    out["pred"] = np.nan
    xt = scrub(test[names].to_numpy(np.float32))
    for bucket, idx in test.groupby("_bucket", sort=False).groups.items():
        loc = np.asarray(idx)
        w = bucket_weights.get(str(bucket), wg)
        pos = test.index.get_indexer(loc)
        out.loc[loc, "pred"] = xt[pos] @ w.astype(np.float32)
    out = out.drop(columns=["_bucket"])
    out = add_cross_sectional_norms(out, "pred")
    rec = pd.DataFrame(records)
    rec.attrs["global_train_ic"] = global_ic
    return out, rec


def run_moe_and_report() -> None:
    base, names = load_component_panel()
    rows = []
    pred_outputs = {}
    static_meta: dict[str, dict[str, object]] = {}

    for signed in [False, True]:
        tag = "moe_static_signed" if signed else "moe_static_nonneg"
        pred, meta = fit_static_gate(base, names, signed=signed, tag=tag)
        pred.to_parquet(OUT_DIR / f"{tag}.parquet", index=False)
        pd.DataFrame([meta]).to_csv(OUT_DIR / f"{tag}_weights.csv", index=False)
        rows.append(summarize(pred, tag) | {"gate_train_ic_2019": meta["train_ic_2019"]})
        pred_outputs[tag] = pred
        static_meta[tag] = meta

    if os.environ.get("STRICT_OPT_STATIC_ONLY", "0") == "1":
        summary = pd.DataFrame(rows)
        summary.to_csv(OUT_DIR / "moe_summary.csv", index=False)
        monthly_rows = []
        for name, pred_df in pred_outputs.items():
            by_m = period_ic(pred_df, "pred", "M")
            for month, ic in by_m.items():
                monthly_rows.append({"model": name, "month": month, "ic": float(ic)})
        pd.DataFrame(monthly_rows).to_csv(OUT_DIR / "moe_monthly_ic.csv", index=False)
        print(summary[["model", "pred_ic_2019", "pred_ic_2020", "pred_monthly_mean_2020", "gate_train_ic_2019"]].to_string(index=False), flush=True)
        return

    for signed in [False, True]:
        tag = "moe_rolling_signed" if signed else "moe_rolling_nonneg"
        pred, weights = rolling_2020_gate(base, names, signed=signed)
        pred.to_parquet(OUT_DIR / f"{tag}.parquet", index=False)
        weights.to_csv(OUT_DIR / f"{tag}_weights.csv", index=False)
        rows.append(summarize(pred, tag) | {"gate_train_ic_2019": float(weights["train_ic"].iloc[0])})
        pred_outputs[tag] = pred

    if FACTOR_PATH.exists() and os.environ.get("STRICT_OPT_SKIP_FACTOR_MOE", "0") != "1":
        pred, bucket_weights = bucketed_moe(base, names)
        pred.to_parquet(OUT_DIR / "moe_bucketed_dynamic.parquet", index=False)
        bucket_weights.to_csv(OUT_DIR / "moe_bucketed_dynamic_weights.csv", index=False)
        rows.append(summarize(pred, "moe_bucketed_dynamic") | {"gate_train_ic_2019": bucket_weights.attrs.get("global_train_ic")})
        pred_outputs["moe_bucketed_dynamic"] = pred

        static_all = apply_static_weights(base, names, static_meta["moe_static_signed"])
        pred, affine_params = affine_calibrate_static(static_all, "moe_affine_signed")
        pred.to_parquet(OUT_DIR / "moe_affine_signed.parquet", index=False)
        affine_params.to_csv(OUT_DIR / "moe_affine_signed_params.csv", index=False)
        rows.append(summarize(pred, "moe_affine_signed") | {"gate_train_ic_2019": affine_params.attrs.get("global_train_ic")})
        pred_outputs["moe_affine_signed"] = pred

        if os.environ.get("STRICT_OPT_RUN_LGB_MOE", "1") == "1":
            pred = lgb_moe_stacker(base, names)
            pred.to_parquet(OUT_DIR / "moe_lgb_stacker.parquet", index=False)
            rows.append(summarize(pred, "moe_lgb_stacker") | {"gate_train_ic_2019": float("nan")})
            pred_outputs["moe_lgb_stacker"] = pred
    else:
        reason = "disabled by STRICT_OPT_SKIP_FACTOR_MOE" if FACTOR_PATH.exists() else f"factor panel is missing: {FACTOR_PATH}"
        print(f"[moe] skip regime/meta MOE because {reason}", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "moe_summary.csv", index=False)

    monthly_rows = []
    for name, pred_df in pred_outputs.items():
        by_m = period_ic(pred_df, "pred", "M")
        for month, ic in by_m.items():
            monthly_rows.append({"model": name, "month": month, "ic": float(ic)})
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(OUT_DIR / "moe_monthly_ic.csv", index=False)
    plot_monthly(monthly, OUT_DIR / "moe_monthly_ic.png")
    print(summary[["model", "pred_ic_2019", "pred_ic_2020", "pred_monthly_mean_2020", "gate_train_ic_2019"]].to_string(index=False), flush=True)


def plot_monthly(monthly: pd.DataFrame, path: Path) -> None:
    if monthly.empty:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pivot = monthly.pivot(index="month", columns="model", values="ic")
    ax = pivot.plot(figsize=(14, 5), marker="o", linewidth=1.5)
    ax.axhline(0.07, color="firebrick", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xlabel("month")
    ax.set_ylabel("IC")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def parse_variants() -> list[Variant]:
    wanted = os.environ.get("STRICT_OPT_VARIANTS", "").strip()
    if not wanted:
        return DEFAULT_VARIANTS
    if wanted in {"__none__", "none", "NONE"}:
        return []
    names = {x.strip() for x in wanted.split(",") if x.strip()}
    out = [v for v in DEFAULT_VARIANTS if v.name in names]
    missing = names - {v.name for v in out}
    if missing:
        raise ValueError(f"unknown variants: {sorted(missing)}")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    variants = parse_variants()
    summaries = []
    monthly_rows = []
    for variant in variants:
        pred = run_variant(variant)
        summaries.append(summarize(pred, variant.name) | {"ablation_type": "base_learner"})
        by_m = period_ic(pred[(pred["datetime"] >= TEST_START) & (pred["datetime"] < TEST_END)], "pred", "M")
        for month, ic in by_m.items():
            monthly_rows.append({"model": variant.name, "month": month, "ic": float(ic)})
    if summaries:
        cur = pd.DataFrame(summaries)
        path = OUT_DIR / "base_ablation_summary.csv"
        if path.exists():
            old = pd.read_csv(path)
            cur = pd.concat([old[~old["model"].isin(cur["model"])], cur], ignore_index=True)
        cur.to_csv(path, index=False)
        monthly = pd.DataFrame(monthly_rows)
        if not monthly.empty:
            monthly.to_csv(OUT_DIR / "base_monthly_ic.csv", index=False)
            plot_monthly(monthly, FIG_DIR / "base_monthly_ic.png")
        print(cur[["model", "pred_ic_2019", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020"]].to_string(index=False), flush=True)
    if os.environ.get("STRICT_OPT_RUN_MOE", "1") == "1":
        run_moe_and_report()


if __name__ == "__main__":
    main()
