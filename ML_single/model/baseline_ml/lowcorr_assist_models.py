#!/usr/bin/env python3
"""Low-correlation assist learners built from causal meta/chain features."""

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
from sklearn.linear_model import Ridge

from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic
from src.plan_a.group_lgb import symbol_group_map
from strict_optimization_ablation import (
    FACTOR_PATH,
    META_COLS,
    OUT_DIR,
    PRED_START,
    TEST_END,
    TEST_START,
    TRAIN_START,
    summarize,
)


EMBARGO_BARS = int(os.environ.get("ASSIST_EMBARGO_BARS", "30"))
MAX_ROWS = int(os.environ.get("ASSIST_MAX_ROWS", "520000"))
OUT_PREFIX = "lowcorr"
META_CACHE = OUT_DIR / "lowcorr_meta_features.parquet"


@dataclass(frozen=True)
class AssistVariant:
    name: str
    model: str = "ridge"
    target: str = "label_xsz"
    feature_set: str = "all"
    alpha: float = 80.0
    seed: int = 7301
    max_rows: int = MAX_ROWS
    n_estimators: int = 120
    learning_rate: float = 0.035
    num_leaves: int = 31
    min_child_samples: int = 180
    reg_lambda: float = 12.0
    colsample_bytree: float = 0.86
    subsample: float = 0.85
    embargo_bars: int = EMBARGO_BARS


VARIANTS = [
    AssistVariant(name="lowcorr_ridge_meta_chain_xsz", model="ridge", feature_set="all", alpha=90.0),
    AssistVariant(name="lowcorr_ridge_chain_only_xsz", model="ridge", feature_set="chain", alpha=70.0),
    AssistVariant(name="lowcorr_lgb_meta_chain_xsz", model="lgb", feature_set="all", max_rows=280000),
    AssistVariant(name="lowcorr_lgb_fwd15_xsz", model="lgb", target="fwd15_xsz", feature_set="all", max_rows=280000, embargo_bars=15),
    AssistVariant(
        name="lowcorr_lgb_fwd15_light_xsz",
        model="lgb",
        target="fwd15_xsz",
        feature_set="all",
        max_rows=160000,
        n_estimators=80,
        min_child_samples=240,
        reg_lambda=14.0,
        colsample_bytree=0.82,
        embargo_bars=15,
    ),
    AssistVariant(name="lowcorr_lgb_fwd60_xsz", model="lgb", target="fwd60_xsz", feature_set="all", max_rows=280000, embargo_bars=60),
]


CHAIN_GROUPS = {
    "ferrous": ["RB", "HC", "I", "J", "JM", "ZC", "SF", "SM", "FG"],
    "oils": ["A", "M", "Y", "P", "OI", "RM", "B"],
    "metals": ["CU", "AL", "ZN", "PB", "NI", "SN", "AU", "AG"],
    "energy_chem": ["FU", "BU", "RU", "L", "PP", "V", "TA", "MA"],
    "agri": ["C", "CS", "JD", "AP", "CF", "SR"],
}


def parse_variants() -> list[AssistVariant]:
    wanted = os.environ.get("ASSIST_VARIANTS", "").strip()
    if not wanted:
        return VARIANTS
    names = {x.strip() for x in wanted.split(",") if x.strip()}
    out = [v for v in VARIANTS if v.name in names]
    missing = names - {v.name for v in out}
    if missing:
        raise ValueError(f"unknown assist variants: {sorted(missing)}")
    return out


def month_starts() -> list[pd.Timestamp]:
    return list(pd.date_range(PRED_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS"))


def scrub_array(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def write_parquet_atomic(data: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        data.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def xsec_z(data: pd.DataFrame, col: str) -> pd.Series:
    g = data.groupby("datetime", sort=False)[col]
    return ((data[col] - g.transform("mean")) / (g.transform("std") + 1e-9)).astype(np.float32)


def add_ret_features(data: pd.DataFrame) -> pd.DataFrame:
    close = data["close"].astype(np.float64).clip(lower=1e-12)
    by_symbol = data.groupby("symbol", sort=False)
    data["ret1"] = np.log(close / by_symbol["close"].shift(1).astype(np.float64).clip(lower=1e-12))
    data["ret1"] = data["ret1"].mask(data["is_long_break_before"].fillna(False), 0.0)
    data["ret1"] = data["ret1"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    for horizon in [3, 6, 12, 24, 48]:
        prev = by_symbol["close"].shift(horizon).astype(np.float64).clip(lower=1e-12)
        col = f"ret{horizon}"
        data[col] = np.log(close / prev).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    open_ = data["open"].astype(np.float64).clip(lower=1e-12)
    high = data["high"].astype(np.float64)
    low = data["low"].astype(np.float64)
    data["intrabar_ret"] = np.log(close / open_).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    data["range_rel"] = ((high - low) / close).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    data["body_rel"] = ((close - open_) / open_).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    data["log_volume"] = np.log1p(data["volume"].clip(lower=0)).astype(np.float32)
    data["log_amount"] = np.log1p(data["amount"].clip(lower=0)).astype(np.float32)
    oi_log = np.log1p(data["oi"].clip(lower=0)).astype(np.float64)
    data["oi_chg"] = by_symbol["oi"].transform(lambda s: np.log1p(s.clip(lower=0)).diff()).fillna(0.0).astype(np.float32)
    data["oi_chg"] = data["oi_chg"].mask(data["is_long_break_before"].fillna(False), 0.0).astype(np.float32)
    del oi_log
    return data


def add_cross_group_features(data: pd.DataFrame) -> pd.DataFrame:
    for col in ["ret1", "ret3", "ret6", "ret12", "ret24", "ret48", "range_rel", "log_amount", "oi_chg"]:
        data[f"x_{col}"] = xsec_z(data, col)

    groups = symbol_group_map()
    data["group"] = data["symbol"].map(groups).fillna("other")
    chain_map = {}
    for group, symbols in CHAIN_GROUPS.items():
        for symbol in symbols:
            chain_map[symbol] = group
    data["chain_group"] = data["symbol"].map(chain_map).fillna(data["group"])

    for col in ["ret1", "ret6", "ret24", "log_amount", "range_rel"]:
        grp_col = f"grp_{col}_mean"
        data[grp_col] = data.groupby(["datetime", "group"], sort=False)[col].transform("mean").astype(np.float32)
        data[f"rel_{col}_grp"] = (data[col] - data[grp_col]).astype(np.float32)
        sum_col = data.groupby(["datetime", "chain_group"], sort=False)[col].transform("sum")
        cnt_col = data.groupby(["datetime", "chain_group"], sort=False)[col].transform("count")
        peer = ((sum_col - data[col]) / (cnt_col - 1).replace(0, np.nan)).fillna(data[grp_col])
        data[f"chain_peer_{col}"] = peer.astype(np.float32)
        data[f"rel_{col}_chain"] = (data[col] - data[f"chain_peer_{col}"]).astype(np.float32)

    data["market_ret1"] = data.groupby("datetime", sort=False)["ret1"].transform("mean").astype(np.float32)
    data["market_ret24"] = data.groupby("datetime", sort=False)["ret24"].transform("mean").astype(np.float32)
    data["rel_ret1_mkt"] = (data["ret1"] - data["market_ret1"]).astype(np.float32)
    data["rel_ret24_mkt"] = (data["ret24"] - data["market_ret24"]).astype(np.float32)

    by_symbol = data.groupby("symbol", sort=False)
    for col in ["grp_ret1_mean", "grp_ret6_mean", "chain_peer_ret1", "chain_peer_ret6", "market_ret1"]:
        data[f"{col}_lag1"] = by_symbol[col].shift(1).fillna(0.0).astype(np.float32)
        data[f"{col}_lag3"] = by_symbol[col].shift(3).fillna(0.0).astype(np.float32)
    return data


def add_time_targets(data: pd.DataFrame) -> pd.DataFrame:
    close = data["close"].astype(np.float64).clip(lower=1e-12)
    by_symbol = data.groupby("symbol", sort=False)
    for horizon in [5, 15, 60]:
        fwd_close = by_symbol["close"].shift(-horizon).astype(np.float64).clip(lower=1e-12)
        fwd = np.log(fwd_close / close).replace([np.inf, -np.inf], np.nan)
        data[f"fwd{horizon}"] = fwd.astype(np.float32)

    g = data.groupby("datetime", sort=False)["label"]
    mu = g.transform("mean")
    sd = g.transform("std")
    rank = g.rank(pct=True).astype(np.float32)
    data["label_xsz"] = ((data["label"] - mu) / (sd + 1e-9)).clip(-8, 8).astype(np.float32)
    data["label_ranknorm"] = ndtri(rank.clip(0.01, 0.99)).astype(np.float32)
    for horizon in [5, 15, 60]:
        col = f"fwd{horizon}"
        g = data.groupby("datetime", sort=False)[col]
        mu = g.transform("mean")
        sd = g.transform("std")
        data[f"{col}_xsz"] = ((data[col] - mu) / (sd + 1e-9)).clip(-8, 8).astype(np.float32)

    minute = (data["datetime"].dt.hour * 60 + data["datetime"].dt.minute).astype(np.float32)
    data["minute_sin"] = np.sin(2 * np.pi * minute / 1440.0).astype(np.float32)
    data["minute_cos"] = np.cos(2 * np.pi * minute / 1440.0).astype(np.float32)
    dow = data["datetime"].dt.dayofweek.astype(np.float32)
    data["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    data["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    data["month"] = data["datetime"].dt.to_period("M").dt.to_timestamp()
    data["_bar_no"] = data.groupby("symbol", sort=False).cumcount().astype(np.int32)
    return data


def build_meta_features() -> pd.DataFrame:
    if META_CACHE.exists() and os.environ.get("ASSIST_REBUILD_META", "0") != "1":
        data = pd.read_parquet(META_CACHE)
        data["datetime"] = pd.to_datetime(data["datetime"])
        data["month"] = pd.to_datetime(data["month"])
        return data

    cols = [c for c in META_COLS if c in {"symbol", "datetime", "label", "is_long_break_before", "close", "open", "high", "low", "volume", "amount", "oi"}]
    data = pd.read_parquet(
        FACTOR_PATH,
        columns=cols,
        filters=[("datetime", ">=", TRAIN_START), ("datetime", "<", TEST_END)],
    )
    data["symbol"] = data["symbol"].astype(str)
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    data = add_ret_features(data)
    data = data.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    data = add_cross_group_features(data)
    data = data.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    data = add_time_targets(data)
    keep_cols = [
        "symbol",
        "datetime",
        "month",
        "label",
        "label_xsz",
        "label_ranknorm",
        "fwd5_xsz",
        "fwd15_xsz",
        "fwd60_xsz",
        "_bar_no",
    ] + feature_columns("all")
    data = data[list(dict.fromkeys(keep_cols))]
    for col in data.columns:
        if col not in {"symbol", "datetime", "month"}:
            data[col] = pd.to_numeric(data[col], errors="coerce").astype(np.float32 if col != "_bar_no" else np.int32)
    write_parquet_atomic(data, META_CACHE)
    return data


def feature_columns(feature_set: str) -> list[str]:
    base = [
        "ret1",
        "ret3",
        "ret6",
        "ret12",
        "ret24",
        "ret48",
        "intrabar_ret",
        "range_rel",
        "body_rel",
        "log_volume",
        "log_amount",
        "oi_chg",
        "x_ret1",
        "x_ret3",
        "x_ret6",
        "x_ret12",
        "x_ret24",
        "x_ret48",
        "x_range_rel",
        "x_log_amount",
        "x_oi_chg",
        "market_ret1",
        "market_ret24",
        "rel_ret1_mkt",
        "rel_ret24_mkt",
        "minute_sin",
        "minute_cos",
        "dow_sin",
        "dow_cos",
    ]
    chain = [
        "grp_ret1_mean",
        "grp_ret6_mean",
        "grp_ret24_mean",
        "grp_log_amount_mean",
        "grp_range_rel_mean",
        "rel_ret1_grp",
        "rel_ret6_grp",
        "rel_ret24_grp",
        "rel_log_amount_grp",
        "rel_range_rel_grp",
        "chain_peer_ret1",
        "chain_peer_ret6",
        "chain_peer_ret24",
        "chain_peer_log_amount",
        "chain_peer_range_rel",
        "rel_ret1_chain",
        "rel_ret6_chain",
        "rel_ret24_chain",
        "rel_log_amount_chain",
        "rel_range_rel_chain",
        "grp_ret1_mean_lag1",
        "grp_ret1_mean_lag3",
        "grp_ret6_mean_lag1",
        "grp_ret6_mean_lag3",
        "chain_peer_ret1_lag1",
        "chain_peer_ret1_lag3",
        "chain_peer_ret6_lag1",
        "chain_peer_ret6_lag3",
        "market_ret1_lag1",
        "market_ret1_lag3",
    ]
    if feature_set == "chain":
        return chain + ["market_ret1", "market_ret24", "rel_ret1_mkt", "rel_ret24_mkt"]
    return base + chain


def first_bar_by_month(data: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    first = data.groupby(["month", "symbol"], sort=False)["_bar_no"].min().reset_index()
    out = {}
    for month, sub in first.groupby("month", sort=False):
        out[pd.Timestamp(month)] = sub.set_index("symbol")["_bar_no"]
    return out


def train_mask_for_month(
    data: pd.DataFrame,
    ms: pd.Timestamp,
    first_pos: dict[pd.Timestamp, pd.Series],
    target: str,
    embargo_bars: int,
) -> np.ndarray:
    mask = (data["datetime"] >= TRAIN_START) & (data["datetime"] < ms) & data[target].notna()
    starts = first_pos.get(ms)
    if starts is None:
        return mask.to_numpy()
    cutoff = data["symbol"].map(starts).astype("float64")
    keep = cutoff.isna() | (data["_bar_no"].astype("float64") < cutoff - embargo_bars)
    return (mask & keep).to_numpy()


def sample_indices(data: pd.DataFrame, mask: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) <= max_rows:
        return idx
    rng = np.random.default_rng(seed)
    event_cols = ["x_ret1", "x_range_rel", "x_log_amount", "x_oi_chg"]
    scores = np.zeros(len(idx), dtype=np.float32)
    for col in event_cols:
        scores += np.abs(data[col].to_numpy(np.float32)[idx])
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    event_need = min(len(idx), max_rows // 4)
    weights = np.sqrt(scores.clip(min=0) + 0.05)
    weights /= weights.sum()
    event_pick = rng.choice(idx, event_need, replace=False, p=weights)
    picked = np.zeros(len(data), dtype=bool)
    picked[event_pick] = True
    rest = idx[~picked[idx]]
    rest_pick = rng.choice(rest, min(len(rest), max_rows - event_need), replace=False)
    return np.sort(np.concatenate([event_pick, rest_pick]))


def fit_predict_month(
    data: pd.DataFrame,
    variant: AssistVariant,
    feat_cols: list[str],
    first_pos: dict[pd.Timestamp, pd.Series],
    ms: pd.Timestamp,
) -> pd.DataFrame:
    train_mask = train_mask_for_month(data, ms, first_pos, variant.target, variant.embargo_bars)
    tr_idx = sample_indices(data, train_mask, variant.max_rows, variant.seed + int(ms.year * 12 + ms.month))
    test_mask = (data["datetime"] >= ms) & (data["datetime"] < ms + pd.DateOffset(months=1))
    te_idx = np.flatnonzero(test_mask.to_numpy())
    xtr = scrub_array(data.iloc[tr_idx][feat_cols].to_numpy(np.float32))
    ytr = scrub_array(data.iloc[tr_idx][variant.target].to_numpy(np.float32))
    xte = scrub_array(data.iloc[te_idx][feat_cols].to_numpy(np.float32))

    if variant.model == "ridge":
        mu = xtr.mean(axis=0, dtype=np.float64).astype(np.float32)
        sd = xtr.std(axis=0, dtype=np.float64).astype(np.float32)
        sd[sd < 1e-6] = 1.0
        xtr = (xtr - mu) / sd
        xte = (xte - mu) / sd
        model = Ridge(alpha=variant.alpha, fit_intercept=True, random_state=variant.seed)
        model.fit(xtr, ytr)
        pred = model.predict(xte)
    elif variant.model == "lgb":
        model = lgb.LGBMRegressor(
            n_estimators=variant.n_estimators,
            learning_rate=variant.learning_rate,
            num_leaves=variant.num_leaves,
            subsample=variant.subsample,
            colsample_bytree=variant.colsample_bytree,
            min_child_samples=variant.min_child_samples,
            reg_lambda=variant.reg_lambda,
            n_jobs=int(os.environ.get("ASSIST_N_JOBS", "1")),
            random_state=variant.seed,
            verbose=-1,
            force_col_wise=True,
        )
        model.fit(xtr, ytr)
        pred = model.predict(xte)
    else:
        raise ValueError(f"unknown model: {variant.model}")

    out = data.iloc[te_idx][["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype(np.float32)
    mic = compute_ic(out["pred"].to_numpy(), out["label"].to_numpy())
    print(f"[assist][{variant.name}][{ms:%Y-%m}] tr={len(tr_idx):7d} pr={len(out):6d} IC={mic:.5f}", flush=True)
    del xtr, xte, ytr, model
    gc.collect()
    return out


def run_variant(data: pd.DataFrame, variant: AssistVariant) -> pd.DataFrame:
    pred_path = OUT_DIR / f"{variant.name}.parquet"
    parts_dir = OUT_DIR / f"{variant.name}_month_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    if pred_path.exists() and os.environ.get("ASSIST_FORCE", "0") != "1":
        return pd.read_parquet(pred_path)
    feat_cols = feature_columns(variant.feature_set)
    first_pos = first_bar_by_month(data)
    pieces = []
    one_month = os.environ.get("ASSIST_ONE_MONTH", "0") == "1"
    if one_month:
        missing = next((ms for ms in month_starts() if not (parts_dir / f"{ms:%Y-%m}.parquet").exists()), None)
        if missing is not None:
            part = fit_predict_month(data, variant, feat_cols, first_pos, missing)
            write_parquet_atomic(part, parts_dir / f"{missing:%Y-%m}.parquet")
            print(f"[assist-one-month][{variant.name}] wrote {missing:%Y-%m}", flush=True)
            return pd.DataFrame()
        pieces = [pd.read_parquet(parts_dir / f"{ms:%Y-%m}.parquet") for ms in month_starts()]
    else:
        for ms in month_starts():
            part_path = parts_dir / f"{ms:%Y-%m}.parquet"
            if part_path.exists() and os.environ.get("ASSIST_FORCE", "0") != "1":
                part = pd.read_parquet(part_path)
            else:
                part = fit_predict_month(data, variant, feat_cols, first_pos, ms)
                write_parquet_atomic(part, part_path)
            pieces.append(part)
    pred = pd.concat(pieces, ignore_index=True)
    pred = add_cross_sectional_norms(pred, "pred")
    write_parquet_atomic(pred, pred_path)
    (OUT_DIR / f"{variant.name}.json").write_text(json.dumps(asdict(variant), indent=2), encoding="utf-8")
    return pred


def write_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = OUT_DIR / "base_ablation_summary.csv"
    cur = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        cur = pd.concat([old[~old["model"].isin(cur["model"])], cur], ignore_index=True)
    cur.to_csv(path, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_meta_features()
    rows = []
    for variant in parse_variants():
        print(f"[assist-variant] {variant.name} model={variant.model} features={variant.feature_set}", flush=True)
        pred = run_variant(data, variant)
        if pred.empty:
            continue
        row = summarize(pred, variant.name) | {"ablation_type": "lowcorr_assist"}
        rows.append(row)
        cols = ["model", "pred_ic_2019", "pred_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020"]
        print(pd.DataFrame([row])[cols].to_string(index=False), flush=True)
    write_summary(rows)


if __name__ == "__main__":
    main()
