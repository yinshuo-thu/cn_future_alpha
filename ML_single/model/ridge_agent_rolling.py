from __future__ import annotations

import argparse
import fnmatch
import gc
import hashlib
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

FU_ROOT = Path("/root/autodl-tmp/fu-alpha-research")
FU_SRC = FU_ROOT / "src"
if str(FU_SRC) not in sys.path:
    sys.path.insert(0, str(FU_SRC))

from fu_alpha_research.config import load_config  # noqa: E402
from fu_alpha_research.feature_matrix import FeatureMatrix  # noqa: E402


CONFIG_PATH = FU_ROOT / "configs/futures.yaml"
FEATURE_SET_DIR = FU_ROOT / "outputs/model_feature_sets"
META_KEEP = ["symbol", "datetime", "label"]
TRAIN_AUX_COLS = ["label_xsz", "label_xrank", "label_ranknorm", "event_score", "_bars_to_month_end"]
PRED_COLS = ["pred", "pred_xsz", "pred_xrank"]


@dataclass(frozen=True)
class RidgeAgentState:
    feature_cols: list[str]
    target_col: str
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    y_mean: float
    alpha: float
    top_k: int
    clip_z: float


@dataclass(frozen=True)
class RidgeAgentVariant:
    name: str
    feature_set: str
    target_col: str = "label_xsz"
    max_train_rows: int = 600_000
    sample_mode: str = "soft_event"
    alpha: float = 1.0
    top_k: int = 0
    seed: int = 20260624
    lookback_months: int = 0
    half_life_months: float = 12.0
    clip_z: float = 6.0


def read_list(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def month_range(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def scrub_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.array(x, dtype=np.float32, copy=True)
    return np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def compute_ic(pred: np.ndarray | pd.Series, label: np.ndarray | pd.Series) -> float:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(p) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    p = p[mask]
    y = y[mask]
    denom = math.sqrt(float(np.mean(p * p) * np.mean(y * y)))
    if denom <= 1e-18:
        return float("nan")
    return float(np.mean(p * y) / denom)


def monthly_ic(df: pd.DataFrame, pred_col: str) -> pd.Series:
    clean = df.dropna(subset=[pred_col, "label"]).copy()
    if clean.empty:
        return pd.Series(dtype=float)
    clean["_month"] = clean["datetime"].dt.to_period("M").astype(str)
    return clean.groupby("_month", sort=True).apply(
        lambda g: compute_ic(g[pred_col].to_numpy(), g["label"].to_numpy()),
        include_groups=False,
    )


def summarize_predictions(df: pd.DataFrame, model: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "rows": int(len(df)),
        "label_rows": int(df["label"].notna().sum()),
    }
    windows = {
        "2019_2020": (pd.Timestamp("2019-01-01"), pd.Timestamp("2021-01-01")),
        "2019": (pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01")),
        "2020": (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")),
    }
    for pred_col in PRED_COLS:
        if pred_col not in df.columns:
            continue
        for suffix, (start, end) in windows.items():
            part = df[(df["datetime"] >= start) & (df["datetime"] < end)]
            mic = monthly_ic(part, pred_col)
            row[f"{pred_col}_ic_{suffix}"] = compute_ic(part[pred_col].to_numpy(), part["label"].to_numpy())
            row[f"{pred_col}_monthly_mean_{suffix}"] = float(mic.mean()) if len(mic) else float("nan")
            row[f"{pred_col}_monthly_std_{suffix}"] = float(mic.std(ddof=1)) if len(mic) > 1 else float("nan")
            std = row[f"{pred_col}_monthly_std_{suffix}"]
            row[f"{pred_col}_monthly_ir_{suffix}"] = (
                row[f"{pred_col}_monthly_mean_{suffix}"] / std
                if np.isfinite(std) and std > 0
                else float("nan")
            )
    return row


def add_prediction_views(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("datetime", sort=False)["pred"]
    out["pred_xsz"] = ((out["pred"] - g.transform("mean")) / (g.transform("std") + 1e-9)).astype(np.float32)
    out["pred_xrank"] = (g.rank(pct=True) - 0.5).astype(np.float32)
    return out


def add_label_views_and_sampling_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("datetime", sort=False)["label"]
    mu = g.transform("mean")
    sd = g.transform("std")
    rank = g.rank(pct=True).astype(np.float32)
    out["label_xsz"] = ((out["label"] - mu) / (sd + 1e-9)).clip(-8, 8).astype(np.float32)
    out["label_xrank"] = (rank - 0.5).astype(np.float32)

    try:
        from scipy.special import ndtri

        out["label_ranknorm"] = ndtri(rank.clip(0.01, 0.99)).astype(np.float32)
    except Exception:
        out["label_ranknorm"] = out["label_xrank"].astype(np.float32)

    close = out["close"].astype(np.float64).abs().clip(lower=1e-12)
    open_ = out["open"].astype(np.float64).abs().clip(lower=1e-12)
    high = out["high"].astype(np.float64)
    low = out["low"].astype(np.float64)
    intrabar = np.log(close / open_).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    range_rel = ((high - low) / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    log_amount = np.log1p(out["amount"].clip(lower=0)).astype(np.float64)

    def xsec_abs(s: pd.Series) -> pd.Series:
        sg = s.groupby(out["datetime"], sort=False)
        z = (s - sg.transform("mean")) / (sg.transform("std") + 1e-9)
        return z.abs().clip(0, 8)

    out["event_score"] = (xsec_abs(intrabar) + xsec_abs(range_rel) + 0.5 * xsec_abs(log_amount)).astype(np.float32)
    pos = out.groupby("symbol", sort=False).cumcount()
    size = out.groupby("symbol", sort=False)["datetime"].transform("size")
    out["_bars_to_month_end"] = (size - pos - 1).astype(np.int32)
    return out


def stratified_pick(data: pd.DataFrame, pool: np.ndarray, need: int, rng: np.random.Generator) -> np.ndarray:
    if need <= 0 or len(pool) == 0:
        return np.empty(0, dtype=np.int64)
    if len(pool) <= need:
        return pool.astype(np.int64, copy=False)
    ranks = np.nan_to_num(data["label_xrank"].to_numpy(np.float32)[pool], nan=0.0)
    bins = np.floor(np.clip((ranks + 0.5) * 8.0, 0, 7)).astype(np.int16)
    pieces: list[np.ndarray] = []
    per = max(1, need // 8)
    for b in range(8):
        loc = pool[bins == b]
        if len(loc):
            pieces.append(rng.choice(loc, min(len(loc), per), replace=False))
    used = sum(len(x) for x in pieces)
    if used < need:
        already = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
        taken = np.zeros(len(data), dtype=bool)
        taken[already] = True
        rest = pool[~taken[pool]]
        if len(rest):
            pieces.append(rng.choice(rest, min(need - used, len(rest)), replace=False))
    out = np.concatenate(pieces) if pieces else pool
    if len(out) > need:
        out = rng.choice(out, need, replace=False)
    return np.sort(out.astype(np.int64, copy=False))


def sample_rows(data: pd.DataFrame, cap: int, mode: str, seed: int) -> pd.DataFrame:
    pool = np.flatnonzero(data["label"].notna().to_numpy() & data["label_xrank"].notna().to_numpy())
    if cap <= 0 or len(pool) <= cap:
        return data.iloc[pool].copy()
    rng = np.random.default_rng(seed)
    if mode == "random":
        idx = rng.choice(pool, cap, replace=False)
    elif mode == "stratified":
        idx = stratified_pick(data, pool, cap, rng)
    else:
        frac = {"soft_event": 0.25, "event35": 0.35, "event50": 0.50}.get(mode)
        if frac is None:
            raise ValueError(f"bad sample mode: {mode}")
        scores = np.nan_to_num(data["event_score"].to_numpy(np.float32)[pool], nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.sqrt(np.maximum(scores, 0.0) + 0.05)
        weights = weights / weights.sum()
        event_need = min(len(pool), int(cap * frac))
        event_pick = rng.choice(pool, event_need, replace=False, p=weights)
        used = np.zeros(len(data), dtype=bool)
        used[event_pick] = True
        rest = pool[~used[pool]]
        rest_pick = stratified_pick(data, rest, cap - len(event_pick), rng)
        idx = np.concatenate([event_pick, rest_pick])
    return data.iloc[np.sort(idx)].copy()


def build_feature_matrix() -> FeatureMatrix:
    cfg = load_config(CONFIG_PATH)
    candidates = [
        cfg.factor_panel_path,
        Path("/root/autodl-tmp/shared-nvme/feature_model/data_factors_big.parquet"),
        Path("/root/shared-nvme/feature_model/data_factors_big.parquet"),
    ]
    for path in candidates:
        if path.exists():
            cfg = replace(cfg, factor_panel_path=path)
            break
    expr_candidates = [
        FU_ROOT / "outputs/expression_sets/combined_for_new1000_models.csv",
        FU_ROOT / "outputs/expression_sets/new100.csv",
    ]
    expression_path = next((path for path in expr_candidates if path.exists()), None)
    return FeatureMatrix(cfg, expression_path=expression_path)


def transform_features(df: pd.DataFrame, features: list[str], feature_transform: str) -> pd.DataFrame:
    if feature_transform == "raw":
        return df
    if feature_transform != "xsz":
        raise ValueError(f"bad feature_transform={feature_transform!r}")
    out = df.copy()
    g = out.groupby("datetime", sort=False)[features]
    mu = g.transform("mean")
    sd = g.transform("std")
    out.loc[:, features] = ((out[features] - mu) / (sd + 1e-8)).astype(np.float32)
    return out


def feature_hash(features: list[str]) -> str:
    payload = "\n".join(features).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def ensure_month_cache(
    fm: FeatureMatrix,
    months: list[str],
    features: list[str],
    cache_rows: int,
    rebuild: bool,
    seed: int,
    feature_transform: str,
    out_dir: Path,
    cache_name: str,
) -> Path:
    cache_dir = out_dir / f"sample_cache_{cache_name}_{feature_transform}_f{len(features)}_{feature_hash(features)}_m{cache_rows}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = META_KEEP + features + TRAIN_AUX_COLS
    for i, month in enumerate(months):
        path = cache_dir / f"{month}.parquet"
        if path.exists() and not rebuild:
            continue
        print(f"[cache][{month}] reading {len(features)} features", flush=True)
        df = fm.read_month(month, features)
        df = transform_features(df, features, feature_transform)
        df = add_label_views_and_sampling_cols(df)
        sample = sample_rows(df[keep_cols], cache_rows, "soft_event", seed + i)
        write_parquet_atomic(sample, path)
        print(f"[cache][{month}] sampled rows={len(sample)} -> {path}", flush=True)
        del df, sample
        gc.collect()
    return cache_dir


def load_train_samples(
    cache_dir: Path,
    all_train_months: list[str],
    test_month: str,
    features: list[str],
    target_col: str,
    max_rows: int,
    sample_mode: str,
    seed: int,
    embargo_bars: int,
    lookback_months: int,
) -> pd.DataFrame:
    test_period = pd.Period(test_month, freq="M")
    prev_month = str(test_period - 1)
    start_period = pd.Period(all_train_months[0], freq="M")
    if lookback_months > 0:
        start_period = max(start_period, test_period - lookback_months)
    read_cols = list(dict.fromkeys(META_KEEP + features + TRAIN_AUX_COLS + [target_col]))
    pieces = []
    for month in all_train_months:
        period = pd.Period(month, freq="M")
        if period < start_period:
            continue
        if period >= test_period:
            break
        path = cache_dir / f"{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_parquet(path, columns=read_cols)
        if embargo_bars > 0 and month == prev_month:
            part = part[part["_bars_to_month_end"] >= embargo_bars].copy()
        pieces.append(part)
    if not pieces:
        return pd.DataFrame()
    train = pd.concat(pieces, ignore_index=True)
    if len(train) > max_rows:
        train = sample_rows(train, max_rows, sample_mode, seed + int(test_period.year * 12 + test_period.month))
    return train


def recency_weights(train: pd.DataFrame, test_month: str, half_life_months: float) -> np.ndarray | None:
    if half_life_months <= 0:
        return None
    test_period = pd.Period(test_month, freq="M")
    periods = train["datetime"].dt.to_period("M")
    age = np.array([test_period.ordinal - p.ordinal for p in periods], dtype=np.float64)
    w = np.exp(-np.log(2.0) * np.maximum(age, 0.0) / half_life_months)
    return w.astype(np.float64)


def weighted_mean_scale(x: np.ndarray, w: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if w is None:
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(x.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    else:
        mean = (w[:, None] * x).sum(axis=0).astype(np.float32)
        var = (w[:, None] * (x - mean) ** 2).sum(axis=0)
        scale = np.maximum(np.sqrt(var), 1e-6).astype(np.float32)
    return mean, scale


def fit_ridge_agent(
    train: pd.DataFrame,
    features: list[str],
    target_col: str,
    alpha: float,
    top_k: int,
    clip_z: float,
    sample_weight: np.ndarray | None = None,
) -> RidgeAgentState:
    y = train[target_col].to_numpy(np.float64, copy=False)
    mask = np.isfinite(y)
    x = scrub_matrix(train.loc[mask, features].to_numpy(np.float32, copy=False))
    y = y[mask]
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64)[mask]
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            w = None
        else:
            w = w / w.sum()
    else:
        w = None

    mean, scale = weighted_mean_scale(x, w)
    xz = ((x - mean) / scale).astype(np.float32)
    if clip_z > 0:
        np.clip(xz, -clip_z, clip_z, out=xz)
    y_mean = float(y.mean()) if w is None else float(np.sum(w * y))
    y0 = y - y_mean
    if w is None:
        n = max(len(xz), 1)
        gram = (xz.T @ xz).astype(np.float64) / n
        cov = (xz.T @ y0).astype(np.float64) / n
    else:
        xzw = xz.astype(np.float64) * w[:, None]
        gram = xzw.T @ xz.astype(np.float64)
        cov = xzw.T @ y0
    if top_k and top_k < len(features):
        keep = np.argpartition(np.abs(cov), -top_k)[-top_k:]
        w_sub = np.linalg.solve(gram[np.ix_(keep, keep)] + alpha * np.eye(len(keep)), cov[keep])
        weight = np.zeros(len(features), dtype=np.float64)
        weight[keep] = w_sub
    else:
        weight = np.linalg.solve(gram + alpha * np.eye(len(features)), cov)
    return RidgeAgentState(features, target_col, mean, scale, weight.astype(np.float32), y_mean, alpha, top_k, clip_z)


def predict_ridge_agent(model: RidgeAgentState, df: pd.DataFrame, chunk_rows: int = 180_000) -> np.ndarray:
    pred = np.empty(len(df), dtype=np.float32)
    for start in range(0, len(df), chunk_rows):
        end = min(start + chunk_rows, len(df))
        x = scrub_matrix(df.iloc[start:end][model.feature_cols].to_numpy(np.float32, copy=False))
        xz = ((x - model.mean) / model.scale).astype(np.float32)
        if model.clip_z > 0:
            np.clip(xz, -model.clip_z, model.clip_z, out=xz)
        pred[start:end] = (xz @ model.weight + model.y_mean).astype(np.float32)
    return pred


def load_feature_sets() -> dict[str, list[str]]:
    files = {
        "validated617": "ridge_validated617.txt",
        "score45": "ridge617_plus_scorecard_ridge_retained45.txt",
        "score100": "ridge617_plus_scorecard100.txt",
        "new1000": "ridge617_plus_new1000.txt",
        "newret463": "ridge617_plus_new1000_retained463.txt",
        "both27": "ridge617_plus_scorecard_both_retained27.txt",
    }
    out = {}
    for key, name in files.items():
        path = FEATURE_SET_DIR / name
        if path.exists():
            out[key] = read_list(path)
    return out


def predefined_variants(preset: str) -> list[RidgeAgentVariant]:
    screen2019 = [
        RidgeAgentVariant("ridge_agent_validated617_hl12_a1_n600k_clip6", "validated617", alpha=1.0, half_life_months=12.0),
        RidgeAgentVariant("ridge_agent_score45_hl12_a1_n600k_clip6", "score45", alpha=1.0, half_life_months=12.0),
        RidgeAgentVariant(
            "ridge_agent_score45_label_hl12_a1_n600k_clip6",
            "score45",
            target_col="label",
            alpha=1.0,
            half_life_months=12.0,
        ),
        RidgeAgentVariant("ridge_agent_score45_hl6_a1_n600k_clip6", "score45", alpha=1.0, half_life_months=6.0),
        RidgeAgentVariant(
            "ridge_agent_score45_hl12_a05_event35_n600k_clip6",
            "score45",
            alpha=0.5,
            sample_mode="event35",
            half_life_months=12.0,
        ),
        RidgeAgentVariant("ridge_agent_newret463_hl12_a1_n600k_clip6", "newret463", alpha=1.0, half_life_months=12.0),
        RidgeAgentVariant(
            "ridge_agent_newret463_label_hl12_a1_n600k_clip6",
            "newret463",
            target_col="label",
            alpha=1.0,
            half_life_months=12.0,
        ),
        RidgeAgentVariant("ridge_agent_newret463_hl6_a1_n600k_clip6", "newret463", alpha=1.0, half_life_months=6.0),
        RidgeAgentVariant(
            "ridge_agent_newret463_hl12_a05_event35_n600k_clip6",
            "newret463",
            alpha=0.5,
            sample_mode="event35",
            half_life_months=12.0,
        ),
        RidgeAgentVariant("ridge_agent_newret463_hl12_a2_n600k_clip6", "newret463", alpha=2.0, half_life_months=12.0),
    ]
    final2020 = [
        RidgeAgentVariant("ridge_agent_score45_hl12_a1_n900k_clip6", "score45", alpha=1.0, max_train_rows=900_000, half_life_months=12.0),
        RidgeAgentVariant(
            "ridge_agent_score45_label_hl12_a1_n900k_clip6",
            "score45",
            target_col="label",
            alpha=1.0,
            max_train_rows=900_000,
            half_life_months=12.0,
        ),
        RidgeAgentVariant("ridge_agent_newret463_hl12_a1_n900k_clip6", "newret463", alpha=1.0, max_train_rows=900_000, half_life_months=12.0),
        RidgeAgentVariant(
            "ridge_agent_newret463_label_hl12_a1_n900k_clip6",
            "newret463",
            target_col="label",
            alpha=1.0,
            max_train_rows=900_000,
            half_life_months=12.0,
        ),
        RidgeAgentVariant(
            "ridge_agent_newret463_hl12_a05_event35_n900k_clip6",
            "newret463",
            alpha=0.5,
            max_train_rows=900_000,
            sample_mode="event35",
            half_life_months=12.0,
        ),
    ]
    robust = [
        RidgeAgentVariant("ridge_agent_score45_hl12_a1_n600k_noclip", "score45", alpha=1.0, half_life_months=12.0, clip_z=0.0),
        RidgeAgentVariant("ridge_agent_score45_hl12_a1_n600k_clip4", "score45", alpha=1.0, half_life_months=12.0, clip_z=4.0),
        RidgeAgentVariant("ridge_agent_score45_hl12_a1_n600k_clip8", "score45", alpha=1.0, half_life_months=12.0, clip_z=8.0),
        RidgeAgentVariant("ridge_agent_newret463_hl12_a1_n600k_noclip", "newret463", alpha=1.0, half_life_months=12.0, clip_z=0.0),
        RidgeAgentVariant("ridge_agent_newret463_hl12_a1_n600k_clip4", "newret463", alpha=1.0, half_life_months=12.0, clip_z=4.0),
        RidgeAgentVariant("ridge_agent_newret463_hl12_a1_n600k_clip8", "newret463", alpha=1.0, half_life_months=12.0, clip_z=8.0),
    ]
    if preset == "screen2019":
        return screen2019
    if preset == "final2020":
        return final2020
    if preset == "robust":
        return robust
    if preset == "all":
        return screen2019 + robust + final2020
    raise ValueError(f"bad preset: {preset}")


def run_variant(
    variant: RidgeAgentVariant,
    fm: FeatureMatrix,
    feature_sets: dict[str, list[str]],
    cache_dir: Path,
    train_months: list[str],
    test_months: list[str],
    embargo_bars: int,
    force: bool,
    feature_transform: str,
    out_dir: Path,
) -> pd.DataFrame:
    features = feature_sets[variant.feature_set]
    model_name = variant.name if feature_transform == "raw" else f"{variant.name}_{feature_transform}feat"
    model_dir = out_dir / model_name
    parts_dir = model_dir / "month_parts"
    pred_path = model_dir / f"{model_name}.parquet"
    summary_path = model_dir / "summary.csv"
    if pred_path.exists() and summary_path.exists() and not force:
        return pd.read_csv(summary_path)
    parts_dir.mkdir(parents=True, exist_ok=True)
    for month in test_months:
        part_path = parts_dir / f"{month}.parquet"
        if part_path.exists() and not force:
            continue
        train = load_train_samples(
            cache_dir,
            train_months,
            month,
            features,
            variant.target_col,
            variant.max_train_rows,
            variant.sample_mode,
            variant.seed,
            embargo_bars,
            variant.lookback_months,
        )
        if len(train) < 10_000:
            raise RuntimeError(f"{variant.name} {month}: too few train rows: {len(train)}")
        test = fm.read_month(month, features)
        test = transform_features(test, features, feature_transform)
        sample_weight = recency_weights(train, month, variant.half_life_months)
        model = fit_ridge_agent(
            train,
            features,
            variant.target_col,
            variant.alpha,
            variant.top_k,
            variant.clip_z,
            sample_weight,
        )
        pred = predict_ridge_agent(model, test)
        out = test[META_KEEP].copy()
        out["pred"] = pred
        write_parquet_atomic(out, part_path)
        ic = compute_ic(out["pred"].to_numpy(), out["label"].to_numpy())
        print(
            f"[predict][{model_name}][{month}] train={len(train)} test={len(out)} raw_ic={ic:.6f}",
            flush=True,
        )
        del train, test, out, pred, model
        gc.collect()

    pieces = [pd.read_parquet(parts_dir / f"{month}.parquet") for month in test_months]
    pred_df = pd.concat(pieces, ignore_index=True)
    pred_df = add_prediction_views(pred_df)
    write_parquet_atomic(pred_df, pred_path)
    row = summarize_predictions(pred_df, model_name)
    row.update({f"cfg_{k}": v for k, v in asdict(variant).items()})
    row["cfg_feature_transform"] = feature_transform
    row["cfg_features"] = len(features)
    summary = pd.DataFrame([row])
    summary.to_csv(summary_path, index=False)
    monthly_rows = []
    for month, grp in pred_df.assign(month=pred_df["datetime"].dt.to_period("M").astype(str)).groupby("month", sort=True):
        mrow = {"model": model_name, "month": month}
        for pred_col in PRED_COLS:
            mrow[f"{pred_col}_ic"] = compute_ic(grp[pred_col].to_numpy(), grp["label"].to_numpy())
        monthly_rows.append(mrow)
    pd.DataFrame(monthly_rows).to_csv(model_dir / "monthly_ic.csv", index=False)
    write_json(
        model_dir / "metadata.json",
        asdict(variant)
        | {"features": len(features), "embargo_bars": embargo_bars, "feature_transform": feature_transform},
    )
    print(
        f"[summary][{model_name}] pred_xsz_ic_2019={row['pred_xsz_ic_2019']:.6f} "
        f"pred_xsz_ic_2020={row['pred_xsz_ic_2020']:.6f}",
        flush=True,
    )
    del pred_df, pieces
    gc.collect()
    return summary


def write_combined_outputs(out_dir: Path, summaries: list[pd.DataFrame]) -> None:
    existing = []
    for path in sorted(out_dir.glob("*/summary.csv")):
        try:
            existing.append(pd.read_csv(path))
        except Exception:
            pass
    if not summaries and not existing:
        return
    out = pd.concat(existing + summaries, ignore_index=True)
    if "model" in out.columns:
        out = out.drop_duplicates(subset=["model"], keep="last")
    sort_col = "pred_xsz_ic_2019" if out["pred_xsz_ic_2019"].notna().any() else "pred_xsz_ic_2020"
    out = out.sort_values(sort_col, ascending=False)
    out.to_csv(out_dir / "summary.csv", index=False)
    monthly_parts = []
    for path in sorted(out_dir.glob("*/monthly_ic.csv")):
        monthly_parts.append(pd.read_csv(path))
    if monthly_parts:
        monthly = pd.concat(monthly_parts, ignore_index=True)
        monthly.to_csv(out_dir / "monthly_ic.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["screen2019", "final2020", "robust", "all"], default="screen2019")
    parser.add_argument("--variant", default="*", help="fnmatch pattern applied to variant names")
    parser.add_argument("--train-start", default="2017-01")
    parser.add_argument("--test-start", default="2019-07")
    parser.add_argument("--test-end", default="2019-12")
    parser.add_argument("--cache-rows-per-month", type=int, default=20_000)
    parser.add_argument("--feature-transform", choices=["raw", "xsz"], default="raw")
    parser.add_argument("--embargo-bars", type=int, default=30)
    parser.add_argument("--out-dir", default="/root/autodl-tmp/quant/ML/ridge_agent_results_20260628")
    parser.add_argument("--cache-name", default="agent")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = load_feature_sets()
    variants = [v for v in predefined_variants(args.preset) if fnmatch.fnmatch(v.name, args.variant)]
    if not variants:
        raise SystemExit("no variants selected")
    missing = sorted({v.feature_set for v in variants} - set(feature_sets))
    if missing:
        raise FileNotFoundError(f"missing feature sets: {missing}")
    union_features = list(dict.fromkeys(x for v in variants for x in feature_sets[v.feature_set]))
    fm = build_feature_matrix()
    train_months = month_range(args.train_start, str(pd.Period(args.test_end, freq="M") - 1))
    test_months = month_range(args.test_start, args.test_end)
    print(
        f"[setup] out_dir={out_dir} variants={len(variants)} union_features={len(union_features)} "
        f"train={train_months[0]}..{train_months[-1]} test={test_months[0]}..{test_months[-1]}",
        flush=True,
    )
    cache_dir = ensure_month_cache(
        fm,
        train_months,
        union_features,
        args.cache_rows_per_month,
        args.rebuild_cache,
        seed=20260624,
        feature_transform=args.feature_transform,
        out_dir=out_dir,
        cache_name=args.cache_name,
    )
    write_json(
        out_dir / "run_metadata.json",
        {
            "args": vars(args),
            "feature_sets": {k: len(v) for k, v in feature_sets.items()},
            "selected_feature_sets": {k: len(feature_sets[k]) for k in sorted({v.feature_set for v in variants})},
            "union_features": len(union_features),
            "variants": [asdict(v) for v in variants],
            "note": "Ridge single-model experiments only. Training uses months strictly before each test month with 30-bar embargo on the previous month by default.",
        },
    )
    summaries = []
    for variant in variants:
        print(f"[run] {variant.name}", flush=True)
        summaries.append(
            run_variant(
                variant,
                fm,
                feature_sets,
                cache_dir,
                train_months,
                test_months,
                args.embargo_bars,
                args.force,
                args.feature_transform,
                out_dir,
            )
        )
        write_combined_outputs(out_dir, summaries)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
