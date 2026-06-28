"""
Leak-free rolling affine calibration by automatic row keys.

For each prediction month, fit

    label ~= slope[key] * base_pred + intercept[key]

using only prior months, with strong shrinkage toward the global rolling affine.
Keys can combine stable metadata and history-defined buckets, e.g.
``symbol+hour``, ``symbol+tod4+conf5`` or ``symbol+tod4+range3``.
"""
import os
import re
import sys

sys.path.insert(0, "/root/feature_model")

import numpy as np
import pandas as pd

from src.data.loader import load_config
from src.evaluation.metrics import compute_ic, ic_by_period
from src.plan_a.data_utils import factor_data_path
from src.plan_a.group_lgb import symbol_group_map


DEFAULT_BASE = "predictions_symbol_tod_affine_lb18_ss400000_si300000_clip0.5_1.5_best056885.parquet"


def parse_key_spec():
    raw = os.environ.get("KEY_SPEC", "symbol+hour").strip().lower()
    parts = [x.strip() for x in raw.split("+") if x.strip()]
    if not parts:
        raise ValueError("KEY_SPEC must contain at least one key part")
    return raw, parts


def add_norms(df):
    out = df.copy()
    g = out.groupby("datetime")["pred"]
    out["pred_xsz"] = (out["pred"] - g.transform("mean")) / (g.transform("std") + 1e-9)
    out["pred_xrank"] = g.rank(pct=True) - 0.5
    return out


def summarize(df):
    rows = []
    for col in ["pred", "pred_xsz", "pred_xrank"]:
        tmp = df.rename(columns={col: "_pred"})
        m = ic_by_period(tmp, "_pred", "label", "M")
        y = ic_by_period(tmp, "_pred", "label", "Y")
        rows.append(
            {
                "pred_col": col,
                "coverage": float((tmp["_pred"].notna() & tmp["label"].notna()).mean()),
                "total_ic": compute_ic(tmp["_pred"].values, tmp["label"].values),
                "monthly_mean": m.mean(),
                "monthly_std": m.std(),
                "ir": m.mean() / m.std(),
                **{f"ic_{k}": v for k, v in y.items()},
            }
        )
    return pd.DataFrame(rows)


def load_base(cfg):
    base_file = os.environ.get("KEY_BASE_FILE", DEFAULT_BASE)
    df = pd.read_parquet(
        os.path.join(cfg["output_dir"], base_file),
        columns=["symbol", "datetime", "label", "pred"],
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.rename(columns={"pred": "base_pred"})
    return df


def needs_market_state(parts):
    return any(p.startswith(("range", "vol", "ret")) for p in parts)


def attach_market_state(df, parts):
    if not needs_market_state(parts):
        return df
    cols = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
    raw = pd.read_parquet(factor_data_path(), columns=cols)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw[(raw["datetime"] >= df["datetime"].min()) & (raw["datetime"] <= df["datetime"].max())].copy()
    raw = raw.sort_values(["symbol", "datetime"])
    raw["range_state"] = ((raw["high"] - raw["low"]) / raw["open"].replace(0.0, np.nan)).astype(np.float32)
    raw["vol_state"] = np.log1p(raw["volume"].clip(lower=0)).astype(np.float32)
    raw["ret1_state"] = raw.groupby("symbol")["close"].pct_change(fill_method=None).astype(np.float32)
    raw = raw[["symbol", "datetime", "range_state", "vol_state", "ret1_state"]]
    return df.merge(raw, on=["symbol", "datetime"], how="left")


def add_static_keys(df, parts):
    out = df.copy()
    if "group" in parts:
        out["group_key"] = out["symbol"].map(symbol_group_map()).fillna("other").astype(str)
    if "hour" in parts:
        out["hour_key"] = out["datetime"].dt.hour.astype(np.int16)
    if "tod4" in parts:
        minute = out["datetime"].dt.hour * 60 + out["datetime"].dt.minute
        out["tod4_key"] = np.select(
            [minute <= 600, minute <= 720, minute <= 840],
            [0, 1, 2],
            default=3,
        ).astype(np.int8)
    if "sign" in parts:
        out["sign_key"] = np.select([out["base_pred"] < 0, out["base_pred"] > 0], [-1, 1], default=0).astype(np.int8)
    return out


def bucket_token(token):
    m = re.fullmatch(r"(conf|range|vol|ret)(\d+)", token)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def bucket_source(df, kind):
    if kind == "conf":
        return df["base_pred"].abs().to_numpy(np.float64)
    if kind == "range":
        return df["range_state"].to_numpy(np.float64)
    if kind == "vol":
        return df["vol_state"].to_numpy(np.float64)
    if kind == "ret":
        return np.abs(df["ret1_state"]).to_numpy(np.float64)
    raise ValueError(kind)


def make_bucket(values, cuts):
    vals = np.nan_to_num(values, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    return np.searchsorted(cuts, vals, side="right").astype(np.int16)


def add_dynamic_buckets(frame, train_idx, apply_idx, parts):
    created = []
    for token in parts:
        parsed = bucket_token(token)
        if parsed is None:
            continue
        kind, bins = parsed
        col = f"{token}_key"
        vals_train = bucket_source(frame.iloc[train_idx], kind)
        finite = vals_train[np.isfinite(vals_train)]
        if len(finite) < max(1000, bins * 100):
            cuts = np.array([], dtype=np.float64)
        else:
            qs = np.linspace(0.0, 1.0, bins + 1)[1:-1]
            cuts = np.unique(np.nanquantile(finite, qs))
        frame.loc[frame.index[train_idx], col] = make_bucket(bucket_source(frame.iloc[train_idx], kind), cuts)
        frame.loc[frame.index[apply_idx], col] = make_bucket(bucket_source(frame.iloc[apply_idx], kind), cuts)
        frame[col] = frame[col].fillna(0).astype(np.int16)
        created.append(col)
    return created


def key_columns(parts):
    cols = []
    for p in parts:
        if p == "symbol":
            cols.append("symbol")
        elif p in {"group", "hour", "tod4", "sign"}:
            cols.append(f"{p}_key")
        elif bucket_token(p) is not None:
            cols.append(f"{p}_key")
        else:
            raise ValueError(f"unknown key part: {p}")
    return cols


def affine_stats(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 1000:
        return 1.0, 0.0, 1.0
    x = x[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    xm = float(x.mean())
    ym = float(y.mean())
    xc = x - xm
    yc = y - ym
    sxx = float(np.dot(xc, xc))
    sxy = float(np.dot(xc, yc))
    slope = sxy / sxx if sxx > 1e-12 else 1.0
    intercept = ym - slope * xm
    var = sxx / max(len(x), 1)
    return slope, intercept, max(var, 1e-12)


def fit_key_affine(train, keys, slope_shrink, intercept_shrink, min_rows, clip_low, clip_high):
    global_slope, global_intercept, global_var = affine_stats(
        train["base_pred"].to_numpy(np.float64),
        train["label"].to_numpy(np.float64),
    )
    tmp = train[keys + ["base_pred", "label"]].copy()
    x = tmp["base_pred"].to_numpy(np.float64)
    y = tmp["label"].to_numpy(np.float64)
    tmp["_x2"] = x * x
    tmp["_xy"] = x * y
    grouped = (
        tmp.groupby(keys, sort=False, observed=True)
        .agg(
            n=("label", "size"),
            sx=("base_pred", "sum"),
            sy=("label", "sum"),
            sxx=("_x2", "sum"),
            sxy=("_xy", "sum"),
        )
    )
    n = grouped["n"].astype(np.float64)
    sxx_c = grouped["sxx"] - grouped["sx"] * grouped["sx"] / n
    sxy_c = grouped["sxy"] - grouped["sx"] * grouped["sy"] / n
    denom = sxx_c + slope_shrink * global_var
    slope = (sxy_c + slope_shrink * global_var * global_slope) / denom.replace(0.0, np.nan)
    slope = slope.fillna(global_slope).clip(clip_low, clip_high)
    xbar = grouped["sx"] / n
    ybar = grouped["sy"] / n
    raw_intercept = ybar - slope * xbar
    intercept = (n * raw_intercept + intercept_shrink * global_intercept) / (n + intercept_shrink)
    if min_rows > 0:
        weak = grouped["n"] < min_rows
        slope.loc[weak] = global_slope
        intercept.loc[weak] = global_intercept
    params = pd.DataFrame({"slope": slope.astype(np.float32), "intercept": intercept.astype(np.float32)})
    params = params.reset_index()
    return params, float(global_slope), float(global_intercept)


def run():
    cfg = load_config()
    spec, parts = parse_key_spec()
    lookback = int(os.environ.get("KEY_LOOKBACK", "18"))
    slope_shrink = float(os.environ.get("KEY_SLOPE_SHRINK", "400000"))
    intercept_shrink = float(os.environ.get("KEY_INTERCEPT_SHRINK", "300000"))
    min_rows = int(os.environ.get("KEY_MIN_ROWS", "0"))
    clip_low = float(os.environ.get("KEY_SLOPE_CLIP_LOW", "0.5"))
    clip_high = float(os.environ.get("KEY_SLOPE_CLIP_HIGH", "1.5"))
    output_mode = os.environ.get("KEY_OUTPUT_MODE", "score").strip().lower()
    if output_mode not in {"score", "label"}:
        raise ValueError("KEY_OUTPUT_MODE must be score or label")

    data = load_base(cfg)
    data = add_static_keys(data, parts)
    data = attach_market_state(data, parts)
    data = data.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    keys = key_columns(parts)
    out = data[["symbol", "datetime", "label", "base_pred"]].copy()
    out["pred"] = np.nan
    records = []
    start = pd.Timestamp(cfg["start_date"])
    end = min(pd.Timestamp(cfg["end_date"]), data["datetime"].max())
    print(
        f"[rolling_key_affine] rows={len(data)} spec={spec} lookback={lookback} "
        f"slope_shrink={slope_shrink:g} intercept_shrink={intercept_shrink:g}",
        flush=True,
    )
    for ms in pd.date_range(start, end, freq="MS"):
        next_ms = ms + pd.DateOffset(months=1)
        tr_start = ms - pd.DateOffset(months=lookback)
        tr_mask = (data["datetime"] >= tr_start) & (data["datetime"] < ms)
        pr_mask = (data["datetime"] >= ms) & (data["datetime"] < next_ms)
        tr_idx = np.flatnonzero(tr_mask.to_numpy())
        pr_idx = np.flatnonzero(pr_mask.to_numpy())
        if len(pr_idx) == 0:
            continue
        if len(tr_idx) < 5000:
            out.loc[pr_mask, "pred"] = data.loc[pr_mask, "base_pred"].to_numpy(np.float32)
            records.append({"month": f"{ms:%Y-%m}", "train_rows": len(tr_idx), "keys": 0, "base_ic": np.nan, "month_ic": np.nan})
            continue
        add_dynamic_buckets(data, tr_idx, pr_idx, parts)
        params, gs, gi = fit_key_affine(
            data.iloc[tr_idx],
            keys,
            slope_shrink=slope_shrink,
            intercept_shrink=intercept_shrink,
            min_rows=min_rows,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        cur = data.iloc[pr_idx][["base_pred"] + keys].copy()
        cur["_row"] = pr_idx
        cur = cur.merge(params, on=keys, how="left")
        cur["slope"] = cur["slope"].fillna(gs).astype(np.float32)
        cur["intercept"] = cur["intercept"].fillna(gi).astype(np.float32)
        yhat = cur["slope"].to_numpy(np.float64) * cur["base_pred"].to_numpy(np.float64) + cur["intercept"].to_numpy(np.float64)
        if output_mode == "score" and abs(gs) > 1e-12:
            pred = ((yhat - gi) / gs).astype(np.float32)
        else:
            pred = yhat.astype(np.float32)
        out.loc[data.index[pr_idx], "pred"] = pred
        base_ic = compute_ic(data.iloc[pr_idx]["base_pred"].values, data.iloc[pr_idx]["label"].values)
        month_ic = compute_ic(pred, data.iloc[pr_idx]["label"].values)
        records.append(
            {
                "month": f"{ms:%Y-%m}",
                "train_rows": len(tr_idx),
                "keys": len(params),
                "global_slope": gs,
                "global_intercept": gi,
                "base_ic": base_ic,
                "month_ic": month_ic,
            }
        )
        print(
            f"  [key-affine][{ms:%Y-%m}] keys={len(params):5d} base={base_ic:.4f} IC={month_ic:.4f}",
            flush=True,
        )

    pred = add_norms(out[["symbol", "datetime", "label", "pred"]])
    safe_spec = spec.replace("+", "_")
    name = (
        f"key_affine_{output_mode}_{safe_spec}_lb{lookback}_ss{slope_shrink:g}_"
        f"si{intercept_shrink:g}_clip{clip_low:g}_{clip_high:g}"
    )
    suffix = os.environ.get("KEY_NAME_SUFFIX", "").strip()
    if suffix:
        safe_suffix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in suffix)
        name = f"{name}_{safe_suffix}"
    pred_path = os.path.join(cfg["output_dir"], f"predictions_{name}.parquet")
    pred.to_parquet(pred_path, index=False)
    rep = os.path.join(cfg["reports_dir"], "plan_a")
    os.makedirs(rep, exist_ok=True)
    pd.DataFrame(records).to_csv(os.path.join(rep, f"key_affine_months_{name}.csv"), index=False)
    res = summarize(pred[pred["datetime"] >= pd.Timestamp(cfg["start_date"])])
    res.to_csv(os.path.join(rep, f"ablation_{name}.csv"), index=False)
    print(res.to_string(index=False), flush=True)
    print(f"[rolling_key_affine] wrote {pred_path}", flush=True)


if __name__ == "__main__":
    run()
