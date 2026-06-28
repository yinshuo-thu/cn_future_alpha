"""
Leak-free rolling bucket scale optimizer.

This calibrator keeps the base signal shape and only learns multiplicative
scales by state buckets.  For each month it defines dynamic buckets from prior
history, then chooses positive bucket scales that maximize the historical pooled
IC with a small shrinkage penalty toward scale 1.
"""
import os
import re
import sys

sys.path.insert(0, "/root/feature_model")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.data.loader import load_config
from src.evaluation.metrics import compute_ic, ic_by_period
from src.plan_a.data_utils import factor_data_path
from src.plan_a.group_lgb import symbol_group_map


DEFAULT_BASE = "predictions_symbol_tod_affine_lb18_ss400000_si300000_clip0.5_1.5_best056885.parquet"


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


def parse_spec():
    raw = os.environ.get("SCALE_SPEC", "range5+vol5").strip().lower()
    parts = [p.strip() for p in raw.split("+") if p.strip()]
    if not parts:
        raise ValueError("SCALE_SPEC is empty")
    parsed = []
    for part in parts:
        m = re.fullmatch(r"(conf|range|vol|ret|group|symbol|tod)(\d*)", part)
        if not m:
            raise ValueError(f"bad SCALE_SPEC part: {part}")
        kind = m.group(1)
        bins = int(m.group(2) or (4 if kind == "tod" else 1))
        if kind in {"conf", "range", "vol", "ret"} and bins < 2:
            raise ValueError(f"{kind} needs a bucket count, e.g. {kind}5")
        parsed.append((part, kind, bins))
    return raw, parsed


def load_frame(cfg, parsed):
    base_file = os.environ.get("SCALE_BASE_FILE", DEFAULT_BASE)
    data = pd.read_parquet(
        os.path.join(cfg["output_dir"], base_file),
        columns=["symbol", "datetime", "label", "pred"],
    )
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.rename(columns={"pred": "base_pred"})
    need_state = any(kind in {"range", "vol", "ret"} for _, kind, _ in parsed)
    if need_state:
        raw = pd.read_parquet(
            factor_data_path(),
            columns=["symbol", "datetime", "open", "high", "low", "close", "volume"],
        )
        raw["datetime"] = pd.to_datetime(raw["datetime"])
        raw = raw[(raw["datetime"] >= data["datetime"].min()) & (raw["datetime"] <= data["datetime"].max())].copy()
        raw = raw.sort_values(["symbol", "datetime"])
        raw["range_state"] = ((raw["high"] - raw["low"]) / raw["open"].replace(0.0, np.nan)).astype(np.float32)
        raw["vol_state"] = np.log1p(raw["volume"].clip(lower=0)).astype(np.float32)
        raw["ret_state"] = raw.groupby("symbol")["close"].pct_change(fill_method=None).abs().astype(np.float32)
        data = data.merge(raw[["symbol", "datetime", "range_state", "vol_state", "ret_state"]], on=["symbol", "datetime"], how="left")
    if any(kind == "group" for _, kind, _ in parsed):
        codes, _ = pd.factorize(data["symbol"].map(symbol_group_map()).fillna("other"), sort=True)
        data["group_state"] = codes.astype(np.int16)
    if any(kind == "symbol" for _, kind, _ in parsed):
        codes, _ = pd.factorize(data["symbol"], sort=True)
        data["symbol_state"] = codes.astype(np.int16)
    if any(kind == "tod" for _, kind, _ in parsed):
        minute = (data["datetime"].dt.hour * 60 + data["datetime"].dt.minute).to_numpy()
        data["tod_state"] = np.select([minute <= 600, minute <= 720, minute <= 840], [0, 1, 2], default=3).astype(np.int16)
    return data.sort_values(["datetime", "symbol"]).reset_index(drop=True)


def source_values(df, kind):
    if kind == "conf":
        return df["base_pred"].abs().to_numpy(np.float64)
    if kind == "range":
        return df["range_state"].to_numpy(np.float64)
    if kind == "vol":
        return df["vol_state"].to_numpy(np.float64)
    if kind == "ret":
        return df["ret_state"].to_numpy(np.float64)
    if kind == "group":
        return df["group_state"].to_numpy(np.int64)
    if kind == "symbol":
        return df["symbol_state"].to_numpy(np.int64)
    if kind == "tod":
        return df["tod_state"].to_numpy(np.int64)
    raise ValueError(kind)


def codes_for(train, apply, parsed):
    codes = []
    dims = []
    for _, kind, bins in parsed:
        if kind in {"group", "symbol", "tod"}:
            tr = source_values(train, kind).astype(np.int64)
            ap = source_values(apply, kind).astype(np.int64)
            dim = int(max(np.nanmax(tr), np.nanmax(ap))) + 1
            codes.append(ap)
            dims.append(dim)
            continue
        tr_vals = source_values(train, kind)
        ap_vals = source_values(apply, kind)
        finite = tr_vals[np.isfinite(tr_vals)]
        if len(finite) < max(1000, bins * 100):
            cuts = np.array([], dtype=np.float64)
        else:
            cuts = np.unique(np.nanquantile(finite, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
        code = np.searchsorted(cuts, np.nan_to_num(ap_vals, nan=-np.inf), side="right").astype(np.int64)
        codes.append(code)
        dims.append(bins)
    if len(codes) == 1:
        return codes[0], dims[0]
    return np.ravel_multi_index(tuple(codes), tuple(dims)), int(np.prod(dims))


def optimize_scales(base, label, code, k, penalty, low, high):
    mask = np.isfinite(base) & np.isfinite(label) & np.isfinite(code)
    if mask.sum() < 5000:
        return np.ones(k, dtype=np.float64), np.nan
    c = code[mask].astype(np.int64)
    p = base[mask].astype(np.float64)
    y = label[mask].astype(np.float64)
    a = np.bincount(c, weights=p * y, minlength=k).astype(np.float64)
    b = np.bincount(c, weights=p * p, minlength=k).astype(np.float64)
    cnt = np.bincount(c, minlength=k).astype(np.float64)
    w = cnt / max(float(cnt.sum()), 1.0)
    yy = float(np.mean(y * y))

    def score_and_grad(s):
        s = np.asarray(s, dtype=np.float64)
        num = float(np.dot(s, a))
        den_p = float(np.dot(s * s, b))
        den = np.sqrt(max(den_p, 1e-18) * max(yy * len(y), 1e-18))
        ic = num / max(den, 1e-12)
        pen = penalty * float(np.dot(w, (s - 1.0) ** 2))
        if den_p <= 1e-18:
            grad = -a
        else:
            grad_ic = a / den - (num * s * b) / (den * den_p)
            grad = -grad_ic + 2.0 * penalty * w * (s - 1.0)
        return -ic + pen, grad

    starts = [np.ones(k, dtype=np.float64)]
    if k <= 10:
        starts.extend([np.linspace(1.15, 0.85, k), np.linspace(0.85, 1.15, k)])
    best_x = starts[0]
    best_obj = np.inf
    for start in starts:
        res = minimize(
            lambda s: score_and_grad(s),
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=[(low, high)] * k,
            options={"maxiter": 300, "ftol": 1e-12},
        )
        if res.success and np.isfinite(res.fun) and res.fun < best_obj:
            best_obj = float(res.fun)
            best_x = res.x
    scale = np.clip(best_x, low, high)
    train_ic = compute_ic(p * scale[c], y)
    return scale, train_ic


def run():
    cfg = load_config()
    spec, parsed = parse_spec()
    lookback = int(os.environ.get("SCALE_LOOKBACK", "18"))
    penalty = float(os.environ.get("SCALE_PENALTY", "0.001"))
    low = float(os.environ.get("SCALE_LOW", "0.5"))
    high = float(os.environ.get("SCALE_HIGH", "1.5"))
    data = load_frame(cfg, parsed)
    out = data[["symbol", "datetime", "label", "base_pred"]].copy()
    out["pred"] = np.nan
    records = []
    start = pd.Timestamp(cfg["start_date"])
    end = min(pd.Timestamp(cfg["end_date"]), data["datetime"].max())
    print(
        f"[bucket_scale_ic] rows={len(data)} spec={spec} lookback={lookback} "
        f"penalty={penalty:g} bounds=({low:g},{high:g})",
        flush=True,
    )
    for ms in pd.date_range(start, end, freq="MS"):
        next_ms = ms + pd.DateOffset(months=1)
        tr = data[(data["datetime"] >= ms - pd.DateOffset(months=lookback)) & (data["datetime"] < ms)]
        pr = data[(data["datetime"] >= ms) & (data["datetime"] < next_ms)]
        if len(pr) == 0:
            continue
        if len(tr) < 5000:
            out.loc[pr.index, "pred"] = pr["base_pred"].to_numpy(np.float32)
            records.append({"month": f"{ms:%Y-%m}", "train_rows": len(tr), "buckets": 0, "train_ic": np.nan, "base_ic": np.nan, "month_ic": np.nan})
            continue
        tr_code, k = codes_for(tr, tr, parsed)
        pr_code, _ = codes_for(tr, pr, parsed)
        scale, train_ic = optimize_scales(
            tr["base_pred"].to_numpy(np.float64),
            tr["label"].to_numpy(np.float64),
            tr_code,
            k,
            penalty,
            low,
            high,
        )
        pred = pr["base_pred"].to_numpy(np.float64) * scale[pr_code]
        out.loc[pr.index, "pred"] = pred.astype(np.float32)
        base_ic = compute_ic(pr["base_pred"].values, pr["label"].values)
        month_ic = compute_ic(pred, pr["label"].values)
        records.append(
            {
                "month": f"{ms:%Y-%m}",
                "train_rows": len(tr),
                "buckets": k,
                "train_ic": train_ic,
                "base_ic": base_ic,
                "month_ic": month_ic,
                "scale_min": float(scale.min()),
                "scale_max": float(scale.max()),
            }
        )
        print(
            f"  [bucket-scale][{ms:%Y-%m}] k={k:3d} train={train_ic:.4f} "
            f"base={base_ic:.4f} IC={month_ic:.4f} scale={scale.min():.2f}-{scale.max():.2f}",
            flush=True,
        )
    pred = add_norms(out[["symbol", "datetime", "label", "pred"]])
    safe_spec = spec.replace("+", "_")
    name = f"bucket_scale_ic_{safe_spec}_lb{lookback}_pen{penalty:g}_b{low:g}_{high:g}"
    suffix = os.environ.get("SCALE_NAME_SUFFIX", "").strip()
    if suffix:
        safe_suffix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in suffix)
        name = f"{name}_{safe_suffix}"
    pred_path = os.path.join(cfg["output_dir"], f"predictions_{name}.parquet")
    pred.to_parquet(pred_path, index=False)
    rep = os.path.join(cfg["reports_dir"], "plan_a")
    os.makedirs(rep, exist_ok=True)
    pd.DataFrame(records).to_csv(os.path.join(rep, f"bucket_scale_months_{name}.csv"), index=False)
    res = summarize(pred[pred["datetime"] >= pd.Timestamp(cfg["start_date"])])
    res.to_csv(os.path.join(rep, f"ablation_{name}.csv"), index=False)
    print(res.to_string(index=False), flush=True)
    print(f"[bucket_scale_ic] wrote {pred_path}", flush=True)


if __name__ == "__main__":
    run()
