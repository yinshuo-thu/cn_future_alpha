"""
Leak-free daily rolling ridge stack over saved component predictions.

Monthly stacks reacted too slowly around regime shifts. This model re-estimates
component weights every trading day using only prior days, then predicts the
current day. It keeps full row coverage: no selective NaNs.
"""
import os
import sys
sys.path.insert(0, "/root/feature_model")

import numpy as np
import pandas as pd

from src.data.loader import load_config
from src.evaluation.metrics import compute_ic, ic_by_period
from src.plan_a.rolling_ensemble import BASE_COMPONENTS


def parse_components():
    raw = os.environ.get("DAILY_COMPONENTS", "").strip()
    comps = list(BASE_COMPONENTS)
    if raw:
        comps = []
        for item in raw.split(";"):
            item = item.strip()
            if not item:
                continue
            name, fname, mode = item.split(":")
            if mode not in {"rank", "z", "raw"}:
                raise ValueError(f"bad mode {mode} in {item}")
            comps.append((name, fname, mode))
    return comps


def component_signal(df, mode):
    if mode == "raw":
        return df["pred"].astype(np.float32)
    g = df.groupby("datetime")["pred"]
    if mode == "rank":
        return (g.rank(pct=True) - 0.5).astype(np.float32)
    if mode == "z":
        return ((df["pred"] - g.transform("mean")) / (g.transform("std") + 1e-9)).astype(np.float32)
    raise ValueError(mode)


def load_components(output_dir, comps):
    base = None
    for name, fname, mode in comps:
        path = os.path.join(output_dir, fname)
        df = pd.read_parquet(path, columns=["symbol", "datetime", "label", "pred"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df[name] = component_signal(df, mode)
        cols = ["symbol", "datetime", "label", name] if base is None else ["symbol", "datetime", name]
        base = df[cols].copy() if base is None else base.merge(df[cols], on=["symbol", "datetime"], how="inner")
    return base


def add_target_transforms(data):
    g = data.groupby("datetime")["label"]
    mu = g.transform("mean")
    sd = g.transform("std")
    data["label_xsz"] = ((data["label"] - mu) / (sd + 1e-9)).astype(np.float32)
    data["label_xrank"] = (g.rank(pct=True) - 0.5).astype(np.float32)
    return data


def fit_ridge(X, y, alpha):
    X = np.nan_to_num(X.astype(np.float32, copy=False), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y = y.astype(np.float64, copy=False)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    mu = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = np.maximum(X.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    Xz = ((X - mu) / sd).astype(np.float32)
    ym = float(y.mean())
    y0 = y - ym
    G = (Xz.T @ Xz).astype(np.float64) / max(len(Xz), 1)
    c = (Xz.T @ y0).astype(np.float64) / max(len(Xz), 1)
    w = np.linalg.solve(G + alpha * np.eye(len(c)), c).astype(np.float32)
    return mu, sd, ym, w


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
        rows.append({
            "pred_col": col,
            "coverage": float((df[col].notna() & df["label"].notna()).mean()),
            "total_ic": compute_ic(df[col].values, df["label"].values),
            "monthly_mean": m.mean(),
            "monthly_std": m.std(),
            "ir": m.mean() / m.std(),
            **{f"ic_{k}": v for k, v in y.items()},
        })
    return pd.DataFrame(rows)


def run():
    cfg = load_config()
    comps = parse_components()
    names = [x[0] for x in comps]
    data = load_components(cfg["output_dir"], comps)
    data = data[data["datetime"] >= pd.Timestamp(cfg["start_date"])].copy()
    data = add_target_transforms(data)
    target_mode = os.environ.get("DAILY_TARGET_MODE", "xsz")
    target_col = {"raw": "label", "xsz": "label_xsz", "xrank": "label_xrank"}[target_mode]
    lookback_days = int(os.environ.get("DAILY_LOOKBACK_DAYS", "60"))
    min_days = int(os.environ.get("DAILY_MIN_DAYS", "20"))
    max_rows = int(os.environ.get("DAILY_MAX_ROWS", "800000"))
    alpha = float(os.environ.get("DAILY_ALPHA", "0.1"))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    data["_day"] = data["datetime"].dt.floor("D")
    days = np.array(sorted(data["_day"].unique()))
    day_code = pd.Categorical(data["_day"], categories=days, ordered=True).codes
    X_all = data[names].to_numpy(np.float32)
    y_all = data[target_col].to_numpy(np.float64)
    out = data[["symbol", "datetime", "label"]].copy()
    out["pred"] = np.nan
    print(
        f"[daily_stack] rows={len(data)} components={len(names)} days={len(days)} "
        f"lookback={lookback_days} min_days={min_days} alpha={alpha} target={target_mode}",
        flush=True,
    )
    default_w = np.ones(len(names), dtype=np.float32) / max(len(names), 1)
    records = []
    for d, day in enumerate(days):
        cur = day_code == d
        if d >= min_days:
            lo = max(0, d - lookback_days)
            tr_mask = (day_code >= lo) & (day_code < d)
            idx = np.flatnonzero(tr_mask & np.isfinite(y_all))
            if max_rows and len(idx) > max_rows:
                idx = rng.choice(idx, max_rows, replace=False)
            mu, sd, ym, w = fit_ridge(X_all[idx], y_all[idx], alpha)
            pred = ((np.nan_to_num(X_all[cur], nan=0.0, posinf=0.0, neginf=0.0) - mu) / sd) @ w + ym
            train_ic = compute_ic(((np.nan_to_num(X_all[idx], nan=0.0, posinf=0.0, neginf=0.0) - mu) / sd) @ w + ym, data.iloc[idx]["label"].values)
        else:
            pred = np.nan_to_num(X_all[cur], nan=0.0, posinf=0.0, neginf=0.0) @ default_w
            train_ic = np.nan
        out.loc[data.index[cur], "pred"] = pred
        if d % 20 == 0 or d == len(days) - 1:
            day_ic = compute_ic(out.loc[data.index[cur], "pred"].values, data.loc[data.index[cur], "label"].values)
            print(f"  [daily-stack][{pd.Timestamp(day):%Y-%m-%d}] train_ic={train_ic:.4f} day_ic={day_ic:.4f}", flush=True)
        records.append({"day": str(pd.Timestamp(day).date()), "train_ic": train_ic})
    pred = add_norms(out)
    name = (
        f"daily_stack_yt{target_mode}_lb{lookback_days}_min{min_days}_"
        f"a{alpha:g}_n{max_rows}_c{len(names)}"
    )
    suffix = os.environ.get("DAILY_NAME_SUFFIX", "").strip()
    if suffix:
        safe_suffix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in suffix)
        name = f"{name}_{safe_suffix}"
    pred.to_parquet(os.path.join(cfg["output_dir"], f"predictions_{name}.parquet"), index=False)
    rep = os.path.join(cfg["reports_dir"], "plan_a")
    os.makedirs(rep, exist_ok=True)
    pd.DataFrame(records).to_csv(os.path.join(rep, f"daily_stack_weights_{name}.csv"), index=False)
    res = summarize(pred)
    res.to_csv(os.path.join(rep, f"ablation_{name}.csv"), index=False)
    print(res.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
