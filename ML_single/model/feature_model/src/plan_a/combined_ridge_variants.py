"""
Rolling ridge on the union of the legacy Plan A factors and mined factors.

Prediction-level stacking topped out near 0.055 IC. This script tests whether
fitting a single linear model on both raw factor libraries can exploit
cross-library covariance that is lost after compressing each library to one
model prediction.
"""
import os
import sys
sys.path.insert(0, "/root/feature_model")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.loader import load_config
from src.evaluation.metrics import compute_ic, ic_by_period
from src.plan_a.data_utils import META_COLS


OLD_PATH = "/root/shared-nvme/feature_model/data_factors_big.parquet"
MINED_PATH = "/root/shared-nvme/feature_model/data_factors_mined.parquet"


def scrub(x):
    return np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def selected_old_cols(cfg, path, top_n):
    names = pq.ParquetFile(path).schema.names
    schema = set(names)
    selected_path = os.path.join(cfg["output_dir"], "selected_factors.txt")
    selected = []
    if os.path.exists(selected_path):
        with open(selected_path) as f:
            selected = [x.strip() for x in f if x.strip()]
    cols, seen = [], set()
    for col in selected:
        if col in schema and col not in META_COLS and col not in seen:
            cols.append(col)
            seen.add(col)
            if len(cols) >= top_n:
                return cols
    for col in names:
        if col not in META_COLS and col not in seen:
            cols.append(col)
            seen.add(col)
            if len(cols) >= top_n:
                break
    return cols


def mined_cols(path, top_n):
    names = pq.ParquetFile(path).schema.names
    cols = [c for c in names if c not in META_COLS]
    return cols[:top_n]


def load_data(cfg, old_top, mined_top):
    old_path = os.environ.get("COMBINED_OLD_PATH", OLD_PATH)
    mined_path = os.environ.get("COMBINED_MINED_PATH", MINED_PATH)
    old_cols = selected_old_cols(cfg, old_path, old_top)
    mined = mined_cols(mined_path, mined_top)
    old_df = pd.read_parquet(old_path, columns=["symbol", "datetime", "label"] + old_cols)
    old_df["datetime"] = pd.to_datetime(old_df["datetime"])
    mined_df = pd.read_parquet(mined_path, columns=["symbol", "datetime", "label"] + mined)
    mined_df["datetime"] = pd.to_datetime(mined_df["datetime"])
    aligned = (
        len(old_df) == len(mined_df)
        and old_df["symbol"].equals(mined_df["symbol"])
        and old_df["datetime"].equals(mined_df["datetime"])
    )
    if not aligned:
        mined_df = mined_df.drop(columns=["label"])
        data = old_df.merge(mined_df, on=["symbol", "datetime"], how="inner", suffixes=("", "_mined"))
    else:
        rename = {c: (c if c not in old_df.columns else f"mined_{c}") for c in mined}
        mined_part = mined_df[mined].rename(columns=rename)
        data = pd.concat([old_df, mined_part], axis=1)
        mined = [rename[c] for c in mined]
    feat_cols = old_cols + mined
    return data, feat_cols, old_cols, mined


def add_target_transforms(data):
    g = data.groupby("datetime")["label"]
    mu = g.transform("mean")
    sd = g.transform("std")
    data["label_xsz"] = ((data["label"] - mu) / (sd + 1e-9)).astype(np.float32)
    data["label_xrank"] = (g.rank(pct=True) - 0.5).astype(np.float32)
    return data


def fit_ridge(X, y, alpha, top_k=0):
    X = scrub(X.astype(np.float32, copy=False))
    y = y.astype(np.float64, copy=False)
    mask = np.isfinite(y)
    X = X[mask]
    y = y[mask]
    mu = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = np.maximum(X.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    Xz = ((X - mu) / sd).astype(np.float32)
    y0 = y - y.mean()
    G = (Xz.T @ Xz).astype(np.float64) / max(len(Xz), 1)
    c = (Xz.T @ y0).astype(np.float64) / max(len(Xz), 1)
    if top_k and top_k < len(c):
        keep = np.argpartition(np.abs(c), -top_k)[-top_k:]
        wk = np.linalg.solve(G[np.ix_(keep, keep)] + alpha * np.eye(len(keep)), c[keep])
        w = np.zeros_like(c)
        w[keep] = wk
    else:
        w = np.linalg.solve(G + alpha * np.eye(len(c)), c)
    return mu, sd, w.astype(np.float32)


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
    old_top = int(os.environ.get("COMBINED_OLD_TOP_N", "1000"))
    mined_top = int(os.environ.get("COMBINED_MINED_TOP_N", "1005"))
    lookback = int(os.environ.get("COMBINED_LOOKBACK", "12"))
    max_rows = int(os.environ.get("COMBINED_MAX_ROWS", "600000"))
    alpha = float(os.environ.get("COMBINED_ALPHA", "1.0"))
    top_k = int(os.environ.get("COMBINED_TOP_K_COEF", "0"))
    target_mode = os.environ.get("COMBINED_TARGET_MODE", "xsz")
    if target_mode not in {"raw", "xsz", "xrank"}:
        raise ValueError(f"unknown COMBINED_TARGET_MODE={target_mode}")
    data, feat_cols, old_cols, mined = load_data(cfg, old_top, mined_top)
    data = add_target_transforms(data)
    target_col = {"raw": "label", "xsz": "label_xsz", "xrank": "label_xrank"}[target_mode]
    print(
        f"[combined_ridge] rows={len(data)} features={len(feat_cols)} old={len(old_cols)} "
        f"mined={len(mined)} lookback={lookback} max_rows={max_rows} alpha={alpha} "
        f"top_k={top_k} target={target_mode}",
        flush=True,
    )
    out = data[["symbol", "datetime", "label"]].copy()
    out["pred"] = np.nan
    start = pd.Timestamp(cfg["start_date"])
    end = min(pd.Timestamp(cfg["end_date"]), data["datetime"].max())
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    for ms in pd.date_range(start, end, freq="MS"):
        me = ms + pd.offsets.MonthEnd(1)
        tr_start = ms - pd.DateOffset(months=lookback)
        tr_end = ms - pd.Timedelta(days=1)
        tr = data[(data.datetime >= tr_start) & (data.datetime <= tr_end)].dropna(subset=[target_col, "label"])
        pr = data[(data.datetime >= ms) & (data.datetime <= me)]
        if len(tr) < 5000 or len(pr) == 0:
            continue
        if max_rows and len(tr) > max_rows:
            tr = tr.iloc[rng.choice(len(tr), max_rows, replace=False)]
        mu, sd, w = fit_ridge(tr[feat_cols].to_numpy(np.float32), tr[target_col].to_numpy(np.float64), alpha, top_k=top_k)
        Xpr = scrub(pr[feat_cols].to_numpy(np.float32))
        out.loc[pr.index, "pred"] = ((Xpr - mu) / sd) @ w
        mic = compute_ic(out.loc[pr.index, "pred"].values, pr["label"].values)
        print(f"  [combined-ridge][{ms:%Y-%m}] tr={len(tr):7d} nnz={int(np.count_nonzero(w)):4d} IC={mic:.4f}", flush=True)
    pred = add_norms(out)
    name = (
        f"combined_ridge_yt{target_mode}_a{alpha:g}_lb{lookback}_old{len(old_cols)}"
        f"_mined{len(mined)}_n{max_rows}_k{top_k}"
    )
    suffix = os.environ.get("COMBINED_NAME_SUFFIX", "").strip()
    if suffix:
        safe_suffix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in suffix)
        name = f"{name}_{safe_suffix}"
    pred.to_parquet(os.path.join(cfg["output_dir"], f"predictions_{name}.parquet"), index=False)
    rep = os.path.join(cfg["reports_dir"], "plan_a")
    os.makedirs(rep, exist_ok=True)
    eval_df = pred[pred.datetime >= pd.Timestamp(cfg["start_date"])].copy()
    res = summarize(eval_df)
    res.to_csv(os.path.join(rep, f"ablation_{name}.csv"), index=False)
    print(res.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
