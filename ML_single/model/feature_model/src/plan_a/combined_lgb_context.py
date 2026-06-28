"""
LightGBM on the union of legacy Plan A factors and mined factors.

This is the nonlinear counterpart to combined_ridge_variants.py. It keeps the
same monthly walk-forward protocol and adds symbol/calendar context features.
"""
import os
import sys
sys.path.insert(0, "/root/feature_model")

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.data.loader import load_config
from src.evaluation.metrics import compute_ic, ic_by_period
from src.plan_a.combined_ridge_variants import load_data
from src.plan_a.group_lgb import symbol_group_map


def scrub(x):
    return np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def add_target_transforms(data):
    g = data.groupby("datetime")["label"]
    mu = g.transform("mean")
    sd = g.transform("std")
    data["label_xsz"] = ((data["label"] - mu) / (sd + 1e-9)).astype(np.float32)
    data["label_xrank"] = (g.rank(pct=True) - 0.5).astype(np.float32)
    return data


def add_context(data):
    groups = symbol_group_map()
    out = data
    symbols = sorted(out["symbol"].unique())
    sym_map = {s: i for i, s in enumerate(symbols)}
    group_names = sorted(set(groups.values()) | {"other"})
    grp_map = {g: i for i, g in enumerate(group_names)}
    out["symbol_code"] = out["symbol"].map(sym_map).astype(np.int16)
    out["group_code"] = out["symbol"].map(groups).fillna("other").map(grp_map).astype(np.int8)
    minute = (out["datetime"].dt.hour * 60 + out["datetime"].dt.minute).astype(np.float32)
    out["minute_sin"] = np.sin(2 * np.pi * minute / 1440.0).astype(np.float32)
    out["minute_cos"] = np.cos(2 * np.pi * minute / 1440.0).astype(np.float32)
    dow = out["datetime"].dt.dayofweek.astype(np.float32)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    month = out["datetime"].dt.month.astype(np.float32)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0).astype(np.float32)
    return out


def make_X(df, feat_cols, use_context):
    X = pd.DataFrame(scrub(df[feat_cols].to_numpy(np.float32)), columns=feat_cols, index=df.index)
    if use_context:
        for c in ["symbol_code", "group_code"]:
            X[c] = df[c].to_numpy()
        for c in ["minute_sin", "minute_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            X[c] = df[c].to_numpy(np.float32)
    return X


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
    old_top = int(os.environ.get("COMBLGB_OLD_TOP_N", "550"))
    mined_top = int(os.environ.get("COMBLGB_MINED_TOP_N", "500"))
    lookback = int(os.environ.get("COMBLGB_LOOKBACK", "12"))
    max_rows = int(os.environ.get("COMBLGB_MAX_ROWS", "500000"))
    target_mode = os.environ.get("COMBLGB_TARGET_MODE", "xsz")
    use_context = os.environ.get("COMBLGB_CONTEXT", "1") == "1"
    if target_mode not in {"raw", "xsz", "xrank"}:
        raise ValueError(f"unknown COMBLGB_TARGET_MODE={target_mode}")
    data, feat_cols, old_cols, mined_cols = load_data(cfg, old_top, mined_top)
    data = add_target_transforms(data)
    if use_context:
        data = add_context(data)
    target_col = {"raw": "label", "xsz": "label_xsz", "xrank": "label_xrank"}[target_mode]
    print(
        f"[combined_lgb] rows={len(data)} features={len(feat_cols)} old={len(old_cols)} "
        f"mined={len(mined_cols)} lookback={lookback} max_rows={max_rows} "
        f"target={target_mode} context={use_context}",
        flush=True,
    )
    params = dict(
        n_estimators=int(os.environ.get("COMBLGB_N_ESTIMATORS", "320")),
        learning_rate=float(os.environ.get("COMBLGB_LR", "0.035")),
        num_leaves=int(os.environ.get("COMBLGB_NUM_LEAVES", "63")),
        subsample=float(os.environ.get("COMBLGB_SUBSAMPLE", "0.8")),
        colsample_bytree=float(os.environ.get("COMBLGB_COLSAMPLE", "0.65")),
        min_child_samples=int(os.environ.get("COMBLGB_MIN_CHILD", "120")),
        reg_lambda=float(os.environ.get("COMBLGB_REG_LAMBDA", "4.0")),
        n_jobs=-1,
        verbose=-1,
    )
    cat_cols = ["symbol_code", "group_code"] if use_context else []
    out = data[["symbol", "datetime", "label"]].copy()
    out["pred"] = np.nan
    start = pd.Timestamp(cfg["start_date"])
    end = min(pd.Timestamp(cfg["end_date"]), data["datetime"].max())
    for ms in pd.date_range(start, end, freq="MS"):
        me = ms + pd.offsets.MonthEnd(1)
        tr_start = ms - pd.DateOffset(months=lookback)
        tr_end = ms - pd.Timedelta(days=1)
        tr = data[(data.datetime >= tr_start) & (data.datetime <= tr_end)].dropna(subset=[target_col, "label"])
        pr = data[(data.datetime >= ms) & (data.datetime <= me)]
        if len(tr) < 5000 or len(pr) == 0:
            continue
        if max_rows and len(tr) > max_rows:
            tr = tr.sample(max_rows, random_state=int(cfg.get("seed", 42)))
        Xtr = make_X(tr, feat_cols, use_context)
        ytr = tr[target_col].to_numpy(np.float32)
        Xpr = make_X(pr, feat_cols, use_context)
        model = lgb.LGBMRegressor(**params)
        model.fit(Xtr, ytr, categorical_feature=cat_cols if cat_cols else "auto")
        out.loc[pr.index, "pred"] = model.predict(Xpr)
        mic = compute_ic(out.loc[pr.index, "pred"].values, pr["label"].values)
        print(f"  [combined-lgb][{ms:%Y-%m}] tr={len(tr):7d} IC={mic:.4f}", flush=True)
    pred = add_norms(out)
    name = (
        f"combined_lgb_ctx{int(use_context)}_yt{target_mode}_lb{lookback}"
        f"_old{len(old_cols)}_mined{len(mined_cols)}_n{max_rows}"
    )
    suffix = os.environ.get("COMBLGB_NAME_SUFFIX", "").strip()
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
