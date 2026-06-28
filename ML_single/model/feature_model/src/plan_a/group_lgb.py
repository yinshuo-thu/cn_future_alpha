"""
Prior-sector grouped LightGBM.

Each month trains one model per sector using only historical rows from symbols in
that sector. Predictions are stitched back together and cross-sectionally ranked
per timestamp so group model scales are comparable under the official pooled IC.
"""
import os
import sys
sys.path.insert(0, "/root/feature_model")

import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq

from src.data.loader import load_config
from src.evaluation.metrics import compute_ic, ic_by_period
from src.plan_a.data_utils import factor_data_path, select_factor_columns


BIG = "/root/shared-nvme/feature_model/data_factors_big.parquet"

PRIOR_GROUPS = {
    "precious": ["AU", "AG"],
    "nonferrous": ["CU", "AL", "ZN", "PB", "NI", "SN", "SS"],
    "ferrous": ["RB", "HC", "I", "J", "JM", "ZC", "SF", "SM"],
    "energy_chem": ["BU", "RU", "NR", "TA", "MA", "EG", "PP", "L", "V", "FU", "SC", "FG", "EB", "SA", "UR", "PG"],
    "agri": ["A", "B", "C", "CS", "M", "Y", "P", "OI", "RM", "CF", "CY", "SR", "AP", "CJ", "JD", "RR", "SP", "FB"],
}


def symbol_group_map():
    out = {}
    for g, syms in PRIOR_GROUPS.items():
        for s in syms:
            out[s] = g
    return out


def scrub(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def load_data(cfg, top_n):
    data_path = factor_data_path()
    feat_cols = select_factor_columns(cfg, data_path, top_n)
    data = pd.read_parquet(data_path, columns=["symbol", "datetime", "label"] + feat_cols)
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["group"] = data["symbol"].map(symbol_group_map()).fillna("other")
    return data, feat_cols


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
            "coverage": float((tmp["_pred"].notna() & tmp["label"].notna()).mean()),
            "total_ic": compute_ic(df[col].values, df["label"].values),
            "monthly_mean": m.mean(),
            "monthly_std": m.std(),
            "ir": m.mean() / m.std(),
            **{f"ic_{k}": v for k, v in y.items()},
        })
    return pd.DataFrame(rows)


def run():
    cfg = load_config()
    top_n = int(os.environ.get("GROUP_LGB_TOP_N", "500"))
    lookback = int(os.environ.get("GROUP_LGB_LOOKBACK", "12"))
    max_rows = int(os.environ.get("GROUP_LGB_MAX_ROWS", "120000"))
    data, feat_cols = load_data(cfg, top_n)
    print(f"[group_lgb] rows={len(data)} features={len(feat_cols)} lookback={lookback} max_rows/group={max_rows}", flush=True)
    params = dict(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_samples=80,
        reg_lambda=4.0,
        n_jobs=-1,
        verbose=-1,
    )
    out = data[["symbol", "datetime", "label", "group"]].copy()
    out["pred"] = np.nan
    start = pd.Timestamp(cfg["start_date"])
    end = min(pd.Timestamp(cfg["end_date"]), data["datetime"].max())
    for ms in pd.date_range(start, end, freq="MS"):
        next_ms = ms + pd.DateOffset(months=1)
        tr_start = ms - pd.DateOffset(months=lookback)
        for grp in PRIOR_GROUPS:
            tr = data[(data.group == grp) & (data.datetime >= tr_start) & (data.datetime < ms)].dropna(subset=["label"])
            pr = data[(data.group == grp) & (data.datetime >= ms) & (data.datetime < next_ms)]
            if len(tr) < 5000 or len(pr) == 0:
                continue
            if len(tr) > max_rows:
                tr = tr.sample(max_rows, random_state=42)
            Xtr = scrub(tr[feat_cols].to_numpy(np.float32))
            ytr = tr["label"].to_numpy(np.float32)
            Xpr = scrub(pr[feat_cols].to_numpy(np.float32))
            model = lgb.LGBMRegressor(**params)
            model.fit(Xtr, ytr)
            out.loc[pr.index, "pred"] = model.predict(Xpr)
        pr_idx = out[(out.datetime >= ms) & (out.datetime < next_ms)].index
        mic = compute_ic(out.loc[pr_idx, "pred"].values, out.loc[pr_idx, "label"].values)
        print(f"  [group-lb{lookback}][{ms:%Y-%m}] IC={mic:.4f}", flush=True)

    pred = add_norms(out)
    name = f"group_lgb_prior_lb{lookback}_top{top_n}_n{max_rows}"
    pred.to_parquet(os.path.join(cfg["output_dir"], f"predictions_{name}.parquet"), index=False)
    eval_df = pred[pred.datetime >= pd.Timestamp(cfg["start_date"])].copy()
    res = summarize(eval_df)
    rep = os.path.join(cfg["reports_dir"], "plan_a")
    os.makedirs(rep, exist_ok=True)
    res.to_csv(os.path.join(rep, f"ablation_{name}.csv"), index=False)
    print(res.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
