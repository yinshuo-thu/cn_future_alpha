"""
Leak-free prior-group rolling ensemble.

This variant keeps the same component pool as rolling_ensemble.py, but allows
each prior sector group to use its own historical ensemble weights. The final
prediction is a fixed shrinkage blend of global and group-specific predictions.
"""
import os
import sys
sys.path.insert(0, "/root/feature_model")

import numpy as np
import pandas as pd

from src.data.loader import load_config
from src.evaluation.metrics import compute_ic, ic_by_period
from src.plan_a.group_lgb import symbol_group_map
from src.plan_a.rolling_ensemble import (
    BASE_COMPONENTS,
    OPTIONAL_COMPONENTS,
    best_weight,
    best_weight_signed_slsqp,
    best_weight_slsqp,
    compute_component_signal,
    parse_excluded_components,
    parse_extra_components,
    simplex_grid,
)


def parse_alphas():
    raw = os.environ.get("GROUP_ENSEMBLE_ALPHAS", "0.25,0.5,0.75,1.0")
    vals = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("GROUP_ENSEMBLE_ALPHAS produced no values")
    return vals


def parse_alpha_grid():
    raw = os.environ.get("GROUP_ENSEMBLE_ALPHA_GRID", "0,0.05,0.1,0.13,0.15,0.2,0.25,0.3,0.4,0.5")
    vals = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError("GROUP_ENSEMBLE_ALPHA_GRID produced no values")
    return vals


def load_components(output_dir, components):
    base = None
    for name, fname, mode in components:
        path = os.path.join(output_dir, fname)
        df = pd.read_parquet(path, columns=["symbol", "datetime", "label", "pred"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df[name] = compute_component_signal(df, mode).astype(np.float32)
        cols = ["symbol", "datetime", "label", name] if base is None else ["symbol", "datetime", name]
        base = df[cols].copy() if base is None else base.merge(df[cols], on=["symbol", "datetime"], how="inner")
    groups = symbol_group_map()
    base["group"] = base["symbol"].map(groups).fillna("other").astype("category")
    return base


def choose_weight(hist, names, W, optimizer, max_weight, min_weight):
    P = hist[names].to_numpy(np.float32)
    y = hist["label"].to_numpy(np.float64)
    if optimizer == "slsqp":
        return best_weight_slsqp(P, y, W, max_weight=max_weight)
    if optimizer == "signed_slsqp":
        return best_weight_signed_slsqp(P, y, W, max_weight=max_weight, min_weight=min_weight)
    return best_weight(P, y, W)


def summarize(pred):
    by_y = ic_by_period(pred, "pred", "label", "Y")
    by_m = ic_by_period(pred, "pred", "label", "M")
    return {
        "total_ic": compute_ic(pred["pred"].values, pred["label"].values),
        "monthly_mean": by_m.mean(),
        "monthly_std": by_m.std(),
        "ir": by_m.mean() / by_m.std(),
        **{f"ic_{k}": v for k, v in by_y.items()},
    }


def blend_arrays(global_pred, group_pred, alpha):
    return (
        (1.0 - alpha) * global_pred.astype(np.float32)
        + alpha * group_pred.astype(np.float32)
    ).astype(np.float32)


def best_alpha_from_history(hist, alpha_grid):
    y = hist["label"].to_numpy(np.float64)
    gp = hist["pred_global"].to_numpy(np.float64)
    pp = hist["pred_group"].to_numpy(np.float64)
    mask = np.isfinite(y) & np.isfinite(gp) & np.isfinite(pp)
    if mask.sum() < 5000:
        return None, np.nan
    y = y[mask]
    gp = gp[mask]
    diff = pp[mask] - gp
    y2 = float(np.mean(y * y))
    gy = float(np.mean(gp * y))
    dy = float(np.mean(diff * y))
    g2 = float(np.mean(gp * gp))
    gd = float(np.mean(gp * diff))
    d2 = float(np.mean(diff * diff))
    best_alpha = None
    best_ic = -np.inf
    for alpha in alpha_grid:
        pred_var = g2 + 2.0 * alpha * gd + alpha * alpha * d2
        num = gy + alpha * dy
        ic = num / np.sqrt(max(pred_var * y2, 1e-30))
        if ic > best_ic:
            best_alpha = alpha
            best_ic = ic
    return best_alpha, float(best_ic)


def rolling_alpha_prediction(pred_base, months, min_months, lookback_months, alpha_grid):
    pred = pred_base[["symbol", "datetime", "label"]].copy()
    pred["pred"] = np.nan
    records = []
    for i, month in enumerate(months):
        cur_mask = pred_base["_month"] == month
        hist_months = months[:i]
        if lookback_months > 0:
            hist_months = hist_months[-lookback_months:]
        if len(hist_months) >= min_months:
            hist = pred_base[pred_base["_month"].isin(hist_months)]
            alpha, train_ic = best_alpha_from_history(hist, alpha_grid)
            if alpha is None:
                alpha, train_ic = alpha_grid[0], np.nan
        else:
            alpha, train_ic = alpha_grid[0], np.nan
        pred.loc[cur_mask, "pred"] = blend_arrays(
            pred_base.loc[cur_mask, "pred_global"].to_numpy(),
            pred_base.loc[cur_mask, "pred_group"].to_numpy(),
            alpha,
        )
        month_ic = compute_ic(pred.loc[cur_mask, "pred"].values, pred.loc[cur_mask, "label"].values)
        records.append({"month": month, "alpha": alpha, "alpha_train_ic": train_ic, "month_ic": month_ic})
        print(f"  [group-alpha][{month}] alpha={alpha:g} train_ic={train_ic:.4f} month_ic={month_ic:.4f}", flush=True)
    return pred, pd.DataFrame(records)


def rolling_group_alpha_prediction(pred_base, months, min_months, lookback_months, alpha_grid, min_rows):
    pred = pred_base[["symbol", "datetime", "label"]].copy()
    pred["pred"] = np.nan
    records = []
    group_names = list(pred_base["group"].cat.categories)
    for i, month in enumerate(months):
        cur_month_mask = pred_base["_month"] == month
        hist_months = months[:i]
        if lookback_months > 0:
            hist_months = hist_months[-lookback_months:]
        month_alpha_rows = []
        for grp in group_names:
            cur_mask = cur_month_mask & (pred_base["group"] == grp)
            if not bool(cur_mask.any()):
                continue
            alpha, train_ic = alpha_grid[0], np.nan
            if len(hist_months) >= min_months:
                hist = pred_base[(pred_base["_month"].isin(hist_months)) & (pred_base["group"] == grp)]
                if len(hist) >= min_rows:
                    cand_alpha, cand_ic = best_alpha_from_history(hist, alpha_grid)
                    if cand_alpha is not None:
                        alpha, train_ic = cand_alpha, cand_ic
            pred.loc[cur_mask, "pred"] = blend_arrays(
                pred_base.loc[cur_mask, "pred_global"].to_numpy(),
                pred_base.loc[cur_mask, "pred_group"].to_numpy(),
                alpha,
            )
            month_alpha_rows.append((grp, alpha, train_ic))
        month_ic = compute_ic(pred.loc[cur_month_mask, "pred"].values, pred.loc[cur_month_mask, "label"].values)
        rec = {"month": month, "month_ic": month_ic}
        for grp, alpha, train_ic in month_alpha_rows:
            rec[f"alpha_{grp}"] = alpha
            rec[f"train_ic_{grp}"] = train_ic
        records.append(rec)
        alpha_text = ",".join(f"{g}:{a:g}" for g, a, _ in month_alpha_rows)
        print(f"  [group-alpha-bygrp][{month}] month_ic={month_ic:.4f} alpha={alpha_text}", flush=True)
    return pred, pd.DataFrame(records)


def run():
    cfg = load_config()
    out_dir = cfg["output_dir"]
    rep = os.path.join(cfg["reports_dir"], "plan_a")
    os.makedirs(rep, exist_ok=True)

    components = list(BASE_COMPONENTS)
    if os.environ.get("ENSEMBLE_INCLUDE_ALIBI", "0") == "1":
        components.extend(OPTIONAL_COMPONENTS)
    components.extend(parse_extra_components())
    excluded = parse_excluded_components()
    if excluded:
        components = [c for c in components if c[0] not in excluded]
        if not components:
            raise ValueError("ENSEMBLE_EXCLUDE_COMPONENTS removed all components")

    names = [x[0] for x in components]
    data = load_components(out_dir, components)
    data = data[data["datetime"] >= pd.Timestamp(cfg["start_date"])].copy()
    data["_month"] = data["datetime"].dt.to_period("M").astype(str)

    W = simplex_grid(len(names), step=float(os.environ.get("ENSEMBLE_STEP", "0.1")))
    optimizer = os.environ.get("ENSEMBLE_OPTIMIZER", "slsqp")
    max_weight = float(os.environ.get("ENSEMBLE_MAX_WEIGHT", "0.4"))
    min_weight = float(os.environ.get("ENSEMBLE_MIN_WEIGHT", "-0.1"))
    min_months = int(os.environ.get("ENSEMBLE_MIN_MONTHS", "10"))
    lookback_months = int(os.environ.get("ENSEMBLE_LOOKBACK_MONTHS", "0"))
    min_group_rows = int(os.environ.get("GROUP_ENSEMBLE_MIN_ROWS", "20000"))
    default_w = np.ones(len(names), dtype=np.float64) / len(names)

    data["pred_global"] = np.nan
    data["pred_group"] = np.nan
    records = []
    months = sorted(data["_month"].unique())
    group_names = list(data["group"].cat.categories)
    for i, month in enumerate(months):
        cur_mask = data["_month"] == month
        hist_months = months[:i]
        if lookback_months > 0:
            hist_months = hist_months[-lookback_months:]

        if len(hist_months) >= min_months:
            hist = data[data["_month"].isin(hist_months)]
            global_w, global_ic = choose_weight(hist, names, W, optimizer, max_weight, min_weight)
            if global_w is None:
                global_w, global_ic = default_w, np.nan
        else:
            hist = None
            global_w, global_ic = default_w, np.nan

        curP = data.loc[cur_mask, names].to_numpy(np.float32)
        global_pred = np.nan_to_num(curP, nan=0.0, posinf=0.0, neginf=0.0) @ global_w.astype(np.float32)
        data.loc[cur_mask, "pred_global"] = global_pred
        data.loc[cur_mask, "pred_group"] = global_pred

        month_records = []
        for grp in group_names:
            grp_cur_mask = cur_mask & (data["group"] == grp)
            if not bool(grp_cur_mask.any()):
                continue
            grp_w, grp_ic = global_w, np.nan
            if hist is not None:
                grp_hist = hist[hist["group"] == grp]
                if len(grp_hist) >= min_group_rows:
                    cand_w, cand_ic = choose_weight(grp_hist, names, W, optimizer, max_weight, min_weight)
                    if cand_w is not None:
                        grp_w, grp_ic = cand_w, cand_ic
            grpP = data.loc[grp_cur_mask, names].to_numpy(np.float32)
            data.loc[grp_cur_mask, "pred_group"] = (
                np.nan_to_num(grpP, nan=0.0, posinf=0.0, neginf=0.0) @ grp_w.astype(np.float32)
            )
            month_records.append((grp, grp_ic, grp_w))

        g_ic = compute_ic(data.loc[cur_mask, "pred_global"].values, data.loc[cur_mask, "label"].values)
        p_ic = compute_ic(data.loc[cur_mask, "pred_group"].values, data.loc[cur_mask, "label"].values)
        records.append({"month": month, "global_train_ic": global_ic, "month_global_ic": g_ic, "month_group_ic": p_ic})
        print(f"  [group-ens][{month}] global_train_ic={global_ic:.4f} global_ic={g_ic:.4f} group_ic={p_ic:.4f}", flush=True)

    alphas = parse_alphas()
    pred_base = data[["symbol", "datetime", "label", "group", "_month", "pred_global", "pred_group"]].copy()
    summaries = []
    best_alpha = None
    best_ic = -np.inf
    best_pred = None
    for alpha in alphas:
        pred = pred_base[["symbol", "datetime", "label"]].copy()
        pred["pred"] = (
            (1.0 - alpha) * pred_base["pred_global"].to_numpy(np.float32)
            + alpha * pred_base["pred_group"].to_numpy(np.float32)
        ).astype(np.float32)
        row = {"alpha": alpha, **summarize(pred)}
        summaries.append(row)
        if row["total_ic"] > best_ic:
            best_ic = row["total_ic"]
            best_alpha = alpha
            best_pred = pred

    stem = os.environ.get("GROUP_ENSEMBLE_OUTPUT_STEM", "group_rolling_ensemble")
    pd.DataFrame(records).to_csv(os.path.join(rep, f"{stem}_monthly.csv"), index=False)
    alpha_mode = os.environ.get("GROUP_ENSEMBLE_ALPHA_MODE", "fixed")
    if alpha_mode == "rolling":
        alpha_grid = parse_alpha_grid()
        alpha_min_months = int(os.environ.get("GROUP_ENSEMBLE_ALPHA_MIN_MONTHS", str(min_months)))
        alpha_lookback = int(os.environ.get("GROUP_ENSEMBLE_ALPHA_LOOKBACK_MONTHS", "0"))
        rolling_pred, alpha_records = rolling_alpha_prediction(
            pred_base, months, alpha_min_months, alpha_lookback, alpha_grid
        )
        rolling_row = {"alpha": "rolling", **summarize(rolling_pred)}
        summaries.append(rolling_row)
        alpha_records.to_csv(os.path.join(rep, f"{stem}_alpha_monthly.csv"), index=False)
        if rolling_row["total_ic"] > best_ic:
            best_ic = rolling_row["total_ic"]
            best_alpha = "rolling"
            best_pred = rolling_pred
    elif alpha_mode == "rolling_group":
        alpha_grid = parse_alpha_grid()
        alpha_min_months = int(os.environ.get("GROUP_ENSEMBLE_ALPHA_MIN_MONTHS", str(min_months)))
        alpha_lookback = int(os.environ.get("GROUP_ENSEMBLE_ALPHA_LOOKBACK_MONTHS", "0"))
        alpha_min_rows = int(os.environ.get("GROUP_ENSEMBLE_ALPHA_MIN_ROWS", str(min_group_rows)))
        rolling_pred, alpha_records = rolling_group_alpha_prediction(
            pred_base, months, alpha_min_months, alpha_lookback, alpha_grid, alpha_min_rows
        )
        rolling_row = {"alpha": "rolling_group", **summarize(rolling_pred)}
        summaries.append(rolling_row)
        alpha_records.to_csv(os.path.join(rep, f"{stem}_alpha_by_group_monthly.csv"), index=False)
        if rolling_row["total_ic"] > best_ic:
            best_ic = rolling_row["total_ic"]
            best_alpha = "rolling_group"
            best_pred = rolling_pred
    elif alpha_mode != "fixed":
        raise ValueError(f"bad GROUP_ENSEMBLE_ALPHA_MODE: {alpha_mode}")
    summary = pd.DataFrame(summaries)
    summary.to_csv(os.path.join(rep, f"{stem}_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)

    alpha_suffix = best_alpha if isinstance(best_alpha, str) else f"alpha{best_alpha:g}"
    out_name = os.environ.get("GROUP_ENSEMBLE_OUTPUT_NAME", f"predictions_{stem}_{alpha_suffix}.parquet")
    best_pred.to_parquet(os.path.join(out_dir, out_name), index=False)
    if os.environ.get("GROUP_ENSEMBLE_WRITE_FINAL", "0") == "1":
        best_pred.to_parquet(os.path.join(out_dir, "predictions.parquet"), index=False)
    best_alpha_text = best_alpha if isinstance(best_alpha, str) else f"{best_alpha:g}"
    print(f"[group-ens] best_alpha={best_alpha_text} best_ic={best_ic:.10f} output={out_name}", flush=True)


if __name__ == "__main__":
    run()
