"""
Leak-free rolling ensemble for component predictions.

For each prediction month, ensemble weights are selected using only component
predictions and labels strictly before that month. This avoids the full-OOS
weight tuning leakage that is fine for diagnostics but invalid for final
sequential predictions.
"""
import os
import sys
sys.path.insert(0, "/root/feature_model")

import numpy as np
import pandas as pd

from src.data.loader import load_config
from src.evaluation.metrics import compute_ic, ic_by_period


BASE_COMPONENTS = [
    ("u12", "predictions_plan_a_lgb.parquet", "rank"),
    ("u6", "predictions_lgb_lb6_top500_n400000.parquet", "rank"),
    ("g12", "predictions_group_lgb_prior_lb12_top500_n120000.parquet", "rank"),
    ("u12t800", "predictions_lgb_lb12_top800_n250000.parquet", "rank"),
    ("resmlp", "predictions_resmlp_lb12_top300_n150000_e2.parquet", "z"),
]
OPTIONAL_COMPONENTS = [
    ("alibi", "predictions_alibi_lb12_top64_seq30_n80000_e1.parquet", "z"),
]


def parse_extra_components():
    """
    ENSEMBLE_EXTRA_COMPONENTS format:
    name:filename:mode;name2:filename2:mode
    mode is rank or z.
    """
    raw = os.environ.get("ENSEMBLE_EXTRA_COMPONENTS", "").strip()
    if not raw:
        return []
    out = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad ENSEMBLE_EXTRA_COMPONENTS item: {item}")
        name, fname, mode = parts
        if mode not in {"rank", "z"}:
            raise ValueError(f"bad component mode for {name}: {mode}")
        out.append((name, fname, mode))
    return out


def parse_excluded_components():
    raw = os.environ.get("ENSEMBLE_EXCLUDE_COMPONENTS", "").strip()
    if not raw:
        return set()
    return {x.strip() for x in raw.replace(";", ",").split(",") if x.strip()}


def compute_component_signal(df, mode):
    g = df.groupby("datetime")["pred"]
    if mode == "rank":
        return g.rank(pct=True) - 0.5
    if mode == "z":
        return (df["pred"] - g.transform("mean")) / (g.transform("std") + 1e-9)
    raise ValueError(mode)


def load_components(output_dir):
    base = None
    for name, fname, mode in COMPONENTS:
        path = os.path.join(output_dir, fname)
        df = pd.read_parquet(path, columns=["symbol", "datetime", "label", "pred"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df[name] = compute_component_signal(df, mode).astype(np.float32)
        cols = ["symbol", "datetime", "label", name] if base is None else ["symbol", "datetime", name]
        base = df[cols].copy() if base is None else base.merge(df[cols], on=["symbol", "datetime"], how="inner")
    return base


def simplex_grid(n, step=0.1):
    units = int(round(1 / step))
    out = []

    def rec(prefix, remain, k):
        if k == 1:
            out.append(prefix + [remain])
            return
        for v in range(remain + 1):
            rec(prefix + [v], remain - v, k - 1)

    rec([], units, n)
    return np.asarray(out, dtype=np.float64) / units


def best_weight(P, y, W):
    mask = np.isfinite(y) & np.all(np.isfinite(P), axis=1)
    if mask.sum() < 5000:
        return None, np.nan
    P = P[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    # IC(w) can be evaluated from sufficient statistics:
    # E[(P w)y] = w'E[P y], E[(P w)^2] = w'E[P'P]w.
    c = (P * y[:, None]).mean(axis=0)
    G = (P.T @ P) / len(P)
    yy = (y * y).mean()
    num = W @ c
    pred_var = np.einsum("ij,jk,ik->i", W, G, W)
    den = np.sqrt(np.maximum(pred_var, 1e-18) * yy)
    ic = num / np.maximum(den, 1e-12)
    j = int(np.nanargmax(ic))
    return W[j], float(ic[j])


def best_weight_slsqp(P, y, W, max_weight=0.5):
    mask = np.isfinite(y) & np.all(np.isfinite(P), axis=1)
    if mask.sum() < 5000:
        return None, np.nan
    P = P[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    c = (P * y[:, None]).mean(axis=0)
    G = (P.T @ P) / len(P)
    yy = (y * y).mean()
    grid_w, grid_ic = best_weight(P, y, W)
    if grid_w is None:
        return None, np.nan
    try:
        from scipy.optimize import minimize
    except Exception:
        return grid_w, grid_ic

    n = len(c)
    max_weight = max(float(max_weight), 1.0 / n)
    bounds = [(0.0, max_weight) for _ in range(n)]
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def ic_value(w):
        pred_var = float(w @ G @ w)
        den = np.sqrt(max(pred_var, 1e-18) * yy)
        return float((w @ c) / max(den, 1e-12))

    def objective(w):
        return -ic_value(w)

    starts = [grid_w, np.ones(n, dtype=np.float64) / n]
    best_w = grid_w.astype(np.float64)
    best_ic = grid_ic
    for w0 in starts:
        w0 = np.clip(w0.astype(np.float64), 0.0, max_weight)
        if w0.sum() <= 1e-12:
            w0 = np.ones(n, dtype=np.float64) / n
        w0 = w0 / w0.sum()
        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 200, "ftol": 1e-10, "disp": False})
        if res.success and np.all(np.isfinite(res.x)):
            w = np.clip(res.x, 0.0, max_weight)
            w = w / w.sum()
            val = ic_value(w)
            if val > best_ic:
                best_w, best_ic = w, val
    return best_w, float(best_ic)


def best_weight_signed_slsqp(P, y, W, max_weight=0.5, min_weight=-0.1):
    mask = np.isfinite(y) & np.all(np.isfinite(P), axis=1)
    if mask.sum() < 5000:
        return None, np.nan
    P = P[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    c = (P * y[:, None]).mean(axis=0)
    G = (P.T @ P) / len(P)
    yy = (y * y).mean()
    grid_w, grid_ic = best_weight(P, y, W)
    if grid_w is None:
        return None, np.nan
    try:
        from scipy.optimize import minimize
    except Exception:
        return grid_w, grid_ic

    n = len(c)
    max_weight = max(float(max_weight), 1.0 / n)
    min_weight = min(float(min_weight), 0.0)
    bounds = [(min_weight, max_weight) for _ in range(n)]
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def ic_value(w):
        pred_var = float(w @ G @ w)
        den = np.sqrt(max(pred_var, 1e-18) * yy)
        return float((w @ c) / max(den, 1e-12))

    def objective(w):
        return -ic_value(w)

    def make_feasible_start(w0):
        w = np.clip(w0.astype(np.float64), min_weight, max_weight)
        for _ in range(8):
            if abs(w.sum() - 1.0) < 1e-9:
                break
            w = np.clip(w + (1.0 - w.sum()) / n, min_weight, max_weight)
        if abs(w.sum() - 1.0) > 1e-6 or np.any(w < min_weight - 1e-9) or np.any(w > max_weight + 1e-9):
            return np.ones(n, dtype=np.float64) / n
        return w

    starts = [grid_w.astype(np.float64), np.ones(n, dtype=np.float64) / n]
    try:
        ridge = 1e-6 * max(float(np.trace(G)) / max(n, 1), 1e-12)
        raw = np.linalg.solve(G + ridge * np.eye(n), c)
        if np.all(np.isfinite(raw)) and abs(raw.sum()) > 1e-12:
            starts.append(raw / raw.sum())
    except np.linalg.LinAlgError:
        pass

    best_w = grid_w.astype(np.float64)
    best_ic = grid_ic
    for w0 in starts:
        w0 = make_feasible_start(w0)
        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 300, "ftol": 1e-10, "disp": False})
        if res.success and np.all(np.isfinite(res.x)):
            w = res.x.astype(np.float64)
            if abs(w.sum() - 1.0) > 1e-6 or np.any(w < min_weight - 1e-8) or np.any(w > max_weight + 1e-8):
                continue
            val = ic_value(w)
            if val > best_ic:
                best_w, best_ic = w, val
    return best_w, float(best_ic)


def add_month_col(df):
    out = df.copy()
    out["_month"] = out["datetime"].dt.to_period("M").astype(str)
    return out


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
    global COMPONENTS
    COMPONENTS = components
    data = load_components(out_dir)
    data = data[data["datetime"] >= pd.Timestamp(cfg["start_date"])].copy()
    data = add_month_col(data)
    names = [x[0] for x in COMPONENTS]
    W = simplex_grid(len(names), step=float(os.environ.get("ENSEMBLE_STEP", "0.1")))
    optimizer = os.environ.get("ENSEMBLE_OPTIMIZER", "grid")
    max_weight = float(os.environ.get("ENSEMBLE_MAX_WEIGHT", "0.5"))
    min_months = int(os.environ.get("ENSEMBLE_MIN_MONTHS", "6"))
    lookback_months = int(os.environ.get("ENSEMBLE_LOOKBACK_MONTHS", "0"))
    default_w = np.ones(len(names), dtype=np.float64) / len(names)
    data["pred"] = np.nan
    records = []
    months = sorted(data["_month"].unique())
    for i, month in enumerate(months):
        cur_idx = data.index[data["_month"] == month]
        hist_months = months[:i]
        if lookback_months > 0:
            hist_months = hist_months[-lookback_months:]
        if len(hist_months) >= min_months:
            hist = data[data["_month"].isin(hist_months)]
            if optimizer == "slsqp":
                w, train_ic = best_weight_slsqp(
                    hist[names].to_numpy(np.float32),
                    hist["label"].to_numpy(np.float64),
                    W,
                    max_weight=max_weight,
                )
            elif optimizer == "signed_slsqp":
                w, train_ic = best_weight_signed_slsqp(
                    hist[names].to_numpy(np.float32),
                    hist["label"].to_numpy(np.float64),
                    W,
                    max_weight=max_weight,
                    min_weight=float(os.environ.get("ENSEMBLE_MIN_WEIGHT", "-0.1")),
                )
            else:
                w, train_ic = best_weight(hist[names].to_numpy(np.float32), hist["label"].to_numpy(np.float64), W)
            if w is None:
                w, train_ic = default_w, np.nan
        else:
            w, train_ic = default_w, np.nan
        curP = data.loc[cur_idx, names].to_numpy(np.float32)
        data.loc[cur_idx, "pred"] = np.nan_to_num(curP, nan=0.0, posinf=0.0, neginf=0.0) @ w.astype(np.float32)
        month_ic = compute_ic(data.loc[cur_idx, "pred"].values, data.loc[cur_idx, "label"].values)
        records.append({"month": month, "train_ic": train_ic, "month_ic": month_ic, **{f"w_{n}": w[j] for j, n in enumerate(names)}})
        print(f"  [rolling-ens][{month}] train_ic={train_ic:.4f} month_ic={month_ic:.4f} w={np.round(w,2)}", flush=True)
    pred = data[["symbol", "datetime", "label", "pred"]].copy()
    pred_name = os.environ.get("ENSEMBLE_OUTPUT_NAME", "predictions_rolling_ensemble.parquet")
    pred_path = os.path.join(out_dir, pred_name)
    pred.to_parquet(pred_path, index=False)
    write_final = os.environ.get("ENSEMBLE_WRITE_FINAL", "1") == "1"
    if write_final:
        pred.to_parquet(os.path.join(out_dir, "predictions.parquet"), index=False)
    weights = pd.DataFrame(records)
    stem = os.path.splitext(os.path.basename(pred_name))[0]
    report_suffix = "" if write_final else f"_{stem}"
    weights.to_csv(os.path.join(rep, f"rolling_ensemble_weights{report_suffix}.csv"), index=False)
    by_y = ic_by_period(pred, "pred", "label", "Y")
    by_m = ic_by_period(pred, "pred", "label", "M")
    summary = {
        "total_ic": compute_ic(pred["pred"].values, pred["label"].values),
        "monthly_mean": by_m.mean(),
        "monthly_std": by_m.std(),
        "ir": by_m.mean() / by_m.std(),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(rep, f"rolling_ensemble_summary{report_suffix}.csv"), index=False)
    print(pd.Series(summary).to_string(), flush=True)
    print(by_y.to_string(), flush=True)


if __name__ == "__main__":
    run()
