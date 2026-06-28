"""
Leak-free daily expert-level MOE.

For each prediction block this model estimates nonnegative expert weights from
prior days by directly maximizing the weighted training IC.  It is an automatic
gating layer over saved model predictions, not a hand-picked monthly switch.
"""
import os
import sys

sys.path.insert(0, "/root/feature_model")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.data.loader import load_config
from src.evaluation.metrics import compute_ic, ic_by_period
from src.plan_a.daily_stack_ridge import component_signal, load_components, parse_components


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
                "coverage": float((df[col].notna() & df["label"].notna()).mean()),
                "total_ic": compute_ic(df[col].values, df["label"].values),
                "monthly_mean": m.mean(),
                "monthly_std": m.std(),
                "ir": m.mean() / m.std(),
                **{f"ic_{k}": v for k, v in y.items()},
            }
        )
    return pd.DataFrame(rows)


def weighted_moments(P, y, weight):
    mask = np.isfinite(y) & np.all(np.isfinite(P), axis=1)
    P = P[mask].astype(np.float64, copy=False)
    y = y[mask].astype(np.float64, copy=False)
    weight = weight[mask].astype(np.float64, copy=False)
    if len(y) < 5000:
        return None
    weight = np.maximum(weight, 0.0)
    sw = float(weight.sum())
    if sw <= 1e-12:
        weight = np.ones_like(y, dtype=np.float64)
        sw = float(weight.sum())
    weight = weight / sw
    c = (P * (y * weight)[:, None]).sum(axis=0)
    G = P.T @ (P * weight[:, None])
    yy = float((weight * y * y).sum())
    return c, G, yy, P, y


def make_feasible(w, min_weight, max_weight):
    n = len(w)
    out = np.clip(np.asarray(w, dtype=np.float64), min_weight, max_weight)
    for _ in range(32):
        diff = 1.0 - out.sum()
        if abs(diff) < 1e-10:
            break
        free = (out > min_weight + 1e-10) & (out < max_weight - 1e-10)
        if not np.any(free):
            out = np.ones(n, dtype=np.float64) / n
            break
        out[free] += diff / free.sum()
        out = np.clip(out, min_weight, max_weight)
    if abs(out.sum() - 1.0) > 1e-6:
        out = np.ones(n, dtype=np.float64) / n
    return out


def make_feasible_bounds(w, lower, upper):
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    out = np.clip(np.asarray(w, dtype=np.float64), lower, upper)
    if lower.sum() > 1.0 + 1e-10 or upper.sum() < 1.0 - 1e-10:
        raise ValueError("infeasible expert bounds")
    for _ in range(64):
        diff = 1.0 - float(out.sum())
        if abs(diff) < 1e-10:
            break
        if diff > 0:
            room = np.maximum(upper - out, 0.0)
            total = float(room.sum())
            if total <= 1e-12:
                break
            out += room * min(1.0, diff / total)
        else:
            room = np.maximum(out - lower, 0.0)
            total = float(room.sum())
            if total <= 1e-12:
                break
            out -= room * min(1.0, -diff / total)
        out = np.clip(out, lower, upper)
    if abs(out.sum() - 1.0) > 1e-6:
        out = lower.copy()
        rem = 1.0 - float(out.sum())
        cap = np.maximum(upper - lower, 0.0)
        if rem > 0 and cap.sum() > 0:
            out += cap * (rem / float(cap.sum()))
        out = np.clip(out, lower, upper)
    return out


def best_ic_weights(P, y, sample_weight, max_weight, min_weight, prev_w=None, lower_bounds=None, upper_bounds=None):
    stats = weighted_moments(P, y, sample_weight)
    if stats is None:
        n = P.shape[1]
        return np.ones(n, dtype=np.float64) / n, np.nan
    c, G, yy, Pm, ym = stats
    n = len(c)
    if lower_bounds is None or upper_bounds is None:
        max_weight = max(float(max_weight), 1.0 / n)
        min_weight = min(float(min_weight), 1.0 / n)
        if min_weight * n > 1.0:
            min_weight = 0.0
        lower = np.full(n, min_weight, dtype=np.float64)
        upper = np.full(n, max_weight, dtype=np.float64)
    else:
        lower = np.asarray(lower_bounds, dtype=np.float64)
        upper = np.asarray(upper_bounds, dtype=np.float64)
        if lower.shape != (n,) or upper.shape != (n,):
            raise ValueError("expert bound arrays have wrong shape")
        if lower.sum() > 1.0 + 1e-10 or upper.sum() < 1.0 - 1e-10:
            raise ValueError("infeasible expert bounds")

    def ic_value(w):
        pred_var = float(w @ G @ w)
        den = np.sqrt(max(pred_var, 1e-18) * max(yy, 1e-18))
        return float((w @ c) / max(den, 1e-12))

    starts = [np.ones(n, dtype=np.float64) / n]
    if prev_w is not None and len(prev_w) == n:
        starts.append(np.asarray(prev_w, dtype=np.float64))
    best_j = int(np.nanargmax(c / np.sqrt(np.maximum(np.diag(G), 1e-18) * max(yy, 1e-18))))
    unit = lower.copy()
    unit[best_j] = min(upper[best_j], 1.0 - float(lower.sum() - lower[best_j]))
    starts.append(unit)
    pos = np.maximum(c, 0.0)
    if pos.sum() > 1e-12:
        starts.append(pos / pos.sum())
    try:
        ridge = 1e-8 * max(float(np.trace(G)) / max(n, 1), 1e-12)
        raw = np.linalg.solve(G + ridge * np.eye(n), c)
        if np.all(np.isfinite(raw)) and abs(raw.sum()) > 1e-12:
            starts.append(raw / raw.sum())
    except np.linalg.LinAlgError:
        pass

    bounds = [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    best_w = make_feasible_bounds(starts[0], lower, upper)
    best_ic = ic_value(best_w)
    for start in starts:
        w0 = make_feasible_bounds(start, lower, upper)
        res = minimize(
            lambda w: -ic_value(w),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-12, "disp": False},
        )
        if not res.success or not np.all(np.isfinite(res.x)):
            continue
        w = make_feasible_bounds(res.x, lower, upper)
        val = ic_value(w)
        if val > best_ic:
            best_w, best_ic = w, val
    train_pred = Pm @ best_w
    train_ic = compute_ic(train_pred, ym)
    return best_w.astype(np.float64), train_ic


def sample_weights(day_code, idx, cur_d, half_life_days):
    if half_life_days <= 0:
        return np.ones(len(idx), dtype=np.float64)
    age = np.maximum(cur_d - day_code[idx], 1).astype(np.float64)
    return np.exp(-np.log(2.0) * age / float(half_life_days))


def parse_component_max_weights(raw):
    out = {}
    for item in raw.replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        name, value = item.split(":", 1)
        out[name.strip()] = float(value)
    return out


def run():
    cfg = load_config()
    comps = parse_components()
    names = [x[0] for x in comps]
    data = load_components(cfg["output_dir"], comps)
    data = data[data["datetime"] >= pd.Timestamp(cfg["start_date"])].copy()
    data["_day"] = data["datetime"].dt.floor("D")
    days = np.array(sorted(data["_day"].unique()))
    day_code = pd.Categorical(data["_day"], categories=days, ordered=True).codes
    X_all = data[names].to_numpy(np.float32)
    y_all = data["label"].to_numpy(np.float64)

    lookback_days = int(os.environ.get("EXPERT_LOOKBACK_DAYS", "90"))
    min_days = int(os.environ.get("EXPERT_MIN_DAYS", "25"))
    retrain_days = int(os.environ.get("EXPERT_RETRAIN_DAYS", "5"))
    max_rows = int(os.environ.get("EXPERT_MAX_ROWS", "900000"))
    half_life = float(os.environ.get("EXPERT_HALF_LIFE_DAYS", "45"))
    max_weight = float(os.environ.get("EXPERT_MAX_WEIGHT", "0.65"))
    min_weight = float(os.environ.get("EXPERT_MIN_WEIGHT", "0.0"))
    anchor_min_weight = float(os.environ.get("EXPERT_ANCHOR_MIN_WEIGHT", "0.0"))
    anchor_max_weight = float(os.environ.get("EXPERT_ANCHOR_MAX_WEIGHT", "1.0"))
    component_max = parse_component_max_weights(os.environ.get("EXPERT_COMPONENT_MAX_WEIGHTS", ""))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    out = data[["symbol", "datetime", "label"]].copy()
    out["pred"] = np.nan
    default_w = np.ones(len(names), dtype=np.float64) / max(len(names), 1)
    prev_w = default_w.copy()
    records = []
    print(
        f"[daily_expert_moe] rows={len(data)} components={len(names)} days={len(days)} "
        f"lookback={lookback_days} min_days={min_days} retrain={retrain_days} "
        f"max_rows={max_rows} half_life={half_life} max_w={max_weight} "
        f"anchor_min={anchor_min_weight}",
        flush=True,
    )
    lower_bounds = None
    upper_bounds = None
    if anchor_min_weight > 0.0:
        lower_bounds = np.full(len(names), min_weight, dtype=np.float64)
        upper_bounds = np.full(len(names), max_weight, dtype=np.float64)
        lower_bounds[0] = max(anchor_min_weight, min_weight)
        upper_bounds[0] = max(anchor_max_weight, lower_bounds[0])
        for i, name in enumerate(names):
            if name in component_max:
                upper_bounds[i] = max(lower_bounds[i], component_max[name])
        default_w = make_feasible_bounds(default_w, lower_bounds, upper_bounds)
        prev_w = default_w.copy()

    d = 0
    while d < len(days):
        block_end = min(len(days), d + retrain_days)
        cur = (day_code >= d) & (day_code < block_end)
        if d >= min_days:
            lo = max(0, d - lookback_days)
            tr_mask = (day_code >= lo) & (day_code < d) & np.isfinite(y_all)
            idx = np.flatnonzero(tr_mask)
            if max_rows and len(idx) > max_rows:
                idx = rng.choice(idx, max_rows, replace=False)
            wgt = sample_weights(day_code, idx, d, half_life)
            w, train_ic = best_ic_weights(
                np.nan_to_num(X_all[idx], nan=0.0, posinf=0.0, neginf=0.0),
                y_all[idx],
                wgt,
                max_weight=max_weight,
                min_weight=min_weight,
                prev_w=prev_w,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            prev_w = w
        else:
            w = default_w
            train_ic = np.nan
        pred = np.nan_to_num(X_all[cur], nan=0.0, posinf=0.0, neginf=0.0) @ w
        out.loc[data.index[cur], "pred"] = pred
        block_ic = compute_ic(pred, data.loc[data.index[cur], "label"].values)
        row = {
            "day_start": str(pd.Timestamp(days[d]).date()),
            "day_end": str(pd.Timestamp(days[block_end - 1]).date()),
            "train_ic": train_ic,
            "block_ic": block_ic,
        }
        row.update({f"w_{name}": float(val) for name, val in zip(names, w)})
        records.append(row)
        top = sorted(zip(names, w), key=lambda x: x[1], reverse=True)[:3]
        print(
            f"  [expert-moe][{pd.Timestamp(days[d]):%Y-%m-%d}.."
            f"{pd.Timestamp(days[block_end - 1]):%Y-%m-%d}] train_ic={train_ic:.4f} "
            f"block_ic={block_ic:.4f} top={','.join(f'{n}:{v:.2f}' for n, v in top)}",
            flush=True,
        )
        d = block_end

    pred = add_norms(out)
    name = (
        f"daily_expert_moe_lb{lookback_days}_min{min_days}_rt{retrain_days}_"
        f"n{max_rows}_hl{half_life:g}_mw{max_weight:g}_c{len(names)}"
    )
    suffix = os.environ.get("EXPERT_NAME_SUFFIX", "").strip()
    if suffix:
        safe_suffix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in suffix)
        name = f"{name}_{safe_suffix}"
    pred_path = os.path.join(cfg["output_dir"], f"predictions_{name}.parquet")
    pred.to_parquet(pred_path, index=False)
    rep = os.path.join(cfg["reports_dir"], "plan_a")
    os.makedirs(rep, exist_ok=True)
    pd.DataFrame(records).to_csv(os.path.join(rep, f"daily_expert_weights_{name}.csv"), index=False)
    res = summarize(pred)
    res.to_csv(os.path.join(rep, f"ablation_{name}.csv"), index=False)
    print(res.to_string(index=False), flush=True)
    print(f"[daily_expert_moe] wrote {pred_path}", flush=True)


if __name__ == "__main__":
    run()
