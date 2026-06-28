#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri


OUT_DIR = Path("/root/autodl-tmp/quant/ML/agent_runs/ridge_parallel_20260628/postprocess_view_scan")
FIT_PATH = Path(
    "/root/autodl-tmp/quant/ML/effective_rolling_results/"
    "ridge_overlap333_xsz_hl12_a05/ridge_overlap333_xsz_hl12_a05.parquet"
)
APPLY_PATH = Path(
    "/root/autodl-tmp/quant/ML/effective_rolling_results/"
    "ridge_overlap333_xsz_hl12_n900k_a02/ridge_overlap333_xsz_hl12_n900k_a02.parquet"
)


def compute_ic(pred: np.ndarray | pd.Series, label: np.ndarray | pd.Series) -> float:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(p) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    p = p[mask]
    y = y[mask]
    den = math.sqrt(float(np.mean(p * p) * np.mean(y * y)))
    if den <= 1e-18:
        return float("nan")
    return float(np.mean(p * y) / den)


def monthly_ic(df: pd.DataFrame, col: str) -> pd.Series:
    clean = df.loc[df[col].notna() & df["label"].notna(), ["_month", col, "label"]]
    if clean.empty:
        return pd.Series(dtype=float)
    return clean.groupby("_month", sort=True).apply(
        lambda g: compute_ic(g[col].to_numpy(), g["label"].to_numpy()),
        include_groups=False,
    )


def period_mask(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    return (df["datetime"] >= pd.Timestamp(start)) & (df["datetime"] < pd.Timestamp(end))


def summarize(df: pd.DataFrame, col: str, start: str, end: str) -> dict[str, float | int]:
    part = df.loc[period_mask(df, start, end)]
    mic = monthly_ic(part, col)
    std = float(mic.std(ddof=1)) if len(mic) > 1 else float("nan")
    return {
        "rows": int(len(part)),
        "label_rows": int(part["label"].notna().sum()),
        "ic": compute_ic(part[col].to_numpy(), part["label"].to_numpy()),
        "monthly_mean": float(mic.mean()) if len(mic) else float("nan"),
        "monthly_std": std,
        "monthly_ir": float(mic.mean() / std) if np.isfinite(std) and std > 0 else float("nan"),
    }


def load_pred(path: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["symbol", "datetime", "label", "pred"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.loc[period_mask(df, start, end)].copy()
    df["_month"] = df["datetime"].dt.strftime("%Y-%m")
    return df


def add_views(df: pd.DataFrame) -> list[str]:
    pred = df["pred"].astype("float64")
    g = df.groupby("datetime", sort=False)["pred"]
    mean = g.transform("mean").astype("float64")
    std = g.transform("std").astype("float64")
    z = (pred - mean) / (std + 1e-9)
    rank = g.rank(pct=True).astype("float64").clip(0.001, 0.999)
    views: dict[str, np.ndarray | pd.Series] = {
        "pred": pred,
        "pred_xcenter": pred - mean,
        "pred_xsz": z,
        "pred_xrank": rank - 0.5,
        "pred_rankgauss": ndtri(rank).clip(-3.1, 3.1),
        "pred_asinh_z": np.arcsinh(z),
        "pred_sqrt_signed_z": np.sign(z) * np.sqrt(np.abs(z)),
    }
    for clip in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0):
        views[f"pred_xsz_clip{clip:g}"] = z.clip(-clip, clip)
    for scale in (0.75, 1.0, 1.5, 2.0, 3.0):
        views[f"pred_tanh_z_s{scale:g}"] = np.tanh(z / scale)
    for name, values in views.items():
        df[name] = np.asarray(values, dtype=np.float32)
    return list(views)


def simplex_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    n = x.shape[1]
    if len(y) < n + 2:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64), float("nan")
    gram = x.T @ x / len(x)
    cov = x.T @ y / len(x)
    best_w = np.full(n, 1.0 / n, dtype=np.float64)
    best_ic = compute_ic(x @ best_w, y)
    for bits in range(1, 1 << n):
        idx = [i for i in range(n) if bits & (1 << i)]
        try:
            w_sub = np.linalg.solve(gram[np.ix_(idx, idx)] + 1e-10 * np.eye(len(idx)), cov[idx])
        except np.linalg.LinAlgError:
            w_sub = np.linalg.pinv(gram[np.ix_(idx, idx)]) @ cov[idx]
        if np.all(w_sub <= 0):
            w_sub = -w_sub
        if np.any(w_sub < -1e-12):
            continue
        w_sub = np.maximum(w_sub, 0.0)
        if w_sub.sum() <= 0:
            continue
        w = np.zeros(n, dtype=np.float64)
        w[idx] = w_sub / w_sub.sum()
        ic = compute_ic(x @ w, y)
        if np.isfinite(ic) and ic > best_ic:
            best_ic = ic
            best_w = w
    return best_w, best_ic


def ridge_view_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[dict[str, np.ndarray | float], float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-9
    xs = (x - mean) / scale
    y_mean = float(y.mean())
    yc = y - y_mean
    gram = xs.T @ xs / len(xs)
    cov = xs.T @ yc / len(xs)
    coef = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), cov)
    pred = xs @ coef + y_mean
    return {"mean": mean, "scale": scale, "coef": coef, "intercept": y_mean}, compute_ic(pred, y)


def apply_ridge_view(x: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    mean = model["mean"]  # type: ignore[assignment]
    scale = model["scale"]  # type: ignore[assignment]
    coef = model["coef"]  # type: ignore[assignment]
    intercept = float(model["intercept"])
    return ((np.asarray(x, dtype=np.float64) - mean) / scale @ coef + intercept).astype(np.float32)


def add_combo(
    df: pd.DataFrame,
    name: str,
    cols: list[str],
    fit_df: pd.DataFrame,
    method: str,
    alpha: float = 0.0,
) -> tuple[str, float, dict[str, float]]:
    if method == "simplex":
        w, fit_ic = simplex_fit(fit_df[cols].to_numpy(np.float64, copy=False), fit_df["label"].to_numpy(np.float64, copy=False))
        df[name] = (df[cols].to_numpy(np.float64, copy=False) @ w).astype(np.float32)
        return name, fit_ic, {col: float(val) for col, val in zip(cols, w)}
    model, fit_ic = ridge_view_fit(
        fit_df[cols].to_numpy(np.float64, copy=False),
        fit_df["label"].to_numpy(np.float64, copy=False),
        alpha,
    )
    df[name] = apply_ridge_view(df[cols].to_numpy(np.float64, copy=False), model)
    weights = {col: float(val) for col, val in zip(cols, model["coef"])}  # type: ignore[index]
    return name, fit_ic, weights


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fit = load_pred(FIT_PATH, "2019-01-01", "2020-01-01")
    apply = load_pred(APPLY_PATH, "2020-01-01", "2021-01-01")
    base_views = add_views(fit)
    add_views(apply)

    windows = {
        "fit_h1": ("2019-01-01", "2019-07-01"),
        "val_h2": ("2019-07-01", "2020-01-01"),
        "fit_q1q3": ("2019-01-01", "2019-10-01"),
        "val_q4": ("2019-10-01", "2020-01-01"),
        "fit_full2019": ("2019-01-01", "2020-01-01"),
        "audit_2020": ("2020-01-01", "2021-01-01"),
    }

    combo_defs: list[tuple[str, list[str], str, float]] = [
        ("simplex_basic", ["pred", "pred_xcenter", "pred_xsz", "pred_xrank"], "simplex", 0.0),
        ("simplex_robust_z", ["pred_xsz", "pred_xsz_clip2", "pred_xsz_clip3", "pred_xsz_clip4", "pred_xrank"], "simplex", 0.0),
        (
            "simplex_rank_tanh",
            ["pred_xsz", "pred_xrank", "pred_rankgauss", "pred_tanh_z_s1", "pred_tanh_z_s2", "pred_asinh_z"],
            "simplex",
            0.0,
        ),
    ]
    for alpha in (0.01, 0.05, 0.2):
        combo_defs.append(
            (
                f"ridgeview_robust_a{alpha:g}",
                ["pred_xsz", "pred_xrank", "pred_rankgauss", "pred_tanh_z_s1", "pred_tanh_z_s2", "pred_asinh_z"],
                "ridge",
                alpha,
            )
        )

    summary_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    for view in base_views:
        for window, (start, end) in windows.items():
            target_df = apply if window == "audit_2020" else fit
            summary_rows.append({"candidate": view, "kind": "single", "fit_rule": "none", "window": window, **summarize(target_df, view, start, end)})

    fit_specs = {
        "h1_to_h2": ("2019-01-01", "2019-07-01"),
        "q1q3_to_q4": ("2019-01-01", "2019-10-01"),
        "full2019_to_2020": ("2019-01-01", "2020-01-01"),
    }
    for rule, (start, end) in fit_specs.items():
        fit_df = fit.loc[period_mask(fit, start, end)]
        for combo_name, cols, method, alpha in combo_defs:
            col, fit_ic, weights = add_combo(fit, f"{combo_name}_{rule}", cols, fit_df, method, alpha)
            add_combo(apply, f"{combo_name}_{rule}", cols, fit_df, method, alpha)
            for v, w in weights.items():
                weight_rows.append({"candidate": col, "base_view": v, "weight": w, "fit_ic": fit_ic, "fit_rule": rule, "method": method})
            for window, (wstart, wend) in windows.items():
                target_df = apply if window == "audit_2020" else fit
                summary_rows.append(
                    {"candidate": col, "kind": "combo", "fit_rule": rule, "window": window, **summarize(target_df, col, wstart, wend)}
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "view_scan_summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUT_DIR / "view_scan_weights.csv", index=False)

    pivot = summary.pivot_table(index=["candidate", "kind", "fit_rule"], columns="window", values=["ic", "monthly_mean"], aggfunc="first")
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    selected = pivot.reset_index()
    selected["selection_score"] = selected["ic_val_q4"].fillna(-999.0) + 0.25 * selected["ic_val_h2"].fillna(-999.0)
    selected = selected.sort_values(["selection_score", "ic_val_q4", "ic_val_h2"], ascending=False)
    selected.to_csv(OUT_DIR / "selected_by_2019_internal_then_2020_audit.csv", index=False)

    monthly_rows = []
    keep = selected.head(20)["candidate"].tolist()
    for candidate in keep:
        if candidate not in apply.columns:
            continue
        for month, ic in monthly_ic(apply, candidate).items():
            monthly_rows.append({"candidate": candidate, "month": month, "ic": float(ic)})
    pd.DataFrame(monthly_rows).to_csv(OUT_DIR / "top20_2020_monthly_ic.csv", index=False)

    (OUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "fit_path": str(FIT_PATH),
                "apply_path": str(APPLY_PATH),
                "selection": "candidate ranking uses 2019 H2/Q4 only; audit_2020 columns are not used for selection",
                "rows_fit_2019": int(len(fit)),
                "rows_apply_2020": int(len(apply)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(selected.head(25).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
