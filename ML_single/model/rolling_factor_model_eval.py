#!/usr/bin/env python3
"""
Rolling evaluation for the archived factor-model route.

The materialized 1,144-factor panel is not present in this migrated archive,
so this script uses the saved factor-model prediction artifacts as the model
layer to re-train monthly blend weights.  For every test month, labels from
that month and later are excluded from weight fitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize


ROOT = Path("/root/autodl-tmp/quant")
WORK = ROOT / "ML"
OUT = WORK / "rolling_results"
FIG = OUT / "figures"
ART = ROOT / "artifacts"
COMP_ART = ART / "component_predictions"
COMP_ALT = ROOT / "component_predictions"

TRAIN_START = pd.Timestamp("2018-01-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    path: Path
    column: str = "pred"
    mode: str = "raw"


def compute_ic(pred: np.ndarray, label: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 2:
        return float("nan")
    p = pred[mask]
    y = label[mask]
    den = np.sqrt(np.mean(p * p) * np.mean(y * y))
    if den < 1e-18:
        return float("nan")
    return float(np.mean(p * y) / den)


def add_cross_sectional_norms(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("datetime", sort=False)[pred_col]
    out["pred_xsz"] = ((out[pred_col] - g.transform("mean")) / (g.transform("std") + 1e-9)).astype("float32")
    out["pred_xrank"] = (g.rank(pct=True) - 0.5).astype("float32")
    return out


def period_ic(df: pd.DataFrame, pred_col: str, period: str = "M") -> pd.Series:
    tmp = df.dropna(subset=[pred_col, "label"]).copy()
    if period == "Y":
        tmp["_period"] = tmp["datetime"].dt.year.astype(str)
    else:
        tmp["_period"] = tmp["datetime"].dt.to_period("M").astype(str)
    return tmp.groupby("_period", sort=True).apply(lambda x: compute_ic(x[pred_col].to_numpy(), x["label"].to_numpy()))


def summarize_predictions(df: pd.DataFrame, name: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    sub = df[(df["datetime"] >= start) & (df["datetime"] < end)].copy()
    row: dict[str, object] = {
        "model": name,
        "rows": int(len(sub)),
        "label_rows": int(sub["label"].notna().sum()),
        "date_min": str(sub["datetime"].min()),
        "date_max": str(sub["datetime"].max()),
    }
    for col in ["pred", "pred_xsz", "pred_xrank"]:
        if col not in sub.columns:
            continue
        by_m = period_ic(sub, col, "M")
        by_y = period_ic(sub, col, "Y")
        row[f"{col}_ic"] = compute_ic(sub[col].to_numpy(), sub["label"].to_numpy())
        row[f"{col}_monthly_mean"] = float(by_m.mean())
        row[f"{col}_monthly_std"] = float(by_m.std())
        row[f"{col}_monthly_ir"] = float(by_m.mean() / by_m.std()) if by_m.std() > 0 else float("nan")
        for year, val in by_y.items():
            row[f"{col}_ic_{year}"] = float(val)
    return row


def parquet_ok(path: Path) -> tuple[bool, str, int | None]:
    try:
        pf = pq.ParquetFile(path)
        return True, ",".join(pf.schema_arrow.names), int(pf.metadata.num_rows)
    except Exception as exc:  # noqa: BLE001 - audit should capture any parquet failure.
        return False, f"{type(exc).__name__}: {str(exc)[:180]}", None


def audit_assets() -> pd.DataFrame:
    paths = [
        ART / "predictions_best_ic0716.parquet",
        ART / "predictions_core_moe_noDL_ic0617.parquet",
    ]
    paths.extend(sorted(COMP_ART.glob("*.parquet")))
    paths.extend(sorted(COMP_ALT.glob("*.parquet")))
    rows = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        ok, detail, nrows = parquet_ok(path)
        rows.append(
            {
                "path": str(path),
                "file": path.name,
                "bytes": path.stat().st_size if path.exists() else None,
                "readable": ok,
                "rows": nrows,
                "detail": detail,
            }
        )
    return pd.DataFrame(rows)


def load_base() -> pd.DataFrame:
    path = ART / "predictions_best_ic0716.parquet"
    base = pd.read_parquet(path, columns=["symbol", "datetime", "label"])
    base["datetime"] = pd.to_datetime(base["datetime"])
    base = base[(base["datetime"] >= TRAIN_START) & (base["datetime"] < TEST_END)].copy()
    base["_month"] = base["datetime"].dt.to_period("M").astype(str)
    return base.reset_index(drop=True)


def resolve_component(name: str) -> Path | None:
    for folder in (COMP_ART, COMP_ALT):
        path = folder / name
        if not path.exists():
            continue
        ok, _, _ = parquet_ok(path)
        if ok:
            return path
    return None


def candidate_specs() -> list[CandidateSpec]:
    best = ART / "predictions_best_ic0716.parquet"
    core = ART / "predictions_core_moe_noDL_ic0617.parquet"
    specs: list[CandidateSpec] = [
        CandidateSpec("best_raw", best, "pred", "raw"),
        CandidateSpec("best_xsz", best, "pred_xsz", "raw"),
        CandidateSpec("best_xrank", best, "pred_xrank", "raw"),
        CandidateSpec("core_raw", core, "pred", "raw"),
        CandidateSpec("core_xsz", core, "pred_xsz", "raw"),
        CandidateSpec("core_xrank", core, "pred_xrank", "raw"),
    ]
    component_defs = [
        ("anchor_raw", "anchor_core_ic0587.parquet", "pred", "raw"),
        ("anchor_xsz", "anchor_core_ic0587.parquet", "pred_xsz", "raw"),
        ("anchor_xrank", "anchor_core_ic0587.parquet", "pred_xrank", "raw"),
        ("flgb1819_z", "predictions_flgb1819_win_aligned.parquet", "pred", "z"),
        ("flgb2021_z", "predictions_flgb2021_seed_aligned.parquet", "pred", "z"),
        ("group650_z", "predictions_group_lgb_prior_lb12_top650_n250000.parquet", "pred", "z"),
        ("group650_xsz", "predictions_group_lgb_prior_lb12_top650_n250000.parquet", "pred_xsz", "raw"),
        ("group650_xrank", "predictions_group_lgb_prior_lb12_top650_n250000.parquet", "pred_xrank", "raw"),
        ("shared_z", "predictions_shared_final_v2_aligned.parquet", "pred", "z"),
        ("relh_z", "predictions_rel_sym_hour075_aligned.parquet", "pred", "z"),
        ("relsess_z", "predictions_rel_sym_session075_aligned.parquet", "pred", "z"),
        ("rels05c_z", "predictions_rel_sym_s05cap_aligned.parquet", "pred", "z"),
        ("rels05_z", "predictions_rel_sym_s05_aligned.parquet", "pred", "z"),
        ("rollv5_z", "predictions_rolling_v5_lb24_aligned.parquet", "pred", "z"),
    ]
    for cand_name, file_name, col, mode in component_defs:
        path = resolve_component(file_name)
        if path is not None:
            specs.append(CandidateSpec(cand_name, path, col, mode))
    return specs


def component_signal(df: pd.DataFrame, column: str, mode: str) -> pd.Series:
    sig = df[column].astype("float32")
    if mode == "raw":
        return sig
    g = df.groupby("datetime", sort=False)[column]
    if mode == "z":
        return ((df[column] - g.transform("mean")) / (g.transform("std") + 1e-9)).astype("float32")
    if mode == "rank":
        return (g.rank(pct=True) - 0.5).astype("float32")
    raise ValueError(f"bad mode: {mode}")


def add_candidate(base: pd.DataFrame, spec: CandidateSpec) -> tuple[pd.DataFrame, str | None]:
    cols = ["symbol", "datetime", spec.column]
    if spec.column != "label":
        cols.append("label")
    try:
        df = pd.read_parquet(spec.path, columns=list(dict.fromkeys(cols)))
    except Exception as exc:  # noqa: BLE001
        return base, f"{spec.name}: skipped read failure from {spec.path}: {exc}"
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[(df["datetime"] >= TRAIN_START) & (df["datetime"] < TEST_END)].copy()
    df[spec.name] = component_signal(df, spec.column, spec.mode)
    before = len(base)
    base = base.merge(df[["symbol", "datetime", spec.name]], on=["symbol", "datetime"], how="left")
    if len(base) != before:
        raise RuntimeError(f"merge changed row count for {spec.name}: {before} -> {len(base)}")
    base[spec.name] = base[spec.name].astype("float32")
    return base, None


def build_candidate_panel() -> tuple[pd.DataFrame, list[str], list[str]]:
    base = load_base()
    logs = []
    names = []
    for spec in candidate_specs():
        if spec.name in base.columns:
            continue
        base, msg = add_candidate(base, spec)
        if msg:
            logs.append(msg)
            continue
        names.append(spec.name)
        non_null = int(base[spec.name].notna().sum())
        logs.append(f"{spec.name}: loaded from {spec.path.name}, non_null={non_null}")
    return base, names, logs


def fit_ic_weights_from_stats(
    c: np.ndarray,
    g: np.ndarray,
    yy: float,
    lower: np.ndarray,
    upper: np.ndarray,
    prev_w: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    n = len(c)
    if n == 1:
        return np.ones(1, dtype=np.float64), float(c[0] / np.sqrt(max(g[0, 0] * yy, 1e-30)))
    if lower.sum() > 1.0 + 1e-10 or upper.sum() < 1.0 - 1e-10:
        lower = np.zeros(n, dtype=np.float64)
        upper = np.ones(n, dtype=np.float64)

    def project_start(w0: np.ndarray) -> np.ndarray:
        w = np.clip(np.asarray(w0, dtype=np.float64), lower, upper)
        for _ in range(64):
            diff = 1.0 - float(w.sum())
            if abs(diff) < 1e-10:
                break
            if diff > 0:
                room = np.maximum(upper - w, 0.0)
                total = float(room.sum())
                if total <= 1e-12:
                    break
                w += room * min(1.0, diff / total)
            else:
                room = np.maximum(w - lower, 0.0)
                total = float(room.sum())
                if total <= 1e-12:
                    break
                w -= room * min(1.0, -diff / total)
            w = np.clip(w, lower, upper)
        if abs(w.sum() - 1.0) > 1e-6:
            w = lower.copy()
            rem = 1.0 - float(w.sum())
            cap = np.maximum(upper - lower, 0.0)
            if rem > 0 and cap.sum() > 0:
                w += cap * (rem / float(cap.sum()))
        return w

    def ic_value(w: np.ndarray) -> float:
        var = float(w @ g @ w)
        den = np.sqrt(max(var, 1e-30) * max(float(yy), 1e-30))
        return float((w @ c) / den)

    starts = [np.ones(n, dtype=np.float64) / n]
    diag = np.maximum(np.diag(g), 1e-18)
    single_ic = c / np.sqrt(diag * max(float(yy), 1e-18))
    best_j = int(np.nanargmax(single_ic))
    unit = lower.copy()
    rem = 1.0 - float(lower.sum())
    unit[best_j] = min(upper[best_j], lower[best_j] + max(rem, 0.0))
    starts.append(unit)
    pos = np.maximum(single_ic, 0.0)
    if pos.sum() > 1e-12:
        starts.append(pos / pos.sum())
    if prev_w is not None and len(prev_w) == n:
        starts.append(prev_w)
    try:
        ridge = 1e-8 * max(float(np.trace(g)) / max(n, 1), 1e-12)
        raw = np.linalg.solve(g + ridge * np.eye(n), c)
        if np.all(np.isfinite(raw)) and abs(raw.sum()) > 1e-12:
            starts.append(raw / raw.sum())
    except np.linalg.LinAlgError:
        pass

    bounds = [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    best_w = project_start(starts[0])
    best_ic = ic_value(best_w)
    for start in starts:
        w0 = project_start(start)
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
        w = project_start(res.x)
        val = ic_value(w)
        if val > best_ic:
            best_w, best_ic = w, val
    return best_w, float(best_ic)


def monthly_stats(data: pd.DataFrame, names: list[str]) -> tuple[dict[str, dict[str, object]], list[str]]:
    stats: dict[str, dict[str, object]] = {}
    months = sorted(data["_month"].unique())
    for month in months:
        cur = data["_month"].to_numpy() == month
        y = data.loc[cur, "label"].to_numpy(np.float64)
        x = data.loc[cur, names].to_numpy(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        label_mask = np.isfinite(y)
        xl = x[label_mask]
        yl = y[label_mask]
        stats[month] = {
            "n": int(len(yl)),
            "c": xl.T @ yl,
            "g": xl.T @ xl,
            "yy": float(yl @ yl),
            "std": np.nanstd(x, axis=0),
        }
    return stats, months


def sum_stats(stats: dict[str, dict[str, object]], months: Iterable[str], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    c = np.zeros(len(indices), dtype=np.float64)
    g = np.zeros((len(indices), len(indices)), dtype=np.float64)
    yy = 0.0
    n = 0
    for month in months:
        s = stats[month]
        c += np.asarray(s["c"])[indices]
        g += np.asarray(s["g"])[np.ix_(indices, indices)]
        yy += float(s["yy"])
        n += int(s["n"])
    return c, g, yy, n


def rolling_blend(
    data: pd.DataFrame,
    names: list[str],
    model_name: str,
    candidates: list[str],
    *,
    max_weight: float = 0.90,
    min_weight: float = 0.0,
    lookback_months: int = 0,
    min_train_months: int = 24,
    anchor_name: str | None = None,
    anchor_min: float = 0.0,
    stats_cache: tuple[dict[str, dict[str, object]], list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    name_to_idx = {name: i for i, name in enumerate(names)}
    selected = [name for name in candidates if name in name_to_idx]
    if not selected:
        raise ValueError(f"{model_name}: no candidates are available")
    sel_idx = np.array([name_to_idx[x] for x in selected], dtype=int)
    stats, months = stats_cache if stats_cache is not None else monthly_stats(data, names)
    test_months = [m for m in months if pd.Period(m).to_timestamp() >= TEST_START and pd.Period(m).to_timestamp() < TEST_END]
    out = data[["symbol", "datetime", "label", "_month"]].copy()
    out["pred"] = np.nan
    records = []
    prev_full_w: np.ndarray | None = None
    for month in test_months:
        i = months.index(month)
        hist_months = months[:i]
        if lookback_months > 0:
            hist_months = hist_months[-lookback_months:]
        if len(hist_months) < min_train_months:
            active_names = selected
            w = np.ones(len(active_names), dtype=np.float64) / len(active_names)
            train_ic = float("nan")
        else:
            cur_std = np.asarray(stats[month]["std"])[sel_idx]
            hist_std = np.zeros(len(sel_idx), dtype=np.float64)
            for hist_m in hist_months:
                hist_std += np.asarray(stats[hist_m]["std"])[sel_idx]
            active_mask = (cur_std > 1e-10) & (hist_std > 1e-10)
            if not np.any(active_mask):
                active_mask[:] = True
            active_names = [name for name, keep in zip(selected, active_mask) if keep]
            active_global_idx = sel_idx[active_mask]
            c, g, yy, n = sum_stats(stats, hist_months, active_global_idx)
            lower = np.full(len(active_names), min_weight, dtype=np.float64)
            upper = np.full(len(active_names), max(max_weight, 1.0 / len(active_names)), dtype=np.float64)
            if anchor_name and anchor_name in active_names and anchor_min > 0:
                j = active_names.index(anchor_name)
                lower[j] = max(lower[j], anchor_min)
                upper[j] = max(upper[j], lower[j])
            prev_active = None
            if prev_full_w is not None:
                prev_active = np.array([prev_full_w[selected.index(x)] for x in active_names], dtype=np.float64)
            w, train_ic = fit_ic_weights_from_stats(c, g, yy, lower, upper, prev_active)
        full_w = np.zeros(len(selected), dtype=np.float64)
        for local, name in enumerate(active_names):
            full_w[selected.index(name)] = w[local]
        prev_full_w = full_w

        cur_mask = out["_month"].to_numpy() == month
        x_cur = data.loc[cur_mask, active_names].to_numpy(np.float32)
        x_cur = np.nan_to_num(x_cur, nan=0.0, posinf=0.0, neginf=0.0)
        pred = x_cur @ w.astype(np.float32)
        out.loc[cur_mask, "pred"] = pred
        month_ic = compute_ic(pred, out.loc[cur_mask, "label"].to_numpy())
        rec: dict[str, object] = {
            "model": model_name,
            "month": month,
            "train_months": len(hist_months),
            "active_components": len(active_names),
            "train_ic": train_ic,
            "month_ic": month_ic,
            "rows": int(cur_mask.sum()),
            "label_rows": int(out.loc[cur_mask, "label"].notna().sum()),
        }
        for name, val in zip(selected, full_w):
            rec[f"w_{name}"] = float(val)
        records.append(rec)
    pred = out.drop(columns=["_month"])
    pred = pred[pred["datetime"].ge(TEST_START) & pred["datetime"].lt(TEST_END)].copy()
    pred = add_cross_sectional_norms(pred, "pred")
    return pred, pd.DataFrame(records)


def static_prediction(data: pd.DataFrame, name: str, candidate: str) -> pd.DataFrame:
    pred = data[["symbol", "datetime", "label", candidate]].copy()
    pred = pred[(pred["datetime"] >= TEST_START) & (pred["datetime"] < TEST_END)].copy()
    pred = pred.rename(columns={candidate: "pred"})
    pred = add_cross_sectional_norms(pred, "pred")
    return pred


def plot_monthly_ic(monthly: pd.DataFrame, best_model: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pivot = monthly.pivot(index="month", columns="model", values="month_ic")
    ax = pivot.plot(figsize=(14, 5), marker="o", linewidth=1.8)
    ax.axhline(0.06, color="firebrick", linestyle="--", linewidth=1.0, label="IC 0.06")
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title("Monthly rolling test IC")
    ax.set_xlabel("Test month")
    ax.set_ylabel("IC")
    ax.legend(loc="best", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "monthly_ic.png", dpi=140)
    plt.close()

    one = monthly[monthly["model"] == best_model].copy()
    colors = ["#2f6f8f" if x >= 0 else "#a23b3b" for x in one["month_ic"]]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(one["month"], one["month_ic"], color=colors)
    ax.axhline(0.06, color="firebrick", linestyle="--", linewidth=1.0)
    ax.axhline(float(one["month_ic"].mean()), color="darkgreen", linestyle=":", linewidth=1.2)
    ax.set_title(f"{best_model} monthly IC")
    ax.set_xlabel("Test month")
    ax.set_ylabel("IC")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / f"{best_model}_monthly_ic.png", dpi=140)
    plt.close()


def plot_weights(weights: pd.DataFrame, model: str) -> None:
    rows = weights[weights["model"] == model].copy()
    weight_cols = [c for c in rows.columns if c.startswith("w_")]
    if rows.empty or not weight_cols:
        return
    mat = rows[weight_cols].to_numpy(dtype=float)
    labels = [c[2:] for c in weight_cols]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows["month"], fontsize=8)
    ax.set_title(f"{model} rolling weights")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    plt.savefig(FIG / f"{model}_weights.png", dpi=140)
    plt.close()


def write_report(
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    weights: pd.DataFrame,
    asset_audit: pd.DataFrame,
    load_logs: list[str],
    best_model: str,
) -> None:
    weights.to_csv(OUT / "rolling_weights.csv", index=False)

    def table_block(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False, floatfmt=".6f")
        except ImportError:
            return "```csv\n" + df.to_csv(index=False) + "```"

    factor_count = sum(1 for _ in open(ART / "selected_factors.txt", "r", encoding="utf-8"))
    bad_assets = asset_audit[~asset_audit["readable"]]
    report = []
    report.append("# Rolling factor-model evaluation\n")
    report.append(f"- Factor catalog retained factors: {factor_count}")
    report.append(f"- Train history starts: {TRAIN_START.date()}")
    report.append(f"- Rolling test window requested/available: {TEST_START.date()} to {TEST_END.date()} exclusive")
    report.append("- Protocol: for each test month, fit blend weights with labels strictly before that month.")
    report.append("- Note: the 50GB+ materialized factor panel is not present; this run uses saved factor-model prediction artifacts.")
    report.append("")
    report.append("## Summary\n")
    report.append(table_block(summary))
    report.append("")
    report.append(f"Best selected model by `pred_ic`: `{best_model}`")
    report.append("")
    report.append("## Monthly IC\n")
    report.append(table_block(monthly))
    report.append("")
    report.append("## Asset Audit\n")
    report.append(f"- Readable parquet files: {int(asset_audit['readable'].sum())}/{len(asset_audit)}")
    if not bad_assets.empty:
        report.append("- Unreadable component files were skipped:")
        for _, row in bad_assets.iterrows():
            report.append(f"  - `{row['path']}`: {row['detail']}")
    report.append("")
    report.append("## Candidate Load Log\n")
    for line in load_logs:
        report.append(f"- {line}")
    report.append("")
    report.append("## Figures\n")
    report.append("- `figures/monthly_ic.png`")
    report.append(f"- `figures/{best_model}_monthly_ic.png`")
    if (FIG / f"{best_model}_weights.png").exists():
        report.append(f"- `figures/{best_model}_weights.png`")
    report.append("")
    OUT.joinpath("rolling_factor_model_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    asset_audit = audit_assets()
    asset_audit.to_csv(OUT / "asset_audit.csv", index=False)

    data, names, load_logs = build_candidate_panel()
    (OUT / "candidate_load_log.txt").write_text("\n".join(load_logs) + "\n", encoding="utf-8")

    variants = {
        "static_core": static_prediction(data, "static_core", "core_raw"),
        "static_best": static_prediction(data, "static_best", "best_raw"),
    }
    rolling_defs = {
        "rolling_clean_no_final": {
            "candidates": [
                "core_raw",
                "core_xsz",
                "core_xrank",
                "anchor_raw",
                "anchor_xsz",
                "anchor_xrank",
                "flgb1819_z",
                "flgb2021_z",
                "group650_z",
                "group650_xsz",
                "group650_xrank",
            ],
            "max_weight": 0.70,
            "min_weight": 0.0,
        },
        "rolling_best_core": {
            "candidates": ["best_raw", "core_raw", "best_xsz", "core_xsz", "best_xrank", "core_xrank"],
            "max_weight": 0.95,
            "min_weight": 0.0,
        },
        "rolling_all_nonneg": {
            "candidates": names,
            "max_weight": 0.90,
            "min_weight": 0.0,
        },
        "rolling_all_signed": {
            "candidates": names,
            "max_weight": 1.10,
            "min_weight": -0.15,
        },
    }
    stats_cache = monthly_stats(data, names)
    weight_frames = []
    for model_name, cfg in rolling_defs.items():
        pred, weights = rolling_blend(data, names, model_name, stats_cache=stats_cache, **cfg)
        variants[model_name] = pred
        weight_frames.append(weights)

    summary_rows = []
    monthly_rows = []
    for model_name, pred in variants.items():
        path = OUT / f"{model_name}_predictions.parquet"
        pred.to_parquet(path, index=False)
        summary_rows.append(summarize_predictions(pred, model_name, TEST_START, TEST_END))
        by_m = period_ic(pred, "pred", "M").rename("month_ic").reset_index().rename(columns={"_period": "month"})
        by_m.insert(0, "model", model_name)
        monthly_rows.append(by_m)

    summary = pd.DataFrame(summary_rows).sort_values("pred_ic", ascending=False)
    monthly = pd.concat(monthly_rows, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    summary.to_csv(OUT / "summary.csv", index=False)
    monthly.to_csv(OUT / "monthly_ic.csv", index=False)

    best_model = str(summary.iloc[0]["model"])
    plot_monthly_ic(monthly, best_model)
    plot_weights(weights, best_model)
    write_report(summary, monthly, weights, asset_audit, load_logs, best_model)

    metadata = {
        "train_start": str(TRAIN_START.date()),
        "test_start": str(TEST_START.date()),
        "test_end_exclusive": str(TEST_END.date()),
        "candidates": names,
        "best_model": best_model,
        "best_pred_ic": float(summary.iloc[0]["pred_ic"]),
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[["model", "rows", "label_rows", "pred_ic", "pred_monthly_mean", "pred_monthly_ir"]].to_string(index=False))
    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
