#!/usr/bin/env python3
"""Minimal diverse ML ensemble over strict rolling component predictions.

Selection is done on 2019 only:
  - 2019Q1-Q3: fit IC weights and compute component correlations.
  - 2019Q4: choose the smallest/highest validation subset.
  - 2020: final rolling month-by-month evaluation only.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic
from strict_optimization_ablation import BASE_STRICT_DIR, OUT_DIR as STRICT_OUT_DIR, PRED_START, TEST_END, TEST_START, summarize


ROOT = Path("/root/autodl-tmp/quant/ML")
EFFECTIVE_DIR = ROOT / "effective_rolling_results"
OUT_DIR = STRICT_OUT_DIR / "minimal_diverse_ensemble"
VAL_START = pd.Timestamp("2019-10-01")
MAX_K = 9
COMPACT_RAW_IC_2020 = 0.05206430357711721
COMPACT_XSZ_IC_2020 = 0.05024390718000563


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    path: Path
    col: str = "pred"


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def read_component(spec: ComponentSpec) -> pd.DataFrame:
    df = pd.read_parquet(spec.path, columns=["symbol", "datetime", "label", spec.col])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[(df["datetime"] >= PRED_START) & (df["datetime"] < TEST_END)]
    df = df.sort_values(["datetime", "symbol"], kind="mergesort").reset_index(drop=True)
    return df


def collect_specs() -> list[ComponentSpec]:
    specs: list[ComponentSpec] = []
    strict_files = [
        ("base_raw", BASE_STRICT_DIR / "strict_lgb_raw_top300_n500000.parquet"),
        ("base_xsz", BASE_STRICT_DIR / "strict_lgb_xsz_top300_n500000.parquet"),
        ("base_xrank", BASE_STRICT_DIR / "strict_lgb_xrank_top300_n500000.parquet"),
    ]
    for prefix, path in strict_files:
        if path.exists():
            specs.append(ComponentSpec(f"{prefix}_raw", path, "pred"))
            specs.append(ComponentSpec(f"{prefix}_xsz", path, "pred_xsz"))
            specs.append(ComponentSpec(f"{prefix}_xrank", path, "pred_xrank"))

    summary_path = STRICT_OUT_DIR / "base_ablation_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        keep = summary[
            (summary["pred_ic_2019"].fillna(-1) >= 0.040)
            | (summary["pred_xsz_ic_2019"].fillna(-1) >= 0.050)
        ]["model"].astype(str)
        for model in keep:
            path = STRICT_OUT_DIR / f"{model}.parquet"
            if path.exists():
                specs.append(ComponentSpec(f"{model}_raw", path, "pred"))

    for name in [
        "mlp_overlap333_xsz_hl12_n1200k",
        "mlp_overlap333_xsz_hl12_n800k",
        "mlp_overlap333_xsz_hl12_n400k",
        "mlp_ridge617_xsz_hl12_n400k",
    ]:
        path = EFFECTIVE_DIR / name / f"{name}.parquet"
        if path.exists():
            specs.append(ComponentSpec(f"{name}_raw", path, "pred"))
            specs.append(ComponentSpec(f"{name}_xsz", path, "pred_xsz"))

    dedup: dict[str, ComponentSpec] = {}
    for spec in specs:
        dedup[spec.name] = spec
    return list(dedup.values())


def load_matrix(specs: list[ComponentSpec]) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    first = read_component(specs[0])
    n = len(first)
    ref_symbol = first["symbol"].astype(str).to_numpy()
    ref_dt = first["datetime"].astype("int64").to_numpy()
    base = first[["symbol", "datetime", "label"]].copy()
    cols: list[np.ndarray] = []
    names: list[str] = []
    for i, spec in enumerate(specs):
        df = first if i == 0 else read_component(spec)
        ok = (
            len(df) == n
            and np.array_equal(df["datetime"].astype("int64").to_numpy(), ref_dt)
            and np.array_equal(df["symbol"].astype(str).to_numpy(), ref_symbol)
        )
        if not ok:
            print(f"[align-skip] {spec.name}", flush=True)
            continue
        cols.append(scrub(df[spec.col].to_numpy(np.float32)))
        names.append(spec.name)
        print(f"[component] {len(names):02d} {spec.name}", flush=True)
        if i != 0:
            del df
    x = np.column_stack(cols).astype(np.float32, copy=False)
    return base, names, x


def stats(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    xm = scrub(x[mask]).astype(np.float64, copy=False)
    ym = y[mask].astype(np.float64, copy=False)
    good = np.isfinite(ym)
    xm = xm[good]
    ym = ym[good]
    return xm.T @ xm, xm.T @ ym, float(ym @ ym)


def fit_subset_from_stats(
    gram: np.ndarray,
    cov: np.ndarray,
    yty: float,
    cols: list[int],
    signed: bool = True,
) -> tuple[np.ndarray, float]:
    idx = np.array(cols, dtype=np.int32)
    lower = np.full(len(idx), -0.14 if signed else 0.0, dtype=np.float64)
    upper = np.full(len(idx), 0.85 if signed else 0.85, dtype=np.float64)
    return fit_ic_weights_from_stats(cov[idx], gram[np.ix_(idx, idx)], yty, lower, upper)


def ic_from_stats(gram: np.ndarray, cov: np.ndarray, yty: float, cols: list[int], w: np.ndarray) -> float:
    idx = np.array(cols, dtype=np.int32)
    num = float(w @ cov[idx])
    den = float((w @ gram[np.ix_(idx, idx)] @ w) * yty)
    if den <= 1e-18:
        return float("nan")
    return num / np.sqrt(den)


def corr_matrix(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xm = scrub(x[mask]).astype(np.float64, copy=False)
    xm -= xm.mean(axis=0, keepdims=True)
    sd = np.maximum(xm.std(axis=0, keepdims=True), 1e-12)
    xm /= sd
    return (xm.T @ xm) / max(len(xm), 1)


def standalone_table(names: list[str], train_cov: np.ndarray, train_gram: np.ndarray, train_yty: float, val_cov: np.ndarray, val_gram: np.ndarray, val_yty: float) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "component_index": i,
                "component": name,
                "train_ic_2019q1q3": train_cov[i] / np.sqrt(max(train_gram[i, i] * train_yty, 1e-18)),
                "val_ic_2019q4": val_cov[i] / np.sqrt(max(val_gram[i, i] * val_yty, 1e-18)),
            }
        )
    return pd.DataFrame(rows).sort_values("val_ic_2019q4", ascending=False).reset_index(drop=True)


def greedy_candidates(
    names: list[str],
    train_gram: np.ndarray,
    train_cov: np.ndarray,
    train_yty: float,
    val_gram: np.ndarray,
    val_cov: np.ndarray,
    val_yty: float,
    corr: np.ndarray,
    seed_components: list[int],
) -> pd.DataFrame:
    rows = []
    pool = list(range(len(names)))
    for corr_penalty in [0.0, 0.003, 0.006, 0.010, 0.016, 0.024]:
        selected: list[int] = []
        for seed in seed_components:
            if seed in pool and seed not in selected:
                selected.append(seed)
        while len(selected) < MAX_K:
            best = None
            for cand in pool:
                if cand in selected:
                    continue
                trial = selected + [cand]
                w, train_ic = fit_subset_from_stats(train_gram, train_cov, train_yty, trial, signed=True)
                val_ic = ic_from_stats(val_gram, val_cov, val_yty, trial, w)
                if len(trial) > 1:
                    subcorr = np.abs(corr[np.ix_(trial, trial)])
                    avg_corr = float((subcorr.sum() - len(trial)) / (len(trial) * (len(trial) - 1)))
                    max_corr = float((subcorr - np.eye(len(trial))).max())
                else:
                    avg_corr = 0.0
                    max_corr = 0.0
                score = val_ic - corr_penalty * avg_corr
                key = (score, val_ic, -avg_corr, -max_corr)
                if best is None or key > best[0]:
                    best = (key, cand, w, train_ic, val_ic, avg_corr, max_corr)
            if best is None:
                break
            selected.append(int(best[1]))
            w = best[2]
            rows.append(
                {
                    "corr_penalty": corr_penalty,
                    "k": len(selected),
                    "train_ic_2019q1q3": float(best[3]),
                    "val_static_ic_2019q4": float(best[4]),
                    "avg_abs_corr_2019q1q3": float(best[5]),
                    "max_abs_corr_2019q1q3": float(best[6]),
                    "component_indices": json.dumps(selected),
                    "components": "|".join(names[i] for i in selected),
                    "weights": json.dumps([float(x) for x in w]),
                }
            )
    return pd.DataFrame(rows)


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS"))


def rolling_predict(base: pd.DataFrame, x: np.ndarray, cols: list[int], alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dt = base["datetime"]
    y = base["label"].to_numpy(np.float64)
    rows = []
    weights = []
    for ms in month_starts(TEST_START, TEST_END):
        train_mask = ((dt >= PRED_START) & (dt < ms) & base["label"].notna()).to_numpy()
        test_mask = ((dt >= ms) & (dt < ms + pd.DateOffset(months=1))).to_numpy()
        gram, cov, yty = stats(x[:, cols], y, train_mask)
        w, train_ic = fit_subset_from_stats(gram, cov, yty, list(range(len(cols))), signed=True)
        pred = scrub(x[test_mask][:, cols]) @ w.astype(np.float32)
        part = base.loc[test_mask, ["symbol", "datetime", "label"]].copy()
        part["pred"] = pred.astype(np.float32) * alpha
        rows.append(part)
        row = {
            "month": f"{ms:%Y-%m}",
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_ic": float(train_ic),
            "month_ic": compute_ic(part["pred"], part["label"]),
        }
        for local_i, component_i in enumerate(cols):
            row[f"w_{component_i}"] = float(w[local_i] * alpha)
        weights.append(row)
    out = pd.concat(rows, ignore_index=True)
    out = add_cross_sectional_norms(out, "pred")
    return out, pd.DataFrame(weights)


def evaluate_val_rolling(base: pd.DataFrame, x: np.ndarray, cols: list[int]) -> tuple[float, pd.DataFrame]:
    dt = base["datetime"]
    y = base["label"].to_numpy(np.float64)
    rows = []
    for ms in month_starts(VAL_START, TEST_START):
        train_mask = ((dt >= PRED_START) & (dt < ms) & base["label"].notna()).to_numpy()
        val_mask = ((dt >= ms) & (dt < ms + pd.DateOffset(months=1))).to_numpy()
        gram, cov, yty = stats(x[:, cols], y, train_mask)
        w, _ = fit_subset_from_stats(gram, cov, yty, list(range(len(cols))), signed=True)
        pred = scrub(x[val_mask][:, cols]) @ w.astype(np.float32)
        part = base.loc[val_mask, ["symbol", "datetime", "label"]].copy()
        part["pred"] = pred.astype(np.float32)
        rows.append(part)
    pred_df = pd.concat(rows, ignore_index=True)
    return compute_ic(pred_df["pred"], pred_df["label"]), pred_df


def plot_outputs(monthly: pd.Series, comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(kind="bar", ax=ax, color="#2364AA")
    ax.axhline(COMPACT_RAW_IC_2020, color="#D1495B", linestyle="--", linewidth=1.2, label="compact MOE raw IC")
    ax.set_title("Minimal Diverse Ensemble 2020 Monthly Raw IC")
    ax.set_xlabel("Month")
    ax.set_ylabel("IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "monthly_ic.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    comparison.set_index("model")[["pred_ic_2020", "pred_xsz_ic_2020"]].plot(kind="bar", ax=ax, color=["#2364AA", "#73BFB8"])
    ax.axhline(COMPACT_RAW_IC_2020, color="#D1495B", linestyle="--", linewidth=1.0)
    ax.set_title("ML Model Comparison, 2020")
    ax.set_xlabel("")
    ax.set_ylabel("IC")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_2020.png", dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.6f}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = collect_specs()
    base, names, x = load_matrix(specs)
    y = base["label"].to_numpy(np.float64)
    dt = base["datetime"]

    train_mask = ((dt >= PRED_START) & (dt < VAL_START) & base["label"].notna()).to_numpy()
    val_mask = ((dt >= VAL_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    train_gram, train_cov, train_yty = stats(x, y, train_mask)
    val_gram, val_cov, val_yty = stats(x, y, val_mask)
    corr = corr_matrix(x, train_mask)

    standalone = standalone_table(names, train_cov, train_gram, train_yty, val_cov, val_gram, val_yty)
    standalone.to_csv(OUT_DIR / "standalone_2019_selection_stats.csv", index=False)
    pd.DataFrame(corr, index=names, columns=names).to_csv(OUT_DIR / "component_corr_2019q1q3.csv")

    seed_components = []
    for preferred in ["mlp_overlap333_xsz_hl12_n1200k_raw", "base_raw_raw"]:
        if preferred in names:
            seed_components.append(names.index(preferred))
    candidates = greedy_candidates(names, train_gram, train_cov, train_yty, val_gram, val_cov, val_yty, corr, seed_components)
    candidates.to_csv(OUT_DIR / "selection_grid.csv", index=False)

    # Select by 2019Q4 static validation, requiring fewer than 10 components; tie-break by diversity and size.
    grid = candidates[candidates["k"] <= MAX_K].copy()
    grid = grid.sort_values(
        ["val_static_ic_2019q4", "avg_abs_corr_2019q1q3", "k"],
        ascending=[False, True, True],
    )
    selected = grid.iloc[0].to_dict()
    cols = [int(x) for x in json.loads(selected["component_indices"])]
    val_roll_ic, val_roll_df = evaluate_val_rolling(base, x, cols)
    selected["val_rolling_ic_2019q4"] = float(val_roll_ic)
    selected["selected_components"] = [names[i] for i in cols]

    pred, weights = rolling_predict(base, x, cols, alpha=1.0)
    pred.to_parquet(OUT_DIR / "minimal_diverse_ensemble.parquet", index=False)
    weights.to_csv(OUT_DIR / "rolling_weights.csv", index=False)
    monthly = period_ic(pred, "pred", "M")
    monthly.to_csv(OUT_DIR / "monthly_ic.csv")

    summary = summarize(pred, "minimal_diverse_ensemble")
    summary.update(
        {
            "k": len(cols),
            "compact_raw_ic_2020": COMPACT_RAW_IC_2020,
            "compact_xsz_ic_2020": COMPACT_XSZ_IC_2020,
            "val_static_ic_2019q4": float(selected["val_static_ic_2019q4"]),
            "val_rolling_ic_2019q4": float(selected["val_rolling_ic_2019q4"]),
            "avg_abs_corr_2019q1q3": float(selected["avg_abs_corr_2019q1q3"]),
            "max_abs_corr_2019q1q3": float(selected["max_abs_corr_2019q1q3"]),
            "components": "|".join(names[i] for i in cols),
        }
    )
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    component_rows = []
    selected_weights = json.loads(selected["weights"])
    for rank, (idx, weight) in enumerate(zip(cols, selected_weights), start=1):
        component_rows.append(
            {
                "rank": rank,
                "component_index": idx,
                "component": names[idx],
                "selection_weight_2019q1q3": float(weight),
            }
        )
    pd.DataFrame(component_rows).to_csv(OUT_DIR / "selected_components.csv", index=False)

    single_path = EFFECTIVE_DIR / "mlp_overlap333_xsz_hl12_n1200k" / "summary.csv"
    single = pd.read_csv(single_path).iloc[0].to_dict() if single_path.exists() else {}
    comparison = pd.DataFrame(
        [
            {
                "model": "Best single MLP",
                "pred_ic_2020": single.get("pred_ic_2020", np.nan),
                "pred_xsz_ic_2020": single.get("pred_xsz_ic_2020", np.nan),
                "components": 1,
            },
            {
                "model": "Compact MOE",
                "pred_ic_2020": COMPACT_RAW_IC_2020,
                "pred_xsz_ic_2020": COMPACT_XSZ_IC_2020,
                "components": 16,
            },
            {
                "model": "Minimal diverse ensemble",
                "pred_ic_2020": summary["pred_ic_2020"],
                "pred_xsz_ic_2020": summary["pred_xsz_ic_2020"],
                "components": len(cols),
            },
        ]
    )
    comparison.to_csv(OUT_DIR / "comparison_2020.csv", index=False)
    plot_outputs(monthly, comparison)

    report = OUT_DIR / "README.md"
    report.write_text(
        "# Minimal Diverse ML Ensemble\n\n"
        "Selection window: 2019Q1-Q3 for fitting/correlation, 2019Q4 for subset validation; 2020 is final rolling OOS only.\n\n"
        f"Selected {len(cols)} components:\n\n"
        + "\n".join(f"- {names[i]}" for i in cols)
        + "\n\n"
        + markdown_table(
            summary_df[
                [
                    "model",
                    "k",
                    "pred_ic_2020",
                    "pred_xsz_ic_2020",
                    "pred_monthly_mean_2020",
                    "pred_monthly_ir_2020",
                    "val_static_ic_2019q4",
                    "val_rolling_ic_2019q4",
                    "avg_abs_corr_2019q1q3",
                ]
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary_df[["model", "k", "pred_ic_2020", "pred_xsz_ic_2020", "val_static_ic_2019q4", "val_rolling_ic_2019q4"]].to_string(index=False), flush=True)
    print("[minimal-ensemble] components:", ", ".join(names[i] for i in cols), flush=True)


if __name__ == "__main__":
    main()
