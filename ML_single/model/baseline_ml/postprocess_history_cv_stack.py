#!/usr/bin/env python3
"""Postprocess history-CV stack outputs and export robust-core diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import expanded_history_gate_clean as eh
import expanded_topk_view_stack_clean as topk
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, period_ic


ROOT = Path("/root/autodl-tmp/quant/ML")
OUT_DIR = ROOT / "strict_opt_results" / "expanded_topk_history_cv_stack"
BASE_PRED_DIR = OUT_DIR / "base_predictions"
ROBUST_MODEL = "history_cv_stack__rolling__pred__top8__xsz__std0__u0.1"
ROBUST_PATH = OUT_DIR / "history_cv_stack_robust_core_pred_only_val_selected.parquet"


def robust_core_configs() -> list[str]:
    configs = pd.read_csv(OUT_DIR / "history_cv_stack_base_configs.csv")
    return configs["config"].astype(str).head(8).tolist()


def read_aligned_feature(cfg: str, view: str, suffix: str, ref: pd.DataFrame | None) -> tuple[pd.DataFrame, np.ndarray]:
    path = BASE_PRED_DIR / f"{cfg.replace('/', '_')}__{suffix}.parquet"
    df = pd.read_parquet(path, columns=["symbol", "datetime", "label", view])
    df["datetime"] = pd.to_datetime(df["datetime"])
    base = df[["symbol", "datetime", "label"]].copy().reset_index(drop=True)
    if ref is not None:
        aligned = (
            len(base) == len(ref)
            and np.array_equal(base["datetime"].to_numpy(), ref["datetime"].to_numpy())
            and np.array_equal(base["symbol"].astype(str).to_numpy(), ref["symbol"].astype(str).to_numpy())
        )
        if not aligned:
            raise ValueError(f"alignment mismatch for {cfg} {suffix}")
    return base, df[view].to_numpy(np.float32, copy=False)


def reconstruct_robust_final(out_path: Path) -> pd.DataFrame:
    configs = robust_core_configs()
    hist_base = None
    test_base = None
    hist_cols = []
    test_cols = []
    feature_names = []
    for cfg in configs:
        hb, hx = read_aligned_feature(cfg, "pred", "hist_roll", hist_base)
        tb, tx = read_aligned_feature(cfg, "pred", "test_roll", test_base)
        if hist_base is None:
            hist_base = eh.add_xsz_label(hb)
            test_base = tb
        if test_base is None:
            test_base = tb
        hist_cols.append(hx)
        test_cols.append(tx)
        feature_names.append(f"rolling::{cfg}::pred")

    assert hist_base is not None and test_base is not None
    xh = np.column_stack(hist_cols).astype(np.float32)
    xt = np.column_stack(test_cols).astype(np.float32)
    y = hist_base["label_xsz_fit"].to_numpy(np.float64)
    spec = topk.StackSpec(
        name="rolling__pred__top8__xsz__std0__u0.1",
        modes=("rolling",),
        views=("pred",),
        top_n=8,
        target="xsz",
        standardize=False,
        upper=0.10,
    )
    cols = list(range(len(feature_names)))
    weights, fit_ic, mean, scale = topk.fit_stack(xh, y, cols, spec)
    pred = topk.apply_stack(xt, cols, weights, mean, scale, spec.standardize)
    out = test_base[["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype(np.float32)
    out = add_cross_sectional_norms(out, "pred")
    out.to_parquet(out_path, index=False)
    pd.DataFrame(
        {
            "feature": feature_names,
            "weight": [float(w) for w in weights],
            "final_fit_ic_2019apr_dec": fit_ic,
        }
    ).to_csv(OUT_DIR / "history_cv_stack_robust_core_final_weights.csv", index=False)
    return out


def summarize(pred: pd.DataFrame, model: str) -> dict[str, object]:
    monthly = period_ic(pred, "pred", "M")
    return {
        "model": model,
        "rows": int(len(pred)),
        "label_rows": int(pred["label"].notna().sum()),
        "pred_ic_2020": compute_ic(pred["pred"], pred["label"]),
        "pred_xsz_ic_2020": compute_ic(pred["pred_xsz"], pred["label"]),
        "pred_xrank_ic_2020": compute_ic(pred["pred_xrank"], pred["label"]),
        "monthly_mean": float(monthly.mean()),
        "monthly_ir": float(monthly.mean() / monthly.std()) if monthly.std() > 0 else float("nan"),
    }


def write_monthly(pred: pd.DataFrame, name: str) -> pd.Series:
    monthly = period_ic(pred, "pred", "M")
    monthly.to_csv(OUT_DIR / f"{name}_monthly_ic.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(kind="bar", ax=ax, color="#2F6B8F")
    ax.axhline(0.06, color="#C44536", linestyle="--", linewidth=1.2, label="IC 0.06")
    ax.set_title(name.replace("_", " "))
    ax.set_xlabel("month")
    ax.set_ylabel("raw IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_monthly_ic.png", dpi=160)
    plt.close(fig)
    return monthly


def main() -> None:
    summary = pd.read_csv(OUT_DIR / "history_cv_stack_summary.csv")
    robust = reconstruct_robust_final(ROBUST_PATH)

    selected_path = OUT_DIR / "history_cv_stack_selected_by_val_raw.parquet"
    diagnostic_path = OUT_DIR / "history_cv_stack_diagnostic_best_2020_raw.parquet"
    selected = pd.read_parquet(selected_path)
    diagnostic = pd.read_parquet(diagnostic_path)

    rows = [
        summarize(selected, "global_val_raw_selected_top10_failure"),
        summarize(robust, "robust_core_pred_only_val_selected"),
        summarize(diagnostic, "diagnostic_best_2020_raw"),
    ]
    out = pd.DataFrame(rows)

    robust_row = summary[summary["model"] == ROBUST_MODEL].iloc[0].to_dict()
    diag_row = summary[summary["diagnostic_best_2020_raw"]].iloc[0].to_dict()
    global_row = summary[summary["selected_by_history_val_raw"]].iloc[0].to_dict()
    meta_rows = []
    for tag, row in [
        ("global_val_raw_selected_top10_failure", global_row),
        ("robust_core_pred_only_val_selected", robust_row),
        ("diagnostic_best_2020_raw", diag_row),
    ]:
        meta_rows.append(
            {
                "tag": tag,
                "source_model": row["model"],
                "stack_val_raw_ic_2019q4": row["stack_val_raw_ic_2019q4"],
                "stack_val_xsz_ic_2019q4": row["stack_val_xsz_ic_2019q4"],
                "views": row["views"],
                "top_n": row["top_n"],
                "target": row["target"],
                "standardize": row["standardize"],
                "upper": row["upper"],
            }
        )
    meta = pd.DataFrame(meta_rows)
    full = out.merge(meta, left_on="model", right_on="tag", how="left").drop(columns=["tag"])
    full.to_csv(OUT_DIR / "history_cv_stack_key_results.csv", index=False)

    monthly_parts = []
    for name, pred in [
        ("global_val_raw_selected_top10_failure", selected),
        ("robust_core_pred_only_val_selected", robust),
        ("diagnostic_best_2020_raw", diagnostic),
    ]:
        monthly = write_monthly(pred, name)
        monthly_parts.append(monthly.rename(name))

    monthly_df = pd.concat(monthly_parts, axis=1)
    monthly_df.to_csv(OUT_DIR / "history_cv_stack_key_monthly_ic.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    full.set_index("model")[["pred_ic_2020", "pred_xsz_ic_2020"]].plot(kind="bar", ax=ax, color=["#2F6B8F", "#76B7B2"])
    ax.axhline(0.06, color="#C44536", linestyle="--", linewidth=1.2, label="IC 0.06")
    ax.set_xlabel("")
    ax.set_ylabel("IC")
    ax.set_title("History-CV Stack Key Results")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "history_cv_stack_key_results.png", dpi=160)
    plt.close(fig)

    print(full.to_string(index=False))


if __name__ == "__main__":
    main()
