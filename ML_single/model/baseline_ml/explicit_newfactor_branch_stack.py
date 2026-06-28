#!/usr/bin/env python3
"""Explicit top-level stack with robust anchors plus FU new-factor branches.

All inputs are already train-before-test predictions.  Stack hyperparameters are
selected on 2019Q4 after fitting weights on 2019Apr-Sep; final weights are fit on
2019Apr-Dec and evaluated on 2020.
"""

from __future__ import annotations

from dataclasses import dataclass
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
STRICT = ROOT / "strict_opt_results"
HIST_DIR = STRICT / "expanded_topk_history_cv_stack"
BASE_PRED_DIR = HIST_DIR / "base_predictions"
TOPK_DIR = STRICT / "fu_newfactor_topk_best_2019q4"
OUT_DIR = STRICT / "explicit_newfactor_branch_stack"

HIST_START = pd.Timestamp("2019-04-01")
VAL_START = pd.Timestamp("2019-10-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")

ANCHOR_CONFIGS = [
    "old_old9_topk_all_xsz_month_decay6_u090",
    "old_family_top24_xsz_month_equal_signed05_u090",
]


@dataclass(frozen=True)
class BranchSpec:
    name: str
    feature_set: str
    target: str
    standardize: bool
    upper: float


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def month_list(start: str, end: str) -> list[str]:
    return [str(x) for x in pd.period_range(start, end, freq="M")]


def read_anchor(cfg: str, suffix: str) -> pd.DataFrame:
    path = BASE_PRED_DIR / f"{cfg}__{suffix}.parquet"
    df = pd.read_parquet(path, columns=["symbol", "datetime", "label", "pred"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.rename(columns={"pred": f"anchor__{cfg}"})


def align_column(base: pd.DataFrame, df: pd.DataFrame, col: str) -> np.ndarray:
    cur = base[["symbol", "datetime"]].merge(df[["symbol", "datetime", col]], on=["symbol", "datetime"], how="left")
    return scrub(cur[col].to_numpy(np.float32, copy=False)).astype(np.float32, copy=False)


def read_topk_part(part: str, months: list[str], view: str, out_col: str) -> pd.DataFrame:
    pieces = []
    part_dir = TOPK_DIR / "prediction_parts" / part
    for month in months:
        path = part_dir / f"{month}.parquet"
        cur = pd.read_parquet(path, columns=["symbol", "datetime", view])
        cur["datetime"] = pd.to_datetime(cur["datetime"])
        pieces.append(cur.rename(columns={view: out_col}))
    return pd.concat(pieces, ignore_index=True)


def read_lowcorr(col: str, out_col: str) -> pd.DataFrame:
    path = TOPK_DIR / "lowcorr_anchor_residual" / "topk_anchor_lowcorr_lgb_meta_chain_xsz.parquet"
    df = pd.read_parquet(path, columns=["symbol", "datetime", col])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.rename(columns={col: out_col})


def build_design() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], dict[str, list[int]]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist_months = month_list("2019-04", "2019-12")
    test_months = month_list("2020-01", "2020-12")

    hist_base = None
    test_base = None
    hist_cols: list[np.ndarray] = []
    test_cols: list[np.ndarray] = []
    names: list[str] = []

    for cfg in ANCHOR_CONFIGS:
        h = read_anchor(cfg, "hist_roll")
        t = read_anchor(cfg, "test_roll")
        col = f"anchor__{cfg}"
        if hist_base is None:
            hist_base = h[["symbol", "datetime", "label"]].copy().reset_index(drop=True)
            hist_base = eh.add_xsz_label(hist_base)
            test_base = t[["symbol", "datetime", "label"]].copy().reset_index(drop=True)
        hist_cols.append(align_column(hist_base, h, col))
        test_cols.append(align_column(test_base, t, col))
        names.append(col)

    assert hist_base is not None and test_base is not None

    for part, family in [("rolling_ridge", "ridge"), ("rolling_lgb", "lgb"), ("rolling_mlp", "mlp")]:
        for view in ["pred", "pred_xsz", "pred_xrank"]:
            col = f"newtopk__{family}__{view}"
            h = read_topk_part(part, hist_months, view, col)
            t = read_topk_part(part, test_months, view, col)
            hist_cols.append(align_column(hist_base, h, col))
            test_cols.append(align_column(test_base, t, col))
            names.append(col)

    for col in ["anchor_pred_xsz", "resid_pred", "combined_pred"]:
        out_col = f"newtopk__lowcorr__{col}"
        h = read_lowcorr(col, out_col)
        t = h
        hist_cols.append(align_column(hist_base, h, out_col))
        test_cols.append(align_column(test_base, t, out_col))
        names.append(out_col)

    xh = np.column_stack(hist_cols).astype(np.float32)
    xt = np.column_stack(test_cols).astype(np.float32)
    feature_sets = {
        "anchor_only": [i for i, n in enumerate(names) if n.startswith("anchor__")],
        "anchor_new_xsz": [
            i
            for i, n in enumerate(names)
            if n.startswith("anchor__") or n.endswith("__pred_xsz") or n == "newtopk__lowcorr__combined_pred"
        ],
        "anchor_new_all": list(range(len(names))),
    }
    pd.DataFrame({"feature": names}).to_csv(OUT_DIR / "features.csv", index=False)
    return hist_base, test_base, xh, xt, names, feature_sets


def specs() -> list[BranchSpec]:
    out = []
    for feature_set in ["anchor_only", "anchor_new_xsz", "anchor_new_all"]:
        for target in ["raw", "xsz"]:
            for standardize in [False, True]:
                for upper in [0.10, 0.25, 0.50, 0.90]:
                    out.append(
                        BranchSpec(
                            name=f"{feature_set}__{target}__std{int(standardize)}__u{upper:g}",
                            feature_set=feature_set,
                            target=target,
                            standardize=standardize,
                            upper=upper,
                        )
                    )
    return out


def fit_apply(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_apply: np.ndarray,
    cols: list[int],
    spec: BranchSpec,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    stack_spec = topk.StackSpec(
        name=spec.name,
        modes=("explicit",),
        views=("pred",),
        top_n=len(cols),
        target=spec.target,
        standardize=spec.standardize,
        upper=spec.upper,
    )
    weights, fit_ic, mean, scale = topk.fit_stack(x_train, y_train, cols, stack_spec)
    pred = topk.apply_stack(x_apply, cols, weights, mean, scale, spec.standardize)
    return pred, weights, fit_ic, mean, scale


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


def main() -> None:
    hist_base, test_base, xh, xt, names, feature_sets = build_design()
    train_mask = (hist_base["datetime"] < VAL_START).to_numpy()
    val_mask = (hist_base["datetime"] >= VAL_START).to_numpy()
    y_train_raw = hist_base.loc[train_mask, "label"].to_numpy(np.float64)
    y_train_xsz = hist_base.loc[train_mask, "label_xsz_fit"].to_numpy(np.float64)
    y_val_raw = hist_base.loc[val_mask, "label"].to_numpy(np.float64)
    y_val_xsz = hist_base.loc[val_mask, "label_xsz_fit"].to_numpy(np.float64)
    y_full_raw = hist_base["label"].to_numpy(np.float64)
    y_full_xsz = hist_base["label_xsz_fit"].to_numpy(np.float64)

    rows = []
    weight_rows = []
    preds: dict[str, pd.DataFrame] = {}
    for spec in specs():
        cols = feature_sets[spec.feature_set]
        y_train = y_train_xsz if spec.target == "xsz" else y_train_raw
        val_pred, train_w, train_ic, _, _ = fit_apply(xh[train_mask], y_train, xh[val_mask], cols, spec)
        val_raw = compute_ic(val_pred, y_val_raw)
        val_xsz = compute_ic(val_pred, y_val_xsz)
        y_full = y_full_xsz if spec.target == "xsz" else y_full_raw
        test_pred_values, final_w, final_fit_ic, _, _ = fit_apply(xh, y_full, xt, cols, spec)
        pred = test_base[["symbol", "datetime", "label"]].copy()
        pred["pred"] = test_pred_values.astype(np.float32)
        pred = add_cross_sectional_norms(pred, "pred")
        model = f"explicit_newfactor_branch__{spec.name}"
        row = summarize(pred, model)
        row.update(
            {
                "feature_set": spec.feature_set,
                "target": spec.target,
                "standardize": spec.standardize,
                "upper": spec.upper,
                "n_features": len(cols),
                "train_ic_2019apr_sep": train_ic,
                "val_raw_ic_2019q4": val_raw,
                "val_xsz_ic_2019q4": val_xsz,
                "final_fit_ic_2019apr_dec": final_fit_ic,
            }
        )
        rows.append(row)
        preds[model] = pred
        for idx, weight in zip(cols, final_w):
            weight_rows.append({"model": model, "feature": names[idx], "weight": float(weight)})
        print(f"[explicit] {model} val={val_raw:.6f} test={row['pred_ic_2020']:.6f}", flush=True)

    summary = pd.DataFrame(rows).sort_values("val_raw_ic_2019q4", ascending=False)
    summary["selected_by_val_raw"] = summary["model"] == summary.iloc[0]["model"]
    summary["diagnostic_best_2020_raw"] = summary["model"] == summary.sort_values("pred_ic_2020", ascending=False).iloc[0]["model"]
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUT_DIR / "weights.csv", index=False)

    selected_model = str(summary.iloc[0]["model"])
    best_model = str(summary.sort_values("pred_ic_2020", ascending=False).iloc[0]["model"])
    preds[selected_model].to_parquet(OUT_DIR / "selected_by_val_raw.parquet", index=False)
    preds[best_model].to_parquet(OUT_DIR / "diagnostic_best_2020_raw.parquet", index=False)

    key = pd.concat(
        [
            summary[summary["selected_by_val_raw"]],
            summary[summary["diagnostic_best_2020_raw"]],
            summary[(summary["feature_set"] == "anchor_new_xsz") & (summary["target"] == "xsz")].head(1),
        ],
        ignore_index=True,
    ).drop_duplicates("model")
    key.to_csv(OUT_DIR / "key_results.csv", index=False)

    monthly_parts = []
    for model in key["model"].astype(str):
        monthly = period_ic(preds[model], "pred", "M")
        monthly_parts.append(monthly.rename(model))
    monthly_df = pd.concat(monthly_parts, axis=1)
    monthly_df.to_csv(OUT_DIR / "key_monthly_ic.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    key.set_index("model")[["pred_ic_2020", "pred_xsz_ic_2020"]].plot(kind="bar", ax=ax, color=["#2F6B8F", "#76B7B2"])
    ax.axhline(0.06, color="#C44536", linestyle="--", linewidth=1.2)
    ax.set_xlabel("")
    ax.set_ylabel("IC")
    ax.set_title("Explicit New-Factor Branch Stack")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "key_results.png", dpi=160)
    plt.close(fig)

    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
