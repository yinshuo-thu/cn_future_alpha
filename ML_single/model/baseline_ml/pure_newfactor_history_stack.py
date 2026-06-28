#!/usr/bin/env python3
"""Pure FU-new-factor history-CV stack.

This script intentionally excludes old clean/old9 anchors.  It only stacks
train-before-test predictions produced from the FU new-factor route:

  - topK rolling Ridge/LGB/MLP;
  - topK low-correlation LGB residual/combined stream;
  - full-retained rolling LGB/MLP streams;
  - full-retained low-correlation residual stream.

Stack selection uses 2019Apr-Sep for weight fitting and 2019Q4 for model
selection.  Final weights are fit on 2019Apr-Dec and evaluated on 2020.
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
TOPK_DIR = STRICT / "fu_newfactor_topk_best_2019q4"
THREE_DIR = STRICT / "fu_newfactor_three_model"
OUT_DIR = STRICT / "pure_newfactor_history_stack"

HIST_START = pd.Timestamp("2019-04-01")
VAL_START = pd.Timestamp("2019-10-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")


@dataclass(frozen=True)
class StackChoice:
    name: str
    feature_set: str
    target: str
    standardize: bool
    upper: float


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def months(start: str, end: str) -> list[str]:
    return [str(x) for x in pd.period_range(start, end, freq="M")]


def read_parts(root: Path, part: str, month_list: list[str], view: str, out_col: str) -> pd.DataFrame:
    pieces = []
    part_dir = root / "prediction_parts" / part
    for month in month_list:
        path = part_dir / f"{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        cur = pd.read_parquet(path, columns=["symbol", "datetime", "label", view])
        cur["datetime"] = pd.to_datetime(cur["datetime"])
        pieces.append(cur.rename(columns={view: out_col}))
    return pd.concat(pieces, ignore_index=True)


def read_full(path: Path, col: str, out_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["symbol", "datetime", "label", col])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.rename(columns={col: out_col})


def align(base: pd.DataFrame, df: pd.DataFrame, col: str) -> np.ndarray:
    cur = base[["symbol", "datetime"]].merge(df[["symbol", "datetime", col]], on=["symbol", "datetime"], how="left")
    return scrub(cur[col].to_numpy(np.float32, copy=False)).astype(np.float32, copy=False)


def add_feature(
    hist_base: pd.DataFrame | None,
    test_base: pd.DataFrame | None,
    hist_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
    names: list[str],
    hist_cols: list[np.ndarray],
    test_cols: list[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hist_base is None:
        hist_base = hist_df[["symbol", "datetime", "label"]].copy().reset_index(drop=True)
        hist_base = eh.add_xsz_label(hist_base)
    if test_base is None:
        test_base = test_df[["symbol", "datetime", "label"]].copy().reset_index(drop=True)
    hist_cols.append(align(hist_base, hist_df, col))
    test_cols.append(align(test_base, test_df, col))
    names.append(col)
    return hist_base, test_base


def build_design() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], dict[str, list[int]]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist_months = months("2019-04", "2019-12")
    test_months = months("2020-01", "2020-12")
    hist_base = None
    test_base = None
    hist_cols: list[np.ndarray] = []
    test_cols: list[np.ndarray] = []
    names: list[str] = []

    for root_name, root, part_prefix in [
        ("topk", TOPK_DIR, "rolling_"),
        ("fullret", THREE_DIR, "rolling_"),
    ]:
        for model in ["ridge", "lgb", "mlp"]:
            if root_name == "fullret" and model == "ridge":
                continue
            part = f"{part_prefix}{model}"
            part_dir = root / "prediction_parts" / part
            if not part_dir.exists():
                continue
            for view in ["pred", "pred_xsz", "pred_xrank"]:
                col = f"{root_name}__{model}__{view}"
                hist = read_parts(root, part, hist_months, view, col)
                test = read_parts(root, part, test_months, view, col)
                hist_base, test_base = add_feature(hist_base, test_base, hist, test, col, names, hist_cols, test_cols)

    topk_lowcorr = TOPK_DIR / "lowcorr_anchor_residual" / "topk_anchor_lowcorr_lgb_meta_chain_xsz.parquet"
    for col in ["anchor_pred_xsz", "resid_pred", "combined_pred"]:
        out_col = f"topk_lowcorr__{col}"
        full = read_full(topk_lowcorr, col, out_col)
        hist_base, test_base = add_feature(hist_base, test_base, full, full, out_col, names, hist_cols, test_cols)

    full_lowcorr = THREE_DIR / "lowcorr_residual" / "new_lgb_resid_lgb_meta_chain_xsz.parquet"
    if full_lowcorr.exists():
        for col in ["pred", "pred_xsz", "pred_xrank"]:
            out_col = f"fullret_lowcorr__{col}"
            full = read_full(full_lowcorr, col, out_col)
            hist_base, test_base = add_feature(hist_base, test_base, full, full, out_col, names, hist_cols, test_cols)

    assert hist_base is not None and test_base is not None
    xh = np.column_stack(hist_cols).astype(np.float32)
    xt = np.column_stack(test_cols).astype(np.float32)
    pd.DataFrame({"feature": names}).to_csv(OUT_DIR / "features.csv", index=False)

    def idx(predicate) -> list[int]:
        return [i for i, name in enumerate(names) if predicate(name)]

    feature_sets = {
        "topk_three_xsz": idx(lambda n: n.startswith("topk__") and n.endswith("__pred_xsz")),
        "topk_three_all": idx(lambda n: n.startswith("topk__")),
        "topk_three_lowcorr_xsz": idx(
            lambda n: (n.startswith("topk__") and n.endswith("__pred_xsz")) or n == "topk_lowcorr__combined_pred"
        ),
        "topk_all_lowcorr": idx(lambda n: n.startswith("topk__") or n.startswith("topk_lowcorr__")),
        "all_new_xsz": idx(lambda n: n.endswith("__pred_xsz") or n.endswith("__combined_pred")),
        "all_new_all_views": list(range(len(names))),
    }
    pd.DataFrame(
        [{"feature_set": key, "n_features": len(value)} for key, value in feature_sets.items()]
    ).to_csv(OUT_DIR / "feature_sets.csv", index=False)
    return hist_base, test_base, xh, xt, names, feature_sets


def stack_choices() -> list[StackChoice]:
    out = []
    for feature_set in [
        "topk_three_xsz",
        "topk_three_all",
        "topk_three_lowcorr_xsz",
        "topk_all_lowcorr",
        "all_new_xsz",
        "all_new_all_views",
    ]:
        for target in ["raw", "xsz"]:
            for standardize in [False, True]:
                for upper in [0.10, 0.25, 0.50, 0.90]:
                    out.append(
                        StackChoice(
                            name=f"{feature_set}__{target}__std{int(standardize)}__u{upper:g}",
                            feature_set=feature_set,
                            target=target,
                            standardize=standardize,
                            upper=upper,
                        )
                    )
    return out


def fit_apply(x_train: np.ndarray, y_train: np.ndarray, x_apply: np.ndarray, cols: list[int], choice: StackChoice):
    spec = topk.StackSpec(
        name=choice.name,
        modes=("pure_new",),
        views=("pred",),
        top_n=len(cols),
        target=choice.target,
        standardize=choice.standardize,
        upper=choice.upper,
    )
    w, fit_ic, mean, scale = topk.fit_stack(x_train, y_train, cols, spec)
    pred = topk.apply_stack(x_apply, cols, w, mean, scale, choice.standardize)
    return pred, w, fit_ic


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
    for choice in stack_choices():
        cols = feature_sets[choice.feature_set]
        if not cols:
            continue
        y_train = y_train_xsz if choice.target == "xsz" else y_train_raw
        val_pred, train_w, train_ic = fit_apply(xh[train_mask], y_train, xh[val_mask], cols, choice)
        val_raw = compute_ic(val_pred, y_val_raw)
        val_xsz = compute_ic(val_pred, y_val_xsz)
        y_full = y_full_xsz if choice.target == "xsz" else y_full_raw
        test_values, final_w, final_ic = fit_apply(xh, y_full, xt, cols, choice)
        pred = test_base[["symbol", "datetime", "label"]].copy()
        pred["pred"] = test_values.astype(np.float32)
        pred = add_cross_sectional_norms(pred, "pred")
        model = f"pure_newfactor_stack__{choice.name}"
        row = summarize(pred, model)
        row.update(
            {
                "feature_set": choice.feature_set,
                "target": choice.target,
                "standardize": choice.standardize,
                "upper": choice.upper,
                "n_features": len(cols),
                "train_ic_2019apr_sep": train_ic,
                "val_raw_ic_2019q4": val_raw,
                "val_xsz_ic_2019q4": val_xsz,
                "final_fit_ic_2019apr_dec": final_ic,
            }
        )
        rows.append(row)
        preds[model] = pred
        for i, w in zip(cols, final_w):
            weight_rows.append({"model": model, "feature": names[i], "weight": float(w)})
        print(f"[pure-new] {model} val={val_raw:.6f} test={row['pred_ic_2020']:.6f}", flush=True)

    summary = pd.DataFrame(rows).sort_values("val_raw_ic_2019q4", ascending=False)
    summary["selected_by_val_raw"] = summary["model"] == summary.iloc[0]["model"]
    summary["diagnostic_best_2020_raw"] = summary["model"] == summary.sort_values("pred_ic_2020", ascending=False).iloc[0]["model"]
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUT_DIR / "weights.csv", index=False)

    selected_model = str(summary.iloc[0]["model"])
    diagnostic_model = str(summary.sort_values("pred_ic_2020", ascending=False).iloc[0]["model"])
    preds[selected_model].to_parquet(OUT_DIR / "selected_by_val_raw.parquet", index=False)
    preds[diagnostic_model].to_parquet(OUT_DIR / "diagnostic_best_2020_raw.parquet", index=False)

    key = pd.concat(
        [
            summary[summary["selected_by_val_raw"]],
            summary[summary["diagnostic_best_2020_raw"]],
            summary[summary["feature_set"].eq("topk_three_lowcorr_xsz")].head(1),
        ],
        ignore_index=True,
    ).drop_duplicates("model")
    key.to_csv(OUT_DIR / "key_results.csv", index=False)
    monthly = []
    for model in key["model"].astype(str):
        monthly.append(period_ic(preds[model], "pred", "M").rename(model))
    monthly_df = pd.concat(monthly, axis=1)
    monthly_df.to_csv(OUT_DIR / "key_monthly_ic.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [x.replace("pure_newfactor_stack__", "") for x in key["model"]]
    y = np.arange(len(key))
    ax.barh(y + 0.18, key["pred_ic_2020"], height=0.34, color="#2F6B8F", label="raw IC")
    ax.barh(y - 0.18, key["pred_xsz_ic_2020"], height=0.34, color="#76B7B2", label="xsz IC")
    ax.axvline(0.06, color="#C44536", linestyle="--", linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("2020 merged IC")
    ax.set_title("Pure New-Factor History-CV Stack")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "key_results.png", dpi=170)
    plt.close(fig)

    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
