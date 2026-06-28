#!/usr/bin/env python3
"""Rolling stack over pure FU-new-factor components.

This is a stricter follow-up to ``pure_newfactor_history_stack.py``:

  - old/old9 anchors are excluded;
  - model selection is done by 2019Q4 rolling validation only;
  - 2020 is evaluated month by month, fitting stack weights only on rows
    strictly before the predicted month.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import expanded_history_gate_clean as eh
import pure_newfactor_history_stack as ph
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic


ROOT = Path("/root/autodl-tmp/quant/ML")
STRICT = ROOT / "strict_opt_results"
OUT_DIR = STRICT / "pure_newfactor_rolling_stack"

HIST_STARTS = [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-04-01")]
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
    signed: bool
    train_start: pd.Timestamp


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def months(start: str, end: str) -> list[str]:
    return [str(x) for x in pd.period_range(start, end, freq="M")]


def add_label_views(base: pd.DataFrame) -> pd.DataFrame:
    base = eh.add_xsz_label(base.copy())
    g = base.groupby("datetime", sort=False)["label"]
    base["label_xrank_fit"] = (g.rank(pct=True) - 0.5).astype(np.float32)
    return base


def build_design() -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, list[int]]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist_months = months("2019-01", "2019-12")
    test_months = months("2020-01", "2020-12")
    hist_base = None
    test_base = None
    hist_cols: list[np.ndarray] = []
    test_cols: list[np.ndarray] = []
    names: list[str] = []

    for root_name, root, part_prefix in [
        ("topk", ph.TOPK_DIR, "rolling_"),
        ("fullret", ph.THREE_DIR, "rolling_"),
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
                hist = ph.read_parts(root, part, hist_months, view, col)
                test = ph.read_parts(root, part, test_months, view, col)
                hist_base, test_base = ph.add_feature(
                    hist_base, test_base, hist, test, col, names, hist_cols, test_cols
                )

    topk_lowcorr = ph.TOPK_DIR / "lowcorr_anchor_residual" / "topk_anchor_lowcorr_lgb_meta_chain_xsz.parquet"
    for col in ["anchor_pred_xsz", "resid_pred", "combined_pred"]:
        out_col = f"topk_lowcorr__{col}"
        full = ph.read_full(topk_lowcorr, col, out_col)
        hist_base, test_base = ph.add_feature(hist_base, test_base, full, full, out_col, names, hist_cols, test_cols)

    full_lowcorr = ph.THREE_DIR / "lowcorr_residual" / "new_lgb_resid_lgb_meta_chain_xsz.parquet"
    if full_lowcorr.exists():
        for col in ["pred", "pred_xsz", "pred_xrank"]:
            out_col = f"fullret_lowcorr__{col}"
            full = ph.read_full(full_lowcorr, col, out_col)
            hist_base, test_base = ph.add_feature(
                hist_base, test_base, full, full, out_col, names, hist_cols, test_cols
            )

    if hist_base is None or test_base is None:
        raise RuntimeError("no pure-new components were loaded")

    base = pd.concat(
        [
            hist_base[["symbol", "datetime", "label"]],
            test_base[["symbol", "datetime", "label"]],
        ],
        ignore_index=True,
    )
    base["datetime"] = pd.to_datetime(base["datetime"])
    base = add_label_views(base)
    x = np.vstack([np.column_stack(hist_cols), np.column_stack(test_cols)]).astype(np.float32, copy=False)

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

    pd.DataFrame({"feature": names}).to_csv(OUT_DIR / "features.csv", index=False)
    pd.DataFrame(
        [{"feature_set": key, "n_features": len(value)} for key, value in feature_sets.items()]
    ).to_csv(OUT_DIR / "feature_sets.csv", index=False)
    return base, x, names, feature_sets


def stack_choices() -> list[StackChoice]:
    out: list[StackChoice] = []
    fast = os.environ.get("PURE_ROLL_FULL", "0") != "1"
    for train_start in HIST_STARTS:
        feature_sets = [
            "topk_three_xsz",
            "topk_three_all",
            "topk_three_lowcorr_xsz",
            "topk_all_lowcorr",
            "all_new_xsz",
            "all_new_all_views",
        ]
        for feature_set in feature_sets:
            targets = ["raw", "xsz", "xrank"]
            uppers = [0.25, 0.50, 0.90]
            signed_values = [False, True]
            if fast:
                targets = ["raw", "xsz"]
                uppers = [0.90]
                signed_values = [False]
                if feature_set in {"all_new_all_views", "topk_all_lowcorr"}:
                    signed_values = [False, True]
            for target in targets:
                for standardize in [False, True]:
                    for upper in uppers:
                        for signed in signed_values:
                            out.append(
                                StackChoice(
                                    name=(
                                        f"{feature_set}__{target}__std{int(standardize)}"
                                        f"__u{upper:g}__{'signed' if signed else 'nonneg'}"
                                        f"__from{train_start:%Y%m}"
                                    ),
                                    feature_set=feature_set,
                                    target=target,
                                    standardize=standardize,
                                    upper=upper,
                                    signed=signed,
                                    train_start=train_start,
                                )
                            )
    return out


def target_values(base: pd.DataFrame, target: str) -> np.ndarray:
    if target == "raw":
        return base["label"].to_numpy(np.float64)
    if target == "xsz":
        return base["label_xsz_fit"].to_numpy(np.float64)
    if target == "xrank":
        return base["label_xrank_fit"].to_numpy(np.float64)
    raise ValueError(target)


def fit_weights(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    cols: list[int],
    choice: StackChoice,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    xx = scrub(x[mask][:, cols].astype(np.float64, copy=False))
    yy = y[mask].astype(np.float64, copy=False)
    good = np.isfinite(yy)
    xx = xx[good]
    yy = yy[good]

    mean = np.zeros(xx.shape[1], dtype=np.float64)
    scale = np.ones(xx.shape[1], dtype=np.float64)
    if choice.standardize:
        mean = xx.mean(axis=0)
        scale = np.maximum(xx.std(axis=0), 1e-9)
        xx = (xx - mean) / scale

    gram = xx.T @ xx
    cov = xx.T @ yy
    yty = float(yy @ yy)
    lower = np.full(len(cols), -0.15 if choice.signed else 0.0, dtype=np.float64)
    upper = np.full(len(cols), choice.upper, dtype=np.float64)
    w, ic = fit_ic_weights_from_stats(cov, gram, yty, lower, upper)
    return w, float(ic), mean, scale


def apply_weights(
    x: np.ndarray,
    mask: np.ndarray,
    cols: list[int],
    w: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    standardize: bool,
) -> np.ndarray:
    xx = scrub(x[mask][:, cols].astype(np.float32, copy=False))
    if standardize:
        xx = ((xx.astype(np.float64) - mean) / scale).astype(np.float32)
    return xx @ w.astype(np.float32)


def rolling_predict(
    base: pd.DataFrame,
    x: np.ndarray,
    cols: list[int],
    choice: StackChoice,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dt = base["datetime"]
    y = target_values(base, choice.target)
    rows = []
    weights = []
    for ms in pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS"):
        train_mask = ((dt >= choice.train_start) & (dt < ms) & base["label"].notna()).to_numpy()
        test_mask = ((dt >= ms) & (dt < ms + pd.DateOffset(months=1))).to_numpy()
        if int(train_mask.sum()) < 50_000 or int(test_mask.sum()) == 0:
            continue
        w, fit_ic, mean, scale = fit_weights(x, y, train_mask, cols, choice)
        pred = apply_weights(x, test_mask, cols, w, mean, scale, choice.standardize)
        part = base.loc[test_mask, ["symbol", "datetime", "label"]].copy()
        part["pred"] = pred.astype(np.float32)
        rows.append(part)
        row = {
            "model": choice.name,
            "month": f"{ms:%Y-%m}",
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "fit_ic": fit_ic,
            "month_ic": compute_ic(part["pred"], part["label"]),
        }
        for local_i, wv in enumerate(w):
            row[f"w_{cols[local_i]}"] = float(wv)
        weights.append(row)
    if not rows:
        raise RuntimeError(f"no rolling rows for {choice.name}")
    pred_df = pd.concat(rows, ignore_index=True)
    pred_df = add_cross_sectional_norms(pred_df, "pred")
    return pred_df, pd.DataFrame(weights)


def summarize(pred: pd.DataFrame, model: str, tag: str) -> dict[str, object]:
    monthly = period_ic(pred, "pred", "M")
    return {
        "model": model,
        "tag": tag,
        "rows": int(len(pred)),
        "label_rows": int(pred["label"].notna().sum()),
        "pred_ic": compute_ic(pred["pred"], pred["label"]),
        "pred_xsz_ic": compute_ic(pred["pred_xsz"], pred["label"]),
        "pred_xrank_ic": compute_ic(pred["pred_xrank"], pred["label"]),
        "monthly_mean": float(monthly.mean()),
        "monthly_ir": float(monthly.mean() / monthly.std()) if monthly.std() > 0 else float("nan"),
    }


def main() -> None:
    base, x, names, feature_sets = build_design()
    all_rows = []
    all_weight_rows = []
    preds_2020: dict[str, pd.DataFrame] = {}

    for i, choice in enumerate(stack_choices(), start=1):
        cols = feature_sets[choice.feature_set]
        if not cols:
            continue
        val_pred, val_weights = rolling_predict(base, x, cols, choice, VAL_START, TEST_START)
        test_pred, test_weights = rolling_predict(base, x, cols, choice, TEST_START, TEST_END)
        model = f"pure_newfactor_rolling_stack__{choice.name}"

        row = summarize(test_pred, model, "2020")
        val_row = summarize(val_pred, model, "2019q4")
        row.update(
            {
                "val_raw_ic_2019q4": val_row["pred_ic"],
                "val_xsz_ic_2019q4": val_row["pred_xsz_ic"],
                "val_monthly_mean_2019q4": val_row["monthly_mean"],
                "feature_set": choice.feature_set,
                "target": choice.target,
                "standardize": choice.standardize,
                "upper": choice.upper,
                "signed": choice.signed,
                "train_start": f"{choice.train_start:%Y-%m}",
                "n_features": len(cols),
            }
        )
        all_rows.append(row)

        val_weights["phase"] = "val_2019q4"
        test_weights["phase"] = "test_2020"
        all_weight_rows.append(val_weights)
        all_weight_rows.append(test_weights)
        preds_2020[model] = test_pred
        print(
            f"[pure-new-roll {i:03d}] {model} val={row['val_raw_ic_2019q4']:.6f} "
            f"test={row['pred_ic']:.6f}",
            flush=True,
        )

    summary = pd.DataFrame(all_rows).sort_values("val_raw_ic_2019q4", ascending=False).reset_index(drop=True)
    summary["selected_by_val_raw"] = summary["model"] == summary.iloc[0]["model"]
    summary["diagnostic_best_2020_raw"] = summary["model"] == summary.sort_values("pred_ic", ascending=False).iloc[0]["model"]
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.concat(all_weight_rows, ignore_index=True).to_csv(OUT_DIR / "rolling_weights.csv", index=False)

    selected_model = str(summary.iloc[0]["model"])
    diagnostic_model = str(summary.sort_values("pred_ic", ascending=False).iloc[0]["model"])
    preds_2020[selected_model].to_parquet(OUT_DIR / "selected_by_val_raw.parquet", index=False)
    preds_2020[diagnostic_model].to_parquet(OUT_DIR / "diagnostic_best_2020_raw.parquet", index=False)

    key = pd.concat(
        [
            summary[summary["selected_by_val_raw"]],
            summary[summary["diagnostic_best_2020_raw"]],
            summary.head(8),
        ],
        ignore_index=True,
    ).drop_duplicates("model")
    key.to_csv(OUT_DIR / "key_results.csv", index=False)

    monthly = []
    for model in key["model"].astype(str):
        monthly.append(period_ic(preds_2020[model], "pred", "M").rename(model))
    pd.concat(monthly, axis=1).to_csv(OUT_DIR / "key_monthly_ic.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [x.replace("pure_newfactor_rolling_stack__", "") for x in key["model"]]
    y = np.arange(len(key))
    ax.barh(y + 0.18, key["pred_ic"], height=0.34, color="#28666E", label="raw IC")
    ax.barh(y - 0.18, key["pred_xsz_ic"], height=0.34, color="#7CB518", label="xsz IC")
    ax.axvline(0.0554976, color="#D1495B", linestyle=":", linewidth=1.2, label="old 9 raw")
    ax.axvline(0.06, color="#C44536", linestyle="--", linewidth=1.2, label="target 0.06")
    ax.set_yticks(y, labels)
    ax.set_xlabel("2020 merged IC")
    ax.set_title("Pure New-Factor Rolling Stack")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "key_results.png", dpi=170)
    plt.close(fig)

    print(key.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
