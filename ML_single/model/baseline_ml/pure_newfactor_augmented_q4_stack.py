#!/usr/bin/env python3
"""Augmented pure-new-factor stack using only 2019Q4 for stack selection.

This follows the train-before-test base learner artifacts already produced by
the FU-new-factor route.  It adds the latest raw-target Ridge/LGB and MLP
variants as extra candidates, selects stack details on 2019-12 after fitting on
2019-10..2019-11, then fits final weights on 2019Q4 and evaluates 2020.
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
from rolling_factor_model_eval import compute_ic, fit_ic_weights_from_stats, period_ic


ROOT = Path("/root/autodl-tmp/quant/ML")
STRICT = ROOT / "strict_opt_results"
OUT_DIR = STRICT / "pure_newfactor_augmented_q4_stack"

TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")
FIT_END = pd.Timestamp("2019-12-01")


@dataclass(frozen=True)
class Component:
    name: str
    root: Path
    part: str
    family: str


@dataclass(frozen=True)
class Spec:
    name: str
    feature_set: str
    target: str
    standardize: bool
    upper: float
    signed: bool


def months(start: str, end: str) -> list[str]:
    return [str(x) for x in pd.period_range(start, end, freq="M")]


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def read_parts(root: Path, part: str, month_list: list[str], view: str, out_col: str) -> pd.DataFrame:
    pieces = []
    for month in month_list:
        path = root / "prediction_parts" / part / f"{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        cur = pd.read_parquet(path, columns=["symbol", "datetime", "label", view])
        cur["datetime"] = pd.to_datetime(cur["datetime"])
        pieces.append(cur.rename(columns={view: out_col}))
    return pd.concat(pieces, ignore_index=True)


def read_full(path: Path, col: str, out_col: str) -> pd.DataFrame:
    cur = pd.read_parquet(path, columns=["symbol", "datetime", "label", col])
    cur["datetime"] = pd.to_datetime(cur["datetime"])
    return cur.rename(columns={col: out_col})


def align(base: pd.DataFrame, frame: pd.DataFrame, col: str) -> np.ndarray:
    cur = base[["symbol", "datetime"]].merge(frame[["symbol", "datetime", col]], on=["symbol", "datetime"], how="left")
    return scrub(cur[col].to_numpy(np.float32, copy=False)).astype(np.float32, copy=False)


def add_component(
    base: pd.DataFrame | None,
    frame: pd.DataFrame,
    col: str,
    cols: list[np.ndarray],
    names: list[str],
    families: list[str],
    family: str,
) -> pd.DataFrame:
    if base is None:
        base = frame[["symbol", "datetime", "label"]].copy().reset_index(drop=True)
    cols.append(align(base, frame, col))
    names.append(col)
    families.append(family)
    return base


def component_specs() -> list[Component]:
    topk = STRICT / "fu_newfactor_topk_best_2019q4"
    fullret = STRICT / "fu_newfactor_three_model"
    raw_ridge = STRICT / "fu_newfactor_topk_ridge_rawtarget"
    raw_lgb = STRICT / "fu_newfactor_topk_lgb_rawtarget"
    cons_lgb = STRICT / "fu_newfactor_topk_lgb_conservative"
    mlp_e6 = STRICT / "fu_newfactor_mlp_ret537_e6_s15k"
    return [
        Component("topk_ridge", topk, "rolling_ridge", "ridge"),
        Component("topk_lgb", topk, "rolling_lgb", "lgb"),
        Component("topk_mlp", topk, "rolling_mlp", "mlp"),
        Component("fullret_lgb", fullret, "rolling_lgb", "lgb"),
        Component("fullret_mlp", fullret, "rolling_mlp", "mlp"),
        Component("raw_ridge", raw_ridge, "rolling_ridge", "ridge"),
        Component("raw_lgb", raw_lgb, "rolling_lgb", "lgb"),
        Component("cons_lgb", cons_lgb, "rolling_lgb", "lgb"),
        Component("mlp_e6", mlp_e6, "rolling_mlp", "mlp"),
    ]


def build_design() -> tuple[pd.DataFrame, np.ndarray, list[str], list[str], dict[str, list[int]]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    month_list = months("2019-10", "2020-12")
    base: pd.DataFrame | None = None
    cols: list[np.ndarray] = []
    names: list[str] = []
    families: list[str] = []

    for comp in component_specs():
        part_dir = comp.root / "prediction_parts" / comp.part
        if not part_dir.exists():
            continue
        for view in ["pred", "pred_xsz", "pred_xrank"]:
            col = f"{comp.name}__{view}"
            frame = read_parts(comp.root, comp.part, month_list, view, col)
            base = add_component(base, frame, col, cols, names, families, comp.family)

    topk_lowcorr = STRICT / "fu_newfactor_topk_best_2019q4" / "lowcorr_anchor_residual" / "topk_anchor_lowcorr_lgb_meta_chain_xsz.parquet"
    if topk_lowcorr.exists():
        for col in ["anchor_pred_xsz", "resid_pred", "combined_pred"]:
            out_col = f"topk_lowcorr__{col}"
            frame = read_full(topk_lowcorr, col, out_col)
            frame = frame[(frame["datetime"] >= "2019-10-01") & (frame["datetime"] < TEST_END)]
            base = add_component(base, frame, out_col, cols, names, families, "lowcorr")

    full_lowcorr = STRICT / "fu_newfactor_three_model" / "lowcorr_residual" / "new_lgb_resid_lgb_meta_chain_xsz.parquet"
    if full_lowcorr.exists():
        for view in ["pred", "pred_xsz", "pred_xrank"]:
            out_col = f"fullret_lowcorr__{view}"
            frame = read_full(full_lowcorr, view, out_col)
            frame = frame[(frame["datetime"] >= "2019-10-01") & (frame["datetime"] < TEST_END)]
            base = add_component(base, frame, out_col, cols, names, families, "lowcorr")

    if base is None:
        raise RuntimeError("no components loaded")
    base["datetime"] = pd.to_datetime(base["datetime"])
    base = eh.add_xsz_label(base)
    g = base.groupby("datetime", sort=False)["label"]
    base["label_xrank_fit"] = (g.rank(pct=True) - 0.5).astype(np.float32)
    x = np.column_stack(cols).astype(np.float32, copy=False)

    def idx(pred) -> list[int]:
        return [i for i, name in enumerate(names) if pred(i, name)]

    feature_sets = {
        "all": list(range(len(names))),
        "xsz_only": idx(lambda _i, n: n.endswith("__pred_xsz") or n.endswith("__combined_pred") or n.endswith("anchor_pred_xsz")),
        "raw_xsz": idx(lambda _i, n: n.endswith("__pred") or n.endswith("__pred_xsz") or n.endswith("__combined_pred")),
        "ridge_lgb_mlp": idx(lambda i, _n: families[i] in {"ridge", "lgb", "mlp"}),
        "ridge_lgb_mlp_xsz": idx(lambda i, n: families[i] in {"ridge", "lgb", "mlp"} and n.endswith("__pred_xsz")),
        "with_lowcorr": idx(lambda i, _n: families[i] in {"ridge", "lgb", "mlp", "lowcorr"}),
    }
    pd.DataFrame({"feature": names, "family": families}).to_csv(OUT_DIR / "features.csv", index=False)
    pd.DataFrame([{"feature_set": k, "n_features": len(v)} for k, v in feature_sets.items()]).to_csv(
        OUT_DIR / "feature_sets.csv", index=False
    )
    return base, x, names, families, feature_sets


def target_values(base: pd.DataFrame, target: str) -> np.ndarray:
    if target == "raw":
        return base["label"].to_numpy(np.float64)
    if target == "xsz":
        return base["label_xsz_fit"].to_numpy(np.float64)
    if target == "xrank":
        return base["label_xrank_fit"].to_numpy(np.float64)
    raise ValueError(target)


def specs() -> list[Spec]:
    out: list[Spec] = []
    core_feature_sets = ["all", "xsz_only", "raw_xsz", "ridge_lgb_mlp_xsz", "with_lowcorr"]
    for feature_set in core_feature_sets:
        for target in ["raw", "xsz"]:
            for standardize in [False, True]:
                for upper in [0.30, 0.90]:
                    out.append(
                        Spec(
                            f"{feature_set}__{target}__std{int(standardize)}__u{upper:g}__nonneg",
                            feature_set,
                            target,
                            standardize,
                            upper,
                            False,
                        )
                    )
    for feature_set in ["all", "with_lowcorr"]:
        for target in ["raw", "xsz"]:
            out.append(
                Spec(
                    f"{feature_set}__{target}__std1__u0.3__signed",
                    feature_set,
                    target,
                    True,
                    0.30,
                    True,
                )
            )
    return out


def fit_weights(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    cols: list[int],
    spec: Spec,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    xx = scrub(x[mask][:, cols].astype(np.float64, copy=False))
    yy = y[mask].astype(np.float64, copy=False)
    good = np.isfinite(yy)
    xx = xx[good]
    yy = yy[good]
    mean = np.zeros(xx.shape[1], dtype=np.float64)
    scale = np.ones(xx.shape[1], dtype=np.float64)
    if spec.standardize:
        mean = xx.mean(axis=0)
        scale = np.maximum(xx.std(axis=0), 1e-9)
        xx = (xx - mean) / scale
    gram = xx.T @ xx
    cov = xx.T @ yy
    yty = float(yy @ yy)
    lower = np.full(len(cols), -0.15 if spec.signed else 0.0, dtype=np.float64)
    upper = np.full(len(cols), spec.upper, dtype=np.float64)
    w, ic = fit_ic_weights_from_stats(cov, gram, yty, lower, upper)
    return w, float(ic), mean, scale


def apply_weights(x: np.ndarray, mask: np.ndarray, cols: list[int], w: np.ndarray, mean: np.ndarray, scale: np.ndarray, standardize: bool) -> np.ndarray:
    xx = scrub(x[mask][:, cols].astype(np.float32, copy=False))
    if standardize:
        mean32 = mean.astype(np.float32, copy=False)
        scale32 = scale.astype(np.float32, copy=False)
        xx = (xx - mean32) / scale32
    return xx @ w.astype(np.float32)


def family_weight_summary(names: list[str], families: list[str], cols: list[int], w: np.ndarray) -> dict[str, float]:
    out = {}
    for family in sorted(set(families)):
        vals = [abs(float(w[j])) for j, i in enumerate(cols) if families[i] == family]
        out[f"absw_{family}"] = float(sum(vals))
    nz = [names[i] for j, i in enumerate(cols) if abs(float(w[j])) > 1e-6]
    out["nonzero_components"] = "|".join(nz[:40])
    return out


def add_pred_xsz(pred: pd.DataFrame) -> pd.DataFrame:
    g = pred.groupby("datetime", sort=False)["pred"]
    mean = g.transform("mean")
    std = g.transform("std").replace(0.0, np.nan)
    pred["pred_xsz"] = ((pred["pred"] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    return pred


def summarize(pred: pd.DataFrame, model: str) -> dict[str, object]:
    monthly = period_ic(pred, "pred", "M")
    return {
        "model": model,
        "rows": int(len(pred)),
        "label_rows": int(pred["label"].notna().sum()),
        "pred_ic_2020": compute_ic(pred["pred"], pred["label"]),
        "pred_xsz_ic_2020": compute_ic(pred["pred_xsz"], pred["label"]) if "pred_xsz" in pred else float("nan"),
        "pred_xrank_ic_2020": compute_ic(pred["pred_xrank"], pred["label"]) if "pred_xrank" in pred else float("nan"),
        "monthly_mean": float(monthly.mean()),
        "monthly_ir": float(monthly.mean() / monthly.std()) if monthly.std() > 0 else float("nan"),
    }


def main() -> None:
    base, x, names, families, feature_sets = build_design()
    fit_mask = (base["datetime"] < FIT_END).to_numpy()
    val_mask = ((base["datetime"] >= FIT_END) & (base["datetime"] < TEST_START)).to_numpy()
    q4_mask = (base["datetime"] < TEST_START).to_numpy()
    test_mask = ((base["datetime"] >= TEST_START) & (base["datetime"] < TEST_END)).to_numpy()

    spec_by_name = {spec.name: spec for spec in specs()}
    val_rows = []
    for spec in spec_by_name.values():
        cols = feature_sets[spec.feature_set]
        if not cols:
            continue
        y = target_values(base, spec.target)
        w, fit_ic, mean, scale = fit_weights(x, y, fit_mask, cols, spec)
        val_pred = apply_weights(x, val_mask, cols, w, mean, scale, spec.standardize)
        val_raw = compute_ic(val_pred, base.loc[val_mask, "label"].to_numpy(np.float64))
        val_xsz = compute_ic(val_pred, base.loc[val_mask, "label_xsz_fit"].to_numpy(np.float64))
        val_rows.append(
            {
                "spec": spec.name,
                "feature_set": spec.feature_set,
                "target": spec.target,
                "standardize": spec.standardize,
                "upper": spec.upper,
                "signed": spec.signed,
                "n_features": len(cols),
                "fit_ic_2019oct_nov": fit_ic,
                "val_raw_ic_2019dec": val_raw,
                "val_xsz_ic_2019dec": val_xsz,
                **family_weight_summary(names, families, cols, w),
            }
        )

    validation = pd.DataFrame(val_rows).sort_values("val_raw_ic_2019dec", ascending=False).reset_index(drop=True)
    validation.to_csv(OUT_DIR / "validation_grid.csv", index=False)
    selected_specs: list[str] = []
    for col, n_top in [("val_raw_ic_2019dec", 12), ("val_xsz_ic_2019dec", 6)]:
        for name in validation.sort_values(col, ascending=False)["spec"].head(n_top):
            if name not in selected_specs:
                selected_specs.append(str(name))
    selected_specs = selected_specs[:6]
    selected_by_val_spec = selected_specs[0]

    rows = []
    weight_rows = []
    selected_pred: pd.DataFrame | None = None
    validation_meta = validation.set_index("spec")
    for spec_name in selected_specs:
        spec = spec_by_name[spec_name]
        cols = feature_sets[spec.feature_set]
        y = target_values(base, spec.target)
        w_final, final_fit_ic, mean_final, scale_final = fit_weights(x, y, q4_mask, cols, spec)
        test_pred = apply_weights(x, test_mask, cols, w_final, mean_final, scale_final, spec.standardize)
        pred = base.loc[test_mask, ["symbol", "datetime", "label"]].copy().reset_index(drop=True)
        pred["pred"] = test_pred.astype(np.float32)
        pred = add_pred_xsz(pred)
        model = f"pure_newfactor_augq4__{spec.name}"
        row = summarize(pred, model)
        row.update(
            {
                "feature_set": spec.feature_set,
                "target": spec.target,
                "standardize": spec.standardize,
                "upper": spec.upper,
                "signed": spec.signed,
                "n_features": len(cols),
                "fit_ic_2019oct_nov": float(validation_meta.loc[spec.name, "fit_ic_2019oct_nov"]),
                "val_raw_ic_2019dec": float(validation_meta.loc[spec.name, "val_raw_ic_2019dec"]),
                "val_xsz_ic_2019dec": float(validation_meta.loc[spec.name, "val_xsz_ic_2019dec"]),
                "final_fit_ic_2019q4": final_fit_ic,
                **family_weight_summary(names, families, cols, w_final),
            }
        )
        rows.append(row)
        for local, idx in enumerate(cols):
            weight_rows.append({"model": model, "feature": names[idx], "family": families[idx], "weight": float(w_final[local])})
        if spec.name == selected_by_val_spec:
            selected_pred = pred

    summary = pd.DataFrame(rows).sort_values("val_raw_ic_2019dec", ascending=False).reset_index(drop=True)
    summary["selected_by_val_raw"] = False
    summary.loc[0, "selected_by_val_raw"] = True
    summary["diagnostic_best_2020_raw"] = False
    summary.loc[summary["pred_ic_2020"].idxmax(), "diagnostic_best_2020_raw"] = True
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUT_DIR / "weights.csv", index=False)

    selected_model = str(summary.loc[0, "model"])
    if selected_pred is None:
        raise RuntimeError("selected prediction was not produced")
    selected_pred.to_parquet(OUT_DIR / "selected_by_2019dec.parquet", index=False)
    period_ic(selected_pred, "pred", "M").to_csv(OUT_DIR / "selected_monthly_ic.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    period_ic(selected_pred, "pred", "M").plot(kind="bar", ax=ax)
    ax.axhline(0.05549757798302793, color="#b23a48", linestyle="--", linewidth=1.0, label="old9 raw IC")
    ax.set_title("Pure new-factor augmented Q4 stack selected by 2019-12")
    ax.set_ylabel("Raw IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "selected_monthly_ic.png", dpi=160)
    plt.close(fig)

    print(summary.head(12).to_string(index=False), flush=True)
    print("[selected]", selected_model, flush=True)


if __name__ == "__main__":
    main()
