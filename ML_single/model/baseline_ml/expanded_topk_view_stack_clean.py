#!/usr/bin/env python3
"""Top-K new-factor aware 2019-only expanded view stack.

This keeps the clean train-before-test protocol from
``expanded_gate_view_stack.py`` but appends the FU top-K Ridge/LGB/MLP and
top-K low-correlation residual streams before searching first-level gates.

No 2020 labels are used for component selection, gate fitting, stack fitting,
or hyperparameter selection.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import expanded_history_gate_clean as eh  # noqa: E402
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic  # noqa: E402


ROOT = Path("/root/autodl-tmp/quant/ML")
STRICT_OUT = ROOT / "strict_opt_results"
OUT_DIR = STRICT_OUT / "expanded_topk_view_stack_clean"
TOPK_DIR = STRICT_OUT / "fu_newfactor_topk_best_2019q4"
OUT_PREFIX = OUT_DIR / "view_stack"

TRAIN_START = pd.Timestamp("2019-01-01")
VAL_START = pd.Timestamp("2019-10-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")


@dataclass(frozen=True)
class StackSpec:
    name: str
    modes: tuple[str, ...]
    views: tuple[str, ...]
    top_n: int
    target: str
    standardize: bool
    upper: float


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def month_range(start: str = "2019-01", end: str = "2020-12") -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def align_to_base(base: pd.DataFrame, df: pd.DataFrame, col: str) -> np.ndarray | None:
    ref = base[["symbol", "datetime"]].copy()
    cur = ref.merge(df[["symbol", "datetime", col]], on=["symbol", "datetime"], how="left")
    vals = scrub(cur[col].to_numpy(np.float32, copy=False)).astype(np.float32, copy=False)
    if float(np.nanstd(vals)) <= 1e-12:
        return None
    return vals


def read_topk_parts(part_name: str, view: str, out_name: str) -> pd.DataFrame:
    pieces = []
    part_dir = TOPK_DIR / "prediction_parts" / part_name
    for month in month_range():
        path = part_dir / f"{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        cur = pd.read_parquet(path, columns=["symbol", "datetime", view])
        cur["datetime"] = pd.to_datetime(cur["datetime"])
        pieces.append(cur.rename(columns={view: out_name}))
    return pd.concat(pieces, ignore_index=True)


def append_topk_components(
    base: pd.DataFrame,
    x: np.ndarray,
    names: list[str],
    families: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    cols: list[np.ndarray] = []
    new_names: list[str] = []
    new_families: list[str] = []
    logs: list[str] = []

    for part_name, family in [("rolling_ridge", "topk_ridge"), ("rolling_lgb", "topk_lgb"), ("rolling_mlp", "topk_mlp")]:
        for view in ["pred", "pred_xsz", "pred_xrank"]:
            out_name = f"topk__{part_name}__{view}"
            try:
                df = read_topk_parts(part_name, view, out_name)
                vals = align_to_base(base, df, out_name)
                if vals is None:
                    logs.append(f"skip {out_name}: degenerate")
                    continue
                cols.append(vals)
                new_names.append(out_name)
                new_families.append(family)
                logs.append(f"loaded {out_name} family={family}")
            except Exception as exc:  # noqa: BLE001
                logs.append(f"skip {out_name}: {exc}")

    lowcorr_path = TOPK_DIR / "lowcorr_anchor_residual" / "topk_anchor_lowcorr_lgb_meta_chain_xsz.parquet"
    if lowcorr_path.exists():
        for col in ["anchor_pred_xsz", "resid_pred", "combined_pred"]:
            out_name = f"topk__lowcorr_lgb_meta_chain_xsz__{col}"
            try:
                df = pd.read_parquet(lowcorr_path, columns=["symbol", "datetime", col])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.rename(columns={col: out_name})
                vals = align_to_base(base, df, out_name)
                if vals is None:
                    logs.append(f"skip {out_name}: degenerate")
                    continue
                cols.append(vals)
                new_names.append(out_name)
                new_families.append("topk_lowcorr_lgb")
                logs.append(f"loaded {out_name} family=topk_lowcorr_lgb")
            except Exception as exc:  # noqa: BLE001
                logs.append(f"skip {out_name}: {exc}")

    if cols:
        x = np.column_stack([x] + cols).astype(np.float32, copy=False)
        names = names + new_names
        families = families + new_families
    (OUT_DIR / "topk_load_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    return x, names, families


def finalize_matrix_topk() -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old_out = eh.OUT_DIR
    try:
        eh.OUT_DIR = OUT_DIR
        base, x, names, families = eh.finalize_matrix()
    finally:
        eh.OUT_DIR = old_out
    x, names, families = append_topk_components(base, x, names, families)
    pd.DataFrame({"component": names, "family": families}).to_csv(OUT_DIR / "components.csv", index=False)
    return base, x, names, families


def component_ic_table(base: pd.DataFrame, x: np.ndarray, names: list[str], families: list[str]) -> pd.DataFrame:
    dt = base["datetime"]
    val_mask = ((dt >= VAL_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    y = base["label"].to_numpy(np.float64)
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "idx": i,
                "component": name,
                "family": families[i],
                "val_ic_2019q4": compute_ic(x[val_mask, i], y[val_mask]),
            }
        )
    out = pd.DataFrame(rows).sort_values("val_ic_2019q4", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_DIR / "component_ic_2019q4.csv", index=False)
    return out


def configs_topk(names: list[str], families: list[str], comp_ic: pd.DataFrame) -> list[eh.GateConfig]:
    by_family: dict[str, list[str]] = {}
    for name, fam in zip(names, families):
        by_family.setdefault(fam, []).append(name)

    old = by_family.get("old_clean", [])
    old9 = by_family.get("old9", [])
    original_new = [n for n in names if n.startswith("new1617_shuffle_rolling_")]
    topk_core = [
        n
        for n in [
            "topk__rolling_ridge__pred_xsz",
            "topk__rolling_lgb__pred_xsz",
            "topk__rolling_mlp__pred_xsz",
            "topk__lowcorr_lgb_meta_chain_xsz__combined_pred",
        ]
        if n in names
    ]
    topk_all = [n for n, fam in zip(names, families) if fam.startswith("topk_")]
    top_val24 = [n for n in comp_ic[~comp_ic["family"].isin(["old_clean"])]["component"].head(24).astype(str).tolist() if n in names]
    topk_val12 = [n for n in comp_ic[comp_ic["family"].str.startswith("topk_")]["component"].head(12).astype(str).tolist() if n in names]

    pools = {
        "old_topk_core": list(dict.fromkeys(old + topk_core)),
        "old_old9_topk_core": list(dict.fromkeys(old + old9 + topk_core)),
        "old_original_topk_core": list(dict.fromkeys(old + original_new + topk_core)),
        "old_old9_original_topk_core": list(dict.fromkeys(old + old9 + original_new + topk_core)),
        "old_topk_val12": list(dict.fromkeys(old + topk_val12)),
        "old_old9_topk_val12": list(dict.fromkeys(old + old9 + topk_val12)),
        "old_top_val24": list(dict.fromkeys(old + top_val24)),
        "old_old9_top_val24": list(dict.fromkeys(old + old9 + top_val24)),
        "old_old9_topk_all": list(dict.fromkeys(old + old9 + topk_all)),
    }

    out: list[eh.GateConfig] = []
    out.extend(eh.configs(names, families))
    for pool_name, comps in pools.items():
        if not comps:
            continue
        for target in ["raw", "xsz"]:
            for scheme in ["row", "month_equal", "month_decay3", "month_decay6"]:
                out.append(eh.GateConfig(f"{pool_name}_{target}_{scheme}_u090", comps, target=target, scheme=scheme, signed=False, upper=0.90))
                if pool_name in {"old_old9_topk_core", "old_old9_topk_val12", "old_old9_top_val24"} and scheme in {"row", "month_equal"}:
                    out.append(
                        eh.GateConfig(
                            f"{pool_name}_{target}_{scheme}_signed03_u090",
                            comps,
                            target=target,
                            scheme=scheme,
                            signed=True,
                            lower=-0.03,
                            upper=0.90,
                        )
                    )
    dedup: dict[str, eh.GateConfig] = {}
    for cfg in out:
        dedup[cfg.name] = cfg
    return list(dedup.values())


def validate_first_level(base: pd.DataFrame, x: np.ndarray, names: list[str], configs: list[eh.GateConfig]) -> pd.DataFrame:
    name_to_idx = {n: i for i, n in enumerate(names)}
    dt = base["datetime"]
    train_mask = eh.mask_between(dt, TRAIN_START, VAL_START, base["label"])
    val_mask = eh.mask_between(dt, VAL_START, TEST_START, base["label"])
    rows = []
    print(f"[topk-view][validate] components={len(names)} configs={len(configs)}", flush=True)
    for pos, cfg in enumerate(configs, start=1):
        cols = [name_to_idx[c] for c in cfg.components if c in name_to_idx]
        if not cols:
            continue
        w, train_ic = eh.fit_weights(base, x, cols, train_mask, cfg)
        pred = eh.predict_frame(base, x, cols, w, val_mask)
        val_ic = compute_ic(pred["pred"], pred["label"])
        rows.append(
            {
                "model": cfg.name,
                "val_ic_2019q4": val_ic,
                "train_ic_2019q1q3": train_ic,
                "k": len(cols),
                "target": cfg.target,
                "scheme": cfg.scheme,
                "signed": cfg.signed,
                "upper": cfg.upper,
            }
        )
        print(f"[topk-view][val {pos:03d}/{len(configs):03d}] {cfg.name} k={len(cols)} val={val_ic:.6f}", flush=True)
    grid = pd.DataFrame(rows).sort_values("val_ic_2019q4", ascending=False)
    grid.to_csv(OUT_DIR / "validation_grid.csv", index=False)
    return grid


def selected_config_names(grid: pd.DataFrame) -> list[str]:
    nonneg = grid[grid["signed"] == False].sort_values("val_ic_2019q4", ascending=False)  # noqa: E712
    selected = list(dict.fromkeys(nonneg.head(10)["model"].astype(str).tolist()))
    topk = nonneg[nonneg["model"].astype(str).str.contains("topk")].head(4)["model"].astype(str).tolist()
    for name in topk:
        if name not in selected:
            selected.append(name)
    for must in [
        "old_family_selected_raw_month_equal_u090",
        "old_family_selected_raw_row_u090",
        "old_old9_topk_core_raw_month_equal_u090",
        "old_old9_topk_core_xsz_month_equal_u090",
        "old_old9_topk_val12_raw_month_equal_u090",
        "old_old9_top_val24_raw_month_equal_u090",
    ]:
        if must in set(grid["model"].astype(str)) and must not in selected:
            selected.append(must)
    return selected


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS"))


def rolling_predict_period(base: pd.DataFrame, x: np.ndarray, cols: list[int], cfg: eh.GateConfig, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dt = base["datetime"]
    pieces = []
    for ms in month_starts(start, end):
        train_mask = ((dt >= TRAIN_START) & (dt < ms) & base["label"].notna()).to_numpy()
        pred_mask = ((dt >= ms) & (dt < ms + pd.DateOffset(months=1))).to_numpy()
        w, train_ic = eh.fit_weights(base, x, cols, train_mask, cfg)
        pred = eh.predict_frame(base, x, cols, w, pred_mask)
        print(f"[topk-view][rolling] {cfg.name} {ms:%Y-%m} train_ic={train_ic:.6f}", flush=True)
        pieces.append(pred)
    return pd.concat(pieces, ignore_index=True)


def add_design_columns(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    names: list[str],
    mode: str,
    cfg_name: str,
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
) -> None:
    for view in ["pred", "pred_xsz", "pred_xrank"]:
        name = f"{mode}::{cfg_name}::{view}"
        arrays[name] = (
            val_pred[view].to_numpy(np.float32, copy=False),
            test_pred[view].to_numpy(np.float32, copy=False),
        )
        names.append(name)


def build_base_predictions(
    base: pd.DataFrame,
    x: np.ndarray,
    names: list[str],
    cfg_by_name: dict[str, eh.GateConfig],
    selected: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    name_to_idx = {n: i for i, n in enumerate(names)}
    dt = base["datetime"]
    train_mask = eh.mask_between(dt, TRAIN_START, VAL_START, base["label"])
    full_train_mask = eh.mask_between(dt, TRAIN_START, TEST_START, base["label"])
    val_mask = ((dt >= VAL_START) & (dt < TEST_START)).to_numpy()
    test_mask = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()

    val_base = base.loc[val_mask, ["symbol", "datetime", "label", "label_xsz_fit"]].copy().reset_index(drop=True)
    test_base = base.loc[test_mask, ["symbol", "datetime", "label", "label_xsz_fit"]].copy().reset_index(drop=True)

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feature_names: list[str] = []
    rows = []
    for pos, cfg_name in enumerate(selected, start=1):
        cfg = cfg_by_name[cfg_name]
        cols = [name_to_idx[c] for c in cfg.components if c in name_to_idx]
        if not cols:
            continue
        w_val, train_ic = eh.fit_weights(base, x, cols, train_mask, cfg)
        val_fixed = eh.predict_frame(base, x, cols, w_val, val_mask).reset_index(drop=True)
        w_final, final_train_ic = eh.fit_weights(base, x, cols, full_train_mask, cfg)
        test_fixed = eh.predict_frame(base, x, cols, w_final, test_mask).reset_index(drop=True)
        add_design_columns(arrays, feature_names, "fixed", cfg_name, val_fixed, test_fixed)

        val_roll = rolling_predict_period(base, x, cols, cfg, VAL_START, TEST_START).reset_index(drop=True)
        test_roll = rolling_predict_period(base, x, cols, cfg, TEST_START, TEST_END).reset_index(drop=True)
        add_design_columns(arrays, feature_names, "rolling", cfg_name, val_roll, test_roll)

        rows.append(
            {
                "rank": pos,
                "config": cfg_name,
                "k": len(cols),
                "fixed_train_ic_q1q3": float(train_ic),
                "fixed_train_ic_2019": float(final_train_ic),
            }
        )
        print(f"[topk-view][base {pos:02d}/{len(selected):02d}] {cfg_name}", flush=True)

    pd.DataFrame(rows).to_csv(f"{OUT_PREFIX}_base_configs.csv", index=False)
    pd.DataFrame({"feature": feature_names}).to_csv(f"{OUT_PREFIX}_features.csv", index=False)
    return val_base, test_base, arrays, pd.DataFrame(rows)


def stack_specs(n_configs: int) -> list[StackSpec]:
    specs: list[StackSpec] = []
    for modes in [("fixed",), ("rolling",), ("fixed", "rolling")]:
        mode_name = "+".join(modes)
        for views in [("pred",), ("pred_xsz",), ("pred", "pred_xsz"), ("pred", "pred_xsz", "pred_xrank")]:
            view_name = "+".join(views)
            for top_n in [4, 6, 8, 10, n_configs]:
                if top_n > n_configs:
                    continue
                for target in ["raw", "xsz"]:
                    for standardize in [False, True]:
                        for upper in [0.25, 0.35, 0.60, 1.00]:
                            specs.append(
                                StackSpec(
                                    name=f"{mode_name}__{view_name}__top{top_n}__{target}__std{int(standardize)}__u{upper:g}",
                                    modes=modes,
                                    views=views,
                                    top_n=top_n,
                                    target=target,
                                    standardize=standardize,
                                    upper=upper,
                                )
                            )
    dedup: dict[str, StackSpec] = {}
    for spec in specs:
        dedup[spec.name] = spec
    return list(dedup.values())


def choose_columns(all_names: list[str], spec: StackSpec, base_order: list[str]) -> list[int]:
    keep_configs = set(base_order[: spec.top_n])
    cols = []
    for i, name in enumerate(all_names):
        mode, cfg, view = name.split("::", 2)
        if mode in spec.modes and cfg in keep_configs and view in spec.views:
            cols.append(i)
    return cols


def fit_stack(xv: np.ndarray, y: np.ndarray, cols: list[int], spec: StackSpec) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    x = scrub(xv[:, cols].astype(np.float64, copy=False))
    mean = np.zeros(x.shape[1], dtype=np.float64)
    scale = np.ones(x.shape[1], dtype=np.float64)
    if spec.standardize:
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-9)
        x = (x - mean) / scale
    good = np.isfinite(y)
    x = x[good]
    yy = y[good]
    gram = x.T @ x
    cov = x.T @ yy
    yty = float(yy @ yy)
    lower = np.zeros(len(cols), dtype=np.float64)
    upper = np.full(len(cols), spec.upper, dtype=np.float64)
    w, ic = fit_ic_weights_from_stats(cov, gram, yty, lower, upper)
    return w, float(ic), mean, scale


def apply_stack(xt: np.ndarray, cols: list[int], w: np.ndarray, mean: np.ndarray, scale: np.ndarray, standardize: bool) -> np.ndarray:
    x = scrub(xt[:, cols].astype(np.float32, copy=False))
    if standardize:
        x = ((x.astype(np.float64) - mean) / scale).astype(np.float32)
    return x @ w.astype(np.float32)


def summarize_2020(pred: pd.DataFrame, model: str) -> dict[str, object]:
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


def run_stack(
    val_base: pd.DataFrame,
    test_base: pd.DataFrame,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    base_configs: pd.DataFrame,
) -> pd.DataFrame:
    feature_names = list(arrays)
    xv = np.column_stack([arrays[n][0] for n in feature_names]).astype(np.float32)
    xt = np.column_stack([arrays[n][1] for n in feature_names]).astype(np.float32)
    y_raw = val_base["label"].to_numpy(np.float64, copy=False)
    y_xsz = val_base["label_xsz_fit"].to_numpy(np.float64, copy=False)
    base_order = base_configs["config"].astype(str).tolist()

    rows = []
    weight_rows = []
    best_val_raw: tuple[float, str, pd.DataFrame] | None = None
    best_test_raw: tuple[float, str, pd.DataFrame] | None = None
    for spec in stack_specs(len(base_order)):
        cols = choose_columns(feature_names, spec, base_order)
        if not cols:
            continue
        y = y_xsz if spec.target == "xsz" else y_raw
        w, val_ic, mean, scale = fit_stack(xv, y, cols, spec)
        val_pred = apply_stack(xv, cols, w, mean, scale, spec.standardize)
        pred = apply_stack(xt, cols, w, mean, scale, spec.standardize)
        out = test_base[["symbol", "datetime", "label"]].copy()
        out["pred"] = pred.astype(np.float32)
        out = add_cross_sectional_norms(out, "pred")
        row = summarize_2020(out, f"expanded_topk_view_stack__{spec.name}")
        val_raw_ic = compute_ic(val_pred, val_base["label"].to_numpy(np.float64, copy=False))
        val_xsz_ic = compute_ic(val_pred, val_base["label_xsz_fit"].to_numpy(np.float64, copy=False))
        row.update(
            {
                "stack_fit_ic_2019q4": val_ic,
                "stack_val_raw_ic_2019q4": val_raw_ic,
                "stack_val_xsz_ic_2019q4": val_xsz_ic,
                "n_features": len(cols),
                "modes": "+".join(spec.modes),
                "views": "+".join(spec.views),
                "top_n": spec.top_n,
                "target": spec.target,
                "standardize": spec.standardize,
                "upper": spec.upper,
            }
        )
        rows.append(row)
        if best_val_raw is None or val_raw_ic > best_val_raw[0]:
            best_val_raw = (float(val_raw_ic), str(row["model"]), out)
        test_raw_ic = float(row["pred_ic_2020"])
        if best_test_raw is None or test_raw_ic > best_test_raw[0]:
            best_test_raw = (test_raw_ic, str(row["model"]), out)
        for feature, weight in zip([feature_names[i] for i in cols], w):
            if abs(float(weight)) > 1e-8:
                weight_rows.append({"model": row["model"], "feature": feature, "weight": float(weight)})
        print(
            f"[topk-view][stack] {row['model']} "
            f"fit={val_ic:.6f} val_raw={val_raw_ic:.6f} test={row['pred_ic_2020']:.6f}",
            flush=True,
        )

    summary = pd.DataFrame(rows).sort_values("stack_val_raw_ic_2019q4", ascending=False)
    summary["selected_by_2019q4_raw"] = summary["model"] == summary.iloc[0]["model"]
    if best_test_raw is not None:
        summary["diagnostic_best_2020_raw"] = summary["model"] == best_test_raw[1]
    summary.to_csv(f"{OUT_PREFIX}_summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(f"{OUT_PREFIX}_weights.csv", index=False)
    if best_val_raw is not None:
        _, best_model, pred = best_val_raw
        pred.to_parquet(f"{OUT_PREFIX}_selected_by_val_raw.parquet", index=False)
        monthly = pred.assign(month=pred["datetime"].dt.to_period("M").astype(str)).groupby("month").apply(
            lambda g: pd.Series(
                {
                    "pred_ic": compute_ic(g["pred"], g["label"]),
                    "pred_xsz_ic": compute_ic(g["pred_xsz"], g["label"]),
                    "pred_xrank_ic": compute_ic(g["pred_xrank"], g["label"]),
                }
            ),
            include_groups=False,
        )
        monthly.reset_index().to_csv(f"{OUT_PREFIX}_selected_by_val_raw_monthly_ic.csv", index=False)
    if best_test_raw is not None:
        _, _, pred = best_test_raw
        pred.to_parquet(f"{OUT_PREFIX}_diagnostic_best_2020_raw.parquet", index=False)
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base, x, names, families = finalize_matrix_topk()
    comp_ic_path = OUT_DIR / "component_ic_2019q4.csv"
    comp_ic = pd.read_csv(comp_ic_path) if comp_ic_path.exists() else component_ic_table(base, x, names, families)
    all_configs = configs_topk(names, families, comp_ic)
    cfg_by_name = {cfg.name: cfg for cfg in all_configs}
    grid_path = OUT_DIR / "validation_grid.csv"
    selected_path = OUT_DIR / "selected_base_config_names.csv"
    if grid_path.exists() and selected_path.exists():
        grid = pd.read_csv(grid_path)
        selected = pd.read_csv(selected_path)["config"].astype(str).tolist()
        print(f"[topk-view] reuse validation grid rows={len(grid)} selected={len(selected)}", flush=True)
    else:
        grid = validate_first_level(base, x, names, all_configs)
        selected = selected_config_names(grid)
        pd.DataFrame({"rank": range(1, len(selected) + 1), "config": selected}).to_csv(selected_path, index=False)
    print("[topk-view] selected base configs:", ", ".join(selected), flush=True)
    val_base, test_base, arrays, base_configs = build_base_predictions(base, x, names, cfg_by_name, selected)
    summary = run_stack(val_base, test_base, arrays, base_configs)
    print(summary.head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
