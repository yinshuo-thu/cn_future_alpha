#!/usr/bin/env python3
"""Expanded no-2020-label history gate over clean old and new predictions.

Selection protocol:
  - Build only train-before-test prediction streams.
  - Fit candidate gates on 2019Q1-Q3.
  - Select the gate configuration by 2019Q4 IC.
  - Refit the selected configuration on all 2019 and evaluate 2020.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

from fu_newfactor_family_ensemble import build_matrix, collect_candidates
from minimal_diverse_ensemble import collect_specs as collect_old9_specs
from rolling_factor_model_eval import (
    TEST_END,
    TEST_START,
    add_candidate,
    add_cross_sectional_norms,
    candidate_specs,
    compute_ic,
    fit_ic_weights_from_stats,
    load_base,
    period_ic,
)


ROOT = Path("/root/autodl-tmp/quant/ML")
OUT_DIR = ROOT / "strict_opt_results" / "expanded_history_gate_clean"
FU_DIR = ROOT / "strict_opt_results" / "fu_newfactor_three_model"

TRAIN_START = pd.Timestamp("2019-01-01")
VAL_START = pd.Timestamp("2019-10-01")

CLEAN_OLD = [
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
]


@dataclass(frozen=True)
class GateConfig:
    name: str
    components: list[str]
    target: str = "raw"
    scheme: str = "row"
    signed: bool = False
    lower: float = 0.0
    upper: float = 0.90


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def add_xsz_label(base: pd.DataFrame) -> pd.DataFrame:
    g = base.groupby("datetime", sort=False)["label"]
    base["label_xsz_fit"] = ((base["label"] - g.transform("mean")) / (g.transform("std") + 1e-9)).clip(-8, 8).astype(np.float32)
    return base


def base_from_family() -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    candidates = collect_candidates()
    base, x, names, families, logs = build_matrix(candidates)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "family_load_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    return base, x, names, families


def append_old_clean(base: pd.DataFrame, names: list[str]) -> tuple[list[np.ndarray], list[str], list[str], list[str]]:
    tmp = base[["symbol", "datetime", "label"]].copy()
    spec_by_name = {s.name: s for s in candidate_specs()}
    logs: list[str] = []
    for name in CLEAN_OLD:
        spec = spec_by_name.get(name)
        if spec is None:
            logs.append(f"missing spec {name}")
            continue
        tmp, err = add_candidate(tmp, spec)
        if err:
            logs.append(err)
    cols: list[np.ndarray] = []
    out_names: list[str] = []
    families: list[str] = []
    for name in CLEAN_OLD:
        if name not in tmp.columns:
            continue
        out_name = f"oldclean__{name}"
        if out_name in names:
            continue
        cols.append(scrub(tmp[name].to_numpy(np.float32, copy=False)).astype(np.float32, copy=False))
        out_names.append(out_name)
        families.append("old_clean")
    return cols, out_names, families, logs


def append_old9(base: pd.DataFrame, names: list[str]) -> tuple[list[np.ndarray], list[str], list[str], list[str]]:
    selected_path = ROOT / "strict_opt_results" / "minimal_diverse_ensemble" / "selected_components.csv"
    selected = pd.read_csv(selected_path)["component"].astype(str).tolist()
    spec_by_name = {s.name: s for s in collect_old9_specs()}
    cols: list[np.ndarray] = []
    out_names: list[str] = []
    families: list[str] = []
    logs: list[str] = []
    ref = base[["symbol", "datetime"]].copy()
    for name in selected:
        spec = spec_by_name.get(name)
        if spec is None:
            logs.append(f"missing old9 spec {name}")
            continue
        out_name = f"old9__{name}"
        if out_name in names:
            continue
        df = pd.read_parquet(spec.path, columns=["symbol", "datetime", spec.col])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[(df["datetime"] >= TRAIN_START) & (df["datetime"] < TEST_END)].copy()
        cur = ref.merge(df[["symbol", "datetime", spec.col]], on=["symbol", "datetime"], how="left")
        vals = scrub(cur[spec.col].to_numpy(np.float32, copy=False)).astype(np.float32, copy=False)
        cols.append(vals)
        out_names.append(out_name)
        families.append("old9")
        logs.append(f"loaded old9 {name}")
    return cols, out_names, families, logs


def finalize_matrix() -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    base, x, names, families = base_from_family()
    base = base[(base["datetime"] >= TRAIN_START) & (base["datetime"] < TEST_END)].copy().reset_index(drop=True)
    base = add_xsz_label(base)

    extra_cols: list[np.ndarray] = []
    extra_names: list[str] = []
    extra_families: list[str] = []
    logs: list[str] = []
    for loader in [append_old_clean, append_old9]:
        cols, new_names, new_families, new_logs = loader(base, names + extra_names)
        extra_cols.extend(cols)
        extra_names.extend(new_names)
        extra_families.extend(new_families)
        logs.extend(new_logs)
    if extra_cols:
        x = np.column_stack([x] + extra_cols).astype(np.float32, copy=False)
        names = names + extra_names
        families = families + extra_families
    (OUT_DIR / "extra_load_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    pd.DataFrame({"component": names, "family": families}).to_csv(OUT_DIR / "components.csv", index=False)
    return base, x, names, families


def mask_between(dt: pd.Series, start: pd.Timestamp, end: pd.Timestamp, label: pd.Series) -> np.ndarray:
    return ((dt >= start) & (dt < end) & label.notna()).to_numpy()


def weighted_stats(
    base: pd.DataFrame,
    x: np.ndarray,
    cols: list[int],
    mask: np.ndarray,
    target: str,
    scheme: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    y_col = "label_xsz_fit" if target == "xsz" else "label"
    y_all = base[y_col].to_numpy(np.float64)
    idx = np.asarray(cols, dtype=np.int32)
    if scheme == "row":
        xm = scrub(x[mask][:, idx]).astype(np.float64, copy=False)
        ym = y_all[mask].astype(np.float64, copy=False)
        good = np.isfinite(ym)
        xm = xm[good]
        ym = ym[good]
        return xm.T @ xm, xm.T @ ym, float(ym @ ym)
    if scheme not in {"month_equal", "month_decay3", "month_decay6"}:
        raise ValueError(f"bad scheme: {scheme}")

    gram = np.zeros((len(cols), len(cols)), dtype=np.float64)
    cov = np.zeros(len(cols), dtype=np.float64)
    yty = 0.0
    month_all = base["datetime"].dt.to_period("M").astype(str).to_numpy()
    months = np.unique(month_all[mask])
    if scheme.startswith("month_decay"):
        half_life = float(scheme.replace("month_decay", ""))
        periods = [pd.Period(m, freq="M") for m in months]
        last = max(periods)
        month_weights = {
            str(p): 0.5 ** ((last.ordinal - p.ordinal) / max(half_life, 1e-6))
            for p in periods
        }
    else:
        month_weights = {str(m): 1.0 for m in months}
    for month in months:
        m = mask & (month_all == month)
        xm = scrub(x[m][:, idx]).astype(np.float64, copy=False)
        ym = y_all[m].astype(np.float64, copy=False)
        good = np.isfinite(ym)
        xm = xm[good]
        ym = ym[good]
        if len(ym) == 0:
            continue
        weight = float(month_weights[str(month)]) / float(len(ym))
        gram += weight * (xm.T @ xm)
        cov += weight * (xm.T @ ym)
        yty += weight * float(ym @ ym)
    return gram, cov, yty


def fit_weights(
    base: pd.DataFrame,
    x: np.ndarray,
    cols: list[int],
    mask: np.ndarray,
    cfg: GateConfig,
) -> tuple[np.ndarray, float]:
    gram, cov, yty = weighted_stats(base, x, cols, mask, cfg.target, cfg.scheme)
    lower = np.full(len(cols), cfg.lower if cfg.signed else 0.0, dtype=np.float64)
    upper = np.full(len(cols), cfg.upper, dtype=np.float64)
    return fit_ic_weights_from_stats(cov, gram, yty, lower, upper)


def predict_frame(base: pd.DataFrame, x: np.ndarray, cols: list[int], w: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    pred = scrub(x[mask][:, cols]) @ w.astype(np.float32)
    out = base.loc[mask, ["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype(np.float32)
    return add_cross_sectional_norms(out, "pred")


def summarize(pred: pd.DataFrame, model: str) -> dict[str, float | str | int]:
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


def configs(names: list[str], families: list[str]) -> list[GateConfig]:
    by_family: dict[str, list[str]] = {}
    for name, fam in zip(names, families):
        by_family.setdefault(fam, []).append(name)

    old = by_family.get("old_clean", [])
    old9 = by_family.get("old9", [])
    newrolling = [n for n in names if n.startswith("new1617_shuffle_rolling_")]

    comp_ic = FU_DIR / "family_ensemble_clean" / "component_ic.csv"
    top_family: list[str] = []
    selected_family: list[str] = []
    if comp_ic.exists():
        df = pd.read_csv(comp_ic).sort_values("val_ic_2019q4", ascending=False)
        top_family = [n for n in df.head(24)["component"].astype(str).tolist() if n in names]
    weights = FU_DIR / "family_ensemble_clean" / "fu_newfactor_family_greedy_nonneg_weights.csv"
    if weights.exists():
        w = pd.read_csv(weights)
        selected_family = [
            n
            for n, val in zip(w["component"].astype(str), w["weight"].astype(float))
            if n in names and abs(val) > 1e-8
        ]

    pools = {
        "old": old,
        "old_newrolling": list(dict.fromkeys(old + newrolling)),
        "old_family_selected": list(dict.fromkeys(old + selected_family)),
        "old_family_top24": list(dict.fromkeys(old + top_family)),
        "old_old9_newrolling": list(dict.fromkeys(old + old9 + newrolling)),
        "old_old9_family_selected": list(dict.fromkeys(old + old9 + selected_family)),
    }

    out: list[GateConfig] = []
    for pool_name, comps in pools.items():
        if not comps:
            continue
        for target in ["raw", "xsz"]:
            for scheme in ["row", "month_equal", "month_decay3", "month_decay6"]:
                out.append(GateConfig(f"{pool_name}_{target}_{scheme}_u090", comps, target=target, scheme=scheme, signed=False, upper=0.90))
                if pool_name in {"old_family_selected", "old_family_top24"} and scheme in {"row", "month_equal", "month_decay6"}:
                    out.append(
                        GateConfig(
                            f"{pool_name}_{target}_{scheme}_signed05_u090",
                            comps,
                            target=target,
                            scheme=scheme,
                            signed=True,
                            lower=-0.05,
                            upper=0.90,
                        )
                    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base, x, names, families = finalize_matrix()
    name_to_idx = {n: i for i, n in enumerate(names)}
    dt = base["datetime"]
    train_mask = mask_between(dt, TRAIN_START, VAL_START, base["label"])
    val_mask = mask_between(dt, VAL_START, TEST_START, base["label"])
    full_train_mask = mask_between(dt, TRAIN_START, TEST_START, base["label"])
    test_mask = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()

    val_rows = []
    cfg_by_name = {}
    all_configs = configs(names, families)
    print(f"[expanded-gate] components={len(names)} configs={len(all_configs)}", flush=True)
    for pos, cfg in enumerate(all_configs, start=1):
        cfg_by_name[cfg.name] = cfg
        cols = [name_to_idx[c] for c in cfg.components if c in name_to_idx]
        if not cols:
            continue
        w_val, train_ic = fit_weights(base, x, cols, train_mask, cfg)
        val_pred = predict_frame(base, x, cols, w_val, val_mask)
        val_ic = compute_ic(val_pred["pred"], val_pred["label"])
        val_rows.append(
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
        print(f"[expanded-gate][val {pos:02d}/{len(all_configs):02d}] {cfg.name} k={len(cols)} val={val_ic:.6f}", flush=True)

    val_df = pd.DataFrame(val_rows).sort_values("val_ic_2019q4", ascending=False)
    val_df.to_csv(OUT_DIR / "validation_grid.csv", index=False)
    eval_models = val_df.head(8)["model"].astype(str).tolist()
    print("[expanded-gate] final-eval models:", ", ".join(eval_models), flush=True)

    rows = []
    weight_rows = []
    for pos, model_name in enumerate(eval_models, start=1):
        cfg = cfg_by_name[model_name]
        cols = [name_to_idx[c] for c in cfg.components if c in name_to_idx]
        if not cols:
            continue
        val_info = val_df[val_df["model"] == cfg.name].iloc[0].to_dict()

        w_final, final_train_ic = fit_weights(base, x, cols, full_train_mask, cfg)
        test_pred = predict_frame(base, x, cols, w_final, test_mask)
        test_summary = summarize(test_pred, cfg.name)
        row = {
            **test_summary,
            "val_ic_2019q4": float(val_info["val_ic_2019q4"]),
            "train_ic_2019q1q3": float(val_info["train_ic_2019q1q3"]),
            "final_train_ic_2019": final_train_ic,
            "k": len(cols),
            "target": cfg.target,
            "scheme": cfg.scheme,
            "signed": cfg.signed,
            "upper": cfg.upper,
        }
        rows.append(row)
        print(
            f"[expanded-gate][test {pos:02d}/{len(eval_models):02d}] {cfg.name} "
            f"val={row['val_ic_2019q4']:.6f} test={row['pred_ic_2020']:.6f}",
            flush=True,
        )
        for comp, weight in zip(cfg.components, w_final):
            if abs(float(weight)) > 1e-8:
                weight_rows.append({"model": cfg.name, "component": comp, "weight": float(weight)})

    summary = pd.DataFrame(rows).sort_values(["val_ic_2019q4", "pred_ic_2020"], ascending=False)
    selected = summary.iloc[0].to_dict()
    summary["selected_by_2019q4"] = summary["model"] == selected["model"]
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUT_DIR / "weights.csv", index=False)
    (OUT_DIR / "selected_by_2019q4.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(summary.head(20).to_string(index=False), flush=True)
    print(f"[selected_by_2019q4] {selected['model']} val={selected['val_ic_2019q4']:.6f} test={selected['pred_ic_2020']:.6f}", flush=True)


if __name__ == "__main__":
    main()
