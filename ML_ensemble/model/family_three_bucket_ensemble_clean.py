#!/usr/bin/env python3
"""2019-only three-family ensemble over clean rolling predictions.

This script intentionally avoids the full-window ``best_ic0716`` artifact and
does not use 2020 labels for component selection or weight fitting.

Pipeline:
  1. Load strict train-before-test component predictions from old clean models
     and the FU top-K new-factor Ridge/LGB/MLP runs.
  2. Select and fit one internal blend per family (ridge/lgb/mlp) using
     2019Q1-Q3 for fitting and 2019Q4 for selection.
  3. Refit each family on all 2019 OOS predictions.
  4. Fit the final three-family ensemble on 2019 and evaluate 2020 only.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

from minimal_diverse_ensemble import collect_specs as collect_old_specs
from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic


ROOT = Path("/root/autodl-tmp/quant/ML")
STRICT_OUT = ROOT / "strict_opt_results"
TOPK_DIR = STRICT_OUT / "fu_newfactor_topk_best_2019q4"
OUT_DIR = STRICT_OUT / "family_three_bucket_ensemble_clean"

TRAIN_START = pd.Timestamp("2019-01-01")
VAL_START = pd.Timestamp("2019-10-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")

OLD_MINIMAL_RAW_IC = 0.05549757798302793
EXPANDED_CLEAN_RAW_IC = 0.05888213437995787


@dataclass(frozen=True)
class LoadedComponent:
    family: str
    name: str
    values: np.ndarray


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def family_for_name(name: str) -> str:
    if name.startswith("mlp_") or "rolling_mlp" in name:
        return "mlp"
    if name.startswith("ridge_") or "lowcorr_ridge" in name or "rolling_ridge" in name:
        return "ridge"
    return "lgb"


def read_component_file(path: Path, col: str, name: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["symbol", "datetime", "label", col])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[(df["datetime"] >= TRAIN_START) & (df["datetime"] < TEST_END)].copy()
    df = df.sort_values(["datetime", "symbol"], kind="mergesort").reset_index(drop=True)
    return df.rename(columns={col: name})


def month_range(start: str = "2019-01", end: str = "2020-12") -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def read_parts(part_name: str, view: str, name: str) -> pd.DataFrame:
    pieces = []
    part_dir = TOPK_DIR / "prediction_parts" / part_name
    for month in month_range():
        path = part_dir / f"{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        cur = pd.read_parquet(path, columns=["symbol", "datetime", "label", view])
        cur["datetime"] = pd.to_datetime(cur["datetime"])
        pieces.append(cur.rename(columns={view: name}))
    df = pd.concat(pieces, ignore_index=True)
    return df.sort_values(["datetime", "symbol"], kind="mergesort").reset_index(drop=True)


def align_values(base: pd.DataFrame, df: pd.DataFrame, name: str) -> np.ndarray | None:
    if len(df) != len(base):
        return None
    if not np.array_equal(df["datetime"].astype("int64").to_numpy(), base["datetime"].astype("int64").to_numpy()):
        return None
    if not np.array_equal(df["symbol"].astype(str).to_numpy(), base["symbol"].astype(str).to_numpy()):
        return None
    vals = scrub(df[name].to_numpy(np.float32, copy=False)).astype(np.float32, copy=False)
    if float(np.nanstd(vals)) <= 1e-12:
        return None
    return vals


def collect_components() -> tuple[pd.DataFrame, list[LoadedComponent], list[str]]:
    logs: list[str] = []
    components: list[LoadedComponent] = []

    old_specs = collect_old_specs()
    if not old_specs:
        raise RuntimeError("no old clean component specs found")

    first = read_component_file(old_specs[0].path, old_specs[0].col, old_specs[0].name)
    base = first[["symbol", "datetime", "label"]].copy()
    first_vals = align_values(base, first, old_specs[0].name)
    if first_vals is None:
        raise RuntimeError("first component is not usable")
    components.append(LoadedComponent(family_for_name(old_specs[0].name), old_specs[0].name, first_vals))
    logs.append(f"loaded old:{old_specs[0].name} family={components[-1].family}")

    for spec in old_specs[1:]:
        try:
            df = read_component_file(spec.path, spec.col, spec.name)
            vals = align_values(base, df, spec.name)
            if vals is None:
                logs.append(f"skip old:{spec.name}: alignment/degenerate")
                continue
            components.append(LoadedComponent(family_for_name(spec.name), spec.name, vals))
            logs.append(f"loaded old:{spec.name} family={components[-1].family}")
        except Exception as exc:  # noqa: BLE001
            logs.append(f"skip old:{spec.name}: {exc}")

    for part_name, family in [("rolling_ridge", "ridge"), ("rolling_lgb", "lgb"), ("rolling_mlp", "mlp")]:
        for view in ["pred", "pred_xsz", "pred_xrank"]:
            name = f"topk_{part_name}__{view}"
            try:
                df = read_parts(part_name, view, name)
                vals = align_values(base, df, name)
                if vals is None:
                    logs.append(f"skip new:{name}: alignment/degenerate")
                    continue
                components.append(LoadedComponent(family, name, vals))
                logs.append(f"loaded new:{name} family={family}")
            except Exception as exc:  # noqa: BLE001
                logs.append(f"skip new:{name}: {exc}")

    resid_path = TOPK_DIR / "lowcorr_anchor_residual" / "topk_anchor_lowcorr_lgb_meta_chain_xsz.parquet"
    if resid_path.exists():
        for col in ["resid_pred", "combined_pred"]:
            name = f"topk_lowcorr_lgb_meta_chain_xsz__{col}"
            try:
                df = pd.read_parquet(resid_path, columns=["symbol", "datetime", "label", col])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df[(df["datetime"] >= TRAIN_START) & (df["datetime"] < TEST_END)].copy()
                df = df.sort_values(["datetime", "symbol"], kind="mergesort").reset_index(drop=True)
                df = df.rename(columns={col: name})
                vals = align_values(base, df, name)
                if vals is None:
                    logs.append(f"skip new:{name}: alignment/degenerate")
                    continue
                components.append(LoadedComponent("lgb", name, vals))
                logs.append(f"loaded new:{name} family=lgb")
            except Exception as exc:  # noqa: BLE001
                logs.append(f"skip new:{name}: {exc}")

    return base, components, logs


def stats(x: np.ndarray, y: np.ndarray, mask: np.ndarray, cols: list[int] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    xm = x[mask] if cols is None else x[mask][:, cols]
    ym = y[mask]
    good = np.isfinite(ym)
    xm = scrub(xm[good]).astype(np.float64, copy=False)
    ym = ym[good].astype(np.float64, copy=False)
    return xm.T @ xm, xm.T @ ym, float(ym @ ym)


def ic_from_stats(g: np.ndarray, c: np.ndarray, yy: float, w: np.ndarray) -> float:
    var = float(w @ g @ w)
    return float((w @ c) / np.sqrt(max(var * yy, 1e-30)))


def fit_weights(g: np.ndarray, c: np.ndarray, yy: float, signed: bool, upper: float = 0.90) -> tuple[np.ndarray, float]:
    n = len(c)
    lower = np.full(n, -0.15 if signed else 0.0, dtype=np.float64)
    upper_arr = np.full(n, max(upper, 1.0 / max(n, 1)), dtype=np.float64)
    return fit_ic_weights_from_stats(c, g, yy, lower, upper_arr)


def corr_matrix(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xm = scrub(x[mask]).astype(np.float64, copy=False)
    xm -= xm.mean(axis=0, keepdims=True)
    xm /= np.maximum(xm.std(axis=0, keepdims=True), 1e-12)
    return (xm.T @ xm) / max(len(xm), 1)


def standalone_table(
    names: list[str],
    families: list[str],
    train_g: np.ndarray,
    train_c: np.ndarray,
    train_yy: float,
    val_g: np.ndarray,
    val_c: np.ndarray,
    val_yy: float,
) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "idx": i,
                "family": families[i],
                "component": name,
                "train_ic_2019q1q3": train_c[i] / np.sqrt(max(train_g[i, i] * train_yy, 1e-18)),
                "val_ic_2019q4": val_c[i] / np.sqrt(max(val_g[i, i] * val_yy, 1e-18)),
            }
        )
    return pd.DataFrame(rows).sort_values("val_ic_2019q4", ascending=False).reset_index(drop=True)


def greedy_family_select(
    family: str,
    standalone: pd.DataFrame,
    train_g: np.ndarray,
    train_c: np.ndarray,
    train_yy: float,
    val_g: np.ndarray,
    val_c: np.ndarray,
    val_yy: float,
    corr: np.ndarray,
    *,
    signed: bool,
    max_k: int,
) -> tuple[list[int], pd.DataFrame]:
    pool = standalone[standalone["family"] == family].sort_values("val_ic_2019q4", ascending=False)
    pool_idx = pool.head(24)["idx"].astype(int).tolist()
    if not pool_idx:
        return [], pd.DataFrame()

    rows = []
    best_overall: tuple[float, list[int]] = (-np.inf, [])
    for corr_penalty in [0.0, 0.004, 0.010]:
        current = [pool_idx[0]]
        while len(current) < min(max_k, len(pool_idx)):
            best = None
            for cand in pool_idx:
                if cand in current:
                    continue
                trial = current + [cand]
                idx = np.asarray(trial, dtype=np.int32)
                w, train_ic = fit_weights(train_g[np.ix_(idx, idx)], train_c[idx], train_yy, signed=signed)
                val_ic = ic_from_stats(val_g[np.ix_(idx, idx)], val_c[idx], val_yy, w)
                subcorr = np.abs(corr[np.ix_(idx, idx)])
                avg_corr = float((subcorr.sum() - len(idx)) / max(len(idx) * (len(idx) - 1), 1))
                score = val_ic - corr_penalty * avg_corr
                key = (score, val_ic, -avg_corr)
                if best is None or key > best[0]:
                    best = (key, cand, w, train_ic, val_ic, avg_corr)
            if best is None:
                break
            current.append(int(best[1]))
            row = {
                "family": family,
                "signed": signed,
                "corr_penalty": corr_penalty,
                "k": len(current),
                "train_ic_2019q1q3": float(best[3]),
                "val_ic_2019q4": float(best[4]),
                "avg_abs_corr_2019q1q3": float(best[5]),
                "indices": json.dumps(current),
            }
            rows.append(row)
            rank_key = (float(best[4]), -float(best[5]), -len(current))
            if rank_key > (best_overall[0], -999.0, -999):  # type: ignore[operator]
                best_overall = (float(best[4]), list(current))

    grid = pd.DataFrame(rows)
    if grid.empty:
        return [pool_idx[0]], grid
    pick = grid.sort_values(["val_ic_2019q4", "avg_abs_corr_2019q1q3", "k"], ascending=[False, True, True]).iloc[0]
    return [int(i) for i in json.loads(pick["indices"])], grid


def global_anchor_sets(names: list[str], standalone: pd.DataFrame) -> dict[str, list[int]]:
    name_to_idx = {name: i for i, name in enumerate(names)}
    top_by_family = []
    for family in ["ridge", "lgb", "mlp"]:
        sub = standalone[standalone["family"] == family].sort_values("val_ic_2019q4", ascending=False)
        if not sub.empty:
            top_by_family.append(int(sub.iloc[0]["idx"]))

    presets = {
        "old_minimal_seed": [
            name_to_idx[x]
            for x in ["mlp_overlap333_xsz_hl12_n1200k_raw", "base_raw_raw"]
            if x in name_to_idx
        ],
        "family_top3_seed": top_by_family,
        "topk_new3_seed": [
            name_to_idx[x]
            for x in ["topk_rolling_ridge__pred_xsz", "topk_rolling_lgb__pred_xsz", "topk_rolling_mlp__pred_xsz"]
            if x in name_to_idx
        ],
    }
    presets["old_plus_new_seed"] = list(dict.fromkeys(presets["old_minimal_seed"] + presets["topk_new3_seed"]))
    return presets


def greedy_global_select(
    names: list[str],
    standalone: pd.DataFrame,
    train_g: np.ndarray,
    train_c: np.ndarray,
    train_yy: float,
    val_g: np.ndarray,
    val_c: np.ndarray,
    val_yy: float,
    corr: np.ndarray,
    *,
    signed: bool,
    max_k: int,
) -> tuple[list[int], pd.DataFrame]:
    rows = []
    top_pool = standalone.sort_values("val_ic_2019q4", ascending=False).head(36)["idx"].astype(int).tolist()
    # Keep the old minimal components in the pool even when their standalone
    # Q4 IC is mediocre; the old model chose them with 2019-only evidence.
    name_to_idx = {name: i for i, name in enumerate(names)}
    for old_name in [
        "base_raw_raw",
        "base_raw_xsz",
        "base_xsz_xrank",
        "chunk_t500_xsz_random_shallow_lb18_raw",
        "lowcorr_lgb_meta_chain_xsz_raw",
        "lowcorr_ridge_chain_only_xsz_raw",
        "mlp_overlap333_xsz_hl12_n1200k_raw",
        "mlp_overlap333_xsz_hl12_n1200k_xsz",
        "mlp_overlap333_xsz_hl12_n800k_xsz",
    ]:
        if old_name in name_to_idx:
            top_pool.append(name_to_idx[old_name])
    pool = list(dict.fromkeys(int(i) for i in top_pool))

    for seed_name, seed in global_anchor_sets(names, standalone).items():
        if not seed:
            continue
        for corr_penalty in [0.0, 0.003, 0.006, 0.012]:
            current = list(dict.fromkeys(seed))
            while len(current) < min(max_k, len(pool)):
                best = None
                for cand in pool:
                    if cand in current:
                        continue
                    trial = current + [cand]
                    idx = np.asarray(trial, dtype=np.int32)
                    w, train_ic = fit_weights(train_g[np.ix_(idx, idx)], train_c[idx], train_yy, signed=signed)
                    val_ic = ic_from_stats(val_g[np.ix_(idx, idx)], val_c[idx], val_yy, w)
                    subcorr = np.abs(corr[np.ix_(idx, idx)])
                    avg_corr = float((subcorr.sum() - len(idx)) / max(len(idx) * (len(idx) - 1), 1))
                    max_corr = float((subcorr - np.eye(len(idx))).max()) if len(idx) > 1 else 0.0
                    score = val_ic - corr_penalty * avg_corr
                    key = (score, val_ic, -avg_corr, -max_corr)
                    if best is None or key > best[0]:
                        best = (key, cand, w, train_ic, val_ic, avg_corr, max_corr)
                if best is None:
                    break
                current.append(int(best[1]))
                rows.append(
                    {
                        "seed": seed_name,
                        "signed": signed,
                        "corr_penalty": corr_penalty,
                        "k": len(current),
                        "train_ic_2019q1q3": float(best[3]),
                        "val_ic_2019q4": float(best[4]),
                        "avg_abs_corr_2019q1q3": float(best[5]),
                        "max_abs_corr_2019q1q3": float(best[6]),
                        "indices": json.dumps(current),
                        "components": "|".join(names[i] for i in current),
                    }
                )

    grid = pd.DataFrame(rows)
    if grid.empty:
        return [], grid
    pick = grid.sort_values(
        ["val_ic_2019q4", "avg_abs_corr_2019q1q3", "k"],
        ascending=[False, True, True],
    ).iloc[0]
    return [int(i) for i in json.loads(pick["indices"])], grid


def summarize_pred(pred: pd.DataFrame, model: str) -> dict[str, object]:
    pred = add_cross_sectional_norms(pred, "pred")
    out: dict[str, object] = {"model": model, "rows": len(pred), "label_rows": int(pred["label"].notna().sum())}
    for col in ["pred", "pred_xsz", "pred_xrank"]:
        mic = period_ic(pred, col, "M")
        out[f"{col}_ic_2020"] = compute_ic(pred[col].to_numpy(), pred["label"].to_numpy())
        out[f"{col}_monthly_mean_2020"] = float(mic.mean())
        out[f"{col}_monthly_ir_2020"] = float(mic.mean() / mic.std(ddof=1)) if mic.std(ddof=1) > 0 else float("nan")
    out["beats_old_minimal_raw"] = bool(out["pred_ic_2020"] > OLD_MINIMAL_RAW_IC)
    out["beats_expanded_clean_raw"] = bool(out["pred_ic_2020"] > EXPANDED_CLEAN_RAW_IC)
    return out


def evaluate_component_set(
    base: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    cols: list[int],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    signed: bool,
    tag: str,
) -> tuple[pd.DataFrame, dict[str, object], np.ndarray, float]:
    idx = np.asarray(cols, dtype=np.int32)
    g, c, yy = stats(x, y, train_mask, cols)
    w, train_ic = fit_weights(g, c, yy, signed=signed)
    out = base.loc[test_mask, ["symbol", "datetime", "label"]].copy()
    out["pred"] = (scrub(x[test_mask][:, cols]) @ w.astype(np.float32)).astype(np.float32)
    out = add_cross_sectional_norms(out, "pred")
    summary = summarize_pred(out, tag)
    summary.update({"signed": signed, "train_ic_2019": float(train_ic), "k": len(cols), "gate_mode": "fixed_2019"})
    return out, summary, w, train_ic


def evaluate_component_set_rolling(
    base: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    cols: list[int],
    *,
    signed: bool,
    tag: str,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    dt = base["datetime"]
    pieces = []
    records = []
    prev_w: np.ndarray | None = None
    for ms in pd.date_range(TEST_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS"):
        train_mask = ((dt >= TRAIN_START) & (dt < ms) & base["label"].notna()).to_numpy()
        test_mask = ((dt >= ms) & (dt < ms + pd.DateOffset(months=1))).to_numpy()
        g, c, yy = stats(x, y, train_mask, cols)
        lower = np.full(len(cols), -0.15 if signed else 0.0, dtype=np.float64)
        upper = np.full(len(cols), max(0.90, 1.0 / max(len(cols), 1)), dtype=np.float64)
        w, train_ic = fit_ic_weights_from_stats(c, g, yy, lower, upper, prev_w)
        prev_w = w
        part = base.loc[test_mask, ["symbol", "datetime", "label"]].copy()
        part["pred"] = (scrub(x[test_mask][:, cols]) @ w.astype(np.float32)).astype(np.float32)
        pieces.append(part)
        rec: dict[str, object] = {
            "month": f"{ms:%Y-%m}",
            "train_ic": float(train_ic),
            "month_ic": compute_ic(part["pred"].to_numpy(), part["label"].to_numpy()),
        }
        for component, weight in zip(cols, w):
            rec[f"w_{component}"] = float(weight)
        records.append(rec)
    out = pd.concat(pieces, ignore_index=True)
    out = add_cross_sectional_norms(out, "pred")
    summary = summarize_pred(out, tag)
    summary.update({"signed": signed, "train_ic_2019": float("nan"), "k": len(cols), "gate_mode": "rolling_train_before_test"})
    return out, summary, pd.DataFrame(records)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base, components, logs = collect_components()
    (OUT_DIR / "load_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")

    names = [c.name for c in components]
    families = [c.family for c in components]
    x = np.column_stack([c.values for c in components]).astype(np.float32, copy=False)
    y = base["label"].to_numpy(np.float64)
    dt = base["datetime"]

    train_mask = ((dt >= TRAIN_START) & (dt < VAL_START) & base["label"].notna()).to_numpy()
    val_mask = ((dt >= VAL_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    all_2019_mask = ((dt >= TRAIN_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    test_mask = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()

    train_g, train_c, train_yy = stats(x, y, train_mask)
    val_g, val_c, val_yy = stats(x, y, val_mask)
    all19_g, all19_c, all19_yy = stats(x, y, all_2019_mask)
    corr = corr_matrix(x, train_mask)

    standalone = standalone_table(names, families, train_g, train_c, train_yy, val_g, val_c, val_yy)
    standalone.to_csv(OUT_DIR / "component_ic_2019.csv", index=False)
    pd.DataFrame(corr, index=names, columns=names).to_csv(OUT_DIR / "component_corr_2019q1q3.csv")

    summaries: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for signed in [False, True]:
        family_preds = {}
        family_train = {}
        for family in ["ridge", "lgb", "mlp"]:
            cols, grid = greedy_family_select(
                family,
                standalone,
                train_g,
                train_c,
                train_yy,
                val_g,
                val_c,
                val_yy,
                corr,
                signed=signed,
                max_k=8,
            )
            grid.to_csv(OUT_DIR / f"{family}_{'signed' if signed else 'nonneg'}_selection_grid.csv", index=False)
            if not cols:
                continue
            idx = np.asarray(cols, dtype=np.int32)
            w, train_ic = fit_weights(all19_g[np.ix_(idx, idx)], all19_c[idx], all19_yy, signed=signed)
            pred_all = scrub(x[:, cols]) @ w.astype(np.float32)
            family_preds[family] = pred_all.astype(np.float32)
            family_train[family] = train_ic
            for component, weight in zip([names[i] for i in cols], w):
                selected_rows.append(
                    {
                        "signed": signed,
                        "family": family,
                        "component": component,
                        "weight_refit_2019": float(weight),
                        "family_train_ic_2019": float(train_ic),
                    }
                )

            fam_pred = base.loc[test_mask, ["symbol", "datetime", "label"]].copy()
            fam_pred["pred"] = pred_all[test_mask].astype(np.float32)
            fam_summary = summarize_pred(fam_pred, f"family_{family}_{'signed' if signed else 'nonneg'}")
            fam_summary.update({"signed": signed, "family_train_ic_2019": float(train_ic), "family_k": len(cols)})
            summaries.append(fam_summary)
            fam_pred.to_parquet(OUT_DIR / f"{fam_summary['model']}.parquet", index=False)

        final_names = [f for f in ["ridge", "lgb", "mlp"] if f in family_preds]
        fmat = np.column_stack([family_preds[f] for f in final_names]).astype(np.float32, copy=False)
        fg, fc, fyy = stats(fmat, y, all_2019_mask)
        fw, final_train_ic = fit_weights(fg, fc, fyy, signed=signed, upper=1.0)
        out = base.loc[test_mask, ["symbol", "datetime", "label"]].copy()
        out["pred"] = (scrub(fmat[test_mask]) @ fw.astype(np.float32)).astype(np.float32)
        tag = f"three_family_{'signed' if signed else 'nonneg'}"
        out = add_cross_sectional_norms(out, "pred")
        out.to_parquet(OUT_DIR / f"{tag}.parquet", index=False)
        period_ic(out, "pred", "M").to_csv(OUT_DIR / f"{tag}_monthly_ic.csv")
        final_summary = summarize_pred(out, tag)
        final_summary.update(
            {
                "signed": signed,
                "final_train_ic_2019": float(final_train_ic),
                "final_families": "|".join(final_names),
                **{f"w_{name}": float(weight) for name, weight in zip(final_names, fw)},
            }
        )
        summaries.append(final_summary)

        global_cols, global_grid = greedy_global_select(
            names,
            standalone,
            train_g,
            train_c,
            train_yy,
            val_g,
            val_c,
            val_yy,
            corr,
            signed=signed,
            max_k=12,
        )
        global_grid.to_csv(OUT_DIR / f"global_greedy_{'signed' if signed else 'nonneg'}_selection_grid.csv", index=False)
        if global_cols:
            tag = f"global_anchor_greedy_{'signed' if signed else 'nonneg'}"
            pred, global_summary, gw, _ = evaluate_component_set(
                base,
                x,
                y,
                global_cols,
                all_2019_mask,
                test_mask,
                signed=signed,
                tag=tag,
            )
            pred.to_parquet(OUT_DIR / f"{tag}.parquet", index=False)
            period_ic(pred, "pred", "M").to_csv(OUT_DIR / f"{tag}_monthly_ic.csv")
            pd.DataFrame(
                {
                    "component": [names[i] for i in global_cols],
                    "family": [families[i] for i in global_cols],
                    "weight_refit_2019": [float(v) for v in gw],
                }
            ).to_csv(OUT_DIR / f"{tag}_weights.csv", index=False)
            summaries.append(global_summary)

            rolling_tag = f"{tag}_rolling_gate"
            rolling_pred, rolling_summary, rolling_weights = evaluate_component_set_rolling(
                base,
                x,
                y,
                global_cols,
                signed=signed,
                tag=rolling_tag,
            )
            rolling_pred.to_parquet(OUT_DIR / f"{rolling_tag}.parquet", index=False)
            period_ic(rolling_pred, "pred", "M").to_csv(OUT_DIR / f"{rolling_tag}_monthly_ic.csv")
            rolling_weights.to_csv(OUT_DIR / f"{rolling_tag}_weights.csv", index=False)
            summaries.append(rolling_summary)

    pd.DataFrame(selected_rows).to_csv(OUT_DIR / "selected_family_components.csv", index=False)
    summary = pd.DataFrame(summaries).sort_values("pred_ic_2020", ascending=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(
        summary[
            [
                "model",
                "pred_ic_2020",
                "pred_xsz_ic_2020",
                "pred_monthly_mean_2020",
                "pred_monthly_ir_2020",
                "beats_old_minimal_raw",
                "beats_expanded_clean_raw",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
