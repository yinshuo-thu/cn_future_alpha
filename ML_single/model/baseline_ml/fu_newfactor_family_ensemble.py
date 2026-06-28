#!/usr/bin/env python3
"""Clean family ensemble over fu-alpha Ridge/LGB/MLP rolling predictions.

The script only consumes strict train-before-test base predictions:
  - effective_rolling_results/*, generated month by month from historical data.
  - fu_newfactor_three_model rolling_lgb/rolling_mlp month parts.

Selection uses 2019Q1-Q3 train stats and 2019Q4 validation stats.  The final
2020 score uses weights refit on all 2019 OOS predictions and never uses 2020
labels for selection or fitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic


ROOT = Path("/root/autodl-tmp/quant/ML")
EFFECTIVE_DIR = ROOT / "effective_rolling_results"
FU_DIR = ROOT / "strict_opt_results" / "fu_newfactor_three_model"
OUT_DIR = FU_DIR / "family_ensemble_clean"

TRAIN_START = pd.Timestamp("2019-01-01")
VAL_START = pd.Timestamp("2019-10-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")
MAX_K = 12
OLD_MINIMAL_RAW_IC = 0.05549757798302793


@dataclass(frozen=True)
class Candidate:
    family: str
    name: str
    path: Path
    is_parts_dir: bool = False


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def month_range(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def read_prediction(candidate: Candidate) -> pd.DataFrame:
    cols = ["symbol", "datetime", "label", "pred", "pred_xsz", "pred_xrank"]
    if candidate.is_parts_dir:
        pieces = []
        for month in month_range("2019-01", "2020-12"):
            path = candidate.path / f"{month}.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            pieces.append(pd.read_parquet(path, columns=cols))
        df = pd.concat(pieces, ignore_index=True)
    else:
        df = pd.read_parquet(candidate.path, columns=cols)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[(df["datetime"] >= TRAIN_START) & (df["datetime"] < TEST_END)].copy()
    df = df.sort_values(["datetime", "symbol"], kind="mergesort").reset_index(drop=True)
    return df


def collect_candidates() -> list[Candidate]:
    out: list[Candidate] = []
    for path in sorted(EFFECTIVE_DIR.glob("*/*.parquet")):
        model = path.parent.name
        if path.name != f"{model}.parquet":
            continue
        if model.startswith("ridge_"):
            out.append(Candidate("ridge", model, path))
        elif model.startswith("lgbm_"):
            out.append(Candidate("lgb", model, path))
        elif model.startswith("mlp_"):
            out.append(Candidate("mlp", model, path))

    parts = FU_DIR / "prediction_parts"
    if (parts / "rolling_lgb").exists():
        out.append(Candidate("lgb", "new1617_shuffle_rolling_lgb838", parts / "rolling_lgb", True))
    if (parts / "rolling_mlp").exists():
        out.append(Candidate("mlp", "new1617_shuffle_rolling_mlp537", parts / "rolling_mlp", True))

    strict_dir = ROOT / "strict_opt_results"
    for model in [
        "lowcorr_lgb_meta_chain_xsz",
        "lowcorr_lgb_fwd15_light_xsz",
        "lowcorr_ridge_chain_only_xsz",
        "lowcorr_ridge_meta_chain_xsz",
    ]:
        path = strict_dir / f"{model}.parquet"
        if path.exists():
            out.append(Candidate("lowcorr", model, path))
    resid_path = FU_DIR / "lowcorr_residual" / "new_lgb_resid_lgb_meta_chain_xsz.parquet"
    if resid_path.exists():
        out.append(Candidate("lowcorr", "new_lgb_resid_lgb_meta_chain_xsz", resid_path))
    return out


def build_matrix(candidates: list[Candidate]) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str], list[str]]:
    first = read_prediction(candidates[0])
    base = first[["symbol", "datetime", "label"]].copy()
    ref_dt = first["datetime"].astype("int64").to_numpy()
    ref_symbol = first["symbol"].astype(str).to_numpy()

    cols: list[np.ndarray] = []
    names: list[str] = []
    families: list[str] = []
    logs: list[str] = []

    for i, cand in enumerate(candidates):
        df = first if i == 0 else read_prediction(cand)
        ok = (
            len(df) == len(base)
            and np.array_equal(df["datetime"].astype("int64").to_numpy(), ref_dt)
            and np.array_equal(df["symbol"].astype(str).to_numpy(), ref_symbol)
        )
        if not ok:
            logs.append(f"skip {cand.name}: alignment mismatch")
            continue
        for view in ["pred", "pred_xsz", "pred_xrank"]:
            name = f"{cand.name}__{view}"
            vals = scrub(df[view].to_numpy(np.float32, copy=False)).astype(np.float32, copy=False)
            if not np.isfinite(vals).any() or float(np.nanstd(vals)) <= 1e-12:
                logs.append(f"skip {name}: degenerate")
                continue
            cols.append(vals)
            names.append(name)
            families.append(cand.family)
        logs.append(f"loaded {cand.family}:{cand.name} rows={len(df)}")
        if i != 0:
            del df

    x = np.column_stack(cols).astype(np.float32, copy=False)
    return base, x, names, families, logs


def stats(x: np.ndarray, y: np.ndarray, mask: np.ndarray, idx: list[int] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    if idx is not None:
        xm = x[mask][:, idx]
    else:
        xm = x[mask]
    xm = scrub(xm).astype(np.float64, copy=False)
    ym = y[mask].astype(np.float64, copy=False)
    good = np.isfinite(ym)
    xm = xm[good]
    ym = ym[good]
    return xm.T @ xm, xm.T @ ym, float(ym @ ym)


def ic_from_stats(gram: np.ndarray, cov: np.ndarray, yty: float, w: np.ndarray) -> float:
    var = float(w @ gram @ w)
    den = np.sqrt(max(var, 1e-30) * max(yty, 1e-30))
    return float((w @ cov) / den)


def standalone_table(
    names: list[str],
    families: list[str],
    train_gram: np.ndarray,
    train_cov: np.ndarray,
    train_yty: float,
    val_gram: np.ndarray,
    val_cov: np.ndarray,
    val_yty: float,
    test_gram: np.ndarray,
    test_cov: np.ndarray,
    test_yty: float,
) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "idx": i,
                "family": families[i],
                "component": name,
                "train_ic_2019q1q3": train_cov[i] / np.sqrt(max(train_gram[i, i] * train_yty, 1e-18)),
                "val_ic_2019q4": val_cov[i] / np.sqrt(max(val_gram[i, i] * val_yty, 1e-18)),
                "test_ic_2020_diag": test_cov[i] / np.sqrt(max(test_gram[i, i] * test_yty, 1e-18)),
            }
        )
    return pd.DataFrame(rows).sort_values("val_ic_2019q4", ascending=False).reset_index(drop=True)


def corr_matrix(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xm = scrub(x[mask]).astype(np.float64, copy=False)
    xm -= xm.mean(axis=0, keepdims=True)
    sd = np.maximum(xm.std(axis=0, keepdims=True), 1e-12)
    xm /= sd
    return (xm.T @ xm) / max(len(xm), 1)


def fit_subset(
    gram: np.ndarray,
    cov: np.ndarray,
    yty: float,
    cols: list[int],
    signed: bool,
) -> tuple[np.ndarray, float]:
    idx = np.asarray(cols, dtype=np.int32)
    lower = np.full(len(idx), -0.12 if signed else 0.0, dtype=np.float64)
    upper = np.full(len(idx), 0.90, dtype=np.float64)
    return fit_ic_weights_from_stats(cov[idx], gram[np.ix_(idx, idx)], yty, lower, upper)


def family_seed(standalone: pd.DataFrame) -> list[int]:
    seeds: list[int] = []
    for family in ["ridge", "lgb", "mlp"]:
        sub = standalone[standalone["family"] == family].sort_values("val_ic_2019q4", ascending=False)
        if not sub.empty:
            seeds.append(int(sub.iloc[0]["idx"]))
    return seeds


def greedy_select(
    names: list[str],
    families: list[str],
    train_gram: np.ndarray,
    train_cov: np.ndarray,
    train_yty: float,
    val_gram: np.ndarray,
    val_cov: np.ndarray,
    val_yty: float,
    corr: np.ndarray,
    standalone: pd.DataFrame,
    signed: bool,
) -> tuple[list[int], pd.DataFrame]:
    selected = family_seed(standalone)
    rows = []
    pool_set = set(standalone.head(18)["idx"].astype(int).tolist())
    for family in ["ridge", "lgb", "mlp"]:
        pool_set.update(
            standalone[standalone["family"] == family]
            .sort_values("val_ic_2019q4", ascending=False)
            .head(8)["idx"]
            .astype(int)
            .tolist()
        )
    pool_set.update(selected)
    pool = [int(i) for i in standalone[standalone["idx"].isin(pool_set)]["idx"].tolist()]
    for corr_penalty in [0.0, 0.006]:
        current = list(dict.fromkeys(selected))
        while len(current) < MAX_K:
            best = None
            for cand in pool:
                if cand in current:
                    continue
                trial = current + [cand]
                w, train_ic = fit_subset(train_gram, train_cov, train_yty, trial, signed=signed)
                idx = np.asarray(trial, dtype=np.int32)
                val_ic = ic_from_stats(val_gram[np.ix_(idx, idx)], val_cov[idx], val_yty, w)
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
                    "signed": signed,
                    "corr_penalty": corr_penalty,
                    "k": len(current),
                    "train_ic_2019q1q3": float(best[3]),
                    "val_ic_2019q4": float(best[4]),
                    "avg_abs_corr_2019q1q3": float(best[5]),
                    "max_abs_corr_2019q1q3": float(best[6]),
                    "indices": json.dumps(current),
                    "components": "|".join(names[i] for i in current),
                    "families": "|".join(families[i] for i in current),
                    "weights": json.dumps([float(v) for v in best[2]]),
                }
            )
    grid = pd.DataFrame(rows)
    if grid.empty:
        return selected, grid
    best_row = grid.sort_values(
        ["val_ic_2019q4", "avg_abs_corr_2019q1q3", "k"],
        ascending=[False, True, True],
    ).iloc[0]
    return [int(i) for i in json.loads(best_row["indices"])], grid


def summarize_prediction(pred: pd.DataFrame, model: str) -> dict[str, object]:
    pred = add_cross_sectional_norms(pred, "pred")
    row: dict[str, object] = {"model": model, "rows": len(pred), "label_rows": int(pred["label"].notna().sum())}
    for col in ["pred", "pred_xsz", "pred_xrank"]:
        mic = period_ic(pred, col, "M")
        row[f"{col}_ic_2020"] = compute_ic(pred[col].to_numpy(), pred["label"].to_numpy())
        row[f"{col}_monthly_mean_2020"] = float(mic.mean())
        row[f"{col}_monthly_std_2020"] = float(mic.std(ddof=1))
        row[f"{col}_monthly_ir_2020"] = float(mic.mean() / mic.std(ddof=1)) if mic.std(ddof=1) > 0 else float("nan")
    return row


def evaluate_final(
    base: pd.DataFrame,
    x: np.ndarray,
    names: list[str],
    families: list[str],
    cols: list[int],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    signed: bool,
    tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    y = base["label"].to_numpy(np.float64)
    gram, cov, yty = stats(x, y, train_mask)
    w, train_ic = fit_subset(gram, cov, yty, cols, signed=signed)
    pred_vals = scrub(x[test_mask][:, cols]) @ w.astype(np.float32)
    pred = base.loc[test_mask, ["symbol", "datetime", "label"]].copy()
    pred["pred"] = pred_vals.astype(np.float32)
    pred = add_cross_sectional_norms(pred, "pred")
    summary = summarize_prediction(pred, tag)
    summary.update(
        {
            "signed": signed,
            "train_ic_2019": float(train_ic),
            "k": len(cols),
            "beats_old_minimal_raw": bool(summary["pred_ic_2020"] > OLD_MINIMAL_RAW_IC),
        }
    )
    weights = pd.DataFrame(
        {
            "component": [names[i] for i in cols],
            "family": [families[i] for i in cols],
            "weight": [float(v) for v in w],
        }
    ).sort_values("weight", ascending=False)
    return pred, weights, summary


def evaluate_rolling_gate(
    base: pd.DataFrame,
    x: np.ndarray,
    names: list[str],
    families: list[str],
    cols: list[int],
    signed: bool,
    tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dt = base["datetime"]
    y = base["label"].to_numpy(np.float64)
    pieces = []
    weight_rows = []
    prev_w: np.ndarray | None = None
    for ms in pd.date_range(TEST_START, TEST_END - pd.offsets.MonthBegin(1), freq="MS"):
        train_mask = ((dt >= TRAIN_START) & (dt < ms) & base["label"].notna()).to_numpy()
        test_mask = ((dt >= ms) & (dt < ms + pd.DateOffset(months=1))).to_numpy()
        gram, cov, yty = stats(x, y, train_mask)
        idx = np.asarray(cols, dtype=np.int32)
        lower = np.full(len(idx), -0.12 if signed else 0.0, dtype=np.float64)
        upper = np.full(len(idx), 0.90, dtype=np.float64)
        w, train_ic = fit_ic_weights_from_stats(cov[idx], gram[np.ix_(idx, idx)], yty, lower, upper, prev_w)
        prev_w = w
        pred_vals = scrub(x[test_mask][:, cols]) @ w.astype(np.float32)
        part = base.loc[test_mask, ["symbol", "datetime", "label"]].copy()
        part["pred"] = pred_vals.astype(np.float32)
        pieces.append(part)
        row = {
            "month": f"{ms:%Y-%m}",
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_ic": float(train_ic),
            "month_ic": compute_ic(part["pred"].to_numpy(), part["label"].to_numpy()),
        }
        for component, family, weight in zip([names[i] for i in cols], [families[i] for i in cols], w):
            row[f"w__{family}__{component}"] = float(weight)
        weight_rows.append(row)
    pred = pd.concat(pieces, ignore_index=True)
    pred = add_cross_sectional_norms(pred, "pred")
    summary = summarize_prediction(pred, tag)
    summary.update(
        {
            "signed": signed,
            "train_ic_2019": float("nan"),
            "k": len(cols),
            "beats_old_minimal_raw": bool(summary["pred_ic_2020"] > OLD_MINIMAL_RAW_IC),
            "gate_mode": "rolling_train_before_test",
        }
    )
    weights = pd.DataFrame(weight_rows)
    return pred, weights, summary


def plot_monthly(preds: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, pred in preds.items():
        mic = period_ic(pred, "pred", "M")
        ax.plot(pd.to_datetime(mic.index), mic.to_numpy(), marker="o", linewidth=1.4, label=name)
    ax.axhline(OLD_MINIMAL_RAW_IC, color="#444444", linestyle="--", linewidth=1.0, label="old minimal raw IC")
    ax.set_title("Fu New-Factor Family Ensemble 2020 Monthly IC")
    ax.set_ylabel("IC")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "monthly_ic.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates()
    base, x, names, families, logs = build_matrix(candidates)
    (OUT_DIR / "candidate_load_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")

    dt = base["datetime"]
    y = base["label"].to_numpy(np.float64)
    train_mask = ((dt >= TRAIN_START) & (dt < VAL_START) & base["label"].notna()).to_numpy()
    val_mask = ((dt >= VAL_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    all_2019_mask = ((dt >= TRAIN_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()
    test_mask = ((dt >= TEST_START) & (dt < TEST_END)).to_numpy()
    test_label_mask = (test_mask & base["label"].notna().to_numpy())

    train_gram, train_cov, train_yty = stats(x, y, train_mask)
    val_gram, val_cov, val_yty = stats(x, y, val_mask)
    test_gram, test_cov, test_yty = stats(x, y, test_label_mask)
    corr = corr_matrix(x, train_mask)

    standalone = standalone_table(names, families, train_gram, train_cov, train_yty, val_gram, val_cov, val_yty, test_gram, test_cov, test_yty)
    standalone.to_csv(OUT_DIR / "component_ic.csv", index=False)
    pd.DataFrame(corr, index=names, columns=names).to_csv(OUT_DIR / "component_corr_2019q1q3.csv")

    summaries = []
    pred_outputs: dict[str, pd.DataFrame] = {}
    for signed in [False]:
        cols, grid = greedy_select(names, families, train_gram, train_cov, train_yty, val_gram, val_cov, val_yty, corr, standalone, signed)
        grid.to_csv(OUT_DIR / f"selection_grid_{'signed' if signed else 'nonneg'}.csv", index=False)
        tag = f"fu_newfactor_family_greedy_{'signed' if signed else 'nonneg'}"
        pred, weights, summary = evaluate_final(base, x, names, families, cols, all_2019_mask, test_mask, signed, tag)
        summary["gate_mode"] = "fixed_2019"
        pred.to_parquet(OUT_DIR / f"{tag}.parquet", index=False)
        weights.to_csv(OUT_DIR / f"{tag}_weights.csv", index=False)
        period_ic(pred, "pred", "M").to_csv(OUT_DIR / f"{tag}_monthly_ic.csv")
        summaries.append(summary)
        pred_outputs[tag] = pred

        rolling_tag = f"{tag}_rolling_gate"
        rolling_pred, rolling_weights, rolling_summary = evaluate_rolling_gate(base, x, names, families, cols, signed, rolling_tag)
        rolling_pred.to_parquet(OUT_DIR / f"{rolling_tag}.parquet", index=False)
        rolling_weights.to_csv(OUT_DIR / f"{rolling_tag}_weights.csv", index=False)
        period_ic(rolling_pred, "pred", "M").to_csv(OUT_DIR / f"{rolling_tag}_monthly_ic.csv")
        summaries.append(rolling_summary)
        pred_outputs[rolling_tag] = rolling_pred

    summary_df = pd.DataFrame(summaries).sort_values("pred_ic_2020", ascending=False)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    plot_monthly(pred_outputs)
    print(summary_df[["model", "pred_ic_2020", "pred_xsz_ic_2020", "pred_monthly_mean_2020", "pred_monthly_ir_2020", "k", "beats_old_minimal_raw"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
