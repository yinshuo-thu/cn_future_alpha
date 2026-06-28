#!/usr/bin/env python3
"""Low-memory 2019-only gate ablation using a float32 memmap component matrix."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats, period_ic
from strict_optimization_ablation import BASE_STRICT_DIR, OUT_DIR as STRICT_DIR

OUT_DIR = Path("/root/autodl-tmp/quant/ML/gate_memmap_results")
PRED_START = np.datetime64("2019-01-01")
VAL_START = np.datetime64("2019-10-01")
TEST_START = np.datetime64("2020-01-01")
TEST_END = np.datetime64("2021-01-01")


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def component_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    strict_files = [
        ("base_raw", BASE_STRICT_DIR / "strict_lgb_raw_top300_n500000.parquet"),
        ("base_xsz", BASE_STRICT_DIR / "strict_lgb_xsz_top300_n500000.parquet"),
        ("base_xrank", BASE_STRICT_DIR / "strict_lgb_xrank_top300_n500000.parquet"),
    ]
    for model, path in strict_files:
        if path.exists():
            for suffix, col in [("raw", "pred"), ("xsz", "pred_xsz"), ("xrank", "pred_xrank")]:
                specs.append({"name": f"{model}_{suffix}", "model": model, "path": path, "col": col, "base_ic_2019": np.inf})
    summary = pd.read_csv(STRICT_DIR / "base_ablation_summary.csv")
    ic_map = dict(zip(summary["model"].astype(str), summary["pred_ic_2019"].astype(float)))
    min_ic = float(os.environ.get("GATE_MEMMAP_MIN_BASE_2019_IC", "0.04"))
    candidate_paths = (
        list(STRICT_DIR.glob("opt_*.parquet"))
        + list(STRICT_DIR.glob("chunk_*.parquet"))
        + list(STRICT_DIR.glob("lowcorr_*.parquet"))
    )
    for path in sorted(candidate_paths):
        model = path.stem
        if model not in ic_map or float(ic_map[model]) < min_ic:
            continue
        for suffix, col in [("raw", "pred"), ("xsz", "pred_xsz"), ("xrank", "pred_xrank")]:
            specs.append({"name": f"{model}_{suffix}", "model": model, "path": path, "col": col, "base_ic_2019": ic_map[model]})
    return specs


def window_mask(base: pd.DataFrame, start: np.datetime64, end: np.datetime64, *, embargo_tail: bool) -> np.ndarray:
    dt = base["datetime"].to_numpy(dtype="datetime64[ns]")
    mask = (dt >= start) & (dt < end) & base["label"].notna().to_numpy()
    embargo_bars = int(os.environ.get("GATE_MEMMAP_EMBARGO_BARS", "30"))
    if not embargo_tail or embargo_bars <= 0 or not mask.any():
        return mask
    sub = base.loc[mask, ["symbol", "datetime"]].sort_values(["symbol", "datetime"])
    drop_idx = sub.groupby("symbol", sort=False).tail(embargo_bars).index.to_numpy()
    mask[drop_idx] = False
    return mask


def load_keys(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["symbol", "datetime", "label"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[(df["datetime"] >= pd.Timestamp(str(PRED_START))) & (df["datetime"] < pd.Timestamp(str(TEST_END)))].copy()
    return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def build_matrix(specs: list[dict[str, object]]) -> tuple[pd.DataFrame, np.memmap, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_keys(Path(specs[0]["path"]))
    n = len(base)
    m = len(specs)
    matrix_path = OUT_DIR / "components_float32.dat"
    x = np.memmap(matrix_path, dtype="float32", mode="w+", shape=(n, m))
    sample = np.linspace(0, n - 1, min(64, n), dtype=np.int64)
    meta = []
    by_path: dict[Path, list[tuple[int, dict[str, object]]]] = {}
    for j, spec in enumerate(specs):
        by_path.setdefault(Path(spec["path"]), []).append((j, spec))

    for path, items in by_path.items():
        cols = ["symbol", "datetime"] + sorted({str(spec["col"]) for _, spec in items})
        df = pd.read_parquet(path, columns=cols)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[(df["datetime"] >= pd.Timestamp(str(PRED_START))) & (df["datetime"] < pd.Timestamp(str(TEST_END)))].copy()
        df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
        if len(df) != n:
            raise RuntimeError(f"row count mismatch for {path}: {len(df)} != {n}")
        sym_ok = (df["symbol"].iloc[sample].to_numpy() == base["symbol"].iloc[sample].to_numpy()).all()
        dt_ok = (df["datetime"].iloc[sample].to_numpy() == base["datetime"].iloc[sample].to_numpy()).all()
        if not (sym_ok and dt_ok):
            raise RuntimeError(f"row order mismatch for {path}")
        for j, spec in items:
            x[:, j] = scrub(df[str(spec["col"])].to_numpy(np.float32, copy=False))
            meta.append({"idx": j, "name": spec["name"], "model": spec["model"], "base_ic_2019": spec["base_ic_2019"]})
            print(f"[component {j + 1:02d}/{m}] {spec['name']} rows={n}", flush=True)
        x.flush()
        del df
    meta_df = pd.DataFrame(meta).sort_values("idx").reset_index(drop=True)
    return base, x, meta_df


def fit_weights(x: np.memmap, y: np.ndarray, mask: np.ndarray, cols: np.ndarray, lower_v: float, upper_v: float) -> tuple[np.ndarray, float]:
    xx = np.asarray(x[mask][:, cols], dtype=np.float64)
    yy = y[mask].astype(np.float64)
    ok = np.isfinite(yy) & np.all(np.isfinite(xx), axis=1)
    xx = xx[ok]
    yy = yy[ok]
    lower = np.full(len(cols), lower_v, dtype=np.float64)
    upper = np.full(len(cols), upper_v, dtype=np.float64)
    return fit_ic_weights_from_stats(xx.T @ yy, xx.T @ xx, float(yy @ yy), lower, upper)


def predict(x: np.memmap, mask: np.ndarray, cols: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (np.asarray(x[mask][:, cols], dtype=np.float32) @ w.astype(np.float32)).astype(np.float32)


def summarize(out: pd.DataFrame, tag: str) -> dict[str, object]:
    row: dict[str, object] = {"model": tag, "rows": len(out), "label_rows": int(out["label"].notna().sum())}
    for col in ["pred", "pred_xsz", "pred_xrank"]:
        by_m = period_ic(out, col, "M")
        row[f"{col}_ic_2020"] = compute_ic(out[col].to_numpy(), out["label"].to_numpy())
        row[f"{col}_monthly_mean_2020"] = float(by_m.mean())
        row[f"{col}_monthly_ir_2020"] = float(by_m.mean() / by_m.std()) if by_m.std() > 0 else float("nan")
    return row


def main() -> None:
    specs = component_specs()
    base, x, meta = build_matrix(specs)
    y = base["label"].to_numpy(np.float64)
    train_mask = window_mask(base, PRED_START, VAL_START, embargo_tail=True)
    val_mask = window_mask(base, VAL_START, TEST_START, embargo_tail=True)
    all_2019_mask = window_mask(base, PRED_START, TEST_START, embargo_tail=True)
    test_mask = window_mask(base, TEST_START, TEST_END, embargo_tail=False)

    candidates = []
    thresholds = [0.04, 0.045, 0.05, 0.052, 0.054]
    bounds = [(-0.12, 0.85, "signed_wide"), (-0.06, 0.75, "signed_tight"), (0.0, 0.75, "nonneg")]
    for thr in thresholds:
        cols = meta.loc[meta["base_ic_2019"] >= thr, "idx"].to_numpy(np.int64)
        if len(cols) < 3:
            continue
        for lower, upper, btag in bounds:
            w, train_ic = fit_weights(x, y, train_mask, cols, lower, upper)
            val_pred = predict(x, val_mask, cols, w)
            val_ic = compute_ic(val_pred, y[val_mask])
            row = {"thr": thr, "bound": btag, "lower": lower, "upper": upper, "n": len(cols), "train_ic_2019_jansep": train_ic, "val_ic_2019q4": val_ic}
            candidates.append(row)
            print(f"[candidate] thr={thr:.3f} bound={btag} n={len(cols)} train={train_ic:.6f} val={val_ic:.6f}", flush=True)
    cand = pd.DataFrame(candidates).sort_values(["val_ic_2019q4", "train_ic_2019_jansep"], ascending=False)
    cand.to_csv(OUT_DIR / "candidate_2019_validation.csv", index=False)
    best = cand.iloc[0].to_dict()
    cols = meta.loc[meta["base_ic_2019"] >= float(best["thr"]), "idx"].to_numpy(np.int64)
    w, train_ic = fit_weights(x, y, all_2019_mask, cols, float(best["lower"]), float(best["upper"]))
    test_pred = predict(x, test_mask, cols, w)
    out = base.loc[test_mask, ["symbol", "datetime", "label"]].copy()
    out["pred"] = test_pred
    out = add_cross_sectional_norms(out, "pred")
    tag = f"gate_thr{best['thr']:.3f}_{best['bound']}"
    out.to_parquet(OUT_DIR / f"{tag}.parquet", index=False)
    summary = pd.DataFrame([summarize(out, tag)])
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    names = meta.loc[meta["idx"].isin(cols), "name"].tolist()
    pd.DataFrame({"name": names, "weight": w}).to_csv(OUT_DIR / "weights.csv", index=False)
    monthly = period_ic(out, "pred", "M")
    monthly.to_csv(OUT_DIR / "monthly_ic.csv")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly.index.astype(str), monthly.to_numpy(), color=["#2f6f8f" if v >= 0 else "#a23b3b" for v in monthly])
    ax.axhline(0.06, color="firebrick", linestyle="--", linewidth=1)
    ax.axhline(float(monthly.mean()), color="darkgreen", linestyle=":", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "monthly_ic.png", dpi=140)
    (OUT_DIR / "metadata.json").write_text(json.dumps({"selected_by": "2019Q4 validation IC", "best": best, "all_2019_train_ic": train_ic}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
