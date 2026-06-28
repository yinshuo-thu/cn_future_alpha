#!/usr/bin/env python3
"""2019-only component filter/gate ablation for strict OOS predictions."""

from __future__ import annotations

import json
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

OUT_DIR = Path("/root/autodl-tmp/quant/ML/gate_threshold_results")
PRED_START = pd.Timestamp("2019-01-01")
VAL_START = pd.Timestamp("2019-10-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2021-01-01")


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def load_all_components() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    for path in sorted(STRICT_DIR.glob("opt_*.parquet")):
        model = path.stem
        if model not in ic_map:
            continue
        for suffix, col in [("raw", "pred"), ("xsz", "pred_xsz"), ("xrank", "pred_xrank")]:
            specs.append({"name": f"{model}_{suffix}", "model": model, "path": path, "col": col, "base_ic_2019": ic_map[model]})

    by_path: dict[Path, list[dict[str, object]]] = {}
    for spec in specs:
        by_path.setdefault(Path(spec["path"]), []).append(spec)

    base = None
    meta = []
    loaded = 0
    for path, path_specs in by_path.items():
        cols = ["symbol", "datetime", "label"] + sorted({str(s["col"]) for s in path_specs})
        try:
            df = pd.read_parquet(path, columns=cols)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {path}: {exc}", flush=True)
            continue
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[(df["datetime"] >= PRED_START) & (df["datetime"] < TEST_END)].copy()
        df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
        if base is None:
            base = df[["symbol", "datetime", "label"]].copy()
        else:
            if len(df) != len(base):
                raise RuntimeError(f"row count mismatch for {path}: {len(df)} != {len(base)}")
            sample = np.linspace(0, len(base) - 1, min(64, len(base)), dtype=np.int64)
            sym_ok = (df["symbol"].iloc[sample].to_numpy() == base["symbol"].iloc[sample].to_numpy()).all()
            dt_ok = (df["datetime"].iloc[sample].to_numpy() == base["datetime"].iloc[sample].to_numpy()).all()
            if not (sym_ok and dt_ok):
                raise RuntimeError(f"component row order mismatch for {path}")
        for spec in path_specs:
            loaded += 1
            name = str(spec["name"])
            col = str(spec["col"])
            base[name] = df[col].to_numpy(np.float32, copy=True)
            meta.append({"name": name, "model": spec["model"], "base_ic_2019": spec["base_ic_2019"]})
            print(f"[component {loaded:02d}/{len(specs)}] {name} rows={len(df)} cols={base.shape[1]}", flush=True)
        del df
    if base is None:
        raise RuntimeError("no components loaded")
    return base, pd.DataFrame(meta)


def fit_weights(df: pd.DataFrame, names: list[str], start: pd.Timestamp, end: pd.Timestamp, lower_v: float, upper_v: float) -> tuple[np.ndarray, float]:
    tr = df[(df["datetime"] >= start) & (df["datetime"] < end)]
    x = scrub(tr[names].to_numpy(np.float32)).astype(np.float64, copy=False)
    y = tr["label"].to_numpy(np.float64)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    lower = np.full(len(names), lower_v, dtype=np.float64)
    upper = np.full(len(names), upper_v, dtype=np.float64)
    return fit_ic_weights_from_stats(x.T @ y, x.T @ x, float(y @ y), lower, upper)


def pred_frame(df: pd.DataFrame, names: list[str], weights: np.ndarray, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    sub = df[(df["datetime"] >= start) & (df["datetime"] < end)].copy()
    sub["pred"] = (scrub(sub[names].to_numpy(np.float32)) @ weights.astype(np.float32)).astype(np.float32)
    return add_cross_sectional_norms(sub[["symbol", "datetime", "label", "pred"]], "pred")


def summarize(out: pd.DataFrame, tag: str) -> dict[str, object]:
    row: dict[str, object] = {"model": tag, "rows": len(out), "label_rows": int(out["label"].notna().sum())}
    for col in ["pred", "pred_xsz", "pred_xrank"]:
        by_m = period_ic(out, col, "M")
        row[f"{col}_ic_2020"] = compute_ic(out[col].to_numpy(), out["label"].to_numpy())
        row[f"{col}_monthly_mean_2020"] = float(by_m.mean())
        row[f"{col}_monthly_ir_2020"] = float(by_m.mean() / by_m.std()) if by_m.std() > 0 else float("nan")
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, meta = load_all_components()
    candidates = []
    thresholds = [0.04, 0.045, 0.05, 0.052, 0.054]
    bounds = [(-0.12, 0.85, "signed_wide"), (-0.06, 0.75, "signed_tight"), (0.0, 0.75, "nonneg")]
    for thr in thresholds:
        names = meta.loc[meta["base_ic_2019"] >= thr, "name"].tolist()
        if len(names) < 3:
            continue
        for lower, upper, btag in bounds:
            w, train_ic = fit_weights(df, names, PRED_START, VAL_START, lower, upper)
            val = pred_frame(df, names, w, VAL_START, TEST_START)
            val_ic = compute_ic(val["pred"].to_numpy(), val["label"].to_numpy())
            candidates.append({"thr": thr, "bound": btag, "lower": lower, "upper": upper, "n": len(names), "train_ic_2019_jansep": train_ic, "val_ic_2019q4": val_ic})
            print(f"[candidate] thr={thr:.3f} bound={btag} n={len(names)} train={train_ic:.6f} val={val_ic:.6f}", flush=True)
    cand = pd.DataFrame(candidates).sort_values(["val_ic_2019q4", "train_ic_2019_jansep"], ascending=False)
    cand.to_csv(OUT_DIR / "candidate_2019_validation.csv", index=False)
    best = cand.iloc[0].to_dict()
    names = meta.loc[meta["base_ic_2019"] >= float(best["thr"]), "name"].tolist()
    w, train_ic = fit_weights(df, names, PRED_START, TEST_START, float(best["lower"]), float(best["upper"]))
    out = pred_frame(df, names, w, TEST_START, TEST_END)
    tag = f"gate_thr{best['thr']:.3f}_{best['bound']}"
    out.to_parquet(OUT_DIR / f"{tag}.parquet", index=False)
    pd.DataFrame([summarize(out, tag)]).to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame({"name": names, "weight": w}).to_csv(OUT_DIR / "weights.csv", index=False)
    monthly = period_ic(out, "pred", "M")
    monthly.to_csv(OUT_DIR / "monthly_ic.csv")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly.index.astype(str), monthly.to_numpy(), color=["#2f6f8f" if x >= 0 else "#a23b3b" for x in monthly])
    ax.axhline(0.06, color="firebrick", linestyle="--", linewidth=1)
    ax.axhline(float(monthly.mean()), color="darkgreen", linestyle=":", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "monthly_ic.png", dpi=140)
    meta_out = {"selected_by": "max 2019Q4 validation IC", "best": best, "all_2019_train_ic": train_ic}
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(pd.read_csv(OUT_DIR / "summary.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
