#!/usr/bin/env python3
"""Low-memory static IC gate over strict component predictions."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/quant/ML")

import numpy as np
import pandas as pd

from rolling_factor_model_eval import add_cross_sectional_norms, compute_ic, fit_ic_weights_from_stats
from strict_optimization_ablation import (
    BASE_STRICT_DIR,
    OUT_DIR,
    PRED_START,
    TEST_END,
    TEST_START,
    summarize,
)


WORK_DIR = OUT_DIR / "lowmem_gate"
WORK_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Spec:
    name: str
    path: Path
    col: str


def collect_specs() -> list[Spec]:
    specs: list[Spec] = []
    strict_files = [
        ("base_raw", BASE_STRICT_DIR / "strict_lgb_raw_top300_n500000.parquet"),
        ("base_xsz", BASE_STRICT_DIR / "strict_lgb_xsz_top300_n500000.parquet"),
        ("base_xrank", BASE_STRICT_DIR / "strict_lgb_xrank_top300_n500000.parquet"),
    ]
    for prefix, path in strict_files:
        if path.exists():
            specs.extend([Spec(f"{prefix}_raw", path, "pred"), Spec(f"{prefix}_xsz", path, "pred_xsz"), Spec(f"{prefix}_xrank", path, "pred_xrank")])

    min_ic = float(os.environ.get("LOWMEM_GATE_MIN_BASE_2019_IC", "0.04"))
    eligible: set[str] | None = None
    summary_path = OUT_DIR / "base_ablation_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        eligible = set(summary.loc[summary["pred_ic_2019"].fillna(-1.0) >= min_ic, "model"].astype(str))

    candidates = list(OUT_DIR.glob("opt_*.parquet")) + list(OUT_DIR.glob("chunk_*.parquet")) + list(OUT_DIR.glob("lowcorr_*.parquet"))
    candidates = [p for p in candidates if not p.name.endswith("_meta_features.parquet")]
    excluded = {x.strip() for x in os.environ.get("LOWMEM_GATE_EXCLUDE_MODELS", "").split(",") if x.strip()}
    raw_only_candidates = os.environ.get("LOWMEM_GATE_RAW_ONLY_CANDIDATES", "0") == "1"
    for path in sorted(candidates):
        prefix = path.stem
        if prefix in excluded:
            print(f"[spec-skip] {prefix}: excluded by LOWMEM_GATE_EXCLUDE_MODELS", flush=True)
            continue
        if eligible is not None and prefix not in eligible:
            print(f"[spec-skip] {prefix}: 2019 IC below {min_ic:.4f}", flush=True)
            continue
        specs.append(Spec(f"{prefix}_raw", path, "pred"))
        if not raw_only_candidates:
            specs.extend([Spec(f"{prefix}_xsz", path, "pred_xsz"), Spec(f"{prefix}_xrank", path, "pred_xrank")])
    return specs


def read_component(spec: Spec) -> pd.DataFrame:
    df = pd.read_parquet(spec.path, columns=["symbol", "datetime", "label", spec.col])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[(df["datetime"] >= PRED_START) & (df["datetime"] < TEST_END)].copy()
    df = df.sort_values(["datetime", "symbol"], kind="mergesort").reset_index(drop=True)
    return df


def scrub(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x).copy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def fit_static(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, signed: bool) -> tuple[np.ndarray, float]:
    xt = scrub(x[train_mask]).astype(np.float64, copy=False)
    yt = y[train_mask].astype(np.float64, copy=False)
    mask = np.isfinite(yt) & np.all(np.isfinite(xt), axis=1)
    xt = xt[mask]
    yt = yt[mask]
    lower = np.full(xt.shape[1], -0.12 if signed else 0.0, dtype=np.float64)
    upper = np.full(xt.shape[1], 0.85 if signed else 0.75, dtype=np.float64)
    return fit_ic_weights_from_stats(xt.T @ yt, xt.T @ xt, float(yt @ yt), lower, upper)


def main() -> None:
    specs = collect_specs()
    if not specs:
        raise RuntimeError("no component specs")
    first = read_component(specs[0])
    n = len(first)
    ref_symbol = first["symbol"].astype(str).to_numpy()
    ref_dt = first["datetime"].astype("int64").to_numpy()
    base = first[["symbol", "datetime", "label"]].copy()
    mat_path = WORK_DIR / "components.float32.memmap"
    xmap = np.memmap(mat_path, mode="w+", dtype=np.float32, shape=(n, len(specs)))

    names: list[str] = []
    for j, spec in enumerate(specs):
        df = first if j == 0 else read_component(spec)
        ok = len(df) == n and np.array_equal(df["datetime"].astype("int64").to_numpy(), ref_dt) and np.array_equal(df["symbol"].astype(str).to_numpy(), ref_symbol)
        if not ok:
            print(f"[align-skip] {spec.name}: key mismatch", flush=True)
            continue
        xmap[:, len(names)] = scrub(df[spec.col].to_numpy(np.float32))
        names.append(spec.name)
        print(f"[component] {len(names):02d}/{len(specs)} {spec.name}", flush=True)
        if j != 0:
            del df
    xmap.flush()
    x = np.asarray(xmap[:, : len(names)])
    y = base["label"].to_numpy(np.float64)
    dt = base["datetime"]
    train_mask = ((dt >= PRED_START) & (dt < TEST_START) & base["label"].notna()).to_numpy()

    rows = []
    for signed in [False, True]:
        tag = "lowmem_moe_static_signed" if signed else "lowmem_moe_static_nonneg"
        weights, train_ic = fit_static(x, y, train_mask, signed=signed)
        pred = scrub(x) @ weights.astype(np.float32)
        out = base.copy()
        out["pred"] = pred.astype(np.float32)
        out = add_cross_sectional_norms(out, "pred")
        out.to_parquet(OUT_DIR / f"{tag}.parquet", index=False)
        wrow = {"model": tag, "train_ic_2019": train_ic, "pred_ic_2020": compute_ic(out.loc[(out["datetime"] >= TEST_START) & (out["datetime"] < TEST_END), "pred"], out.loc[(out["datetime"] >= TEST_START) & (out["datetime"] < TEST_END), "label"])}
        wrow.update({f"w_{n}": float(w) for n, w in zip(names, weights)})
        pd.DataFrame([wrow]).to_csv(OUT_DIR / f"{tag}_weights.csv", index=False)
        rows.append(summarize(out, tag) | {"gate_train_ic_2019": train_ic})
        print(f"[gate] {tag} train_ic={train_ic:.6f} pred_ic_2020={rows[-1]['pred_ic_2020']:.6f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT_DIR / "lowmem_moe_summary.csv", index=False)
    (OUT_DIR / "lowmem_moe_components.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["model", "pred_ic_2019", "pred_ic_2020", "pred_monthly_mean_2020", "gate_train_ic_2019"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
