#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ML_DIR = Path("/root/autodl-tmp/quant/ML")
RUN_DIR = ML_DIR / "agent_runs" / "lgb_parallel_20260628"
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import coarse_2019_search as coarse  # noqa: E402


PRED_2019 = coarse.PRED_2019
FOLDS = coarse.FOLDS


@dataclass(frozen=True)
class Candidate:
    name: str
    spec: dict[str, Any]
    params: int


def signed_power(x: np.ndarray, gamma: float) -> np.ndarray:
    return (np.sign(x) * np.power(np.abs(x), gamma)).astype(np.float32)


def transform(df: pd.DataFrame, fit: dict[str, Any]) -> np.ndarray:
    x = df["pred"].to_numpy(np.float64)
    kind = fit["kind"]
    if kind == "power":
        return signed_power(x, float(fit["gamma"]))
    if kind == "winsor_power":
        y = np.clip(x, float(fit["lo"]), float(fit["hi"]))
        return signed_power(y, float(fit["gamma"]))
    if kind == "softsign":
        scale = float(fit["scale"])
        return (x / (1.0 + np.abs(x) / max(scale, 1e-12))).astype(np.float32)
    if kind == "asinh":
        scale = float(fit["scale"])
        return np.arcsinh(x / max(scale, 1e-12)).astype(np.float32)
    raise ValueError(kind)


def fit_candidate(train: pd.DataFrame, cand: Candidate) -> dict[str, Any]:
    kind = cand.spec["kind"]
    if kind == "power":
        return dict(cand.spec)
    if kind == "winsor_power":
        out = dict(cand.spec)
        out["lo"] = float(train["pred"].quantile(out["qlo"]))
        out["hi"] = float(train["pred"].quantile(out["qhi"]))
        return out
    if kind in {"softsign", "asinh"}:
        out = dict(cand.spec)
        out["scale"] = float(train["pred"].abs().quantile(out["scale_q"]))
        return out
    raise ValueError(kind)


def candidate_grid() -> list[Candidate]:
    out: list[Candidate] = []
    gammas = [0.45, 0.50, 0.60, 2.0 / 3.0, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0, 1.10, 1.25]
    for gamma in gammas:
        out.append(Candidate(f"power_g{gamma:g}", {"kind": "power", "gamma": gamma}, 1))
    for qlo, qhi in [(0.001, 0.999), (0.0025, 0.9975), (0.005, 0.995), (0.01, 0.99)]:
        for gamma in gammas:
            out.append(
                Candidate(
                    f"winsor_power_q{qlo:g}_{qhi:g}_g{gamma:g}",
                    {"kind": "winsor_power", "qlo": qlo, "qhi": qhi, "gamma": gamma},
                    3,
                )
            )
    for kind in ["softsign", "asinh"]:
        for scale_q in [0.90, 0.95, 0.975, 0.99]:
            out.append(Candidate(f"{kind}_scaleq{scale_q:g}", {"kind": kind, "scale_q": scale_q}, 1))
    return out


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    base = coarse.load_base(PRED_2019)
    fold_data = {}
    base_fold_ic = {}
    for fold, tr_s, tr_e, val_s, val_e in FOLDS:
        train = coarse.slice_period(base, tr_s, tr_e)
        val = coarse.slice_period(base, val_s, val_e)
        fold_data[fold] = (train, val)
        base_fold_ic[fold] = float(coarse.compute_ic(val["pred"].to_numpy(), val["label"].to_numpy()))
    rows = []
    candidates = candidate_grid()
    for cand in candidates:
        row = {"candidate": cand.name, "params": cand.params, "spec_json": json.dumps(cand.spec, sort_keys=True)}
        ics = []
        deltas = []
        ok = True
        for fold, (train, val) in fold_data.items():
            try:
                fit = fit_candidate(train, cand)
                pred = transform(val, fit)
                ic = float(coarse.compute_ic(pred, val["label"].to_numpy()))
            except Exception as exc:  # noqa: BLE001
                row[f"{fold}_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                ic = float("nan")
                ok = False
            delta = ic - base_fold_ic[fold] if math.isfinite(ic) else float("nan")
            row[f"{fold}_ic"] = ic
            row[f"{fold}_delta"] = delta
            ics.append(ic)
            deltas.append(delta)
        vals = np.asarray(ics, dtype=np.float64)
        ds = np.asarray(deltas, dtype=np.float64)
        row["cv_mean_ic"] = float(np.nanmean(vals))
        row["cv_min_ic"] = float(np.nanmin(vals))
        row["delta_mean"] = float(np.nanmean(ds))
        row["delta_min"] = float(np.nanmin(ds))
        row["delta_std"] = float(np.nanstd(ds, ddof=0))
        row["selection_score"] = row["delta_mean"] - 0.50 * row["delta_std"] + 0.25 * row["delta_min"] - 0.00005 * cand.params
        row["conservative_ok"] = bool(ok and row["delta_min"] >= -0.0005)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    summary.to_csv(RUN_DIR / "shape_refine_search_2019_only.csv", index=False)

    primary = summary[summary["conservative_ok"]].head(1)
    if primary.empty:
        primary = summary.head(1)
    selected = primary.copy()
    selected.insert(0, "selection_slot", "shape_refine_primary")
    selected["selection_reason"] = "best monotone shape-compression candidate by 2019 expanding-fold score; no 2020 read in search"
    cand = {c.name: c for c in candidates}[selected.iloc[0]["candidate"]]
    full = base.copy()
    fit = fit_candidate(full, cand)
    pred_full = transform(full, fit)
    selected["fit_json"] = json.dumps(fit, sort_keys=True)
    full_metrics = coarse.metrics(full, pred_full, "fit2019")
    for key, value in full_metrics.items():
        selected[key] = value
    selected.to_csv(RUN_DIR / "shape_refine_selected_2019_only.csv", index=False)

    config = {
        "protocol": {
            "selection_data": str(PRED_2019),
            "folds": FOLDS,
            "score": "delta_mean - 0.50*delta_std + 0.25*delta_min - 0.00005*params",
            "primary_filter": "delta_min >= -0.0005",
            "no_2020_read_by_search_script": True,
        },
        "base_fold_ic": base_fold_ic,
        "selected": selected.replace({np.nan: None}).to_dict(orient="records"),
    }
    with (RUN_DIR / "shape_refine_selected_config_2019_only.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print("Top 20 shape refine 2019-only:")
    print(summary[["candidate", "params", "selection_score", "delta_mean", "delta_min", "cv_mean_ic"]].head(20).to_string(index=False))
    print("\nSelected:")
    print(selected[["selection_slot", "candidate", "selection_score", "delta_mean", "delta_min", "cv_mean_ic", "fit2019_ic", "fit2019_monthly_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
