#!/usr/bin/env python3
"""Retained three-single-model ensemble scaffold.

This module preserves the current best strict ensemble that uses only the three
ML_single models: MLP, LGB, and Ridge.  The original full rebuild needs large
intermediate prediction parquet files that are intentionally not archived under
/root/jump_model.  When those files are present, this scaffold documents the
required inputs; when they are absent, it materializes the archived strict
selector result and candidate catalog so the retained model remains auditable.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


JUMP = Path("/root/jump_model")
ROOT = JUMP / "new_ensemble"
ML_SINGLE = JUMP / "ML_single"
ML_ENSEMBLE = JUMP / "ML_ensemble"

REQUIRED_INPUTS = [
    (
        "mlp_month_parts",
        Path("/root/autodl-tmp/quant/ML/effective_rolling_results/mlp_overlap333_xsz_hl12_n1200k/month_parts"),
        "Monthly MLP prediction/view panels for 2019-2020.",
    ),
    (
        "mlp_base_script",
        Path("/root/autodl-tmp/quant/ML/agent_runs/mlp_20260628/mlp_same_model_viewcal_screen.py"),
        "Original MLP view-calibration helper imported by the selected MLP calibrator.",
    ),
    (
        "mlp_selected_config",
        Path("/root/autodl-tmp/quant/ML/agent_runs/mlp_parallel_20260628/time120_slope_a025_strong/selected_config.json"),
        "2019-only selected MLP calibration config.",
    ),
    (
        "lgb_2019_predictions",
        Path("/root/autodl-tmp/quant/ML/agent_runs/lgb_reg8_viewcal_20260628/opt_lgb_worker_t500_xsz_random_lb18_reg8_seed140_2019full.parquet"),
        "LGB 2019 prediction panel for outer-fold fitting.",
    ),
    (
        "lgb_2020_predictions",
        Path("/root/autodl-tmp/quant/ML/strict_opt_results/opt_lgb_worker_t500_xsz_random_lb18_reg8_seed140_audit2020.parquet"),
        "LGB raw 2020 audit prediction panel.",
    ),
    (
        "lgb_2020_pass",
        Path("/root/autodl-tmp/quant/ML/agent_runs/lgb_reg8_shape_seqcal_20260628/ref_time90_a1_then_signed_abs12_a0.8_recent_weak_selector_winner_audit2020_predictions.parquet"),
        "Materialized selected LGB postprocessed 2020 pass.",
    ),
    (
        "ridge_fit_predictions",
        Path("/root/autodl-tmp/quant/ML/effective_rolling_results/ridge_overlap333_xsz_hl12_a05/ridge_overlap333_xsz_hl12_a05.parquet"),
        "Ridge 2019 fit-side prediction panel.",
    ),
    (
        "ridge_apply_predictions",
        Path("/root/autodl-tmp/quant/ML/effective_rolling_results/ridge_overlap333_xsz_hl12_n900k_a02/ridge_overlap333_xsz_hl12_n900k_a02.parquet"),
        "Ridge 2020 apply-side prediction panel.",
    ),
]

ARCHIVED_REPORT_JSON = ML_ENSEMBLE / "metrics" / "current_three_models_ensemble_report.json"
ARCHIVED_SELECTOR_CSV = ML_ENSEMBLE / "metrics" / "current_three_models_selector_winners.csv"
ARCHIVED_SINGLE_CSV = ML_SINGLE / "metrics" / "single_model_audit_metrics.csv"


def ensure_dirs() -> None:
    for child in ["configs", "metrics", "weights", "figures", "scripts", "model"]:
        (ROOT / child).mkdir(parents=True, exist_ok=True)


def input_status() -> pd.DataFrame:
    rows = []
    for name, path, purpose in REQUIRED_INPUTS:
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "purpose": purpose,
            }
        )
    return pd.DataFrame(rows)


def candidate_catalog() -> pd.DataFrame:
    comps = ["mlp", "lgb", "ridge"]
    sets: dict[str, list[str]] = {
        "raw3": comps,
        "xcenter3": [f"{c}_xcenter" for c in comps],
        "xsz3": [f"{c}_xsz" for c in comps],
        "xrank3": [f"{c}_xrank" for c in comps],
        "rankgauss3": [f"{c}_rankgauss" for c in comps],
        "tanh3": [f"{c}_tanh" for c in comps],
        "raw_xsz6": comps + [f"{c}_xsz" for c in comps],
        "xsz_rank6": [f"{c}_xsz" for c in comps] + [f"{c}_xrank" for c in comps],
        "mlp_lgb_raw2": ["mlp", "lgb"],
        "mlp_lgb_xsz2": ["mlp_xsz", "lgb_xsz"],
        "mlp_lgb_rank2": ["mlp_rankgauss", "lgb_rankgauss"],
    }
    methods = [
        "equal",
        "top1",
        "icpos",
        "simplex",
        "signed_ridge_a001",
        "signed_ridge_a01",
        "signed_ridge_a1",
    ]
    rows = []
    for set_name, cols in sets.items():
        for method in methods:
            rows.append(
                {
                    "candidate": f"{set_name}__{method}",
                    "feature_set": set_name,
                    "method": method,
                    "cols_json": json.dumps(cols),
                    "postcal": "",
                }
            )
    cal_sets = {
        "raw3": sets["raw3"],
        "raw_xsz6": sets["raw_xsz6"],
        "mlp_lgb_raw2": sets["mlp_lgb_raw2"],
    }
    cal_methods = ["equal", "simplex", "signed_ridge_a001", "signed_ridge_a01", "signed_ridge_a1"]
    for set_name, cols in cal_sets.items():
        for method in cal_methods:
            for minutes, alpha in [(60, 0.25), (90, 0.25), (120, 0.25), (120, 0.50)]:
                rows.append(
                    {
                        "candidate": f"{set_name}__{method}__time{minutes}_a{alpha:g}",
                        "feature_set": set_name,
                        "method": f"{method}_timecal",
                        "cols_json": json.dumps(cols),
                        "postcal": json.dumps({"bucket_minutes": minutes, "alpha": alpha}),
                    }
                )
    return pd.DataFrame(rows)


def _first(report: dict, key: str) -> dict:
    value = report[key]
    if isinstance(value, list):
        return value[0]
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def flatten_outer_fold_weights(row: dict) -> pd.DataFrame:
    cols = json.loads(row["cols_json"])
    by_fold = json.loads(row["weights_by_fold_json"])
    rows = []
    for fold, weights in by_fold.items():
        for col, weight in zip(cols, weights):
            rows.append({"candidate": row["candidate"], "fold": fold, "feature": col, "weight": float(weight)})
    return pd.DataFrame(rows)


def materialize_archived_outputs() -> None:
    ensure_dirs()
    status = input_status()
    status.to_csv(ROOT / "configs" / "required_inputs.csv", index=False)
    candidate_catalog().to_csv(ROOT / "configs" / "candidate_catalog.csv", index=False)

    if ARCHIVED_SINGLE_CSV.exists():
        shutil.copy2(ARCHIVED_SINGLE_CSV, ROOT / "metrics" / "single_model_audit_metrics.csv")
    if ARCHIVED_SELECTOR_CSV.exists():
        shutil.copy2(ARCHIVED_SELECTOR_CSV, ROOT / "metrics" / "selector_winners_2020_audit.csv")

    report = json.loads(ARCHIVED_REPORT_JSON.read_text(encoding="utf-8"))
    strict = _first(report, "selected_strict_by_2019_selectors")
    diagnostic = _first(report, "best_2020_diagnostic")

    _write_json(
        ROOT / "configs" / "selected_by_2019.json",
        {
            "retained_model": "three_model_strict_2019_selector",
            "candidate": strict["candidate"],
            "method": strict["method"],
            "cols": json.loads(strict["cols_json"]),
            "selector_winner": True,
            "winner_selectors": strict["winner_selectors"],
            "audit2020_pooled_ic": strict["audit2020_pooled_ic"],
            "audit2020_monthly_mean": strict["audit2020_monthly_mean"],
            "target_note": "This exceeds 0.056 pooled IC; 0.56 would be a decimal-scale typo for these IC metrics.",
        },
    )
    _write_json(
        ROOT / "configs" / "best_2020_diagnostic.json",
        {
            "candidate": diagnostic["candidate"],
            "method": diagnostic["method"],
            "cols": json.loads(diagnostic["cols_json"]),
            "selector_winner": False,
            "audit2020_pooled_ic": diagnostic["audit2020_pooled_ic"],
            "audit2020_monthly_mean": diagnostic["audit2020_monthly_mean"],
            "note": "Diagnostic only: best 2020 result among audited candidates, not selected by 2019 selectors.",
        },
    )

    pd.DataFrame(
        [
            {
                "model": "three_model_strict_2019_selector",
                "candidate": strict["candidate"],
                "pooled_ic": strict["audit2020_pooled_ic"],
                "monthly_mean": strict["audit2020_monthly_mean"],
                "monthly_std": strict["audit2020_monthly_std"],
                "monthly_ir": strict["audit2020_monthly_ir"],
                "rows": strict["audit2020_rows"],
                "label_rows": strict["audit2020_label_rows"],
                "selection": strict["winner_selectors"],
            }
        ]
    ).to_csv(ROOT / "metrics" / "best_ensemble_audit_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "model": "three_model_best_2020_diagnostic",
                "candidate": diagnostic["candidate"],
                "pooled_ic": diagnostic["audit2020_pooled_ic"],
                "monthly_mean": diagnostic["audit2020_monthly_mean"],
                "monthly_std": diagnostic["audit2020_monthly_std"],
                "monthly_ir": diagnostic["audit2020_monthly_ir"],
                "rows": diagnostic["audit2020_rows"],
                "label_rows": diagnostic["audit2020_label_rows"],
                "selection": "diagnostic_not_2019_selected",
            }
        ]
    ).to_csv(ROOT / "metrics" / "best_diagnostic_audit_metrics.csv", index=False)

    flatten_outer_fold_weights(strict).to_csv(ROOT / "weights" / "selected_outer_fold_weights.csv", index=False)
    pd.DataFrame(report.get("previous_benchmarks", [])).to_csv(ROOT / "metrics" / "previous_ensemble_benchmarks.csv", index=False)


def print_table(rows: Iterable[dict]) -> None:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return
    print(df.to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true", help="Only check whether full rebuild inputs exist.")
    parser.add_argument("--materialize-archived", action="store_true", help="Write retained archived results to new_ensemble.")
    args = parser.parse_args()

    ensure_dirs()
    status = input_status()
    status.to_csv(ROOT / "configs" / "required_inputs.csv", index=False)

    if args.check_inputs:
        print_table(status.to_dict(orient="records"))
        return 0 if bool(status["exists"].all()) else 2

    if args.materialize_archived:
        materialize_archived_outputs()
        print(f"materialized archived retained ensemble under {ROOT}", flush=True)
        return 0

    missing = status.loc[~status["exists"]]
    if not missing.empty:
        materialize_archived_outputs()
        print("[missing full rebuild inputs]", flush=True)
        print_table(missing.to_dict(orient="records"))
        print(f"archived retained ensemble materialized under {ROOT}", flush=True)
        return 2

    print(
        "All required inputs exist. Run the original full evaluator from "
        "/root/jump_model/ML_single/model/ensemble_current_three.py, then copy its outputs here.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

