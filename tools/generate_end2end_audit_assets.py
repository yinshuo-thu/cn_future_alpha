#!/usr/bin/env python3
"""Generate lightweight audit assets for migrated end-to-end runs.

This script reads original prediction parquet files from /root/autodl-tmp and
writes only small CSV/PNG files under /root/jump_model. It does not copy raw
data, feature caches, or prediction parquet files.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
import torch

from generate_ml_audit_assets import save_assets


ROOT = Path("/root/autodl-tmp/quant/end2end")
JUMP = Path("/root/jump_model")

RUNS = [
    {
        "version": "single",
        "name": "factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44",
        "audit_name": "single",
        "metrics_dir": JUMP / "end2end_single" / "metrics",
        "figures_dir": JUMP / "end2end_single" / "figures",
        "checkpoint_dir": ROOT / "checkpoints" / "factor_operator_market_full_oldarch_fullsample_m05_m12_raw_seed44",
    },
    {
        "version": "large_v1",
        "name": "factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1",
        "audit_name": "large_v1",
        "metrics_dir": JUMP / "end2end_large" / "version1" / "metrics",
        "figures_dir": JUMP / "end2end_large" / "version1" / "figures",
        "checkpoint_dir": ROOT / "checkpoints" / "factor_operator_extra_market_lite_seq_top48_huber_moe4_monthly12_val1",
    },
    {
        "version": "large_v2",
        "name": "factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1",
        "audit_name": "large_v2",
        "metrics_dir": JUMP / "end2end_large" / "version2" / "metrics",
        "figures_dir": JUMP / "end2end_large" / "version2" / "figures",
        "checkpoint_dir": ROOT / "checkpoints" / "factor_operator_extra_market_full_seq_top48_huber_moe4_monthly12_val1",
    },
    {
        "version": "large_v3",
        "name": "review_sttopk_xsz_dtmean_2019_2020_e3",
        "audit_name": "large_v3",
        "metrics_dir": JUMP / "end2end_large" / "version3" / "metrics",
        "figures_dir": JUMP / "end2end_large" / "version3" / "figures",
        "checkpoint_dir": ROOT / "checkpoints" / "review_sttopk_xsz_dtmean_2019_2020_e3",
        "eval_start": "2020-01-01",
        "eval_end": "2021-01-01",
    },
]


def count_checkpoint_params(checkpoint_dir: Path | None) -> dict[str, int | str]:
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return {"checkpoint_count": 0, "checkpoint_bytes": 0, "param_count": "branch_selector"}
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    total_bytes = sum(path.stat().st_size for path in checkpoints)
    if not checkpoints:
        return {"checkpoint_count": 0, "checkpoint_bytes": int(total_bytes), "param_count": 0}
    ckpt = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    n_params = sum(int(value.numel()) for value in state.values() if hasattr(value, "numel"))
    feature_cols = ckpt.get("feature_cols", []) if isinstance(ckpt, dict) else []
    return {
        "checkpoint_count": len(checkpoints),
        "checkpoint_bytes": int(total_bytes),
        "param_count": int(n_params),
        "feature_cols": len(feature_cols),
        "first_checkpoint": checkpoints[0].name,
    }


def main() -> None:
    rows = []
    for spec in RUNS:
        name = str(spec["name"])
        pred_path = ROOT / "runs" / name / "predictions_with_label.parquet"
        print(f"[e2e] {name}", flush=True)
        df = pd.read_parquet(pred_path, columns=["symbol", "datetime", "label", "pred"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        if "eval_start" in spec:
            start = pd.Timestamp(str(spec["eval_start"]))
            end = pd.Timestamp(str(spec["eval_end"]))
            df = df[(df["datetime"] >= start) & (df["datetime"] < end)].copy()
            eval_window = f"{start.date()} to {(end - pd.Timedelta(days=1)).date()}"
        else:
            eval_window = f"{df['datetime'].min()} to {df['datetime'].max()}"
        row = save_assets(df, "pred", str(spec["audit_name"]), Path(spec["metrics_dir"]), Path(spec["figures_dir"]))
        row.update({"version": spec["version"], "run": name, "eval_window": eval_window})
        row.update(count_checkpoint_params(spec["checkpoint_dir"]))
        rows.append(row)
        pd.DataFrame([row]).to_csv(Path(spec["metrics_dir"]) / "audit_metrics.csv", index=False)
        print(f"[e2e] pooled={row['pooled_ic']:.6f} sn={row['SN_nonoverlap_ic']:.6f}", flush=True)
        del df
        gc.collect()
    pd.DataFrame(rows).to_csv(JUMP / "common_docs" / "end2end_selected_audit_metrics.csv", index=False)


if __name__ == "__main__":
    main()
