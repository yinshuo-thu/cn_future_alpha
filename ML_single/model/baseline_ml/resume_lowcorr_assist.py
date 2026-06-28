#!/usr/bin/env python3
"""Resume one low-correlation assist variant month by month."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/root/autodl-tmp/quant/ML")
OUT_DIR = ROOT / "strict_opt_results"
SCRIPT = ROOT / "lowcorr_assist_models.py"


def part_count(variant: str) -> int:
    parts_dir = OUT_DIR / f"{variant}_month_parts"
    return len(list(parts_dir.glob("*.parquet"))) if parts_dir.exists() else 0


def run_variant(variant: str, one_month: bool) -> int:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONFAULTHANDLER": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYARROW_NUM_THREADS": "1",
            "ASSIST_VARIANTS": variant,
            "ASSIST_N_JOBS": "1",
        }
    )
    if one_month:
        env["ASSIST_ONE_MONTH"] = "1"
    else:
        env.pop("ASSIST_ONE_MONTH", None)
    proc = subprocess.run([sys.executable, "-u", str(SCRIPT)], cwd=str(ROOT), env=env, check=False)
    return int(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--target-parts", type=int, default=24)
    ap.add_argument("--max-attempts", type=int, default=80)
    ap.add_argument("--backoff", type=float, default=5.0)
    args = ap.parse_args()
    final_path = OUT_DIR / f"{args.variant}.parquet"
    attempts = 0
    while attempts < args.max_attempts and not final_path.exists():
        parts = part_count(args.variant)
        print(f"[resume-assist] {args.variant} parts={parts}/{args.target_parts} attempts={attempts}", flush=True)
        if parts >= args.target_parts:
            rc = run_variant(args.variant, one_month=False)
        else:
            rc = run_variant(args.variant, one_month=True)
        attempts += 1
        if final_path.exists():
            break
        if rc != 0:
            print(f"[resume-assist] child rc={rc}; continuing after backoff", flush=True)
        time.sleep(args.backoff)
    print(f"[resume-assist] final_exists={final_path.exists()} parts={part_count(args.variant)}", flush=True)
    raise SystemExit(0 if final_path.exists() else 1)


if __name__ == "__main__":
    main()
