#!/usr/bin/env python3
"""
Resume one chunked strict-LightGBM variant month by month.

The chunked runner intentionally supports CHUNKED_ONE_MONTH=1 so a native
failure never loses completed month parts.  This supervisor wraps that mode with
progress checks and a small backoff, then runs one final pass to concatenate all
parts and write the ablation summary.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/root/autodl-tmp/quant/ML")
OUT_DIR = ROOT / "strict_opt_results"
RUNNER = ROOT / "chunked_strict_lgb.py"


def count_parts(variant: str) -> int:
    parts_dir = OUT_DIR / f"{variant}_month_parts"
    return len(list(parts_dir.glob("*.parquet"))) if parts_dir.exists() else 0


def run_once(variant: str, one_month: bool, n_jobs: int) -> int:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONFAULTHANDLER": "1",
            "PYTHONMALLOC": "malloc",
            "MALLOC_ARENA_MAX": "2",
            "OMP_NUM_THREADS": str(n_jobs),
            "MKL_NUM_THREADS": str(n_jobs),
            "OPENBLAS_NUM_THREADS": str(n_jobs),
            "NUMEXPR_NUM_THREADS": str(n_jobs),
            "PYARROW_NUM_THREADS": "1",
            "CHUNKED_VARIANTS": variant,
            "CHUNKED_N_JOBS": str(n_jobs),
        }
    )
    if one_month:
        env["CHUNKED_ONE_MONTH"] = "1"
    else:
        env.pop("CHUNKED_ONE_MONTH", None)
    return subprocess.run([sys.executable, "-u", str(RUNNER)], cwd=str(ROOT), env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("--target-parts", type=int, default=24)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--backoff", type=float, default=5.0)
    args = parser.parse_args()

    final_path = OUT_DIR / f"{args.variant}.parquet"
    stale = 0
    for attempt in range(1, args.max_attempts + 1):
        before = count_parts(args.variant)
        print(f"[resume][{args.variant}] attempt={attempt} before={before}", flush=True)
        if before >= args.target_parts:
            break
        code = run_once(args.variant, one_month=True, n_jobs=args.n_jobs)
        after = count_parts(args.variant)
        print(f"[resume][{args.variant}] exit={code} after={after}", flush=True)
        if after > before:
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                print(f"[resume][{args.variant}] no progress for {stale} attempts", flush=True)
                return 2
        if after >= args.target_parts:
            break
        time.sleep(args.backoff)

    parts = count_parts(args.variant)
    if parts < args.target_parts:
        print(f"[resume][{args.variant}] incomplete parts={parts}/{args.target_parts}", flush=True)
        return 3

    if not final_path.exists():
        print(f"[resume][{args.variant}] finalizing", flush=True)
        code = run_once(args.variant, one_month=False, n_jobs=args.n_jobs)
        if code != 0:
            return code
    print(f"[resume][{args.variant}] done parts={count_parts(args.variant)} final={final_path.exists()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
