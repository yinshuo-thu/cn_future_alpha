#!/usr/bin/env python3
"""Dispatch retained single-model training/evaluation scripts by model name."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COMMANDS = {
    "mlp": [sys.executable, str(ROOT / "model" / "effective_rolling_mlp_single.py")],
    "lgb": [sys.executable, str(ROOT / "model" / "chunked_strict_lgb.py")],
    "ridge": [sys.executable, str(ROOT / "model" / "effective_rolling_single_models.py"), "--preset", "ridge"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(COMMANDS), required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args, passthrough = parser.parse_known_args()

    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    command = COMMANDS[args.model] + passthrough
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'model'}:{env.get('PYTHONPATH', '')}"
    print(" ".join(command), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
