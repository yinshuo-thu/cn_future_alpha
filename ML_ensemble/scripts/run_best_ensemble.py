#!/usr/bin/env python3
"""Run/materialize the retained strict three-ML-single ensemble."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'model'}:{env.get('PYTHONPATH', '')}"
    command = [sys.executable, str(ROOT / "model" / "three_model_ensemble.py"), "--materialize-archived"]
    print(" ".join(command), flush=True)
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
