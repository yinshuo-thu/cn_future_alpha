#!/usr/bin/env python3
"""Materialize or check the retained three-model ensemble."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [sys.executable, str(ROOT / "model" / "three_model_ensemble.py"), "--materialize-archived"]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

