#!/usr/bin/env python3
"""Run the retained strict ML ensemble implementation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'model'}:{env.get('PYTHONPATH', '')}"
    command = [sys.executable, str(ROOT / "model" / "expanded_gate_stack.py")]
    print(" ".join(command), flush=True)
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
