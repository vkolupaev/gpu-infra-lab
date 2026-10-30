"""Shared paths for the test suite. Tests read only committed data and never touch the network or a GPU."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
FIRST_NUMBER = "20261025T101616Z_qwen3-8b_0798e2f"

for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
