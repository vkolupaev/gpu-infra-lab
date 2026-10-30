"""scripts/table.py on the first-number run group: golden snapshot plus the key numbers spelled out."""

import subprocess
import sys

from conftest import FIRST_NUMBER, ROOT, SCRIPTS

GOLDEN = ROOT / "tests" / "golden" / "first-number-table.md"


def run_table():
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "table.py"), "results/runs.jsonl", "--group", FIRST_NUMBER],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def column(lines, label):
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    i = header.index(label)
    return [[c.strip() for c in row.strip("|").split("|")][i] for row in lines[2:]]


def test_matches_golden():
    assert run_table() == GOLDEN.read_text()


def test_key_numbers():
    out = run_table()
    rows = [line for line in out.splitlines() if line.startswith("|")]
    assert column(rows, "C") == ["1", "8", "32"]
    assert column(rows, "out tok/s") == ["87", "568", "1346"]
    assert column(rows, "$/1M out") == ["3.563", "0.545", "0.230"]
    assert column(rows, "std@E2EL, ms") == ["3.1", "7.9", "114.8"]
    physics = next(line for line in out.splitlines() if line.startswith("C=1:"))
    assert "(72% of spec bandwidth)" in physics
    assert "(83% of measured bandwidth)" in physics
