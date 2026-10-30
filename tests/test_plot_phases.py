"""Benchmark phases of the first-number run, as plot_gpu_monitor.py places them on the time axis."""

import datetime as dt

import pytest

from conftest import FIRST_NUMBER, ROOT

pytest.importorskip("matplotlib")
import plot_gpu_monitor  # noqa: E402 - needs matplotlib, skipped above when it is missing

EXPECTED = {1: ("10:17:30", "10:19:57"), 8: ("10:20:09", "10:20:45"), 32: ("10:20:59", "10:22:00")}


def test_read_phases():
    phases = plot_gpu_monitor.read_phases(str(ROOT / "results" / "raw" / FIRST_NUMBER))
    assert [c for c, _, _ in phases] == sorted(EXPECTED)
    for c, start, end in phases:
        for got, want in zip((start, end), EXPECTED[c], strict=True):
            want = dt.datetime.combine(got.date(), dt.time.fromisoformat(want))
            assert abs((got - want).total_seconds()) <= 1, f"C={c}: {got:%H:%M:%S} != {want:%H:%M:%S}"
