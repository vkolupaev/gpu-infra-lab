"""bench.py run end to end against tests/mock (fake vllm, nvidia-smi, huggingface_hub), in a temp copy of the repo."""

import json
import os
import shutil
import socket
import subprocess
import sys

import pytest

from conftest import ROOT


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def repo(tmp_path):
    """Code, configs and mocks only: no .git, no committed results, so nothing real gets appended to."""
    shutil.copy2(ROOT / "bench.py", tmp_path)
    for d in ("configs", "scripts", "tests/mock"):
        shutil.copytree(ROOT / d, tmp_path / d)
    return tmp_path


def run_bench(repo, *extra, env=None):
    env = {
        **os.environ,
        "PATH": f"{repo / 'tests/mock/bin'}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(repo / "tests/mock/py"),
        "GIT_COMMIT": "mock000",
        **(env or {}),
    }
    cmd = [
        sys.executable,
        "bench.py",
        "run",
        "--config",
        "configs/qwen3-8b-bf16.yaml",
        "--provider",
        "mock",
        "--price-per-hour",
        "1.0",
        "--skip-hwcheck",
        "--tag",
        "mock",
        "--port",
        str(free_port()),
        *extra,
    ]
    return subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=60)


def test_run_writes_valid_records_and_note(repo):
    p = run_bench(repo, "--concurrency", "1,8")
    assert p.returncode == 0, p.stdout + p.stderr
    runs = repo / "results" / "runs.jsonl"
    v = subprocess.run([sys.executable, "bench.py", "validate", str(runs)], cwd=repo, capture_output=True, text=True)
    assert v.returncode == 0, v.stdout
    recs = [json.loads(line) for line in runs.read_text().splitlines()]
    assert [r["workload"]["concurrency"] for r in recs] == [1, 8]
    assert len(list((repo / "experiments").glob("*-mock.md"))) == 1


def test_failed_server_surfaces_its_own_error(repo):
    p = run_bench(repo, "--concurrency", "1", env={"MOCK_SERVE_FAIL": "1"})
    out = p.stdout + p.stderr
    assert p.returncode != 0
    assert "libnvrtc.so.13" in out
    assert "ProcessLookupError" not in out
    assert "copy them off the host" in out


def test_random_range_ratio_reaches_the_record(repo):
    p = run_bench(repo, "--config", "configs/qwen3-8b-bf16-rr05.yaml")
    assert p.returncode == 0, p.stdout + p.stderr
    recs = [json.loads(line) for line in (repo / "results" / "runs.jsonl").read_text().splitlines()]
    assert [r["workload"]["random_range_ratio"] for r in recs] == [0.5]
