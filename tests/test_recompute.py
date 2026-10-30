"""Every committed run must be reproducible from its raw outputs with the current bench.py.

A failure means the code behind a record changed: bump schema_version (CLAUDE.md, Results schema)
or fix the formula.
"""

import json
import types

import pytest
import yaml

import bench
from conftest import ROOT


def committed_runs():
    for line in (ROOT / "results" / "runs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["schema_version"] == bench.SCHEMA_VERSION and (ROOT / "results" / "raw" / rec["group_id"]).is_dir():
            yield pytest.param(rec, id=rec["run_id"])


@pytest.mark.parametrize("rec", committed_runs())
def test_record_recomputes_from_raw(rec):
    raw_dir = ROOT / rec["raw"]["dir"]
    cfg = yaml.safe_load((raw_dir / "config.yaml").read_text())
    res = json.loads((raw_dir / rec["raw"]["vllm_result"]).read_text())
    p = rec["provider"]
    args = types.SimpleNamespace(
        provider=p["name"],
        instance_type=p["instance_type"],
        price_per_hour=p["price_per_hour_usd"],
        price_basis=p["price_basis"],
        gpu_sharing=p["gpu_sharing"],
        gpu_fraction=p["gpu_fraction"],
        tag=rec["tag"],
        note=rec["note"],
        extra=rec["extra"],
    )
    new = bench.build_record(
        cfg,
        args,
        res,
        c=rec["workload"]["concurrency"],
        n=rec["workload"]["num_prompts"],
        group_id=rec["group_id"],
        git=rec["git"],
        hw=rec["hardware"],
        sw=rec["software"],
        raw_dir=raw_dir,
        ready_s=rec["engine"]["server_ready_s"],
    )
    for section in ("results", "derived", "workload", "model"):
        assert new[section] == rec[section], (
            f"derived fields changed for {rec['run_id']} ({section}): bump schema_version or fix the formula"
        )
