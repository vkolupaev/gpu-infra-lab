# CLAUDE.md

Working rules for Claude Code in this repository.

## What this repo is
Companion lab to the "GPU Infra Notebook" site: reproducible measurements of LLM
inference systems and their economics (throughput, p50/p95/p99 latency, TTFT,
cost per 1M tokens) on rented GPUs. Public repository.

The learning plan and the detailed Block 0 instructions live OUTSIDE this repo
(private): `../LEARNING PLAN/response01.md`. Read it before Block 0 work; do not
copy its private context (employer, career notes) into this repo.

## Language
- Talk to the user in Russian.
- Everything committed (code, comments, README, notes, experiment records,
  commit messages) is in English.

## Hard rules
1. No employer data here: no numbers from work GPUs/VMs, no project or company
   details. Those go to a separate private repo.
2. Commit code BEFORE a measurement run: every record stores the git commit.
   Results are committed AFTER the run as a separate `exp(...)` commit.
3. Never amend, rebase or force-push commits that records in
   `results/runs.jsonl` point to.
4. Always keep raw tool outputs (`results/raw/<group_id>/`): derived fields must
   be recomputable without renting a GPU again.
5. No secrets on rented hosts beyond an SSH key. Never ask for or store provider
   API keys (Vast.ai, RunPod, ...): creating and destroying instances is done by
   the user. Read-only HF token only if unavoidable.
6. Do not commit mock results (`tests/mock/` runs go to a temp copy of the repo).

## Results schema
- `bench.py` defines `SCHEMA_VERSION = 1` and `REQUIRED` fields.
- Adding an optional field: no version bump.
- Renaming a field or changing units/meaning: bump `schema_version` and add a
  v1 -> v2 converter in the same commit.
- Run `python3 bench.py validate results/runs.jsonl` before every `exp(...)` commit.

## Measurement method (do not "simplify")
- One variable changes per experiment; each config is its own file in `configs/`.
- `--ignore-eos`, closed loop (`--max-concurrency`) unless the experiment says otherwise.
- p99 from fewer than 200 requests is reported as approximate (`p99_reliable: false`).
- Model revision pinned as a quoted commit sha; base image pinned by digest.
- Every result gets a sanity check against a first-principles bound
  (memory bandwidth, FLOPs, memory capacity) in the experiment note.
- Experiment notes: `experiments/YYYY-MM-DD-<slug>.md` from `experiments/TEMPLATE.md`;
  Question and Hypothesis are written before results are looked at.

## Commits
Conventional Commits, imperative, subject <= 72 chars:
`<type>(<scope>): <summary>`; types: feat, fix, refactor, build, docs, test, chore, exp.
`exp` bodies carry the key number with its conditions and end with
`Run-Group: <group_id>`.

## Layout (minimal; add a directory only when a file needs it)
- `Dockerfile` - targets `lab` (vLLM + nsys) and `lab-ncu` (+ Nsight Compute)
- `bench.py` - `vllm serve` + `vllm bench serve` wrapper, schema v1, experiment drafts
- `configs/` - engine configs (YAML)
- `scripts/` - provider-check, hwcheck, start, bootstrap
- `results/` - `runs.jsonl`, `raw/`, `provider-check/`
- `experiments/` - experiment notes
- `notes/` - comparisons and decisions
- `tests/mock/` - fake `vllm`, `nvidia-smi`, `huggingface_hub` for offline runs

## Open placeholders (fill on Block 0, day 1)
- `Dockerfile`: `VLLM_IMAGE` digest, `NSYS_VERSION`, `NCU_VERSION` (from `apt-cache madison`
  inside the base image), `NSYS_PKG` if `nsight-systems-cli` is not available.
- `configs/qwen3-8b-bf16.yaml`: `revision` (quoted HF commit sha).
- Image name: `ghcr.io/<user>/gpu-infra-lab`.

## Status
Block 0 not started. Order of blocks: 0 -> 1 -> 2 -> 5 -> 4 -> 3.
