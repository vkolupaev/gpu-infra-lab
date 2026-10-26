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
- Percentiles in prose (experiment notes, notes, commit messages):
  `p<N>@<METRIC> = <value> <unit>`, e.g. `p50@TPOT = 11.2 ms`, `p95@TTFT = 1968 ms`,
  `p99@E2EL = 6430 ms`; mean: `mean@TPOT`. Metric names as in vLLM: TTFT, TPOT, ITL,
  E2EL. Auto-generated `bench.py` Results tables keep their column headers.
- Model revision pinned as a quoted commit sha; base image pinned by digest.
- Every result gets a sanity check against a first-principles bound
  (memory bandwidth, FLOPs, memory capacity) in the experiment note.
- Experiment notes: `experiments/YYYY-MM-DD-<slug>.md` from `experiments/TEMPLATE.md`;
  Question and Hypothesis are written before results are looked at.

## Commits
- Conventional Commits, imperative, subject <= 72 chars:
  `<type>(<scope>): <summary>`; types: feat, fix, refactor, build, docs, test, chore, exp.
- `exp` bodies carry the key number with its conditions and end with
  `Run-Group: <group_id>`.
- Stage explicit paths in `exp` commits (`git add results/runs.jsonl results/raw/<group_id>
  experiments/<note>.md`), never `git add -A` or `git add .`: raw directories with no records
  in `runs.jsonl` stay untracked unless an experiment note or `notes/` refers to them.

## Layout (minimal; add a directory only when a file needs it)
- `Dockerfile` - targets `lab` (vLLM + nsys) and `lab-ncu` (+ Nsight Compute)
- `bench.py` - `vllm serve` + `vllm bench serve` wrapper, schema v1, experiment drafts
- `configs/` - engine configs (YAML)
- `scripts/` - provider-check, hwcheck, start, bootstrap, table (sweep points side by side), plot_sweep, plotstyle (shared plot style)
- `results/` - `runs.jsonl`, `raw/`, `provider-check/`, `<experiment>/points.jsonl` (one sweep point per line, see README)
- `experiments/` - experiment notes
- `notes/` - comparisons and decisions
- `tests/mock/` - fake `vllm`, `nvidia-smi`, `huggingface_hub` for offline runs

## Status
Block 0 closed on 2026-10-25 (M1-M5 done: repo, Vast as benchmark provider, schema v1,
TEMPLATE, first number in `experiments/2026-10-25-first-number.md`).
Lab image `v0.32.0-cu129-daffd21` is in GHCR (torchcodec removed); digests in `notes/images.md`.
Next: Block 1. Deferred per plan: host repeat -> 2.1; `lab-ncu`, ncu/MIG provider -> 1.3/4.4.
Order of blocks: 0 -> 1 -> 2 -> 5 -> 4 -> 3.
