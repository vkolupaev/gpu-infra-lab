# gpu-infra-lab

Reproducible measurements of LLM inference systems and their economics:
throughput, latency (p50/p95/p99), TTFT, and cost per 1M tokens on rented GPUs.

## Latest result
Qwen3-8B (`Qwen/Qwen3-8B` @ `b968826d9c46`, BF16) on 1x A100-SXM4-80GB (SXM, power limit 400 W),
vLLM 0.32.0 with default engine settings (not tuned); random prompts, 1024 input / 256 output
tokens, closed loop at fixed concurrency; on-demand $1.114/h. Measured 2026-10-25.

| Concurrency | Out tok/s | TTFT p50/p95 ms | TPOT p50 ms | $/1M out |
|---|---|---|---|---|
| 1 | 87 | 86 / 90 | 11.2 | 3.563 |
| 8 | 568 | 485 / 538 | 12.3 | 0.545 |
| 32 | 1346 | 1334 / 1968 | 18.5 | 0.230 |

- At C=1 decode is memory-bound: every step reads all weights for one token; prefill is ~3%
  of request time.
- C=8 -> C=32 gives 2.37x output, not 4x: batching shares only the weight reads; prefill and
  KV-cache reads belong to each request and grow linearly with concurrency.
- Closed loop with `--request-rate inf` and fixed lengths keeps requests synchronized in waves,
  so TTFT here includes the prefill of the whole wave; it is not TTFT under staggered arrivals.

[Full experiment note](experiments/2026-10-25-first-number.md)

## How numbers are produced
- Question and hypothesis, with falsification criteria, are committed before the run in a
  separate commit.
- Code is committed before the run and every record stores its git commit; results arrive
  afterwards in a separate `exp(...)` commit.
- Every result is checked against a first-principles bound: memory bandwidth, FLOPs or capacity.
- Raw tool outputs are kept in `results/raw/<group_id>/`, so derived fields can be recomputed
  without renting a GPU again.
- One variable changes per experiment; each engine configuration is its own file in `configs/`.
- p99 from fewer than 200 requests is flagged as approximate (`p99_reliable: false`).
- Model revision is pinned by commit sha, the base image by digest.
- Derived fields of every committed run are recomputed from raw outputs by the test suite on
  every commit. After cloning, enable the hook: `git config core.hooksPath .githooks`.

## Layout
- `Dockerfile` - lab image: pinned official vLLM image + profiling tools
- `bench.py` - runs `vllm serve` + `vllm bench serve`, writes one record per run (schema v1)
- `configs/` - one YAML per engine configuration
- `scripts/` - provider checks, hardware sanity checks, bootstrap
- `results/runs.jsonl` - all runs, one JSON record per line; `results/raw/` - untouched tool outputs
- `experiments/` - experiment notes (question -> hypothesis -> result -> interpretation)
- `notes/` - comparisons and decisions
- `CLAUDE.md` - working rules for Claude Code in this repository

## Reproduce a run
On the GPU host, inside the lab image, export the image ref with its digest from
[`notes/images.md`](notes/images.md), then run:

    export LAB_IMAGE=ghcr.io/vkolupaev/gpu-infra-lab:<tag>@<digest>
    python3 bench.py run --config configs/qwen3-8b-bf16.yaml --provider <name> --price-per-hour <usd>

`bench.py run` checks the hardware first (`scripts/hwcheck.py` -> `hwcheck.json`; it stops
unless the verdict is OK, `--allow-suspicious` continues) and logs `nvidia-smi` once a second
to `gpu-monitor.csv` while the server runs. Raw outputs go to `results/raw/<group_id>/`.

Copy `results/` back to the laptop before destroying the instance, also when a run fails:
if the server fails before the first record, its `server.log` is the only copy of the error.
Raw directories with no records in `runs.jsonl` are committed only when an experiment note
or a file in `notes/` refers to them as evidence.

Plots are not drawn on the host. After copying `results/` back, on the laptop:

    python3 scripts/plot_gpu_monitor.py results/raw/<group_id>

## Compare points
One JSONL line is one sweep point. For `bench.py` runs that is `results/runs.jsonl`
(schema v1). Experiments without `bench.py` write `results/<experiment>/points.jsonl`:
each line holds the swept variable, the metrics with units in the key name
(`ttft_ms`, `mem_peak_mib`, `tokens_per_s`) and the context (`gpu_name`, git commit,
`timestamp_utc`). Nesting is fine; columns are addressed by dotted paths.

    python3 scripts/table.py results/runs.jsonl --group <group_id>
    python3 scripts/table.py results/<experiment>/points.jsonl --col batch --col 'ttft_ms=p50@TTFT, ms'

`--col path[=label]` adds one column (repeatable). `--cols` is a shorthand for several:
it splits on a comma not followed by a space, so `p50@TTFT, ms` stays one label.
`raw:<key>` reads a field of the raw `vllm bench serve` result of a bench record.

    python3 scripts/plot_sweep.py results/runs.jsonl --group <group_id> --logx --out <png>

One panel per `--y`, panels share the X axis; `+` puts metrics of the same unit on one
panel. `--x` can be any metric, e.g. `--x results.output_throughput_tps --y results.ttft_ms.p95`.

## Commit rules
Conventional Commits, English, imperative, subject <= 72 chars:
`<type>(<scope>): <summary>`; types: feat, fix, refactor, build, docs, test, chore, exp.
- Commit code BEFORE a run: every record stores the git commit that produced it.
- Commit results AFTER a run as a separate `exp(...)` commit with the key number in the body
  and `Run-Group: <group_id>` in the footer.
- Never amend or force-push a commit that a record points to.
- No employer data in this repository.

Status: Block 0 closed on 2026-10-25; next is Block 1. Block order: 0 -> 1 -> 2 -> 5 -> 4 -> 3.
