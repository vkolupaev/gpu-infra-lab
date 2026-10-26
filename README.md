# gpu-infra-lab

Reproducible measurements of LLM inference systems and their economics:
throughput, latency (p50/p95/p99), TTFT, and cost per 1M tokens on rented GPUs.

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

## Commit rules
Conventional Commits, English, imperative, subject <= 72 chars:
`<type>(<scope>): <summary>`; types: feat, fix, refactor, build, docs, test, chore, exp.
- Commit code BEFORE a run: every record stores the git commit that produced it.
- Commit results AFTER a run as a separate `exp(...)` commit with the key number in the body
  and `Run-Group: <group_id>` in the footer.
- Never amend or force-push a commit that a record points to.
- No employer data in this repository.
