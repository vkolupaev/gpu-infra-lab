#!/usr/bin/env bash
# run_session.sh - Block 1 GPU session: lesson scripts in a fixed order, logs next to the points.
# On the GPU host, inside tmux, from the repo root, image ref exported as LAB_IMAGE:
#   DATE=2026-11-07 PRICE=1.114 R06=<Qwen/Qwen3-0.6B sha> bash pytorch/run_session.sh a b
# Part a: hwcheck, 1.4 (Qwen3-8B, Qwen3-0.6B, nsys), 1.1, debt N.  Part b: 1.2, 1.3, 1.5.
# A failed step is logged and the session goes on; check results/<DATE>-session/session.log.
set -uo pipefail
cd "$(dirname "$0")/.."
D=${DATE:?set DATE=YYYY-MM-DD}
R8=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/qwen3-8b-bf16.yaml'))['revision'])")
S=results/$D-session
mkdir -p "$S"
log() { echo "== $(date -u +%H:%M:%S) $*" | tee -a "$S/session.log"; }
step() { # step <slug> <command...>
  local out=results/$D-$1
  shift
  mkdir -p "$out"
  log "start $out: $*"
  "$@" 2>&1 | tee -a "$out/stdout.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" = 0 ]; then log "ok $out"; else log "FAILED rc=$rc $out"; fi
}
NSYS=(nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node
  --capture-range=cudaProfilerApi --capture-range-end=stop -f true)

python3 scripts/hwcheck.py > "$S/hwcheck.json"
nvidia-smi -q > "$S/nvidia-smi-q.txt"
python3 -c "import json,sys; v=json.load(open('$S/hwcheck.json')).get('verdict'); sys.exit(v != 'OK')" ||
  { log "hwcheck verdict is not OK: destroy the instance, take another host"; exit 1; }
log "hwcheck OK: $(tr -d '\n' < "$S/hwcheck.json" | cut -c1-200)"

for part in "$@"; do
  case $part in
  a)
    R06=${R06:?set R06=<Qwen/Qwen3-0.6B commit sha>}
    python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-8B', revision='$R8'); s('Qwen/Qwen3-0.6B', revision='$R06')" \
      > "$S/hf-download.log" 2>&1 &
    DL=$!
    step h2d-copy python3 pytorch/h2d_copy.py --out "results/$D-h2d-copy"
    wait $DL || log "FAILED model download, see $S/hf-download.log"
    step prefill-decode python3 pytorch/prefill_decode.py --model Qwen/Qwen3-8B --revision "$R8" \
      --out "results/$D-prefill-decode" --busy
    step prefill-decode python3 pytorch/prefill_decode.py --model Qwen/Qwen3-0.6B --revision "$R06" \
      --out "results/$D-prefill-decode" --busy --append
    O=results/$D-prefill-decode-nsys
    step prefill-decode-nsys "${NSYS[@]}" -o "$O/decode" python3 pytorch/prefill_decode.py \
      --model Qwen/Qwen3-8B --revision "$R8" --out "$O" --nsys-decode
    step prefill-decode-nsys nsys stats -f csv -o "$O/decode" \
      -r cuda_gpu_kern_sum,cuda_api_sum,cuda_kern_exec_sum,nvtx_gpu_proj_sum "$O/decode.nsys-rep"
    step train-step python3 pytorch/train_step.py --out "results/$D-train-step" --trace
    O=results/$D-train-step-nsys
    step train-step-nsys "${NSYS[@]}" -o "$O/train" python3 pytorch/train_step.py \
      --out "$O/points" --precision bf16 --steps 10 --nsys-range
    step train-step-nsys nsys stats -f csv -o "$O/train" \
      -r nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum "$O/train.nsys-rep"
    log "start debt N: bench.py, C=8, random_range_ratio 0.5"
    python3 bench.py run --config configs/qwen3-8b-bf16-rr05.yaml --provider vast \
      --price-per-hour "${PRICE:?set PRICE}" --tag waves-rr05 ${BENCH_ARGS:-} 2>&1 | tee -a "$S/bench-rr05.log"
    log "debt N rc=${PIPESTATUS[0]}"
    ;;
  b)
    step backward-cost python3 pytorch/backward_cost.py --out "results/$D-backward-cost"
    step activation-memory python3 pytorch/activation_memory.py --out "results/$D-activation-memory"
    step attention-ops python3 pytorch/attention_ops.py --out "results/$D-attention-ops" --profile-seq 4096
    O=results/$D-compile
    mkdir -p "$O"
    log "start $O (part A, TORCH_LOGS=output_code -> output_code.txt)"
    TORCH_LOGS=output_code python3 pytorch/compile_speedup.py --out "$O" 2> "$O/output_code.txt" | tee -a "$O/stdout.log"
    log "compile part A rc=${PIPESTATUS[0]}"
    step compile timeout 1500 python3 pytorch/compile_speedup.py --out "$O" --append --hf --revision "$R8"
    ;;
  *) log "unknown part $part (use a, b)" ;;
  esac
done
log "done. On the laptop: copy results/ and experiments/ back, then DESTROY the instance."
