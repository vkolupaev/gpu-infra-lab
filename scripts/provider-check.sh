#!/usr/bin/env bash
# provider-check.sh - run INSIDE a fresh instance. ~15-25 min, mostly the model download.
# Usage: PROVIDER=vast PRICE=1.10 T_CREATE="2026-10-21T15:02Z" bash scripts/provider-check.sh
#   T_CREATE = when you clicked "create" (UTC); the script start time ~= "SSH ready".
#   Plain vLLM image: add INSTALL_PROF=1 NSYS_VERSION=<pin> NCU_VERSION=<pin> (pins from Dockerfile).
set -uo pipefail
cd "$(dirname "$0")/.."
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=results/provider-check/${PROVIDER:?set PROVIDER}-${TS}
mkdir -p "$OUT"
log() { echo "== $*" | tee -a "$OUT/summary.txt"; }

log "provider=$PROVIDER price_per_hour=${PRICE:-?} t_create=${T_CREATE:-?} t_ssh_ready=$(date -u +%FT%TZ)"

# 1. hardware honesty
nvidia-smi -q > "$OUT/nvidia-smi-q.txt"
nvidia-smi topo -m > "$OUT/topo.txt" 2>&1
nvidia-smi --query-gpu=name,memory.total,power.limit,power.max_limit,clocks.max.sm,clocks.max.memory,pcie.link.gen.max,pcie.link.width.max,driver_version,mig.mode.current --format=csv | tee -a "$OUT/summary.txt"
python3 scripts/hwcheck.py | tee "$OUT/hwcheck.json"
log "cpu: $(nproc) vCPU, $(free -g | awk '/Mem/{print $2}') GB RAM, $(lscpu | awk -F: '/Model name/{print $2}' | xargs)"

# 2. profiler counters (go/no-go for ncu); nsys tracing does not need counters
if [ "${INSTALL_PROF:-0}" = 1 ]; then   # plain vLLM image: install pinned profilers on the fly
  apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    "${NSYS_PKG:-cuda-nsight-systems-12-9}=${NSYS_VERSION:?}" "${NCU_PKG:-cuda-nsight-compute-12-9}=${NCU_VERSION:?}" > "$OUT/apt.txt" 2>&1
  export PATH=/usr/local/cuda/bin:$PATH
fi
TEST='import torch; x=torch.randn(4096,4096,device="cuda"); y=x@x; torch.cuda.synchronize()'
if command -v ncu >/dev/null; then
  ncu --metrics sm__cycles_elapsed.avg -c 1 python3 -c "$TEST" > "$OUT/ncu.txt" 2>&1
  if grep -q ERR_NVGPUCTRPERM "$OUT/ncu.txt"; then log "ncu: NO (ERR_NVGPUCTRPERM)";
  elif grep -q "sm__cycles_elapsed.avg" "$OUT/ncu.txt"; then log "ncu: YES";
  else log "ncu: UNKNOWN, see ncu.txt"; fi
else log "ncu: not installed in image"; fi
if command -v nsys >/dev/null; then
  nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none -o "$OUT/nsys-test" -f true python3 -c "$TEST" > "$OUT/nsys.txt" 2>&1 \
    && log "nsys cuda trace: YES" || log "nsys cuda trace: NO, see nsys.txt"
  nsys profile --gpu-metrics-devices=all --sample=none -o "$OUT/nsys-gm" -f true python3 -c "$TEST" > "$OUT/nsys-gm.txt" 2>&1 \
    && ! grep -qi "insufficient\|permission\|not supported" "$OUT/nsys-gm.txt" \
    && log "nsys gpu-metrics: YES" || log "nsys gpu-metrics: NO, see nsys-gm.txt"
  rm -f "$OUT"/*.nsys-rep "$OUT"/*.sqlite
else log "nsys: not installed in image"; fi
grep -h RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null | tee -a "$OUT/summary.txt"

# 3. MIG: can we switch it on? (expected NO in containers)
nvidia-smi -i 0 -mig 1 > "$OUT/mig.txt" 2>&1; log "mig enable attempt: $(tail -1 "$OUT/mig.txt")"
nvidia-smi -i 0 -mig 0 > /dev/null 2>&1

# 4. disk and network
df -h . | tee -a "$OUT/summary.txt"
dd if=/dev/zero of=./ddtest bs=1M count=4096 oflag=direct 2>&1 | tail -1 | sed 's/^/disk write: /' | tee -a "$OUT/summary.txt"; rm -f ./ddtest
REV=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/qwen3-8b-bf16.yaml'))['revision'])")
T0=$(date +%s)
python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-8B', revision='$REV')" > "$OUT/hf.txt" 2>&1
T1=$(date +%s); SZ=$(du -sb ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B 2>/dev/null | cut -f1)
log "hf download: $((T1-T0)) s, $((SZ/1000000)) MB, $(( SZ / ( (T1-T0>0 ? T1-T0 : 1) * 1000000 ) )) MB/s"
log "done $(date -u +%FT%TZ). Now copy results/provider-check back and DESTROY the instance."
