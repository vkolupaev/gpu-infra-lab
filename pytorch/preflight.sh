#!/usr/bin/env bash
# preflight.sh - environment checks before a GPU session (~2 minutes).
# On the host, from the repo root, after the destroy timer `( sleep 9000; vastai destroy instance <ID> ) &`
# and before pytorch/run_session.sh:
#   DATE=2026-11-21 R06=<Qwen/Qwen3-0.6B sha> [LAB_IMAGE=<ref@digest>] bash pytorch/preflight.sh
# Prints one line per check (PASS / WARN / FAIL / SKIP) and a DECISION; paste the summary into the chat.
# It only reports: never stops, destroys or deletes anything, always exits 0 (except a usage error).
# Writes only untracked files: results/<DATE>-session/preflight*.
# Test input without a GPU: PREFLIGHT_HWCHECK_JSON=<file> parses that file instead of running hwcheck.py.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
D=${DATE:-}
if ! [[ $D =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "usage: DATE=YYYY-MM-DD [R06=<sha>] [LAB_IMAGE=<ref>] bash pytorch/preflight.sh" >&2
  exit 2
fi
S=results/$D-session
mkdir -p "$S" || exit 2
REPORT=$S/preflight.txt
: > "$REPORT"
TMP=$(mktemp -d) || TMP=$S/preflight-tmp
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

WARNS=()
FAILS=()
NO_GPU=""
TORCH_K=""  # kernel counts seen by torch.profiler and nsys: number, 0, or empty = unknown
NSYS_K=""

out() { echo "$*" | tee -a "$REPORT"; }
report() { # report <STATUS> <name> <value> [action]
  local line="$1 $2: $3"
  [ -n "${4:-}" ] && line="$line -> $4"
  out "$line"
  case $1 in
  WARN) WARNS+=("$2") ;;
  FAIL) FAILS+=("$2") ;;
  esac
}
oneline() { tr '\n' ' ' | tr -s ' ' | sed 's/^ //; s/ $//' | cut -c1-200; }

SMOKE='import torch
x = torch.randn(1024, 1024, device="cuda")
(x @ x).sum().item()'

detect_gpu() {
  if ! command -v nvidia-smi > /dev/null; then
    NO_GPU="nvidia-smi not found"
  elif ! timeout 120 python3 -c "import sys, torch; sys.exit(not torch.cuda.is_available())" > /dev/null 2>&1; then
    NO_GPU="nvidia-smi found, but torch.cuda.is_available() is False or torch failed to import"
  fi
}

HW=""
check_gpu() {
  if [ -n "${PREFLIGHT_HWCHECK_JSON:-}" ]; then
    HW=$PREFLIGHT_HWCHECK_JSON
    out "   (test input: $HW instead of running hwcheck.py)"
  elif [ -n "$NO_GPU" ]; then
    report SKIP gpu "no GPU"
    return
  else
    HW=$S/preflight-hwcheck.json
    if ! timeout 600 python3 scripts/hwcheck.py > "$HW" 2> "$S/preflight-hwcheck.err"; then
      report FAIL gpu "hwcheck.py failed, see $S/preflight-hwcheck.err" "change host"
      HW=""
      return
    fi
  fi
  local r
  r=$(python3 - "$HW" << 'EOF' 2>&1
import json, sys
h = json.load(open(sys.argv[1]))
ok = "A100-SXM4-80GB" in h.get("gpu_name", "") and h.get("verdict") == "OK"
print("PASS" if ok else "FAIL", f"{h.get('gpu_name')}, hbm {h.get('hbm_pct_of_spec')}%, "
      f"gemm {h.get('gemm_pct_of_spec')}% of spec, verdict {h.get('verdict')}")
EOF
  )
  case $r in
  PASS\ *) report PASS gpu "${r#PASS }" ;;
  FAIL\ *) report FAIL gpu "${r#FAIL }" "change host" ;;
  *) report FAIL gpu "cannot parse $HW: $(echo "$r" | tail -n 1 | oneline)" "change host" ;;
  esac
}

check_pcie() {
  if [ -z "$HW" ]; then
    if [ -n "$NO_GPU" ] && [ -z "${PREFLIGHT_HWCHECK_JSON:-}" ]; then report SKIP pcie "no GPU"
    else report WARN pcie "no hwcheck output" "go, note \"other PCIe mode\" in the h2d-copy record"; fi
    return
  fi
  local r gen width
  r=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('nvidia_smi_after_load', ''))" "$HW" 2>&1)
  gen=$(echo "$r" | cut -d, -f1 | tr -d ' ')
  width=$(echo "$r" | cut -d, -f2 | tr -d ' ')
  if [ "$gen" = 4 ] && [ "$width" = 16 ]; then
    report PASS pcie "gen $gen, width $width"
  else
    report WARN pcie "gen ${gen:-?}, width ${width:-?} (expected gen 4, width 16)" \
      "go, note \"other PCIe mode\" in the h2d-copy record"
  fi
}

check_torch_profiler() {
  if [ -n "$NO_GPU" ]; then report SKIP torch-profiler "no GPU"; return; fi
  local r
  r=$(timeout 180 python3 - << EOF 2>&1
from torch.profiler import profile, ProfilerActivity
import torch
$SMOKE
with profile(activities=[ProfilerActivity.CUDA]) as p:
    (x @ x).sum().item()
print("CUDA_EVENTS", sum(e.device_type == torch.autograd.DeviceType.CUDA for e in p.events()))
EOF
  )
  local n
  n=$(echo "$r" | sed -n 's/^CUDA_EVENTS \([0-9][0-9]*\)$/\1/p' | tail -n 1)
  if [ -z "$n" ]; then
    TORCH_K=0
    report WARN torch-profiler "test failed: $(echo "$r" | tail -n 1 | oneline)" \
      "profiler fields will be empty; kernel counts from nsys stats"
  elif [ "$n" -gt 0 ]; then
    TORCH_K=$n
    report PASS torch-profiler "$n CUDA events"
  else
    TORCH_K=0
    report WARN torch-profiler "0 CUDA events" "profiler fields will be empty; kernel counts from nsys stats"
  fi
}

check_nsys() {
  if [ -n "$NO_GPU" ]; then report SKIP nsys "no GPU"; return; fi
  local act="nsys steps will fail; points without nsys are still recorded"
  local log=$S/preflight-nsys.log
  if ! command -v nsys > /dev/null; then
    NSYS_K=0
    report WARN nsys "nsys not found" "$act"
    return
  fi
  if ! timeout 300 nsys profile -t cuda --sample=none --cpuctxsw=none -f true -o "$TMP/smoke" \
    python3 -c "$SMOKE" > "$log" 2>&1 || ! [ -f "$TMP/smoke.nsys-rep" ]; then
    NSYS_K=0
    report WARN nsys "nsys profile failed, see $log" "$act"
    return
  fi
  timeout 300 nsys stats -r cuda_gpu_kern_sum -f csv -o "$TMP/smoke" "$TMP/smoke.nsys-rep" >> "$log" 2>&1
  local csv
  csv=$(find "$TMP" -maxdepth 1 -name 'smoke*cuda_gpu_kern_sum*.csv' | head -n 1)
  if [ -n "$csv" ]; then
    cp "$csv" "$S/preflight-nsys-kern.csv"
    NSYS_K=$(awk 'NF' "$csv" | tail -n +2 | wc -l)
    if [ "$NSYS_K" -gt 0 ]; then report PASS nsys "$NSYS_K kernel rows in cuda_gpu_kern_sum"
    else report WARN nsys "0 kernel rows in cuda_gpu_kern_sum, see $S/preflight-nsys-kern.csv" "$act"; fi
  elif grep -q 'does not contain' "$log"; then # nsys writes no CSV for a report without kernel data
    NSYS_K=0
    report WARN nsys "no CUDA kernel data in the report, see $log" "$act"
  else
    report WARN nsys "could not parse stats, see $log" "$act"
  fi
}

check_profilers() {
  if [ -n "$NO_GPU" ]; then report SKIP profilers "no GPU"; return; fi
  local v="torch.profiler ${TORCH_K:-?}, nsys ${NSYS_K:-?}"
  if [ "$TORCH_K" = 0 ] && [ "$NSYS_K" = 0 ]; then
    report FAIL profilers "$v" "no kernel tracing on this host: change host"
  elif [ "${TORCH_K:-0}" != 0 ] || [ "${NSYS_K:-0}" != 0 ]; then
    report PASS profilers "$v"
  else
    report WARN profilers "$v (nsys result unknown)" "check nsys stats by hand (gpu-only-risks 5a)"
  fi
}

free_gb() { # free_gb <dir>: nearest existing parent, prints "<GB> <mount>"
  local d=$1
  while ! [ -e "$d" ] && [ "$d" != / ]; do d=$(dirname "$d"); done
  df -Pk "$d" | awk 'NR == 2 { printf "%.1f %s\n", $4 * 1024 / 1e9, $6 }'
}

check_disk() {
  local hf=${HF_HOME:-$HOME/.cache/huggingface} a b min v
  a=$(free_gb .)
  b=$(free_gb "$hf")
  if [ -z "$a" ] || [ -z "$b" ]; then
    report WARN disk "df failed" "check df -h by hand: >= 40 GB free"
    return
  fi
  if [ "${a#* }" = "${b#* }" ]; then
    v="${a% *} GB free on ${a#* } (repo and HF cache)"
    min=${a% *}
  else
    v="repo ${a% *} GB on ${a#* }, HF cache ${b% *} GB on ${b#* }"
    min=$(printf '%s\n%s\n' "${a% *}" "${b% *}" | sort -g | head -n 1)
  fi
  if awk -v m="$min" 'BEGIN { exit !(m >= 40) }'; then report PASS disk "$v"
  elif awk -v m="$min" 'BEGIN { exit !(m >= 25) }'; then report WARN disk "$v (< 40 GB)" "recreate with a bigger disk"
  else report FAIL disk "$v (< 25 GB)" "recreate with a bigger disk"; fi
}

check_insurance() {
  local timer vast
  timer=$(pgrep -af 'sleep 9000' | head -n 1)
  vast=$(command -v vastai)
  if [ -n "$timer" ] && [ -n "$vast" ]; then
    report PASS insurance "timer pid ${timer%% *}, vastai at $vast"
  elif [ -z "$timer" ] && [ -z "$vast" ]; then
    report FAIL insurance "no 'sleep 9000' process, no vastai" "start the destroy timer first; timer cannot destroy the instance"
  elif [ -z "$timer" ]; then
    report FAIL insurance "no 'sleep 9000' process" "start the destroy timer first"
  else
    report FAIL insurance "vastai not found" "timer cannot destroy the instance"
  fi
}

check_hf() {
  local r8 r v="" bad=""
  r8=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/qwen3-8b-bf16.yaml'))['revision'])" 2>&1) || {
    report FAIL hf "cannot read revision from configs/qwen3-8b-bf16.yaml: $(echo "$r8" | tail -n 1 | oneline)" \
      "fix HF access in 10 min, else change host"
    return
  }
  local pairs=("Qwen/Qwen3-8B $r8")
  if [ -n "${R06:-}" ]; then pairs+=("Qwen/Qwen3-0.6B $R06"); fi
  local p repo rev
  for p in "${pairs[@]}"; do
    repo=${p% *}
    rev=${p#* }
    r=$(timeout 60 python3 -c "
import sys
from huggingface_hub import HfApi
print('SHA', HfApi().model_info(sys.argv[1], revision=sys.argv[2]).sha)" "$repo" "$rev" 2>&1)
    if echo "$r" | grep -q '^SHA '; then
      v="$v$repo@${rev:0:12} found; "
    else
      bad=1
      v="$v$repo@${rev:0:12}: $(echo "$r" | tail -n 1 | oneline); "
    fi
  done
  [ -z "${R06:-}" ] && v="${v}Qwen/Qwen3-0.6B skipped (R06 not set); "
  v=${v%; }
  if [ -z "$bad" ]; then report PASS hf "$v"; else report FAIL hf "$v" "fix HF access in 10 min, else change host"; fi
}

check_git() {
  local head st tracked untracked
  head=$(git rev-parse --short HEAD 2>&1) || { report WARN git "not a git checkout: $head" "records get git.dirty = null"; return; }
  # git.dirty (bench.py git_info) ignores untracked files; this run's own session directory is untracked
  st=$(git status --short 2>&1 | grep -v "^?? $S/\$")
  tracked=$(echo "$st" | grep -c '^[^?]')
  untracked=$(echo "$st" | grep -c '^??')
  if [ "$tracked" -gt 0 ]; then
    report WARN git "HEAD $head, $tracked changed tracked file(s): $(echo "$st" | grep '^[^?]' | oneline)" \
      "commit or stash before the session (git.dirty)"
  elif [ "$untracked" -gt 0 ]; then
    report PASS git "HEAD $head, clean ($untracked untracked, not in git.dirty: $(echo "$st" | grep '^??' | oneline))"
  else
    report PASS git "HEAD $head, clean"
  fi
}

check_image() {
  if [ -z "${LAB_IMAGE:-}" ]; then report SKIP image "LAB_IMAGE not set"; return; fi
  local ref=${LAB_IMAGE%@*} digest=${LAB_IMAGE##*@} tag
  tag=${ref##*:}
  # notes/images.md has tag and digest in separate columns of one table row
  if [ "$digest" != "$LAB_IMAGE" ] && grep -F "\`$tag\`" notes/images.md | grep -qF "\`$digest\`"; then
    report PASS image "$tag@${digest:0:19} in notes/images.md"
  else
    report WARN image "$LAB_IMAGE not in notes/images.md (tag and digest in one row)" "image differs from notes/images.md"
  fi
}

out "== preflight $D $(date -u +%Y-%m-%dT%H:%M:%SZ) host $(hostname) commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
detect_gpu
[ -n "$NO_GPU" ] && out "   no GPU: $NO_GPU"
check_gpu
check_pcie
check_torch_profiler
check_nsys
check_profilers
check_disk
check_insurance
check_hf
check_git
check_image

if [ -n "$NO_GPU" ]; then DEC="NO GPU (dry run)"
elif [[ " ${FAILS[*]} " == *" insurance "* ]] && [ "${#FAILS[@]}" = 1 ]; then DEC="FIX AND RERUN"
elif [ "${#FAILS[@]}" -gt 0 ]; then DEC="CHANGE HOST"
elif [ "${#WARNS[@]}" -gt 0 ]; then DEC="GO WITH NOTES"
else DEC="GO"; fi
out "DECISION: $DEC"
out "notes: ${WARNS[*]:--}"
out "fail:  ${FAILS[*]:--}"
out "paste everything from \"== preflight\" to here into the chat"
exit 0
