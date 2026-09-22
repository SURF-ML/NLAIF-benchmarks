#!/usr/bin/env bash
# What does --decode-context-parallel-size buy on the KV pool, and therefore on N?
#
# DeepSeek uses MLA, whose KV cache is REPLICATED across TP ranks at dcp=1 — measured at
# 575.9 KiB/token against an MLA theoretical of 34.3 KiB/token. DCP shards the decode KV
# across ranks WITHOUT expanding the world size, so TP*PP (the tok/s/GPU denominator) is
# unchanged. Arithmetic alone bounded N at 32..256, an 8x spread, so measure it.
#
# Startup only: the engine prints the KV pool and max concurrency before serving anything.
# Each engine runs in a NAMED container that is removed by name afterwards — an earlier
# version used `pkill -f "vllm serve"`, which matched broadly enough to kill this script.
set -uo pipefail
cd /home/amd/NLAIF-benchmarks/application-benchmarks/vLLM
REPO_ROOT=$PWD
source configs/amd/mi355x-n32/system.sh      # N=32 envelope: GPU_MEM_UTIL=0.97, v0.29.0
eval "$(sed -n '/^SERVE_CMD=/,/"$/p' scripts/vllm_sweep_parallel.sh)"
OUT=$REPO_ROOT/profiling/dcp; mkdir -p "$OUT"

for DCP in 1 2 4 8; do
    NAME=dcpprobe$DCP
    LOG=$OUT/dcp$DCP.log
    echo "########## -dcp $DCP ##########"
    podman rm -f "$NAME" >/dev/null 2>&1
    # CONTAINER_EXEC is "podman run --rm ARGS IMAGE"; splice in a --name we can target.
    ${CONTAINER_EXEC/podman run --rm/podman run --rm --name $NAME} $SERVE_CMD \
        --model=deepseek-ai/DeepSeek-V4-Pro --trust-remote-code \
        --tokenizer-mode=deepseek_v4 --enable-expert-parallel \
        --tensor-parallel-size=8 --max-num-seqs=32 \
        --decode-context-parallel-size=$DCP --port 8010 > "$LOG" 2>&1 &
    for _ in $(seq 1 240); do
        grep -qE "Maximum concurrency|RuntimeError|ValueError|AssertionError|Engine core init|not supported|out of memory" "$LOG" 2>/dev/null && break
        sleep 5
    done
    grep -hE "Available KV cache memory|Maximum concurrency" "$LOG" | sed 's/^([^)]*) //' | sort -u | head -2
    grep -hiE "RuntimeError|ValueError|AssertionError|not supported|out of memory" "$LOG" | sed 's/^([^)]*) //' | sort -u | head -2 | cut -c1-150
    podman rm -f "$NAME" >/dev/null 2>&1
    sleep 20
done

echo "########## summary ##########"
python3 - <<'PY'
import re, pathlib
print(f"{'dcp':>4}{'KV pool (tok)':>16}{'max conc':>10}{'N (pow2)':>10}{'vs dcp=1':>10}")
print("-"*50)
base = None
for d in (1, 2, 4, 8):
    f = pathlib.Path(f"profiling/dcp/dcp{d}.log")
    if not f.exists():
        continue
    t = f.read_text(errors="replace")
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens.*?concurrency for [\d,]+ tokens per request: ([\d.]+)x", t)
    if not m:
        why = "FAILED" if re.search(r"RuntimeError|AssertionError|not supported", t) else "no line"
        print(f"{d:>4}{why:>16}")
        continue
    pool = int(m.group(1).replace(",", "")); conc = float(m.group(2))
    n = 2 ** (int(conc).bit_length() - 1)
    base = base or pool
    print(f"{d:>4}{pool:>16,}{conc:>10.2f}{n:>10}{pool/base:>9.2f}x")
PY
echo "DCP SCAN DONE"
