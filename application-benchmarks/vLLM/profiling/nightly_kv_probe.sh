#!/usr/bin/env bash
# Does the pinned v0.29.0 allocate 16x more KV per token than it needs?
#
# SemiAnalysis run the same model, hardware and TP8/EP1 layout on a vLLM nightly and report
# a KV pool of 4,386,107 tokens (~39.5 KiB/token, ~1.1x MLA theoretical). We measure 300,533
# (575.9 KiB/token, 16.8x theoretical) on v0.29.0. Everything else has been ruled out — DCP
# is unsupported for this model, their metric is not summing DP ranks (dp_attn=False), the
# CUDA-graph profiler costs 0.7%, and their recipe sets no memory-utilisation override.
#
# So: same serve flags, same checkpoint, only the image differs. Startup only.
set -uo pipefail
cd /home/amd/NLAIF-benchmarks/application-benchmarks/vLLM
REPO_ROOT=$PWD
source configs/amd/mi355x-n32/system.sh
eval "$(sed -n '/^SERVE_CMD=/,/"$/p' scripts/vllm_sweep_parallel.sh)"
OUT=$REPO_ROOT/profiling/nightly; mkdir -p "$OUT"
# The exact nightly SemiAnalysis pinned has been garbage-collected (nightly tags rotate),
# so probe the released v0.30.0 instead: reproducible, and a concrete candidate for re-pinning.
NIGHTLY=docker.io/vllm/vllm-openai-rocm:v0.30.0

probe() {  # $1=label $2=image
    local NAME=kvprobe_$1 LOG=$OUT/$1.log
    echo "########## $1 ##########"
    podman rm -f "$NAME" >/dev/null 2>&1
    ${CONTAINER_EXEC/podman run --rm/podman run --rm --name $NAME} $SERVE_CMD \
        --model=deepseek-ai/DeepSeek-V4-Pro --trust-remote-code \
        --tokenizer-mode=deepseek_v4 --enable-expert-parallel \
        --tensor-parallel-size=8 --max-num-seqs=32 --port 8012 > "$LOG" 2>&1 &
    for _ in $(seq 1 300); do
        grep -qE "Maximum concurrency|RuntimeError|ValueError|AssertionError|Engine core init|not supported|out of memory" "$LOG" 2>/dev/null && break
        sleep 5
    done
    grep -hE "Available KV cache memory|GPU KV cache size|use_fp4_cache|indexer cache" "$LOG" \
        | sed 's/^([^)]*) //' | sort -u | head -4 | cut -c1-165
    grep -hiE "RuntimeError|AssertionError|not supported|out of memory" "$LOG" \
        | sed 's/^([^)]*) //' | sort -u | head -2 | cut -c1-150
    podman rm -f "$NAME" >/dev/null 2>&1
    sleep 20
}

# The nightly ships no vllm[bench]/ray layer, but we only need `vllm serve` here.
CONTAINER_EXEC="${CONTAINER_EXEC/$CONTAINER_IMAGE/$NIGHTLY}" probe v0300 "$NIGHTLY"

echo "########## comparison ##########"
python3 - <<'PY'
import pathlib, re
mla = (512 + 64) * 61 / 1024
rows = [("v0.29.0 (pinned, measured earlier)", 165.05, 300_533)]
f = pathlib.Path("profiling/nightly/v0300.log")
if f.exists():
    t = f.read_text(errors="replace")
    g = re.search(r"Available KV cache memory: ([\d.]+) GiB", t)
    k = re.search(r"GPU KV cache size: ([\d,]+) tokens", t)
    if g and k:
        rows.append(("v0.30.0", float(g.group(1)), int(k.group(1).replace(",", ""))))
    else:
        print("  v0.30.0 produced no KV line — see profiling/nightly/v0300.log")
print(f"{'image':<38}{'KV GiB':>9}{'pool tok':>12}{'KiB/tok':>10}{'x MLA':>8}{'max conc':>10}")
print("-"*87)
for name, gib, pool in rows:
    per = gib * 2**30 / pool / 1024
    print(f"{name:<38}{gib:>9.2f}{pool:>12,}{per:>10.1f}{per/mla:>8.1f}{pool/9216:>10.1f}")
print(f"\n  SemiAnalysis reported, same shape:            4,386,107 tokens   ~39.5 KiB/tok   475.9 conc")
PY
echo "NIGHTLY KV PROBE DONE"
