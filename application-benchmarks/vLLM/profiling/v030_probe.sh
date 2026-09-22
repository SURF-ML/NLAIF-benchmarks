#!/usr/bin/env bash
# Does v0.30.0 change the KV pool for DeepSeek-V4, and if so which knob?
# Three startup probes, same checkpoint and layout, only the named flag differing.
# No pkill anywhere: `pkill -f <pattern>` matches this script's own command line and kills it.
set -uo pipefail
cd /home/amd/NLAIF-benchmarks/application-benchmarks/vLLM
REPO_ROOT=$PWD
source configs/amd/mi355x-n32-v030/system.sh
eval "$(sed -n '/^SERVE_CMD=/,/"$/p' scripts/vllm_sweep_parallel.sh)"
OUT=$REPO_ROOT/profiling/v0300; mkdir -p "$OUT"

probe() {   # $1=label  $2=extra/override serve args
    local LOG=$OUT/$1.log
    echo "########## $1  ${2:-(spec-mandated flags)} ##########"
    timeout 900 $CONTAINER_EXEC $SERVE_CMD \
        --model=deepseek-ai/DeepSeek-V4-Pro --trust-remote-code \
        --tokenizer-mode=deepseek_v4 --enable-expert-parallel \
        --tensor-parallel-size=8 --max-num-seqs=32 --port 8013 $2 > "$LOG" 2>&1
    grep -hE "Available KV cache memory|GPU KV cache size|use_fp4_cache|kv_cache_dtype" "$LOG" \
        | sed 's/^([^)]*) //' | sort -u | head -3 | cut -c1-170
    grep -hiE "NotImplementedError|RuntimeError|AssertionError|error:" "$LOG" \
        | sed 's/^([^)]*) //' | sort -u | head -2 | cut -c1-160
    sleep 20
}

probe fp8_specfixed ""
probe auto_dsmla    "--kv-cache-dtype auto"
probe dcp8          "--decode-context-parallel-size 8"

echo "########## summary ##########"
python3 - <<'PY'
import pathlib, re
mla = (512 + 64) * 61 / 1024
print(f"{'config':<34}{'KV GiB':>9}{'pool tok':>12}{'KiB/tok':>9}{'xMLA':>7}{'max conc':>10}")
print("-"*81)
print(f"{'v0.29.0 pinned (baseline)':<34}{165.05:>9.2f}{300533:>12,}{575.9:>9.1f}{16.8:>7.1f}{32.6:>10.1f}")
for p, label in (("fp8_specfixed", "v0.30.0, spec-mandated fp8"),
                 ("auto_dsmla",    "v0.30.0, --kv-cache-dtype auto"),
                 ("dcp8",          "v0.30.0, -dcp 8")):
    f = pathlib.Path(f"profiling/v0300/{p}.log")
    if not f.exists():
        print(f"{label:<34}{'(no log)':>9}"); continue
    t = f.read_text(errors="replace")
    g = re.search(r"Available KV cache memory: ([\d.]+) GiB", t)
    k = re.search(r"GPU KV cache size: ([\d,]+) tokens", t)
    if not (g and k):
        why = "NotImplemented" if "NotImplementedError" in t else ("FAILED" if re.search(r"RuntimeError|error:", t) else "no line")
        print(f"{label:<34}{why:>9}"); continue
    gib, pool = float(g.group(1)), int(k.group(1).replace(",", ""))
    per = gib * 2**30 / pool / 1024
    print(f"{label:<34}{gib:>9.2f}{pool:>12,}{per:>9.1f}{per/mla:>7.1f}{pool/9216:>10.1f}")
print(f"\n  SemiAnalysis (nightly, same shape):      4,386,107 tok   ~39.5 KiB/tok    475.9 conc")
PY
echo "V030 PROBE DONE"
