#!/usr/bin/env bash
# MFMA operand-width counters on mi355-gpu-13, run standalone (no repo config needed).
#
# gfx950 counts 8-bit and 4/6-bit matrix ops separately:
#   SQ_INSTS_VALU_MFMA_MOPS_F8     8-bit
#   SQ_INSTS_VALU_MFMA_MOPS_F6F4   4/6-bit
# vLLM's logs say v0.29.0 serves DeepSeek-V4 as W4A8 (afp8_wfp4) and v0.23.0 as W4A4
# (fp4 x fp4). This checks that claim against the hardware. EP is held ON for both, so the
# only thing that varies is the engine.
set -uo pipefail
OUT=/home/amd/mfma_probe; mkdir -p "$OUT"
PMC="SQ_INSTS_VALU_MFMA_MOPS_F8 SQ_INSTS_VALU_MFMA_MOPS_F6F4 SQ_INSTS_VALU_MFMA_MOPS_BF16"
RUNFLAGS=(--rm --device /dev/kfd --device /dev/dri --ipc=host --network=host
          --security-opt seccomp=unconfined --cap-add=SYS_PTRACE
          -v /home/amd:/home/amd -w /home/amd
          -e HF_HOME=/dev/shm/hf_hub -e HF_HUB_OFFLINE=1 -e VLLM_ROCM_USE_AITER=1
          -e PYTHONNOUSERSITE=1)

probe() {   # $1=label  $2=image
    echo "########## $1 ##########"
    podman run "${RUNFLAGS[@]}" "$2" bash -c "
        rocprofv3 --pmc $PMC -d $OUT/$1 -o $1 -- python3 /home/amd/mfma_probe/probe.py
    " > "$OUT/$1.log" 2>&1
    echo "  exit=$?  log=$OUT/$1.log"
}

echo "########## control (do the counters register at all?) ##########"
podman run "${RUNFLAGS[@]}" localhost/vllm-bench:amd-v0.29.0 bash -c "
    rocprofv3 --pmc $PMC -d $OUT/control -o control -- python3 /home/amd/mfma_probe/control.py
" > "$OUT/control.log" 2>&1
echo "  exit=$?  log=$OUT/control.log"

probe w4a8_v029 localhost/vllm-bench:amd-v0.29.0
probe w4a4_v023 localhost/vllm-bench:amd-v0.23.0

echo "########## counter totals ##########"
for d in control w4a8_v029 w4a4_v023; do
    echo "--- $d ---"
    f=$(find "$OUT/$d" -name "*counter_collection.csv" 2>/dev/null | head -1)
    [ -z "$f" ] && { echo "  no counter CSV (see $OUT/$d.log)"; continue; }
    python3 - "$f" <<'PY'
import csv, sys, collections
tot = collections.Counter()
for r in csv.DictReader(open(sys.argv[1])):
    n = r.get("Counter_Name") or r.get("Counter")
    try: tot[n] += float(r.get("Counter_Value") or r.get("Value") or 0)
    except ValueError: pass
for k, v in sorted(tot.items()):
    print(f"  {k:36s} {v:,.0f}")
f8 = tot.get("SQ_INSTS_VALU_MFMA_MOPS_F8", 0); f4 = tot.get("SQ_INSTS_VALU_MFMA_MOPS_F6F4", 0)
if f8 + f4:
    print(f"  -> 4/6-bit share of low-precision MFMA: {100*f4/(f8+f4):.2f}%")
else:
    print("  -> no low-precision MFMA counted")
PY
done
echo "GPU13 MFMA DONE"
