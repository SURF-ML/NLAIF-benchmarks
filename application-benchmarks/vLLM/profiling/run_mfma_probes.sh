#!/usr/bin/env bash
# Run the MFMA counter probes once the sweeps are done. Control first: if the control
# shows zero counters the real results are uninterpretable and we say so rather than
# concluding "no FP4".
set -uo pipefail
cd /home/amd/NLAIF-benchmarks/application-benchmarks/vLLM
REPO_ROOT=$PWD
OUT=$REPO_ROOT/profiling/results
mkdir -p "$OUT"
PMC="SQ_INSTS_VALU_MFMA_MOPS_F8 SQ_INSTS_VALU_MFMA_MOPS_F6F4 SQ_INSTS_VALU_MFMA_MOPS_BF16"

run_probe() {   # $1=label  $2=cluster-config  $3=model  $4=EP(0/1)
    local label="$1" cfg="$2" model="$3" ep="$4"
    echo "########## $label ##########"
    ( source "$REPO_ROOT/configs/$cfg/system.sh"
      PROBE_EP=$ep $CONTAINER_EXEC bash -c "
        cd $REPO_ROOT
        export PROBE_EP=$ep
        rocprofv3 --pmc $PMC -d $OUT/$label -o $label -- \
            python3 profiling/mfma_probe.py '$model' 2>&1 | tail -25
      " ) > "$OUT/$label.log" 2>&1
    echo "  exit=$? log=$OUT/$label.log"
}

# Control — does counter collection work at all here?
echo "########## control ##########"
( source "$REPO_ROOT/configs/amd/mi355x-n32/system.sh"
  $CONTAINER_EXEC bash -c "cd $REPO_ROOT && rocprofv3 --pmc $PMC -d $OUT/control -o control -- \
      python3 profiling/control_matmul.py 2>&1 | tail -15" ) > "$OUT/control.log" 2>&1
echo "  control exit=$? log=$OUT/control.log"

# W4A8 on the pinned engine vs W4A4 on v0.23.0 — the comparison that matters
run_probe w4a8_v029_ep8 amd/mi355x-n32          deepseek-ai/DeepSeek-V4-Pro 1
run_probe w4a4_v023_ep1 amd/mi355x-n32-v023-ep1 deepseek-ai/DeepSeek-V4-Pro 0

echo "########## counter totals ##########"
for d in control w4a8_v029_ep8 w4a4_v023_ep1; do
    echo "--- $d ---"
    f=$(find "$OUT/$d" -name "*counter_collection.csv" 2>/dev/null | head -1)
    if [ -n "$f" ]; then
        python3 - "$f" <<'PY'
import csv, sys, collections
tot = collections.Counter()
with open(sys.argv[1]) as fh:
    for r in csv.DictReader(fh):
        name = r.get("Counter_Name") or r.get("Counter")
        val = r.get("Counter_Value") or r.get("Value") or "0"
        try: tot[name] += float(val)
        except ValueError: pass
for k, v in sorted(tot.items()):
    print(f"  {k:36s} {v:,.0f}")
f8, f6f4 = tot.get("SQ_INSTS_VALU_MFMA_MOPS_F8",0), tot.get("SQ_INSTS_VALU_MFMA_MOPS_F6F4",0)
if f8 + f6f4 > 0:
    print(f"  -> 4/6-bit share of low-precision MFMA: {100*f6f4/(f8+f6f4):.2f}%")
PY
    else
        echo "  no counter CSV produced (see $d.log)"
    fi
done
echo "MFMA PROBES DONE"
