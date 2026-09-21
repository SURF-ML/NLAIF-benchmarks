#!/usr/bin/env bash
# Wait for both EP1 sweeps, pull gpu-13's results over, regenerate the findings doc and
# commit. Written so the overnight work lands without further prompting.
set -uo pipefail
cd /home/amd/NLAIF-benchmarks/application-benchmarks/vLLM
G13=/home/amd/NLAIF-benchmarks/application-benchmarks/vLLM

cells_local() { R=$(ls -d results/*v023-ep1 2>/dev/null | grep -v smoke | tail -1); find "$R" -name summary.json 2>/dev/null | wc -l; }
cells_g13()   { ssh -o BatchMode=yes -o ConnectTimeout=10 mi355-gpu-13 "R=\$(ls -d $G13/results/*gpu13_amd-mi355x-n32-ep1 2>/dev/null | tail -1); find \"\$R\" -name summary.json 2>/dev/null | wc -l" 2>/dev/null || echo 0; }

# Bounded wait: 5 h, then finalise with whatever exists rather than hanging forever.
deadline=$(( $(date +%s) + 18000 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    l=$(cells_local); g=$(cells_g13)
    echo "$(date +%H:%M) local v023-ep1 ${l}/6   gpu13 v029-ep1 ${g}/6"
    [ "$l" = "6" ] && [ "$g" = "6" ] && break
    sleep 600
done

echo "=== pulling gpu-13 results ==="
rsync -az mi355-gpu-13:"$G13/results/" results/ 2>&1 | tail -2
echo "  local result dirs now: $(ls -d results/2026* 2>/dev/null | wc -l)"

echo "=== regenerating findings ==="
python3 docs/generate_findings.py

echo "=== committing ==="
git add -A docs logs/mfma-counters.txt configs/amd scripts/gen_configs.py profiling 2>/dev/null
if ! git diff --cached --quiet; then
    git commit -q -m "Findings: EP1 sweeps complete, regenerate results document

Pulls mi355-gpu-13's v0.29.0 EP1 sweep over and regenerates
docs/AMD-MI355X-FINDINGS.md from every scores.csv on disk.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N61ZH2WSxtF5S2Lprgyj8y"
    echo "  committed: $(git log --oneline -1)"
else
    echo "  nothing to commit"
fi
echo "FINALIZE DONE"
