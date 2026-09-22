#!/usr/bin/env bash
# Wait for gpu-37's repeat of the v0.29.0 EP1 config, then compare it against gpu-13's run
# of the identical config. Two identical configs on two identical nodes: this says whether
# the -8.4% prefill-mid cell is a real effect or run-to-run variance.
set -uo pipefail
cd /home/amd/NLAIF-benchmarks/application-benchmarks/vLLM
deadline=$(( $(date +%s) + 14400 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    R=$(ls -d results/*_amd-mi355x-n32-ep1 2>/dev/null | grep -v gpu13 | tail -1)
    n=$( [ -n "$R" ] && find "$R" -name summary.json 2>/dev/null | wc -l || echo 0 )
    echo "$(date +%H:%M) gpu37 repeat: ${n}/6"
    [ "$n" = "6" ] && break
    sleep 600
done
echo "=== cross-node comparison, identical config ==="
python3 - <<'PY'
import csv, glob
def load(p): return {(r['input_len'],r['output_len'],r['point']):float(r['tokens_per_gpu']) for r in csv.DictReader(open(p))}
g13 = glob.glob("results/*gpu13_amd-mi355x-n32-ep1/scores.csv")
g37 = [f for f in glob.glob("results/*_amd-mi355x-n32-ep1/scores.csv") if "gpu13" not in f]
ep8 = glob.glob("results/*_amd-mi355x-n32/scores.csv")
if not (g13 and g37 and ep8):
    print("  missing a run; g13=%d g37=%d ep8=%d" % (len(g13), len(g37), len(ep8))); raise SystemExit
a, b, base = load(g13[0]), load(g37[0]), load(ep8[0])
print(f"{'cell':<22}{'EP8':>9}{'EP1 g13':>9}{'EP1 g37':>9}{'spread':>9}")
print("-"*58)
for k in sorted(a, key=lambda k:(k[1],['low','mid','high'].index(k[2]))):
    x, y, z = base[k], a[k], b.get(k, float('nan'))
    spread = abs(y-z)/max(y,z)*100 if z==z else float('nan')
    print(f"{k[0]+'->'+k[1]+' '+k[2]:<22}{x:9.1f}{y:9.1f}{z:9.1f}{spread:8.1f}%")
print("-"*58)
n=len(a)
print(f"{'mean':<22}{sum(base.values())/n:9.1f}{sum(a.values())/n:9.1f}{sum(b.values())/n:9.1f}")
k=("8192","1024","mid")
print(f"\nprefill-mid, the anomaly: EP8={base[k]:.1f}  EP1-g13={a[k]:.1f}  EP1-g37={b.get(k,float('nan')):.1f}")
print("  -> reproduces (real effect)" if b.get(k,0) < base[k] else "  -> does NOT reproduce (was variance)")
PY
python3 docs/generate_findings.py
echo "REPEAT COMPARISON DONE"
