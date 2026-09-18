#!/bin/bash
# Orchestrator job: runs every operating point for one cluster, then plots + scores.
#
# Usually launched via ./run.sh <cluster>, but can be submitted directly:
#     CLUSTER=snellius/h100 sbatch scripts/vllm_sweep_parallel.sh
#     CLUSTER=snellius/h100 bash  scripts/vllm_sweep_parallel.sh   # in an allocation
#
#SBATCH --job-name=vllm-sweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/sweep-%j.out
#SBATCH --error=logs/sweep-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && ! [ -t 0 ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

CLUSTER="${CLUSTER:-snellius/h100}"
CONFIG_ROOT="$REPO_ROOT/configs/$CLUSTER"
[ -f "$CONFIG_ROOT/system.sh" ] || { echo "No system.sh for cluster '$CLUSTER'"; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG_ROOT/system.sh"

DATASET_DIR="$REPO_ROOT/datasets"
CONTAINER_EXEC="${CONTAINER_EXEC:-}"

# ── base vllm commands ────────────────────────────────────────────────────────
# sweep_parallel.py appends per-config flags (model, TP/PP/EP, port, …) from the
# serve/bench JSON. Keep only flags that are constant across every run here.

SERVE_CMD="vllm serve
  --kv-cache-dtype fp8
  --enable-chunked-prefill
  --max-model-len=9216
  --max-num-batched-tokens=9216
  --no-enable-prefix-caching
  --download-dir=$HF_HOME/hub
  --gpu-memory-utilization=${GPU_MEM_UTIL:-0.95}
  --safetensors-load-strategy=prefetch
  --api-server-count=4"

BENCH_CMD="vllm bench serve
  --backend=vllm
  --dataset-name=custom
  --skip-chat-template
  --request-rate=inf
  --ignore-eos
  --save-result
  --percentile-metrics=ttft,tpot,itl,e2el
  --ready-check-timeout-sec=600"

# ── run every (layout, operating point, shape) in the cluster ─────────────────
mkdir -p logs
RESULTS_DIR="${RESULTS_DIR:-results/$(date +%Y%m%d_%H%M%S)}"

# Orchestrator runs on the host python (needs sbatch/squeue/sacct), NOT in the container.
/usr/bin/python3.11 "$REPO_ROOT/src/sweep_parallel.py" \
    --config-root="$CONFIG_ROOT" \
    --serve-cmd="$SERVE_CMD" \
    --bench-cmd="$BENCH_CMD" \
    --link-vars="model=model,max_num_seqs=_n" \
    --results-dir="$RESULTS_DIR" \
    --dataset-dir="$DATASET_DIR" \
    --gpus-per-node="${GPUS_PER_NODE:-0}" \
    ${GPU_REQUEST_TMPL:+--gpu-request-tmpl="$GPU_REQUEST_TMPL"} \
    ${WORKER_TIME:+--worker-time="$WORKER_TIME"} \
    ${SERVE_TIMEOUT:+--serve-timeout="$SERVE_TIMEOUT"} \
    ${DRY_RUN:+--dry-run} \
    ${SBATCH_ARGS:+--sbatch-args="$SBATCH_ARGS"} \
    ${CONTAINER_EXEC:+--container-exec="$CONTAINER_EXEC"}

if [ -n "${DRY_RUN:-}" ]; then
    echo "Dry run complete — no jobs submitted, nothing to plot or score."
    exit 0
fi

# ── plot + score (need pandas/matplotlib → run inside the container) ───────────
# Visual sanity check only — the score comes from score.py, not from the plot.
${CONTAINER_EXEC:-} python3 "$REPO_ROOT/src/utils/plot_pareto.py" "$RESULTS_DIR" \
    --fig-by=model --curve-by=custom-output-len || echo "[warn] plotting failed"

${CONTAINER_EXEC:-} python3 "$REPO_ROOT/src/utils/score.py" "$RESULTS_DIR"

echo "Sweep complete. Results + scores in: $RESULTS_DIR"
