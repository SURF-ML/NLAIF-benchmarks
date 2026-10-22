#!/bin/bash
# One-shot benchmark runner: run every measurement for a cluster, then plot + score.
#
#   ./run.sh <cluster> [--local] [--dry-run]
#
#   <cluster>        configs/<cluster>/ must exist (e.g. snellius/h100).
#                    Defaults to $CLUSTER, else snellius/h100.
#   --local          run the orchestrator in this shell instead of submitting it as a
#                    job (use inside an existing allocation / for debugging).
#   --dry-run        print the worker allocation each model would request and exit;
#                    submits nothing, needs no GPUs. Implies --local.
#
# Requires SLURM: one sbatch job per measurement (multi-node via srun + Ray). The orchestrator runs
# here (--local, e.g. on a login/interactive node) or as a submitted job (default).
# Datasets are downloaded from the organisers (see README §4) and must not be regenerated.
#
# Environment overrides (all optional): HF_HOME, SBATCH_ARGS (worker partition/account),
# ORCH_SBATCH_ARGS (orchestrator job partition; defaults to SBATCH_ARGS), CONTAINER_EXEC.
#
# Examples:
#   ./run.sh snellius/h100 --dry-run
#   SBATCH_ARGS="-p gpu_h100 --account=proj1" ./run.sh snellius/h100
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CLUSTER="${CLUSTER:-snellius/h100}"
LOCAL=0
DRY_RUN=""
for arg in "$@"; do
    case "$arg" in
        --local)        LOCAL=1 ;;
        --dry-run)      LOCAL=1; export DRY_RUN=1 ;;
        -*)             echo "Unknown flag: $arg" >&2; exit 1 ;;
        *)              CLUSTER="$arg" ;;
    esac
done

CONFIG_ROOT="$REPO_ROOT/configs/$CLUSTER"
if [ ! -f "$CONFIG_ROOT/system.sh" ]; then
    echo "No cluster config at configs/$CLUSTER/system.sh. Available clusters:" >&2
    find configs -name system.sh -printf '  %h\n' | sed 's#configs/##' >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_ROOT/system.sh"
export CLUSTER

mkdir -p logs

echo "Cluster:        $CLUSTER  (GPUS_PER_NODE=${GPUS_PER_NODE:-?})"
echo "Models:        $(find "$CONFIG_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f ' )"
echo "Worker sbatch: ${SBATCH_ARGS:-<none>}"

# ── run every measurement + plot + score ──────────────────────────────────────
if [ "$LOCAL" -eq 1 ]; then
    echo "Running orchestrator locally ..."
    CLUSTER="$CLUSTER" bash scripts/vllm_sweep_parallel.sh
else
    SWEEP_ID=$(CLUSTER="$CLUSTER" sbatch --parsable \
                   ${ORCH_SBATCH_ARGS:-${SBATCH_ARGS}} scripts/vllm_sweep_parallel.sh)
    echo "Submitted sweep orchestrator job: $SWEEP_ID"
    echo "Watch:   tail -f logs/sweep-${SWEEP_ID}.out"
    echo "Results: results/<timestamp>/  (scores.csv + scores.json on completion)"
fi
