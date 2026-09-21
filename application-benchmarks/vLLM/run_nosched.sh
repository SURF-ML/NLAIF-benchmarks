#!/usr/bin/env bash
# Non-SLURM driver for the vLLM inference benchmark.
#
# Runs exactly the same measurement code as the SLURM harness — src/sweep_parallel.py
# --worker, one invocation per (serve, bench) pair, same SERVE_CMD/BENCH_CMD, same
# summary.json layout — but replaces the two things that need a scheduler:
#   * the orchestrator's sbatch/squeue loop  →  a serial for-loop here
#   * the worker's srun-based Ray bootstrap  →  plain ssh (so --ray is NOT passed)
#
# Single node (TP*PP <= GPUS_PER_NODE):
#     CLUSTER=amd/mi355x ./run_nosched.sh
# Two nodes, Ray over ssh (first entry is the head = this machine):
#     CLUSTER=amd/mi355x NODES="10.2.84.4 10.2.84.5" ./run_nosched.sh
#
# Container runtimes: apptainer needs nothing beyond CONTAINER_EXEC. podman/docker cannot
# host a Ray daemon through a one-shot `run` (the container dies when `ray start` returns),
# so when NODES is set this script boots ONE persistent container per node from
# CONTAINER_RUN_ARGS + CONTAINER_IMAGE and drives everything through `podman exec`.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CLUSTER="${CLUSTER:?set CLUSTER, e.g. amd/mi355x}"
CONFIG_ROOT="$REPO_ROOT/configs/$CLUSTER"
[ -f "$CONFIG_ROOT/system.sh" ] || { echo "No system.sh for cluster '$CLUSTER'"; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG_ROOT/system.sh"

DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-8000}"
RAY_PORT="${RAY_PORT:-6379}"
NODES="${NODES:-}"                 # "head other ..."; empty => single node, no Ray
CONTAINER_EXEC="${CONTAINER_EXEC:-}"
CONTAINER_ENGINE="${CONTAINER_ENGINE:-podman}"
CNAME="${CNAME:-vllm-bench-$USER}"
mkdir -p tmp logs "$RESULTS_DIR"

# The Fixed parameters have exactly one home: scripts/vllm_sweep_parallel.sh. Lift the two
# command strings straight out of it rather than copying them, so this driver can never
# drift from the spec when they change there.
eval "$(sed -n '/^SERVE_CMD=/,/"$/p; /^BENCH_CMD=/,/"$/p' "$REPO_ROOT/scripts/vllm_sweep_parallel.sh")"

# ── Ray across the nodes (only when the layout spans more than one) ───────────
if [ -n "$NODES" ]; then
    read -r -a NODE_ARR <<< "$NODES"
    HEAD_IP="${NODE_ARR[0]}"
    RAY_ADDRESS="$HEAD_IP:$RAY_PORT"

    # Persistent container per node when the runtime is podman/docker. `ray start` (no
    # --block) daemonises and returns, which would end a one-shot `podman run` and take the
    # daemon with it; a container that sleeps forever outlives it and `exec` reuses it for
    # the engine too. apptainer has no such lifecycle, so it keeps its plain prefix and gets
    # RAY_ADDRESS through the APPTAINERENV_ passthrough instead.
    if [ -n "${CONTAINER_IMAGE:-}" ] && [[ "$CONTAINER_EXEC" == *"$CONTAINER_ENGINE run"* ]]; then
        boot_container() {  # $1 = "" for local, else ssh host
            local run="${1:+ssh $1} $CONTAINER_ENGINE"
            $run rm -f "$CNAME" >/dev/null 2>&1 || true
            $run run -d --name "$CNAME" -e RAY_ADDRESS="$RAY_ADDRESS" \
                $CONTAINER_RUN_ARGS "$CONTAINER_IMAGE" sleep infinity >/dev/null
        }
        boot_container ""
        for h in "${NODE_ARR[@]:1}"; do boot_container "$h"; done
        CONTAINER_EXEC="$CONTAINER_ENGINE exec -w $REPO_ROOT $CNAME"
        REMOTE_EXEC="$CONTAINER_ENGINE exec -w $REPO_ROOT $CNAME"
        CLEANUP_CONTAINERS=1
    else
        export RAY_ADDRESS
        export APPTAINERENV_RAY_ADDRESS="$RAY_ADDRESS"
        REMOTE_EXEC="$CONTAINER_EXEC"
        CLEANUP_CONTAINERS=0
    fi

    stop_ray() {
        echo "[nosched] stopping ray"
        for h in "${NODE_ARR[@]:1}"; do ssh "$h" "$REMOTE_EXEC ray stop --force" || true; done
        $CONTAINER_EXEC ray stop --force || true
        if [ "$CLEANUP_CONTAINERS" = 1 ]; then
            for h in "${NODE_ARR[@]:1}"; do ssh "$h" "$CONTAINER_ENGINE rm -f $CNAME" || true; done
            $CONTAINER_ENGINE rm -f "$CNAME" >/dev/null 2>&1 || true
        fi
    }
    trap stop_ray EXIT

    echo "[nosched] ray head on $HEAD_IP (gcs=$RAY_ADDRESS, ${GPUS_PER_NODE} GPUs/node)"
    $CONTAINER_EXEC ray start --head --disable-usage-stats \
        --node-ip-address="$HEAD_IP" --port="$RAY_PORT" --num-gpus="$GPUS_PER_NODE"
    for h in "${NODE_ARR[@]:1}"; do
        echo "[nosched] ray worker on $h"
        ssh "$h" "$REMOTE_EXEC ray start --address=$RAY_ADDRESS --num-gpus=$GPUS_PER_NODE"
    done

    # GPU count is a stricter readiness signal than node count — nodes register before
    # their GPUs do. Same check the SLURM path makes.
    EXPECTED=$(( ${#NODE_ARR[@]} * GPUS_PER_NODE ))
    for _ in $(seq 1 100); do
        HAVE=$($CONTAINER_EXEC ray status --address "$RAY_ADDRESS" 2>/dev/null \
               | sed -n 's|.*/\([0-9.]*\)\s*GPU.*|\1|p' | head -1)
        [ "${HAVE%%.*}" = "$EXPECTED" ] && break
        sleep 3
    done
    echo "[nosched] ray ready: ${HAVE:-?}/$EXPECTED GPUs"
    SERVE_CMD="$SERVE_CMD --distributed-executor-backend ray"
fi

# ── pair serve x bench the way the orchestrator's --link-vars does ────────────
# model=model, max_num_seqs=_n — one job per pair, each with a freshly started engine.
TMPD="$(mktemp -d "$REPO_ROOT/tmp/nosched.XXXXXX")"
NPAIRS=$(python3 - "$CONFIG_ROOT" "$TMPD" <<'PY'
import json, pathlib, sys
config_root, tmpd = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
i = 0
for sp in sorted(config_root.glob("*/serve-params.json")):
    serves = json.loads(sp.read_text())
    benches = json.loads((sp.parent / "bench-params.json").read_text())
    for s in serves:
        for b in benches:
            if s.get("model") == b.get("model") and s.get("max_num_seqs") == b.get("_n"):
                (tmpd / f"serve_{i}.json").write_text(json.dumps(s))
                (tmpd / f"bench_{i}.json").write_text(json.dumps([b]))
                i += 1
print(i)
PY
)
echo "[nosched] $NPAIRS measurement(s) → $RESULTS_DIR"

# ── run them, one at a time ───────────────────────────────────────────────────
# No --ray: that flag is what makes the worker reach for srun/scontrol. Ray is already
# up (above) and RAY_ADDRESS is inherited, so the engine joins the existing cluster.
# The worker runs inside the container, exactly as _submit_group launches it.
for i in $(seq 0 $((NPAIRS - 1))); do
    echo "[nosched] === measurement $i/$((NPAIRS - 1)) ==="
    ${CONTAINER_EXEC:-} python3 "$REPO_ROOT/src/sweep_parallel.py" --worker \
        --serve-config "$TMPD/serve_$i.json" \
        --bench-configs "$TMPD/bench_$i.json" \
        --serve-cmd "$SERVE_CMD" \
        --bench-cmd "$BENCH_CMD" \
        --port "$PORT" \
        --serve-timeout "${SERVE_TIMEOUT:-1800}" \
        --results-dir "$RESULTS_DIR" \
        --dataset-dir "$DATASET_DIR" \
        2>&1 | tee -a "$RESULTS_DIR/worker_$i.log" \
        || echo "[nosched] measurement $i FAILED — continuing"
done
rm -rf "$TMPD"

# ── plot + score (unchanged; they never touch SLURM) ──────────────────────────
${CONTAINER_EXEC:-} python3 "$REPO_ROOT/src/utils/plot_pareto.py" "$RESULTS_DIR" \
    --fig-by=model --curve-by=custom-output-len || echo "[warn] plotting failed"
${CONTAINER_EXEC:-} python3 "$REPO_ROOT/src/utils/score.py" "$RESULTS_DIR"
echo "[nosched] done → $RESULTS_DIR"
