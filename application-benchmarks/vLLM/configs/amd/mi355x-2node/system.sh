# AMD MI355X, 2-node TP/PP layout sweep (16 GPUs total via Ray) — same node type and container
# as ../mi355x (single-node TP8); only the parallelism layout (and the timeouts below) change.
#
# Every layout here spans more than one node's GPUS_PER_NODE, so sweep_parallel.py routes it
# through Ray automatically (nodes_for() in src/sweep_parallel.py) — no NODES=... env var, no
# manual ssh/podman bootstrap like run_nosched.sh needs; that's the point of SLURM working now.
#
# There is still no CPU-only partition. Use `./run.sh amd/mi355x-2node --local` (orchestrator
# runs right here in the submitting shell, e.g. the control node) rather than the default
# sbatch-the-orchestrator mode: a plain sbatch job with no explicit --mem gets the partition's
# default of the WHOLE node's memory, which starves every worker job of that same node and
# wedges the sweep in PENDING (Resources) forever — measured directly on this cluster.
#
# Both timeout overrides below must be set BEFORE sourcing ../mi355x/system.sh: that file only
# applies its own default via `${VAR:-...}`, so setting VAR first (not after) is what makes the
# override actually take — an override placed after sourcing is silently a no-op, since the
# variable is already set to the base's default by then (measured directly on this cluster).

# ../mi355x/system.sh's VLLM_ENGINE_READY_TIMEOUT_S=1800 is vLLM's OWN internal gate (the API
# server's patience for its EngineCore workers to finish loading). It was calibrated for
# single-node TP8 with an already-warmed AITER kernel cache; a layout that spans nodes for the
# first time (e.g. TP8/PP2) has to JIT-compile many kernel shapes from scratch and measurably
# needs longer — a real run hit vLLM's own "Timed out waiting for engine core processes to
# start... Waited 1800s" and was killed by vLLM itself at the 30-minute mark while still
# actively compiling, not stuck.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-5400}"

# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x" && pwd)/system.sh"

# DeepSeek-V4's ~806 GiB checkpoint plus the cross-node Ray bootstrap take longer to reach
# /health than the single-node default (1800s) covers; sweep_parallel.py's own --serve-timeout
# help text recommends ~5400s for exactly this case (DeepSeek-V4 across nodes). Unlike
# VLLM_ENGINE_READY_TIMEOUT_S above, mi355x/system.sh never sets this one, so ordering doesn't
# matter here — but it's kept alongside it for the same reason.
export SERVE_TIMEOUT="${SERVE_TIMEOUT:-5400}"

# ../mi355x/system.sh's CONTAINER_RUN_ARGS bind-mounts $HOME and /mnt/data, matching where that
# cluster's REPO_ROOT lives (under $HOME). This cluster's code was moved to /shared/data instead
# (an NFS mount reachable from both nodes, needed since they have no shared disk of their own —
# see the top-level repo's run_nosched.sh history for why), so the container also needs that path
# bind-mounted, or CONTAINER_RUN_ARGS's `-w $REPO_ROOT` fails with "workdir ... does not exist".
# This is still used to start the Ray head/worker containers (src/sweep_parallel.py's
# start_ray_cluster) even though multi-node `vllm serve`/bench now run via `podman exec` into the
# already-running head container instead (see that file) — that exec inherits every mount/env
# below for free from when the container was created, RAY_ADDRESS excepted (added explicitly at
# exec time, since it's only known once the head is actually up).
#
# VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS (vLLM default 300) caps a single engine step. The first
# 8192-token prefill on TP4/PP4 also lazily creates the cross-node PP send/recv NCCL
# communicators, and measurably exceeded 300 s ("RPC call to execute_model timed out"),
# killing the engine on its first request; the same layout at 1k input was fine.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
CONTAINER_RUN_ARGS="$CONTAINER_RUN_ARGS -v /shared/data:/shared/data \
-e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
CONTAINER_EXEC="podman run --rm $CONTAINER_RUN_ARGS $CONTAINER_IMAGE"
