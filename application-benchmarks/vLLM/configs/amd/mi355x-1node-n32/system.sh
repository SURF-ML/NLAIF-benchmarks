# Single-node MI355X layout sweep at N=32 (max_num_seqs=32), shared by the -low/-high configs,
# which differ only in which node they pin to and which operating point they run. Same engine
# everywhere so concurrency 1 vs 32 is the only variable between the two nodes.
#
# GPU_MEM_UTIL=0.97 comes from ../mi355x-n32 (TP8 needs it to fit 32 x 9216 tokens: 32.61x at
# 0.97 vs 28.63x at 0.90). Timeouts are set BEFORE sourcing it: ../mi355x only applies its own
# `${VAR:-...}` defaults, and bakes VLLM_ENGINE_READY_TIMEOUT_S into CONTAINER_RUN_ARGS.
# TP4/PP2 and TP2/PP4 start with a cold AITER kernel cache, which measurably needs > 1800 s.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-5400}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-n32" && pwd)/system.sh"
export SERVE_TIMEOUT="${SERVE_TIMEOUT:-5400}"
# Repo lives on /shared/data (see ../mi355x-2node/system.sh); the container needs it for -w.
CONTAINER_RUN_ARGS="$CONTAINER_RUN_ARGS -v /shared/data:/shared/data \
-e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
CONTAINER_EXEC="podman run --rm $CONTAINER_RUN_ARGS $CONTAINER_IMAGE"
