# Single node, N=512 (see scripts/gen_configs.py) — same engine, GPU_MEM_UTIL=0.97, mounts and
# timeouts as ../mi355x-1node-n32, not pinned to a node.
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-1node-n32" && pwd)/system.sh"
# Whole node per worker job (see ../mi355x-2node-rdma-sweep/system.sh).
SBATCH_ARGS="--exclusive"
