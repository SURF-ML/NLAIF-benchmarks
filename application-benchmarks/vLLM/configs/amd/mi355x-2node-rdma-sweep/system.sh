# 2-node TP/PP × EP grid (see scripts/gen_configs.py) over the Pollara RDMA fabric — same
# container, mounts, timeouts and RCCL settings as ../mi355x-2node-rdma.
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-2node-rdma" && pwd)/system.sh"
# Whole nodes per worker job: vLLM's API servers, engine core and per-GPU workers need the
# node's CPUs, and a plain GPU request is allotted a single core.
SBATCH_ARGS="--exclusive"
