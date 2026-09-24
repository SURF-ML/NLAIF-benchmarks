# Two nodes, TP8/PP2 at N=1024 (see scripts/gen_configs.py) — same RDMA fabric settings,
# timeouts and whole-node worker jobs as ../mi355x-2node-rdma-sweep.
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-2node-rdma-sweep" && pwd)/system.sh"
