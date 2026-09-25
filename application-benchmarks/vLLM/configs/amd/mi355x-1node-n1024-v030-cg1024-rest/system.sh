# The 1024→8192 low/mid cells of ../mi355x-1node-n1024-v030-cg1024, run into the same results dir as its high cell.
# bench-params.json is that cluster's generated file filtered to these two entries.
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-1node-n1024-v030-cg1024" && pwd)/system.sh"
