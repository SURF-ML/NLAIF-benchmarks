# AMD MI355X, N=32 variant. Identical hardware and container to ../mi355x — the only
# difference is that this envelope spends more VRAM on KV cache to make max_num_seqs=32
# reachable.
#
# N is a declared, self-policed submission parameter, so the value here is a measurement,
# not a preference. DeepSeek-V4 uses MLA, whose KV cache is replicated rather than split
# across the TP ranks, so aggregate concurrency tracks the per-GPU cache directly. Measured
# on this node, TP8, --max-model-len=9216:
#
#     GPU_MEM_UTIL   KV cache    tokens     Maximum concurrency
#     0.90           144.89 GiB  263,820    28.63x     <- ../mi355x uses this, with N=16
#     0.95           159.29 GiB  290,043    31.47x        short of 32: high point preempts
#     0.96           162.17 GiB  295,287    32.04x        clears 32 by 0.04x — too thin
#     0.97           165.05 GiB  300,533    32.61x     <- this config
#
# 0.97 is the first value with real margin over 32. Do not go higher without cause: the
# base config's 0.90 was chosen to leave room for the fragmentation spike at the high
# operating point, and that headroom is mostly spent here.
#
# Re-measure after any change to the model, the layout or the vLLM version:
#     grep "Maximum concurrency" results/<run>/*/serve.log
# If it drops below 32.0 this envelope is no longer honest — use ../mi355x (N=16) instead.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"

# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x" && pwd)/system.sh"
