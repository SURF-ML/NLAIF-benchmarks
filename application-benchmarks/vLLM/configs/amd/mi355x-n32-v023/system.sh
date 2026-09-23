# vLLM v0.23.0 comparison against the v0.29.0 baseline. Same node, same TP8 layout, same
# N=32 @ GPU_MEM_UTIL=0.97 envelope as ../mi355x-n32 — the ONLY difference is the engine
# version, so any delta is attributable to vLLM itself.
#
# NOT SUBMITTABLE, on precision grounds: v0.23.0 is a stable release (so ruling 10 allows it),
# but it runs the MoE as W4A4 — its older code infers activation precision from weight
# precision — and §8.5 prohibits additional activation quantisation. This config exists to
# measure how much of the result is the engine rather than the hardware.
#
# Build the image first:
#   cd containers && BUILDER=podman GPU_VENDOR=amd VLLM_VERSION=v0.23.0 \
#       bash build_vllm_bench.slurm
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-localhost/vllm-bench:amd-v0.23.0}"

# Version-specific JIT caches. AITER and Triton artifacts are keyed by their own version, not
# just by GPU arch, so pointing 0.23.0 at the 0.29.0 caches risks loading kernels built
# against a different aiter. Separate dirs keep the two engines from contaminating each other.
export AITER_CACHE="${AITER_CACHE:-$HOME/.cache/aiter-v023/flydsl_cache}"
export TRITON_CACHE="${TRITON_CACHE:-$HOME/.cache/triton-v023}"

# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-n32" && pwd)/system.sh"
