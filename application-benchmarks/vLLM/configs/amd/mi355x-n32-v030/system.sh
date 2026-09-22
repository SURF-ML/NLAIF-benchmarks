# vLLM v0.30.0 on the same node, layout and N=32 @ 0.97 envelope as ../mi355x-n32.
# Only the engine differs, so any delta is attributable to the version.
#
# NOT SUBMITTABLE as things stand: README §9 ruling 10 pins the version. This exists to test
# whether the pin is what caps our KV pool. v0.29.0 gives 300,533 KV tokens (575.9 KiB/token,
# 16.8x the ~34.3 KiB/token the MLA tensors imply); SemiAnalysis report 4,386,107 (~39.5
# KiB/token) on a nightly for the same model and hardware. The v0.30.0 release notes name two
# candidates for this model:
#   * "--kv-cache-dtype auto resolving to fp8_ds_mla with FlashMLA" (#45091) — a compact,
#     MLA-aware FP8 KV layout. §8.1 fixes --kv-cache-dtype to the literal "fp8", not "auto",
#     so the spec may be selecting a less efficient implementation of the same precision.
#   * "PCP+DCP on sparse-MLA models" (#56157) — exactly the NotImplementedError that made
#     --decode-context-parallel-size unusable on v0.29.0.
#
# Build the image with the repo's own script, which clears the ENTRYPOINT the upstream image
# sets and adds vllm[bench]+ray, so it drops into the harness unchanged:
#   cd containers && BUILDER=podman GPU_VENDOR=amd VLLM_VERSION=v0.30.0 bash build_vllm_bench.slurm
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-localhost/vllm-bench:amd-v0.30.0}"

# Version-specific JIT caches so v0.30.0 cannot load kernels built against v0.29.0's aiter.
export AITER_CACHE="${AITER_CACHE:-$HOME/.cache/aiter-v030/flydsl_cache}"
export TRITON_CACHE="${TRITON_CACHE:-$HOME/.cache/triton-v030}"

# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-n32" && pwd)/system.sh"
