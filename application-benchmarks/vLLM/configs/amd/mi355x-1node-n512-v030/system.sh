# ../mi355x-1node-n512 on vLLM v0.30.0 (image and kernel caches as ../mi355x-n32-v030).
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-localhost/vllm-bench:amd-v0.30.0}"
export AITER_CACHE="${AITER_CACHE:-$HOME/.cache/aiter-v030/flydsl_cache}"
export TRITON_CACHE="${TRITON_CACHE:-$HOME/.cache/triton-v030}"
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-1node-n512" && pwd)/system.sh"
