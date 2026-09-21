# AMD — 8x Instinct MI355X (288 GiB) per node, no scheduler, podman instead of apptainer.
# Sourced by run.sh / scripts/vllm_sweep_parallel.sh / run_nosched.sh with $REPO_ROOT set.
# Every value honours a pre-set environment variable so users can override without editing
# this file (e.g. HF_HOME=/scratch/hf ./run_nosched.sh).

GPUS_PER_NODE=8
# 8x288 GiB = 2.3 TiB per node, so DeepSeek-V4-Pro's ~806 GiB checkpoint fits in ONE node at
# TP8 — no Ray, no second node needed for the scored config. The SLURM knobs below are kept
# only so this cluster dir stays usable from run.sh on a scheduled machine; run_nosched.sh
# ignores them.
SBATCH_ARGS="${SBATCH_ARGS:-}"
ORCH_SBATCH_ARGS="${ORCH_SBATCH_ARGS:-}"
[ -n "${GPU_REQUEST_TMPL:-}" ] || GPU_REQUEST_TMPL='--gpus-per-node={n}'

# Fraction of VRAM vLLM may use. MI355X has 288 GiB/GPU; 0.90 leaves headroom for the RCCL
# buffers and the fragmentation spike at the high operating point.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# Model cache. This box has no shared filesystem, so the weights live on the local RAM disk
# (/dev/shm, 1.5 TiB). For a two-node run every node needs its OWN copy at this same path.
export HF_HOME="${HF_HOME:-/dev/shm/hf_hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# torch.compile / inductor artifacts. Deliberately NOT on /dev/shm: the checkpoint already
# occupies ~850 GiB of it, and this cache must survive a reboot to be worth having.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/vllm_cache}"

# The API-server frontend's patience while EngineCore loads. vLLM's 600s default is too short
# for an 806 GiB checkpoint + graph capture; 1800s gives ~3x margin.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"

# DeepSeek-V4's sparse attention indexer has no generic ROCm kernel — vLLM refuses to build the
# executor without it ("Sparse attention indexer ROCm path is only supported on AITER. Please
# enable aiter with VLLM_ROCM_USE_AITER=1"). AITER is AMD's fused-kernel library, shipped in the
# ROCm vLLM image, so this is a platform requirement rather than a tuning choice.
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"

# Pin RCCL to the inter-node NIC. Only read by a multi-node (NODES=...) run; a single-node TP8
# serve stays on XGMI and ignores it. enp193s0f1np1 is the 10.2.84.0/23 fabric interface here —
# change it if your second node is on a different subnet.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp193s0f1np1}"

# ── container ────────────────────────────────────────────────────────────────────────────────
# podman, not apptainer. The flags are split out because run_nosched.sh needs them twice: once
# as a one-shot `podman run` prefix (single node) and once to boot a persistent container it
# can `podman exec` into (multi-node, where `ray start` daemonises and would otherwise die with
# the one-shot container that launched it).
#   --device kfd/dri   ROCm devices (both are world-rw here, so no --group-add is needed)
#   --ipc=host         vLLM's workers talk over POSIX shm; also makes the host /dev/shm — where
#                      the weights and the benchmark datasets live — the container's /dev/shm
#   --network=host     the bench client reaches the server on localhost:8000, and Ray's GCS +
#                      worker ports are reachable from the other node
# Deliberately NO --userns=keep-id. Rootless podman's default mapping already puts container
# root at your host UID, so results/ and logs/ still come back owned by you — but it ALSO makes
# the image's own tree writable, which vLLM on ROCm needs: AITER JIT-compiles the kernels it has
# no prebuilt entry for straight into its dist-packages dir, and Triton caches under $HOME
# (= /root in the image). Under keep-id both are root-owned and read-only, and the engine dies
# with "[Errno 13] Permission denied: .../aiter/jit/flydsl_cache/launch_mixed_moe_gemm1_...".
#
# The AITER kernel cache is then bind-mounted out to the host (seeded once from the image with
# `podman cp`, 2.2 GiB / ~3.5k prebuilt kernels) so JIT results survive --rm. Without it every
# measurement restarts the engine into a fresh container and recompiles from scratch.
#   podman create --name seed $CONTAINER_IMAGE true
#   mkdir -p ~/.cache/aiter/flydsl_cache
#   podman cp seed:/usr/local/lib/python3.12/dist-packages/aiter/jit/flydsl_cache/. \
#             ~/.cache/aiter/flydsl_cache/ && podman rm seed
AITER_CACHE="${AITER_CACHE:-$HOME/.cache/aiter/flydsl_cache}"
TRITON_CACHE="${TRITON_CACHE:-$HOME/.cache/triton}"
AITER_JIT_DIR=/usr/local/lib/python3.12/dist-packages/aiter/jit

# CONTAINER_EXTRA_ENV lets a derived cluster add `-e VAR=VAL` flags without restating the
# whole arg string (see ../mi355x-n32-w4a4, which needs a checkpoint-specific toggle).
CONTAINER_IMAGE="${CONTAINER_IMAGE:-localhost/vllm-bench:amd-v0.29.0}"
CONTAINER_RUN_ARGS="${CONTAINER_RUN_ARGS:---device /dev/kfd --device /dev/dri \
--ipc=host --network=host \
--security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
-v $HOME:$HOME -w $REPO_ROOT \
-v $AITER_CACHE:$AITER_JIT_DIR/flydsl_cache \
-e FLYDSL_RUNTIME_CACHE_DIR=$AITER_JIT_DIR/flydsl_cache \
-e TRITON_CACHE_DIR=$TRITON_CACHE \
-e HF_HOME=$HF_HOME -e HF_HUB_OFFLINE=$HF_HUB_OFFLINE \
-e VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT \
-e VLLM_ENGINE_READY_TIMEOUT_S=$VLLM_ENGINE_READY_TIMEOUT_S \
-e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME -e VLLM_ROCM_USE_AITER=$VLLM_ROCM_USE_AITER \
-e PYTHONNOUSERSITE=1 ${CONTAINER_EXTRA_ENV:-}}"

CONTAINER_EXEC="${CONTAINER_EXEC:-podman run --rm $CONTAINER_RUN_ARGS $CONTAINER_IMAGE}"
