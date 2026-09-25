# AMD — 8x Instinct MI355X (288 GiB) per node, one node, rootless podman instead of apptainer.
# Sourced by run.sh / scripts/vllm_sweep_parallel.sh with $REPO_ROOT already set.
# Every value honours a pre-set environment variable so users can override without editing
# this file (e.g. HF_HOME=/scratch/hf ./run.sh amd/mi355x).
#
# There is no CPU-only partition on this cluster. Run the orchestrator in the submitting shell:
#     ./run.sh amd/mi355x --local
# A plain sbatch orchestrator job with no explicit --mem gets the partition's default of the
# whole node's memory, which starves the worker job of that same node and leaves the sweep
# PENDING (Resources) forever.

GPUS_PER_NODE=8
# 8x288 GiB = 2.3 TiB per node, so DeepSeek-V4-Pro's ~806 GiB checkpoint fits in ONE node at
# TP8. Whole node per worker job: vLLM's API servers, engine core and per-GPU workers need the
# node's CPUs, and a plain GPU request is allotted a single core.
SBATCH_ARGS="${SBATCH_ARGS:---exclusive}"
ORCH_SBATCH_ARGS="${ORCH_SBATCH_ARGS:-}"
[ -n "${GPU_REQUEST_TMPL:-}" ] || GPU_REQUEST_TMPL='--gpus-per-node={n}'

# Fraction of VRAM vLLM may use. Measured on this node, TP8, --max-model-len=9216:
#
#     GPU_MEM_UTIL   KV cache    tokens     Maximum concurrency
#     0.90           144.89 GiB  263,820    28.63x
#     0.95           159.29 GiB  290,043    31.47x
#     0.96           162.17 GiB  295,287    32.04x
#     0.97           165.05 GiB  300,533    32.61x     <- this config
#
# N (max_num_seqs=1024, see scripts/gen_configs.py) is far above that startup line: for
# DeepSeek-V4 it is a worst-case reservation summed over its many KV groups, and measured KV
# usage at runtime is much lower. 0.97 still buys the most KV headroom; 0.90 would leave more
# room for the fragmentation spike at the high operating point.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"

# Model cache. Weights live on the local SSD array at /mnt/data, NOT on /dev/shm: /dev/shm loads
# faster, but it is not durable — an 806 GiB checkpoint was silently cleared from it overnight,
# mid campaign, without a reboot. There is no shared filesystem, so each node needs its own copy.
export HF_HOME="${HF_HOME:-/mnt/data/hf_hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# torch.compile / inductor artifacts. Kept off /dev/shm so the cache survives a reboot.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/vllm_cache}"

# The API-server frontend's patience while EngineCore loads (vLLM default 600 s). A cold AITER
# kernel cache measurably needs more than 1800 s before the engine is ready.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-5400}"
# Cap on a single engine step (vLLM default 300 s); the first long prefill can exceed it.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
# The harness's own wait for /health (sweep_parallel.py --serve-timeout).
export SERVE_TIMEOUT="${SERVE_TIMEOUT:-5400}"

# DeepSeek-V4's sparse attention indexer has no generic ROCm kernel — vLLM refuses to build the
# executor without it ("Sparse attention indexer ROCm path is only supported on AITER. Please
# enable aiter with VLLM_ROCM_USE_AITER=1"). AITER is AMD's fused-kernel library, shipped in the
# ROCm vLLM image, so this is a platform requirement rather than a tuning choice.
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"

# RCCL's inter-node interface. A single-node TP8 serve stays on XGMI and ignores it.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp193s0f1np1}"

# ── container ────────────────────────────────────────────────────────────────────────────────
#   --device kfd/dri   ROCm devices (both are world-rw here, so no --group-add is needed)
#   --ipc=host         vLLM's workers talk over POSIX shm; the host /dev/shm — where the
#                      benchmark datasets are copied — becomes the container's /dev/shm
#   --network=host     the bench client reaches the server on localhost
#   /shared/data       the NFS mount the repo lives on; the container needs it for `-w $REPO_ROOT`
# Deliberately NO --userns=keep-id. Rootless podman's default mapping already puts container
# root at your host UID, so results/ and logs/ still come back owned by you — but it ALSO makes
# the image's own tree writable, which vLLM on ROCm needs: AITER JIT-compiles the kernels it has
# no prebuilt entry for straight into its dist-packages dir, and Triton caches under $HOME
# (= /root in the image). Under keep-id both are root-owned and read-only, and the engine dies
# with "[Errno 13] Permission denied: .../aiter/jit/flydsl_cache/launch_mixed_moe_gemm1_...".
#
# The AITER kernel cache is bind-mounted out to the host so JIT results survive --rm; without it
# every measurement restarts the engine into a fresh container and recompiles from scratch. The
# cache is per vLLM version. Seed it once from the image (2.2 GiB, ~3.5k prebuilt kernels), and
# create it before the first run — podman will not bind-mount a path that does not exist:
#   podman create --name seed $CONTAINER_IMAGE true
#   mkdir -p $AITER_CACHE
#   podman cp seed:/usr/local/lib/python3.12/dist-packages/aiter/jit/flydsl_cache/. \
#             $AITER_CACHE/ && podman rm seed
CONTAINER_IMAGE="${CONTAINER_IMAGE:-localhost/vllm-bench:amd-v0.30.0}"
AITER_CACHE="${AITER_CACHE:-$HOME/.cache/aiter-v030/flydsl_cache}"
TRITON_CACHE="${TRITON_CACHE:-$HOME/.cache/triton-v030}"
AITER_JIT_DIR=/usr/local/lib/python3.12/dist-packages/aiter/jit

CONTAINER_RUN_ARGS="${CONTAINER_RUN_ARGS:---device /dev/kfd --device /dev/dri \
--ipc=host --network=host \
--security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
-v $HOME:$HOME -v /mnt/data:/mnt/data -v /shared/data:/shared/data -w $REPO_ROOT \
-v $AITER_CACHE:$AITER_JIT_DIR/flydsl_cache \
-e FLYDSL_RUNTIME_CACHE_DIR=$AITER_JIT_DIR/flydsl_cache \
-e TRITON_CACHE_DIR=$TRITON_CACHE \
-e HF_HOME=$HF_HOME -e HF_HUB_OFFLINE=$HF_HUB_OFFLINE \
-e VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT \
-e VLLM_ENGINE_READY_TIMEOUT_S=$VLLM_ENGINE_READY_TIMEOUT_S \
-e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS \
-e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME -e VLLM_ROCM_USE_AITER=$VLLM_ROCM_USE_AITER \
-e PYTHONNOUSERSITE=1 ${CONTAINER_EXTRA_ENV:-}}"

CONTAINER_EXEC="${CONTAINER_EXEC:-podman run --rm $CONTAINER_RUN_ARGS $CONTAINER_IMAGE}"
