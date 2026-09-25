# AMD — 2 nodes x 8 Instinct MI355X (288 GiB) as one 16-GPU deployment, rootless podman.
# Sourced by run.sh / scripts/vllm_sweep_parallel.sh with $REPO_ROOT already set.
# Every value honours a pre-set environment variable so users can override without editing
# this file (e.g. HF_HOME=/scratch/hf ./run.sh amd/mi355x-2node).
#
# Every layout here spans more than one node's GPUS_PER_NODE, so sweep_parallel.py stands up a
# Ray cluster across the SLURM allocation automatically; vllm serve/bench then run via
# `podman exec` into the Ray head's container.
#
# There is no CPU-only partition on this cluster. Run the orchestrator in the submitting shell:
#     ./run.sh amd/mi355x-2node --local
# A plain sbatch orchestrator job with no explicit --mem gets the partition's default of the
# whole node's memory, which starves the worker jobs and leaves the sweep PENDING forever.

GPUS_PER_NODE=8
# Whole nodes per worker job: vLLM's API servers, engine core and per-GPU workers need the
# node's CPUs, and a plain GPU request is allotted a single core.
SBATCH_ARGS="${SBATCH_ARGS:---exclusive}"
ORCH_SBATCH_ARGS="${ORCH_SBATCH_ARGS:-}"
[ -n "${GPU_REQUEST_TMPL:-}" ] || GPU_REQUEST_TMPL='--gpus-per-node={n}'

# Fraction of VRAM vLLM may use. 0.90 leaves headroom for the RCCL buffers and the
# fragmentation spike at the high operating point. N (max_num_seqs=1024, see
# scripts/gen_configs.py) is far above the startup "Maximum concurrency" line (78.4x for
# TP8/PP2): for DeepSeek-V4 that line is a worst-case reservation summed over its many KV
# groups, and measured KV usage at runtime is much lower.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# Model cache. Weights live on the local SSD array at /mnt/data, NOT on /dev/shm: /dev/shm loads
# faster, but it is not durable — an 806 GiB checkpoint was silently cleared from it overnight,
# mid campaign, without a reboot. There is no shared filesystem, so each node needs its own copy.
export HF_HOME="${HF_HOME:-/mnt/data/hf_hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# torch.compile / inductor artifacts. Kept off /dev/shm so the cache survives a reboot.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/vllm_cache}"

# The API-server frontend's patience while EngineCore loads (vLLM default 600 s). A layout that
# spans nodes for the first time JIT-compiles many kernel shapes from scratch: a real run was
# killed by vLLM at 1800 s ("Timed out waiting for engine core processes to start") while still
# actively compiling.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-5400}"
# Cap on a single engine step (vLLM default 300 s). The first 8192-token prefill also lazily
# creates the cross-node PP send/recv communicators, and measurably exceeded 300 s ("RPC call to
# execute_model timed out"), killing the engine on its first request.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
# The harness's own wait for /health (sweep_parallel.py --serve-timeout): the checkpoint load
# plus the cross-node Ray bootstrap take longer than a single node.
export SERVE_TIMEOUT="${SERVE_TIMEOUT:-5400}"

# DeepSeek-V4's sparse attention indexer has no generic ROCm kernel — vLLM refuses to build the
# executor without it ("Sparse attention indexer ROCm path is only supported on AITER. Please
# enable aiter with VLLM_ROCM_USE_AITER=1"). AITER is AMD's fused-kernel library, shipped in the
# ROCm vLLM image, so this is a platform requirement rather than a tuning choice.
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"

# Interface for RCCL's bootstrap / socket traffic. enp193s0f1np1 is the 10.2.84.0/23 fabric
# interface here — change it if your second node is on a different subnet. Bulk cross-node
# traffic goes over RDMA (below), not this interface.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp193s0f1np1}"

# ── RDMA: RCCL over the per-GPU Pollara (AMD AINIC) RoCE NICs ────────────────────────────────
# Without these the container sees no RDMA device and RCCL falls back to NET/Socket over the
# front-end port: measured ~1 GB/s peak cross-node vs ~4 GB/s with GPU-direct RDMA.
#   /dev/infiniband        verbs devices for the 8 per-GPU Pollara NICs
#   host libionic          the image's libionic (1.0.54.0-167) rejects the host kernel driver
#                          ("Driver ionic does not support the kernel ABI of 1"); the host's
#                          AINIC-bundle build (1.1.54.0-184) matches it
#   NCCL_IB_HCA            only the per-GPU NICs, not the two bnxt front-end ports
#   NCCL_IB_GID_INDEX=1    RoCE v2 over the routed IPv6 fabric addresses
POLLARA_HCA="${POLLARA_HCA:-rocep9s0,rocep25s0,rocep105s0,rocep121s0,rocep137s0,rocep153s0,rocep233s0,rocep249s0}"

# ── container ────────────────────────────────────────────────────────────────────────────────
#   --device kfd/dri   ROCm devices (both are world-rw here, so no --group-add is needed)
#   --ipc=host         vLLM's workers talk over POSIX shm; the host /dev/shm — where the
#                      benchmark datasets are copied — becomes the container's /dev/shm
#   --network=host     the bench client reaches the server on localhost, and Ray's GCS +
#                      worker ports are reachable from the other node
#   /shared/data       the NFS mount the repo lives on (the nodes have no shared disk of their
#                      own); the container needs it for `-w $REPO_ROOT`
# Deliberately NO --userns=keep-id. Rootless podman's default mapping already puts container
# root at your host UID, so results/ and logs/ still come back owned by you — but it ALSO makes
# the image's own tree writable, which vLLM on ROCm needs: AITER JIT-compiles the kernels it has
# no prebuilt entry for straight into its dist-packages dir, and Triton caches under $HOME
# (= /root in the image). Under keep-id both are root-owned and read-only, and the engine dies
# with "[Errno 13] Permission denied: .../aiter/jit/flydsl_cache/launch_mixed_moe_gemm1_...".
#
# The AITER kernel cache is bind-mounted out to the host so JIT results survive --rm; without it
# every measurement restarts the engine into a fresh container and recompiles from scratch. The
# cache is per vLLM version and per node. Seed it once on each node from the image (2.2 GiB,
# ~3.5k prebuilt kernels), before the first run — podman will not bind-mount a missing path:
#   podman create --name seed $CONTAINER_IMAGE true
#   mkdir -p $AITER_CACHE
#   podman cp seed:/usr/local/lib/python3.12/dist-packages/aiter/jit/flydsl_cache/. \
#             $AITER_CACHE/ && podman rm seed
# vLLM v0.30.0, the same engine as the single-node config (../mi355x). The AITER and Triton
# caches are per vLLM version, so they are the -v030 ones, as in ../mi355x.
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
--device /dev/infiniband --ulimit memlock=-1:-1 \
-v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
-e NCCL_IB_HCA=$POLLARA_HCA -e NCCL_IB_GID_INDEX=1 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
-e PYTHONNOUSERSITE=1 ${CONTAINER_EXTRA_ENV:-}}"

CONTAINER_EXEC="${CONTAINER_EXEC:-podman run --rm $CONTAINER_RUN_ARGS $CONTAINER_IMAGE}"
