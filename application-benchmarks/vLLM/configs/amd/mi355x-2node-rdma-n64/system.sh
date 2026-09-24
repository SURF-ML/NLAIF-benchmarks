# Same as ../mi355x-2node, plus RCCL over the Pollara (AMD AINIC) RoCE NICs instead of TCP.
# Without these the container sees no RDMA device and RCCL falls back to NET/Socket over the
# front-end port: measured ~1 GB/s peak cross-node vs ~4 GB/s with GPU-direct RDMA here.
#   /dev/infiniband        verbs devices for the 8 per-GPU Pollara NICs
#   host libionic          the image's libionic (1.0.54.0-167) rejects the host kernel driver
#                          ("Driver ionic does not support the kernel ABI of 1"); the host's
#                          AINIC-bundle build (1.1.54.0-184) matches it
#   NCCL_IB_HCA            only the per-GPU NICs, not the two bnxt front-end ports
#   NCCL_IB_GID_INDEX=1    RoCE v2 over the routed IPv6 fabric addresses
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-2node" && pwd)/system.sh"
POLLARA_HCA=rocep9s0,rocep25s0,rocep105s0,rocep121s0,rocep137s0,rocep153s0,rocep233s0,rocep249s0
CONTAINER_RUN_ARGS="$CONTAINER_RUN_ARGS --device /dev/infiniband --ulimit memlock=-1:-1 \
-v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
-e NCCL_IB_HCA=$POLLARA_HCA -e NCCL_IB_GID_INDEX=1 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET"
CONTAINER_EXEC="podman run --rm $CONTAINER_RUN_ARGS $CONTAINER_IMAGE"
SBATCH_ARGS="--exclusive"  # else Slurm pins each job to 2 hw threads and vLLM is CPU-starved
