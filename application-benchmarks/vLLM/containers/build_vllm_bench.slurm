#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH --exclusive

set -e

# ── Configuration ────────────────────────────────────────────────────────────
# BUILDER: apptainer | podman (default: apptainer)
BUILDER="${BUILDER:-apptainer}"

# GPU_VENDOR: nvidia | amd (default: nvidia)
GPU_VENDOR="${GPU_VENDOR:-nvidia}"

# VLLM_VERSION: upstream vllm/vllm-openai image tag (default: v0.23.0)
VLLM_VERSION="${VLLM_VERSION:-v0.29.0}"

NVIDIA_BASE="vllm/vllm-openai:${VLLM_VERSION}"
AMD_BASE="vllm/vllm-openai-rocm:${VLLM_VERSION}"
# ─────────────────────────────────────────────────────────────────────────────

if [ -n "${SLURM_SUBMIT_DIR}" ] && ! [ -t 0 ]; then
    SCRIPT_DIR=$SLURM_SUBMIT_DIR
    echo "Batch job detected, using output dir $SCRIPT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "Interactive job detected, using output dir $SCRIPT_DIR"
fi

# Select base image
case "$GPU_VENDOR" in
    nvidia) BASE_IMAGE="$NVIDIA_BASE" ;;
    amd)    BASE_IMAGE="$AMD_BASE" ;;
    *)      echo "ERROR: GPU_VENDOR must be 'nvidia' or 'amd' (got: '$GPU_VENDOR')"; exit 1 ;;
esac

echo "Builder:    $BUILDER"
echo "GPU vendor: $GPU_VENDOR"
echo "Base image: $BASE_IMAGE"
echo "Script dir: $SCRIPT_DIR"

case "$BUILDER" in

    apptainer)
        # Patch the From: line in the def file on the fly
        PATCHED_DEF="/dev/shm/vllm-bench-patched.def"
        sed "s|^From:.*|From: $BASE_IMAGE|" "$SCRIPT_DIR/vllm-bench.def" > "$PATCHED_DEF"

        CONTAINER_NAME="vllm-bench_${GPU_VENDOR}_${VLLM_VERSION}.sif"

        export APPTAINER_CACHEDIR=/dev/shm
        export APPTAINER_TMPDIR=/dev/shm
        export TMPDIR=/dev/shm

        cd /dev/shm
        apptainer build --force "$CONTAINER_NAME" "$PATCHED_DEF"
        cp "$CONTAINER_NAME" "$SCRIPT_DIR/"
        echo "Container built: $SCRIPT_DIR/$CONTAINER_NAME"
        ;;

    podman)
        IMAGE_TAG="vllm-bench:${GPU_VENDOR}-${VLLM_VERSION}"
	OUTPUT_TAR="$SCRIPT_DIR/vllm-bench_${GPU_VENDOR}_${VLLM_VERSION}.tar"

        export TMPDIR=/dev/shm

        cd /dev/shm
        podman build \
            --no-cache \
            --build-arg BASE_IMAGE="docker.io/$BASE_IMAGE" \
            --tag "$IMAGE_TAG" \
            --file "$SCRIPT_DIR/vllm-bench.containerfile"
	
	echo "Exporting image to $OUTPUT_TAR..."
	podman save "$IMAGE_TAG" -o "$OUTPUT_TAR"
        
	echo "Image saved: $OUTPUT_TAR"
        ;;

    *)
        echo "ERROR: BUILDER must be 'apptainer' or 'podman' (got: '$BUILDER')"
        exit 1
        ;;
esac
