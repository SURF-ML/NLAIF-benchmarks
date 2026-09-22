#!/usr/bin/env bash
# Can v0.30.0 serve AMD's FP4 checkpoints, and at what activation precision?
#
# On v0.29.0, amd/DeepSeek-V4-Pro-MXFP4 failed to load at all with expert parallelism on:
# vLLM builds a dense DeepseekV4MLP for the shared expert, routes it to the DeepSeek-native
# FP8 path and allocates [3072, 7168], while the checkpoint ships it FP4-packed [3072, 3584].
# Dropping EP let shared-expert fusion engage and it loaded — but still served W4A8.
#
# v0.30.0 source says the relevant machinery is unchanged: from_config still discards the
# Quark schema, the ROCm DeepSeek-V4 priority list is still [AITER_MXFP4_BF16,
# TRITON_UNFUSED], and aiter_mxfp4_mxfp4 is still not a --moe-backend value. So the
# expectation is "loads under some configuration, serves W4A8". Measured here rather than
# assumed, since two of my predictions this session were wrong.
#
# No pkill anywhere: it matches this script's own command line and kills it (exit 144).
set -uo pipefail
cd /home/amd/NLAIF-benchmarks/application-benchmarks/vLLM
REPO_ROOT=$PWD
source configs/amd/mi355x-n32-v030/system.sh
eval "$(sed -n '/^SERVE_CMD=/,/"$/p' scripts/vllm_sweep_parallel.sh)"
OUT=$REPO_ROOT/profiling/mxfp4v030; mkdir -p "$OUT"

probe() {   # $1=label  $2=model  $3=extra serve args  $4=extra podman -e flags
    local LOG=$OUT/$1.log
    echo "########## $1 ##########"
    timeout 1200 $CONTAINER_EXEC ${4:-} $SERVE_CMD \
        --model="$2" --trust-remote-code --tokenizer-mode=deepseek_v4 \
        --tensor-parallel-size=8 --max-num-seqs=32 --port 8016 $3 > "$LOG" 2>&1
    grep -hE "expert_dtype resolved|Mxfp4 MoE backend|Using .*Experts|GPU KV cache size" "$LOG" \
        | sed 's/^([^)]*) //' | sort -u | head -4 | cut -c1-170
    echo "  operand dtypes: $(grep -ohE "'torch\.float[0-9a-z_]*', 'torch\.float[0-9a-z_]*'" "$LOG" | sort -u | head -1)"
    grep -hiE "AssertionError|NotImplementedError|RuntimeError|Simulated weight dequant" "$LOG" \
        | sed 's/^([^)]*) //' | sort -u | head -2 | cut -c1-150
    sleep 20
}

# EP on = the benchmark's own configuration. EP off = what made it load on v0.29.0.
# CONTAINER_EXEC ends with the image name, so an extra -e must be spliced in before it,
# not appended — hence the substitution rather than a 4th positional arg to podman.
FUSE_EXEC="${CONTAINER_EXEC/$CONTAINER_IMAGE/-e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 $CONTAINER_IMAGE}"

probe mxfp4_ep8 amd/DeepSeek-V4-Pro-MXFP4 "--enable-expert-parallel"
CONTAINER_EXEC="$FUSE_EXEC" probe mxfp4_ep1 amd/DeepSeek-V4-Pro-MXFP4 ""
probe nvfp4_ep8 amd/DeepSeek-V4-Pro-NVFP4 "--enable-expert-parallel"
echo "MXFP4 V030 PROBE DONE"
