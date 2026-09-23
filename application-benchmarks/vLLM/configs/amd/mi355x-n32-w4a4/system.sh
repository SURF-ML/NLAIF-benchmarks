# W4A4 probe: amd/DeepSeek-V4-Pro-MXFP4 on the v0.29.0 engine.
#
# Same node, same TP8 layout, same N=32 @ GPU_MEM_UTIL=0.97 envelope and same engine as
# ../mi355x-n32 — the ONLY difference is the checkpoint, so the delta isolates activation
# precision. The base deepseek-ai/DeepSeek-V4-Pro declares FP8 dynamic activations, so
# v0.29.0 serves it W4A8 (kernels: flydsl_moe1_afp8_wfp4_bf16, ~2490 TFLOPS ceiling, about
# half of FP8 peak). This checkpoint declares MXFP4 activations
# (global_quant_config.input_tensors: fp4 / per_group 32 / e8m0), which routes the routed
# experts through AITER's W4A4 CK path and onto the FP4 matrix cores.
#
# NOT SUBMITTABLE under the rules as written: README §8.5 bars "additional weight or
# activation quantisation", and §8.1/§3 pin the checkpoint. Kept as a separate cluster so its
# results can never be confused with a scored run. It exists to answer whether FP4 compute
# actually wins on MI355X, which the v0.23.0 comparison could not (that run was confounded by
# six versions of kernel maturity and came out 5.5x SLOWER despite using W4A4).
#
# Expected ceiling ~1.4x, not 2x: computed from this checkpoint's own tensor shapes, 51.3% of
# per-token GEMM FLOPs are routed experts (-> FP4 here) and 38.8% are attention projections
# that stay FP8 in every published build. This checkpoint also pushes the shared expert (8.6%)
# to FP4, which nvidia/DeepSeek-V4-Pro-NVFP4 leaves at FP8 — so it is NOT coverage-matched
# with NVIDIA's build and the two are not directly comparable without aligning exclude lists.
MODEL_OVERRIDE="${MODEL_OVERRIDE:-amd/DeepSeek-V4-Pro-MXFP4}"

# Version-specific JIT caches: the W4A4 CK kernels are a different set from the W4A8 FlyDSL
# ones the baseline compiled, so keep them apart rather than mixing two kernel families.
export AITER_CACHE="${AITER_CACHE:-$HOME/.cache/aiter-w4a4/flydsl_cache}"
export TRITON_CACHE="${TRITON_CACHE:-$HOME/.cache/triton-w4a4}"

# This checkpoint quantises the SHARED expert to MXFP4 too, and vLLM only has a loader for
# that when shared-expert fusion is on: with fusion off, DeepseekV4MoE builds a dense
# DeepseekV4MLP for the shared expert, routes it to the DeepSeek-native FP8 path, allocates a
# [3072, 7168] parameter, and dies on the FP4-packed [3072, 3584] tensor —
#   parameter.py:175 load_merged_column_weight: assert param_data.shape == loaded_weight.shape
# With fusion on, `self.shared_experts = None` and the weights are consumed by the MXFP4
# expert loader instead, which is the layout amd/DeepSeek-V4-Pro-MXFP4 actually ships (the
# mapper comment in amd/model.py says shared-expert w{1,2,3}.weight_scale are "left untouched
# ... consumed as-is by the MXFP4 expert loader").
#
# Consequence for comparability: the shared expert stays FP4 here, so this build is NOT
# module-matched with nvidia/DeepSeek-V4-Pro-NVFP4, which leaves it at FP8. Worth ~8.6% of
# per-token GEMM FLOPs — fine for a W4A4-vs-W4A8 delta against our own baseline, not fine for
# a cross-vendor number without aligning the exclude lists first.
export CONTAINER_EXTRA_ENV="-e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1"

# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-n32" && pwd)/system.sh"
