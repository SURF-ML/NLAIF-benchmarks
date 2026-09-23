# v0.23.0 with expert parallelism OFF (TP8 / EP1), on the base checkpoint.
#
# Single-variable change from ../mi355x-n32-v023: same image, same TP8 layout, same N=32 @
# GPU_MEM_UTIL=0.97, same checkpoint — only --enable-expert-parallel is dropped. Shared-expert
# fusion is left at its default (VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS defaults False in
# 0.23.0), so EP really is the only thing that moves.
#
# Why: ../mi355x-n32-v023 measured 5.5x SLOWER than v0.29.0 and I attributed that to kernel
# immaturity ("using 2stage default" for every MoE shape, no tuned FlyDSL configs). But EP8 may
# simply be a poor layout for 384 experts over 8 ranks on that stack. If EP1 closes most of the
# gap, the version comparison was measuring layout, not maturity, and the "engine is worth 5.5x"
# claim needs withdrawing.
#
# Also the closest analogue to the SemiAnalysis InferenceX config (dsv4-fp4-mi355x-vllm: TP8,
# EP1, base checkpoint, vllm-openai-rocm v0.22.0), which is worth a sanity comparison.
#
# NOT SUBMITTABLE: v0.23.0 runs the MoE as W4A4, which §8.5 prohibits. (The version itself is
# allowed — any stable release available at acceptance is.) EP is tunable (§8.4); the
# submittable EP1 variants are ../mi355x-n32-ep1 (v0.29.0) and ../mi355x-n32-v030-ep1 (v0.30.0).
#
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-n32-v023" && pwd)/system.sh"
