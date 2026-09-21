#!/usr/bin/env python3
"""Does an FP4-activation MoE GEMM actually reach the FP4 matrix cores?

Single process, one GPU, AITER's fused MoE at DeepSeek-V4's real shapes. Run twice —
activations quantised to FP4 (A4W4) and to FP8 (A8W4) — and compare the hardware
counters. This is the same kernel family vLLM dispatches; isolating it avoids profiling
an 8-process TP engine, which destabilises under PMC collection.

Recipe follows aiter/ops/flydsl/test_flydsl_moe_a4w4.py.

Usage: moe_mfma_micro.py {a4w4|a8w4}
"""
import sys, torch, aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_moe

MODE = sys.argv[1] if len(sys.argv) > 1 else "a4w4"
# DeepSeek-V4-Pro MoE, one TP8 rank: 384 experts / 8 = 48 local, hidden 7168, inter 3072.
TOKENS, MODEL_DIM, INTER_DIM, EXPERTS, TOPK = 4096, 7168, 3072, 48, 6
ITERS = 20

dev = "cuda"
torch.manual_seed(0)
q = aiter.get_torch_quant(QuantType.per_1x32)          # weights: torch path, fp4 only
# Activations: the HIP quantiser is the one that also does MXFP8, which is what the
# W4A8 arm needs (get_torch_quant asserts fp4x2).
qa = aiter.get_hip_quant(QuantType.per_1x32)
a_dtype = dtypes.fp4x2 if MODE == "a4w4" else dtypes.fp8

inp = torch.randn(TOKENS, MODEL_DIM, device=dev, dtype=torch.bfloat16) / 10
w1 = torch.randn(EXPERTS, INTER_DIM * 2, MODEL_DIM, device=dev, dtype=torch.bfloat16) / 20
w2 = torch.randn(EXPERTS, MODEL_DIM, INTER_DIM, device=dev, dtype=torch.bfloat16) / 20

w1_qt, w1_scale = q(w1, quant_dtype=dtypes.fp4x2)
w2_qt, w2_scale = q(w2, quant_dtype=dtypes.fp4x2)
w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)
a_qt, a_scale = qa(inp, quant_dtype=a_dtype)

score = torch.randn(TOKENS, EXPERTS, device=dev, dtype=torch.bfloat16)
topk_weight, topk_ids = torch.topk(torch.softmax(score.float(), dim=-1), TOPK, dim=-1)
topk_weight = topk_weight.to(torch.float32)   # aiter moe_sorting requires fp32 weights
topk_ids = topk_ids.to(torch.int32)

print(f"[micro] mode={MODE} a_dtype={a_dtype} tokens={TOKENS} experts={EXPERTS} topk={TOPK}")
print(f"[micro] a_qt={tuple(a_qt.shape)}/{a_qt.dtype}  w1_qt={tuple(w1_qt.shape)}/{w1_qt.dtype}")

def run():
    return fused_moe(a_qt, w1_qt, w2_qt, topk_weight, topk_ids,
                     quant_type=QuantType.per_1x32,
                     w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a_scale,
                     activation=ActivationType.Silu, dtype=torch.bfloat16)

out = run()                      # warm up / JIT
torch.cuda.synchronize()
for _ in range(ITERS):
    run()
torch.cuda.synchronize()
print(f"[micro] done, out={tuple(out.shape)}/{out.dtype}, {ITERS} iters")
