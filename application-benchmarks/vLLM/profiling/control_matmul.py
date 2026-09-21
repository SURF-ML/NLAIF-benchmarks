#!/usr/bin/env python3
"""Control: prove the MFMA counters actually register on this setup.

Runs a bf16 matmul and an fp8 matmul. If SQ_INSTS_VALU_MFMA_MOPS_BF16 and _F8 both
come back non-zero here, the collection path works — so a zero F6F4 in the real probe
means "no 4-bit matrix ops", not "counters broken". Without this control a null result
is uninterpretable.
"""
import torch
n = 4096
a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
for _ in range(20):
    torch.mm(a, b)
try:
    af = a.to(torch.float8_e4m3fnuz if torch.cuda.get_device_capability()[0] < 9 else torch.float8_e4m3fn)
    bf = b.to(af.dtype)
    scale = torch.tensor(1.0, device="cuda")
    for _ in range(20):
        torch._scaled_mm(af, bf.t().contiguous().t(), scale_a=scale, scale_b=scale,
                         out_dtype=torch.bfloat16)
    print("[control] bf16 + fp8 matmuls done")
except Exception as e:
    print(f"[control] fp8 matmul unavailable: {type(e).__name__}: {e}")
torch.cuda.synchronize()
