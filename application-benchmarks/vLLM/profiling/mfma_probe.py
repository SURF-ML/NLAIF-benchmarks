#!/usr/bin/env python3
"""Count MFMA ops by operand width while serving DeepSeek-V4, to check whether the
FP4 matrix path is actually used.

gfx950 separates 8-bit from 4/6-bit matrix ops in hardware:
    SQ_INSTS_VALU_MFMA_MOPS_F8     8-bit MFMA
    SQ_INSTS_VALU_MFMA_MOPS_F6F4   4/6-bit MFMA
so the F8:F6F4 ratio settles what vLLM's log strings only claim. Run under rocprofv3.

Kept deliberately small: one short generate on an already-warm engine. We want the
counter ratio, not a throughput number.
"""
import os, sys, time

def main():
    model = sys.argv[1]
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model, trust_remote_code=True, tokenizer_mode="deepseek_v4",
        tensor_parallel_size=8, max_model_len=2048, max_num_seqs=8,
        gpu_memory_utilization=0.90, kv_cache_dtype="fp8",
        enforce_eager=True,                 # skip graph capture: faster, and counters
                                            # attribute to real kernels not replays
        enable_expert_parallel=os.environ.get("PROBE_EP", "1") == "1",
    )
    # warm up so one-off init kernels are not what we count
    llm.generate(["warmup"] * 4, SamplingParams(max_tokens=8, ignore_eos=True))
    t = time.time()
    out = llm.generate(["The capital of France is"] * 8,
                       SamplingParams(max_tokens=64, ignore_eos=True))
    print(f"[probe] generated {len(out)} seqs in {time.time()-t:.1f}s", flush=True)

if __name__ == "__main__":
    main()
