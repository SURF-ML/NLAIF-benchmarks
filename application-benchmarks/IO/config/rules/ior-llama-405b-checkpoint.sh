#!/usr/bin/env bash
# Model:   LLaMA 3.1 405B — training checkpoint I/O (weights + optimiser states)
# Size:    ~4.05 TiB per checkpoint  (FP16 weights: 810 GiB + FP32 Adam states: 3240 GiB)
# Pattern: Sequential write then read, file-per-process (one shard per rank)
# Note:    Scaling/ntasks/ppn are CLI args (--scaling, --ntasks, --ppn)
#
# Purpose / what we are testing
#   Same pattern as the LLaMA 2 70B checkpoint rule, scaled to LLaMA 3.1 405B. Each rank
#   saves its shard of model weights and optimiser states; the benchmark measures how long
#   that save (and a subsequent restore) takes at this scale.
#
#   Size breakdown per checkpoint:
#     FP16 weights:       405B params × 2 bytes              =  810 GiB
#     FP32 Adam states:   405B params × 4 bytes × 2 (m+v)   = 3240 GiB
#     Total:                                                   4050 GiB ≈ 4 TiB
#
#   At 4 TiB per checkpoint, checkpoint latency becomes a dominant training overhead.
#   A cluster saving 4 TiB at 100 GB/s takes ~40 seconds per checkpoint; at 10 GB/s it
#   takes ~7 minutes. For a model trained for weeks, this cost recurs every few hundred
#   steps and adds up. This benchmark directly measures the achievable checkpoint bandwidth
#   and lets operators determine safe checkpoint frequencies.
#
#   The 405B rule differs from 70B not just in size but in the stress it places on the FS:
#   with more ranks (typical 405B training uses 512-2048 GPUs), the file-per-process pattern
#   creates thousands of simultaneous large sequential writes — a worst-case for MDS and
#   OST contention. This exercises both the data path (write throughput) and the namespace
#   path (file creation rate).

# Total I/O volume: 4050 GiB per checkpoint (distributed across tasks)
AGGREGATE_SIZE="4050GiB"

# Block and transfer sizes (starting hints; scaled automatically by runner)
BLOCK_SIZE="1g"
TRANSFER_SIZE="64m"    # Large sequential shard writes; smaller values waste syscall overhead
SEGMENT_COUNT="1"      # Runner auto-scales based on scaling mode and AGGREGATE_SIZE

# Access pattern flags: sequential file-per-process checkpoint write + restore
IOR_FLAGS=(
  "-w"              # Write phase: checkpoint save (critical path, blocks training)
  "-r"              # Read phase: checkpoint restore (recovery after failure)
  "-F"              # File-per-process: each rank owns its own shard
  "-C"              # Reorder tasks for read: cold cache (rank i reads rank i+1's shard)
  "-e"              # fsync after close: required for checkpoint durability
  "-g"              # Barriers: synchronised phases (all ranks must finish before next phase)
  "-a" "POSIX"      # POSIX API (fopen/fwrite/fread, same as torch.save / torch.load)
)
