#!/usr/bin/env bash
# Model:   LLaMA 2 70B — training checkpoint I/O (weights + optimiser states)
# Size:    ~700 GiB per checkpoint  (FP16 weights: 140 GiB + FP32 Adam states: 560 GiB)
# Pattern: Sequential write then read, file-per-process (one shard per rank)
# Note:    Scaling/ntasks/ppn are CLI args (--scaling, --ntasks, --ppn)
#
# Purpose / what we are testing
#   Models the checkpoint save/restore cycle during LLaMA 2 70B training with FSDP or ZeRO:
#   each rank writes its own shard (FP16 model parameters + FP32 Adam m/v states) to a
#   separate file, then reads it back (resuming from a checkpoint exercises the read path).
#
#   Size breakdown per checkpoint:
#     FP16 weights:       70B params × 2 bytes              = 140 GiB
#     FP32 Adam states:   70B params × 4 bytes × 2 (m+v)   = 560 GiB
#     Total:                                                  700 GiB
#   (Excludes FP32 master weights and gradients, which are not always checkpointed.)
#
#   The write phase is the critical path: checkpointing blocks training and its latency
#   directly determines how long the cluster sits idle. The read phase (restore) is less
#   frequent but must also complete quickly to minimise recovery time after failure.
#
#   -e (fsync) is mandatory: without it, a node crash after IOR reports "done" but before
#   the OS flushes dirty pages would produce a corrupt checkpoint. This is a correctness
#   requirement, not just a performance concern.
#
#   TRANSFER_SIZE=64m targets large contiguous writes per rank. Smaller values increase
#   syscall count without improving throughput on Lustre/GPFS stripe boundaries.

# Total I/O volume: 700 GiB per checkpoint (distributed across tasks)
AGGREGATE_SIZE="700GiB"

# Block and transfer sizes (starting hints; scaled automatically by runner)
BLOCK_SIZE="256m"
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
  #"--posix.odirect" # O_DIRECT: bypass page cache — useful on systems with small RAM/node
)
