#!/usr/bin/env bash
# Model:   LLaMA 3.1 405B — training dataset I/O
# Size:    ~27.3 TiB  (15T tokens × 2 bytes/token, uint16 tokenisation)
# Pattern: Sequential by default; pass --randomize for shuffle-buffer access (-z)
# Note:    Scaling/ntasks/ppn are CLI args (--scaling, --ntasks, --ppn)
#
# Purpose / what we are testing
#   Same pattern as the LLaMA 2 70B dataset rule, scaled to LLaMA 3.1 405B's training corpus:
#   15T tokens streamed sequentially across all ranks in file-per-process shard layout.
#   The aggregate (27.94 TiB) represents one full epoch over the pretraining corpus.
#
#   The key difference from the 70B rule is scale: 7.5× more data means proportionally more
#   time under I/O load per epoch, and a larger shard per rank at the same task count. This
#   stresses sustained throughput over longer durations and exercises the FS at higher aggregate
#   bandwidth demand (more ranks × larger per-rank files).
#
#   TRANSFER_SIZE is larger than the 70B rule (32m vs 16m) reflecting that at 405B scale,
#   larger shard sizes favour bigger read chunks to amortise open/seek overhead per rank.
#
#   Write phase creates synthetic shards; the harness does not pre-stage data.

# Total I/O volume: 15T tokens × 2 bytes/token = 27940 GiB (distributed across tasks)
AGGREGATE_SIZE="27940GiB"

# Block and transfer sizes (starting hints; scaled automatically by runner)
BLOCK_SIZE="1g"
TRANSFER_SIZE="32m"    # Sequential shard streaming chunk size
SEGMENT_COUNT="1"      # Runner auto-scales based on scaling mode and AGGREGATE_SIZE

# Access pattern flags
IOR_FLAGS=(
  "-w"              # Write synthetic shards (harness does not pre-create data files)
  "-r"              # Read phase: models training dataloader streaming tokenised shards
  "-F"              # File-per-process: each GPU/task streams its own shard
  "-C"              # Reorder tasks for read: cold cache (rank i reads rank i+1's shard)
  "-g"              # Barriers: synchronised phases
  "-a" "POSIX"      # POSIX API (fopen/fread, same as torch DataLoader with mmap disabled)
)
