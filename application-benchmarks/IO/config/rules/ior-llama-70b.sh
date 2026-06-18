#!/usr/bin/env bash
# Model:   LLaMA 2 70B — training dataset I/O
# Size:    ~3.64 TiB  (2T tokens × 2 bytes/token, uint16 tokenisation)
# Pattern: Sequential by default; pass --randomize for shuffle-buffer access (-z)
# Note:    Scaling/ntasks/ppn are CLI args (--scaling, --ntasks, --ppn)
#
# Purpose / what we are testing
#   Models the dataloader I/O pattern during LLaMA 2 70B pretraining: the tokenised dataset
#   is stored as a set of binary shard files (one per GPU/rank), and the training loop streams
#   each shard sequentially, one transfer-sized chunk at a time. The write phase creates the
#   synthetic shards (the harness does not pre-stage data); the read phase is what matters.
#
#   The aggregate (3.64 TiB = 2T tokens × uint16) represents one full epoch — one complete
#   sequential pass through the training corpus. LLaMA 2 does not repeat data, so one run of
#   this benchmark corresponds to the I/O load of one full training run.
#
#   File-per-process (-F) matches the real shard layout. -C reorders tasks so each rank reads
#   a different rank's shard, ensuring the FS cache does not contain the data from the just-
#   completed write. TRANSFER_SIZE represents the streaming chunk size from each shard file.
#
#   Sequential mode measures sustained streaming throughput (the common case).
#   Random mode (-z) approximates a shuffle-buffer dataloader that reads random chunks from
#   each shard rather than linearly streaming it.

# Total I/O volume: 2T tokens × 2 bytes/token = 3725 GiB (distributed across tasks)
AGGREGATE_SIZE="3726GiB"

# Block and transfer sizes (starting hints; scaled automatically by runner)
BLOCK_SIZE="512m"
TRANSFER_SIZE="16m"    # Sequential shard streaming chunk size
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
