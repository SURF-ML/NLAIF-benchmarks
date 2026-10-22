#!/usr/bin/env bash
# Workload: IOR smoke test — the ERA5 rule (B6) at a size that runs in seconds
# Size:     10 GiB
# Pattern:  As ior-scientific.sh: sequential write step, then random-offset read step on one
#           shared file, incompressible data
# Note:     ntasks and ppn are CLI args (--ntasks, --ppn)
#
# Purpose / what we are testing
#   Not a performance benchmark. Checks the IOR binary, the system file, the target directory,
#   and job submission, on the same code path as ior-scientific.sh: shared file,
#   split write/read steps, random offsets (-z) with a separate read seed, fsync at close and
#   random data. Run it before the full ERA5 run. Performance numbers are meaningless; 10 GiB
#   is served largely from caches.

# Total I/O volume (distributed across tasks)
AGGREGATE_SIZE="10GiB"

# Block and transfer sizes (starting hints; scaled automatically by runner)
BLOCK_SIZE="64m"
TRANSFER_SIZE="4m"
SEGMENT_COUNT="1"

# Sequential write step, then random read step, as in ior-scientific.sh
IOR_SPLIT_WR=true

IOR_FLAGS=(
  "-w"              # Write step
  "-r"              # Read step
  "-z"              # Random offsets for the read step
  "-e"              # fsync on close
  "-g"              # Barriers: synchronised phases
  "-a" "POSIX"      # POSIX API
  "--dataPacketType=random"  # Incompressible data
)
