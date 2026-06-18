#!/usr/bin/env bash
# Workload: IOR smoketest — minimal write + read to validate harness wiring and job submission
# Use before full benchmark campaigns to verify system config, SLURM submission, and output parsing
# Note: Scaling/ntasks/ppn are CLI args (--scaling, --ntasks, --ppn)
#
# Purpose / what we are testing
#   Not a performance benchmark. Validates that the IOR binary, filesystem path, SLURM job
#   submission, and output parsing all work correctly. Should complete in seconds.
#
#   Uses file-per-process (-F) to exercise the same code path as the checkpoint rules — a
#   shared-file smoketest would pass while the file-per-process path is broken. The write +
#   fsync + read + cold-cache sequence mirrors the checkpoint pattern end-to-end, making a
#   passing smoketest a meaningful pre-flight check before a production checkpoint run.
#
#   Performance numbers from this rule are meaningless — 10 GiB is too small to saturate
#   anything and will be dominated by job startup overhead. Interpret results as pass/fail.

# Total I/O volume (distributed across tasks)
AGGREGATE_SIZE="10GiB"

# Block and transfer sizes (starting hints; scaled automatically by runner)
BLOCK_SIZE="1g"
TRANSFER_SIZE="1m"
SEGMENT_COUNT="1"

# Access pattern: sequential write + read, file-per-process (mirrors checkpoint rules)
IOR_FLAGS=(
  "-w"              # Write phase
  "-r"              # Read phase
  "-F"              # File-per-process: exercises same path as checkpoint rules
  "-C"              # Reorder tasks for read: cold cache simulation
  "-e"              # fsync after close: validates durability path
  "-g"              # Barriers: synchronised phases
  "-a" "POSIX"      # POSIX API
)
