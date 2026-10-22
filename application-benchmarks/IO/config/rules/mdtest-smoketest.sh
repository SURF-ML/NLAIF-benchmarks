#!/usr/bin/env bash
# Workload: mdtest smoke test — the LVD-1689M rule (B7) at a size that runs in seconds
# Total dataset size: ~4.0 GiB  (13,824 files × 300 KiB each)
# Pattern: As mdtest-lvd-1689m-shallow.sh: create + write, random-order read (-e), remove;
#          no stat phase. Verbose (-v) to expose per-phase timing and errors quickly.
# Note: ntasks and ppn are CLI args (--ntasks, --ppn)
#
# Purpose / what we are testing
#   Not a performance benchmark. Checks the mdtest binary, the system file, the target
#   directory and job submission, with the flags of the LVD-1689M rule on a small
#   tree. The tree (b=3, depth=3, 27 leaf directories) is small and asymmetric to exercise tree
#   traversal without accumulating enough files to make failures slow to surface. The output
#   must show rates for File creation, File read and File removal, and 0.000 for File stat (that
#   phase does not run). Performance numbers are meaningless.

# Dataset: 13,824 files, 300 KiB each (as LVD-1689M)
TOTAL_FILES=13824
AVG_FILE_SIZE="300KiB"

# Directory tree: b=3, depth=3 → 27 leaf dirs; small enough for a fast sanity check
MDTEST_BRANCH=3
MDTEST_DEPTH=3
MDTEST_TOTAL_ITEMS=13824    # 27 leaves × 512 files (smoke-test scale)
MDTEST_ITERATIONS=1         # Single iteration (full dataset pass)

# Access pattern flags: as the LVD-1689M rule, plus -v
MDTEST_FLAGS=(
  "-b" "${MDTEST_BRANCH}"      # Branch factor
  "-z" "${MDTEST_DEPTH}"       # Tree depth
  "-i" "${MDTEST_ITERATIONS}"  # Iterations
  "-N" "1"                     # Node offset between phases: reads hit files from another node
  "-L"                         # Leaf only: files in leaf directories only
  "-F"                         # Files only: no separate directory test
  "-w" "307200"                # Write size: 300 KiB per file
  "-e" "307200"                # Read size: read each file back in full
  "-C"                         # Phase: create
  "-E"                         # Phase: read
  "-r"                         # Phase: remove
  "-R"                         # Random read order
  "--dataPacketType=random"    # Incompressible file contents
  "-v"                         # Verbose: per-phase timing
)
