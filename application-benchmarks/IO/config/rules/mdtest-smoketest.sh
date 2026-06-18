#!/usr/bin/env bash
# Workload: Smoketest — tiny synthetic dataset for functional verification
# Total dataset size: ~2.3 GiB  (16384 files × 150 KiB each)
# Pattern: Random (-R) + verbose (-v) to expose per-phase timing and any errors quickly
# Note: Scaling/ntasks/ppn are CLI args (--scaling, --ntasks, --ppn)
#
# Purpose / what we are testing
#   Not a performance benchmark. This rule exists to verify that the mdtest binary, the
#   target filesystem path, and the harness plumbing all work correctly before committing
#   to a multi-hour production run. It should complete in seconds at any task count.
#
#   Directory structure: b=3, depth=3 → 27 leaf dirs, ~607 files each (at default task count).
#   The tree is intentionally small and asymmetric (non-power-of-2 branching) to exercise
#   tree traversal without accumulating enough files to make failures slow to surface.
#
#   Random mode (-R) and verbose (-v) are fixed here: random access catches stat/lookup
#   failures that sequential access might mask, and verbose output makes it easy to see
#   which phase failed without digging into aggregate summaries.
#
#   Performance numbers from this rule are meaningless — dataset too small to saturate
#   anything. Interpret results as pass/fail only.

# Dataset: 16384 files, ~150 KiB avg
TOTAL_FILES=16384
AVG_FILE_SIZE="150KiB"

# Directory tree: b=3, depth=3 → 27 leaf dirs; small enough for a fast sanity check
MDTEST_BRANCH=3
MDTEST_DEPTH=3
MDTEST_TOTAL_ITEMS=16384    # 16384 files total (smoketest scale)
MDTEST_ITERATIONS=1         # Single iteration (full dataset pass)

# Access pattern flags: random lookup with persistent files (trains on shuffled dataset)
MDTEST_FLAGS=(
  "-b" "${MDTEST_BRANCH}"      # Branch factor: filesystem hierarchy structure
  "-z" "${MDTEST_DEPTH}"       # Tree depth: levels of directory nesting
  "-i" "${MDTEST_ITERATIONS}"  # Iterations: repeated phases to measure consistency
  "-N" "1"                     # Skip inode cache: each task uses a unique node offset
  "-L"                         # Leaf only: place files in leaf directories only
  "-w" "153600"                # Write size: 150 KiB per file (avg ImageNet-1k file)
  "-v"                         # Verbose: per-phase timing and stat output
)
