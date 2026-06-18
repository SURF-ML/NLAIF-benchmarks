#!/usr/bin/env bash
# Workload: LVD-1689M metadata benchmark — ~1.7B files, shallow prefix (25000-way × depth-1)
# Total dataset size: ~475 TiB  (1.7B files × 300 KiB each)
# Pattern: Sequential by default; pass --randomize for random mini-batch lookup (-R)
# Note: Scaling/ntasks/ppn are CLI args (--scaling, --ntasks, --ppn)
#
# Purpose / what we are testing
#   Same corpus as the hashed variant (LVD-1689M, 1.7B files) but with a flat single-level
#   layout. The directory count (25000) is inferred from the DINOv3 paper's top-level k-means
#   clustering granularity (25k clusters), which is the coarsest semantic grouping used to
#   balance the dataset and a plausible basis for shallow directory organization.
#
#   Directory structure: 25000 leaf dirs × ~67600 files each, 1 level deep.
#   Compared to the hashed variant: 2.6× fewer directories, 2.6× more files per directory,
#   one fewer MDS lookup per file open (depth-1 vs depth-2). This isolates two competing
#   effects and lets us compare them directly on the same corpus:
#     - Hashed: more directories (65k), lower per-dir occupancy (~26k files), deeper path
#     - Shallow: fewer directories (25k), higher per-dir occupancy (~68k files), flat path
#
#   High per-directory file counts (~68k) stress directory entry lookup and readdir performance.
#   Many filesystems (including Lustre with large_dir) handle this well, but the cost of
#   opening a random file in a 68k-entry directory differs from a 26k-entry one depending on
#   whether the FS uses a hash-tree or linear scan internally.
#
#   The pair of hashed + shallow rules answers: given a fixed file count and file size, does
#   directory depth or directory occupancy matter more for metadata throughput on this FS?

# Dataset: 1.7B files, ~300 KiB avg
TOTAL_FILES=1700000000
AVG_FILE_SIZE="300KiB"

# Directory tree: shallow prefix (1 level, 25k dirs) — inferred from DINOv3 top-level k-means
# cluster count (25k clusters); ~67k files per dir vs ~26k for the hashed variant
MDTEST_BRANCH=25000
MDTEST_DEPTH=1
MDTEST_TOTAL_ITEMS=1700000000  # 1.7B files in LVD-1689M dataset
MDTEST_ITERATIONS=1            # Single iteration (full dataset pass)

# Access pattern flags: sequential by default; --randomize appends -R at runtime
MDTEST_FLAGS=(
  "-b" "${MDTEST_BRANCH}"      # Branch factor: filesystem hierarchy structure
  "-z" "${MDTEST_DEPTH}"       # Tree depth: levels of directory nesting
  "-i" "${MDTEST_ITERATIONS}"  # Iterations: repeated phases to measure consistency
  "-N" "1"                     # Skip inode cache: each task uses a unique node offset
  "-L"                         # Leaf only: place files in leaf directories only
  "-w" "307200"                # Write size: 300 KiB per file (avg LVD-1689M file)
)
