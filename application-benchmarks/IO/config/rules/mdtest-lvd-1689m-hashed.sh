#!/usr/bin/env bash
# Workload: LVD-1689M metadata benchmark — ~1.7B files, hashed prefix (256-way × depth-2)
# Total dataset size: ~475 TiB  (1.7B files × 300 KiB each)
# Pattern: Sequential by default; pass --randomize for random mini-batch lookup (-R)
# Note: Scaling/ntasks/ppn are CLI args (--scaling, --ntasks, --ppn)
#
# Purpose / what we are testing
#   LVD-1689M is Meta's DINOv3 pretraining corpus: 1.689B web images curated from ~17B Instagram
#   images. At this scale no semantic organization is practical; real storage pipelines shard by
#   content hash. This rule models the standard 2-byte hex-prefix layout:
#     <hash[0:2]>/<hash[2:4]>/<image_file>
#   yielding 256² = 65536 leaf directories with ~25900 files each.
#
#   Directory structure: 65536 leaf dirs × ~25900 files, 2 levels deep.
#   This is the highest directory-occupancy rule in the suite (aside from the shallow variant).
#   Each file open requires two MDS lookups (one per path component above the file), so path
#   resolution cost is doubled relative to depth-1 rules. The MDS must also handle 65k hot
#   directory inodes concurrently.
#
#   The billion-file scale makes total inode count the primary FS constraint: 1.7B inodes
#   represents a significant fraction of what most parallel filesystems can hold. This rule is
#   the most demanding in the suite and is intended to reveal whether the target FS can sustain
#   AI training workloads at production scale, not just at toy dataset sizes.
#
#   Sequential mode tests epoch-level throughput (how fast can all 1.7B files be stat-ed once).
#   Random mode (-R) approximates a pretraining dataloader that samples uniformly across the
#   full corpus, scattering requests across all 65k directories.

# Dataset: 1.7B files, ~300 KiB avg
TOTAL_FILES=1700000000
AVG_FILE_SIZE="300KiB"

# Directory tree: hashed prefix (256-way branching, depth 2)
MDTEST_BRANCH=256
MDTEST_DEPTH=2
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
