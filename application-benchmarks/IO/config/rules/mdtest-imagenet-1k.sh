#!/usr/bin/env bash
# Workload: ImageNet-1k metadata benchmark — ~1.2M files, 1k classes, 1000-way × depth-1 tree
# Total dataset size: ~172 GiB  (1.2M files × 150 KiB each)
# Pattern: Sequential by default; pass --randomize for random mini-batch lookup (-R)
# Note: Scaling/ntasks/ppn are CLI args (--scaling, --ntasks, --ppn)
#
# Purpose / what we are testing
#   Represents a medium-scale supervised image classification dataset in its real on-disk layout:
#   one directory per class (synset), all images for that class stored flat inside it. The shared
#   directory tree (no -u) means all MPI tasks contend on the same 1000 class directories, which
#   is the actual condition during distributed training where every GPU worker reads from the same
#   mounted dataset.
#
#   Directory structure: 1000 leaf dirs × ~1200 files each.
#   This is moderate directory count with moderate occupancy — neither extreme. The MDS must hold
#   ~1000 directory inodes hot in cache and handle concurrent stat/open from all tasks into the
#   same small set of directories.
#
#   File size (~150 KiB) is small enough that file-data I/O is not the bottleneck; this is
#   primarily a metadata throughput test: how many file open/stat/close ops/sec can the filesystem
#   sustain under concurrent access from many ranks.
#
#   Sequential mode measures bulk ingestion and full-epoch read throughput.
#   Random mode (-R) measures mini-batch access latency, where a shuffled training loader
#   scatters requests across all class directories simultaneously.

# Dataset: 1.2M files, ~150 KiB avg (simulates real ImageNet-1k structure)
TOTAL_FILES=1200000
AVG_FILE_SIZE="150KiB"

# Directory tree: flat structure (1 dir per class) — 1000 leaf dirs, ~1200 files each
MDTEST_BRANCH=1000
MDTEST_DEPTH=1
MDTEST_TOTAL_ITEMS=1200000    # 1.2M files in ImageNet-1k dataset
MDTEST_ITERATIONS=1           # Single iteration (full dataset pass)

# Access pattern flags: sequential by default; --randomize appends -R at runtime
MDTEST_FLAGS=(
  "-b" "${MDTEST_BRANCH}"      # Branch factor: 1000 = one dir per ImageNet-1k class
  "-z" "${MDTEST_DEPTH}"       # Tree depth: 1 = flat (root → class_dir → files)
  "-i" "${MDTEST_ITERATIONS}"  # Iterations: repeated phases to measure consistency
  "-N" "1"                     # Skip inode cache: each task uses a unique node offset
  "-L"                         # Leaf only: place files in leaf directories only
  "-w" "153600"                # Write size: 150 KiB per file (avg ImageNet-1k file)
)
