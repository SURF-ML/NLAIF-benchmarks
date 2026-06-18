#!/usr/bin/env bash
# Workload: ImageNet-21k metadata benchmark — ~14M files, 21841 classes, 21841-way × depth-1 tree
# Total dataset size: ~2.0 TiB  (14M files × 150 KiB each)
# Pattern: Sequential by default; pass --randomize for random mini-batch lookup (-R)
# Note: Scaling/ntasks/ppn are CLI args (--scaling, --ntasks, --ppn)
#
# Purpose / what we are testing
#   Represents the full ImageNet-21k (Winter 2021) dataset in its real on-disk layout: one flat
#   directory per synset, images stored directly inside. Compared to ImageNet-1k, the directory
#   count is 22× higher while files per directory drops from ~1200 to ~641.
#
#   Directory structure: 21841 leaf dirs × ~641 files each.
#   The primary stress is on the MDS inode namespace: 21k+ directory inodes must be enumerated,
#   stat-ed, and kept accessible. This exercises directory cache capacity and the cost of
#   distributing concurrent requests across a much wider set of directories than 1k. With a
#   shuffled training loader (-R), requests scatter across all 21k dirs simultaneously, which
#   fragments the MDS working set far more than the 1k case.
#
#   File size and per-directory occupancy are unchanged from ImageNet-1k (~150 KiB, ~641 files).
#   The key comparison with the 1k rule is therefore a pure directory-count scaling test:
#   does metadata throughput degrade as the number of directories grows by 22×, and if so, how?

# Dataset: 14M files, ~150 KiB avg (simulates real ImageNet-21k structure)
TOTAL_FILES=14000000
AVG_FILE_SIZE="150KiB"

# Directory tree: flat structure (1 dir per class) — 21841 leaf dirs, ~641 files each
MDTEST_BRANCH=21841
MDTEST_DEPTH=1
MDTEST_TOTAL_ITEMS=14000000   # 14M files in ImageNet-21k dataset
MDTEST_ITERATIONS=1           # Single iteration (full dataset pass)

# Access pattern flags: sequential by default; --randomize appends -R at runtime
MDTEST_FLAGS=(
  "-b" "${MDTEST_BRANCH}"      # Branch factor: 21841 = one dir per ImageNet-21k class
  "-z" "${MDTEST_DEPTH}"       # Tree depth: 1 = flat (root → class_dir → files)
  "-i" "${MDTEST_ITERATIONS}"  # Iterations: repeated phases to measure consistency
  "-N" "1"                     # Skip inode cache: each task uses a unique node offset
  "-L"                         # Leaf only: place files in leaf directories only
  "-w" "153600"                # Write size: 150 KiB per file (avg ImageNet-21k file)
)
