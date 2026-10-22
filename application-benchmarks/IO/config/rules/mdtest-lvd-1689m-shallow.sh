#!/usr/bin/env bash
# Workload: LVD-1689M image data set (B7) — ~1.7B files in 25,000 directories (one level)
# Total dataset size: ~475 TiB  (1.7B files × 300 KiB each)
# Pattern: Create + write 300 KiB per file, then read every file back (-e) in random
#          order (-R), then remove; no stat phase. Only the read rate is reported.
# Note: ntasks and ppn are CLI args (--ntasks, --ppn)
#
# Purpose / what we are testing
#   LVD-1689M is Meta's DINOv3 pretraining corpus: 1.689B web images curated from ~17B Instagram
#   images, stored as individual files. This rule models a flat layout of 25,000 directories
#   with ~68,000 files each (25,000 is the number of top-level clusters in DINOv3's curation).
#   Meta does not document the storage layout; this is a model, not a copy.
#
#   The billion-file scale makes total inode count the primary FS constraint: 1.7B inodes
#   represents a significant fraction of what most parallel filesystems can hold. The rule is
#   intended to reveal whether the target FS can sustain AI training workloads at production
#   scale, not just at toy dataset sizes.
#
#   Directory count and process count: mdtest puts the same whole number of each process's files
#   in every leaf directory (it derives each file's directory from its index), and creates nothing
#   for a process with fewer files than directories. Every directory therefore holds k × ntasks
#   files, k >= 1. The run script chooses the smallest k for which ntasks × k × 25,000 reaches
#   1.7B, so the data set is never smaller: 1.7B at 68,000, 34,000, 22,667, ...
#   processes (k = 1, 2, 3, ...), larger in between by less than 1/(k - 1), which is less than
#   20% up to 14,336 processes. Above 68,000 processes k = 1, and every additional process adds
#   25,000 files (one per directory, ~7.15 GiB). A deeper hashed layout (256 × 256 = 65,536
#   directories) has fewer files per process per directory, so more process counts overshoot.
#
#   Phases run in order within one mdtest invocation: create (each file written with 300 KiB),
#   read (open + read 300 KiB + close), remove. -C -E -r select exactly these phases: a
#   training epoch opens, reads and closes every sample without a separate stat, and a stat
#   pass over the same files would only warm metadata caches for the read phase. -F skips
#   mdtest's separate directory test, which would otherwise create, stat, rename and remove
#   another 1.7B directories that are not part of the dataset. The read phase is what a
#   dataloader does per sample; without -e it would only open and close each file.
#   -R shuffles the read order, as a pretraining dataloader does, scattering requests across
#   all 25,000 directories.
#
#   -N 1 shifts the node offset between phases: mdtest multiplies the stride by the tasks per
#   node when ranks are block-placed on several nodes, so reads hit files created on another
#   node and the client cache cannot serve them. The shift accumulates per phase; on 2 nodes
#   it can wrap back to the writing node. Run on 3 or more nodes.
#
#   --dataPacketType=random makes the file contents incompressible. It does not prevent
#   deduplication: mdtest seeds each file's contents with process number + file index, so files
#   repeat across processes. Data reduction must be disabled for the benchmark data.

# Dataset: 1.7B files, ~300 KiB avg
TOTAL_FILES=1700000000
AVG_FILE_SIZE="300KiB"

# Directory tree: 25,000 directories, one level
MDTEST_BRANCH=25000
MDTEST_DEPTH=1
MDTEST_TOTAL_ITEMS=1700000000  # At least 1.7B files (LVD-1689M); more if the process count requires
MDTEST_ITERATIONS=1            # Single iteration (full dataset pass)

# Access pattern flags: create, random-order read, remove (no stat)
MDTEST_FLAGS=(
  "-b" "${MDTEST_BRANCH}"      # Branch factor: 25,000 directories
  "-z" "${MDTEST_DEPTH}"       # Tree depth: one level
  "-i" "${MDTEST_ITERATIONS}"  # Iterations: one pass over the data set
  "-N" "1"                     # Node offset between phases: reads hit files from another node
  "-L"                         # Leaf only: place files in leaf directories only
  "-F"                         # Files only: no separate directory test (create/stat/rename/remove of n dirs per task)
  "-w" "307200"                # Write size: 300 KiB per file (avg LVD-1689M file)
  "-e" "307200"                # Read size: read each file back in full in the read phase
  "-C"                         # Phase: create (and write 300 KiB per file)
  "-E"                         # Phase: read (open, read 300 KiB, close)
  "-r"                         # Phase: remove
  "-R"                         # Random read order (shuffled dataloader)
  "--dataPacketType=random"    # Incompressible file contents
)
