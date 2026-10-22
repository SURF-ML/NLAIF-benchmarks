#!/usr/bin/env bash
# Workload: ERA5 climate data set (B6) — random patch reads from a shared large file
# Size:     540 TiB  (one epoch over a 10× GraphCast-scale subset; full ERA5 is ~5 PB)
# Pattern:  Sequential write step creates the file, then a separate read step does random
#           patch reads (-z); only the read step is measured
# Note:     ntasks and ppn are CLI args (--ntasks, --ppn)
#
# Purpose / what we are testing
#   Models the I/O pattern of AI weather/climate models (e.g. GraphCast, Pangu-Weather)
#   that train on ERA5 reanalysis data stored as large gridded files (NetCDF/Zarr/HDF5).
#   Multiple dataloader workers read non-contiguous spatial patches or temporal windows from
#   the same shared file simultaneously — the defining characteristic of this workload.
#
#   Aggregate size derivation (GraphCast variable set, 0.25° resolution):
#     Grid:        721 × 1440 = 1,038,240 points
#     Fields:      6 surface + 6 variables × 37 pressure levels = 228 fields × float32
#     Per snapshot: 228 × 1,038,240 × 4 bytes ≈ 950 MiB
#     Training period: 40 years at 6-hourly resolution = 58,440 timesteps
#     GraphCast epoch: 58,440 × 950 MiB ≈ 54 TiB
#     This rule:   10 × 54 TiB = 540 TiB
#   An epoch reads a subset of ERA5 (~5 PB in total, per the ARCO-ERA5 copy). The 10×
#   factor represents training on a larger part of the dataset (~12% instead of ~1%); it
#   is NOT ten epochs over the same 54 TiB, which would mostly measure re-read caching.
#   This benchmark does NOT model a single training step (which reads O(1–32) snapshots,
#   i.e. ~1–30 GiB); it models one full epoch — one complete read of the training subset.
#
#   Two steps in one job (IOR_SPLIT_WR=true). The write step only creates the dataset and is
#   not measured: it writes sequentially, the fastest way to create the file (the runner
#   drops -z from it). The read step reads the file at random offsets, which are assigned to
#   ranks at random, so on N nodes a fraction (N-1)/N of the reads hits data written by
#   another node, defeating the client page cache. -C cannot do this here: IOR rejects -z
#   combined with -C on a shared file.
#
#   A single 540 TiB file needs wide striping on filesystems with a per-object size limit
#   (e.g. Lustre on ldiskfs); pre-create and tune the target directory accordingly.
#
#   No -F (shared file, not file-per-process): ERA5-based training stores data in one or a
#   few large files (NetCDF/Zarr); all dataloader workers read from the same file
#   concurrently, hitting the same storage targets and creating lock and cache contention.
#
#   TRANSFER_SIZE=4m represents one spatial patch: a 0.25° global field at one pressure
#   level is 721 × 1440 × 4 bytes ≈ 4.15 MiB, making 4m a natural transfer size.
#
#   -z (random offsets) is essential: dataloaders shuffle training samples, so consecutive
#   reads from different workers land at unrelated file offsets, defeating prefetch/readahead.
#
#   API: POSIX pread() is the correct model here, not MPI-IO. Real AI training dataloaders
#   (PyTorch DataLoader with h5py/zarr workers) are Python multiprocessing-based — each worker
#   process reads independently via standard file I/O, with no MPI layer. MPI-IO collective
#   I/O (MPI_File_read_at_all) would only help for regular strided access patterns where
#   ROMIO can coalesce scattered reads into contiguous chunks; with random offsets (-z) the
#   collective algorithm covers the same scattered regions and adds communication overhead
#   on top, giving no benefit over independent reads. POSIX is both representative and correct.
#
#   --dataPacketType=random regenerates every transfer, so the data cannot be compressed. It
#   does not prevent deduplication: IOR seeds each transfer with process number + file offset
#   (32 bits), so each process's data repeats every 4 GiB. Data reduction must be disabled for
#   the benchmark data. IOR's default (timestamp) is worse: the same buffer for every transfer.

# Total I/O volume: 540 TiB = one epoch over a 10× GraphCast-scale subset
AGGREGATE_SIZE="540TiB"

# Block and transfer sizes (starting hints; scaled automatically by runner)
BLOCK_SIZE="64m"
TRANSFER_SIZE="4m"     # One spatial patch: 0.25° global field at one level ≈ 4 MiB
SEGMENT_COUNT="1"

# Sequential write step, then random read step (see header)
IOR_SPLIT_WR=true

# Access pattern: random patch reads from a single shared file (no -F, no -C)
IOR_FLAGS=(
  "-w"              # Write step: create the synthetic dataset (no ERA5 pre-staging)
  "-r"              # Read step: random patch reads modelling dataloader access
  "-z"              # Random offsets for the read step: shuffled patch reads (mini-batch)
  "-e"              # fsync on close: the read step reads data that is on storage
  "-g"              # Barriers: synchronised phases
  "-a" "POSIX"      # POSIX API (pread proxy for HDF5/NetCDF parallel I/O)
  "--dataPacketType=random"  # Incompressible data (see header)
)
