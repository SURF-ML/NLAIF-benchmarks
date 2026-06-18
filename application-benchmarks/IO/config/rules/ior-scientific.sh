#!/usr/bin/env bash
# Workload: ERA5-inspired scientific dataset — random patch reads from a shared large file
# Size:     ~54 TiB  (one training epoch at GraphCast scale; full ERA5 reanalysis is ~3 PB)
# Pattern:  Random spatial/temporal patch reads (-z) from a single shared file
# Note:     Scaling/ntasks/ppn are CLI args (--scaling, --ntasks, --ppn)
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
#     One epoch:   58,440 × 950 MiB ≈ 54 TiB
#   This benchmark does NOT model a single training step (which reads O(1–32) snapshots,
#   i.e. ~1–30 GiB); it models one full epoch — one complete read of the training subset.
#
#   Key distinction from LLaMA dataset rules: no -F (shared file, not file-per-process).
#   ERA5-based training stores data in one or a few large files (NetCDF/Zarr); all dataloader
#   workers read from the same file concurrently, hitting the same OSTs and creating lock and
#   cache contention absent in the per-rank shard pattern of LLaMA rules.
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

# Total I/O volume: ~54 TiB = one training epoch at GraphCast scale (58,440 snapshots × 950 MiB)
AGGREGATE_SIZE="54TiB"

# Block and transfer sizes (starting hints; scaled automatically by runner)
BLOCK_SIZE="64m"
TRANSFER_SIZE="4m"     # One spatial patch: 0.25° global field at one level ≈ 4 MiB
SEGMENT_COUNT="1"

# Access pattern: random patch reads from a single shared file (no -F)
IOR_FLAGS=(
  "-w"              # Write synthetic dataset (harness does not pre-stage ERA5 data)
  "-r"              # Read phase: random patch reads modelling dataloader access
  "-z"              # Random offsets: non-contiguous patch reads (shuffled mini-batch)
  "-C"              # Reorder tasks for read: cold cache simulation
  "-g"              # Barriers: synchronised read phases
  "-a" "POSIX"      # POSIX API (pread proxy for HDF5/NetCDF parallel I/O)
)
