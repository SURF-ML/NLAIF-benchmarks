# System: Snellius (SURF) — Rome partition, scratch-shared filesystem
#
# IOR installation (if not present):
#   module load 2025 foss/2025a Autoconf/2.72-GCCcore-14.2.0
#   git clone https://github.com/hpc/ior.git && cd ior
#   ./bootstrap && ./configure --prefix=$HOME/.local && make install

# SLURM configuration
CLUSTER="snellius"
PARTITION="rome"
TIME="12:00:00"
RUNNER="srun"
RUNNER_NTASKS_OPT="-n"
RUNNER_PPN_OPT="--ntasks-per-node"
FEATURES=""

# Hardware: 256 GiB RAM, 128 CPUs per node (used for memory hogging and task affinity)
DRAM_PER_NODE=256
CPUS_PER_NODE=128

# Module environment (loaded in order before benchmark)
ENV_SETUP=(
  "module load 2025"
  "module load foss/2025a"
)

# Benchmark executables (must be in PATH or use absolute paths)
IOR="$HOME/.local/bin/ior"
MDTEST="$HOME/.local/bin/mdtest"

# Benchmark output: results written to scratch-shared filesystem
FILESYSTEM="scratch-shared"
TARGET_DIR="/scratch-shared/$USER"
