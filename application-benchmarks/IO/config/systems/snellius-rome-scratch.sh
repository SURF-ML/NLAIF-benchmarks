# System: Snellius (SURF) — Rome partition, scratch-shared filesystem
# Example system file: copy it and set every value for the system under test (see INSTRUCTIONS.md).
#
# One IO500 build provides io500, ior and mdtest: release io500-sc26, checked out by commit.
# On Snellius the Autoconf module is needed for IOR's configure step:
#   module load 2025 foss/2025a Autoconf/2.72-GCCcore-14.2.0
#   git clone https://github.com/IO500/io500.git $HOME/.local/io500-sc26
#   cd $HOME/.local/io500-sc26 && git checkout ddabf2efc978f8650c29176ca422bb0bf3e8990a
#   ./prepare.sh

# SLURM configuration
CLUSTER="snellius"
PARTITION="rome"
TIME="12:00:00"
RUNNER="srun"
RUNNER_NTASKS_OPT="-n"
RUNNER_PPN_OPT="--ntasks-per-node"
FEATURES=""

# Hardware: 224 GiB of memory available to a job and 128 cores per node
# (used for memory hogging and task affinity)
DRAM_PER_NODE=224
CPUS_PER_NODE=128

# Module environment (loaded in order before benchmark)
ENV_SETUP=(
  "module load 2025"
  "module load foss/2025a"
)

# Benchmark executables from the IO500 build
IOR="$HOME/.local/io500-sc26/bin/ior"
MDTEST="$HOME/.local/io500-sc26/bin/mdtest"

# Benchmark output: results written to scratch-shared filesystem
FILESYSTEM="scratch-shared"
TARGET_DIR="/scratch-shared/$USER"
