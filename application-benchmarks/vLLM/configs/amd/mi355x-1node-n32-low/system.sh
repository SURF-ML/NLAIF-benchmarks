# Single-node N=32 layout sweep, "low" operating point only, pinned to mi355-gpu-13 so the two
# operating points run side by side on different nodes (see ../mi355x-1node-n32/system.sh).
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../mi355x-1node-n32" && pwd)/system.sh"
SBATCH_ARGS="-w mi355-gpu-13 --exclusive"  # --exclusive: without it Slurm pins the job to 2 hw threads (cpu=1) and vLLM is CPU-starved (~2x slower TPOT)
