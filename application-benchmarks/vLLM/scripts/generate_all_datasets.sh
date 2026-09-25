#!/bin/bash
# Regenerate every benchmark dataset: the scored model × each workload shape.
#
# Datasets are tokenizer-specific (token counts depend on the model's tokenizer), so each
# model gets its own files. Output filenames match what the bench configs reference:
#     datasets/<short>_i<in>_o<out>_n1024.jsonl
#
# Runs the generator inside $CONTAINER_EXEC (transformers + numpy live in the container).
# CPU-only — no GPU needed. Submit as a SLURM job or run directly in an allocation:
#
#     sbatch scripts/generate_all_datasets.sh            # as a batch job
#     CONTAINER_EXEC="apptainer exec ... sif" bash scripts/generate_all_datasets.sh
#
#SBATCH --job-name=gen-datasets
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/gen-datasets-%j.out
#SBATCH --error=logs/gen-datasets-%j.err

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && ! [ -t 0 ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

# Optional: source a cluster system.sh (sets CONTAINER_EXEC, HF_HOME) when CLUSTER is given.
if [ -n "${CLUSTER:-}" ] && [ -f "configs/$CLUSTER/system.sh" ]; then
    source "configs/$CLUSTER/system.sh"
fi
CONTAINER_EXEC="${CONTAINER_EXEC:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Each run needs num_prompts = 2 x max_concurrency, and the max batch size N is vendor-chosen
# per layout, so the hosted files must cover the largest N anyone plausibly declares. 32768
# prompts covers N up to 16384; raise it (and DATASET_SIZES in gen_configs.py) if that is ever
# not enough. The count lands in the filename as the _n<N> suffix.
NUM_REQUESTS="${NUM_REQUESTS:-32768}"
OUT_DIR="$REPO_ROOT/datasets"
GEN="$REPO_ROOT/scripts/generate_custom_dataset.py"

# model short-name | tokenizer | extra generator flags
# Datasets are tokenizer-specific, so only the scored model needs one. The unscored smoke-test
# models in gen_configs.py need their own files if you run them; add them here temporarily.
MODELS=(
    "dsv4|deepseek-ai/DeepSeek-V4-Pro|--trust-remote-code"
)
# Fixed workload shapes: "input_len output_len". Both sum to 9216 = --max-model-len.
SHAPES=("1024 8192" "8192 1024")

for entry in "${MODELS[@]}"; do
    IFS='|' read -r short tokenizer extra <<<"$entry"
    for shape in "${SHAPES[@]}"; do
        read -r in_len out_len <<<"$shape"
        echo "=== $short  i${in_len}_o${out_len}  (tokenizer: $tokenizer) ==="
        # shellcheck disable=SC2086
        ${CONTAINER_EXEC:-} python3 "$GEN" \
            --tokenizer "$tokenizer" \
            --model-short-name "$short" \
            --input-len "$in_len" \
            --output-len "$out_len" \
            --num-requests "$NUM_REQUESTS" \
            --range-ratio 0.0 \
            --seed 0 \
            --output-dir "$OUT_DIR" \
            $extra
    done
done

echo "All datasets written to $OUT_DIR"
