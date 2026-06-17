#!/bin/bash
# Regenerate every benchmark dataset: 3 models × 3 workload shapes, n=1024 prompts each.
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

NUM_REQUESTS="${NUM_REQUESTS:-1024}"
OUT_DIR="$REPO_ROOT/datasets"
GEN="$REPO_ROOT/scripts/generate_custom_dataset.py"

# model short-name | tokenizer | extra generator flags
MODELS=(
    "llama70b|nvidia/Llama-3.3-70B-Instruct-FP8|"
    "gpt120b|openai/gpt-oss-120b|"
    "dsv4|deepseek-ai/DeepSeek-V4-Pro|--trust-remote-code"
)
# Fixed workload shapes: "input_len output_len"
SHAPES=("1024 8192" "4608 4608" "8192 1024")

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
