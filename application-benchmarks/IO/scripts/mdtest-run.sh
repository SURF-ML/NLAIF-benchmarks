#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Helpers
# ============================================================
usage() {
  cat <<'EOF'
Usage:
  ./scripts/mdtest-run.sh --system=<path> --rule=<path> [options]

Required:
  --system=PATH    Path to system config file (e.g. config/systems/snellius-rome-scratch.sh)
  --rule=PATH      Path to benchmark rule file (e.g. config/rules/mdtest-imagenet-21k-rand-14m.sh)

Scaling options:
  --scaling=TYPE   single (default), weak, or strong
                   weak:   per-task items fixed (MDTEST_TOTAL_ITEMS); aggregate grows with ntasks
                   strong: total items fixed (MDTEST_TOTAL_ITEMS); per-task = total / ntasks
  --ntasks=LIST    Comma or space-separated task counts
                   (default: CPUS_PER_NODE for single; 1,2,4,8,16 nodes for weak/strong)
  --ppn=N          Tasks per node (default: CPUS_PER_NODE from system file)

Optional:
  --randomize      Add -R (random lookup) to mdtest flags; results go to <rule>-randomize/
  --interactive    Print commands to stdout instead of generating SLURM scripts
  --dry-run        Generate scripts without submitting
  --chain          Submit jobs sequentially (each depends on the previous)
EOF
}

# Detect if variable is an array
is_array() { declare -p "$1" 2>/dev/null | grep -q "declare -a" || return 1; }

# Calculate total nodes in directory tree: (b^(d+1) - 1) / (b - 1) for b > 1, else d + 1
# where b = branching factor, d = depth
tree_node_count() {
  local branch=$1 depth=$2
  if (( branch > 1 )); then
    echo $(( (branch ** (depth + 1) - 1) / (branch - 1) ))
  else
    echo $(( depth + 1 ))
  fi
}

# ============================================================
#  Parse arguments
# ============================================================
SYSTEM_FILE=""
RULE_FILE=""
DRY_RUN=false
CHAIN=false
RANDOMIZE=false
INTERACTIVE=false
SCALING="single"   # default: one-shot run; use weak/strong for sweep
NTASKS_LIST=""     # default: derived from CPUS_PER_NODE after system file is sourced
PPN=""             # default: CPUS_PER_NODE from system file; override with --ppn

for arg in "$@"; do
  case "$arg" in
    --system=*) SYSTEM_FILE="${arg#--system=}" ;;
    --rule=*)   RULE_FILE="${arg#--rule=}" ;;
    --scaling=*) SCALING="${arg#--scaling=}" ;;
    --ntasks=*) NTASKS_LIST="${arg#--ntasks=}" ;;
    --ppn=*)    PPN="${arg#--ppn=}" ;;
    --randomize)    RANDOMIZE=true ;;
    --interactive)  INTERACTIVE=true ;;
    --dry-run)      DRY_RUN=true ;;
    --chain)    CHAIN=true ;;
    -h|--help)  usage; exit 0 ;;
    *)          echo "Unknown argument: $arg" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$SYSTEM_FILE" ]]; then
  echo "ERROR: --system is required." >&2
  usage
  exit 1
fi

if [[ -z "$RULE_FILE" ]]; then
  echo "ERROR: --rule is required." >&2
  usage
  exit 1
fi

# ============================================================
#  Validate system and rule files, set up environment
# ============================================================
if [[ ! -f "$SYSTEM_FILE" ]]; then
  echo "ERROR: system file not found: $SYSTEM_FILE" >&2
  exit 1
fi
# System file contract: CLUSTER, PARTITION, TIME, RUNNER, FEATURES, CPUS_PER_NODE,
# DRAM_PER_NODE, ENV_SETUP, IOR, MDTEST, FILESYSTEM, TARGET_DIR
# shellcheck source=/dev/null
source "$SYSTEM_FILE"

if [[ -z "${MDTEST:-}" || ! -x "${MDTEST}" ]]; then
  echo "ERROR: MDTEST executable not found or not executable: ${MDTEST:-}" >&2
  exit 1
fi

if ! $DRY_RUN && ! $INTERACTIVE && ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is required but not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$RULE_FILE" ]]; then
  echo "ERROR: rule file not found: $RULE_FILE" >&2
  exit 1
fi
# Rule file contract: MDTEST_BRANCH, MDTEST_DEPTH, MDTEST_TOTAL_ITEMS, MDTEST_ITERATIONS,
# MDTEST_FLAGS (all flags except -n, which the runner derives from scaling mode and ntasks)
# shellcheck source=/dev/null
source "$RULE_FILE"

# Validate mandatory rule parameters
for var in MDTEST_BRANCH MDTEST_DEPTH MDTEST_TOTAL_ITEMS MDTEST_ITERATIONS; do
  [[ -n "${!var:-}" ]] || { echo "ERROR: rule file must define $var" >&2; exit 1; }
done

if ! is_array MDTEST_FLAGS || [[ ${#MDTEST_FLAGS[@]} -eq 0 ]]; then
  echo "ERROR: rule file must define MDTEST_FLAGS as a non-empty array" >&2
  exit 1
fi

# Compute total nodes in the directory tree for validation
MDTEST_TREE_NODES=$(tree_node_count "$MDTEST_BRANCH" "$MDTEST_DEPTH")

# Determine tasks per node: --ppn flag > system CPUS_PER_NODE > fallback 1
TASKS_PER_NODE="${PPN}"
if [[ -z "$TASKS_PER_NODE" ]]; then
  TASKS_PER_NODE=${CPUS_PER_NODE:-${cpus_per_node:-1}}
fi

# Guard against requesting more tasks per node than CPUs available
cpus_avail=${CPUS_PER_NODE:-${cpus_per_node:-${NODE_CPUS:-}}}
if [[ -n "$cpus_avail" ]]; then
  if (( TASKS_PER_NODE > cpus_avail )); then
    echo "WARNING: TASKS_PER_NODE ($TASKS_PER_NODE) > CPUS_PER_NODE ($cpus_avail). Adjusting to $cpus_avail." >&2
    TASKS_PER_NODE=$cpus_avail
  fi
fi

# Accept comma- or space-separated --ntasks list and split into an array
NTASKS_ARR=()
if [[ -n "$NTASKS_LIST" ]]; then
  NTASKS_LIST="${NTASKS_LIST//,/ }"
  read -r -a NTASKS_ARR <<< "$NTASKS_LIST"
fi

# Validate TOTAL_ITEMS required (primary size parameter)
if [[ -z "$MDTEST_TOTAL_ITEMS" ]]; then
  echo "ERROR: rule file must define MDTEST_TOTAL_ITEMS" >&2
  exit 1
fi

# Resolve defaults and validate for each scaling mode
# Note: MDTEST_TOTAL_ITEMS is total items for the benchmark (like AGGREGATE_SIZE for IOR)
case "${SCALING,,}" in
  weak)
    # Per-node items constant; aggregate metadata load grows with nodes
    # MDTEST_TOTAL_ITEMS = per-node items (fixed regardless of ppn or ntasks)
    if [[ -z "$NTASKS_LIST" ]]; then
      NTASKS_ARR=()
      for n in 1 2 4 8 16; do NTASKS_ARR+=( $(( n * TASKS_PER_NODE )) ); done
    fi
    echo "Scaling mode: WEAK (per-node items: $MDTEST_TOTAL_ITEMS, aggregate grows with nodes)"
    ;;
  strong)
    # Per-node items constant; total items stay constant
    # Distributes across increasing nodes as ntasks grows
    if [[ -z "$NTASKS_LIST" ]]; then
      NTASKS_ARR=()
      for n in 1 2 4 8 16; do NTASKS_ARR+=( $(( n * TASKS_PER_NODE )) ); done
    fi
    echo "Scaling mode: STRONG (constant total items: $MDTEST_TOTAL_ITEMS, per-node shrinks with ntasks)"
    ;;
  single)
    # Total items = MDTEST_TOTAL_ITEMS for that specific ntasks value
    if [[ -z "$NTASKS_LIST" ]]; then
      NTASKS_ARR=( "$TASKS_PER_NODE" )
    fi
    if (( ${#NTASKS_ARR[@]} != 1 )); then
      echo "ERROR: --scaling=single expects a single ntasks value, got: $NTASKS_LIST" >&2
      exit 1
    fi
    echo "Scaling mode: SINGLE (ntasks=${NTASKS_ARR[0]}, total items: $MDTEST_TOTAL_ITEMS)"
    ;;
  *)
    echo "ERROR: unknown scaling type: $SCALING. Use single, weak, or strong." >&2
    exit 1
    ;;
esac

NTASKS_LIST="${NTASKS_ARR[*]}"

# Inject random-lookup flag when --randomize is set
if $RANDOMIZE; then
  MDTEST_FLAGS+=("-R")
fi

# Shell-quote MDTEST_FLAGS for safe heredoc embedding (protects special chars and spaces)
# Note: -n (items per task) is computed dynamically in the loop based on scaling mode
MDTEST_FLAGS_STR="$(printf '%q ' "${MDTEST_FLAGS[@]}")"

# Results path: results/<rule>[-randomize]/<cluster>/<filesystem>/<timestamp>/
RULE_NAME=$(basename "$RULE_FILE" .sh)
$RANDOMIZE && RULE_NAME="${RULE_NAME}-randomize"
TARGET_DIR="${TARGET_DIR}/${RULE_NAME}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
CASE_DIR="$REPO_ROOT/results/$RULE_NAME/$CLUSTER/$FILESYSTEM/$TIMESTAMP"

mkdir -p "$CASE_DIR"

# Write metadata: system config, benchmark parameters, and scaling setup
cat > "$CASE_DIR/mdtest.md" <<EOF
# MDTEST ${SCALING^} Scaling

cluster = $CLUSTER
filesystem = $FILESYSTEM
partition = $PARTITION
date = $TIMESTAMP
time_limit = $TIME
runner = $RUNNER
scaling_mode = $SCALING

## MDTEST Parameters

mdtest = $MDTEST
branch = $MDTEST_BRANCH
depth = $MDTEST_DEPTH
total_items = $MDTEST_TOTAL_ITEMS
randomize = $RANDOMIZE
flags = $MDTEST_FLAGS_STR
total_files = ${TOTAL_FILES:-}
avg_file_size = ${AVG_FILE_SIZE:-}

## Scaling

scaling_type = $SCALING
tasks_per_node = $TASKS_PER_NODE
ntasks_list = $NTASKS_LIST
target_dir = $TARGET_DIR
EOF

echo "Case: $CASE_DIR"
echo "  branch=$MDTEST_BRANCH  depth=$MDTEST_DEPTH  total_items=$MDTEST_TOTAL_ITEMS  iterations=$MDTEST_ITERATIONS"

# For weak/single modes the items per task is fixed; warn if it doesn't divide evenly
if [[ "${SCALING,,}" != "strong" ]]; then
  remainder=$(( MDTEST_TOTAL_ITEMS % MDTEST_TREE_NODES ))
  if (( remainder != 0 )); then
    effective=$(( (MDTEST_TOTAL_ITEMS / MDTEST_TREE_NODES) * MDTEST_TREE_NODES ))
    echo "WARNING: MDTEST_TOTAL_ITEMS ($MDTEST_TOTAL_ITEMS) not divisible by tree nodes ($MDTEST_TREE_NODES); mdtest will use $effective items/task."
  fi
fi
echo ""

# Generate one SLURM script per ntasks value
PREV_JOBID=""
for ntasks in $NTASKS_LIST; do
  nodes=$(( (ntasks + TASKS_PER_NODE - 1) / TASKS_PER_NODE ))
  ppn=$(( ntasks < TASKS_PER_NODE ? ntasks : TASKS_PER_NODE ))

  # Compute per-task items based on scaling mode and MDTEST_TOTAL_ITEMS
  case "${SCALING,,}" in
    strong)
      # Strong: total items stay constant, per-node shrinks as nodes increase
      PER_NODE_ITEMS=$(( MDTEST_TOTAL_ITEMS / nodes ))
      items_per_task=$(( PER_NODE_ITEMS / ppn ))
      total_items=$MDTEST_TOTAL_ITEMS
      ;;
    weak)
      # Weak: per-node items constant, total grows with nodes
      PER_NODE_ITEMS=$MDTEST_TOTAL_ITEMS
      items_per_task=$(( PER_NODE_ITEMS / ppn ))
      total_items=$(( PER_NODE_ITEMS * nodes ))
      ;;
    single)
      # Single: total items = MDTEST_TOTAL_ITEMS, distributed across nodes and tasks
      PER_NODE_ITEMS=$(( MDTEST_TOTAL_ITEMS / nodes ))
      items_per_task=$(( PER_NODE_ITEMS / ppn ))
      total_items=$MDTEST_TOTAL_ITEMS
      ;;
  esac

  # Ensure at least 1 item per task
  (( items_per_task < 1 )) && items_per_task=1

  job_name="mdtest_${SCALING}_n${ntasks}"

  if $INTERACTIVE; then
    echo ""
    echo "# ntasks=${ntasks}  nodes=${nodes}  ppn=${ppn}  items/task=${items_per_task}  total=${total_items}"
    echo "mkdir -p \"${TARGET_DIR}/${TIMESTAMP}\""
    echo "${RUNNER} ${RUNNER_NTASKS_OPT} ${ntasks} ${RUNNER_PPN_OPT} ${ppn} ${MDTEST} ${MDTEST_FLAGS_STR} -n ${items_per_task} -d \"${TARGET_DIR}/${TIMESTAMP}\" \\"
    echo "  > \"${CASE_DIR}/${job_name}.out\" 2> \"${CASE_DIR}/${job_name}.err\""
    continue
  fi
  script="$CASE_DIR/${job_name}.sh"

  cat > "$script" <<SLURM
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${ntasks}
#SBATCH --ntasks-per-node=${ppn}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${CASE_DIR}/${job_name}_%j.out
#SBATCH --error=${CASE_DIR}/${job_name}_%j.err
#SBATCH --constraint=${FEATURES}
#SBATCH --exclusive

# Load system-specific modules and environment setup
$(printf '%s\n' "${ENV_SETUP[@]}")

# Keep file permissions predictable and owner-writable
umask 022

# Per-job directory prevents collisions with stale test-dir.* trees from reruns
TARGET_DIR="${TARGET_DIR}/\${SLURM_JOB_ID}"
mkdir -p "\${TARGET_DIR}"

echo "MDTEST_METADATA: scaling_mode = ${SCALING}"
echo "MDTEST_METADATA: ntasks = ${ntasks}"
echo "MDTEST_METADATA: nodes = ${nodes}"
echo "MDTEST_METADATA: ppn = ${ppn}"
echo "MDTEST_METADATA: items_per_task = ${items_per_task}"
echo "MDTEST_METADATA: total_items = ${total_items}"
echo "MDTEST_METADATA: total_files = ${TOTAL_FILES:-}"
echo "MDTEST_METADATA: avg_file_size = ${AVG_FILE_SIZE:-}"

echo "MDTEST_COMMAND: ${RUNNER} ${RUNNER_NTASKS_OPT} ${ntasks} ${RUNNER_PPN_OPT} ${ppn} ${MDTEST} ${MDTEST_FLAGS_STR} -n ${items_per_task} -d \"\${TARGET_DIR}\""

# MDTEST benchmark command with ${SCALING} scaling
echo "MDTEST_METADATA: started_at = \$(date '+%H:%M:%S')"
${RUNNER} ${RUNNER_NTASKS_OPT} ${ntasks} ${RUNNER_PPN_OPT} ${ppn} ${MDTEST} ${MDTEST_FLAGS_STR} -n ${items_per_task} -d "\${TARGET_DIR}"
echo "MDTEST_METADATA: finished_at = \$(date '+%H:%M:%S')"
SLURM

  chmod +x "$script"
  echo "  Generated: ${job_name}.sh  (ntasks=${ntasks}, items/task=${items_per_task})"

  if ! $DRY_RUN; then
    sbatch_args=()
    # Chaining: each job waits for previous (afterany) to avoid parallel metadata
    # contention and ensure fair performance comparison across ntasks values
    if $CHAIN && [[ -n "$PREV_JOBID" ]]; then
      sbatch_args+=(--dependency=afterany:"$PREV_JOBID")
    fi
    PREV_JOBID=$(sbatch "${sbatch_args[@]}" "$script" | awk '{print $NF}')
    echo "    Submitted job $PREV_JOBID"
  fi
done

if $INTERACTIVE; then
  echo ""
  echo "# After running the commands above, extract results:"
  echo "#   ./scripts/mdtest-parse.sh $CASE_DIR"
elif $DRY_RUN; then
  echo "(dry-run mode — no jobs submitted)"
else
  echo ""
  echo "Jobs submitted. After completion, extract results:"
  echo "  ./scripts/mdtest-parse.sh $CASE_DIR"
fi
