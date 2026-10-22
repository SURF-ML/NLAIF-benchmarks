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
  --system=PATH    Path to system config file (e.g. config/systems/snellius-genoa-scratch.sh)
  --rule=PATH      Path to benchmark rule file (e.g. config/rules/mdtest-lvd-1689m-shallow.sh)

Options:
  --ntasks=N       Total number of processes (default: CPUS_PER_NODE from system file)
  --ppn=N          Processes per node (default: CPUS_PER_NODE from system file)
  --interactive    Print commands to stdout instead of generating a SLURM script.
                   All status messages go to stderr so the command stream can be piped
                   directly to bash.
  --dry-run        Generate the SLURM script without submitting it
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

# Commit and working-tree state of the source the binary was built from. In an IO500
# build mdtest is <io500>/bin/mdtest, built from the IOR source tree <io500>/build/ior.
build_commit() {
  local src
  src="$(cd "$(dirname "$1")/.." 2>/dev/null && pwd)/build/ior"
  git -C "$src" rev-parse HEAD 2>/dev/null || echo "unknown"
}
build_tree_state() {
  local src
  src="$(cd "$(dirname "$1")/.." 2>/dev/null && pwd)/build/ior"
  if ! git -C "$src" rev-parse HEAD >/dev/null 2>&1; then echo "unknown"
  elif [[ -z "$(git -C "$src" status --porcelain 2>/dev/null)" ]]; then echo "clean"
  else echo "modified"
  fi
}

# ============================================================
#  Parse arguments
# ============================================================
SYSTEM_FILE=""
RULE_FILE=""
DRY_RUN=false
INTERACTIVE=false
NTASKS=""          # default: CPUS_PER_NODE from system file
PPN=""             # default: CPUS_PER_NODE from system file

for arg in "$@"; do
  case "$arg" in
    --system=*) SYSTEM_FILE="${arg#--system=}" ;;
    --rule=*)   RULE_FILE="${arg#--rule=}" ;;
    --ntasks=*) NTASKS="${arg#--ntasks=}" ;;
    --ppn=*)    PPN="${arg#--ppn=}" ;;
    --interactive)  INTERACTIVE=true ;;
    --dry-run)      DRY_RUN=true ;;
    -h|--help)  usage; exit 0 ;;
    *)          echo "Unknown argument: $arg" >&2; usage; exit 1 ;;
  esac
done

# In interactive mode all status output goes to stderr so the command
# stream on stdout can be piped cleanly to bash.
info() { if $INTERACTIVE; then echo "$@" >&2; else echo "$@"; fi; }

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
# System file contract: CLUSTER, PARTITION, TIME, RUNNER, RUNNER_NTASKS_OPT, RUNNER_PPN_OPT,
# FEATURES, CPUS_PER_NODE, DRAM_PER_NODE, ENV_SETUP, IOR, MDTEST, FILESYSTEM, TARGET_DIR
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
# MDTEST_FLAGS (all flags except -n and -d, which the runner sets)
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

# Directories that hold files: the leaves with -L, otherwise every directory of the tree
if printf '%s\n' "${MDTEST_FLAGS[@]}" | grep -qx -- "-L"; then
  FILE_DIRS=$(( MDTEST_BRANCH ** MDTEST_DEPTH ))
else
  FILE_DIRS=$(tree_node_count "$MDTEST_BRANCH" "$MDTEST_DEPTH")
fi

# Determine tasks per node: --ppn flag > system CPUS_PER_NODE > fallback 1
TASKS_PER_NODE="${PPN}"
if [[ -z "$TASKS_PER_NODE" ]]; then
  TASKS_PER_NODE=${CPUS_PER_NODE:-1}
fi

# Guard against requesting more tasks per node than CPUs available
if [[ -n "${CPUS_PER_NODE:-}" ]] && (( TASKS_PER_NODE > CPUS_PER_NODE )); then
  echo "WARNING: TASKS_PER_NODE ($TASKS_PER_NODE) > CPUS_PER_NODE ($CPUS_PER_NODE). Adjusting to $CPUS_PER_NODE." >&2
  TASKS_PER_NODE=$CPUS_PER_NODE
fi

# Total processes: --ntasks > one full node
ntasks="${NTASKS:-$TASKS_PER_NODE}"
if ! [[ "$ntasks" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --ntasks must be a single positive integer, got: $ntasks" >&2
  exit 1
fi

# Shell-quote MDTEST_FLAGS for safe heredoc embedding (protects special chars and spaces)
MDTEST_FLAGS_STR="$(printf '%q ' "${MDTEST_FLAGS[@]}")"

# Results path: results/<rule>/<cluster>-<partition>-<filesystem>/<timestamp>/
RULE_NAME=$(basename "$RULE_FILE" .sh)
TARGET_DIR="${TARGET_DIR}/${RULE_NAME}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
CASE_DIR="$REPO_ROOT/results/$RULE_NAME/$CLUSTER-$PARTITION-$FILESYSTEM/$TIMESTAMP"

# ============================================================
#  Task layout and per-process share of the dataset
# ============================================================
nodes=$(( (ntasks + TASKS_PER_NODE - 1) / TASKS_PER_NODE ))
ppn=$(( ntasks < TASKS_PER_NODE ? ntasks : TASKS_PER_NODE ))
# mdtest puts the same whole number of each process's files in every file directory, and
# creates nothing for a process with fewer files than directories. Choose that number, k,
# as the smallest whole number for which the total reaches MDTEST_TOTAL_ITEMS, and pass
# k x FILE_DIRS files per process: the data set is never smaller than the target. With more
# processes than MDTEST_TOTAL_ITEMS / FILE_DIRS, k = 1 and the data set grows with the processes.
files_per_dir=$(( (MDTEST_TOTAL_ITEMS + ntasks * FILE_DIRS - 1) / (ntasks * FILE_DIRS) ))
(( files_per_dir < 1 )) && files_per_dir=1
items_per_task=$(( files_per_dir * FILE_DIRS ))
total_items=$(( items_per_task * ntasks ))
deviation_pct=$(awk -v t="$total_items" -v g="$MDTEST_TOTAL_ITEMS" 'BEGIN { printf "%+.1f", (t - g) * 100 / g }')

mkdir -p "$CASE_DIR"

# Write metadata: system, rule, task layout and mdtest build
cat > "$CASE_DIR/mdtest.md" <<EOF
# MDTEST run

cluster = $CLUSTER
partition = $PARTITION
filesystem = $FILESYSTEM
date = $TIMESTAMP
time_limit = $TIME
runner = $RUNNER

## MDTEST Parameters

mdtest = $MDTEST
mdtest_commit = $(build_commit "$MDTEST")
mdtest_source_tree = $(build_tree_state "$MDTEST")
rule = $RULE_NAME
branch = $MDTEST_BRANCH
depth = $MDTEST_DEPTH
target_items = $MDTEST_TOTAL_ITEMS
file_dirs = $FILE_DIRS
flags = $MDTEST_FLAGS_STR
total_files = ${TOTAL_FILES:-}
avg_file_size = ${AVG_FILE_SIZE:-}

## Task layout

ntasks = $ntasks
nodes = $nodes
tasks_per_node = $ppn
files_per_dir_per_task = $files_per_dir
items_per_task = $items_per_task
total_items = $total_items
target_dir = $TARGET_DIR
EOF

info "Case: $CASE_DIR"
info "  branch=$MDTEST_BRANCH  depth=$MDTEST_DEPTH  total_items=$MDTEST_TOTAL_ITEMS  iterations=$MDTEST_ITERATIONS"

info "  files: ${ntasks} processes x ${files_per_dir} per directory x ${FILE_DIRS} directories = ${total_items} (target ${MDTEST_TOTAL_ITEMS}, ${deviation_pct}%)"
info ""

job_name="mdtest_n${ntasks}"
metadata=( "ntasks = ${ntasks}" "nodes = ${nodes}" "ppn = ${ppn}" \
           "items_per_task = ${items_per_task}" "total_items = ${total_items}" \
           "total_files = ${TOTAL_FILES:-}" "avg_file_size = ${AVG_FILE_SIZE:-}" )

if $INTERACTIVE; then
  echo ""
  echo "# ntasks=${ntasks}  nodes=${nodes}  ppn=${ppn}  items/task=${items_per_task}  total=${total_items}"
  echo "mkdir -p \"${TARGET_DIR}/${TIMESTAMP}\""
  # Same MDTEST_METADATA lines as the SLURM script, so the output records the task layout
  mdtest_cmd="${RUNNER} ${RUNNER_NTASKS_OPT} ${ntasks} ${RUNNER_PPN_OPT} ${ppn} ${MDTEST} ${MDTEST_FLAGS_STR} -n ${items_per_task} -d \"${TARGET_DIR}/${TIMESTAMP}\""
  echo "{"
  for meta in "${metadata[@]}"; do
    echo "  echo \"MDTEST_METADATA: ${meta}\""
  done
  echo "  echo \"MDTEST_METADATA: started_at = \$(date '+%Y-%m-%d %H:%M:%S')\""
  echo "  ${mdtest_cmd}"
  echo "  echo \"MDTEST_METADATA: finished_at = \$(date '+%Y-%m-%d %H:%M:%S')\""
  echo "} < /dev/null > \"${CASE_DIR}/${job_name}.out\" 2> \"${CASE_DIR}/${job_name}.err\""
  echo ""
  echo "# After running the commands above, the file read rate is the File read row, Max column,"
  echo "# of the SUMMARY rate table in: ${CASE_DIR}/${job_name}.out"
  exit 0
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

$(for meta in "${metadata[@]}"; do echo "echo \"MDTEST_METADATA: ${meta}\""; done)

echo "MDTEST_COMMAND: ${RUNNER} ${RUNNER_NTASKS_OPT} ${ntasks} ${RUNNER_PPN_OPT} ${ppn} ${MDTEST} ${MDTEST_FLAGS_STR} -n ${items_per_task} -d \"\${TARGET_DIR}\""

# MDTEST benchmark command
echo "MDTEST_METADATA: started_at = \$(date '+%Y-%m-%d %H:%M:%S')"
${RUNNER} ${RUNNER_NTASKS_OPT} ${ntasks} ${RUNNER_PPN_OPT} ${ppn} ${MDTEST} ${MDTEST_FLAGS_STR} -n ${items_per_task} -d "\${TARGET_DIR}"
echo "MDTEST_METADATA: finished_at = \$(date '+%Y-%m-%d %H:%M:%S')"
SLURM

chmod +x "$script"
info "  Generated: ${job_name}.sh  (ntasks=${ntasks}, items/task=${items_per_task})"

if $DRY_RUN; then
  info "(dry-run mode — no job submitted)"
  exit 0
fi

JOBID=$(sbatch "$script" | awk '{print $NF}')
info "    Submitted job $JOBID (${job_name})"
info ""
info "Job submitted. After completion, the file read rate is the File read row, Max column,"
info "of the SUMMARY rate table in: ${CASE_DIR}/${job_name}_${JOBID}.out"
