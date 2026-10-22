#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Helpers
# ============================================================
usage() {
  cat <<'EOF'
Usage:
  ./scripts/ior-run.sh --system=<path> --rule=<path> [options]

Required:
  --system=PATH    Path to system config file (e.g. config/systems/snellius-genoa-scratch.sh)
  --rule=PATH      Path to benchmark rule file (e.g. config/rules/ior-scientific.sh)

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

# Convert size string (e.g. 256GiB, 4g, 1m, 1024) to bytes
size_to_bytes() {
  local val="${1//[^0-9.]}"
  local unit="${1//[0-9.]}"
  case "${unit,,}" in
    *t*) echo "$val * 1024^4"     | bc | cut -d. -f1 ;;  # TiB, T, Tb, etc
    *g*) echo "$val * 1024^3"     | bc | cut -d. -f1 ;;  # GiB, G, Gb, etc
    *m*) echo "$val * 1024^2"     | bc | cut -d. -f1 ;;  # MiB, M, Mb, etc
    *k*) echo "$val * 1024"       | bc | cut -d. -f1 ;;  # KiB, K, Kb, etc
    *)   echo "$val"              | cut -d. -f1 ;;        # Plain bytes
  esac
}

# Format bytes as an IOR size argument with the largest exact unit (e.g. 1474560k, 2g)
bytes_to_ior() {
  local bytes=$1
  if   (( bytes % 1024**3 == 0 )); then echo "$(( bytes / 1024**3 ))g"
  elif (( bytes % 1024**2 == 0 )); then echo "$(( bytes / 1024**2 ))m"
  elif (( bytes % 1024    == 0 )); then echo "$(( bytes / 1024 ))k"
  else echo "$bytes"
  fi
}

# Format bytes to human-readable (TiB, GiB, MiB, KiB, B)
bytes_to_human() {
  local bytes=$1
  if   (( bytes >= 1024**4 )); then echo "$(echo "scale=2; $bytes / 1024^4" | bc) TiB"
  elif (( bytes >= 1024**3 )); then echo "$(echo "scale=2; $bytes / 1024^3" | bc) GiB"
  elif (( bytes >= 1024**2 )); then echo "$(echo "scale=2; $bytes / 1024^2" | bc) MiB"
  elif (( bytes >= 1024    )); then echo "$(echo "scale=2; $bytes / 1024"   | bc) KiB"
  else echo "$bytes B"
  fi
}

# Commit and working-tree state of the IOR source the binary was built from. In an IO500
# build the binary is <io500>/bin/ior and its source tree <io500>/build/ior.
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
# shellcheck source=/dev/null
source "$SYSTEM_FILE"

if [[ -z "${IOR:-}" || ! -x "${IOR}" ]]; then
  echo "ERROR: IOR executable not found or not executable: ${IOR:-}" >&2
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
# shellcheck source=/dev/null
source "$RULE_FILE"

# Validate mandatory rule parameters
for var in AGGREGATE_SIZE BLOCK_SIZE TRANSFER_SIZE SEGMENT_COUNT; do
  [[ -n "${!var:-}" ]] || { echo "ERROR: rule file must define $var" >&2; exit 1; }
done

if ! is_array IOR_FLAGS || [[ ${#IOR_FLAGS[@]} -eq 0 ]]; then
  echo "ERROR: rule file must define IOR_FLAGS as a non-empty array" >&2
  exit 1
fi

# Memory hogging thresholds: prevent warm buffer cache inflation from prior jobs.
# Reserve max(32 GiB, 1 GiB/task) free on each node. At low ppn the cache limit
# dominates; at high ppn the per-task overhead does. Rule files may override.
MAX_CACHE_MIB=${MAX_CACHE_MIB:-$((32 * 1024))}
OVERHEAD_PER_TASK_MIB=${OVERHEAD_PER_TASK_MIB:-1024}

# Split write/read: run the write and read phases as two IOR invocations in the same job.
# The write step only creates the dataset and is not measured, so it runs as fast as
# possible: sequential (no -z). With -z the read step assigns offsets to ranks at random
# (seed IOR_READ_SEED), so each rank mostly reads data written by other nodes; this
# replaces -C, which IOR rejects for -z on a shared file. Rule files may enable it and
# set the seed.
IOR_SPLIT_WR=${IOR_SPLIT_WR:-false}
IOR_READ_SEED=${IOR_READ_SEED:-1}
if [[ "$IOR_SPLIT_WR" != "true" && "$IOR_SPLIT_WR" != "false" ]]; then
  echo "ERROR: IOR_SPLIT_WR must be true or false, got: $IOR_SPLIT_WR" >&2
  exit 1
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

has_flag() { local f; for f in "${IOR_FLAGS[@]}"; do [[ "$f" == "$1" ]] && return 0; done; return 1; }

# IOR refuses random offsets combined with constant task reordering on a shared file
if has_flag "-z" && has_flag "-C" && ! has_flag "-F"; then
  echo "ERROR: IOR rejects -z with -C on a shared file (no -F). Drop -C; for a" >&2
  echo "       cold-cache read set IOR_SPLIT_WR=true (read step uses a different seed)." >&2
  exit 1
fi

# Shell-quote IOR_FLAGS for safe heredoc embedding (protects special chars and spaces)
IOR_FLAGS_STR="$(printf '%q ' "${IOR_FLAGS[@]}")"

# One IOR invocation per step: the full flag set, or a write step and a read step.
# Write step: all flags except -r and -z (sequential), plus -k so the read step finds the file.
# Read step: all flags except -w; it removes the file at the end unless the rule sets -k.
if $IOR_SPLIT_WR; then
  if ! has_flag "-w" || ! has_flag "-r"; then
    echo "ERROR: IOR_SPLIT_WR=true requires both -w and -r in IOR_FLAGS" >&2
    exit 1
  fi
  IOR_WRITE_FLAGS=(); IOR_READ_FLAGS=()
  for f in "${IOR_FLAGS[@]}"; do
    [[ "$f" != "-r" && "$f" != "-z" ]] && IOR_WRITE_FLAGS+=("$f")
    [[ "$f" != "-w" ]] && IOR_READ_FLAGS+=("$f")
  done
  has_flag "-k" || IOR_WRITE_FLAGS+=("-k")
  if has_flag "-z"; then
    IOR_READ_FLAGS+=("--random-offset-seed=${IOR_READ_SEED}")
  fi
  STEP_FLAGS=("$(printf '%q ' "${IOR_WRITE_FLAGS[@]}")" "$(printf '%q ' "${IOR_READ_FLAGS[@]}")")
else
  STEP_FLAGS=("$IOR_FLAGS_STR")
fi
LAST_STEP=$(( ${#STEP_FLAGS[@]} - 1 ))

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

AGG_BYTES=$(size_to_bytes "$AGGREGATE_SIZE")
XFER_BYTES=$(size_to_bytes "$TRANSFER_SIZE")
MAX_BLOCK_BYTES=$(( 2 * 1024**3 ))   # upper bound of one IOR block (-b) per segment
PER_TASK_BYTES=$(( AGG_BYTES / ntasks ))

# Split the per-task volume into equal segments of at most MAX_BLOCK_BYTES, each a
# whole number of transfers. The actual volume stays within one transfer per segment
# per task of the target (IOR requires the block size to be a multiple of -t).
SEGMENT_COUNT_RUN=$(( (PER_TASK_BYTES + MAX_BLOCK_BYTES - 1) / MAX_BLOCK_BYTES ))
(( SEGMENT_COUNT_RUN < 1 )) && SEGMENT_COUNT_RUN=1
block_bytes=$(( PER_TASK_BYTES / SEGMENT_COUNT_RUN / XFER_BYTES * XFER_BYTES ))
(( block_bytes < XFER_BYTES )) && block_bytes=$XFER_BYTES
BLOCK_SIZE_RUN=$(bytes_to_ior "$block_bytes")

# Actual volume that IOR will move (shown instead of the target)
PROCESS_BYTES=$(( block_bytes * SEGMENT_COUNT_RUN ))
global_bytes=$(( PROCESS_BYTES * ntasks ))

# Memory hogging: use IOR's -M flag to fill RAM before the benchmark
MEM_HOG_ARGS=""
if [[ -n "${DRAM_PER_NODE:-}" ]]; then
  NODE_MEM_MIB=$(( DRAM_PER_NODE * 1024 ))
  TASK_OVERHEAD_MIB=$(( ppn * OVERHEAD_PER_TASK_MIB ))
  FREE_MIB=$(( MAX_CACHE_MIB > TASK_OVERHEAD_MIB ? MAX_CACHE_MIB : TASK_OVERHEAD_MIB ))
  if (( FREE_MIB < NODE_MEM_MIB )); then
    MEM_HOG_PCT=$(( (NODE_MEM_MIB - FREE_MIB) * 100 / NODE_MEM_MIB ))
    MEM_HOG_ARGS="-M ${MEM_HOG_PCT}"
  fi
fi

mkdir -p "$CASE_DIR"

# Write metadata: system, rule, task layout and IOR build
cat > "$CASE_DIR/ior.md" <<EOF
# IOR run

cluster = $CLUSTER
partition = $PARTITION
filesystem = $FILESYSTEM
date = $TIMESTAMP
time_limit = $TIME
runner = $RUNNER

## IOR Parameters

ior = $IOR
ior_commit = $(build_commit "$IOR")
ior_source_tree = $(build_tree_state "$IOR")
rule = $RULE_NAME
aggregate_size = $AGGREGATE_SIZE
transfer_size = $TRANSFER_SIZE
extra_args = $IOR_FLAGS_STR
split_wr = $IOR_SPLIT_WR
$($IOR_SPLIT_WR && printf 'write_args = %s\nread_args = %s\n' "${STEP_FLAGS[0]}" "${STEP_FLAGS[1]}")

## Task layout

ntasks = $ntasks
nodes = $nodes
tasks_per_node = $ppn
block_size = $BLOCK_SIZE_RUN
segment_count = $SEGMENT_COUNT_RUN
target_dir = $TARGET_DIR
EOF

info "Case: $CASE_DIR"
info "  ntasks=${ntasks}, total: $AGGREGATE_SIZE, per-task: $(bytes_to_human $PROCESS_BYTES)"
$IOR_SPLIT_WR && info "  Split write/read: sequential write step, then read step (random-offset seed ${IOR_READ_SEED} if -z)"

# IOR command line for step $1 writing to path $2
ior_cmd() {
  echo "${RUNNER} ${RUNNER_NTASKS_OPT} ${ntasks} ${RUNNER_PPN_OPT} ${ppn} ${IOR} -b ${BLOCK_SIZE_RUN} -t ${TRANSFER_SIZE} -s ${SEGMENT_COUNT_RUN} ${STEP_FLAGS[$1]} ${MEM_HOG_ARGS} -o \"$2\""
}

job_name="ior_n${ntasks}"

if $INTERACTIVE; then
  echo ""
  echo "# ntasks=${ntasks}  nodes=${nodes}  ppn=${ppn}  block=${BLOCK_SIZE_RUN}  per-task=$(bytes_to_human $PROCESS_BYTES)  global=$(bytes_to_human $global_bytes)"
  $IOR_SPLIT_WR && echo "# Split write/read: the read step runs only if the write step succeeded"
  echo "mkdir -p \"${TARGET_DIR}/${TIMESTAMP}\""
  for (( s=0; s<=LAST_STEP; s++ )); do
    redir=">"; (( s > 0 )) && redir=">>"
    cont="";   (( s < LAST_STEP )) && cont=" &&"
    echo "$(ior_cmd "$s" "${TARGET_DIR}/${TIMESTAMP}/iorfile") < /dev/null ${redir} \"${CASE_DIR}/${job_name}.out\" 2${redir} \"${CASE_DIR}/${job_name}.err\"${cont}"
  done
  echo ""
  echo "# After running the commands above, the read bandwidth is the read row, Max(MiB) column,"
  echo "# of the last 'Summary of all tests' in: ${CASE_DIR}/${job_name}.out"
  exit 0
fi

ior_lines=""
for (( s=0; s<=LAST_STEP; s++ )); do
  cont=""; (( s < LAST_STEP )) && cont=" &&"
  ior_lines+="$(ior_cmd "$s" "\${TARGET_DIR}/iorfile")${cont}"$'\n'
done

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

# IOR benchmark command
$($IOR_SPLIT_WR && echo "# Split write/read: the read step runs only if the write step succeeded")

TARGET_DIR="${TARGET_DIR}/\${SLURM_JOB_ID}"
mkdir -p "\${TARGET_DIR}"
${ior_lines%$'\n'}
SLURM

chmod +x "$script"
info "  Generated: ${job_name}.sh  (ntasks=${ntasks}, block=${BLOCK_SIZE_RUN}, per-task=$(bytes_to_human $PROCESS_BYTES), global=$(bytes_to_human $global_bytes))"

if $DRY_RUN; then
  info "(dry-run mode — no job submitted)"
  exit 0
fi

JOBID=$(sbatch "$script" | awk '{print $NF}')
info "    Submitted job $JOBID (${job_name})"
info ""
info "Job submitted. After completion, the read bandwidth is the read row, Max(MiB) column,"
info "of the last 'Summary of all tests' in: ${CASE_DIR}/${job_name}_${JOBID}.out"
