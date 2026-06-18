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
  --system=PATH    Path to system config file (e.g. config/systems/snellius-rome-scratch.sh)
  --rule=PATH      Path to benchmark rule file (e.g. config/rules/ior-llama-405b.sh)

Scaling options:
  --scaling=TYPE   single (default), weak, or strong
  --ntasks=LIST    Comma or space-separated task counts
                   (default: CPUS_PER_NODE for single; 1,2,4,8,16 nodes for weak/strong)
  --ppn=N          Tasks per node (default: CPUS_PER_NODE from system file)

Optional:
  --randomize      Add -z (random offsets) to IOR flags; results go to <rule>-randomize/
  --interactive    Print commands to stdout instead of generating SLURM scripts
  --dry-run        Generate scripts without submitting
  --chain          Submit jobs sequentially (each depends on the previous)
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
for var in BLOCK_SIZE TRANSFER_SIZE SEGMENT_COUNT; do
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

# Validate AGGREGATE_SIZE required for all modes (primary size parameter)
if [[ -z "$AGGREGATE_SIZE" ]]; then
  echo "ERROR: rule file must define AGGREGATE_SIZE" >&2
  exit 1
fi

# Resolve defaults and validate for each scaling mode
case "${SCALING,,}" in
  weak)
    # Per-node (per-task×ppn) constant; aggregate grows with nodes
    # AGGREGATE_SIZE = per-node aggregate (fixed regardless of ppn or ntasks)
    if [[ -z "$NTASKS_LIST" ]]; then
      NTASKS_ARR=()
      for n in 1 2 4 8 16; do NTASKS_ARR+=( $(( n * TASKS_PER_NODE )) ); done
    fi
    echo "Scaling mode: WEAK (constant per-node: $AGGREGATE_SIZE, aggregate grows with nodes)"
    ;;
  strong)
    # Per-node constant; total aggregate stays constant (AGGREGATE_SIZE)
    # Distributes across increasing nodes as ntasks grows
    if [[ -z "$NTASKS_LIST" ]]; then
      NTASKS_ARR=()
      for n in 1 2 4 8 16; do NTASKS_ARR+=( $(( n * TASKS_PER_NODE )) ); done
    fi
    echo "Scaling mode: STRONG (constant total: $AGGREGATE_SIZE, per-node shrinks with ntasks)"
    ;;
  single)
    # Total aggregate = AGGREGATE_SIZE for that specific ntasks value
    if [[ -z "$NTASKS_LIST" ]]; then
      NTASKS_ARR=( "$TASKS_PER_NODE" )
    fi
    if (( ${#NTASKS_ARR[@]} != 1 )); then
      echo "ERROR: --scaling=single expects a single ntasks value, got: $NTASKS_LIST" >&2
      exit 1
    fi
    echo "Scaling mode: SINGLE (ntasks=${NTASKS_ARR[0]}, total: $AGGREGATE_SIZE)"
    ;;
  *)
    echo "ERROR: unknown scaling type: $SCALING. Use weak, strong, or single." >&2
    exit 1
    ;;
esac

NTASKS_LIST="${NTASKS_ARR[*]}"

# Inject random-offset flag when --randomize is set
if $RANDOMIZE; then
  IOR_FLAGS+=("-z")
fi

# Shell-quote IOR_FLAGS for safe heredoc embedding (protects special chars and spaces)
IOR_FLAGS_STR="$(printf '%q ' "${IOR_FLAGS[@]}")"

# Results path: results/<rule>[-randomize]/<cluster>-<partition>-<filesystem>/<timestamp>/
RULE_NAME=$(basename "$RULE_FILE" .sh)
$RANDOMIZE && RULE_NAME="${RULE_NAME}-randomize"
TARGET_DIR="${TARGET_DIR}/${RULE_NAME}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
CASE_DIR="$REPO_ROOT/results/$RULE_NAME/$CLUSTER-$PARTITION-$FILESYSTEM/$TIMESTAMP"

AGG_BYTES=$(size_to_bytes "$AGGREGATE_SIZE")

# Compute display labels: aggregate and per-node for first ntasks value
FIRST_NTASKS=${NTASKS_ARR[0]:-1}
FIRST_NODES=$(( (FIRST_NTASKS + TASKS_PER_NODE - 1) / TASKS_PER_NODE ))

case "${SCALING,,}" in
  strong)
    # Strong: total stays constant, per-node shrinks as nodes increase
    PER_NODE_BYTES=$(( AGG_BYTES / FIRST_NODES ))
    PER_TASK_BYTES=$(( PER_NODE_BYTES / TASKS_PER_NODE ))
    SCALING_DESC="Strong scaling (total: $AGGREGATE_SIZE → per-node: $(bytes_to_human $PER_NODE_BYTES) @ ${NTASKS_ARR[0]} tasks)"
    ;;
  weak)
    # Weak: per-node stays constant, total grows with nodes
    PER_NODE_BYTES=$AGG_BYTES
    PER_TASK_BYTES=$(( PER_NODE_BYTES / TASKS_PER_NODE ))
    SCALING_DESC="Weak scaling (per-node: $AGGREGATE_SIZE → total grows, @ ${NTASKS_ARR[0]} tasks: $(bytes_to_human $(( PER_NODE_BYTES * FIRST_NODES ))))"
    ;;
  single)
    # Single: total = AGGREGATE_SIZE
    PER_NODE_BYTES=$(( AGG_BYTES / FIRST_NODES ))
    PER_TASK_BYTES=$(( PER_NODE_BYTES / TASKS_PER_NODE ))
    SCALING_DESC="Single run (ntasks=${NTASKS_ARR[0]}, total: $AGGREGATE_SIZE, per-task: $(bytes_to_human $PER_TASK_BYTES))"
    ;;
esac

mkdir -p "$CASE_DIR"

# Write metadata: system config, benchmark parameters, and scaling setup
cat > "$CASE_DIR/ior.md" <<EOF
# IOR ${SCALING^} Scaling

cluster = $CLUSTER
partition = $PARTITION
filesystem = $FILESYSTEM
date = $TIMESTAMP
time_limit = $TIME
runner = $RUNNER
scaling_mode = $SCALING

## IOR Parameters

ior = $IOR
block_size = $BLOCK_SIZE
transfer_size = $TRANSFER_SIZE
segment_count = $SEGMENT_COUNT
randomize = $RANDOMIZE
extra_args = $IOR_FLAGS_STR

## Scaling

scaling_type = $SCALING
tasks_per_node = $TASKS_PER_NODE
ntasks_list = $NTASKS_LIST
target_dir = $TARGET_DIR
EOF

echo "Case: $CASE_DIR"
echo "  $SCALING_DESC"

# Generate one SLURM script per ntasks value
PREV_JOBID=""
for ntasks in $NTASKS_LIST; do
  nodes=$(( (ntasks + TASKS_PER_NODE - 1) / TASKS_PER_NODE ))
  ppn=$(( ntasks < TASKS_PER_NODE ? ntasks : TASKS_PER_NODE ))

  # Compute per-task bytes based on scaling mode and AGGREGATE_SIZE
  case "${SCALING,,}" in
    strong)
      # Strong: total stays constant, per-node shrinks as nodes increase
      PER_NODE_BYTES=$(( AGG_BYTES / nodes ))
      PER_TASK_BYTES=$(( PER_NODE_BYTES / ppn ))
      global_bytes=$AGG_BYTES
      ;;
    weak)
      # Weak: per-node constant, total grows with nodes
      PER_NODE_BYTES=$AGG_BYTES
      PER_TASK_BYTES=$(( PER_NODE_BYTES / ppn ))
      global_bytes=$(( PER_NODE_BYTES * nodes ))
      ;;
    single)
      # Single: total = AGGREGATE_SIZE, distributed across nodes and tasks
      PER_NODE_BYTES=$(( AGG_BYTES / nodes ))
      PER_TASK_BYTES=$(( PER_NODE_BYTES / ppn ))
      global_bytes=$AGG_BYTES
      ;;
  esac

  # Scale block size and segment count to achieve per-task target
  # Try to keep block size in a reasonable range (64m - 2g)
  if (( PER_TASK_BYTES >= 2*1024**3 )); then
    BLOCK_SIZE_RUN="2g"
    SEGMENT_COUNT_RUN=$(( PER_TASK_BYTES / (2*1024**3) ))
  elif (( PER_TASK_BYTES >= 1024**3 )); then
    BLOCK_SIZE_RUN="1g"
    SEGMENT_COUNT_RUN=$(( PER_TASK_BYTES / (1024**3) ))
  elif (( PER_TASK_BYTES >= 256*1024**2 )); then
    BLOCK_SIZE_RUN="256m"
    SEGMENT_COUNT_RUN=$(( PER_TASK_BYTES / (256*1024**2) ))
  elif (( PER_TASK_BYTES >= 128*1024**2 )); then
    BLOCK_SIZE_RUN="128m"
    SEGMENT_COUNT_RUN=$(( PER_TASK_BYTES / (128*1024**2) ))
  else
    BLOCK_SIZE_RUN="64m"
    SEGMENT_COUNT_RUN=$(( PER_TASK_BYTES / (64*1024**2) ))
  fi

  # Ensure at least 1 segment
  (( SEGMENT_COUNT_RUN < 1 )) && SEGMENT_COUNT_RUN=1

  PROCESS_BYTES=$PER_TASK_BYTES

  job_name="ior_${SCALING}_n${ntasks}"

  if $INTERACTIVE; then
    echo ""
    echo "# ntasks=${ntasks}  nodes=${nodes}  ppn=${ppn}  block=${BLOCK_SIZE_RUN}  per-task=$(bytes_to_human $PROCESS_BYTES)  global=$(bytes_to_human $global_bytes)"
    echo "mkdir -p \"${TARGET_DIR}/${TIMESTAMP}\""
    echo "${RUNNER} ${RUNNER_NTASKS_OPT} ${ntasks} ${RUNNER_PPN_OPT} ${ppn} ${IOR} -b ${BLOCK_SIZE_RUN} -t ${TRANSFER_SIZE} -s ${SEGMENT_COUNT_RUN} ${IOR_FLAGS_STR} -o \"${TARGET_DIR}/${TIMESTAMP}/iorfile\" \\"
    echo "  > \"${CASE_DIR}/${job_name}.out\" 2> \"${CASE_DIR}/${job_name}.err\""
    continue
  fi
  script="$CASE_DIR/${job_name}.sh"

  # Memory hogging: use IOR's -M flag to fill RAM before the benchmark, preventing
  # false high throughput from warm buffer cache (kept by OS from prior jobs on shared node).
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

# IOR benchmark command with ${SCALING} scaling

TARGET_DIR="${TARGET_DIR}/\${SLURM_JOB_ID}"
mkdir -p "\${TARGET_DIR}"
${RUNNER} ${RUNNER_NTASKS_OPT} ${ntasks} ${RUNNER_PPN_OPT} ${ppn} ${IOR} -b ${BLOCK_SIZE_RUN} -t ${TRANSFER_SIZE} -s ${SEGMENT_COUNT_RUN} ${IOR_FLAGS_STR} ${MEM_HOG_ARGS} -o "\${TARGET_DIR}/iorfile"
SLURM

  chmod +x "$script"
  echo "  Generated: ${job_name}.sh  (ntasks=${ntasks}, block=${BLOCK_SIZE_RUN}, per-task=$(bytes_to_human $PROCESS_BYTES), global=$(bytes_to_human $global_bytes))"

  if ! $DRY_RUN; then
    sbatch_args=()
    # Chaining: each job waits for previous (afterany) to avoid parallel I/O
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
  echo "#   ./scripts/ior-parse.sh $CASE_DIR"
elif $DRY_RUN; then
  echo "(dry-run mode — no jobs submitted)"
else
  echo ""
  echo "Jobs submitted. After completion, extract results:"
  echo "  ./scripts/ior-parse.sh $CASE_DIR"
fi
