#!/usr/bin/env bash
set -euo pipefail

# Extract MDTEST metadata operation rates and write to TSV for plotting
CASE_DIR="${1:?Usage: $0 <case-directory>}"

if [[ ! -d "$CASE_DIR" ]]; then
  echo "ERROR: directory not found: $CASE_DIR" >&2
  exit 1
fi

# Detect scaling mode from run metadata
SCALING_MODE=$(grep '^scaling_mode' "$CASE_DIR/mdtest.md" 2>/dev/null | awk -F' = ' '{print $2}')
SCALING_MODE="${SCALING_MODE:-unknown}"

DAT="$CASE_DIR/mdtest-${SCALING_MODE}.dat"

# Column layout:
#   topology    : ntasks nodes ppn items_per_task total_items total_files avg_file_size
#   dir ops     : dir_create dir_stat dir_rename dir_remove
#   file ops    : file_create file_stat file_read file_remove
#   tree ops    : tree_create tree_remove
#   timing      : elapsed_s
#
# All ops/sec values are Max across iterations (= Mean when iterations=1).
# StdDev > 0 only when run with MDTEST_ITERATIONS > 1.
echo "# ntasks nodes ppn items_per_task total_items total_files avg_file_size dir_create dir_stat dir_rename dir_remove file_create file_stat file_read file_remove tree_create tree_remove elapsed_s" > "$DAT"

# Gather fallback metadata from the case description
CASE_TOTAL_ITEMS=$(awk -F' = ' '/^total_items/ {print $2; exit}' "$CASE_DIR/mdtest.md" 2>/dev/null || true)
CASE_TOTAL_FILES=$(awk -F' = ' '/^total_files/ {print $2; exit}' "$CASE_DIR/mdtest.md" 2>/dev/null || true)
CASE_AVG_FILE_SIZE=$(awk -F' = ' '/^avg_file_size/ {print $2; exit}' "$CASE_DIR/mdtest.md" 2>/dev/null || true)

extract_metadata() {
  local file="$1"
  local pat="$2"
  awk -F' = ' -v pat="$pat" '$0 ~ pat {print $2; exit}' "$file" 2>/dev/null || true
}

extract_job_script_value() {
  local script="$1"
  local pat="$2"
  awk -F'=' -v pat="$pat" '$0 ~ pat {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "$script" 2>/dev/null || true
}

# Extract the Max value for an mdtest SUMMARY operation line.
# SUMMARY format: "   <Op name>    Max    Min    Mean    StdDev"
# First float found on the line is Max.
extract_op() {
  local file="$1"
  local pattern="$2"
  awk -v pat="$pattern" '$0 ~ pat {
    for(i=1;i<=NF;i++) if ($i ~ /^[0-9][0-9]*\.[0-9][0-9]*$/) {print $i; exit}
  }' "$file" | awk 'NR==1 {print; exit}'
}

# Collect matching job output files and fail early if there are none
file_pattern="$CASE_DIR/mdtest_${SCALING_MODE}_n*_*.out"
shopt -s nullglob
out_files=( $file_pattern )
shopt -u nullglob
if (( ${#out_files[@]} == 0 )); then
  echo "WARNING: no mdtest log files found for pattern: $file_pattern" >&2
  echo "Wrote: $DAT (header only)"
  exit 0
fi

for f in "${out_files[@]}"; do

  # Parse ntasks from job name: mdtest_<scaling>_n<N>_<jobid>.out
  base="$(basename "$f" .out)"
  ntasks="${base#mdtest_${SCALING_MODE}_n}"
  ntasks="${ntasks%%_*}"

  # --- Job metadata from MDTEST_METADATA lines emitted by the runner ---
  nodes=$(extract_metadata "$f" '^MDTEST_METADATA: nodes =')
  ppn=$(extract_metadata "$f" '^MDTEST_METADATA: ppn =')
  items_per_task=$(extract_metadata "$f" '^MDTEST_METADATA: items_per_task =')
  total_items=$(extract_metadata "$f" '^MDTEST_METADATA: total_items =')
  total_files=$(extract_metadata "$f" '^MDTEST_METADATA: total_files =')
  avg_file_size=$(extract_metadata "$f" '^MDTEST_METADATA: avg_file_size =')

  # Fallbacks for older logs without MDTEST_METADATA
  script_base="${base%_*}"
  job_script="$CASE_DIR/${script_base}.sh"
  if [[ -z "$nodes" && -f "$job_script" ]]; then
    nodes=$(extract_job_script_value "$job_script" '^#SBATCH --nodes=')
  fi
  if [[ -z "$ppn" && -f "$job_script" ]]; then
    ppn=$(extract_job_script_value "$job_script" '^#SBATCH --ntasks-per-node=')
  fi
  if [[ -z "$items_per_task" ]]; then
    items_per_task=$(awk '/items[[:space:]]*:/ {print $NF; exit}' "$f" 2>/dev/null || true)
  fi
  if [[ -z "$items_per_task" && -f "$job_script" ]]; then
    items_per_task=$(awk '/mdtest/ && /-n/ {for(i=1;i<NF;i++) if($i=="-n") print $(i+1)}' "$job_script" | head -n 1 2>/dev/null || true)
  fi
  if [[ -z "$total_items" ]]; then
    total_items="$CASE_TOTAL_ITEMS"
    if [[ "$SCALING_MODE" == "weak" && -n "$nodes" ]]; then
      total_items=$(( total_items * nodes ))
    fi
  fi
  if [[ -z "$total_files" ]]; then
    total_files=$(awk '/files\/directories/ { for(i=1;i<NF;i++) if ($(i+1)=="files/directories") print $i; exit }' "$f" 2>/dev/null || true)
  fi
  [[ -z "$total_files"    ]] && total_files="${CASE_TOTAL_FILES:-0}"
  [[ -z "$avg_file_size"  ]] && avg_file_size="${CASE_AVG_FILE_SIZE:-}"

  # --- Operation rates (ops/sec, Max across iterations) ---
  dir_create=$(extract_op  "$f" '^[[:space:]]*Directory creation[[:space:]]')
  dir_stat=$(extract_op    "$f" '^[[:space:]]*Directory stat[[:space:]]')
  dir_rename=$(extract_op  "$f" '^[[:space:]]*Directory rename[[:space:]]')
  dir_remove=$(extract_op  "$f" '^[[:space:]]*Directory removal[[:space:]]')
  file_create=$(extract_op "$f" '^[[:space:]]*File creation[[:space:]]')
  file_stat=$(extract_op   "$f" '^[[:space:]]*File stat[[:space:]]')
  file_read=$(extract_op   "$f" '^[[:space:]]*File read[[:space:]]')
  file_remove=$(extract_op "$f" '^[[:space:]]*File removal[[:space:]]')
  tree_create=$(extract_op "$f" '^[[:space:]]*Tree creation[[:space:]]')
  tree_remove=$(extract_op "$f" '^[[:space:]]*Tree removal[[:space:]]')

  # --- Wall-clock elapsed (s) ---
  # Prefer MDTEST_METADATA timestamps (always present from current run scripts).
  # Fall back to mdtest verbose output "-- started at MM/DD/YYYY HH:MM:SS --"
  # for older logs where the metadata lines were not emitted.
  started_t=$(extract_metadata "$f" '^MDTEST_METADATA: started_at =')
  finished_t=$(extract_metadata "$f" '^MDTEST_METADATA: finished_at =')
  [[ -z "$started_t"  ]] && started_t=$(awk '/^-- started at/  {print $5; exit}' "$f")
  [[ -z "$finished_t" ]] && finished_t=$(awk '/^-- finished at/ {print $5; exit}' "$f")
  elapsed_s=$(awk -v t1="$started_t" -v t2="$finished_t" 'BEGIN {
    split(t1,a,":"); split(t2,b,":")
    s1=a[1]*3600+a[2]*60+a[3]; s2=b[1]*3600+b[2]*60+b[3]
    d=s2-s1; if(d<0) d+=86400; printf "%d", d
  }')

  # --- Fallbacks ---
  nodes="${nodes:-0}"; ppn="${ppn:-0}"
  items_per_task="${items_per_task:-0}"; total_items="${total_items:-0}"
  total_files="${total_files:-0}"; avg_file_size="${avg_file_size:-}"
  dir_create="${dir_create:-0}";  dir_stat="${dir_stat:-0}"
  dir_rename="${dir_rename:-0}";  dir_remove="${dir_remove:-0}"
  file_create="${file_create:-0}"; file_stat="${file_stat:-0}"
  file_read="${file_read:-0}";     file_remove="${file_remove:-0}"
  tree_create="${tree_create:-0}"; tree_remove="${tree_remove:-0}"
  elapsed_s="${elapsed_s:-0}"

  printf "%-8s %-5s %-4s %-14s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %s\n" \
    "$ntasks" "$nodes" "$ppn" "$items_per_task" "$total_items" "$total_files" "$avg_file_size" \
    "$dir_create" "$dir_stat" "$dir_rename" "$dir_remove" \
    "$file_create" "$file_stat" "$file_read" "$file_remove" \
    "$tree_create" "$tree_remove" \
    "$elapsed_s" >> "$DAT"
done

# Sort by ntasks ascending (preserve header row at top)
{ head -1 "$DAT"; tail -n +2 "$DAT" | sort -n; } > "$DAT.tmp" && mv "$DAT.tmp" "$DAT"

echo "Wrote: $DAT"
echo ""
echo "Plot with:"
echo "  gnuplot -e \"case_dir='$CASE_DIR'\" ./scripts/mdtest-plot.gpi"
