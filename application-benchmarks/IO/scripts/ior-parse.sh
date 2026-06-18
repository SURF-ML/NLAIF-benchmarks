#!/usr/bin/env bash
set -euo pipefail

# Extract IOR results and write to TSV for plotting
CASE_DIR="${1:?Usage: $0 <case-directory>}"

if [[ ! -d "$CASE_DIR" ]]; then
  echo "ERROR: directory not found: $CASE_DIR" >&2
  exit 1
fi

# Detect scaling mode from run metadata
SCALING_MODE=$(grep '^scaling_mode' "$CASE_DIR/ior.md" 2>/dev/null | awk -F' = ' '{print $2}')
SCALING_MODE="${SCALING_MODE:-unknown}"

DAT="$CASE_DIR/ior-${SCALING_MODE}.dat"

# Column layout:
#   topology    : ntasks nodes ppn
#   bandwidth   : write_MiB_s read_MiB_s
#   IOPS        : write_IOPS read_IOPS
#   latency     : write_lat_s read_lat_s
#   write phases: w_open_s w_wrrd_s w_close_s
#   read phases : r_open_s r_rdrd_s r_close_s
#   timing      : elapsed_s
#   filesystem  : fs_size_TiB fs_used_pct inode_used_pct
echo "# ntasks nodes ppn write_MiB_s read_MiB_s write_IOPS read_IOPS write_lat_s read_lat_s w_open_s w_wrrd_s w_close_s r_open_s r_rdrd_s r_close_s elapsed_s fs_size_TiB fs_used_pct inode_used_pct" > "$DAT"

# Collect matching job output files and fail early if there are none
file_pattern="$CASE_DIR/ior_${SCALING_MODE}_n*_*.out"
shopt -s nullglob
out_files=( $file_pattern )
shopt -u nullglob
if (( ${#out_files[@]} == 0 )); then
  echo "WARNING: no IOR log files found for pattern: $file_pattern" >&2
  echo "Wrote: $DAT (header only)"
  exit 0
fi

for f in "${out_files[@]}"; do

  # Parse ntasks from job name: ior_<scaling>_n<N>_<jobid>.out
  base="$(basename "$f" .out)"
  ntasks="${base#ior_${SCALING_MODE}_n}"
  ntasks="${ntasks%%_*}"

  # --- Bandwidth (MiB/s) ---
  # Primary: results table lines ("write  <bw> ...")
  # Fallback: older IOR "Max Write: X MiB/sec" format
  write_bw=$(awk '/^write[[:space:]]/ {print $2; exit}' "$f")
  read_bw=$(awk '/^read[[:space:]]/  {print $2; exit}' "$f")
  [[ -z "$write_bw" ]] && write_bw=$(awk '/^Max Write:/ {print $3; exit}' "$f")
  [[ -z "$read_bw"  ]] && read_bw=$(awk '/^Max Read:/  {print $3; exit}' "$f")

  # --- IOPS ---
  write_iops=$(awk '/^write[[:space:]]/ {print $3; exit}' "$f")
  read_iops=$(awk '/^read[[:space:]]/  {print $3; exit}' "$f")

  # --- Latency (s) ---
  write_lat=$(awk '/^write[[:space:]]/ {print $4; exit}' "$f")
  read_lat=$(awk '/^read[[:space:]]/  {print $4; exit}' "$f")

  # --- Phase timing ---
  # Results table: bw IOPS lat block xfer open wr/rd close total iter
  #                $2  $3   $4  $5    $6   $7   $8    $9    $10   $11
  w_open_s=$(awk '/^write[[:space:]]/ {print $7; exit}' "$f")
  w_wrrd_s=$(awk '/^write[[:space:]]/ {print $8; exit}' "$f")
  w_close_s=$(awk '/^write[[:space:]]/ {print $9; exit}' "$f")
  r_open_s=$(awk '/^read[[:space:]]/  {print $7; exit}' "$f")
  r_rdrd_s=$(awk '/^read[[:space:]]/  {print $8; exit}' "$f")
  r_close_s=$(awk '/^read[[:space:]]/  {print $9; exit}' "$f")

  # --- Topology ---
  nodes=$(awk '/^nodes[[:space:]]*:/       {print $NF; exit}' "$f")
  ppn=$(awk '/^clients per node[[:space:]]*:/ {print $NF; exit}' "$f")

  # --- Filesystem state at job start ---
  # Line format: "FS : X TiB   Used FS: Y%   Inodes: Z Mi   Used Inodes: W%"
  fs_line=$(grep '^FS ' "$f" | head -1)
  fs_size=$(echo "$fs_line" | awk '{print $3}')
  fs_used=$(echo "$fs_line" | grep -oE 'Used FS: [0-9.]+' | awk '{print $NF}')
  inode_used=$(echo "$fs_line" | grep -oE 'Used Inodes: [0-9.]+' | awk '{print $NF}')

  # --- Wall-clock elapsed (s) ---
  # "Began    : Day Mon DD HH:MM:SS YYYY"  →  field 6 is the time
  began_t=$(awk '/^Began[[:space:]]*:/    {print $6; exit}' "$f")
  finished_t=$(awk '/^Finished[[:space:]]*:/ {print $6; exit}' "$f")
  elapsed_s=$(awk -v t1="$began_t" -v t2="$finished_t" 'BEGIN {
    split(t1,a,":"); split(t2,b,":")
    s1=a[1]*3600+a[2]*60+a[3]; s2=b[1]*3600+b[2]*60+b[3]
    d=s2-s1; if(d<0) d+=86400; printf "%d", d
  }')

  # --- Fallbacks ---
  # Write-phase fields use NA (not 0) when absent so that read-only rules
  # are unambiguous in the output — gnuplot treats NA as missing data.
  write_bw="${write_bw:-NA}";   read_bw="${read_bw:-NA}"
  write_iops="${write_iops:-NA}"; read_iops="${read_iops:-NA}"
  write_lat="${write_lat:-NA}";   read_lat="${read_lat:-NA}"
  w_open_s="${w_open_s:-NA}";  w_wrrd_s="${w_wrrd_s:-NA}";  w_close_s="${w_close_s:-NA}"
  r_open_s="${r_open_s:-NA}";  r_rdrd_s="${r_rdrd_s:-NA}";  r_close_s="${r_close_s:-NA}"
  elapsed_s="${elapsed_s:-0}"
  nodes="${nodes:-0}"; ppn="${ppn:-0}"
  fs_size="${fs_size:-0}"; fs_used="${fs_used:-0}"; inode_used="${inode_used:-0}"

  printf "%-8s %-5s %-4s %-12s %-12s %-12s %-12s %-12s %-12s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-12s %-12s %s\n" \
    "$ntasks" "$nodes" "$ppn" \
    "$write_bw" "$read_bw" \
    "$write_iops" "$read_iops" \
    "$write_lat" "$read_lat" \
    "$w_open_s" "$w_wrrd_s" "$w_close_s" \
    "$r_open_s" "$r_rdrd_s" "$r_close_s" \
    "$elapsed_s" "$fs_size" "$fs_used" "$inode_used" >> "$DAT"
done

# Sort by ntasks ascending (preserve header row at top)
{ head -1 "$DAT"; tail -n +2 "$DAT" | sort -n; } > "$DAT.tmp" && mv "$DAT.tmp" "$DAT"

echo "Wrote: $DAT"
echo ""
echo "Plot with:"
echo "  gnuplot -e \"case_dir='$CASE_DIR'\" ./scripts/ior-plot.gpi"
