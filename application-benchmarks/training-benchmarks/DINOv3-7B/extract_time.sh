#!/bin/bash
# Extract training time to target from log file and update timing file
#
# Usage: ./extract_time.sh OUTPUT_DIR CHECKPOINT_ITERATION TOP5_ACCURACY

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 OUTPUT_DIR CHECKPOINT_ITERATION TOP5_ACCURACY"
    exit 1
fi

OUTPUT_DIR="$1"
EVAL="$2"
BEST_TOP5="$3"

LOG_FILE="${OUTPUT_DIR}/logs/log.txt"
TIMING_FILE="${OUTPUT_DIR}/timing.log"

echo "========================================"
echo "Extracting Training Time to Target"
echo "========================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Checkpoint iteration: ${EVAL}"
echo "Top-5 accuracy: ${BEST_TOP5}%"
echo ""

# Check if log file exists
if [ ! -f "${LOG_FILE}" ]; then
    echo "ERROR: Log file not found: ${LOG_FILE}"
    exit 1
fi

# Check if timing file exists
if [ ! -f "${TIMING_FILE}" ]; then
    echo "ERROR: Timing file not found: ${TIMING_FILE}"
    exit 1
fi

echo "Parsing log file for checkpoint save time..."

# Look for the checkpoint save completion in the log (searching for "Saved:" with iteration number)
# The log format is: I<YYYYMMDD HH:MM:SS> <pid> dinov3 ...
CHECKPOINT_LOG_ENTRY=$(grep -E "Saved:.*training_${EVAL}/sharded_teacher_checkpoint" "${LOG_FILE}" | tail -1)

if [ -z "${CHECKPOINT_LOG_ENTRY}" ]; then
    echo "WARNING: Could not find checkpoint save entry in log for iteration ${EVAL}"
    exit 1
fi

# Extract timestamp from log entry format: I20260422 05:23:05 or W20260527 17:21:07
CHECKPOINT_TIMESTAMP=$(echo "${CHECKPOINT_LOG_ENTRY}" | grep -oP '^[IWE]\K[0-9]{8} [0-9]{2}:[0-9]{2}:[0-9]{2}')

if [ -z "${CHECKPOINT_TIMESTAMP}" ]; then
    echo "WARNING: Could not extract timestamp from log entry"
    exit 1
fi

# Convert to unix timestamp: YYYYMMDD HH:MM:SS -> epoch
CHECKPOINT_DATE=$(echo "${CHECKPOINT_TIMESTAMP}" | cut -d' ' -f1)
CHECKPOINT_TIME=$(echo "${CHECKPOINT_TIMESTAMP}" | cut -d' ' -f2)

# Format: YYYYMMDD -> YYYY-MM-DD
FORMATTED_DATE="${CHECKPOINT_DATE:0:4}-${CHECKPOINT_DATE:4:2}-${CHECKPOINT_DATE:6:2}"
CHECKPOINT_EPOCH=$(date -d "${FORMATTED_DATE} ${CHECKPOINT_TIME}" +%s 2>/dev/null || echo "")

if [ -z "${CHECKPOINT_EPOCH}" ]; then
    echo "WARNING: Could not convert checkpoint timestamp to epoch"
    exit 1
fi

# Convert to 12-hour format with AM/PM to match record_time_and_run.sh format
CHECKPOINT_TIME_HUMAN=$(date -d "${FORMATTED_DATE} ${CHECKPOINT_TIME}" +"%Y-%m-%d %r" 2>/dev/null || echo "${FORMATTED_DATE} ${CHECKPOINT_TIME}")

echo "Checkpoint saved at: ${CHECKPOINT_TIME_HUMAN} (epoch: ${CHECKPOINT_EPOCH})"

# Read start time from timing file
START_TIMESTAMP=$(grep "^Start timestamp:" "${TIMING_FILE}" | awk '{print $3}')

if [ -z "${START_TIMESTAMP}" ]; then
    echo "WARNING: Could not read start timestamp from timing file"
    exit 1
fi

# Calculate duration in seconds and minutes
DURATION_SECONDS=$((CHECKPOINT_EPOCH - START_TIMESTAMP))
DURATION_MINUTES=$(echo "scale=2; ${DURATION_SECONDS} / 60" | bc)

HOURS=$((DURATION_SECONDS / 3600))
MINUTES=$(((DURATION_SECONDS % 3600) / 60))
SECONDS=$((DURATION_SECONDS % 60))

START_TIME_HUMAN=$(grep '^Start time:' "${TIMING_FILE}" | cut -d':' -f2- | xargs)

echo ""
echo "========================================"
echo "TRAINING TIME TO TARGET"
echo "========================================"
echo "Start time: ${START_TIME_HUMAN}"
echo "Checkpoint save time: ${CHECKPOINT_TIME_HUMAN}"
echo "Target accuracy: ${BEST_TOP5}% (Top-5)"
echo "Checkpoint iteration: ${EVAL}"
echo ""
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Total duration: ${DURATION_MINUTES} minutes"
echo "========================================"

# Append to timing file
echo "" >> "${TIMING_FILE}"
echo "Target Reached:" >> "${TIMING_FILE}"
echo "==============" >> "${TIMING_FILE}"
echo "Checkpoint iteration: ${EVAL}" >> "${TIMING_FILE}"
echo "Checkpoint save time: ${CHECKPOINT_TIME_HUMAN}" >> "${TIMING_FILE}"
echo "Checkpoint save timestamp: ${CHECKPOINT_EPOCH}" >> "${TIMING_FILE}"
echo "Target Top-5 accuracy: ${BEST_TOP5}%" >> "${TIMING_FILE}"
echo "" >> "${TIMING_FILE}"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "${TIMING_FILE}"
echo "Duration (seconds): ${DURATION_SECONDS}" >> "${TIMING_FILE}"
echo "Duration (minutes): ${DURATION_MINUTES}" >> "${TIMING_FILE}"

echo ""
echo "Results saved to: ${TIMING_FILE}"
