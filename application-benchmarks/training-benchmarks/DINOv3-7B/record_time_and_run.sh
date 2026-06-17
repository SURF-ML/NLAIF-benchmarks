#!/bin/bash

echo "============================================"
echo "DINOv3 ViT-7B/16 ImageNet Pretraining"
echo "============================================"
echo "Config file: ${CONFIG_FILE}"
echo "Dataset: ${DATA_ARGS}"
echo "Output Path:   ${OUTPUT_DIR}"
echo "Seed:   ${SEED}"
echo "============================================"

# Record start time
start_time=$(date +%s)
start_time_human=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_time_human"

# Create timing file
mkdir -p "${OUTPUT_DIR}"
TIMING_FILE="${OUTPUT_DIR}/timing.log"
echo "Training Timing Information" > "${TIMING_FILE}"
echo "===========================" >> "${TIMING_FILE}"
echo "Start time: $start_time_human" >> "${TIMING_FILE}"
echo "Start timestamp: $start_time" >> "${TIMING_FILE}"
echo "" >> "${TIMING_FILE}"

srun \
    --ntasks-per-node=${SLURM_NTASKS_PER_NODE} \
    python dinov3_src/dinov3/train/train.py \
    --config-file "${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    "${DATA_ARGS}" \
    "train.seed=${SEED}"


    