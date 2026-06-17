#!/bin/bash
#
# Wrapper script to run training inside Apptainer container
# This script sources the config file and runs the training
#

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

CONFIG_FILE="$1"

cd /workspace/code

# Source the configuration file to set environment variables
if [ -f "${CONFIG_FILE}" ]; then
    echo "Loading configuration from: ${CONFIG_FILE}"
    source "${CONFIG_FILE}"
else
    echo "ERROR: Configuration file not found: ${CONFIG_FILE}"
    exit 1
fi

# Set up SLURM environment for multi-node training
# srun automatically sets SLURM_PROCID, SLURM_LOCALID, etc. for each task
export NODE_RANK="${SLURM_PROCID}"
export LOCAL_RANK="${SLURM_LOCALID:-0}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-${GPUS_PER_NODE}}"
export NNODES="${SLURM_JOB_NUM_NODES}"
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno2np0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eno2np0}"

echo "============================================"
echo "Apptainer Training Launch"
echo "============================================"
echo "Node: ${SLURM_NODEID:-0}/${SLURM_JOB_NUM_NODES:-1}"
echo "NODE_RANK: ${NODE_RANK}"
echo "LOCAL_RANK: ${LOCAL_RANK}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "============================================"

# Call the original run_and_time.sh script
exec ./run_and_time.sh
