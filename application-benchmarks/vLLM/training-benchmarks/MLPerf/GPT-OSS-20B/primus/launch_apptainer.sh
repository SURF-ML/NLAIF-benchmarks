#!/bin/bash
#
# Example launch script for Apptainer-based GPT-OSS-20B training on Snellius
#
# Usage: ./launch_apptainer.sh [config_name]
#
# Example: ./launch_apptainer.sh config_snellius_H100.sh 
#

set -e

# Configuration
CONFIG="${1:-config_snellius_H100_2x4.sh}"
source "${CONFIG}"
WALLTIME="${WALLTIME:-1-00:00:00}"

# Paths - ADJUST THESE FOR YOUR SETUP
CODEDIR="${CODEDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATADIR="${DATADIR:-/projects/hpmlprjs/MLPerfTraining/data/gpt_oss_20b/data}"
MODELDIR="${MODELDIR:-/projects/hpmlprjs/MLPerfTraining/data/gpt_oss_20b/model}"
SIF="${SIF:-/projects/hpmlprjs/MLPerfTraining/training-benchmarks/MLPerf/GPT-OSS-20B/primus/primus.sif}"
LOGDIR="${LOGDIR:-${CODEDIR}/primus_logs}"

# SLURM configuration. Read from config file or set defaults here
PARTITION="${PARTITION:-gpu_h100}"
ACCOUNT="${ACCOUNT:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-64}"

# Experiment settings
NEXP="${NEXP:-1}"
SEED="${SEED:-$RANDOM}"

# Validate required paths
if [ ! -f "${SIF}" ]; then
    echo "ERROR: SIF file not found: ${SIF}"
    echo "Please build the container first or set SIF environment variable"
    exit 1
fi

if [ ! -f "${CODEDIR}/${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CODEDIR}/${CONFIG}"
    echo "Available configs:"
    ls -1 "${CODEDIR}"/config_*.sh 2>/dev/null || echo "  (none found)"
    exit 1
fi

if [ ! -d "${DATADIR}" ]; then
    echo "ERROR: Data directory not found: ${DATADIR}"
    echo "Please set DATADIR environment variable"
    exit 1
fi

if [ ! -d "${MODELDIR}" ]; then
    echo "ERROR: Model directory not found: ${MODELDIR}"
    echo "Please set MODELDIR environment variable"
    exit 1
fi

# Create results directory
mkdir -p "${LOGDIR}"

# Build sbatch command
SBATCH_CMD="sbatch"
SBATCH_CMD="${SBATCH_CMD} --job-name=gpt-oss-20b"
SBATCH_CMD="${SBATCH_CMD} --nodes=${NNODES}"
SBATCH_CMD="${SBATCH_CMD} --ntasks-per-node=1"
SBATCH_CMD="${SBATCH_CMD} --gpus-per-node=${GPUS_PER_NODE}"
SBATCH_CMD="${SBATCH_CMD} --cpus-per-task=${CPUS_PER_TASK}"
SBATCH_CMD="${SBATCH_CMD} --time=${WALLTIME}"
SBATCH_CMD="${SBATCH_CMD} --partition=${PARTITION}"
SBATCH_CMD="${SBATCH_CMD} --switches=1"

if [ -n "${ACCOUNT}" ]; then
    SBATCH_CMD="${SBATCH_CMD} --account=${ACCOUNT}"
fi

if [ -n "${RESERVATION}" ]; then
    SBATCH_CMD="${SBATCH_CMD} --reservation=${RESERVATION}"
fi

SBATCH_CMD="${SBATCH_CMD} --output=${LOGDIR}/slurm-%j.out"
SBATCH_CMD="${SBATCH_CMD} --error=${LOGDIR}/slurm-%j.err"

# Export environment variables for the job
export CODEDIR
export DATADIR
export MODELDIR
export SIF
export CONFIG
export LOGDIR
export NEXP
export SEED
export WALLTIME
export GPUS_PER_NODE

echo "========================================"
echo "Submitting GPT-OSS-20B Training Job"
echo "========================================"
echo "Config:          ${CONFIG}"
echo "Nodes:           ${NNODES}"
echo "GPUs per node:   ${GPUS_PER_NODE}"
echo "Walltime:        ${WALLTIME}"
echo "Partition:       ${PARTITION}"
echo "Code dir:        ${CODEDIR}"
echo "Data dir:        ${DATADIR}"
echo "Model dir:       ${MODELDIR}"
echo "SIF file:        ${SIF}"
echo "Results dir:     ${LOGDIR}"
echo "Experiments:     ${NEXP}"
echo "Seed:            ${SEED}"
echo "========================================"

# Submit the job
JOBID=$(${SBATCH_CMD} --export=ALL run_apptainer.sub | awk '{print $NF}')

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u ${USER}"
echo "  tail -f ${LOGDIR}/slurm-${JOBID}.out"
