#!/bin/bash

export DGXSYSTEM=H100_2x4
export GPUS_PER_NODE=4
export CPUS_PER_TASK=64
export NNODES=4
export MASTER_PORT=29501

export PRIMUS_PATH=/workspace/deps/Primus
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH}"
export EXP=/workspace/code/conf/gpt_oss_20B-pretrain-nvidia.yaml
export DATA_PATH=/data
# export MODEL=/model  # Commented out to enable auto-download

# Set all HuggingFace cache directories to avoid accessing /home
export HF_HOME=/model
export TRANSFORMERS_CACHE=/model
export HF_HUB_CACHE=/model
export HF_DATASETS_CACHE=/model
# Override HOME to prevent any fallback to user home directory
export TRITON_LIBCUDA_PATH=/usr/local/cuda-13/compat/lib.real/libcuda.so.1
export TRITON_CACHE_DIR=/tmp/.triton

export PRIMUS_MICRO_BATCH_SIZE=1
export PRIMUS_GLOBAL_BATCH_SIZE=16
export PRIMUS_LR=4.0e-4
export PRIMUS_MIN_LR=4.0e-5             # Set to 10% of max LR
export PRIMUS_TRAIN_ITERS=1200000       # 1.2M iters × 16 GBS = 19.2B samples
export PRIMUS_LR_WARMUP_ITERS=128
export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-PRIMUS_LR_WARMUP_ITERS))

# NCCL settings for better debugging
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export NCCL_TIMEOUT=1800
# export NCCL_BLOCKING_WAIT=1

# Evaluation frequency (sample-based, adjusts automatically with GBS)
export EVAL_SAMPLES_INTERVAL=12288   # Evaluate every 12,288 samples
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))  # Auto-computed

export PRIMUS_BF16=true
export PRIMUS_FP16=false
export PRIMUS_FP8=null

export PRIMUS_TURBO_ENABLED=false
export USE_TURBO_ATTENTION=false
export USE_TURBO_GROUPED_MLP=false
export USE_TURBO_DEEPEP=false
export TURBO_DEEPEP_NUM_CU=0
export TURBO_SYNC_FREE_MOE_STAGE=0

export PRIMUS_APPLY_ROPE_FUSION=false
export USE_ROCM_MEM_INFO=false

export OVERLAP_GRAD_REDUCE=true
export OVERLAP_PARAM_GATHER=true
export GRADIENT_ACCUMULATION_FUSION=false

export PRIMUS_TP=1
export PRIMUS_PP=4
export PRIMUS_EP=4
export PRIMUS_VP_LAYERS_PER_STAGE=3
export PRIMUS_VP=$([[ "${PRIMUS_VP_LAYERS_PER_STAGE}" != "null" ]] && echo $(( 24 / PRIMUS_PP / PRIMUS_VP_LAYERS_PER_STAGE )) || echo null)
export PRIMUS_OVERLAP_P2P=$([[ "${PRIMUS_VP_LAYERS_PER_STAGE}" != "null" ]] && echo true || echo false)
# For hopper do not enable overlap_moe_ep when using TP > 1
export PRIMUS_OVERLAP_MOE_EP=$([[ "${PRIMUS_TP}" -le 1 && ( "${PRIMUS_PP}" -le 1 || "${PRIMUS_VP_LAYERS_PER_STAGE}" != "null" ) ]] && echo 1  || echo 0)

export ENABLE_MLLOG=1
export MLLOG_OUTPUT_FILE=/results/mlperf_output.log
export MLLOG_TRAIN_LOSS_LOG_FREQ=32
export MLLOG_TARGET_EVAL_LOSS=3.2
export MLLOG_SUBMISSION_BENCHMARK=gpt-oss-20b   
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=SURF
export MLLOG_SUBMISSION_PLATFORM=H100

export HF_TOKEN="${HF_TOKEN:-}"

