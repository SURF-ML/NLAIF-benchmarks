# DINOv3 7B Pre-training Benchmark Rules

## 1. Overview

### 1.1 Benchmark Objective
Pre-train a 7B parameter Vision Transformer (ViT) using Meta's DINOv3 repository on ImageNet-1k dataset with FSDP distributed training on 32 GPUs.

### 1.2 Quality Target
- **Metric**: Top-5 k-NN classification accuracy on ImageNet-1k validation set
- **Target**: 36% using the best performing k value from {10, 20, 100, 200}

### 1.3 Required Runs
Three runs **consecutive** runs on different node-subsets with different random seeds (2, 4, and 6) must be completed for a valid result. There is no limit on the total number of runs, meaning that in case of a hardware failure the benchmark can simply re-run the three runs.

**Configuration Consistency**: All three runs must be executed with exactly the same configuration, hyperparameters, node types, and software environment. The **only** differences between runs must be the random seed value, and the subset of the nodes.

### 1.4 Scoring Methodology
- Measure time-to-target for each of the 3 runs (from start to checkpoint that reaches quality target)
- Calculate the average run-time across the 3 runs
- Weak scale to the entire system (average_runtime / system_size * 32_GPUs)
- Convert to runs per hour for the full system: **Score = system_size / (32 × average_runtime_hours)**

## 2. System Requirements

### 2.1 Hardware
- **GPUs**: 32 GPUs/accelerators per run

### 2.2 Software Stack
- **Framework**: Native PyTorch with GPU support (required)
- **Precision**: BF16/FP16 (mixed precision with BF16 or FP16 weights) and FP32 (TF32 permissable). Tunable through config file.
- **Job Scheduler**: SLURM system required
- **Attention Implementation**: Default PyTorch SDPA (Scaled Dot Product Attention) must be used. Manual installation of Flash Attention or other attention implementations beyond what PyTorch ships with is **not permitted**
- **Python Containers**: Permissible (e.g., PyTorch containers/Apptainers), subject to general rules
- The latest stable version of the software (e.g., PyTorch) available at the time of release of the benchmark suite shall be used, but bugfix releases released after this date may also be used.

### 2.3 Code Base
- **Repository**: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) 
- **Specific Version**: [Pull Request #337](https://github.com/facebookresearch/dinov3/pull/337) (NumPy 2.0 compatibility fix)

## 3. Dataset Requirements

### 3.1 Dataset Specification
- **Dataset**: ImageNet-1k ILSVRC2012
- **Source**: [Kaggle ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)
- **Root Path**: `ILSVRC/Data/CLS_LOC` from downloaded contents
- **Required Files**: `labels.txt` (provided in repository) must be placed in root

### 3.2 Data Preprocessing
- **Global Crop Size**: 512×512 (upsampled using bicubic interpolation)
- **Local Crop Size**: 224×224 (upsampled using bicubic interpolation)
- **Metadata**: Additional metadata required by DINOv3 (generated using `imagenet1k_prep.py`)
- **Storage**: Dataset should be stored on shared (non-local) scratch storage that is accessible for all nodes. Use of scratch local, ramdisk or similar in-memory storage is **not permitted**

## 4. Configuration Rules

### 4.1 Base Configuration
- **Starting Point**: `configs/train/benchmark.yaml` must be used as the base
- **Required Preservation**: All parameters not listed as tunable must remain at their default values

### 4.2 Tunable Hyperparameters

| Parameter | Restrictions | Default | Notes |
|-----------|--------------|---------|-------|
| `train.batch_size_per_gpu` | Unrestricted (integer) | 16 | Also controls global batch size (gbs=GPU×batch_size_per_gpu) |
| `train.num_workers` | Unrestricted (integer) | 10 | Number of dataloader workers |
| `train.compile` | Unrestricted (True/False) | TRUE | Controls use of torch.compile |
| `dino.local_crops_number` | Unrestricted (integer) | TRUE | Number of local crops per image |
| `train.checkpointing_full` | Unrestricted (True/False) | FALSE | Controls the degree of activation checkpointing |
| `train.optim.epochs` | Unrestricted (integer) | 100 | Total training epochs |
| `evaluation.eval_period_iterations` | At least every 1.28M samples. So minimum: 1.28M/(`train.batch_size_per_gpu` / 32) | default=2500 (`train.batch_size_per_gpu`=16) | Controls frequency of saving teacher checkpoint for evaluation |
| `checkpointing.period` | At least every 1.28M samples. So minimum: 1.28M×(`train.batch_size_per_gpu` * 32) | default=2500 (`train.batch_size_per_gpu`=16) | Controls frequency of saving full checkpoint |
| `compute_precision.param_dtype` | fp32,fp16,bf16 | default=bf16 | Controls the precision of the weights |

The precision is only tunable through the `compute_precision.param_dtype` parameter. Using TF32 for FP32 operations is permitted.

### 4.3 Learning Rate Schedule Parameters

| Parameter | Restrictions | Default | Notes |
|-----------|--------------|---------|-------|
| `schedules.lr.peak` | Between `sqrt(train.batch_size_per_gpu / 16)` × DEFAULT_LR and `train.batch_size_per_gpu / 16` × DEFAULT_LR. Must be same as `schedules.lr.end` | 0.0001421 | Only allows tuning learning rate to be between sqrt and linear scaling |
| `schedules.lr.end` | Between `sqrt(train.batch_size_per_gpu / 16)` × DEFAULT_LR and `train.batch_size_per_gpu / 16` × DEFAULT_LR. Must be same as `schedules.lr.peak` | 0.0001421 | Must be same as peak learning rate (lr is constant) |

**Note**: DEFAULT_LR = 0.0001421 (for batch_size_per_gpu=16)

## 5. Code Modification Rules

### 5.1 When a Run Fails
If the benchmark does not run on your platform, there are two sanctioned paths. Never silently work around the code:

1. **Your system needs different settings or a platform-specific adjustment.** You may change system/operational settings to make it run, such as the SLURM jobscript and launch script (see [README](README.md)), interconnect/NIC selection, communication library environment variables, container-runtime flags, cache locations and timeouts. You may also alter the codebase where needed for a fair platform-to-platform comparison. For example, the `filter` for activation checkpointing (`fsdp2.compile_params.parallelize_py`) might not work correctly on every platform. It may be changed, as long as the result follows the original intent and stays mathematically equivalent. **Every such change must be documented** in the submission ([§7.3](#73-documentation-requirements)). You may **not** change any restricted parameter ([§4](#4-configuration-rules)) or make a prohibited change ([§5.2](#52-prohibited-code-changes)) to work around a failure.

2. **The benchmark code or specification itself is broken** (a bug in the benchmark code, not your system). Do not patch around it. Escalate so it is fixed centrally and all parties run the same corrected code: contact **EuroHPC**, who coordinates with the benchmark maintainers to fix the repository.

### 5.2 Prohibited Code Changes
- No changes to model architecture
- No changes to loss computation
- No changes to the evaluation script (`dinov3_src/dinov3/eval/knn.py`)
- No changes that fundamentally alter the training algorithm
- No changes to data augmentation strategies beyond the allowed parameters

### 5.3 Garbage Collection
It is permitted to remove garbage collection or alter the frequency. Requires documentation.

### 5.4 Script/Environment Changes
- **Launch Scripts**: `record_time_and_run.sh` and `run_eval.job` may be altered as needed
  - Time measurement methodology must remain equivalent
  - Training must be initiated by directly calling `python dinov3_src/dinov3/train/train.py`
  - The `train.py` file must be unmodified (except for allowed exceptions)
  
- **Job Scripts**: SLURM job scripts may be adapted to the specific system/setup

- **Environment Setup**: Python environment setup can use virtual environments, Conda, or containers

-  **Environment Variables**: Environment variables (e.g., NCCL/RCCL) may be used, as long as the model execution remains mathemathically equivalent.

## 6. Execution Requirements

### 6.1 Pre-training Execution
- **Configuration File**: Must adhere to all configuration rules
- **Random Seeds**: Must use seeds 2, 4, and 6 for the three required runs
- **Configuration Consistency**: The same configuration file, hyperparameters, and system setup must be used for all 3 runs (only seed changes)
- **Starting Time**: Must be recorded at the beginning of training in `${OUTPUT_DIR}/timing.log`
- **Checkpoints**: Teacher checkpoints stored in `${OUTPUT_DIR}/eval` for evaluation
- **Checkpoint Storage**: Checkpoints must be written to shared (non-local) scratch storage (e.g., parallel filesystem). Use of scratch local, ramdisk or similar in-memory storage is **not permitted**

### 6.2 Time Measurement
- **Start Time**: Recorded when training script begins (`timing.log`)
- **End Time**: Time at which the checkpoint reaching the quality target is saved (from `${OUTPUT_DIR}/logs/log.txt`)
- **Total Time**: Difference between start time and time at which the qualifying checkpoint is saved for each run

### 6.3 Evaluation Process
- **Timing**: Evaluation is performed separately and is **not timed**
- **Checkpoints**: Evaluate checkpoint(s)
- **k-NN Training**: Train k-NN classifiers for k ∈ {10, 20, 100, 200}
- **Metrics**: Compute top-5 accuracy for each classifier
- **Target Achievement**: Checkpoint reaches target when any k ∈ {10, 20, 100, 200} k-NN classifier achieves ≥36% top-5 accuracy

### 6.4 Automation
- The evaluation process may be automated (e.g., using `run_eval.job`)
- Time extraction from logs may be automated (e.g., using `extract_time.sh`)
- The methodology must produce equivalent results to manual evaluation

## 7. Validation, Scoring and Reporting

### 7.1 Required Outputs
For each of the 3 runs, the following has to be provided:
- Training logs (`${OUTPUT_DIR}/logs/log.txt`)
- Timing log (`${OUTPUT_DIR}/timing.log`) with start time, end time, and total duration
- Evaluation results showing the checkpoint that reached the target
- Teacher checkpoints
- Configuration file used (`benchmark.yaml`)
- Documentation (see below)

### 7.2 Final Score Calculation
1. Compute average runtime from start of training to production of eligible checkpoint: `avg_time = (run1_time + run2_time + run3_time) / 3`
2. Using the resulting average runtime and the size of the job (32 GPUs), the score is then weak scaled to the entire partition, system and eventually lifetime of the system as described in the requirements document. 

### 7.3 Documentation Requirements
- All allowed code changes must be documented with justification
- System configuration and hardware specifications
- Any deviations from the reference implementation must be explained
- Python package versions and environment details

## 8. Summary of Key Constraints

| Aspect | Requirement |
|--------|-------------|
| **Model** | 7B parameter ViT (DINOv3 teacher-student) |
| **Dataset** | ImageNet-1k ILSVRC2012 |
| **Precision** | Mixed 16-bit precision (BF16/FP16) and 32-bit (FP32, TF32 permissible) |
| **Image Sizes** | 512×512 (global), 224×224 (local) |
| **GPUs per Run** | 32 |
| **Quality Target** | 36% top-5 k-NN accuracy |
| **Required Runs** | 3 (seeds: 2, 4, 6) |
| **Framework** | Native PyTorch with FSDP |
| **Attention** | Default PyTorch SDPA (no manual Flash Attention) |
| **Code Base** | facebookresearch/dinov3 PR #337 |
| **Configuration** | Based on `configs/train/benchmark.yaml`, limited hyperparameter tuning |
| **Storage** | Dataset and checkpoints on shared scratch storage (no ramdisk) |