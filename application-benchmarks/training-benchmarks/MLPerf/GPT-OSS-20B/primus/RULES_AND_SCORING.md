# GPT-OSS-20B Pre-training Benchmark Rules and Scoring

## 1 Benchmark Objective
Pre-train a 20B parameter GPT Mixture of Experts (MoE) model using any framework following the MLPerf Training v6.0 small_llm_moe_pretraining benchmark specification.
## 2 Benchmark Rules
### 2.1 Base Rules
This benchmark follows the **MLPerf Training CLOSED division** rules as defined in the [MLPerf Training Rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc).

Key aspects of MLPerf CLOSED division:
- Framework implementation is flexible (e.g., PyTorch, Megatron-LM, Primus)
- Implementation (Model/dataset/loss/training) must be mathematically equivalent to the reference implementation
- Allowed hyperparameter tuning (batch size, learning rate, parallelization strategies, etc.)
- Must achieve the specified quality target on the specified dataset
- Configuration/set-up must be consistent between runs

### 2.2 Additional Restrictions
On top of the MLPerf rules, the following additional restrictions apply:

| Constraint | Requirement |
|------------|-------------|
| **GPUs per Run** | Exactly 8 GPUs/accelerators |
| **Precision** | 8-bit precision (e.g., FP8, MXFP8) or higher (BF16/FP16, FP32). Lower precisions are **not** allowed. |
| **Software Version** | The latest stable version of the software available at the time of release of the benchmark suite shall be used, but bugfix releases released after this date may also be used |

### 2.3 Quality Target
As per MLPerf Training v6.0:
- **Target**: ≤ 3.34 log perplexity loss on C4 dataset

## 3. Validation, Scoring and Reporting

### 3.1 Required Outputs
For each run:
- Log files
- Training configuration (hyperparameters, system setup)
- Validation loss trajectory
- Time-to-target measurement
- System information (GPU type, count, interconnect)

### 3.2 Compliance
- Results must pass MLPerf log compliance checker
- Code and configuration must be available for review
- Follow MLPerf submission guidelines

## 4. Scoring
The single-run score for an eligible partition is determined using MLPerf's methodology: 
- Perform 10 runs to the loss target
- Discard slowest and fastest run
- Compute average runtime in minutes over 8 remaining runs

Using the resulting average runtime and the size of the job (8 GPUs), the score is then weak scaled to the entire partition, system and eventually lifetime of the system as described in the requirements document. 


## 5. Reference Implementation

### 5.1 Base Implementation
- **Repository**: [MLPerf Training v6.0 - small_llm_moe_pretraining/primus](https://github.com/mlcommons/training/tree/master/small_llm_moe_pretraining/primus)
- **Framework**: Primus v0.7.0
- **Reference**: Follow mathematical equivalence to reference

### 5.2 Allowed Modifications
Per MLPerf CLOSED rules:
- Framework changes (as long as mathematically equivalent)
- Performance optimizations (compiler flags, kernel fusion, etc.)
- System-specific adaptations (e.g., Apptainer vs Docker)
- Communication optimizations (NCCL/RCCL tuning)

### 5.3 Prohibited Modifications
Per MLPerf CLOSED rules:
- Changes to model architecture
- Changes to dataset or data preprocessing
- Changes to loss calculation
- Changes to quality target or measurement methodology

## 6. Summary

| Aspect | Requirement |
|--------|-------------|
| **Benchmark** | MLPerf Training v6.0 small_llm_moe_pretraining |
| **Division** | CLOSED |
| **Model** | GPT-OSS-20B (MoE) |
| **Dataset** | C4 English (v3.0.1) |
| **GPUs per Run** | 8 |
| **Precision** | 8-bit (FP8) or higher |
| **Quality Target** | ≤3.34 validation loss |
| **Required Runs** | 10 |
| **Framework** | Any |
| **Scoring** | Weak-scaled runs/minute based on geometric mean of middle 8 out of 10 runs|
| **Rules** | [MLPerf Training Rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc) + additional constraints |
