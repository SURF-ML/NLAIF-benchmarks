# DeepseekV3 Pre-training Benchmark Rules and Scoring

## 1 Benchmark Objective
Pre-train a 671B parameter GPT Mixture of Experts (MoE) model using any framework following the MLPerf Training v6.0 llm_moe_pretraining benchmark specification.
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
| **Precision** | 8-bit precision (e.g., FP8, MXFP8) or higher (BF16/FP16, FP32). Lower precisions are **not** allowed. |
| **Software Version** | The latest stable version of the software available at the time of release of the benchmark suite shall be used, but bugfix releases released after this date may also be used |

### 2.3 Quality Target
As per MLPerf Training v6.0:
- **Target**: ≤ 3.6 log perplexity loss on C4 dataset

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
- Perform 3 runs to the loss target
- Discard slowest and fastest run
- Take runtime of remaining run

**IMPORTANT**:Unlike the other benchmarks, the resulting runtime is **not** weak-scaled over the partition and is therefore only scaled over the lifetime of the system as described in the requirements document. This benchmark is intended to be a full system run. Explicitly, that means that `PPJOBS(K,J,P)` is defined as follows:
```
PPJOBS(K,J,P) = LIFESPAN(J,P) / PERFTIME(K,J,P)
```
Where `PPJOBS(K,J,P)` is the performance of the lifetime of the system for benchmark `K`=DeepseekV3, vendor system `J` and partition `P`. `LIFESPAN(J,P)` is the lifespan of the partition `P` of vendor system `J`. `PERFTIME(K,J,P)` is the performance of the run, which is time to quality target as determined above.


## 5. Reference Implementation

### 5.1 Base Implementation
- **Repository**: [MLPerf Training v6.0 - llm_moe_pretraining/nemo](https://github.com/mlcommons/training/tree/master/llm_moe_pretraining/nemo)
- **Framework**: Nemo
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
| **Benchmark** | MLPerf Training v6.0 llm_moe_pretraining |
| **Division** | CLOSED |
| **Model** | DeepseekV3 |
| **Dataset** | C4 English (v3.0.1) |
| **GPUs per Run** | (near) full system |
| **Precision** | 8-bit (FP8) or higher |
| **Quality Target** | ≤3.6 validation loss |
| **Required Runs** | 3 |
| **Framework** | Any |
| **Scoring** | Runtime to loss target of middle run out of 3 runs |
| **Rules** | [MLPerf Training Rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc) + additional constraints |
