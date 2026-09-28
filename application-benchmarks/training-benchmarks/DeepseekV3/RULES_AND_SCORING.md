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
| **Precision** | 8-bit precision (e.g., FP8, MXFP8) or higher (BF16/FP16, FP32, TF32). Lower than 8-bit precisions are **not** allowed. |
| **Software Version** | **Up to** the latest stable version of the software available at the time of acceptance shall be used. |

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
### 4.1 Single run
The single-run score for the accelerated partition is determined using MLPerf's methodology: 
- Perform a run
- Take runtime of remaining run

The exact method of measuring the runtime of a single run is detailed in the [MLPerf Training Rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc). Paraphrasing it here (but refer to the MLPerf rules for the full description): the clock starts either before any part of the system touches the dataset, or once the maximum model initialization time is exceed (30 minutes), whichever comes first. The clock stops when the system detects that it has reached the target (3.6 log perplexity).

### 4.2 System score

As per the MLPerf rules a total of ***three*** runs are required to provide a final score. However, this benchmark adds the constraint that these are 3 **consecutive** runs, and the final reported runtime `PERFTIME(K,J,P)` is the ***mean*** runtime of these three runs (without discarding the slowest and fastest run).


**IMPORTANT**:Unlike the other benchmarks, the resulting runtime is **not** weak-scaled over the partition and is therefore only scaled over the lifetime of the system as described in the requirements document. This benchmark is intended to be a full system run. Explicitly, that means that `PPJOBS(K,J,P)` is defined as follows:
```
PPJOBS(K,J,P) = LIFESPAN(J,P) / PERFTIME(K,J,P)
```
Where `PPJOBS(K,J,P)` is the performance of the lifetime of the system for benchmark `K`=DeepseekV3, vendor system `J` and partition `P`. `LIFESPAN(J,P)` is the lifespan of the partition `P` of vendor system `J`. `PERFTIME(K,J,P)` is the performance of the run, which is time to quality target as determined above.

A submission with a subset of the accelerated partition is admissable, and may actually yield a higher score than the full system if it allows a more favorable parallelization strategy, but will be counted **as-is**.

### 4.3 Hardware failures
In case of a failed run (e.g., from a hardware failure) the benchmarker can simply keep performing runs there are 3 consecutive runs without failures. There is no limit on the total number of runs that are performed, as long as the reported runtimes are from 3 consecutive runs.


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
| **Required Runs** | 3 consecutive |
| **Framework** | Any |
| **Scoring** | Mean average runtime of 3 consecutive runs |
| **Rules** | [MLPerf Training Rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc) + additional constraints in this document |
