# MLPerf GPT-OSS-20B Pre-training benchmark
## Introduction

 Adaption of https://github.com/mlcommons/training/tree/master/small_llm_moe_pretraining/primus on [Snellius](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660184/Snellius) hardware and software. A part of the reason for this adaption the lack of Pyxis/container support on Snellius, requiring a port to apptainer. See their README for more information.


This pre-training benchmark targets single (or two) node (8 GPUs), small input data, small/medium Mixture-of-Experts model (20B) pre-training, using communication intensive parallelization strategies (expert parallel). 

This benchmark follows all the CLOSED division rule-sets from the [MLPerf training rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc), as well as some additional rules of our own. In short, this means that the framework and exact implementation of the training script is flexible, as long as the result is mathematically equivalent to the reference implementation beyond the allowed tunable hyper-parameters (such as batch size and learning rate). 


Some modifications on top of the MLPerf rules are made for this benchmark. Specifically:
- Precision format is restricted to be **at least** 8-bit precision (e.g., FP8/MXFP8). Higher precisions (e.g., BF16/FP16, TF32, FP32, FP64) are still permitted according to the MLPerf rules. 
- The final score is derived as the ***mean*** of 10 **consecutive** runs, without discarding the best and worst performing run.
- Each run is performed using exactly 8 GPUs
- Each run must be on a different subset of nodes on the accelerated partition

For more details on these rules, as well as the scoring mechanism we refer to [`RULES_AND_SCORING.md`](RULES_AND_SCORING.md).

Below is the original README, supplemented with descriptions of our additions/changes to the reference implementation.


# GPT-OSS-20B Pretraining Benchmark

GPT-OSS 20B (Mixture of Experts)

## Overview

This benchmark trains a 20B parameter GPT model with Mixture of Experts (MoE) architecture using the Primus framework on AMD and NVIDIA GPUs.

# 1. Setup apptainer image
Since Snellius does not support Docker containers/Pyxis the Dockerfile for Nvidia was ported to an apptainer def file in `gpt-oss-20b-primus.def`. Here, the version of Primus was also fixed to v0.7.0 to fix a version mismatch between Primus and the patch. To build the apptainer, adapt and run `build_primus.job`, which creates a `primus.sif` apptainer image.

# 2. Prepare Dataset

The current codebase uses the c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4) for training and evaluation.

## Download Preprocessed Data

The pre-tokenized dataset is available for download. Navigate to your desired download directory (`DATA_DIR`)and run the following commands:

```bash
# Create desired download directory with the right permission 
cd ${DATA_DIR}

# Download training and validation data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri
```

After download, you should see files with the following naming conventions:
- Training: `c4-train.en_6_text_document.bin` and `.idx`
- Validation: `c4-validation-91205-samples.en_text_document.bin` and `.idx`

The data directory is approximately **80 GB**.

# 3. Run Training

## Set Environment Variables

Set the directory for data and results and apptainer image. Ensure `$LOGDIR` has write access. Optionally set these environment variables, or edit them later in the `launch_apptainer.sh` script.

```bash
export DATADIR=...
export MODELDIR=...
export LOGDIR=...
export SIF=primus.sif
export WALLTIME=... # E.g., 1-00:00:00
export CODEDIR=... # location of current code directory

# Create results directory
mkdir -p $LOGDIR
sudo chmod -R 777 $LOGDIR
```

## Choose Configuration

Choose or create appropriate configuration and system-specific hyper-parameters based on hardware type:

| Config File | System | GPUs | TP | EP | PP | Precision |
|-------------|--------|------|----|----|----|-----------|
| `config_MI355X_1x8x1.sh` | MI355X | 1 node × 8 GPUs | 1 | 8 | 1 | BF16
| `config_B200_1x8x1.sh` | B200 | 1 node × 8 GPUs | 1 | 8 | 1 | BF16
| `config_snellius_H100_4x4.sh` | H100 | 4 nodes × 4 GPUs | 1 | 4 | 4 | BF16
| `config_snellius_H100_4x4_fp8.sh` | H100 | 4 nodes × 4 GPUs | 1 | 4 | 4 | BF16/FP8
| `config_snellius_H100_2x4.sh` | H100 | 2 nodes × 4 GPUs | 1 | 4 | 4 | BF16

```bash
export CONFIG=... # e.g., config_snellius_H100_4x4.sh
```

## Launch Training

### Apptainer
To run with apptainer, there is a `launch_apptainer.sh` script that takes a config file, and performs an `sbatch run_apptainer.sub` with the proper arguments to submit the job. This job runs a pre-training for each of the experiments (`NEXP`), using the `run_apptainer_wrapper.sh` that uses the apptainer image to run `run_and_time.sh`
#### Single Run

```bash
export NEXP=1
./launch_apptainer.sh $CONFIG_FILE
```

#### Multiple Runs (for submission)

```bash
export NEXP=10
./launch_apptainer.sh $CONFIG_FILE
```


After completion, logs will be available under `$LOGDIR`.

# 4. Quality Metrics

## Target loss

3.34

## Quality Metric

Validation loss (log perplexity)

## Evaluation Frequency

Evaluation every **12,288 samples** (768 iterations with GBS=16)

## Evaluation Thoroughness

We evaluate using the first **1,024 samples** from the validation dataset.

# 5. Model Architecture

| Parameter | Value |
|-----------|-------|
| Model Size | 20B parameters |
| Architecture | GPT with Mixture of Experts |
| Sequence Length | 8192 |
| Expert Parallelism | 8 |

# 6. Training Configuration
Reference:
| Hyperparameter | Value |
|----------------|-------|
| Micro Batch Size | 2 |
| Global Batch Size | 16 |
| Learning Rate | 8e-4 |
| LR Schedule | Cosine decay with warmup |
| Weight Decay | 0.1 |
| Adam β1, β2, eps | 0.9, 0.95, 1e-5 |
| Max Training Iterations | 1,200,000 |

Snellius:
| Hyper-parameter | Value |
|----------------|-------|
| Micro Batch Size | 1 |
| Global Batch Size | 16 |
| Learning Rate | 8e-4 |
| LR Schedule | Cosine decay with warmup |
| Weight Decay | 0.1 |
| Adam β1, β2, eps | 0.9, 0.95, 1e-5 |
| Max Training Iterations | 1,200,000 |

# 7. Directory Structure

```
gpt-oss-20b/primus/
├── conf/                           # Configuration files
│   └── gpt_oss_20B-pretrain.yaml
│   └── gpt_oss_20B-pretrain-nvidia.yaml
└── gpt_oss_20B-pretrain-nvidia.yaml
├── src/                            # Training source code
│   └── train.py
├── build_primus.job                # Builds apptainer image
├── config_MI355X_1x8x1.sh          # System configuration (MI355 - AMD)
├── config_B200_1x8x1.sh            # System configuration (B200 - NVIDIA)
├── config_snellius_H100_4x4.sh     # System configuration (H100 4 nodes - Snellius)
├── config_snellius_H100_2x4.sh     # System configuration (H100 2 nodes - Snellius)
├── Dockerfile                      # Dockerfile (MI355 - AMD)
├── Dockerfile.nvidia               # Dockerfile (B200 - NVIDIA)
├── gpt-oss-20b-primus.def          # Apptainer definition (H100 - Snellius)
├── launch_apptainer.sh             # Launches apptainer job from config (run_apptainer.sub)
├── run_and_time.sh                 # Script to run and time a single run
├── run_apptainer_wrapper.sh        # Launches run_and_time from apptainer image
├── run_apptainer.sub               # Apptainer Slurm batch script
└── requirements.txt                # Python dependencies (includes primus-mllog)
```
# 8. Approximate runtime
Reference (AMD)
- Reference (8xMi355x single node) Approximate train time to convergence is ~6.5 hours.
**correction**: From provided RCP logs time to 3.34 loss is ~5.5 hours. 6.5 hours was to 3.2 loss

Snellius (Nvidia)
- 4N (16xH100): BF16: ~4.5 hours (~160TFLOP/s/GPU), hybrid FP8: ~4 hours (~180TFLOP/s/GPU)
- 2N (8xH100): BF16: ~7 hours (~205TFLOPs/s/GPU)


# 9. Overview of changes
- Provided a way to port the implementation to apptainer and regular SLURM batch jobs
- Fixed the PRIMUS versioning in the docker and apptainer build files.
- Added configuration files for the Snellius H100 partition
- Added FLOP reporting in the `src/train.py` script.
- Patched mlperf_logger.py to support proper logging when PP>1
