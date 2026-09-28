# DINOv3 7B Pre-training benchmark
## Introduction
This pre-training benchmark targets "few node" (32 GPUs), computer vision, medium to large (512x512) data input pre-training runs using FSDP. 

The benchmark uses Meta's native PyTorch DINOv3 repository to pre-train a 7B parameter ViT in a teacher-student set-up on the ImageNet-1k dataset. It uses the ImageNet-1k dataset, but upsamples the images to 512x512 for the global crop size and 224x224 for the local crop size using bicubic interpolation. This results in larger resolution inputs, makes sure local batch sizes do not need to be too large and simulates more intensive CPU data pre-processing. Furthermore, while the original pre-training script uses a combination of BF16 and (manually implemented) FP8, this is not typical for a native PyTorch implementation. The benchmark will therefore be performed in BF16.


A single run constitutes the time from starting the train script to the time that the checkpoint is produced that meets the quality target (36% top-5 kNN on ImageNet validation set) on 32 GPUs/accelerators. The average run-time of 3 runs with different seeds is then weak scaled to the entire system, and converted to #completed runs per hour for the full system. See the [rules](RULES_AND_SCORING.md) for more information.

## Seting up
### DINOv3 repository
This benchmark requires the use of native PyTorch with GPU support and a SLURM system, and uses the [DINOv3](https://github.com/facebookresearch/dinov3) repository in `dinov3_src`. Specifically, this [pull request](https://github.com/facebookresearch/dinov3/pull/337) which solves an issue with newer versions of NumPy. To initialize the modules, run:

```bash
git submodule update --init --recursive
```
Code changes to the repository are generally not allowed, barring exceptions discussed in the [rules](RULES_AND_SCORING.md).


### Python environment
The simplest approach is to run the following in a virtual environment:
```bash
python -m venv dino_venv
pip install -r requirements.txt
```
This installs the requirements from [dino_src/requirements.txt] as well as Pandas (which is used by the evaluation in the DINOv3 repository). This is the approach assumed in the current scripts, but this can be changed to an alternative approach.

It is also possible to create a Conda environment using (see [`dinov3_src/README.md`](dinov3_src/README.md)):
```bash
micromamba env create -f conda.yaml
micromamba activate dinov3
```

The use of PyTorch containers/apptainers is also permissible, as long as the general rules are respected.


### Dataset
This benchmark uses the [ImageNet-1k ILSVRC2012] version, which can be downloaded [here](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data). For pre-training, this data should be stored on scratch storage.

The root of the required input data data is located in `ILSVRC/Data/CLS_LOC` of the downloaded contents, as detailed [`dinov3_src/README.md` ImageNet-1k preparation section](dinov3_src/README.md#data-preparation). However, the download does not include the required [`labels.txt`](labels.txt) file (included in this repo), which needs to be placed in the root. 

For dataloading, the DINOv3 repository depends on some additional metadata. This metadata can be generated using [`imagenet1k_prep.py`](imagenet1k_prep.py) as follows:
```bash
export ROOT_PATH=...
export EXTRA_PATH=...
python imagenet1k_prep.py # generates and stores the extra metadata
```

In the example SLURM jobscripts below the metadata generation is disabled (meaning it has to be done once beforehand), but it can be enabled there too.

## Running the Benchmark
### Pre-training
Given the following variables:
```bash
export CONFIG_FILE=... # configs/train/benchmark.yaml
export OUTPUT_DIR=... # Location to save output
export DATA_ARGS=... # In the form: DATA_ARGS="train.dataset_path=ImageNet:split=TRAIN:root=${ROOT_PATH}:extra=${EXTRA_PATH}"
export SLURM_NTASKS_PER_NODE=... 
export SEED=... # 2, 4 or 6
```
the script in [`record_time_and_run.sh`](record_time_and_run.sh) saves the starting time in `${OUTPUT_DIR}/timing.log` and starts the pre-training run. The resulting checkpoints will be stored in `${OUTPUT_DIR}/eval` for the teacher checkpoints (used for evaluation).


This script can be used via SLURM jobscripts, with an example in [`run_benchmark.job`](run_benchmark.job) for Snellius. The jobscript and the launchscript may be altered based on the system/set-up, as long as the starting time measurement is equivalent, the run is initiated by directly calling `python dinov3_src/dinov3/train/train.py` with an unmodified (barring listed exceptions) `train.py` file and the general rules are respected.

The configuration file (`CONFIG_FILE`) that is used for the runs must adhere to the general rules. `configs/train/benchmark.yaml` should be used as a starting base, with some allowed changes. See the rules and comments for more information.


For a valid result, three runs with different seeds (2, 4 and 6) must be completed.


### Evaluation and Time Measurement

#### Checkpoint Generation
The pre-training script periodically saves checkpoints for evaluation:
- **Periodicity**: Controlled by `evaluation.eval_period_iterations` parameter (see rules for restrictions)
- **Location**: Checkpoints stored in `${OUTPUT_DIR}/eval`

#### Time Measurement
Total time is measured from:
- **Start**: Starting time saved in `${OUTPUT_DIR}/timing.log`
- **End**: Time at which the checkpoint reaching the quality target is saved (logged in `${OUTPUT_DIR}/logs/log.txt`)
- **Quality target**: 36% top-5 accuracy using the best k-NN classifier

#### Manual Evaluation
A specific checkpoint can be evaluated using the following command:

```bash
srun \
    --ntasks-per-node=${SLURM_NTASKS_PER_NODE} \
    python dinov3_src/dinov3/eval/knn.py \
    model.config_file="${OUTPUT_DIR}/config.yaml" \
    output_dir="${RESULTS_DIR}" \
    model.pretrained_weights="${CHECKPOINT_PATH}" \
    "train.dataset=ImageNet:split=TRAIN:root=${ROOT_PATH}:extra=${EXTRA_PATH}" \
    "eval.test_dataset=ImageNet:split=VAL:root=${ROOT_PATH}:extra=${EXTRA_PATH}"
```

The evaluation script:
- Trains k-NN classifiers for k ∈ {10, 20, 100, 200}
- Evaluates each classifier in terms of top-1 and top-5 accuracy
- Saves results in `${RESULTS_DIR}`
- **Target**: 36% top-5 k-NN classification accuracy using the best performing k value

#### Automated Evaluation
The [`run_eval.job`](run_eval.job) script automates the evaluation process:
- Evaluates checkpoints consecutively (starting from the first by default)
- Stops when a checkpoint reaches the required target
- Extracts and saves the checkpoint timestamp from logs
- Records the total duration in `${OUTPUT_DIR}/timing.log`

**Note**: The submitter is free to alter [`run_eval.job`](run_eval.job) and [`record_time_and_run.sh`](record_time_and_run.sh) as needed, as long as the final timing result is determined equivalently. The evaluation script and configuration itself cannot be altered, barring the general exceptions.

## Snellius H100 Results
Snellius has two GPU partitions; 72 4-way 40GB A100 nodes and 88 4-way H100 94 GB (HBM2e) nodes. Tests are performed on the H100 partition.

### Single Run Performance
- **GPUs**: 32 × H100 94GB (HBM2e)
- **Local batch size**: 16 (gbs: 32*16=512)
- **Runtime**: ~34 hours (25k iterations) to reach produce checkpoint that reaches quality target

### H100 Partition Score
- **H100 partition size**: 88 nodes × 4 GPUs = 352 GPUs
- **Weak scaling factor**: 352 / 32 = 11
- **System throughput**: 11 / 34 hours = **0.32 runs per hour**




