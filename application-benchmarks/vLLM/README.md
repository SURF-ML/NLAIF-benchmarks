# LLM Inference Benchmark — Cross-Hardware Evaluation Suite

This document is the authoritative specification for the benchmark. It defines what is
measured, how to reproduce a result, which parameters are fixed, and how submissions are
scored. **Where this document is silent, the configuration is prohibited by default**; contact EuroHPC before deviating from any stated requirement.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Metric Definition](#2-metric-definition)
3. [Models & Precision Policy](#3-models--precision-policy)
4. [Workloads & Datasets](#4-workloads--datasets)
5. [Running the Benchmark](#5-running-the-benchmark)
6. [Submitting Results](#6-submitting-results)
7. [Scoring](#7-scoring)
8. [Rules](#8-rules)
9. [Edge-Case Rulings](#9-edge-case-rulings)
10. [Flag Reference](#10-flag-reference)

---

## 1. Overview

The benchmark measures **LLM serving throughput and interactivity** on a fixed software stack
(vLLM) across accelerator platforms. Model checkpoints, precision, datasets, and workload
parameters are fully specified; the only variable under evaluation is the hardware (and the
software ecosystem that drives it).

Production LLM inference occupies two operating regimes that trade off against each other —
larger batches raise throughput per GPU but lower throughput per user:

| Regime | Optimises for | Dominant bottleneck |
|--------|--------------|---------------------|
| **Online** (interactive) | Low latency per user | Memory bandwidth (decode-bound) |
| **Offline** (batch) | Aggregate throughput | Compute (prefill-bound) |

A benchmark fixed at a single operating point can be tuned for that point alone. This benchmark
therefore measures **throughput per GPU at two operating points spanning both regimes**
([§4](#4-workloads--datasets)) and scores each separately, so a submission tuned for one regime
cannot score well overall. The measurement increases with compute (FLOPS/GPU), memory capacity
and bandwidth per GPU, the number of GPUs in a high-bandwidth domain, and host/interconnect
performance.

---

## 2. Metric Definition

The primary measurement is `total_token_throughput`, as reported by `vllm bench serve`:

```
total_token_throughput = (Σ input_tokens + Σ output_tokens) / wall_clock_seconds
```

It is normalised two ways:

| Quantity | Formula | Interpretation | Role |
|----------|---------|----------------|------|
| **tok/s/GPU** | `total_token_throughput / (TP × PP)` | Hardware efficiency | **Scored** |
| **tok/s/user** | `total_token_throughput × OSL/(ISL+OSL) / max_concurrency` | Per-user interactivity (output tokens a user sees per second) | Reported |

**tok/s/GPU is the scored quantity at every operating point.** Dividing by `TP × PP` keeps
hardware cost in the denominator, so additional GPUs are rewarded only through the larger KV
cache and lower per-GPU load they buy, not through their raw count.

tok/s/user is recorded alongside for interpretation, but is not scored. It counts only output
tokens (ISL and OSL are the measured input and output token totals), since those are what a user
waits on. At a fixed workload shape and concurrency the two are proportional, so scoring both
would count the same measurement twice; and at the highest operating point, scoring tok/s/user
would reward a *smaller* KV cache, since it divides by the concurrency the hardware can sustain.

Each measurement is taken for a given `(workload shape, parallelism layout, operating point)`, with
one engine start per layout covering all of its points ([§5.5](#55-launching-a-sweep)). A **cell**
is one `(workload shape, operating point)` pair: each cell takes the best measurement across
layouts, and every cell is scored independently ([§7](#7-scoring)).

---

## 3. Models & Precision Policy

One model is evaluated:

| Model | Checkpoint | Precision | Architecture |
|-------|------------|-----------|--------------|
| **DeepSeek V4 Pro** | `deepseek-ai/DeepSeek-V4-Pro`, or a permitted FP4 expert checkpoint (below) | FP4+FP8 mixed (native) | MoE, 1.6T total / 49B active |

**Precision is fixed, with one exception.** The score contains no accuracy term, so a free choice
of precision would reward throughput gained by degrading output quality — gaming the metric cannot
detect. Every tensor therefore runs at the base checkpoint's precision, except the expert layers,
which may be FP4 in a permitted FP4 expert checkpoint.

### Base checkpoint precision

DeepSeek V4 Pro ships with mixed precision. From the checkpoint's tensor headers:

| Tensors | Base checkpoint | In a permitted FP4 expert checkpoint |
|---------|-----------------|--------------------------------------|
| Routed experts `ffn.experts.*` | MXFP4 (block 32, `F8_E8M0` scale) | MXFP4 or NVFP4; FP4 activations permitted |
| Shared experts `ffn.shared_experts.*` | FP8 `F8_E4M3` + `F8_E8M0` scale | MXFP4 or NVFP4; FP4 activations permitted |
| Attention projections `attn.{wq_a, wq_b, wkv, wo_a, wo_b}`, `attn.indexer.wq_b` | FP8 `F8_E4M3` + `F8_E8M0` scale | unchanged |
| Compressors `attn.compressor.*`, `attn.indexer.compressor.*`, `attn.indexer.weights_proj` | BF16 | unchanged |
| Router `ffn.gate`, embedding, head, all norms | BF16 | unchanged |

### FP4 expert checkpoints

A submission may use a quantised build of the base model instead of the base checkpoint, provided
all of the following hold:

1. **Official and public.** It is published on Hugging Face by the model developer or by an
   accelerator vendor's official organisation (e.g. `nvidia/`, `amd/`), is publicly available at
   acceptance, and is derived from `deepseek-ai/DeepSeek-V4-Pro`.
2. **Only the experts differ.** Only routed-expert (`ffn.experts.*`) and shared-expert
   (`ffn.shared_experts.*`) tensors may differ from the base checkpoint. Every other tensor —
   attention, indexer, compressors, router, norms, embedding, head, MTP layers — is byte-identical
   to the base, with its original dtype and scales. The tokenizer is identical too, since the
   datasets' token counts depend on it ([§4](#4-workloads--datasets)).
3. **FP4 weights, quantised directly.** Expert weights are **MXFP4** (block 32, E8M0 scale) or
   **NVFP4** (block 16, FP8-E4M3 scale plus a per-tensor FP32 scale); the base MXFP4 weights may be
   requantised to NVFP4. Weights are quantised directly from the base checkpoint's weights, per
   block. Training, fine-tuning, distillation, error-compensating quantisation (GPTQ/AWQ-style),
   pruning, and merging or dropping experts are prohibited.
4. **FP4 activations, experts only.** The inputs to the expert GEMMs may be quantised to the same
   FP4 format as the weights. Block scales are computed at run time; NVFP4's per-tensor global scale
   may be static. Any calibration data must not come from the benchmark datasets. Nothing outside
   the experts runs below the base checkpoint's precision, and the KV cache stays as fixed in
   [§8.1](#81-fixed-parameters).
5. **Runs in an allowed image.** It loads and runs in an official vLLM image at a stable release
   ([§5.1](#51-system-requirements)) at acceptance, with no additional packages ([§8.5](#85-prohibited)).
   Whether a given release already executes the FP4 format natively on a given platform is a
   software matter: the checkpoint is permitted either way, and the measured throughput reflects it.
6. **Disclosed.** The submission names the checkpoint (repository and revision) and its FP4 format
   ([§6](#6-submitting-results)).

Published builds that fit this rule, by a check of their tensor headers: `nvidia/DeepSeek-V4-Pro-NVFP4`
and `amd/DeepSeek-V4-Pro-NVFP4` (routed experts NVFP4, shared experts FP8 as in the base), and
`amd/DeepSeek-V4-Pro-MXFP4` (routed and shared experts MXFP4).

To use one, set `"model"` for that cluster in `CLUSTERS` in
[`scripts/gen_configs.py`](scripts/gen_configs.py) and regenerate. Its cells score as
`deepseek-ai/DeepSeek-V4-Pro`, so they are compared with base-checkpoint results, and `scores.csv`
records the checkpoint that produced each cell.

### DeepSeek V4 Pro layout

Every platform loads the base `deepseek-ai/DeepSeek-V4-Pro` checkpoint (or a permitted FP4 expert
checkpoint) and uses tensor-,
pipeline-, and expert-parallelism ([§8.4](#84-parallelism)), sized to the model's
~1.6T-parameter footprint (≈16 GPUs). Lay it out to match the node's GPU count and memory: a
single node where enough high-bandwidth GPUs are available, or across nodes (the Snellius
H100 reference serves it as **TP4 × PP4 across four 4-GPU nodes** via Ray).

---

## 4. Workloads & Datasets

### Datasets are fixed inputs — download, never regenerate

The benchmark datasets are vLLM `CustomDataset` JSONL files. **Download them
from the organisers:**

> **https://surfdrive.surf.nl/s/MjZ7zQ2ZrZY7nnX**

They **must not be regenerated, reshuffled, truncated, re-tokenised, or substituted.** They are
deterministic random-token corpora (seed `0`, `--range-ratio 0`), so every prompt has an exact
token count; each model's files use that model's own tokeniser, so token lengths are not portable
across models. The harness points each model at its matching files — cross-assigning them is
invalid.

Place the files under `datasets/`. The harness copies each one to **`/dev/shm`** before serving
(local RAM disk, so every worker reads it at memory speed); the bench configs reference the
`/dev/shm/...` path. The `n1024` / `n32768` / `n131072` filename suffix is the **prompt count** —
**the same dataset at different sizes** (the larger files extend the smaller ones, so they are
prefix-consistent; a given concurrency always selects the same file and therefore the same
prompts). Each run needs `num_prompts = 2 × max_concurrency`, and
[`scripts/gen_configs.py`](scripts/gen_configs.py) selects the smallest file that covers it
automatically — you do not choose a size by hand.

### Workload shapes

The context window is fixed at **`--max-model-len = 9216`**. Two shapes test opposite regimes:

| Shape | Input tokens | Output tokens | Primary stress |
|-------|:------------:|:-------------:|----------------|
| `i8192_o1024` | 8192 | 1024 | Prefill — FLOPS-bound |
| `i1024_o8192` | 1024 | 8192 | Decode — memory-bandwidth-bound |

Both sum to exactly 9216, so a sequence occupies the same peak KV-cache footprint under either
shape. That is why a single maximum batch size covers both (below).

### Operating points

Each shape is run at a small set of fixed operating points, defined relative to **N**, the
maximum batch size the deployment sustains:

| Point | Concurrency | Regime |
|-------|-------------|--------|
| `low` | 1 | Single-stream interactivity |
| `high` | N | Saturated — peak per-GPU efficiency |

`max_concurrency` (client) is set to the point's value; `max_num_seqs` (server) is set to **N for
every point**, so one deployment configuration is measured under different load levels and client
load is the only variable that moves.

The point set is defined once, in `POINTS` in [`scripts/gen_configs.py`](scripts/gen_configs.py). Nothing else in the harness or this document encodes how many points there are.

#### Choosing N

**N is tunable per parallelism layout** and declared by the vendor in `serve-params.json` (see
[§8.4](#84-parallelism)). It is not a free scoring lever: tok/s/GPU at the `high` point rises with
N while throughput still scales, and *falls* once N exceeds what the KV cache holds and vLLM
begins preempting. **Declaring your true maximum is also your best score**, so no harness
enforcement is needed.

Determine N yourself, by measurement. vLLM's startup line (`Maximum concurrency for 9216 tokens
per request: …x`) is not an accurate ceiling — for DeepSeek-V4-Pro it understates what fits by
more than an order of magnitude — so do not take N from it. Because both shapes have identical
peak KV footprint, **one N covers both**.

The scored configuration space per submission is:

```
1 model  ×  2 workload shapes  ×  2 concurrency settings
```

---

## 5. Running the Benchmark

### 5.1 System requirements

| Requirement | Detail |
|-------------|--------|
| Job scheduler | **SLURM (required).** One worker job per `(serve, bench)` pair; multi-node serves use `srun` + Ray. The orchestrator may run as a submitted job or locally with `--local` (see §5.5). |
| Container runtime | Apptainer / Singularity, or Podman. Workers run exclusively inside the official image. |
| Official base image | Any official vLLM image from [`hub.docker.com/r/vllm/`](https://hub.docker.com/r/vllm/) — e.g. `vllm/vllm-openai` (CUDA) or `vllm/vllm-openai-rocm` (ROCm) — at a stable release (see below). Images from any other source are not permitted ([§8.5](#85-prohibited)). |
| Accelerators | NVIDIA or AMD. Single-node and multi-node (NVIDIA validated; AMD validated single-node and on 2× 8 MI355X — see [§5.4](#54-platform-configuration)). |
| Host Python | ≥ 3.7 (orchestrator only — runs on the host for SLURM access). |
| Model cache | A shared `HF_HOME` visible from all nodes. Download weights once before the run. |
| Datasets | `datasets/*.jsonl` from the organisers ([§4](#4-workloads--datasets)). |

**vLLM version.** Use any **stable, tagged vLLM release** (`vX.Y.Z`) that is publicly available
at acceptance, with its official image. Nightly, release-candidate (`rc`), development and
self-built-from-source versions are not permitted. Declare the exact version in the submission
([§6](#6-submitting-results)).

Different submissions may therefore use different releases. That is intended: the benchmark
evaluates the hardware together with the software ecosystem that drives it ([§1](#1-overview)), and
engine maturity is part of that ecosystem. It is also material — on MI355X, v0.29.0 to v0.30.0 was
worth ~13% for this model, and v0.23.0 to v0.29.0 a factor of 5.5 — so choose deliberately.

`VLLM_VERSION` in [`containers/build_vllm_bench.slurm`](containers/build_vllm_bench.slurm) is only
the default the build script uses; set it to the release you run.

### 5.2 Building the container

The image adds only what the harness requires: the benchmark client (`vllm[bench]`) and the
multi-node serving transport (`ray[cgraph]` + its matching CuPy build, for cross-node collectives
when serving DeepSeek-V4-Pro across nodes). No model-execution packages — alternative kernels,
attention libraries, or quantisation backends — may be added.

```bash
sbatch containers/build_vllm_bench.slurm                          # NVIDIA (default)
GPU_VENDOR=amd sbatch containers/build_vllm_bench.slurm           # AMD (ROCm)
BUILDER=podman GPU_VENDOR=amd sbatch containers/build_vllm_bench.slurm   # Podman builder
```

Output: `containers/vllm-bench_<vendor>_<version>.sif` (or `.tar` for Podman). Point the
`containers/latest.sif` symlink at the new image.

### 5.3 Downloading model weights

```bash
export HF_HOME=/path/to/shared/model-cache   # must be readable on every compute node

huggingface-cli download deepseek-ai/DeepSeek-V4-Pro
# or a permitted FP4 expert checkpoint (§3), e.g.:
huggingface-cli download nvidia/DeepSeek-V4-Pro-NVFP4
```

### 5.4 Platform configuration

Per-cluster settings live in `configs/<cluster>/system.sh`. Copy the reference
[`configs/snellius/h100/system.sh`](configs/snellius/h100/system.sh) and adjust for your platform
(on ROCm, use `--rocm` in `CONTAINER_EXEC` and your fabric NIC in `NCCL_SOCKET_IFNAME`):

| Variable | Meaning / Snellius H100 reference |
|----------|-----------------------------------|
| `GPUS_PER_NODE` | GPUs per node (`4`); serve configs needing more span nodes via Ray |
| `SBATCH_ARGS` | `-p gpu_h100` (your GPU partition / account / QOS) — used for the **worker** jobs |
| `ORCH_PYTHON` | Host interpreter for the orchestrator (≥ 3.7). Default `python3`; the Snellius reference sets `/usr/bin/python3.11`, since its `python3` is 3.9 |
| `ORCH_SBATCH_ARGS` | `-p rome` (a **CPU** partition). The orchestrator only submits workers and polls `squeue`, so it requests no GPU. It defaults to `SBATCH_ARGS`, which is a GPU partition — and many schedulers reject a GPU-less job there ("You should request at least one GPU when running a job in a GPU partition"), so set this explicitly. Giving it a GPU instead would idle one for the length of the campaign. |
| `GPU_REQUEST_TMPL` | GPU-per-node request; `{n}` = per-node count. Default `--gpus-per-node={n}`; use `--gres=gpu:{n}` for gres-based schedulers |
| `HF_HOME` | shared model cache, readable from every node |
| `CONTAINER_EXEC` | `apptainer exec --nv --bind /path/to/shared …latest.sif` (NVIDIA) / `--rocm …` (AMD) |




Container-runtime flags and any system-level environment required for correct operation (NIC
selection on a multi-node fabric, `HSA_*` / RCCL tuning on ROCm, etc.) are platform-specific and
**permitted** — see [§8.3](#83-tunable-parameters); document any such change in your submission
([§6](#6-submitting-results)). The structure of `system.sh` is identical everywhere.

**Single-node AMD.** Set `VLLM_ROCM_USE_AITER=1` in your `system.sh`. DeepSeek-V4-Pro does not
start on ROCm without it in the releases tested, and the failure is not always self-explanatory:

- **v0.29.0:** `Sparse attention indexer ROCm path is only supported on AITER. Please enable aiter
  with VLLM_ROCM_USE_AITER=1`
- **v0.30.0:** `CUDA error: operation not permitted when stream is capturing` — which does not
  mention AITER and reads like a CUDA-graph problem. Check this variable before debugging graph
  capture.

**Multi-node AMD.** The multi-node path (Ray over the allocation) is vendor-agnostic — RCCL honours
the `NCCL_*` env vars and the container runs the official ROCm image unmodified. On ROCm the
CUDA-only compiled-graph transport (CuPy) is skipped and Ray falls back to RCCL collectives; if a
multi-node ROCm serve fails to bring up, set `VLLM_USE_RAY_COMPILED_DAG=0` in your `system.sh`.
Validate multi-node AMD on your hardware; if it cannot run due to a bug in this harness, see
[§5.6](#56-when-a-run-fails).

The serve- and bench-parameter JSON files are the **scored configuration** and are authored by the
canonical generator [`scripts/gen_configs.py`](scripts/gen_configs.py). Do not edit them except
where [§8](#8-rules) explicitly permits; see [`configs/README.md`](configs/README.md) for the
config layout.

### 5.5 Launching a sweep

One command runs the full sweep for a cluster, then plots and scores:

```bash
./run.sh snellius/h100              # submit the orchestrator as a SLURM job
./run.sh snellius/h100 --local      # run the orchestrator in this shell (login/interactive node)
./run.sh snellius/h100 --dry-run    # preview worker allocations; submits nothing, needs no GPUs
```

The orchestrator is a lightweight coordinator (it only `sbatch`es and polls workers), so it can run
either as a submitted job (default) or directly in your shell with `--local` — handy on a login or
interactive node outside an allocation. The GPU **workers** are always SLURM jobs.

As a submitted job it goes to `ORCH_SBATCH_ARGS` ([§5.4](#54-platform-configuration)), which must
name a **CPU** partition — it requests no GPU, and a GPU partition will usually refuse it. Prefer
this over `--local` for a real campaign: with `--local` the orchestrator must survive as long as the
run does, and if your shell dies the workers keep going.

`<cluster>` selects `configs/<cluster>/`. Override `HF_HOME`, `SBATCH_ARGS`, `CONTAINER_EXEC`,
etc. via the environment. [`run.sh`](run.sh) submits the orchestrator
[`scripts/vllm_sweep_parallel.sh`](scripts/vllm_sweep_parallel.sh), which runs
[`src/sweep_parallel.py`](src/sweep_parallel.py) to:

1. Merge the serve/bench params and pair them by `--link-vars` (`model=model, max_num_seqs=_n`),
   so each layout is matched only with the operating points derived from *its own* N.
2. Submit **one worker per layout** — one SLURM job that runs all of that layout's measurements.
   GPUs per job = `TP × PP`. A layout needing more GPUs than one node holds is served across
   nodes via a Ray cluster the worker starts with `srun` (this is how DeepSeek-V4-Pro spans four
   4-GPU nodes on the H100 reference).
3. Each worker copies its datasets to `/dev/shm`, starts `vllm serve` once, runs the layout's
   measurements in ascending concurrency (`low` before `high`), and writes a `summary.json` per
   cell.
4. Poll SLURM until all workers finish, then generate the Pareto plots and score files.

A worker whose engine dies or wedges (e.g. an out-of-memory crash at high concurrency) aborts
itself promptly and releases its nodes — it does **not** hang to walltime. The cells it had
already finished keep their results; the failed cell and any after it have none and must be
re-run ([§6](#6-submitting-results)). Running `high` last means a crash there costs only the
`high` cells.

### 5.6 When a run fails

If the suite does not run on your platform, there are two sanctioned paths — never silently work
around the harness:

1. **Your system needs different settings.** You may change **system/operational** flags to make it
   run — interconnect/NIC selection (`NCCL_SOCKET_IFNAME`), `HSA_*` / RCCL env, container-runtime
   flags, cache locations, timeouts, the Ray transport knob (`VLLM_USE_RAY_COMPILED_DAG`), etc.
   These are permitted ([§8.3](#83-tunable-parameters)) — they affect operability, not the
   numerical result. **Document every such change in your submission** ([§6](#6-submitting-results)).
   You may **not** change any fixed parameter ([§8.1](#81-fixed-parameters)) to work around a failure.

2. **The harness or spec itself is broken** (a bug in this code, not your system). Do not patch
   around it. Escalate so it is fixed centrally and all parties run the same corrected code:
   contact **EuroHPC**, who coordinates with the benchmark maintainers to fix the repository.

---

## 6. Submitting Results

The deliverable is **`scores.csv`** — the file `run.sh` writes at the end of a run. It is
self-describing: one row per cell with

```
model, input_len, output_len, point, max_concurrency, tokens_per_gpu, tokens_per_user,
tensor_parallel_size, pipeline_parallel_size, max_num_seqs, n_layouts, checkpoint
```

`model` is the base model the cell is scored under; `checkpoint` is the checkpoint that produced it
(the base, or a permitted FP4 expert checkpoint — [§3](#3-models--precision-policy)).

so it conveys exactly what was run, at what concurrency, on which layout. **A complete submission
covers every cell** — each workload shape at each operating point. A missing cell makes the
submission incomplete rather than low-scoring, so re-run any cell that failed.

Submit, alongside `scores.csv`:
- Each layout's **`serve.log`**, the engine's own record of how each layout was started and
  what it reported ([§4](#4-workloads--datasets)).
- A **deviations note** — any system/operational flag you changed to make the suite run on your
  platform ([§5.6](#56-when-a-run-fails)); "none" if unchanged.
- The **container image** identity (vendor + exact vLLM release, e.g. `v0.30.0`) and confirmation that **no fixed
  parameter ([§8.1](#81-fixed-parameters)) was overridden**.
- If a permitted FP4 expert checkpoint was used: its repository and revision, and its FP4 format
  ([§3](#3-models--precision-policy)).

Optional supporting evidence (kept under `results/<timestamp>/` for audit, not required for
scoring): `scores.json`, the plots, the per-cell `summary.json` files, and the `slurm/worker_*`
logs. Re-plot or re-score an existing run without re-benchmarking:

```bash
RESULTS=results/<timestamp>
EXEC="apptainer exec --nv --bind /path/to/shared containers/latest.sif"   # --rocm on AMD
$EXEC python3 src/utils/plot_pareto.py "$RESULTS" --fig-by=model --curve-by=custom-output-len
$EXEC python3 src/utils/score.py "$RESULTS"
```

---

## 7. Scoring

The final score is in **(0, 1]**, where **1.0** is the best result observed across all
submissions for a given cell.

### Step 1 — tok/s/GPU per cell

A **cell** is one (workload shape, operating point) pair. For each cell, the submission's score is
the **best tok/s/GPU achieved across every parallelism layout that was run**:

```
cell(party, shape, point) = max over layouts of  tokens_per_gpu
```

Taking the maximum *is* the pooling across layouts: a submission may run several layouts and each
cell is credited with its best result. Note that this is an envelope — different cells may be won
by different layouts, so the reported values need not all come from one configuration.

This is computed by [`src/utils/score.py`](src/utils/score.py), which writes the raw values to
`scores.csv` / `scores.json`. The plots from [`src/utils/plot_pareto.py`](src/utils/plot_pareto.py)
are a visual sanity check only and play no part in the score.

### Step 2 — Normalisation across submissions

For each cell, each submission's value is divided by the best value achieved on that same cell:

```
norm(party, shape, point) = cell(party, shape, point) / max_p cell(p, shape, point)   ∈ (0, 1]
```

Only like-for-like measurements are compared: the same shape at the same operating point.

**Normalisation happens per cell, before any averaging.** The prefill-bound shape's raw throughput
is roughly an order of magnitude above the decode-bound shape's, so averaging raw values would let
one shape drown out the other entirely.

### Step 3 — Aggregation

```
final_score(party) = mean over all cells of  norm(party, shape, point)
```

Every cell contributes equally. Equivalently — and this is how results are usually presented — the
per-point scores are the mean of that point's shapes, and the final score is the mean of the
per-point scores.

Steps 2 and 3 are performed by the organisers across all submissions; `scores.csv` deliberately
contains **raw** values, in a flat schema designed to be concatenated across submissions for
exactly this purpose.

---

## 8. Rules

Every parameter is **Fixed**, **Operational**, **Tunable**, or **Prohibited**:

| | Category | Meaning |
|:-:|---------|---------|
| 🔒 | **Fixed** | Enforced by the harness; identical on all platforms. Overriding it invalidates the submission. |
| ⚙️ | **Operational** | Harness mechanics (I/O, process layout, timeouts). Not a scoring lever; affects neither the numerical output nor the comparison. |
| ✅ | **Tunable** | Set freely per platform. Affects only *how fast* the same computation runs — not the result. Asymmetry between platforms is expected and fair. |
| ⛔ | **Prohibited** | Must not be used. Breaks comparability, introduces numerical asymmetry, or violates the container requirement. |

**Governing principle:** model weights, numerical precision, KV-cache precision, and workload
shape must be identical across platforms — the one exception being the FP4 expert checkpoints of
[§3](#3-models--precision-policy). Every choice that affects only *how quickly* the same
computation executes — kernel selection, memory layout, fusion, communication backend, system
tuning — is tunable.

### 8.1 Fixed parameters

Set by the harness ([`scripts/vllm_sweep_parallel.sh`](scripts/vllm_sweep_parallel.sh)); must not
be overridden:

| Parameter | Required value | Rationale |
|-----------|---------------|-----------|
| Model checkpoint | `deepseek-ai/DeepSeek-V4-Pro`, or a permitted FP4 expert checkpoint ([§3](#3-models--precision-policy)) | A different checkpoint is a different model. |
| Weight and activation precision | The base checkpoint's, except FP4 in the experts ([§3](#3-models--precision-policy)) | No accuracy term in the score; free precision would reward quality loss. |
| `--kv-cache-dtype` | `fp8` | FP8 KV cache trades accuracy for capacity; fixed symmetrically. |
| `--max-model-len` | `9216` | All datasets fit within this budget. |
| `--max-num-batched-tokens` | `9216` | Any full-length prompt prefills in one pass; equal scheduling across platforms. |
| `--enable-chunked-prefill` | enabled | Identical scheduling behaviour. |
| `--no-enable-prefix-caching` | caching off | Prefix-cache hits would skip computation and inflate throughput. |
| `--skip-chat-template` | enabled | Raw-token prompts; no chat formatting. |
| `--request-rate` | `inf` | Open-loop: the concurrency ceiling governs load, not an arrival rate. |
| `--ignore-eos` | enabled | Forces generation to the prescribed output length. |
| `--percentile-metrics` | `ttft,tpot,itl,e2el` | Uniform recorded metric set. |
| Operating points | `POINTS` in [`scripts/gen_configs.py`](scripts/gen_configs.py) | The set of points, and each point's position relative to N, is fixed. N itself is tunable ([§8.4](#84-parallelism)). |
| `max_concurrency` | The point's value | Defines the load level being measured. |
| Dataset files | `datasets/*.jsonl` | Frozen inputs ([§4](#4-workloads--datasets)). |

### 8.2 Operational parameters

Harness mechanics — set uniformly by the harness, **not scoring levers**. Do not change them;
they do not affect the numerical result or the comparison:

`--download-dir`, `--safetensors-load-strategy`, `--api-server-count`, `--backend=vllm`,
`--dataset-name=custom`, `--save-result`, `--ready-check-timeout-sec`, `--num-prompts` (set to
`2 × max_concurrency`), and the DeepSeek-V4 load requirements `--trust-remote-code` /
`--tokenizer-mode`.

### 8.3 Tunable parameters

Each selects *how* the computation runs without changing *what* is computed. Use your platform's
optimal implementation — asymmetry is **expected and fair**, as it reflects the software ecosystem
being evaluated alongside the hardware.

| Domain | What is tunable | Basis for fairness |
|--------|-----------------|--------------------|
| Attention backend | The platform's optimal attention kernel. | Same attention, different kernel — mathematically equivalent. |
| Graph capture & compilation (`-cc.mode`, `cudagraph_mode`, torch.compile / inductor) | Capture mode and fusion strategy. | Fusion does not alter results (within float rounding). |
| MoE & linear kernels (`--moe-backend`, `--linear-backend`) | The platform's optimal GEMM/MoE kernels. | Kernel selection on identical weights. |
| Kernel-selection environment variables (`VLLM_*`, e.g. `VLLM_ROCM_USE_AITER`) | Env vars that choose between implementations already in the official image. | Same class as `--moe-backend` / attention backend; an env var that alters precision, quantisation or caching behaviour falls under [§8.5](#85-prohibited) instead. |
| KV-cache memory layout (`--block-size`) | Block size and cache layout. | Memory layout only. |
| Collective / all-to-all backend (MoE) | The platform's native collective-communication kernels. | Kernel/transport selection — same class as `--moe-backend` / `-cc.mode`. |
| `--gpu-memory-utilization` | Fraction of VRAM vLLM may use. | Different GPU memory sizes warrant different values to maximise KV cache without OOM. |
| **System / operational environment** | Settings required for correct operation: network-interface selection (e.g. `NCCL_SOCKET_IFNAME`), interconnect env, container-runtime flags, engine/ready timeouts. | Tuning software side of the ecosystem for correct performance on the target system. |

> The Snellius H100 reference sets `-cc.mode=3 -cc.cudagraph_mode=FULL_DECODE_ONLY` for
> DeepSeek-V4-Pro (its default compilation does not serve on the TP4×PP4 layout). This is a
> per-platform compilation choice, so it lives in that model's config, not the shared harness
> command.

### 8.4 Parallelism

The parallelism **layout is tunable**: each vendor sizes it to their hardware (GPU count and memory
per GPU). `TP × PP` is the tok/s/GPU denominator, so additional GPUs are rewarded only through the
larger KV cache and lower per-GPU load they provide. The shipped `serve-params.json` carry the
Snellius H100 reference's choices; adjust them for your platform.

**Any number of layouts may be submitted — there is no limit.** Test whatever layouts of the model
your hardware supports. Add entries to the model's `parallels` list in
[`scripts/gen_configs.py`](scripts/gen_configs.py) and regenerate; each layout carries its own N,
is run at its own operating points, and each cell is credited with the best layout's result
([§7](#7-scoring)). Because `TP × PP` is the tok/s/GPU denominator, a layout that spends more GPUs
has to earn them back through throughput, so there is nothing to bound.

By default every layout runs every operating point. A layout may instead declare
`"_points": [...]` (labels from `POINTS`) in its `parallels` entry to run only those points — for
example a layout that cannot win the `low` cells. Every cell must still be run by at least one
layout; the generator refuses a configuration that leaves one uncovered.

| Parameter | Rule |
|-----------|------|
| `--tensor-parallel-size` | Tunable — choose per your hardware. Counts toward the `TP × PP` denominator. |
| `--pipeline-parallel-size` | Tunable — combine with TP to fit the model across your GPUs/nodes. Counts toward the denominator. |
| `--enable-expert-parallel` | Tunable; typically required for the MoE models. Must work within the official container. |
| `--data-parallel-size` | Outside the swept configuration — GPU scaling uses TP and PP only, kept this way to bound the search space. |
| `max_num_seqs` (**N**) | Tunable, **one per layout**. It is the maximum batch size the deployment sustains, and is set on the server for every operating point. See [§4](#4-workloads--datasets) for how to choose it; it is self-policing, so declare your true maximum. |

### 8.5 Prohibited

| Parameter / method | Reason |
|--------------------|--------|
| A base image not from the official [`hub.docker.com/r/vllm/`](https://hub.docker.com/r/vllm/) org, or installing any model-execution package beyond the harness image's own deps | Reproducibility requires an official vLLM image at the pinned version. |
| Any checkpoint other than the base or a permitted FP4 expert checkpoint ([§3](#3-models--precision-policy)) | A different checkpoint/precision is a different model. |
| Modifying or regenerating dataset files | Frozen inputs ([§4](#4-workloads--datasets)). |
| `--enable-prefix-caching` | Skips computation on repeated prefixes; inflates throughput. |
| `--speculative-config` (EAGLE, MTP, n-gram) | Acceptance rates are hardware-speed-dependent, making token throughput non-comparable. |
| `--performance-mode` | Opaque meta-flag that sets different parameters per platform; not auditable. |
| Weight or activation quantisation beyond [§3](#3-models--precision-policy) (outside the experts, below FP4, or at run time without a permitted checkpoint) | Equivalent to running a different precision. |

---

## 9. Edge-Case Rulings

The principle of [§8](#8-rules) governs: a change affecting only execution speed is tunable; one
altering the computation or workload shape is prohibited.

1. **Lower-precision weights or activations** — prohibited (precision is fixed to prevent
   undetectable quality loss), except FP4 in the experts through a permitted FP4 expert checkpoint
   ([§3](#3-models--precision-policy)).
2. **Prefix caching on the random-token datasets** — prohibited regardless of shared prefixes;
   the parameter is fixed off.
3. **Truncating output to inflate throughput** — impossible; `--ignore-eos` forces generation to
   `custom-output-len`.
4. **Speculative decoding** — prohibited; see [§8.5](#85-prohibited).
5. **Installing extra packages or custom kernels** — prohibited; only what ships in the official
   image at the pinned version may be used.
6. **Optimising for a single operating point** — pointless; every cell is scored and normalised
   separately ([§7](#7-scoring)), so a submission strong at one point and weak at another cannot
   average well.
7. **Declaring an N beyond what the KV cache holds** — permitted, and self-defeating. vLLM
   preempts and recomputes rather than failing, so throughput falls and the score falls with it.
   There is no guard because none is needed; `serve.log` shows the reported capacity either way.
8. **Declaring an N well below the reported capacity** — permitted, and equally self-defeating:
   it leaves throughput unclaimed at the `high` point. A large gap usually indicates a
   misconfiguration.
9. **Submitting several parallelism layouts** — permitted, with no limit on how many
   ([§8.4](#84-parallelism)); each cell takes the best. The reported cells are an envelope and need
   not all come from one layout, so a submission is not required to be a single deployable
   configuration.
10. **A vLLM version other than a stable release available at acceptance** — invalid. Nightly,
    release-candidate and development builds are not permitted. Different submissions may use
    different stable releases; the version used must be declared ([§6](#6-submitting-results)).

Configurations not addressed here are **prohibited by default**. Contact the organisers before
proceeding with anything not explicitly sanctioned.

---

## 10. Flag Reference

Condensed classification of all evaluated parameters. This document governs in all cases.

| Parameter | Category | Notes |
|-----------|:--------:|-------|
| Model checkpoint / weight and activation precision | 🔒 | Fixed, except a permitted FP4 expert checkpoint ([§3](#3-models--precision-policy)). |
| `--kv-cache-dtype fp8` | 🔒 | Fixed; symmetric across platforms. |
| `--max-model-len`, `--max-num-batched-tokens` | 🔒 | Fixed at `9216`. |
| `--enable-chunked-prefill`, `--no-enable-prefix-caching` | 🔒 | Fixed. |
| `--skip-chat-template`, `--request-rate inf`, `--ignore-eos`, `--percentile-metrics` | 🔒 | Fixed bench parameters. |
| Operating points, `max_concurrency` | 🔒 | The point set and each point's position relative to N are fixed ([§4](#4-workloads--datasets)). |
| `--download-dir`, `--safetensors-load-strategy`, `--api-server-count` | ⚙️ | Operational; harness mechanics. |
| `--backend`, `--dataset-name`, `--save-result`, `--ready-check-timeout-sec`, `--num-prompts` | ⚙️ | Operational bench mechanics. |
| `--trust-remote-code` / `--tokenizer-mode` (DeepSeek-V4) | ⚙️ | Operational model-load requirements. |
| `--tensor-parallel-size`, `--pipeline-parallel-size` | ✅ | Vendor sizes the layout to their hardware; counts toward the `TP × PP` denominator. |
| `--gpu-memory-utilization` | ✅ | Fraction of VRAM; tune per GPU memory size. |
| Attention backend | ✅ | Vendor-optimal kernel; mathematically equivalent. |
| `-cc.mode`, `cudagraph_mode`, torch.compile / inductor | ✅ | Kernel fusion / capture; no change to results. |
| `--moe-backend`, `--linear-backend`, collective / all-to-all backend | ✅ | Kernel/transport selection on identical weights. |
| `--block-size` | ✅ | Memory layout only. |
| `--enable-expert-parallel` | ✅ | Tunable; identical both sides; official image only. |
| System / network environment (`NCCL_SOCKET_IFNAME`, runtime flags, …) | ✅ | Required for correct operation; affects speed, not results. |
| Kernel-selection env vars (`VLLM_*`), e.g. `VLLM_ROCM_USE_AITER=1` | ✅ | Select among kernels shipped in the official image; install nothing and do not change precision or what is computed. Some are required on certain platforms (see [§5.4](#54-platform-configuration)). |
| `--data-parallel-size` | — | Outside the swept configuration (scale with TP/PP). |
| `--speculative-config` | ⛔ | Hardware-speed-dependent acceptance rate. |
| `--performance-mode` | ⛔ | Sets different parameters per platform; not auditable. |
