# LLM Inference Benchmark — Cross-Hardware Evaluation Suite

This document is the authoritative specification for the benchmark. It defines what is
measured, how to reproduce a result, which parameters are fixed, and how submissions are
scored. **Where this document is silent, the configuration is prohibited by default**; contact
the organisers before deviating from any stated requirement.

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
- [Appendix A — Draft: vendor FP4 activation quantisation (not in force)](#appendix-a--draft-vendor-fp4-activation-quantisation-not-in-force)

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
therefore measures **throughput per GPU at several operating points spanning both regimes**
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
| **tok/s/user** | `total_token_throughput / max_concurrency` | Per-user interactivity | Reported |

**tok/s/GPU is the scored quantity at every operating point.** Dividing by `TP × PP` keeps
hardware cost in the denominator, so additional GPUs are rewarded only through the larger KV
cache and lower per-GPU load they buy, not through their raw count.

tok/s/user is recorded alongside for interpretation, but is not scored. At a fixed concurrency
the two are proportional (`tok/s/user = total_token_throughput / concurrency`), so scoring both
would count the same measurement twice; and at the highest operating point, scoring tok/s/user
would reward a *smaller* KV cache, since it divides by the concurrency the hardware can sustain.

Each run produces one measurement for a given `(model, workload shape, parallelism layout,
operating point)`. That tuple is a **cell**; every cell is scored independently
([§7](#7-scoring)).

---

## 3. Models & Precision Policy

One model is evaluated:

| Model | Checkpoint | Precision | Architecture |
|-------|------------|-----------|--------------|
| **DeepSeek V4 Pro** | `deepseek-ai/DeepSeek-V4-Pro` | FP4+FP8 mixed (native) | MoE, 1.6T total / 49B active |

**Precision is fixed and may not be substituted.** The score contains no accuracy term, so a free
choice of precision would reward throughput gained by degrading output quality — gaming the metric
cannot detect.

DeepSeek V4 Pro is trained with MoE expert weights in FP4 and attention/router/normalisation
weights in FP8. FP4 encodings differ across vendors (block-16 with FP8 scales vs block-32 with
power-of-two scales), so a byte-identical checkpoint is not possible — each platform runs its own
native FP4 build of the base weights.

### DeepSeek V4 Pro layout and FP4 accuracy gate

Every platform loads the base `deepseek-ai/DeepSeek-V4-Pro` checkpoint and uses tensor-,
pipeline-, and expert-parallelism ([§8.4](#84-parallelism)), sized to the model's
~1.6T-parameter footprint (≈16 GPUs). Lay it out to match the node's GPU count and memory: a
single node where enough high-bandwidth GPUs are available, otherwise across nodes (the Snellius
H100 reference serves it as **TP4 × PP4 across four 4-GPU nodes** via Ray).

---

## 4. Workloads & Datasets

### Datasets are fixed inputs — download, never regenerate

The benchmark datasets are vLLM `CustomDataset` JSONL files, one per workload shape. **Download
them from the organisers:**

> **https://surfdrive.surf.nl/s/MjZ7zQ2ZrZY7nnX**

They **must not be regenerated, reshuffled, truncated, re-tokenised, or substituted.** They are
deterministic random-token corpora (seed `0`, `--range-ratio 0`), so every prompt has an exact
token count, generated with the model's own tokeniser.

Place the files under `datasets/`. The harness copies each one to **`/dev/shm`** before serving
(local RAM disk, so every worker reads it at memory speed); the bench configs reference the
`/dev/shm/...` path. The `n1024` / `n32768` / `n131072` filename suffix is the **prompt count** —
**the same dataset at different sizes** (the larger files extend the smaller ones, so they are
consistent). Each run needs `num_prompts = 2 × max_concurrency`, and
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
| `mid` | N / 2 | Partially loaded |
| `high` | N | Saturated — peak per-GPU efficiency |

`max_concurrency` (client) is set to the point's value; `max_num_seqs` (server) is set to **N for
every point**, so one deployment configuration is measured under different load levels and client
load is the only variable that moves.

The point set is defined once, in `POINTS` in
[`scripts/gen_configs.py`](scripts/gen_configs.py). Nothing else in the harness or this document
encodes how many points there are.

#### Choosing N

**N is tunable per parallelism layout** and declared by the vendor in `serve-params.json` (see
[§8.4](#84-parallelism)). It is not a free scoring lever: tok/s/GPU at the `high` point rises with
N while throughput still scales, and *falls* once N exceeds what the KV cache holds and vLLM
begins preempting. **Declaring your true maximum is also your best score**, so no harness
enforcement is needed.

You do not need to search for it. Start the layout once and read vLLM's own startup line:

```
Maximum concurrency for 9216 tokens per request: 42.65x
```

Use that value, or the nearest lower power of two, as N. The harness captures this line to
`serve.log` beside the results, which is submitted as evidence ([§6](#6-submitting-results)).
Because both shapes have identical peak KV footprint, **one N covers both**.

The scored configuration space per submission is:

```
1 model  ×  2 workload shapes  ×  {vendor's parallelism layouts}  ×  {operating points}
```

---

## 5. Running the Benchmark

### 5.1 System requirements

| Requirement | Detail |
|-------------|--------|
| Job scheduler | **SLURM (required).** One worker job per `(serve, bench)` pair; multi-node serves use `srun` + Ray. The orchestrator may run as a submitted job or locally with `--local` (see §5.5). |
| Container runtime | Apptainer / Singularity, or Podman. Workers run exclusively inside the official image. |
| Official base image | Any official vLLM image from [`hub.docker.com/r/vllm/`](https://hub.docker.com/r/vllm/) — e.g. `vllm/vllm-openai` (CUDA) or `vllm/vllm-openai-rocm` (ROCm) — at a stable release (see below). Images from any other source are not permitted ([§8.5](#85-prohibited)). |
| Accelerators | NVIDIA or AMD. Single-node and multi-node (NVIDIA validated; AMD single-node validated, multi-node AMD — see §5.4). |
| Host Python | ≥ 3.11 (orchestrator only — runs on the host for SLURM access). |
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
```

### 5.4 Platform configuration

Per-cluster settings live in `configs/<cluster>/system.sh`. Copy the reference
[`configs/snellius/h100/system.sh`](configs/snellius/h100/system.sh) and adjust for your platform
(on ROCm, use `--rocm` in `CONTAINER_EXEC` and your fabric NIC in `NCCL_SOCKET_IFNAME`):

| Variable | Meaning / Snellius H100 reference |
|----------|-----------------------------------|
| `GPUS_PER_NODE` | GPUs per node (`4`); serve configs needing more span nodes via Ray |
| `SBATCH_ARGS` | `-p gpu_h100` (your GPU partition / account / QOS) — used for the **worker** jobs |
| `ORCH_SBATCH_ARGS` | `-p rome` (a **CPU** partition). The orchestrator only submits workers and polls `squeue`, so it requests no GPU. It defaults to `SBATCH_ARGS`, which is a GPU partition — and many schedulers reject a GPU-less job there ("You should request at least one GPU when running a job in a GPU partition"), so set this explicitly. Giving it a GPU instead would idle one for the length of the campaign. |
| `GPU_REQUEST_TMPL` | GPU-per-node request; `{n}` = per-node count. Default `--gpus-per-node={n}`; use `--gres=gpu:{n}` for gres-based schedulers |
| `HF_HOME` | shared model cache, readable from every node |
| `CONTAINER_EXEC` | `apptainer exec --nv --bind /path/to/shared …latest.sif` (NVIDIA) / `--rocm …` (AMD) |

Container-runtime flags and any system-level environment required for correct operation (NIC
selection on a multi-node fabric, `HSA_*` / RCCL tuning on ROCm, `VLLM_ROCM_USE_AITER=1`, etc.)
are platform-specific and **permitted** — see [§8.3](#83-tunable-parameters); document any such
change in your submission ([§6](#6-submitting-results)). The structure of `system.sh` is
identical everywhere.

**Single-node AMD.** Set `VLLM_ROCM_USE_AITER=1` in your `system.sh`. DeepSeek-V4-Pro does not
start on ROCm without it in the releases tested, and the failure is not always self-explanatory:

- **v0.29.0:** `Sparse attention indexer ROCm path is only supported on AITER. Please enable aiter
  with VLLM_ROCM_USE_AITER=1`
- **v0.30.0:** `CUDA error: operation not permitted when stream is capturing` — which does not
  mention AITER and reads like a CUDA-graph problem. Check this variable before debugging graph
  capture.

Two container-runtime notes if you use podman rather than apptainer: AITER JIT-compiles into its
own `dist-packages` directory and Triton caches under `$HOME`, so the image tree must be
writable — rootless podman's default user namespace already gives you this, but `--userns=keep-id`
does not, and the engine then dies with `[Errno 13] Permission denied` on `aiter/jit/flydsl_cache`.
Bind-mounting that cache directory out to the host is worth it as well, since each measurement
starts a fresh engine and would otherwise recompile.

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

### 5.5 Launching a run

One command runs every cell for a cluster, then plots and scores:

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
run does, and if your shell dies the workers keep going (SIGTERM is deliberately not propagated to
them) but nothing polls them or runs the plot and score at the end — you would re-run `score.py` by
hand afterwards.

`<cluster>` selects `configs/<cluster>/`. Override `HF_HOME`, `SBATCH_ARGS`, `CONTAINER_EXEC`,
etc. via the environment. [`run.sh`](run.sh) submits the orchestrator
[`scripts/vllm_sweep_parallel.sh`](scripts/vllm_sweep_parallel.sh), which runs
[`src/sweep_parallel.py`](src/sweep_parallel.py) to:

1. Merge the serve/bench params and pair them by `--link-vars` (`model=model, max_num_seqs=_n`),
   so each layout is matched only with the operating points derived from *its own* N.
2. Submit **one worker per (layout, shape, operating point)** — one SLURM job per measurement.
   GPUs per job = `TP × PP`. A layout needing more GPUs than one node holds is served across
   nodes via a Ray cluster the worker starts with `srun` (this is how DeepSeek-V4-Pro spans four
   4-GPU nodes on the H100 reference).
3. Each worker copies its dataset to `/dev/shm`, starts `vllm serve`, runs its single bench, and
   writes `summary.json` plus the engine's own `serve.log`.
4. Poll SLURM until all workers finish, then generate the plots and score files.

**Every measurement gets a freshly started engine.** Sharing one server across several benches
would be cheaper, but back-to-back runs contaminate each other — allocator fragmentation after a
saturated run, lazy CUDA-graph and kernel warm-up that only the first bench pays, page-cache
warmth, and GPU clock and thermal state all differ between the first and last bench on one
engine. Those effects depend on run order and cannot be detected afterwards, so the model load is
paid per measurement deliberately.

A worker whose engine dies or wedges aborts itself promptly and releases its nodes — it does
**not** hang to walltime. Because each cell is its own job, a failure costs exactly that cell;
re-running it is just re-submitting one job.

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
tensor_parallel_size, pipeline_parallel_size, max_num_seqs, n_layouts
```

so it conveys exactly what was run, at what concurrency, on which layout. **A complete submission
covers every cell** — each workload shape at each operating point. A missing cell makes the
submission incomplete rather than low-scoring, so re-run any cell that failed.

Submit, alongside `scores.csv`:
- Each layout's **`serve.log`**, so the declared N can be checked against the KV capacity the
  engine actually reported ([§4](#4-workloads--datasets)).
- A **deviations note** — any system/operational flag you changed to make the suite run on your
  platform ([§5.6](#56-when-a-run-fails)); "none" if unchanged.
- The **container image** identity (vendor + exact vLLM release, e.g. `v0.30.0`) and confirmation that **no fixed
  parameter ([§8.1](#81-fixed-parameters)) was overridden**.

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
shape must be identical across platforms. Every choice that affects only *how quickly* the same
computation executes — kernel selection, memory layout, fusion, communication backend, system
tuning — is tunable.

### 8.1 Fixed parameters

Set by the harness ([`scripts/vllm_sweep_parallel.sh`](scripts/vllm_sweep_parallel.sh)); must not
be overridden:

| Parameter | Required value | Rationale |
|-----------|---------------|-----------|
| Model checkpoint | Per [§3](#3-models--precision-policy) | A different checkpoint is a different model. |
| Weight precision | Native FP4+FP8 | No accuracy term in the score; free precision would reward quality loss. |
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
`2 × max_concurrency`), the dataset-size selection that follows from it, and the DeepSeek-V4 load
requirements `--trust-remote-code` / `--tokenizer-mode`.

### 8.3 Tunable parameters

Each selects *how* the computation runs without changing *what* is computed. Use your platform's
optimal implementation — asymmetry is **expected and fair**, as it reflects the software ecosystem
being evaluated alongside the hardware.

| Domain | What is tunable | Basis for fairness |
|--------|-----------------|--------------------|
| Attention backend | The platform's optimal attention kernel. | Same attention, different kernel — mathematically equivalent. |
| Graph capture & compilation (`-cc.mode`, `cudagraph_mode`, torch.compile / inductor) | Capture mode and fusion strategy. | Fusion does not alter results (within float rounding). |
| MoE & linear kernels (`--moe-backend`, `--linear-backend`) | The platform's optimal GEMM/MoE kernels. | Kernel selection on identical weights. |
| KV-cache memory layout (`--block-size`) | Block size and cache layout. | Memory layout only. |
| Collective / all-to-all backend (MoE) | The platform's native collective-communication kernels. | Kernel/transport selection — same class as `--moe-backend` / `-cc.mode`. |
| `--gpu-memory-utilization` | Fraction of VRAM vLLM may use. | Different GPU memory sizes warrant different values to maximise KV cache without OOM. |
| **System / operational environment** | Settings required for correct operation: network-interface selection (e.g. `NCCL_SOCKET_IFNAME`), interconnect env, container-runtime flags, engine/ready timeouts. | Tuning software side of the ecosystem for correct performance on the target system. |

> The Snellius H100 reference sets `-cc.mode=3 -cc.cudagraph_mode=FULL_DECODE_ONLY` for
> DeepSeek-V4-Pro (its default compilation does not serve on the TP4×PP4 layout). This is a
> per-platform compilation choice, so it lives in that model's config, not the shared harness
> command.

> **ROCm:** `VLLM_ROCM_USE_AITER=1` is permitted. It selects kernels already present in the
> official image and installs nothing, so [§8.5](#85-prohibited) and ruling 5 do not apply. See
> [§10](#10-flag-reference) and, for the symptoms when it is missing, [§5.4](#54-platform-configuration).

### 8.4 Parallelism

The parallelism **layout is tunable**: each vendor sizes it to their hardware (GPU count and memory
per GPU). `TP × PP` is the tok/s/GPU denominator, so additional GPUs are rewarded only through the
larger KV cache and lower per-GPU FLOPS they provide. The shipped `serve-params.json` carries the
Snellius H100 reference's choice; adjust it for your platform.

**Any number of layouts may be submitted — there is no limit.** Test whatever layouts of the model
your hardware supports. Add entries to the model's `parallels` list in
[`scripts/gen_configs.py`](scripts/gen_configs.py) and regenerate; each layout carries its own N,
is run at its own operating points, and each cell is credited with the best layout's result
([§7](#7-scoring)). Because `TP × PP` is the tok/s/GPU denominator, a layout that spends more GPUs
has to earn them back through throughput, so there is nothing to bound.

| Parameter | Rule |
|-----------|------|
| `--tensor-parallel-size` | Tunable — choose per your hardware. Counts toward the `TP × PP` denominator. |
| `--pipeline-parallel-size` | Tunable — combine with TP to fit the model across your GPUs/nodes. Counts toward the denominator. |
| `--enable-expert-parallel` | Tunable; typically required for this MoE model. Expert parallelism redistributes experts across the existing `TP × PP` world, so it adds no GPUs and never changes the tok/s/GPU denominator. Must work within the official container. |
| `--data-parallel-size` | Outside the scored configuration — GPU scaling uses TP and PP only, kept this way to bound the search space. |
| `max_num_seqs` (**N**) | Tunable, **one per layout**. It is the maximum batch size the deployment sustains, and is set on the server for every operating point. See [§4](#4-workloads--datasets) for how to choose it; it is self-policing, so declare your true maximum. |

### 8.5 Prohibited

| Parameter / method | Reason |
|--------------------|--------|
| A base image not from the official [`hub.docker.com/r/vllm/`](https://hub.docker.com/r/vllm/) org, or installing any model-execution package beyond the harness image's own deps | Reproducibility requires an unmodified official vLLM image of a stable release. |
| Alternative checkpoint or precision | A different checkpoint/precision is a different model ([§3](#3-models--precision-policy)). |
| Modifying or regenerating dataset files | Frozen inputs ([§4](#4-workloads--datasets)). |
| `--enable-prefix-caching` | Skips computation on repeated prefixes; inflates throughput. |
| `--speculative-config` (EAGLE, MTP, n-gram) | Acceptance rates are hardware-speed-dependent, making token throughput non-comparable. |
| `--performance-mode` | Opaque meta-flag that sets different parameters per platform; not auditable. |
| Additional weight or activation quantisation | Equivalent to running a different precision. |

---

## 9. Edge-Case Rulings

The principle of [§8](#8-rules) governs: a change affecting only execution speed is tunable; one
altering the computation or workload shape is prohibited.

1. **Lower-precision weights for a fixed-precision model** — prohibited (precision is fixed to
   prevent undetectable quality loss).
2. **Prefix caching on the random-token datasets** — prohibited regardless of shared prefixes;
   the parameter is fixed off.
3. **Truncating output to inflate throughput** — impossible; `--ignore-eos` forces generation to
   `custom-output-len`.
4. **Speculative decoding** — prohibited; see [§8.5](#85-prohibited).
5. **Installing extra packages or custom kernels** — prohibited; only what ships in the official
   image of the stable release used may be used.
6. **Optimising for a single operating point** — pointless; every cell is scored and normalised
   separately ([§7](#7-scoring)), so a submission strong at one point and weak at another cannot
   average well.
7. **Declaring an N beyond what the KV cache holds** — permitted, and self-defeating. vLLM
   preempts and recomputes rather than failing, so throughput falls and the score falls with it.
   There is no guard because none is needed; `serve.log` shows the reported capacity either way.
8. **Declaring an N well below the reported capacity** — permitted, and equally self-defeating:
   it leaves throughput unclaimed at the `high` point. A large gap usually indicates a
   misconfiguration, so check `serve.log` before submitting.
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
| Model checkpoint / weight precision | 🔒 | Fixed ([§3](#3-models--precision-policy)). |
| `--kv-cache-dtype fp8` | 🔒 | Fixed; symmetric across platforms. |
| `--max-model-len`, `--max-num-batched-tokens` | 🔒 | Fixed at `9216`. |
| `--enable-chunked-prefill`, `--no-enable-prefix-caching` | 🔒 | Fixed. |
| `--skip-chat-template`, `--request-rate inf`, `--ignore-eos`, `--percentile-metrics` | 🔒 | Fixed bench parameters. |
| Operating points, `max_concurrency` | 🔒 | The point set and each point's position relative to N are fixed ([§4](#4-workloads--datasets)). |
| `max_num_seqs` (**N**) | ✅ | The layout's max batch size; vendor-declared, one per layout ([§8.4](#84-parallelism)). |
| `--download-dir`, `--safetensors-load-strategy`, `--api-server-count` | ⚙️ | Operational; harness mechanics. |
| `--backend`, `--dataset-name`, `--save-result`, `--ready-check-timeout-sec`, `--num-prompts` | ⚙️ | Operational bench mechanics. |
| `--trust-remote-code` / `--tokenizer-mode` (DeepSeek-V4) | ⚙️ | Operational model-load requirements. |
| `--tensor-parallel-size`, `--pipeline-parallel-size` | ✅ | Vendor sizes the layout to their hardware; counts toward the `TP × PP` denominator. |
| `--gpu-memory-utilization` | ✅ | Fraction of VRAM; tune per GPU memory size. |
| Attention backend | ✅ | Vendor-optimal kernel; mathematically equivalent. |
| `-cc.mode`, `cudagraph_mode`, torch.compile / inductor | ✅ | Kernel fusion / capture; no change to results. |
| `--moe-backend`, `--linear-backend`, collective / all-to-all backend | ✅ | Kernel/transport selection on identical weights. |
| `--block-size` | ✅ | Memory layout only. |
| `--enable-expert-parallel` | ✅ | Redistributes experts across the existing `TP × PP` world; adds no GPUs to the denominator. Official image only. |
| System / network environment (`NCCL_SOCKET_IFNAME`, runtime flags, …) | ✅ | Required for correct operation; affects speed, not results. |
| `VLLM_ROCM_USE_AITER=1` | ✅ | ROCm only. Required for DeepSeek-V4 in the releases tested (v0.29.0, v0.30.0). Selects kernels inside the official image; installs nothing. |
| `--data-parallel-size` | — | Outside the scored configuration (scale with TP/PP). |
| `--speculative-config` | ⛔ | Hardware-speed-dependent acceptance rate. |
| `--performance-mode` | ⛔ | Sets different parameters per platform; not auditable. |

---

## Appendix A — Draft: vendor FP4 activation quantisation (not in force)

> **This appendix is a proposal and has no effect.** [§8.5](#85-prohibited) continues to
> prohibit additional activation quantisation. It records how such a permission would be scoped
> if adopted, so the wording is ready when the preconditions below are met.

### A.1 Proposed rule

> A vendor may quantise **activations** to FP4 for the **routed-expert GEMMs only** — tensors
> matching `*.ffn.experts.*`.
>
> **Every tensor outside `*.ffn.experts.*` must keep exactly the dtype of the corresponding tensor
> in the base `deepseek-ai/DeepSeek-V4-Pro` checkpoint**, for both weights and activations.
>
> Activation scales must be computed **dynamically**. Static or calibrated activation scales are
> prohibited.
>
> The FP4 encoding must be one of:
> - **MXFP4** — block of 32 values, E8M0 (power-of-two) scale; or
> - **NVFP4** — block of 16 values, FP8-E4M3 scale, plus a per-tensor FP32 scale.

The normative clause is the one in bold. The table below is **informative only**: it was generated
from the checkpoint's safetensors headers (layer 10) and must be regenerated, not hand-edited, if
the checkpoint changes.

| tensor pattern | base-checkpoint dtype | under this rule |
|---|---|---|
| `ffn.experts.*` | FP4 packed (`I8`) + `F8_E8M0` scale | weights unchanged; **activations FP4 permitted** |
| `ffn.shared_experts.{w1,w2,w3}` | `F8_E4M3` | unchanged |
| `attn.{wq_a, wq_b, wkv, wo_a, wo_b}`, `attn.indexer.wq_b` | `F8_E4M3` | unchanged |
| `attn.compressor.*`, `attn.indexer.compressor.*`, `attn.indexer.weights_proj` | `BF16` | unchanged |
| `ffn.gate`, `head`, all `*_norm` | `BF16` | unchanged |

### A.2 Why this boundary

- **Scope by tensor pattern, not by prose.** "The experts" is ambiguous about the shared expert,
  and the two published vendor builds already disagree on it: `amd/DeepSeek-V4-Pro-MXFP4`
  quantises `ffn.shared_experts.*` to FP4, `nvidia/DeepSeek-V4-Pro-NVFP4` leaves it FP8. That is
  a quantisation-policy difference, not a hardware one.
- **The gain is bounded.** From the checkpoint's tensor shapes, routed experts are 51.3% of
  per-token GEMM FLOPs, attention projections 38.8%, the shared expert 8.6%. At a nominal 2x FP4
  rate the theoretical ceiling is **~1.34x** with routed experts only, **~1.42x** if the shared
  expert were included. Both figures count GEMM FLOPs only — attention score computation is
  excluded — so realised gains will be lower.
- **Dynamic scaling** removes calibration as a hidden lever (the published NVIDIA build uses static
  scales, the AMD builds dynamic) and needs no calibration data to audit. Whether static scales
  are materially faster on either platform has not been measured.

### A.3 Verification

A submission using this permission would supply:

1. **`config.json` → `quantization_config`**, whose `exclude` / `ignore` lists define module
   coverage. A reviewer diffs coverage against A.1.
2. **A per-module precision map** — tensor name → weight dtype, activation dtype, scale scheme.
3. The existing **`serve.log`** per layout, which records the selected MoE backend.

Tensor dtypes can be checked independently from the safetensors headers without trusting the
vendor. Two signals that look useful are **not** reliable and should not be required:

- **The operand-dtype tuple in `serve.log`** (e.g. `'torch.float4_e2m1fn_x2',
  'torch.float4_e2m1fn_x2'`) is only printed on AITER's "no tuned config, heuristic fallback" path.
  A tuned configuration does not emit it.
- **Hardware MFMA counters** (`SQ_INSTS_VALU_MFMA_MOPS_F8` vs `..._F6F4` on gfx950). A control run
  showed the F8 and BF16 counters accurate to the digit, and an FP4xFP4 MoE registered 100% in
  F6F4 (to within ~12% of the expected FLOP count). But which counter a *mixed* FP8xFP4 operation
  increments was never measured, so the counters may not distinguish W4A8 from W4A4 — the exact
  distinction an audit needs. Counter collection on a served multi-GPU engine also crashes vLLM's
  workers.

### A.4 Preconditions before adopting

1. **A real accuracy gate.** §3's heading *"DeepSeek V4 Pro layout and FP4 accuracy gate"* has no
   gate in its body. The score has no accuracy term, so without a fixed evaluation and threshold
   this permission trades output quality for throughput undetectably.
2. **Both platforms able to execute it.** Measured on MI355X with vLLM v0.29.0 and v0.30.0, the
   DeepSeek-V4 ROCm path cannot reach FP4 activations by any checkpoint or flag:
   `DeepseekV4FP8Config.from_config` discards the Quark schema that declares FP4 activations, the
   ROCm MoE priority list is `[AITER_MXFP4_BF16, TRITON_UNFUSED]` with no W4A4 entry, and
   `aiter_mxfp4_mxfp4` is not an accepted `--moe-backend` value. Adopting the rule before that
   changes would create an asymmetry favouring whichever platform can execute it. Whether the CUDA
   path reaches W4A4 for this model has **not** been verified.

### A.5 Corrections to the earlier draft

An earlier version of this proposal, circulated informally, contained the following errors:

| earlier claim | correction |
|---|---|
| `*.attn.*`, including `compressor.*` and `indexer.*`, is FP8 | Wrong. The compressor and most indexer tensors are **BF16**; only `wq_a/wq_b/wkv/wo_a/wo_b` and `indexer.wq_b` are FP8. The normative clause is now "keep the base dtype", with the table derived from the checkpoint. |
| Weights "may be re-encoded but not re-quantised" into the native FP4 format | Not a real distinction. MXFP4 -> NVFP4 changes block size and scale format, which re-quantises the values. Only MXFP4 matches the base checkpoint byte-for-byte. Removed. |
| Hardware counters verified "accurate to the digit" and usable for audit | Only F8/BF16 were verified that closely; F6F4 to ~12%. The mixed FP8xFP4 case is unmeasured, and engine-level collection crashes vLLM. Demoted to "not reliable". |
| The `serve.log` operand-dtype tuple proves the activation dtype | Printed only on the untuned fallback path. Replaced by a required per-module precision map. |
| Including the shared expert raises the ceiling to "~1.42%" | Typo for **~1.42x**. |
| Shared-expert coverage is worth "~6%" | A difference between two theoretical ceilings, not a measurement. |
| Static scales are faster | An expectation, not measured. |
| "NVIDIA's path may well reach W4A4" | Unverified. The leading CUDA MXFP4 backends for this model are W4A8; a `ModelOptNvFp4FusedMoE` branch for NVFP4 checkpoints may be W4A4 but was not tested. |
