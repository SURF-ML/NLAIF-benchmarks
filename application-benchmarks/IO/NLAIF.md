# NL AI Factory — Storage Benchmark Strategy

## Overview

Scalable benchmark toolkit for shared parallel file system procurements on SURF-operated HPC systems. The suite measures two complementary axes of filesystem performance under controlled scaling conditions.

**IOR** is a synthetic I/O workload generator that measures aggregate bandwidth of shared parallel file systems. It drives read and write phases against user-defined access patterns, operating either file-per-process or on a shared file, at configurable transfer sizes and dataset volumes. Each IOR rule encodes a real-world I/O pattern (e.g., LLM checkpoint writes, scientific simulation random I/O) parameterized to realistic dataset sizes.

**mdtest** is a POSIX metadata operation microbenchmark. It creates, stats, and removes files and directories arranged in a configurable hierarchical tree, reporting operations per second for each phase. Each mdtest rule models a real dataset structure (e.g., ImageNet class hierarchy) and exercises the filesystem metadata path under concurrent multi-task load.

The scripts generate SLURM job files, submit them to the scheduler, parse benchmark output, and produce plot-ready data files.

## Context

The Netherlands AI Factory (NLAIF) is a European, shared AI infrastructure. The storage system is a parallel filesystem serving multiple independent tenants simultaneously, and must sustain production AI and scientific workloads at two levels of scale.

A **high-bandwidth domain (HBD)** is a self-contained GPU cluster connected by a high-speed scale-up network. The system is composed of multiple HBDs sharing common storage.

Three benchmark scopes are defined:

- **Single-HBD**: one HBD running in isolation at full capacity. Establishes the performance baseline.
- **Full-system parallel**: all nodes participating in a single job, exercising the full filesystem fabric under maximum single-tenant aggregate load.
- **Full-system concurrent**: all HBDs running independent single-HBD jobs simultaneously. Validates that per-HBD performance is preserved under multi-tenant load.

Two performance axes are measured. **Bandwidth** (IOR) covers training dataset reads and checkpoint write/restore. **Metadata throughput** (mdtest) covers file creation, stat, and removal under the directory structures of real AI training datasets. Rules are parameterised to realistic workloads: language model training at 70B and 405B scale, computer vision datasets from ImageNet to LVD, and climate reanalysis data for scientific AI.

---

## Dependencies

**Mandatory** — the suite will not run without these:

| Tool | Purpose |
|------|---------|
| `bash` | Script runtime |
| `ior` | Bandwidth benchmark executable (path set via `IOR` in system config) |
| `mdtest` | Metadata benchmark executable (path set via `MDTEST` in system config) |

**Recommended** — missing tools degrade specific functionality:

| Tool | Needed for |
|------|-----------|
| Slurm (`sbatch`) | Job submission; without it the run scripts cannot submit jobs |
| `bc`, `awk`, `sort` | Arithmetic and result post-processing in parse scripts |
| `gnuplot` | Visualisation; without it only the TSV output is produced |

---

## Architecture

Each benchmark tool (IOR, mdtest) is served by a triplet of scripts: a **run** script that generates and submits SLURM jobs, a **parse** script that extracts results into a TSV file, and a **plot** script that renders graphs from that TSV. Configuration is split into two orthogonal layers — system files and rule files — described below.

```
IO/
├── scripts/
│   ├── ior-run.sh          # Generate and submit IOR SLURM jobs
│   ├── ior-parse.sh        # Extract bandwidth results → TSV
│   ├── ior-plot.gpi        # gnuplot visualization
│   ├── mdtest-run.sh       # Generate and submit mdtest SLURM jobs
│   ├── mdtest-parse.sh     # Extract ops/s results → TSV
│   └── mdtest-plot.gpi     # gnuplot visualization
├── config/
│   ├── systems/            # Hardware-specific facts (one file per target system)
│   │   ├── snellius-rome-scratch.sh
│   └── rules/              # Workload tuning intent (one file per benchmark scenario)
│       ├── ior-smoketest.sh
│       ├── ior-llama-405b-checkpoint.sh
│       ├── ior-llama-405b.sh
│       ├── ior-llama-70b-checkpoint.sh
│       ├── ior-llama-70b.sh
│       ├── ior-scientific.sh
│       ├── mdtest-smoketest.sh
│       ├── mdtest-imagenet-1k.sh
│       ├── mdtest-imagenet-21k.sh
│       ├── mdtest-lvd-1689m-shallow.sh
│       └── mdtest-lvd-1689m-hashed.sh
└── results/                # Generated output (git-ignored)
    └── <rule>/<cluster>[…]/<timestamp>/
```

**Output structure:**

```
results/<rule>/<cluster>-<partition>-<filesystem>/<timestamp>/
├── ior.md                          # Run metadata (system, scaling, parameters)
├── ior_<mode>_n<N>.sh              # Generated SLURM script per ntasks value
├── ior_<mode>_n<N>_<jobid>.out     # Benchmark stdout
├── ior_<mode>_n<N>_<jobid>.err     # Benchmark stderr
└── ior-<mode>.dat                  # Parsed TSV: ntasks  write_MiB_s  read_MiB_s
```

**Separation of concerns:**

System files and rule files are fully orthogonal. Any rule can run on any system without modification: all hardware-specific details (filesystem paths, SLURM partition, module load commands, executable locations, memory per node) live exclusively in system files; all workload parameters (dataset size, access pattern, tree structure) live exclusively in rule files. Scaling mode and task counts are CLI flags, so the same `rule + system` pair can be executed as a single production-like run, a strong scaling sweep, or a weak scaling sweep by changing only `--scaling` and `--ntasks`.

### Extending the Suite

**New system:** copy an existing `config/systems/*.sh` and update hardware constants and filesystem paths. No changes to scripts or rules required.

**New rule:** copy an existing `config/rules/*.sh` and update size and flag parameters. No changes to scripts or systems required.

**New scaling point:** pass different `--ntasks` values on the CLI. No configuration changes required.

---

## Configuration

### System Files (`config/systems/*.sh`)

Each system file defines hardware-specific, immutable facts sourced by the run scripts at startup.

| Variable | Description |
|----------|-------------|
| `CLUSTER` | Cluster name (used in results path) |
| `PARTITION` | SLURM partition |
| `TIME` | Job time limit |
| `RUNNER` | Launcher command (`srun` or `mpirun`) |
| `FEATURES` | SLURM `--constraint` string (e.g., `vastnfs`) |
| `CPUS_PER_NODE` | Logical CPUs per node |
| `DRAM_PER_NODE` | RAM per node in GiB (used for buffer cache eviction) |
| `ENV_SETUP` | Ordered bash array of module load and environment commands |
| `IOR` | Absolute path to IOR executable |
| `MDTEST` | Absolute path to mdtest executable |
| `FILESYSTEM` | Filesystem label (used in results path) |
| `TARGET_DIR` | Working directory on the target filesystem |

Available systems:

| File | Cluster | Partition | Filesystem |
|------|---------|-----------|------------|
| `snellius-rome-scratch.sh` | Snellius | rome | scratch-shared (GPFS) |

### Benchmark Rules (`config/rules/*.sh`)

Rules encode workload intent and are sourced alongside a system file at runtime. A rule never references hardware, filesystem paths, or task counts — those are supplied by the system file and CLI flags respectively.

**IOR rule variables:**

| Variable | Description |
|----------|-------------|
| `AGGREGATE_SIZE` | Total I/O volume; interpretation depends on scaling mode (see [Scaling Modes](#scaling-modes)) |
| `BLOCK_SIZE` | Starting block size hint; auto-scaled by the runner at runtime |
| `TRANSFER_SIZE` | I/O transfer unit |
| `SEGMENT_COUNT` | Starting segment count hint; auto-scaled by the runner |
| `IOR_FLAGS` | Bash array of all IOR CLI flags encoding the access pattern |

**mdtest rule variables:**

| Variable | Description |
|----------|-------------|
| `MDTEST_BRANCH` | Directory tree branching factor |
| `MDTEST_DEPTH` | Directory tree depth |
| `MDTEST_TOTAL_ITEMS` | Total files/dirs; interpretation depends on scaling mode |
| `MDTEST_ITERATIONS` | Number of repeated benchmark phases per job |
| `MDTEST_FLAGS` | Bash array of all mdtest flags (excludes `-n`, which the runner derives from scaling mode and ntasks) |
| `TOTAL_FILES` | Informational — intended dataset file count |
| `AVG_FILE_SIZE` | Informational — intended average file size |

---

## Scaling Modes

All run scripts accept `--scaling=TYPE` and compute per-task parameters at runtime from the rule's size constant. The same rule file supports all three modes without modification.

### Strong Scaling (`--scaling=strong`)

The total I/O volume (`AGGREGATE_SIZE` for IOR; `MDTEST_TOTAL_ITEMS` for mdtest) is held **constant** across all task counts. Per-task data volume shrinks as the number of tasks increases.

```
total  = AGGREGATE_SIZE            (fixed)
per_task = total / ntasks          (decreases as ntasks grows)
```

Strong scaling measures filesystem parallelism: how effectively throughput or operation rate increases as more clients share a fixed-size workload. It is the primary mode for procurements where the objective is demonstrating a peak aggregate capability on a representative dataset.

### Weak Scaling (`--scaling=weak`)

The per-node data volume is held **constant** (`AGGREGATE_SIZE` is interpreted as per-node). The aggregate I/O volume grows proportionally with the number of nodes.

```
per_node = AGGREGATE_SIZE          (fixed)
total    = per_node × nodes        (increases as nodes grow)
```

Weak scaling measures throughput retention: whether per-node performance degrades as the aggregate client count increases. Ideal weak scaling yields constant per-node throughput regardless of system size.

### Single Run (`--scaling=single`)

One fixed task count. The total I/O volume equals `AGGREGATE_SIZE` distributed across the requested tasks. Equivalent to a strong scaling measurement at a single operating point — a production-like steady-state measurement at the intended deployment scale.

```
--ntasks=N  →  one SLURM job,  total = AGGREGATE_SIZE
```

---

## Running the Benchmarks

### CLI Reference

Both `ior-run.sh` and `mdtest-run.sh` accept the same flag interface:

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--system=PATH` | yes | — | Path to system config file |
| `--rule=PATH` | yes | — | Path to benchmark rule file |
| `--scaling=TYPE` | no | `single` | `single`, `strong`, or `weak` |
| `--ntasks=LIST` | no | mode-specific | Comma- or space-separated task counts |
| `--ppn=N` | no | `CPUS_PER_NODE` | Tasks per node; capped at `CPUS_PER_NODE` |
| `--randomize` | no | off | IOR: append `-z` (random offsets within each shard); mdtest: append `-R` (random lookup). Results land in `<rule>-randomize/` instead of `<rule>/` |
| `--interactive` | no | off | Print bare benchmark commands to stdout instead of generating SLURM scripts; bypasses `sbatch`, useful for interactive sessions or non-SLURM environments |
| `--dry-run` | no | off | Generate SLURM scripts without submitting |
| `--chain` | no | off | Chain jobs sequentially via `afterany` dependency |
| `-h` / `--help` | no | — | Print usage |

**Default `--ntasks` by scaling mode** (identical for IOR and mdtest):

| Mode | Default ntasks |
|------|---------------|
| `single` | `CPUS_PER_NODE` — one fully packed node |
| `strong` | `1×, 2×, 4×, 8×, 16× CPUS_PER_NODE` — sweep over 1–16 nodes |
| `weak` | `1×, 2×, 4×, 8×, 16× CPUS_PER_NODE` — sweep over 1–16 nodes |

`--chain` is recommended for scaling sweeps: it submits jobs with `afterany` SLURM dependencies so they run sequentially, preventing concurrent filesystem contention between measurement points.

**NLAIF protocol:** All benchmark runs use `--scaling=single`. Task counts and node assignment must always be specified explicitly via `--ppn` and `--ntasks` to pin each job to the intended hardware; do not rely on the scheduler's default node selection.

**HBD scope** — one fully packed HBD node:

```bash
./scripts/<tool>-run.sh \
  --system=config/systems/<system>.sh \
  --rule=config/rules/<rule>.sh \
  --scaling=single \
  --ppn=<ppn> \
  --ntasks=<ppn>
```

**Full-system parallel scope** — one job spanning all system nodes simultaneously:

```bash
./scripts/<tool>-run.sh \
  --system=config/systems/<system>.sh \
  --rule=config/rules/<rule>.sh \
  --scaling=single \
  --ppn=<ppn> \
  --ntasks=<ppn × total_nodes>
```

**Full-system concurrent scope** — N independent HBD-scope jobs submitted simultaneously, one per HBD, each pinned to its own nodelist:

```bash
for i in $(seq 0 $((N_HBDS - 1))); do
  ./scripts/<tool>-run.sh \
    --system=config/systems/<system>.sh \
    --rule=config/rules/<rule>.sh \
    --scaling=single \
    --ppn=<ppn> \
    --ntasks=<ppn>
done
```

Nodelist pinning must be applied per iteration so that SLURM places each job on its own HBD rather than filling the same nodes. This is typically done via per-HBD SLURM partitions, `--nodelist` directives added to the generated scripts, or a `--constraint` value unique to each HBD's hardware.

---

## Benchmark Rules

### Overview

#### Metadata (mdtest)

Sequential and randomized (`--randomize`, adds `-R`) modes are both required for every mdtest rule.

| Rule | Modeled Dataset | Scope | Concurrent | Total Files | Tree (branch × depth) | Files/dir | File Size | Extra Mode | 
|------|----------------|-------|:----------:|-------------|----------------------|-----------|-----------| -- |
| `mdtest-imagenet-21k` | ImageNet-21k — 21,841 classes | HBD | System × 0.25 | ~14 M | 21,841 × 1 | ~641 | 150 KiB | --randomize |
| `mdtest-imagenet-21k` | ImageNet-21k — 21,841 classes | HBD | System × 0.25 | ~14 M | 21,841 × 1 | ~641 | 150 KiB | |
| `mdtest-lvd-1689m-shallow` | LVD-1689M — shallow prefix | System | — | ~1.7 B | 25,000 × 1 | ~67,600 | 300 KiB | --randomize |
| `mdtest-lvd-1689m-shallow` | LVD-1689M — shallow prefix | System | — | ~1.7 B | 25,000 × 1 | ~67,600 | 300 KiB | |

#### Bandwidth (IOR)

Training dataset rules require `--randomize` (`-z`). Checkpoint rules are always sequential.

| Rule | Modeled Workload | Scope | Concurrent | Aggregate Size | Transfer Size | Access Pattern |
|------|-----------------|-------|:----------:|----------------|---------------|----------------|
| `ior-llama-70b` | LLaMA 2 70B training dataset | HBD | System x 0.25 | 3,726 GiB | 16 MiB | rand, file-per-process |
| `ior-llama-70b-checkpoint` | LLaMA 2 70B checkpoint | HBD | System × 0.25 | 700 GiB | 64 MiB | seq write + read, file-per-process |
| `ior-llama-405b` | LLaMA 3.1 405B training dataset | System | — | 27,940 GiB | 32 MiB | rand, file-per-process |
| `ior-llama-405b-checkpoint` | LLaMA 3.1 405B checkpoint | System | — | 4,050 GiB | 64 MiB | seq write + read, file-per-process |
| `ior-scientific` | ERA5 climate reanalysis (one epoch) | System | — | 54 TiB | 4 MiB | rand, shared file |

---

### `mdtest-imagenet-21k`

ImageNet-21k (~14M files, 21,841 synset classes) in its real on-disk layout: one flat directory per class, ~641 images each. This is the canonical metadata pattern for supervised image classification training — every training epoch stats and opens files from this structure. Sequential mode models dataset ingest; randomized mode (`-R`) models the shuffled dataloader, which scatters requests simultaneously across all 21,841 class directories.

Run at HBD scope (single instance) and full-system concurrent (N HBD-scope jobs, one per HBD) to measure multi-tenant MDS isolation.

**Primary metric:** file create ops/s (ingest), file stat ops/s (training).

---

### `mdtest-lvd-1689m-shallow`

LVD-1689M is Meta's DINOv3 pretraining corpus: 1.689 billion web images (~475 TiB). The shallow variant (25,000 directories, ~67,600 files/dir) models a flat layout inferred from DINOv3's 25k top-level k-means cluster hierarchy. At full-system parallel scale this is the heaviest metadata workload in the suite: 1.7 billion inodes under concurrent access, with directory entries of ~67,600 files stressing directory lookup and readdir beyond what the ImageNet rules exercise.

**Primary metric:** file create ops/s (ingest), file stat ops/s (training).

---

### `ior-llama-70b`

LLaMA 2 70B training dataset: 3,726 GiB (2T uint16 tokens). Each rank streams its own pre-tokenised shard file sequentially; `--randomize` switches to random offsets within each shard, modelling a shuffle-buffer dataloader that defeats OS readahead. Run at HBD scope and full-system concurrent. The randomized mode is required; sequential is informational.

**Primary metric:** MiB/s read throughput (random).

---

### `ior-llama-70b-checkpoint`

LLaMA 2 70B checkpoint: 700 GiB per save (FP16 weights 140 GiB + FP32 Adam states 560 GiB). Each rank writes its shard to a separate file and fsyncs before returning — no training steps run during this window. The read phase models recovery from failure. Run at HBD scope and full-system concurrent; the concurrent run directly measures whether checkpoint latency degrades when multiple HBDs save simultaneously.

**Primary metric:** MiB/s write throughput (fsynced), MiB/s read throughput.

---

### `ior-llama-405b`

LLaMA 3.1 405B training dataset: 27,940 GiB (15T uint16 tokens). Same shard-streaming pattern as the 70B rule, scaled to frontier model size. Run at full-system parallel: all nodes read simultaneously, driving maximum aggregate bandwidth demand on the filesystem fabric. Randomized mode required.

**Primary metric:** MiB/s read throughput (random).

---

### `ior-llama-405b-checkpoint`

LLaMA 3.1 405B checkpoint: 4,050 GiB per save (FP16 weights 810 GiB + FP32 Adam states 3,240 GiB). At full-system parallel scale, thousands of ranks write large sequential files concurrently — the scenario most likely to saturate OST bandwidth and expose MDS bottlenecks from simultaneous file creation. Checkpoint frequency is directly limited by write throughput: at 100 GB/s the save takes ~40 s; at 10 GB/s it takes ~7 min.

**Primary metric:** MiB/s write throughput (fsynced), MiB/s read throughput.

---

### `ior-scientific`

ERA5 climate reanalysis training at GraphCast scale: 54 TiB per epoch (58,440 timesteps × ~950 MiB/snapshot, representing 40 years at 6-hourly resolution with 228 fields at 0.25° resolution). Unlike the LLaMA rules, all dataloader workers read from a single shared file concurrently (no `-F`), generating random 4 MiB patch reads across different variables, pressure levels, and timesteps. This creates shared-OST contention absent in shard layouts and defeats OS readahead entirely. Run at full-system parallel.

**Primary metric:** MiB/s read throughput (random, shared file).

