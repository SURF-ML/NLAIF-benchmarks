# NL AI Factory — Storage Benchmarks: Instructions

This document describes how to build, configure, run and check the storage benchmarks B5, B6 and B7. The benchmarks, rules, results and reporting are defined in `README.md`; where the two documents differ, `README.md` applies.

## Contents of this directory

```
application-benchmarks/IO/
├── README.md                         # Benchmarks, rules and reporting
├── INSTRUCTIONS.md                   # This document
├── config/
│   ├── rules/
│   │   ├── io500-posix.ini           # IO500 configuration (B5)
│   │   ├── ior-scientific.sh         # ERA5 workload (B6)
│   │   ├── mdtest-lvd-1689m-shallow.sh  # LVD-1689M workload (B7)
│   │   ├── io500-smoketest.ini       # Smoke tests
│   │   ├── ior-smoketest.sh
│   │   └── mdtest-smoketest.sh
│   └── systems/
│       ├── snellius-rome-scratch.sh   # Example system files (Snellius Rome
│       └── snellius-genoa-scratch.sh  #   and Genoa partitions)
└── scripts/
    ├── ior-run.sh                    # Run script for B6
    └── mdtest-run.sh                 # Run script for B7
```

## 1. Build

One build of the pinned IO500 commit provides the `io500` program and the `ior`, `mdtest` and `pfind` binaries it uses. B6 and B7 run the same `ior` and `mdtest`.

```bash
git clone https://github.com/IO500/io500.git
cd io500 && git checkout ddabf2efc978f8650c29176ca422bb0bf3e8990a
./prepare.sh          # builds ./io500 and bin/ior, bin/mdtest, bin/pfind
```

`prepare.sh` needs MPI, a C compiler, git and autotools. Extra configure and compile flags can be passed with `IOR_EXTRA_CONFIGURE_FLAGS`, `IO500_EXTRA_CFLAGS` and `IO500_EXTRA_LDFLAGS`; report any that are used.

**Build check.** In the IO500 directory, these commands must print the pinned commits of IO500, IOR and pfind, and nothing else (no local changes). `prepare.sh` leaves untracked files, such as `schema-tools/`, which the commands ignore.

```bash
git rev-parse HEAD && git status --porcelain --untracked-files=no
git -C build/ior rev-parse HEAD && git -C build/ior status --porcelain --untracked-files=no
git -C build/pfind rev-parse HEAD && git -C build/pfind status --porcelain --untracked-files=no
```

## 2. B5: IO500

**Configuration.** Copy `config/rules/io500-posix.ini` and set `datadir` (a directory on the storage system under test) and `resultdir` in `[global]`. Only the keys listed in `README.md` may be tuned.

**Run.**

```bash
<launcher> <N tasks, P per node> ./io500 io500-posix.ini
```

**Result summary.** IO500 prints a summary and writes it to `result_summary.txt` in a time-stamped directory under `resultdir`:

```
IO500 version <version> (standard)
[RESULT]       ior-easy-write        <value> GiB/s : time <seconds> seconds
[RESULT]    mdtest-easy-write        <value> kIOPS : time <seconds> seconds
[      ]            timestamp        0.000000 kIOPS : time 0.000 seconds
[RESULT]       ior-hard-write        <value> GiB/s : time <seconds> seconds
[RESULT]    mdtest-hard-write        <value> kIOPS : time <seconds> seconds
[RESULT]                 find        <value> kIOPS : time <seconds> seconds
[RESULT]        ior-easy-read        <value> GiB/s : time <seconds> seconds
[RESULT]     mdtest-easy-stat        <value> kIOPS : time <seconds> seconds
[RESULT]        ior-hard-read        <value> GiB/s : time <seconds> seconds
[RESULT]     mdtest-hard-stat        <value> kIOPS : time <seconds> seconds
[RESULT]   mdtest-easy-delete        <value> kIOPS : time <seconds> seconds
[RESULT]     mdtest-hard-read        <value> kIOPS : time <seconds> seconds
[RESULT]   mdtest-hard-delete        <value> kIOPS : time <seconds> seconds
[      ]  ior-rnd4K-easy-read        <value> GiB/s : time <seconds> seconds
[SCORE ] Bandwidth <value> GiB/s : IOPS <value> kiops : TOTAL <value>
```

- The result for B5 is the TOTAL on the `[SCORE ]` line.
- `[RESULT]` lines enter the Bandwidth and IOPS scores; `[      ]` lines do not (`timestamp` only marks the start time used by `find`).
- `[INVALID]` at the end of a line marks a phase, or on the `[SCORE ]` line the run, that failed a validity check. The reason is recorded in `result.txt`.
- The version line must contain the first 12 digits of the pinned commit, `ddabf2efc978`, and must not end in `-<n>`, which IO500 appends when its `src/` has local changes. A clone of the release reports `io500-sc26_v1-2-gddabf2efc978`.

The result directory also contains `result.txt` (the exact command, timing and any validity errors for each phase), `config-orig.ini` and `config.ini` (the configuration as given and as applied), and a log and a per-process CSV file for each phase. Each phase log starts with a `FS ... Used FS` line that shows the file system fill level.

## 3. B6 and B7: run scripts

Each workload has a run script, configured by two files:

- A **system file** (`config/systems/`) holds the facts about the system: launcher, job limits, node size, environment, binaries and target directory.
- A **rule file** (`config/rules/`) holds the workload: data set size and layout, transfer sizes and access-pattern flags. The rule files for B6 and B7 are used unchanged.

The run script combines the two with the requested task layout, computes the per-process share of the data set, writes one job script into a new case directory and submits it with `sbatch`. With `--interactive` it prints the commands instead, for systems without Slurm or inside an existing allocation; pipe them to `bash`, and load the environment from `ENV_SETUP` yourself in that case.

| Option | Meaning |
|---|---|
| `--system=PATH` | System file |
| `--rule=PATH` | Rule file |
| `--ppn=P` | Processes per node |
| `--ntasks=N` | Total processes, P × nodes |
| `--interactive` | Print the commands instead of submitting a job |
| `--dry-run` | Generate the job script without submitting it |

The case directory holds everything to deliver for the run:

```
results/<rule>/<cluster>-<partition>-<filesystem>/<timestamp>/
├── ior.md | mdtest.md         # Run metadata: system, rule, task layout, binary commit
├── <job>.sh                   # Generated job script
└── <job>_<jobid>.out / .err   # Benchmark output
```

The metadata file records the commit of the source tree the binary was built from (`ior_commit`, `mdtest_commit`) and whether that tree has local changes (`..._source_tree = clean | modified`).

**System file.** Copy `config/systems/snellius-rome-scratch.sh` and set:

| Variable | Description |
|---|---|
| `CLUSTER`, `PARTITION`, `FILESYSTEM` | Labels used in the results path; `PARTITION` is also the Slurm partition |
| `TIME` | Job time limit; long enough for a full-system run |
| `RUNNER`, `RUNNER_NTASKS_OPT`, `RUNNER_PPN_OPT` | MPI launcher and its options for total tasks and tasks per node (e.g. `srun`, `-n`, `--ntasks-per-node`) |
| `FEATURES` | Slurm `--constraint` value, may be empty |
| `CPUS_PER_NODE` | Cores per node (upper bound for `--ppn`) |
| `DRAM_PER_NODE` | Memory in GiB available to a job on one node. IOR fills all but max(32 GiB, 1 GiB per process) of it before running, so reads cannot be served from the client page cache |
| `ENV_SETUP` | Bash array of commands run at job start, e.g. module loads |
| `IOR`, `MDTEST` | Paths to `bin/ior` and `bin/mdtest` of the IO500 build |
| `TARGET_DIR` | Directory on the storage system under test |

The scripts need `bash` 4 or later, `awk`, `bc` and `git`.

### B6: ERA5 climate data set (IOR)

```bash
./scripts/ior-run.sh --system=<system file> \
  --rule=config/rules/ior-scientific.sh \
  --ppn=<P> --ntasks=<P × nodes>
```

The job runs two IOR steps on one shared file: a sequential write step and a random-order read step; the read step runs only if the write step succeeded, and deletes the file at the end. The result is the `read` row, `Max(MiB)` column, of the read step's `Summary of all tests` in the job output.

The run needs 540 TiB of free capacity on top of the prefill. A file of this size needs wide striping on file systems with a per-object size limit; tune the target directory accordingly.

### B7: LVD-1689M image data set (mdtest)

```bash
./scripts/mdtest-run.sh --system=<system file> \
  --rule=config/rules/mdtest-lvd-1689m-shallow.sh \
  --ppn=<P> --ntasks=<P × nodes>
```

The result is the `File read` row, Max column, of the `SUMMARY rate` table in the job output.

> [!WARNING]
> **The size of the data set depends on the number of processes.** mdtest gives every process the same whole number of files, k, in each of the 25,000 directories, and creates nothing for a process with fewer files than directories. With P processes, B7 therefore creates P × k × 25,000 files, k ≥ 1, and every directory holds k × P files.
>
> Before generating the job, the run script chooses the smallest k for which the total reaches 1.7 billion files and prints the result and its deviation from the target (`files: P processes x k per directory x 25000 directories = … (target 1700000000, …%)`). The data set is never smaller than 1.7 billion files, and usually larger:
>
> - 68,000, 34,000, 22,667, … processes (k = 1, 2, 3, …) give 1.7 billion files.
> - Other counts up to 68,000 give more, by less than 1/(k − 1): less than 20% with up to 14,336 processes, but up to twice as many just below 68,000. With the 2 processes per accelerator instance of the rules, 512 GPUs with 7 instances each run 7,168 processes: k = 10 and 1,792,000,000 files, 5.4% above the target.
> - Above 68,000 processes, k = 1, and every additional process adds 25,000 files (25,000 inodes and 7.15 GiB): 80,000 processes, for example, create 2 billion files (+17.6%, about 559 TiB). This is not expected at 2 processes per accelerator instance: 68,000 processes correspond to about 4,860 GPUs with 7 instances each.
>
> The `… tasks, … files` line of the output confirms the number of files created.

The run needs one inode and 300 KiB of free capacity per file on top of the prefill: 1.7 billion inodes and about 475 TiB for 1.7 billion files, more for a larger data set. Run on 3 or more nodes: on fewer, the read phase can return to the node that wrote the files.

## 4. Smoke tests

Before a full run, check the build, the system file and the scripts on a few nodes with `config/rules/io500-smoketest.ini`, `config/rules/ior-smoketest.sh` and `config/rules/mdtest-smoketest.sh`. The IOR and mdtest smoke tests use the flags of the B6 and B7 rules on small data sets. They complete in minutes; their performance numbers are meaningless, and IO500 marks the smoke-test run `[INVALID]` by design.

## 5. Example: smoke tests on Snellius

This example runs the three smoke tests on four nodes of the Snellius `rome` partition (2 × AMD EPYC 7H12, 128 cores and 224 GiB available to a job per node) with 64 processes per node, against the scratch-shared GPFS file system. The Rome nodes have no accelerators, so the process count of the rules does not apply; 64 processes per node are used instead. The four nodes stand in for the full accelerated partition. It logs every step from login to the results, as run on 19 October 2026.

The numbers only show that everything works. With four nodes, tiny data sets and a 30 s IO500 stonewall, reads are served largely from caches, IO500 marks the run `[INVALID]`, and the file system was 26% full, below the prefill level.

### Step 0 — Access and allocation

```bash
ssh <user>@snellius.surf.nl
salloc -N4 -p rome -t 60
```

```
salloc: Granted job allocation 28285920
salloc: Nodes tcn[230,246,249-250] are ready for job
```

All following commands run in this shell on the login node; `srun` starts the processes on the allocated nodes. From another login shell, `export SLURM_JOB_ID=<job id>` has the same effect.

### Step 1 — Benchmarks

Copy this directory (`application-benchmarks/IO`) to the system, here with `rsync` from a local clone of the repository:

```bash
ssh <user>@snellius.surf.nl mkdir -p nlaif-example
rsync -a NLAIF-benchmarks/application-benchmarks/IO/ <user>@snellius.surf.nl:nlaif-example/IO/
```

### Step 2 — Software environment and IO500 build

The build of the pinned commit, into `$HOME/.local/io500-sc26`, where the system file of step 3 expects it. `prepare.sh` compiles, so it runs on an allocated node:

```bash
module load 2025 foss/2025a Autoconf/2.72-GCCcore-14.2.0
git clone https://github.com/IO500/io500.git $HOME/.local/io500-sc26
cd $HOME/.local/io500-sc26
git checkout ddabf2efc978f8650c29176ca422bb0bf3e8990a
srun --ntasks=1 --cpus-per-task=16 ./prepare.sh > $HOME/nlaif-example/io500-prepare.log 2>&1
grep '^OK' $HOME/nlaif-example/io500-prepare.log
ls io500 bin
```

```
OK: All required software packages are now prepared
io500

bin:
ior  mdtest  md-workbench  pfind
```

The build takes about 45 s. The Autoconf module is only needed for the build; the runs below need `module load 2025 foss/2025a`. The build check and the compiler and MPI versions:

```bash
git rev-parse HEAD && git status --porcelain --untracked-files=no
git -C build/ior rev-parse HEAD && git -C build/ior status --porcelain --untracked-files=no
git -C build/pfind rev-parse HEAD && git -C build/pfind status --porcelain --untracked-files=no
gcc --version | head -1
mpirun --version | head -1
```

```
ddabf2efc978f8650c29176ca422bb0bf3e8990a
f1172addabada77d13cf37773d1ffdb202b4fc4a
27427343bb66f525b93d225d1f124cbed440d0b5
gcc (GCC) 14.2.0
mpirun (Open MPI) 5.0.7
```

The three commits are the pinned ones, and no `git status` lines follow them: the trees have no local changes.

### Step 3 — System file

The Rome example file `config/systems/snellius-rome-scratch.sh` is used unchanged. Its `IOR` and `MDTEST` point at the binaries of the IO500 build of step 2, so B6 and B7 run the same `ior` and `mdtest` as B5, and `TARGET_DIR` is on scratch-shared:

```bash
IOR="$HOME/.local/io500-sc26/bin/ior"
MDTEST="$HOME/.local/io500-sc26/bin/mdtest"
TARGET_DIR="/scratch-shared/$USER"
```

Its node values are 128 cores per node and 224 GiB available to a job (`memory.max` of the job's cgroup: 240,518,168,576 bytes).

For another partition, copy the file and set its partition and node values. `snellius-genoa-scratch.sh`, also included in `config/systems/`, was made this way, with the values taken from a Genoa node (`lscpu`: 2 sockets × 96 cores, 1 thread per core; `memory.max`: 360,777,252,864 bytes = 336 GiB):

```diff
-# System: Snellius (SURF) — Rome partition, scratch-shared filesystem
+# System: Snellius (SURF) — Genoa partition, scratch-shared filesystem
-PARTITION="rome"
+PARTITION="genoa"
-# Hardware: 224 GiB of memory available to a job and 128 cores per node
+# Hardware: 336 GiB of memory available to a job and 192 cores per node (2 × AMD EPYC 9654)
-DRAM_PER_NODE=224
-CPUS_PER_NODE=128
+DRAM_PER_NODE=336
+CPUS_PER_NODE=192
```

### Step 4 — IO500 smoke test (procedure of B5)

```bash
module load 2025 foss/2025a
cd $HOME/nlaif-example
cp IO/config/rules/io500-smoketest.ini .
```

Set the data and result directories in the copy:

```diff
 [global]
-datadir = ./datafiles
-resultdir = ./results
+datadir = /scratch-shared/<user>/nlaif-example/io500
+resultdir = /home/<user>/nlaif-example/io500-results
```

Run on 4 nodes × 64 processes:

```bash
srun -N 4 -n 256 --ntasks-per-node 64 $HOME/.local/io500-sc26/io500 io500-smoketest.ini > io500-smoketest.out 2>&1
```

The run takes 220 s. IO500 writes its results to `io500-results/2026.10.19-14.32.03/`; `result_summary.txt`:

```
IO500 version io500-sc26_v1-2-gddabf2efc978 (standard)
[RESULT]       ior-easy-write        2.201852 GiB/s : time 7.316 seconds [INVALID]
[RESULT]    mdtest-easy-write       36.038292 kIOPS : time 12.373 seconds [INVALID]
[      ]            timestamp        0.000000 kIOPS : time 0.000 seconds
[RESULT]       ior-hard-write        0.028492 GiB/s : time 78.958 seconds [INVALID]
[RESULT]    mdtest-hard-write        0.227948 kIOPS : time 44.805 seconds [INVALID]
[RESULT]                 find       50.357377 kIOPS : time 1.222 seconds
[RESULT]        ior-easy-read       13.176024 GiB/s : time 1.220 seconds
[RESULT]     mdtest-easy-stat       43.397851 kIOPS : time 2.184 seconds
[RESULT]        ior-hard-read        1.907297 GiB/s : time 1.179 seconds
[RESULT]     mdtest-hard-stat        3.172046 kIOPS : time 4.153 seconds
[RESULT]   mdtest-easy-delete        4.526463 kIOPS : time 18.925 seconds
[RESULT]     mdtest-hard-read       13.283273 kIOPS : time 1.756 seconds
[RESULT]   mdtest-hard-delete        2.165592 kIOPS : time 6.575 seconds
[      ]  ior-rnd4K-easy-read        8.190830 GiB/s : time 1.963 seconds
[SCORE ] Bandwidth 1.120543 GiB/s : IOPS 7.223769 kiops : TOTAL 2.845091 [INVALID]
```

The version line contains the pinned commit without a `-<n>` suffix. The write phases are `[INVALID]` because they ran for less than the 300 s stonewall, as intended for a smoke test.

### Step 5 — IOR smoke test (procedure of B6)

```bash
cd $HOME/nlaif-example/IO
./scripts/ior-run.sh --system=config/systems/snellius-rome-scratch.sh \
  --rule=config/rules/ior-smoketest.sh --ppn=64 --ntasks=256 --interactive | bash
```

The run script prints, and `bash` executes, a write step and a read step on one shared file (output redirections omitted):

```bash
srun -n 256 --ntasks-per-node 64 /home/<user>/.local/io500-sc26/bin/ior -b 40m -t 4m -s 1 -w -e -g -a POSIX --dataPacketType=random -k  -M 71 -o "/scratch-shared/<user>/ior-smoketest/2026-10-19_143511/iorfile" &&
srun -n 256 --ntasks-per-node 64 /home/<user>/.local/io500-sc26/bin/ior -b 40m -t 4m -s 1 -r -z -e -g -a POSIX --dataPacketType=random --random-offset-seed=1  -M 71 -o "/scratch-shared/<user>/ior-smoketest/2026-10-19_143511/iorfile"
```

The 10 GiB target is split into one 40 MiB block per process, ten 4 MiB transfers. The write step is sequential; the read step reads at random offsets. `-M 71` makes IOR fill most of each node's memory before the run, so that reads cannot be served from the page cache.

The output of the read step shows the random order, the memory IOR fills on each node and the read bandwidth (excerpt):

```
Path                : /scratch-shared/<user>/ior-smoketest/2026-10-19_143511/iorfile
FS                  : 2473.6 TiB   Used FS: 26.2%   Inodes: 300.0 Mi   Used Inodes: 60.5%
access              : single-shared-file
ordering in a file  : random
tasks               : 256
clients per node    : 64
memoryPerNode       : 178.68 GiB
aggregate filesize  : 10 GiB

Results:

access    bw(MiB/s)  IOPS       Latency(s)  block(KiB) xfer(KiB)  open(s)    wr/rd(s)   close(s)   total(s)   iter
------    ---------  ----       ----------  ---------- ---------  --------   --------   --------   --------   ----
read      5520       1385.53    0.070156    40960      4096       0.003808   1.85       0.002961   1.85       0

Summary of all tests:
Operation   Max(MiB)   Min(MiB)  Mean(MiB)     StdDev   Max(OPs)   Min(OPs)  Mean(OPs)     StdDev    Mean(s) Stonewall(s) Stonewall(MiB) Test# #Tasks tPN reps fPP reord reordoff reordrand seed segcnt   blksiz    xsize aggs(MiB)   API RefNum
read         5520.22    5520.22    5520.22       0.00    1380.06    1380.06    1380.06       0.00    1.85500         NA            NA     0    256  64    1   0     0        1         0    0      1 41943040  4194304   10240.0 POSIX      0
```

The result is the `read` row, `Max(MiB)` column, of `Summary of all tests`: 5520.22 MiB/s; the `Results` table above it prints the same bandwidth rounded. The `FS` line shows the file system 26% full. The run takes 124 s. The case directory `results/ior-smoketest/snellius-rome-scratch-shared/2026-10-19_143511/` then holds `ior.md` (with `ior_commit = f1172addabada77d13cf37773d1ffdb202b4fc4a` and `ior_source_tree = clean`), `ior_n256.out` and `ior_n256.err`.

### Step 6 — mdtest smoke test (procedure of B7)

```bash
./scripts/mdtest-run.sh --system=config/systems/snellius-rome-scratch.sh \
  --rule=config/rules/mdtest-smoketest.sh --ppn=64 --ntasks=256 --interactive | bash
```

The output shows rates for File creation, File read and File removal, and 0.000 for File stat, which does not run:

```
256 tasks, 13824 files
SUMMARY rate (in ops/sec): (of 1 iterations)
   Operation                     Max            Min           Mean        Std Dev
   File creation                 395.408        395.408        395.408          0.000
   File stat                       0.000          0.000          0.000          0.000
   File read                     562.102        562.102        562.102          0.000
   File removal                 1578.167       1578.167       1578.167          0.000
   Tree creation                 300.618        300.618        300.618          0.000
   Tree removal                  110.875        110.875        110.875          0.000
```

mdtest gives every process the same whole number of files in each of the 27 leaf directories; with 256 processes, 2 per directory (54 per process) reach the smoke test's target of 13,824 files exactly. The verbose output also shows `Shifting ranks by 64 for each phase`: the read phase reads files written two nodes away. The run takes 107 s. The case directory `results/mdtest-smoketest/snellius-rome-scratch-shared/2026-10-19_143717/` holds `mdtest.md` (with `mdtest_commit = f1172addabada77d13cf37773d1ffdb202b4fc4a` and `mdtest_source_tree = clean`), `mdtest_n256.out` and `mdtest_n256.err`.

### Step 7 — Results

The three results, exactly as printed (B6 from the `Summary of all tests` of the read step):

```bash
cd $HOME/nlaif-example
grep '^\[SCORE' io500-results/2026.10.19-14.32.03/result_summary.txt
grep -A2 '^Summary of all tests' IO/results/ior-smoketest/snellius-rome-scratch-shared/2026-10-19_143511/ior_n256.out | grep '^read'
grep -E '^ +File read ' IO/results/mdtest-smoketest/snellius-rome-scratch-shared/2026-10-19_143717/mdtest_n256.out
```

```
[SCORE ] Bandwidth 1.120543 GiB/s : IOPS 7.223769 kiops : TOTAL 2.845091 [INVALID]
read         5520.22    5520.22    5520.22       0.00    1380.06    1380.06    1380.06       0.00    1.85500         NA            NA     0    256  64    1   0     0        1         0    0      1 41943040  4194304   10240.0 POSIX      0
   File read                     562.102        562.102        562.102          0.000
```

| Benchmark | Result | Value |
|---|---|---|
| B5 | TOTAL | 2.845091 (smoke test, `[INVALID]`) |
| B6 | Read bandwidth | 5520.22 MiB/s |
| B7 | File read rate | 562.102 operations/s |

The output files to deliver are the IO500 result directory `io500-results/2026.10.19-14.32.03/` with the job script, and the two case directories of steps 5 and 6.
