# NL AI Factory — Storage Benchmarks

## 1. Objective

The NL AI Factory storage system serves many independent AI training and scientific workloads at the same time. These benchmarks measure the hot storage tier of the offered system, through its POSIX interface, under load from all nodes of the accelerated partition, on a file system that is filled as it will be in production.

There are three scored storage benchmarks, all run on the full accelerated partition: B5, the standard IO500 benchmark, and B6 and B7, two workloads that replicate how AI training reads its data sets. Scoring, weights and acceptance criteria are defined in the Technical Specifications. How to build, run and check the benchmarks is described in `INSTRUCTIONS.md` in this directory.

## 2. Benchmarks

| ID | Benchmark | Scored result | Reference configuration |
|---|---|---|---|
| B5 | IO500, standard configuration, POSIX | TOTAL score | `config/rules/io500-posix.ini` |
| B6 | ERA5 climate data set (IOR) | Read bandwidth, MiB/s | `config/rules/ior-scientific.sh` |
| B7 | LVD-1689M image data set (mdtest) | File read rate, operations/s | `config/rules/mdtest-lvd-1689m-shallow.sh` |

### B5: IO500

IO500 is the standard benchmark for parallel storage systems; running it unmodified makes the results comparable between Offers and with published systems. B5 is one complete IO500 run in standard mode, with these phases:

| Phase | Access pattern |
|---|---|
| ior-easy | Each process writes and then reads its own large file sequentially; transfer and file size are tunable. Best-case bandwidth. |
| ior-hard | All processes write and then read one shared file in interleaved 47,008-byte records. Worst-case bandwidth. |
| mdtest-easy | Each process creates, stats and deletes empty files in its own directory. |
| mdtest-hard | All processes create, stat, read and delete 3,901-byte files in one shared directory. |
| find | Parallel search of the namespace for files matching name, size and age. |
| ior-rnd4K-easy-read | Random 4 KiB reads from the ior-easy files. Reported, not part of the IO500 scores. |

Every write and create phase runs for at least 300 s (the stonewall); the matching read, stat and delete phases then process everything that was written. The Bandwidth score is the geometric mean of the four ior phases, the IOPS score the geometric mean of the seven mdtest phases and `find`, and the TOTAL, the scored result, the geometric mean of the two.

The configuration `io500-posix.ini` is fixed, including the 300 s stonewall, random data (`dataPacketType = random`) and standard mode, except for these keys, which the Tenderer may tune:

| Section | Tunable keys |
|---|---|
| `[ior-easy]` | `transferSize`, `blockSize`, `filePerProc`, `uniqueDir` |
| `[ior-hard]` | `segmentCount` |
| `[mdtest-easy]` | `n` |
| `[mdtest-hard]` | `n` |

A run is valid only if IO500 marks neither a phase nor the run `[INVALID]`. The IO500 submission rules (https://io500.org/rules/submission) apply in addition to this document.

### B6: ERA5 climate data set (IOR)

Replicates AI weather and climate models such as GraphCast, which train on ERA5 reanalysis data stored as large gridded files. During an epoch, many dataloader processes read spatial patches of about 4 MiB (one 0.25° field at one pressure level) from the same file, in shuffled order. B6 models one epoch over a 540 TiB subset of ERA5: ten times the GraphCast training set, or about 12% of the full ~5 PB reanalysis.

One job runs two IOR steps on one shared 540 TiB file, with random data:

1. **Write step**: all processes create the file sequentially in 4 MiB transfers, with an fsync at close. This step only creates the data set and is not measured.
2. **Read step**: all processes read the whole file in 4 MiB transfers in random order. Offsets are assigned to processes at random, so each process mostly reads data written by another node.

The read bandwidth of the read step is the scored result. The run needs 540 TiB of free capacity on top of the prefill.

### B7: LVD-1689M image data set (mdtest)

Replicates self-supervised vision pre-training on a web-scale image corpus: LVD-1689M, the 1.7-billion-image data set of DINOv3. At this scale images are stored as individual files spread over many directories, and a training epoch opens, reads and closes every image in shuffled order. B7 models a data set of at least 1.7 billion files of 300 KiB with random content in 25,000 directories, one level deep (about 68,000 files each), shared by all processes.

One mdtest run executes three phases:

1. **Create**: every file is created and written (300 KiB).
2. **Read**: every file is opened, read in full and closed, in random order, by a process on a different node from the one that created it.
3. **Remove**: all files and directories are deleted.

The file read rate is the scored result; create and remove only prepare and clean up the data set. The run needs one inode and 300 KiB of free capacity per file on top of the prefill: 1.7 billion inodes and about 475 TiB for 1.7 billion files. Depending on the number of processes, the data set can be larger than 1.7 billion files, by less than 20% with up to 14,336 processes; `INSTRUCTIONS.md` explains how its size is set.

## 3. General rules

- **Clients.** B5, B6 and B7 use all nodes of the accelerated partition as clients.
- **Processes.** B5, B6 and B7 use 2 processes per accelerator instance as defined in ACC-012 on every node, for example 14 per GPU with 7 instances per GPU.
- **Prefill.** Before B5 to B7, the storage system is filled with prefill data. When the read phases of a run start, prefill plus the data written by the run occupy at least 60% of the net capacity, the usable capacity the file system reports to clients. Prefill data is written with random content, like the benchmark data, stays in place for all runs and sits in the same file system and tier as the benchmark data.
- **No data reduction.** Deduplication, compression and any other data reduction are disabled for the benchmark and prefill data and must not reduce the amount of data written. 
- **Directory tuning.** The top-level benchmark directories may be pre-created and tuned, for example with striping. Directories below them may not be pre-created.
- **Caches.** The Tenderer states the aggregate server-side cache capacity (DRAM and other caches that are not a storage tier) per storage tier.

## 4. Software

- **Reference implementation.** All runs use one build of the official IO500 release `io500-sc26`, commit `ddabf2efc978f8650c29176ca422bb0bf3e8990a`, checked out by commit, not by tag: upstream has moved its release tags before. The release builds IOR commit `f1172addabada77d13cf37773d1ffdb202b4fc4a` and pfind commit `27427343bb66f525b93d225d1f124cbed440d0b5`. B5 runs the `io500` program of this build. B6 and B7 run its `ior` and `mdtest` binaries through the run scripts and rule files in this directory; each run records the commit its binary was built from, which must be the IOR commit above.
- **Allowed.** The compiler, MPI library and configure and compile flags of the Tenderer's choice; the system file that describes the offered system (launcher, node size, paths); the IO500 tunable keys listed for B5; directory tuning.
- **Not allowed.** Changes to any benchmark source, rule file or run script; any other IOR, mdtest or pfind; data reduction.
- **Acceptance.** The Hosting Entity re-runs the committed configurations with its own build of the same commit.

## 5. Reporting

### 5.1 Results

For each benchmark, report one value, exactly as the tool prints it:

| Benchmark | Result | Where it is printed |
|---|---|---|
| B5 | TOTAL | `[SCORE ]` line of `result_summary.txt` |
| B6 | Read bandwidth, MiB/s | `read` row, `Max(MiB)` column, of the `Summary of all tests` in the IOR output |
| B7 | File read rate, operations/s | `File read` row, Max column, of the `SUMMARY rate` table in the mdtest output |

### 5.2 Output files

For every result that was measured, deliver the complete output:

- **B5**: the complete IO500 result directory and the job script.
- **B6, B7**: the complete case directory written by the run script: job script, benchmark output and metadata file.
- **Build**: the compiler and MPI library with their versions, the configure and compile flags, and the build check described in `INSTRUCTIONS.md`, which shows the pinned commits and no local changes.

### 5.3 Statements

The output files do not show the following; state them with the results:

- the number of nodes in the accelerated partition and, per node, the processes used;
- how data reduction is disabled;
- the directory tuning commands;
- the prefill: amount of data and number of files.

### 5.4 Projected results

A result that is projected or extrapolated instead of measured on the offered configuration comes with the output files of the measurements it is based on and a description of the method.