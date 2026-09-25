# Bringing the vLLM benchmark up on MI355X — troubleshooting log

Chronological record of everything that broke and how it was resolved, bringing this
benchmark up on an 8× MI355X node with **no SLURM and no apptainer**. Kept because most
of these are not obvious from the error messages, and the next person on ROCm will hit
the same list.

Platform: `mi355-gpu-37`, 8× Instinct MI355X (288 GiB each, gfx950), ROCm 7.2, rootless
podman 4.7.1, no shared filesystem, 1.5 TiB `/dev/shm`.

§1–§11 are the single-node bring-up. §12 is the later two-node setup (`mi355-gpu-13` +
`mi355-gpu-37`, 16 GPUs) once SLURM and shared storage became available.

---

## 1. No scheduler, no apptainer

**Symptom.** `run.sh` submits `sbatch` jobs; every `configs/*/system.sh` builds an
`apptainer exec` prefix. Neither exists on this box.

**Fix.** `run_nosched.sh` replaces the two scheduler-dependent pieces and nothing else:

| SLURM harness | scheduler-free driver |
|---|---|
| orchestrator's `sbatch` + `squeue` poll | serial `for` loop |
| worker's `srun`-based Ray bootstrap | plain `ssh` (so `--ray` is never passed) |

It still calls `src/sweep_parallel.py --worker` with `SERVE_CMD`/`BENCH_CMD` lifted out of
`scripts/vllm_sweep_parallel.sh` with `sed`, rather than copied — so the fixed parameters
cannot drift from the spec. Measurement path and `summary.json` layout are unchanged.

Three defects fixed in the driver itself:

- `mktemp -d "$REPO_ROOT/tmp/..."` failed on a fresh clone — `tmp/` is gitignored and absent.
  Added `mkdir -p tmp logs`.
- **podman cannot host a Ray daemon through a one-shot `run`.** `ray start` daemonises and
  returns, the container's PID 1 exits, and the daemon dies with it. For `NODES=...` runs the
  driver now boots one persistent container per node (`sleep infinity`) and drives everything
  through `podman exec`. apptainer has no such lifecycle and keeps its plain prefix.
- Worker output now `tee`s into `results/`, so a failed measurement is diagnosable without
  re-running a 10-minute engine load.

---

## 2. ROCm: sparse attention indexer requires AITER

**Symptom.** Engine dies during executor init:

```
RuntimeError: Worker failed with error 'Sparse attention indexer ROCm path is only
supported on AITER. Please enable aiter with VLLM_ROCM_USE_AITER=1'
```

**Fix.** `export VLLM_ROCM_USE_AITER=1`.

**On v0.30.0 the symptom changes and gets worse.** The guard moved (to
`sparse_attn_indexer_kpool.py`) and no longer fires first. Without the variable, v0.30.0 gets
past it, proceeds into CUDA graph capture, and dies with

```
CUDA error: operation not permitted when stream is capturing
```

which does not mention AITER at all. Verified as a single-variable comparison: the same v0.30.0
config runs all six cells with the variable set. The default is still `False` in both releases.

**AITER JIT-compiles.** For any kernel the image has no prebuilt entry for, AITER compiles from
source bundled in the same image, for the local `gfx` arch, at first use. That is the shipped
package's normal operation — the same class as torch.compile / inductor — not an installed
custom kernel. It is why the image tree must be writable (§3) and why seeding the kernel cache
matters: a cold cache costs roughly ten minutes per engine start.

**Rules note.** This is *permitted* — AITER ships inside the official
`vllm/vllm-openai-rocm` image (verified: 116 prebuilt `.so`, 3494 prebuilt kernels), so the
env var selects a code path already present at the pinned version and installs nothing.
§8.5's "installing any model-execution package" and ruling 5's "custom kernels" do not
apply. Documented in README §8.3 and §5.4.

---

## 3. podman `--userns=keep-id` breaks AITER's JIT

**Symptom.**

```
RuntimeError: Worker failed with error '[Errno 13] Permission denied:
.../aiter/jit/flydsl_cache/launch_mixed_moe_gemm1_7d59791ea8d748ec85f9bfc383674078'
```

**Cause.** AITER JIT-compiles kernels it has no prebuilt entry for straight into its
`dist-packages` directory, and Triton caches under `$HOME` (= `/root` in the image). Under
`--userns=keep-id` you are uid 1002 inside and both are root-owned and read-only.

**Fix.** Drop `--userns=keep-id`. Rootless podman's *default* mapping already puts container
root at your host uid, so `results/` still comes back owned by you **and** the image tree is
writable. Verified: `touch` in dist-packages succeeds, and files created on the bind mount
are owned by `amd` on the host.

Also bind-mount the AITER kernel cache out to the host, seeded once from the image, so JIT
output survives `--rm` across the six engine restarts in a sweep:

```bash
podman create --name seed localhost/vllm-bench:amd-v0.29.0 true
mkdir -p ~/.cache/aiter/flydsl_cache
podman cp seed:/usr/local/lib/python3.12/dist-packages/aiter/jit/flydsl_cache/. \
          ~/.cache/aiter/flydsl_cache/ && podman rm seed
```

---

## 4. Declared N exceeded what the KV cache holds

**Symptom.** Not an error — `serve.log` reported

```
GPU KV cache size: 263,820 tokens, Maximum concurrency for 9,216 tokens per request: 28.63x
```

against a declared `max_num_seqs = 32`. The `high` operating point benches at concurrency 32,
so it would have measured preemption rather than throughput (§9 ruling 7).

**Fix.** N is a declared, self-policed parameter, so the value has to be measured, not
assumed. DeepSeek-V4 uses MLA, whose KV cache is replicated rather than split across TP ranks,
so aggregate concurrency tracks the per-GPU cache directly:

| GPU_MEM_UTIL | KV cache | max concurrency |
|---|---|---|
| 0.90 | 144.89 GiB | 28.63x |
| 0.95 | 159.29 GiB | 31.47x — short of 32 |
| 0.96 | 162.17 GiB | 32.04x — clears by 0.04x, too thin |
| 0.97 | 165.05 GiB | **32.61x** |

Shipped as two configs rather than one guess: `amd/mi355x` (N=16 @ 0.90, the nearest lower
power of two) and `amd/mi355x-n32` (N=32 @ 0.97). N=32 won every cell by 52–82%.

**Always check before trusting a run:**
```bash
grep "Maximum concurrency" results/<run>/*/serve.log
```

**Superseded (§12.6):** that line turned out to be a worst-case bound, not the real capacity.
Measured, one node sustains N=1024 with at most ~50% KV-cache usage and no preemption.

---

## 5. Worker stalls 30 minutes on a dead engine

**Symptom.** After an engine died during startup, the worker sat for the full
`--serve-timeout` (1800 s) before giving up. Two failed probes cost ~60 idle minutes.

**Cause.** `src/sweep_parallel.py`'s `run_worker()` calls
`wait_for_server(port, args.serve_timeout)` without checking `server.poll()`. The same file
already has exactly this guard in the bench loop (`if server.poll() is not None: ... aborting
remaining shapes`) — it is a one-line omission in the startup path.

**Status: fixed** in `src/sweep_parallel.py` (shared harness code, not AMD-specific).
`wait_for_server()` now takes the server process and its log: it raises as soon as
`server.poll()` reports an exit, and also aborts if `serve.log` stops growing for 20 minutes
(a wedged, still-running engine). On SLURM the old behaviour held an allocation idle for
30 min per dead engine; a dead engine now releases its nodes within seconds.

---

## 6. `amd/DeepSeek-V4-Pro-MXFP4` does not load on v0.29.0

Investigated while trying to reach FP4 compute. Two distinct blockers.

**6a. Shared expert shape mismatch.**

```
vllm/models/deepseek_v4/amd/model.py:838  weight_loader(param, loaded_weight, shard_id)
  parameter.py:175  load_merged_column_weight
    assert param_data.shape == loaded_weight.shape   AssertionError
```

vLLM builds a dense `DeepseekV4MLP` for the shared expert and routes it to the
DeepSeek-native FP8 path, allocating `[3072, 7168]`; the checkpoint ships it FP4-packed as
`[3072, 3584]`. The FP4 layout is only loadable via shared-expert fusion, and

```python
def _fuse_shared_experts_enabled(config) -> bool:
    return bool(getattr(config, "n_shared_experts", None)
                and envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
                and not get_current_vllm_config().parallel_config.enable_expert_parallel)
```

**fusion and expert parallelism are mutually exclusive.** With `--enable-expert-parallel`
dropped, the checkpoint loads and serves (32.44x KV headroom).

**6b. It still serves W4A8 anyway.** Operands come out
`'torch.float8_e4m3fn', 'torch.float4_e2m1fn_x2'` — identical numerics to the baseline,
because `DeepseekV4FP8Config.from_config` **discards the entire Quark schema**:

```python
if config.get("quant_method") == "quark":
    quark_exclude = config.get("exclude") or []
    config = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3",
              "scale_fmt": "ue8m0", "weight_block_size": [128,128],
              "ignored_layers": [name for name in quark_exclude ...]}
```

All 1497 `layer_quant_config` entries are thrown away; only `exclude` survives. So the
checkpoint's `input_tensors: {dtype: fp4}` is never read.

**Conclusion: FP4 compute is unreachable on v0.29.0's ROCm DeepSeek-V4 path, by any
checkpoint.** `aiter_mxfp4_mxfp4` is not an accepted `--moe-backend` value, and the ROCm
priority list is `[AITER_MXFP4_BF16, TRITON_UNFUSED]` with no W4A4 entry.

**Cost of learning this: 864.7 GB evicted + 804 GB downloaded + ~2 h, for a negative result
the backend priority list alone would have predicted.** Lesson: check backend *reachability*
in `oracle/mxfp4.py` before moving weights around.

---

## 7. Bind-mounted cache must exist before the container starts

**Symptom.** On a freshly prepared second node, every measurement failed instantly:

```
Error: statfs /home/amd/.cache/aiter/flydsl_cache: no such file or directory
[nosched] measurement 5 FAILED — continuing
```

**Cause.** `configs/amd/mi355x/system.sh` bind-mounts the AITER kernel cache
(`-v $AITER_CACHE:.../aiter/jit/flydsl_cache`). podman refuses to start when the *host*
side of a bind mount does not exist — it will not create it for you. The sweep then
"succeeds" six times over in the sense that the driver keeps going, producing zero cells.

**Fix.** Seed the caches before the first run on any new node (§3 has the `podman cp`
recipe), or at minimum `mkdir -p` every path the container mounts:

```bash
mkdir -p ~/.cache/aiter/flydsl_cache ~/.cache/triton ~/vllm_cache
```

**Worth noting** the failure mode is quiet: `run_nosched.sh` logs `measurement N FAILED —
continuing` and exits 0 after producing no `summary.json` at all. Always check the cell
count, not the exit status:

```bash
find results/<run> -name summary.json | wc -l    # expect 6
```

## 8. /dev/shm is not durable — an 806 GB checkpoint vanished overnight

**Symptom.** A sweep launched at 02:50 died instantly on every measurement:

```
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find an appropriate cached
snapshot folder for the specified revision on the local disk and outgoing traffic
has been disabled.
```

`/dev/shm/hf_hub` was simply gone. The weights were verified present and serving at 17:22
the previous evening — two full sweeps ran off them — and by 02:50 `/dev/shm` was back to
1.5 TiB free on **both** nodes.

**Not a reboot**: `uptime` showed 3 days. `systemd-tmpfiles-clean.timer` last fired 18 h
earlier, i.e. *before* the checkpoint was restored, so that is not the mechanism either.
Cause still unidentified — but the operational fact stands: on these nodes `/dev/shm` can
be cleared out from under a long-running campaign, simultaneously across nodes.

**Implications.**

- Do not treat an 800 GB `/dev/shm` checkpoint as durable across a multi-day campaign.
  Budget for an ~18 min re-download (at ~750–900 MiB/s) or stage the weights on real disk
  and accept the slower load. `/` had 564 GB free here — not enough for this checkpoint,
  so that trade-off needs planning, not improvising.
- Verify the weights resolve *before* launching a sweep, rather than discovering it six
  measurements later:
  ```bash
  ls /dev/shm/hf_hub/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/*/ | wc -l   # 74
  ```
- This interacts badly with §5: the worker does not notice the engine died, so each of the
  six measurements waits out the full 1800 s `--serve-timeout`. A missing checkpoint costs
  three hours of nothing before the driver reports failure.

## 9. DCP is unavailable for DeepSeek-V4 — do not plan around it

`--decode-context-parallel-size` shards the decode KV cache across TP ranks without
expanding the world size, which makes it the obvious lever against MLA's replicated KV
(575.9 KiB/token measured vs 34.3 KiB/token theoretical, a factor of 16.8). It does not
work for this model:

```
NotImplementedError: DCP is not supported with sparse indexer KV compression
(compress_ratio=4).
  vllm/v1/attention/backends/mla/indexer.py:630
```

The guard is unconditional on size:

```python
if self.dcp_world_size > 1 and self.compress_ratio > 1:
    raise NotImplementedError(...)
```

DeepSeek-V4's sparse indexer sets `compress_ratio=4`, so **any** `dcp > 1` refuses at engine
init. Measured: `dcp=1` serves normally, `dcp=2` and `dcp=8` both fail. Nothing to tune,
and nothing for the benchmark spec to permit — it is an engine limitation, not a rule.

**Two process notes from running this scan**, both of which cost a rerun:

- `pkill -f "vllm serve"` in a probe loop matched broadly enough to kill the scan script
  itself (exit 144). Give each engine a named container and `podman rm -f` it by name.
- Waiting for `Maximum concurrency` alone is not enough — the engine prints `Available KV
  cache memory` a few seconds earlier and can then die during graph capture. Wait on the
  error strings too, or a failure looks like a hang.

## 10. AMD's FP4 checkpoints do not help, on v0.29.0 or v0.30.0

`amd/DeepSeek-V4-Pro-MXFP4` and `amd/DeepSeek-V4-Pro-NVFP4` both declare FP4 *activations*
(`global_quant_config.input_tensors: {dtype: fp4}`), unlike the base checkpoint's FP8. That
makes them look like the route to FP4 matrix compute. Measured on both engine versions:

| checkpoint | config | result |
|---|---|---|
| MXFP4 | EP8 (the benchmark's own) | **fails to load** — `assert param_data.shape == loaded_weight.shape` |
| MXFP4 | EP off + shared-expert fusion | loads; serves **W4A8** (`float8_e4m3fn x float4_e2m1fn_x2`) |
| NVFP4 | EP8 | loads; **emulated** — `Nvfp4QuantizationEmulationTritonExperts ... dequantize weights on the fly` |

The load failure is the shared expert: the checkpoint ships it FP4-packed `[3072, 3584]`,
vLLM builds a dense `DeepseekV4MLP` and allocates `[3072, 7168]` via the DeepSeek-native FP8
path. Only shared-expert fusion consumes the FP4 layout, and
`_fuse_shared_experts_enabled()` requires `not enable_expert_parallel` — so fusion and EP are
mutually exclusive, and the benchmark uses EP.

Even when it loads, the activation declaration is ignored: `DeepseekV4FP8Config.from_config`
discards the entire Quark schema (keeping only `exclude`), the ROCm DeepSeek-V4 priority list
is `[AITER_MXFP4_BF16, TRITON_UNFUSED]` with no W4A4 entry, and `aiter_mxfp4_mxfp4` is not an
accepted `--moe-backend` value. All four facts are identical in v0.29.0 and v0.30.0.

NVFP4 additionally reports only 31.51x KV concurrency, below a declared N=32.

**Check reachability in the source before moving weights.** All three outcomes above were
predicted in about five minutes by reading `from_config`, the priority list and the
`--moe-backend` choices. An earlier round of this same question cost 864 GB of eviction and
804 GB of download to reach the same conclusion.

## 11. Smaller things

- **`hf` CLI**: `~/.local/bin/hf` is a broken symlink to a nonexistent path. The working
  binary is `/home/amd/.hf-cli/venv/bin/hf`.
- **Deleting checkpoints**: use `hf cache rm model/<repo> --yes`, not `rm -rf`. It is the
  purpose-built tool, it handles the shared blob store, and a blunt 800 GB `rm -rf` is
  (rightly) refused by the safety classifier.
- **`vllm serve --help` is a summary** in these versions; flags only show under
  `--help=all`. An early flag-support check reported everything MISSING because of this.
- **Any `vllm`/python CLI call inside the ROCm image needs `--device /dev/kfd --device
  /dev/dri`**, or platform init fails with `No CUDA GPUs are available` and the command
  returns misleading output.
- **Two python installs in the image**: the real one is `python3.12`; `python3.10`
  dist-packages exists and is nearly empty. Probing the wrong one reports aiter and the
  mxfp4 oracle as absent when they are present.
- **Stale `.incomplete` blobs**: an interrupted `hf download` left 8 files / ~50 GB behind.
  `hf cache prune` clears them.

---

## 12. Two nodes under SLURM (16× MI355X, RDMA)

The working two-node configuration, and what each part of it is for. Cluster config:
`configs/amd/mi355x-2node-rdma-sweep` (generated by `scripts/gen_configs.py`); launch with

```bash
./run.sh amd/mi355x-2node-rdma-sweep --local
```

**Platform.** `mi355-gpu-13` + `mi355-gpu-37`, SLURM partition `gpu` (cons_tres,
`--gpus-per-node=8`, accounting disabled — no `sacct`), repo on the shared NFS mount
`/shared/data`, weights and datasets node-local (`/mnt/data/hf_hub`, `/dev/shm`). Each node
has eight AMD Pollara AI NICs (`rocep9s0 … rocep249s0`, one per GPU) on a routed RoCE v2
fabric, plus the front-end Ethernet port.

### 12.1 Where things run

- **Orchestrator: `--local` on the control node.** There is no CPU partition, and a plain
  `sbatch` job with no `--mem` receives the partition default — the whole node's memory — which
  leaves the node's GPUs unschedulable for the worker jobs. The orchestrator only submits and
  polls, so running it in the submitting shell costs nothing.
- **Workers: whole nodes.** `system.sh` sets `SBATCH_ARGS="--exclusive"`, so each worker job
  owns every CPU and GPU of its node(s) for its lifetime.
- **Repo on shared storage.** Both nodes and the control node see the same checkout, so
  `results/`, `tmp/` and the per-job configs are written once and read everywhere. Each
  container bind-mounts `/shared/data` (`-v /shared/data:/shared/data`) in addition to the
  single-node mounts, otherwise `-w $REPO_ROOT` fails with `workdir ... does not exist`.

### 12.2 Identity and rootless podman

- **Same uid on every node.** SLURM launches job steps under the submitting uid; the account
  must map to the same uid on the control node and both compute nodes.
- **`loginctl enable-linger <user>` on each compute node.** Rootless podman needs
  `/run/user/<uid>`, which systemd only keeps for users with a session or linger; SLURM job
  steps are not a login session.

### 12.3 Ray under podman

A layout wider than one node is served through a Ray cluster that the worker starts with
`srun`. With podman (unlike apptainer) the Ray daemons must live in long-running containers,
and everything else must run *inside* them:

- `start_ray_cluster` names the containers `ray-head-$SLURM_JOB_ID` / `ray-worker-$SLURM_JOB_ID`.
- `vllm serve` and `vllm bench` run via `podman exec -e RAY_ADDRESS=<gcs> ray-head-…`. A
  separate `podman run` cannot attach to the head's raylet.
- The bench `exec` gets `--workdir` set to its per-measurement result directory, so
  `--save-result` lands where the worker reads it back.
- Teardown removes each container **by name, one at a time, and retries until `podman ps`
  shows none left**. `podman rm -f <missing> <present>` exits without removing the present
  one, and a container left behind makes the *next* job's Ray cluster hang after `Connected
  to Ray cluster`.
- Without `sacct`, the orchestrator treats a job that has left `squeue` for three polls as
  finished, and decides success from whether its `summary.json` exists.

These are harness changes in `src/sweep_parallel.py`; apptainer keeps its original code path.

### 12.4 RCCL over the Pollara NICs

`configs/amd/mi355x-2node-rdma/system.sh` adds, on top of the single-node container flags:

```bash
POLLARA_HCA=rocep9s0,rocep25s0,rocep105s0,rocep121s0,rocep137s0,rocep153s0,rocep233s0,rocep249s0
CONTAINER_RUN_ARGS="$CONTAINER_RUN_ARGS --device /dev/infiniband --ulimit memlock=-1:-1 \
-v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-184:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
-e NCCL_IB_HCA=$POLLARA_HCA -e NCCL_IB_GID_INDEX=1 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET"
```

| setting | why |
|---|---|
| `--device /dev/infiniband`, `memlock=-1` | verbs devices and pinned-memory registration inside the container |
| host `libionic` bind-mounted over the image's | the image's provider (1.0.54.0-167) rejects the host kernel driver — `Driver ionic does not support the kernel ABI of 1`; the host's AINIC-bundle build (1.1.54.0-184) matches it |
| `NCCL_IB_HCA` | only the eight per-GPU Pollara NICs, not the front-end ports |
| `NCCL_IB_GID_INDEX=1` | RoCE v2 on the fabric's routed IPv6 addresses |

**Verify the transport in every run** — `NCCL_DEBUG=INFO` makes it one grep:

```bash
grep -c "via NET/IB/.*/GDRDMA" results/<run>/*/serve.log     # expect > 0 per cell
```

### 12.5 Timeouts

Exported in `configs/amd/mi355x-2node/system.sh`, **before** it sources `../mi355x/system.sh`
(that file applies defaults with `${VAR:-…}`, so an override placed after sourcing is a no-op):

| variable | value | why |
|---|---|---|
| `VLLM_ENGINE_READY_TIMEOUT_S` | 5400 | a two-node engine's first start JIT-compiles AITER kernels for shapes the single-node cache does not hold |
| `SERVE_TIMEOUT` | 5400 | same, at the harness's `/health` wait |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | 1800 | the first long prefill also creates the cross-node pipeline send/recv communicators lazily; vLLM's 300 s default per step is too tight for that step |

### 12.6 Layouts and N

Sixteen GPUs admit TP8/PP2, TP4/PP4 and TP2/PP8. **TP16 does not load**: the shared-expert MLP
split 16 ways gives a 192-wide partition, not divisible by the checkpoint's FP8 block
(`Weight input_size_per_partition = 192 is not divisible by weight quantization block_k = 128`).
Not fixable by configuration.

**N is measured, not read off the startup line.** For DeepSeek-V4 the `Maximum concurrency`
line is a worst-case reservation and understates what fits by one to two orders of magnitude
(README §4). Measured under the `high` load from the server's `/metrics` (`kv_cache_usage_perc`,
`num_requests_running`, `num_preemptions_total`):

| deployment | reported capacity | requests running | KV-cache usage | preemptions |
|---|---|---|---|---|
| 1 node, TP8, N=512 | 31.16x | 512 (1024→8192) | 21% | 0 |
| 1 node, TP8, N=1024 | 31.16x (v0.29) / 32.31x (v0.30) | 1024 (1024→8192) | 44–50% | 0 |
| 2 nodes, TP8/PP2, N=1024 | 76.71x | 1024 (1024→8192) | 17% | 0 |
| 2 nodes, TP2/PP8, N=256 | 293.71x | 256 (1024→8192) | 1.5% | 0 |

EP on or off moves both figures by <1%. Two related effects:

- **8192→1024 is capped by prefill admission, not by KV.** `--max-num-batched-tokens=9216` is
  fixed, so each engine step admits roughly one 8k prompt. At concurrency 512 on one node only
  ~320 requests run (the rest wait, KV at ~22%); on TP8/PP2 ~155 run at concurrency 512.
- **The reported figure still rises as TP falls** (78.4x TP8/PP2 → 147.9x TP4/PP4 → 293.7x
  TP2/PP8): fewer TP ranks replicate the MLA latent, and each pipeline stage holds fewer layers.

### 12.7 Operating notes

- **Give concurrent sweeps distinct results directories.** The default `RESULTS_DIR` is a
  timestamp to the second; two orchestrators started in the same second share it, *and* share
  `tmp/<same>/bench_<i>.json`, so one sweep's jobs silently run the other's operating points.
  Set `RESULTS_DIR=results/<stamp>_<name>` per launch.
- **`scancel` does not stop a running worker's container.** Rootless podman runs it under
  `conmon` in the user's systemd session, outside the SLURM job's processes, so a cancelled
  worker's engine keeps running and keeps its GPU memory. The next job on that node then fails
  at startup: `Free memory on device cuda:N (171.88/287.98 GiB) on startup is less than desired
  GPU memory utilization`, or `HIP failure: 'out of memory'` in the profile run. After cancelling,
  check `podman ps` on the node and remove leftovers by ID (`podman rm -f <id>`); then
  `scontrol requeue <job>` re-runs the failed cell with its original id and output paths.
- **The final plot/score step needs the container**, and the control node has no podman. After
  a `--local` run, score from any compute node:
  ```bash
  podman run --rm -v /shared/data:/shared/data -w $PWD localhost/vllm-bench:amd-v0.29.0 \
      python3 src/utils/score.py results/<run>
  ```
