# Bringing the vLLM benchmark up on MI355X — troubleshooting log

Chronological record of everything that broke and how it was resolved, bringing this
benchmark up on an 8× MI355X node with **no SLURM and no apptainer**. Kept because most
of these are not obvious from the error messages, and the next person on ROCm will hit
the same list.

Platform: `mi355-gpu-37`, 8× Instinct MI355X (288 GiB each, gfx950), ROCm 7.2, rootless
podman 4.7.1, no shared filesystem, 1.5 TiB `/dev/shm`.

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

---

## 5. Worker stalls 30 minutes on a dead engine

**Symptom.** After an engine died during startup, the worker sat for the full
`--serve-timeout` (1800 s) before giving up. Two failed probes cost ~60 idle minutes.

**Cause.** `src/sweep_parallel.py`'s `run_worker()` calls
`wait_for_server(port, args.serve_timeout)` without checking `server.poll()`. The same file
already has exactly this guard in the bench loop (`if server.poll() is not None: ... aborting
remaining shapes`) — it is a one-line omission in the startup path.

**Status: NOT FIXED** (shared harness code, not AMD-specific — flagged for the maintainers).
On SLURM this holds an allocation idle for 30 min per dead engine, which is worse than it was
here.

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

## 10. Smaller things

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
