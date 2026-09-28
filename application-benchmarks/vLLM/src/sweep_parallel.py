#!/usr/bin/env python3
"""
Parallel vLLM benchmark sweep orchestrator.

Replaces `vllm bench sweep serve` with individual sbatch-based workers —
one worker per serve config. Each worker is submitted as its own SLURM job
requesting only the GPUs it needs. Results are written as `summary.json`
files compatible with `vllm bench sweep plot`.

Two modes
---------
Orchestrator (default):
    Reads serve/bench params, applies link-vars pairing, then submits one
    `sbatch` job per serve config (layout), carrying all its linked benches. Polls squeue.

Worker (--worker flag, launched by sbatch):
    Starts one `vllm serve`, runs its bench configs against it in order, merges serve+bench
    params into the result record, writes summary.json, then stops the server.
    The server's own stdout/stderr is captured to serve.log next to the results.

Usage
-----
    python sweep_parallel.py \\
        --serve-params sweep_configs/serve-params-small.json \\
        --bench-params sweep_configs/bench-params-small.json \\
        --serve-cmd  "vllm serve --kv-cache-dtype auto ..." \\
        --bench-cmd  "vllm bench serve --backend vllm ..." \\
        --link-vars  model=model,max_num_seqs=_n \\
        --results-dir results/20250227_120000 \\
        --sbatch-args "-p gpu_h100 --account=myaccount"
"""

# Runs on the host python, whichever that is per cluster (ORCH_PYTHON in system.sh). Lazy
# annotations keep `str | None`-style hints parseable down to Python 3.7.
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_link_vars(s: str) -> dict[str, str]:
    """'a=b,c=d' → {'a': 'b', 'c': 'd'}"""
    if not s:
        return {}
    out = {}
    for pair in s.split(","):
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _norm(k: str) -> str:
    """Treat hyphens and underscores as equivalent."""
    return k.replace("-", "_")


def _get(d: dict, key: str):
    return d.get(key, d.get(_norm(key)))


def configs_match(serve: dict, bench: dict, link_vars: dict) -> bool:
    return all(_get(serve, sk) == _get(bench, bk)
               for sk, bk in link_vars.items())


def point_selected(serve: dict, bench: dict) -> bool:
    """A layout may restrict itself to some operating points via its `_points` meta-key."""
    points = serve.get("_points")
    return points is None or bench.get("_point") in points


def n_gpus_for(serve: dict) -> int:
    tp = _get(serve, "tensor-parallel-size") or 1
    pp = _get(serve, "pipeline-parallel-size") or 1
    dp = _get(serve, "data-parallel-size") or 1
    return int(tp) * int(pp) * int(dp)


def nodes_for(n_gpus: int, gpus_per_node: int) -> tuple[int, int]:
    """Map a GPU count onto (n_nodes, gpus_per_node_request).

    Single node when the request fits one node (or gpus_per_node is unknown, i.e. 0).
    Multi-node requires an exact multiple of gpus_per_node — partial-node spreads are
    rejected early so a misconfigured sweep fails before any allocation is made.
    """
    if gpus_per_node <= 0 or n_gpus <= gpus_per_node:
        return 1, n_gpus
    if n_gpus % gpus_per_node != 0:
        raise ValueError(
            f"n_gpus={n_gpus} is not a multiple of gpus_per_node={gpus_per_node}; "
            "multi-node configs must request whole nodes"
        )
    return n_gpus // gpus_per_node, gpus_per_node


def apply_params_to_cmd(cmd: str, params: dict) -> str:
    """Append params dict as CLI flags to a command string."""
    extra = []
    for k, v in params.items():
        if k.startswith("_"):
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            extra.append(flag if v else "--no-" + k.replace("_", "-"))
        else:
            extra.append(f"{flag}={v}")
    return (cmd.strip() + " " + " ".join(extra)).strip()


_SERVE_STALL_S = 1200


def wait_for_server(port: int, timeout: int = 600, server: "subprocess.Popen | None" = None,
                    log: "pathlib.Path | None" = None):
    """Poll /health until it responds, the server process exits, or `timeout` elapses.

    Without the `server` check this blocks for the FULL timeout (up to 5400s for a
    multi-node DeepSeek-V4 config) even when the engine already crashed on startup and
    the process is long gone — measured directly on this cluster. Checking server.poll()
    lets a startup crash fail in seconds instead of holding the allocation for ~90 minutes.
    """
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    # A startup that stops logging entirely is wedged, not slow: a healthy DeepSeek-V4 load
    # logs continuously (shard progress, kernel imports, and a 60 s shm_broadcast heartbeat
    # while ranks wait). Measured here: an engine silent after "Connected to Ray cluster" for
    # 45+ min never recovered, and would otherwise hold both nodes for the full timeout.
    last_size, last_growth = -1, time.time()
    while time.time() < deadline:
        if server is not None and server.poll() is not None:
            raise RuntimeError(
                f"Server on port {port} exited (code {server.returncode}) before becoming ready"
            )
        if log is not None:
            size = log.stat().st_size if log.exists() else 0
            if size != last_size:
                last_size, last_growth = size, time.time()
            elif time.time() - last_growth > _SERVE_STALL_S:
                raise RuntimeError(
                    f"Server on port {port} stalled: {log.name} unchanged for {_SERVE_STALL_S}s"
                )
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Server on port {port} not ready after {timeout}s")


def _is_healthy(port: int) -> bool:
    """True if the server's /health endpoint responds within a short timeout.

    A dead/wedged EngineCore makes /health fail (or hang, caught by the timeout), so this
    is the signal used to abort a bench that would otherwise block forever on requests that
    never return (e.g. after a rank crashes mid-run)."""
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
        return True
    except Exception:
        return False


def combo_subdir(serve: dict, bench: dict) -> str:
    """
    Build a filesystem-safe subdirectory name encoding all params.
    Mirrors the key=value concatenation that vllm bench sweep serve uses,
    but truncated+hashed if too long for the filesystem.
    """
    parts = [f"{k}={v}" for k, v in sorted(serve.items()) if not k.startswith("_")]
    parts += [f"{k}={v}" for k, v in sorted(bench.items()) if not k.startswith("_")]
    name = ",".join(parts).replace("/", "--")
    if len(name) > 200:
        import hashlib
        name = name[:180] + "_" + hashlib.md5(name.encode()).hexdigest()[:8]
    return name


# ── Ray multi-node cluster ──────────────────────────────────────────────────────
# Serve configs whose TP×PP×DP exceeds one node are served by a single logical vLLM
# process spanning the allocation. Ray is the cross-node transport: the worker (on the
# head node) places one Ray daemon per node via `srun`, inside the same container as
# vLLM. `srun apptainer exec …` (srun OUTSIDE the container) is the only shape that
# propagates the allocation + env across nodes. Single-node configs never touch this.

class RayCtx:
    """Handles for a Ray cluster started across the worker's sbatch allocation."""

    def __init__(self, head_proc, worker_proc, gcs_address, container, env, nodes,
                 head_container=None, container_engine=None, worker_container=None):
        self.head_proc = head_proc
        self.worker_proc = worker_proc      # may be None when only the head node exists
        self.gcs_address = gcs_address
        self.container = container          # container-exec prefix as a list (may be [])
        self.env = env
        self.nodes = nodes
        self.head_container = head_container    # head's own container name, podman/docker only
        self.container_engine = container_engine  # "podman"/"docker", or None (e.g. apptainer)
        self.worker_container = worker_container  # worker nodes' container name, podman/docker only


def _slurm_nodes() -> list[str]:
    """Expand SLURM_JOB_NODELIST (e.g. 'gcn[001-004]') into a list of hostnames."""
    nodelist = os.environ["SLURM_JOB_NODELIST"]
    out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist]).decode()
    return out.split()


def _host_ip(hostname: str) -> str:
    try:
        out = subprocess.check_output(["getent", "hosts", hostname]).decode().split()
        if out:
            return out[0]
    except Exception:
        pass
    return socket.gethostbyname(hostname)


def _ray_total_gpus(container: list[str], gcs_address: str, env: dict) -> float:
    """Total GPUs the Ray cluster currently advertises (the y in 'x.0/y.0 GPU')."""
    try:
        out = subprocess.run(
            [*container, "ray", "status", "--address", gcs_address],
            capture_output=True, text=True, env=env, timeout=30,
        ).stdout
    except Exception:
        return 0.0
    m = re.search(r"([\d.]+)\s*/\s*([\d.]+)\s+GPU", out)
    return float(m.group(2)) if m else 0.0


def start_ray_cluster(args, ray_port: int, gpus_per_node: int) -> "RayCtx":
    nodes = _slurm_nodes()
    head_node = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    head_ip = _host_ip(head_node)
    gcs_address = f"{head_ip}:{ray_port}"
    container = shlex.split(args.container_exec) if args.container_exec else []
    env = os.environ | {"RAY_ADDRESS": gcs_address}

    # podman/docker isolate each container's /tmp, hostname, etc., which breaks Ray's
    # same-node driver attachment for a later *separate* container running vllm serve (see
    # run_worker) — apptainer shares the host's namespaces by default and never hits this,
    # so it keeps the original fresh-invocation behavior untouched (engine stays None).
    container_engine = container[0] if container and container[0] in ("podman", "docker") else None
    run_id = os.environ.get('SLURM_JOB_ID', os.getpid())
    head_container = f"ray-head-{run_id}" if container_engine else None
    worker_container = f"ray-worker-{run_id}" if container_engine else None
    name_flag = ["--name", head_container] if head_container else []
    worker_name_flag = ["--name", worker_container] if worker_container else []

    print(f"[worker] Ray: {len(nodes)} node(s), head={head_node}({head_ip}) "
          f"gcs={gcs_address} gpus/node={gpus_per_node}", flush=True)

    # Ray head on the local (head) node. --block keeps the daemon in this srun step's
    # foreground so the step handle is our teardown lever (no daemonised orphans).
    head_cmd = [
        "srun", "--nodes=1", "--ntasks=1", f"--nodelist={head_node}",
        "--overlap", "--export=ALL",
        *container[:-1], *name_flag, *container[-1:],
        "ray", "start", "--head", "--disable-usage-stats",
        f"--node-ip-address={head_ip}", f"--port={ray_port}",
        f"--num-gpus={gpus_per_node}", "--block",
    ]
    head_proc = subprocess.Popen(head_cmd, start_new_session=True, env=env)

    # Ray workers on every other allocated node (one srun step, one task per node).
    worker_nodes = [h for h in nodes if h != head_node]
    worker_proc = None
    if worker_nodes:
        worker_cmd = [
            "srun", f"--nodes={len(worker_nodes)}", "--ntasks-per-node=1",
            f"--nodelist={','.join(worker_nodes)}", "--overlap", "--export=ALL",
            *container[:-1], *worker_name_flag, *container[-1:],
            "ray", "start", "--address", gcs_address,
            f"--num-gpus={gpus_per_node}", "--block",
        ]
        worker_proc = subprocess.Popen(worker_cmd, start_new_session=True, env=env)

    ctx = RayCtx(head_proc, worker_proc, gcs_address, container, env, nodes,
                 head_container, container_engine, worker_container)

    # Wait until every node's GPUs have registered (GPU count is a stricter
    # readiness signal than node count — nodes appear before their GPUs do).
    expected = len(nodes) * gpus_per_node
    deadline = time.time() + args.ray_timeout
    while time.time() < deadline:
        if head_proc.poll() is not None:
            stop_ray_cluster(ctx)
            raise RuntimeError("Ray head exited before the cluster became ready")
        if _ray_total_gpus(container, gcs_address, env) >= expected:
            print(f"[worker] Ray ready: {expected} GPUs across {len(nodes)} node(s)",
                  flush=True)
            return ctx
        time.sleep(3)

    stop_ray_cluster(ctx)
    raise TimeoutError(
        f"Ray cluster did not reach {expected} GPUs within {args.ray_timeout}s"
    )


def stop_ray_cluster(ctx: "RayCtx") -> None:
    """Graceful `ray stop` broadcast, then hard-kill the held --block srun steps."""
    if ctx.container_engine:
        # podman/docker: `ray stop` in a fresh container sees no Ray processes, and killing the
        # srun client below doesn't reach a rootless container either (it lives in the user's
        # systemd slice, not the job's cgroup) — measured directly on this cluster: every job
        # leaked its Ray containers, and a leftover raylet on the shared host network wedged
        # the next job's engine right after "Connected to Ray cluster". Remove them by name.
        # `rm -f` can give up ("given PID did not die within timeout") while the GPU driver is
        # still releasing the engine's ~260 GiB/GPU — measured here at ~3 min. Retry until the
        # containers are really gone, so this job holds the nodes until their GPUs are free and
        # the next job's startup memory check doesn't see them still allocated.
        present = [n for n in (ctx.head_container, ctx.worker_container) if n]
        names = " ".join(present)
        filters = " ".join(f"--filter name=^{n}$" for n in present)
        eng = ctx.container_engine
        # One `rm` per name: each node has only one of these containers, and `podman rm -f
        # missing present` silently removes nothing (exit 0) — measured here, it left every
        # worker-node container running.
        stop_cmd = ["sh", "-c",
                    f"for i in $(seq 60); do for n in {names}; do {eng} rm -f $n >/dev/null 2>&1; done; "
                    f"[ -z \"$({eng} ps -aq {filters})\" ] && exit 0; sleep 10; done; exit 0"]
    else:
        stop_cmd = [*ctx.container, "ray", "stop", "--force"]
    try:
        subprocess.run(
            ["srun", f"--nodes={len(ctx.nodes)}", "--ntasks-per-node=1",
             "--overlap", "--export=ALL", *stop_cmd],
            env=ctx.env, timeout=720,
        )
    except Exception:
        pass
    for p in (ctx.head_proc, ctx.worker_proc):
        if p is None:
            continue
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    # Escalate: a wedged collective can leave the held `srun --block` steps alive after SIGTERM.
    # Give them a moment, then SIGKILL anything still holding the allocation.
    deadline = time.time() + 15
    for p in (ctx.head_proc, ctx.worker_proc):
        if p is None:
            continue
        try:
            p.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


# ── worker ────────────────────────────────────────────────────────────────────

# How often to check the server while a bench is running, and how long the server may be
# unhealthy before we conclude the engine is dead and abort the bench. A crashed rank leaves
# the leader process alive but wedged, so a blocking bench would otherwise wait on in-flight
# requests forever (until the SLURM walltime), holding the whole allocation idle.
_BENCH_POLL_S = 15
_BENCH_UNHEALTHY_GRACE_S = 60


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL a Popen's whole process group, tolerating races."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def _run_bench_monitored(cmd: list[str], cwd: str, server: subprocess.Popen, port: int) -> str:
    """Run one bench while watching the serve process, returning its outcome.

      "ok"          bench exited 0
      "failed"      bench exited non-zero on its own (server still up)
      "server_dead" we killed the bench because the engine died/wedged

    The serve leader can stay alive but wedged after a rank crashes, leaving the bench
    blocked on requests that never return; polling /health (and the leader's exit) lets us
    abort within ~a minute instead of hanging to walltime."""
    proc = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
    unhealthy_since: float | None = None
    while True:
        try:
            rc = proc.wait(timeout=_BENCH_POLL_S)
            return "ok" if rc == 0 else "failed"
        except subprocess.TimeoutExpired:
            pass
        if server.poll() is not None:          # leader exited → engine gone
            _kill_group(proc)
            return "server_dead"
        if _is_healthy(port):
            unhealthy_since = None
        else:
            unhealthy_since = unhealthy_since or time.time()
            if time.time() - unhealthy_since >= _BENCH_UNHEALTHY_GRACE_S:
                _kill_group(proc)
                return "server_dead"


def _copy_datasets_to_shm(bench_configs: list[dict], dataset_dir: str | None):
    """Copy dataset files to /dev/shm on the current node."""
    import shutil

    if not dataset_dir:
        return
    seen: set[str] = set()
    for bc in bench_configs:
        dst = bc.get("dataset-path")
        if dst and dst not in seen:
            seen.add(dst)
            src = os.path.join(dataset_dir, os.path.basename(dst))
            if os.path.exists(dst):
                print(f"[worker] Dataset already present: {dst}", flush=True)
            elif os.path.exists(src):
                print(f"[worker] Copying dataset {src} → {dst}", flush=True)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
            else:
                print(f"[worker] WARNING: dataset not found at {src}", flush=True)


def run_worker(args):
    serve_config: dict = json.loads(pathlib.Path(args.serve_config).read_text())
    bench_configs: list[dict] = json.loads(pathlib.Path(args.bench_configs).read_text())
    results_dir = pathlib.Path(args.results_dir)
    port = args.port

    _copy_datasets_to_shm(bench_configs, args.dataset_dir)

    # Multi-node configs need a Ray cluster spanning the allocation before vLLM starts.
    # In that case the worker runs on the host (for srun/scontrol), so the vllm commands
    # must be wrapped in the container. Single-node workers already run inside it, so the
    # prefix is empty and the commands run directly as before.
    cmd_prefix: list[str] = []
    ray_ctx = None
    serve_env = os.environ | {"VLLM_SERVER_DEV_MODE": "1"}
    serve_cmd_str = apply_params_to_cmd(args.serve_cmd, serve_config) + f" --port {port}"
    # Per-model serve flags that can't go through the --key=value flag builder (e.g. vLLM's
    # dotted -cc.* compilation flags) are carried verbatim in the `_extra_serve_args` meta-key
    # and appended raw so they survive shlex.split as plain tokens.
    extra_serve_args = serve_config.get("_extra_serve_args", "").strip()
    if extra_serve_args:
        serve_cmd_str += " " + extra_serve_args
    if args.ray:
        ray_ctx = start_ray_cluster(args, port + 1000, args.ray_gpus_per_node)
        serve_cmd_str += " --distributed-executor-backend ray"
        serve_env = serve_env | {"RAY_ADDRESS": ray_ctx.gcs_address}
        if ray_ctx.head_container:
            # podman/docker: a fresh, separate container can't join this Ray cluster even
            # with --network=host — Ray's driver-side "is there a raylet on my node" check
            # keys off node-local state (under /tmp/ray) that a second container never sees,
            # so it looks like a different, raylet-less node and refuses to attach ("No node
            # info found matching attributes"). `podman exec` into the head's own
            # already-running container sidesteps this: it already has every mount/env var
            # from CONTAINER_RUN_ARGS baked in from when it was created, so only the
            # (dynamically-known-only-now) RAY_ADDRESS needs adding here.
            cmd_prefix = [ray_ctx.container_engine, "exec",
                          "-e", f"RAY_ADDRESS={ray_ctx.gcs_address}", ray_ctx.head_container]
        elif args.container_exec:
            cmd_prefix = shlex.split(args.container_exec)  # apptainer: original behavior

    # Capture the engine's own stdout/stderr next to the results instead of letting it
    # inherit into the SLURM log. serve.log holds vLLM's startup report, including
    #   "Maximum concurrency for <max-model-len> tokens per request: X.XXx"
    # which is how a vendor picks max_num_seqs (N) and how organisers audit that a declared
    # N is consistent with the KV cache the engine actually allocated. The worker's own
    # prints still go to the SLURM log, so progress remains visible there.
    serve_log = results_dir / combo_subdir(serve_config, bench_configs[0]) / "serve.log"
    serve_log.parent.mkdir(parents=True, exist_ok=True)

    print(f"[worker port={port}] Starting server ...", flush=True)
    print(f"[worker port={port}]   {serve_cmd_str}", flush=True)
    print(f"[worker port={port}]   server log → {serve_log}", flush=True)

    with serve_log.open("wb") as serve_log_fh:
        server = subprocess.Popen(
            [*cmd_prefix, *shlex.split(serve_cmd_str)],
            start_new_session=True,      # own process group for clean kill
            env=serve_env,
            stdout=serve_log_fh,
            stderr=subprocess.STDOUT,
        )

    server_died = False
    try:
        wait_for_server(port, args.serve_timeout, server=server, log=serve_log)
        print(f"[worker port={port}] Server ready", flush=True)

        for bench_config in bench_configs:
            # If the serve process has exited, the engine is gone (e.g. a rank crashed). Don't
            # grind the rest of the sweep against a dead server — abort so SLURM frees the nodes
            # instead of holding them idle until walltime.
            if server.poll() is not None:
                print(f"[worker port={port}] ERROR: server exited (code {server.returncode}) "
                      f"before bench — aborting remaining shapes", flush=True)
                server_died = True
                break
            # Created under results_dir (not the default /tmp) so it's reachable from inside a
            # multi-node podman-exec bench invocation too — see bench_prefix below.
            with tempfile.TemporaryDirectory(dir=str(results_dir)) as tmp:
                bench_cmd_str = (
                    apply_params_to_cmd(args.bench_cmd, bench_config)
                    + f" --host localhost --port {port}"
                )
                print(f"[worker port={port}] Bench: {bench_cmd_str}", flush=True)

                # Run bench with cwd=tmp so --save-result lands there, under a health monitor:
                # a single bench failure is isolated so the remaining shapes still run, but if the
                # serve engine dies/wedges we abort the whole worker so SLURM frees the nodes
                # instead of the bench blocking on never-returning requests until walltime.
                #
                # `cwd=tmp` below only sets the *host* podman-client process's directory — podman
                # exec does not inherit it for the process it starts inside the container, which
                # instead defaults to the container's own configured workdir ($REPO_ROOT). Without
                # an explicit --workdir here, --save-result silently writes there instead of `tmp`,
                # and this measurement's result is never found (measured directly on this cluster:
                # the bench completed successfully — 16/16 requests — but its JSON was never seen).
                # `tmp` lives under results_dir, itself under REPO_ROOT, so the same path resolves
                # identically inside the container (bind-mounted) and outside it.
                bench_prefix = cmd_prefix
                if ray_ctx is not None and ray_ctx.head_container:
                    bench_prefix = [cmd_prefix[0], cmd_prefix[1], "--workdir", tmp, *cmd_prefix[2:]]
                outcome = _run_bench_monitored(
                    [*bench_prefix, *shlex.split(bench_cmd_str)], tmp, server, port)
                if outcome == "server_dead" or (outcome == "failed" and not _is_healthy(port)):
                    print(f"[worker port={port}] ERROR: server died/wedged during bench — "
                          f"aborting remaining shapes", flush=True)
                    server_died = True
                    break
                if outcome == "failed":
                    print(f"[worker port={port}] WARNING: bench failed — skipping", flush=True)
                    continue

                result_files = sorted(pathlib.Path(tmp).glob("*.json"))
                if not result_files:
                    print(f"[worker port={port}] WARNING: no result JSON — skipping", flush=True)
                    continue

                record: dict = json.loads(result_files[0].read_text())

            # Merge serve + bench params so sweep plot can group/filter by them
            record.update(serve_config)
            record.update(bench_config)

            subdir = results_dir / combo_subdir(serve_config, bench_config)
            subdir.mkdir(parents=True, exist_ok=True)
            summary = subdir / "summary.json"
            existing = json.loads(summary.read_text()) if summary.exists() else []
            existing.append(record)
            summary.write_text(json.dumps(existing, indent=2))
            print(f"[worker port={port}] Saved → {summary}", flush=True)

    finally:
        print(f"[worker port={port}] Stopping server (pid={server.pid})", flush=True)
        try:
            # Signal only the vLLM leader (not the whole process group): vLLM's own SIGTERM
            # handler tears down the EngineCore + API servers in order, which avoids the
            # EngineDeadError ("AsyncLLM output_handler failed") and the leaked semaphore /
            # shared-memory warnings that a simultaneous group-kill (killpg) produces by racing
            # the engine's death against the API servers' graceful shutdown.
            server.terminate()
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            # Graceful shutdown stalled — hard-kill the whole group so nothing is orphaned.
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.wait()
        except ProcessLookupError:
            pass
        if ray_ctx is not None:
            print(f"[worker port={port}] Stopping Ray cluster", flush=True)
            stop_ray_cluster(ray_ctx)

    # A dead engine means the sweep is incomplete. Exit non-zero so SLURM marks the job FAILED
    # rather than reporting success; for a multi-node job, also cancel it outright so a wedged
    # collective can't keep the allocation alive after teardown.
    if server_died:
        job_id = os.environ.get("SLURM_JOB_ID")
        if ray_ctx is not None and job_id:
            print(f"[worker port={port}] Cancelling job {job_id} to release the allocation",
                  flush=True)
            subprocess.run(["scancel", job_id], check=False)
        sys.exit(1)


# ── sbatch helpers ─────────────────────────────────────────────────────────────

def _submit_group(i, serve_config, bench_configs, port, n, tmp_root, results_dir, args) -> str:
    """Write config files, submit an sbatch job, return SLURM job ID.

    `bench_configs` holds every bench linked to this serve config (layout), in ascending
    concurrency: the worker starts the engine once and runs them in that order.
    """
    serve_file = tmp_root / f"serve_{i}.json"
    bench_file = tmp_root / f"bench_{i}.json"
    serve_file.write_text(json.dumps(serve_config))
    bench_file.write_text(json.dumps(bench_configs))

    log_out = results_dir / "slurm" / f"worker_{i}.out"
    log_err = results_dir / "slurm" / f"worker_{i}.err"
    log_out.parent.mkdir(parents=True, exist_ok=True)

    nodes, gpn = nodes_for(n, args.gpus_per_node)
    multinode = nodes > 1

    # Single-node workers run *inside* the container — they exec `vllm serve`/`vllm bench`
    # directly. Multi-node workers must run on the *host*: they drive Ray across the
    # allocation with `srun`/`scontrol`, which cannot run inside the container. They use the
    # host python and wrap only the inner vllm/ray commands with --container-exec instead.
    python_prefix = ([sys.executable] if multinode
                     else (shlex.split(args.container_exec) + ["python3"]
                           if args.container_exec else [sys.executable]))

    # The worker runs with cwd=<script dir> (see `wrap` below), so every path handed to it
    # must be absolute — a relative results-dir (the default `results/<ts>`) would otherwise
    # resolve against the wrong directory and the config files would not be found.
    worker_argv = [
        *python_prefix,
        str(pathlib.Path(__file__).resolve()),
        "--worker",
        "--serve-config", str(serve_file.resolve()),
        "--bench-configs", str(bench_file.resolve()),
        "--serve-cmd", args.serve_cmd,
        "--bench-cmd", args.bench_cmd,
        "--port", str(port),
        "--serve-timeout", str(args.serve_timeout),
        "--results-dir", str(pathlib.Path(results_dir).resolve()),
        *(["--dataset-dir", str(pathlib.Path(args.dataset_dir).resolve())]
          if args.dataset_dir else []),
        # Multi-node: the worker fans Ray out across the allocation and needs the
        # container-exec prefix to wrap each per-node `ray start`.
        *(["--ray", "--ray-gpus-per-node", str(gpn),
           "--ray-timeout", str(args.ray_timeout),
           *(["--container-exec", args.container_exec] if args.container_exec else [])]
          if multinode else []),
    ]

    script_dir = str(pathlib.Path(__file__).resolve().parent)
    wrap = f"cd {shlex.quote(script_dir)} && " + " ".join(shlex.quote(str(a)) for a in worker_argv)

    # Single-node requests one task; multi-node requests whole nodes with one task each, so the
    # worker (on the head node) can place Ray itself. The per-node GPU request uses the cluster's
    # template ('--gpus-per-node={n}' by default; '--gres=gpu:{n}' etc. for gres-based schedulers).
    gpu_req = shlex.split(args.gpu_request_tmpl.format(n=gpn))
    node_args = (
        ["--nodes=1", "--ntasks=1", *gpu_req] if not multinode
        else [f"--nodes={nodes}", "--ntasks-per-node=1", *gpu_req]
    )
    worker_time = args.worker_time or ("12:00:00" if multinode else "08:00:00")

    sbatch_cmd = [
        "sbatch", "--parsable", f"--time={worker_time}",
        f"--job-name=vllm-worker-{i}",
        *node_args,
        f"--output={log_out}",
        f"--error={log_err}",
        *shlex.split(args.sbatch_args or ""),
        f"--wrap={wrap}",
    ]

    if args.dry_run:
        print(f"[dry-run] group {i}: {' '.join(node_args)} time={worker_time} "
              f"ray={'yes' if multinode else 'no'}")
        return f"DRYRUN-{i}"

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


# States a job cannot move out of. Anything else (PENDING, RUNNING, CONFIGURING,
# …) means SLURM has not caught up yet, not that the job is over.
_TERMINAL_STATES = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED",
})


def _poll_job_states(job_ids: list[str]) -> dict[str, tuple[str, str]] | None:
    """Query squeue; return {job_id: (slurm_state, nodelist)} for active jobs.

    Returns None when squeue could not be reached — which is NOT the same as an
    empty dict (all jobs have left the queue). Conflating the two let a single
    slow squeue, routine once a campaign has a dozen multi-node jobs queued,
    mark every worker — running and pending alike — as failed and abort the
    whole sweep seconds after submitting it.
    """
    if not job_ids:
        return {}
    try:
        out = subprocess.check_output(
            ["squeue", f"--jobs={','.join(job_ids)}", "--noheader", "--format=%i %T %N"],
            stderr=subprocess.DEVNULL, timeout=60,
        ).decode()
    except Exception:
        return None
    result = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            result[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return result


def _get_final_state(job_id: str) -> str:
    """Use sacct to resolve the final state of a job that left squeue."""
    try:
        out = subprocess.check_output(
            ["sacct", f"--jobs={job_id}", "--noheader", "--parsable2", "--format=State"],
            stderr=subprocess.DEVNULL, timeout=60,
        ).decode().strip()
        lines = [l for l in out.splitlines() if l]
        return lines[0].split()[0] if lines else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


# ── orchestrator ──────────────────────────────────────────────────────────────

def _load_params(config_root: str | None, explicit: str | None, kind: str) -> list[dict]:
    """Load a params list, either from an explicit file or by globbing a config root.

    With --config-root <dir>, every <dir>/*/<kind>-params.json (one per model) is
    concatenated — so a whole cluster's models sweep in a single orchestrator run.
    """
    if config_root:
        files = sorted(pathlib.Path(config_root).glob(f"*/{kind}-params.json"))
        if not files:
            raise FileNotFoundError(f"No */{kind}-params.json under {config_root}")
        merged: list[dict] = []
        for f in files:
            merged.extend(json.loads(f.read_text()))
        return merged
    if not explicit:
        raise ValueError(f"Provide --{kind}-params or --config-root")
    return json.loads(pathlib.Path(explicit).read_text())


def run_orchestrator(args):
    serve_params = _load_params(args.config_root, args.serve_params, "serve")
    bench_params = _load_params(args.config_root, args.bench_params, "bench")
    link_vars = parse_link_vars(args.link_vars)

    # One worker per serve config (layout): one engine start serves all of that layout's cells.
    # Loading the model dominated a job-per-cell sweep — measured on the Snellius H100 reference
    # (DeepSeek-V4-Pro, N=32), engine start/stop took 51% of all job time, more than the
    # measurements themselves. Cells run in ascending concurrency (low before high), so a crash or
    # OOM at the high point, where they happen, loses only the cells still to run; the worker
    # aborts the rest of its list when the engine dies.
    groups: list[tuple[dict, list[dict]]] = [
        (s, sorted((b for b in bench_params
                    if configs_match(s, b, link_vars) and point_selected(s, b)),
                   key=lambda b: _get(b, "max_concurrency") or 0))
        for s in serve_params
    ]
    groups = [(s, benches) for s, benches in groups if benches]

    print(f"[orchestrator] {len(serve_params)} serve configs / {len(bench_params)} benches "
          f"→ {len(groups)} sbatch workers (one per layout, "
          f"{sum(len(b) for _, b in groups)} measurements)")

    results_dir = pathlib.Path(
        args.results_dir or f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[orchestrator] Results dir: {results_dir}")

    tmp_root = results_dir.parent.parent / "tmp" / results_dir.name
    tmp_root.mkdir(parents=True, exist_ok=True)

    job_ids: dict[int, str] = {}   # pair index → SLURM job ID
    status: dict[int, dict] = {}   # pair index → {state, nodes}
    cancelled = False

    def _cancel_all(signum, frame):
        nonlocal cancelled
        cancelled = True
        print("\n[orchestrator] Interrupted — cancelling all jobs ...", flush=True)
        for jid in job_ids.values():
            subprocess.run(["scancel", jid], stderr=subprocess.DEVNULL)
        sys.exit(1)

    signal.signal(signal.SIGINT, _cancel_all)
    # SIGTERM is left as default: orchestrator job ending (e.g. time limit) does NOT
    # cancel independent worker jobs — they continue running to completion.

    for i, (serve_config, bench_configs) in enumerate(groups):
        if cancelled:
            break
        port = args.base_port + i
        n = n_gpus_for(serve_config)
        model = serve_config.get("model", "?")
        tp = _get(serve_config, "tensor-parallel-size") or 1
        print(f"[orchestrator] Submitting group {i}: n_gpus={n} port={port} "
              f"model={model} TP={tp} benches={len(bench_configs)}")

        try:
            job_id = _submit_group(i, serve_config, bench_configs, port, n, tmp_root, results_dir, args)
        except subprocess.CalledProcessError as e:
            print(f"[orchestrator] ERROR: sbatch failed for group {i}: {e.stderr}", flush=True)
            status[i] = {"state": "FAILED", "nodes": ""}
            continue

        job_ids[i] = job_id
        status[i] = {"state": "pending", "nodes": ""}
        if not args.dry_run:
            print(f"[orchestrator]   → job {job_id}", flush=True)

    if args.dry_run:
        print(f"[dry-run] would submit {len(job_ids)} worker jobs. No jobs were submitted.")
        return results_dir

    if not job_ids:
        print("[orchestrator] No jobs submitted.")
        return results_dir

    STATE_ICON = {"pending": "⏳", "running": "🟢", "done": "✅", "FAILED": "❌", "cancelled": "🚫"}

    print(f"[orchestrator] Waiting for {len(job_ids)} jobs ...", flush=True)
    poll_failures = 0
    absent_counts: dict[int, int] = {}
    while True:
        active_ids = [jid for i, jid in job_ids.items()
                      if status[i]["state"] not in ("done", "FAILED", "cancelled")]

        if active_ids:
            sq = _poll_job_states(active_ids)
            if sq is None:
                # squeue was unreachable this cycle. Leave every job's state
                # untouched and retry — a failed query must never be read as
                # "all jobs finished".
                poll_failures += 1
                if poll_failures in (5, 20) or (poll_failures and poll_failures % 100 == 0):
                    print(f"\n[orchestrator] WARNING: squeue has failed {poll_failures} "
                          f"consecutive times; still waiting on {len(active_ids)} job(s).",
                          flush=True)
            else:
                poll_failures = 0
                for i, jid in job_ids.items():
                    if status[i]["state"] in ("done", "FAILED", "cancelled"):
                        continue
                    if jid in sq:
                        slurm_state, nodes = sq[jid]
                        status[i]["nodes"] = nodes
                        absent_counts[i] = 0
                        if slurm_state in ("RUNNING", "COMPLETING"):
                            status[i]["state"] = "running"
                        # else stays "pending" (PENDING, CONFIGURING, etc.)
                    else:
                        # Job left squeue — resolve its final state. Only a terminal
                        # sacct state is conclusive: PENDING/RUNNING/UNKNOWN mean sacct
                        # is lagging squeue, so leave it and re-check next cycle — UNLESS
                        # sacct is unable to resolve ANY job here (accounting storage
                        # disabled is a real cluster config, not a transient lag — measured
                        # directly on this cluster: without this fallback the orchestrator
                        # polls forever, never reaching plot/score, for jobs that had
                        # already finished hours earlier). After a few consecutive cycles
                        # absent from squeue with no sacct answer, fall back to the result
                        # file the worker itself would have written — ground truth that
                        # doesn't depend on accounting at all.
                        final = _get_final_state(jid)
                        if final in _TERMINAL_STATES:
                            status[i]["state"] = "done" if final == "COMPLETED" else "FAILED"
                            status[i]["nodes"] = ""
                        else:
                            absent_counts[i] = absent_counts.get(i, 0) + 1
                            if absent_counts[i] >= 3:
                                serve_config, bench_configs = groups[i]
                                status[i]["state"] = ("done" if all(
                                    (results_dir / combo_subdir(serve_config, b)
                                     / "summary.json").exists() for b in bench_configs)
                                    else "FAILED")
                                status[i]["nodes"] = ""

        parts = []
        for i in sorted(status):
            s = status[i]
            icon = STATE_ICON.get(s["state"], "?")
            node = f"[{s['nodes']}]" if s["state"] == "running" and s["nodes"] else ""
            jid = f"/{job_ids[i]}" if i in job_ids else ""
            parts.append(f"{icon}#{i}{jid}{node}")
        sys.stdout.write(f"\r  {' '.join(parts)}   ")
        sys.stdout.flush()

        if all(s["state"] in ("done", "FAILED", "cancelled") for s in status.values()):
            break
        time.sleep(5)

    sys.stdout.write("\n")

    failed = [i for i, s in status.items() if s["state"] != "done"]
    if failed:
        print(f"[orchestrator] WARNING: groups {failed} did not complete successfully")
    else:
        print("[orchestrator] All workers completed successfully")

    print(f"[orchestrator] Results: {results_dir}")
    return results_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", action="store_true",
                   help="Run in worker mode (launched by sbatch, not by user)")

    # Orchestrator
    p.add_argument("--serve-params", help="Path to a serve-params JSON file")
    p.add_argument("--bench-params", help="Path to a bench-params JSON file")
    p.add_argument("--config-root",
                   help="Cluster config dir; merges every <root>/*/{serve,bench}-params.json "
                        "so all models sweep in one run. Overrides --serve/--bench-params.")
    p.add_argument("--link-vars", default="",
                   help="Serve→bench variable links, e.g. max_num_seqs=max_concurrency,model=model")
    p.add_argument("--results-dir", help="Output dir (default: results/TIMESTAMP)")
    p.add_argument("--base-port", type=int, default=8000,
                   help="First port; each worker gets base_port+i (default: 8000)")
    p.add_argument("--container-exec", default=None,
                   help="Container prefix inserted before python3 in worker jobs, "
                        "e.g. 'apptainer exec /path/to/vllm.sif'")
    p.add_argument("--sbatch-args", default="",
                   help="Extra sbatch flags for worker jobs, e.g. '-p gpu_h100 --account=foo'")
    p.add_argument("--dataset-dir", default=None,
                   help="Directory containing source dataset files to copy to /dev/shm on each worker node")
    p.add_argument("--gpus-per-node", type=int, default=0,
                   help="GPUs per node on this cluster. Serve configs needing more GPUs than "
                        "this span multiple nodes via Ray. 0 (default) = always single-node.")
    p.add_argument("--gpu-request-tmpl", default="--gpus-per-node={n}",
                   help="sbatch flag template for requesting GPUs per node; '{n}' is the per-node "
                        "GPU count. Default '--gpus-per-node={n}'. Override per cluster for "
                        "gres-based schedulers, e.g. '--gres=gpu:{n}' or '--gres=gpu:mi300:{n}'.")
    p.add_argument("--worker-time", default=None,
                   help="sbatch --time for worker jobs. Default: 08:00:00 single-node, "
                        "12:00:00 multi-node.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the worker allocation each group would request; submit nothing.")

    # Worker
    p.add_argument("--serve-config", help="(worker) Path to single serve-config JSON")
    p.add_argument("--bench-configs", help="(worker) Path to bench-configs list JSON")
    p.add_argument("--port", type=int, default=8000, help="(worker) Server port")
    p.add_argument("--ray", action="store_true",
                   help="(worker) Span the allocation with a Ray cluster (multi-node serve)")
    p.add_argument("--ray-gpus-per-node", type=int, default=0,
                   help="(worker) GPUs each Ray node advertises")
    p.add_argument("--ray-timeout", type=int, default=600,
                   help="(worker) Seconds to wait for all Ray nodes/GPUs to register")
    p.add_argument("--serve-timeout", type=int, default=1800,
                   help="Seconds to wait for `vllm serve` /health after launch. Large "
                        "checkpoints (e.g. DeepSeek-V4 across nodes) load slowly — raise "
                        "this (e.g. 5400) for them. Default: 1800.")

    # Shared
    p.add_argument("--serve-cmd", required=True, help="Base vllm serve command string")
    p.add_argument("--bench-cmd", required=True, help="Base vllm bench serve command string")

    args = p.parse_args()

    if args.worker:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()