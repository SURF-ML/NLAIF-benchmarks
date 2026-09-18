# Snellius — 4× NVIDIA H100 (94 GiB) per node.
# Sourced by run.sh / scripts/vllm_sweep_parallel.sh with $REPO_ROOT already set.
# Every value honours a pre-set environment variable so users can override without
# editing this file (e.g. HF_HOME=/scratch/hf ./run.sh snellius/h100).

GPUS_PER_NODE=4
SBATCH_ARGS="${SBATCH_ARGS:--p gpu_h100}"
# Partition for the ORCHESTRATOR job (the coordinator that only sbatches workers and polls
# squeue — it uses no GPU). It must not land in a GPU partition: Snellius rejects a GPU-less
# job there ("You should request at least one GPU when running a job in a GPU partition"), and
# giving it a GPU would idle one for the length of the whole campaign. Defaults to SBATCH_ARGS
# in run.sh, which is a GPU partition, so set a CPU partition here. Only `./run.sh <cluster>`
# uses this; `--local` runs the orchestrator in your shell instead.
ORCH_SBATCH_ARGS="${ORCH_SBATCH_ARGS:--p rome}"
# How workers request GPUs per node. '{n}' is the per-node GPU count. The default
# '--gpus-per-node={n}' works here; clusters whose GRES is configured differently can override,
# e.g. GPU_REQUEST_TMPL='--gres=gpu:{n}' or '--gres=gpu:<type>:{n}'.
# (Assigned in two steps: a '{n}' inside a ${VAR:-default} expansion mis-parses the braces.)
[ -n "${GPU_REQUEST_TMPL:-}" ] || GPU_REQUEST_TMPL='--gpus-per-node={n}'

# Fraction of VRAM vLLM may use. Tunable per platform (different GPU memory sizes warrant
# different values to maximise KV cache without OOM); 0.95 is a sensible default.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
# Shared model cache, readable from every node (multi-node Ray workers all read HF_HOME). Set it to
# your shared scratch and make sure the --bind below covers that path. Exported so the sbatch workers
# inherit it (the HF tokenizer/cache lookup reads HF_HOME, not just --download-dir).
export HF_HOME="${HF_HOME:-/path/to/shared/hf_home}"

# Weights are pre-downloaded into HF_HOME, so run fully offline: vLLM/HF never contacts the Hub.
# This skips per-file etag checks at startup, avoids a request storm against the Hub from many
# workers, and silences the "sending unauthenticated requests to the HF Hub" warnings (no token).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Persist vLLM's torch.compile / inductor artifacts on shared storage so they are reused across runs
# and across nodes — warm starts skip recompilation. Keyed by model + parallel layout + GPU +
# vLLM version, so a populated cache only ever helps.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/path/to/shared/vllm_cache}"

# How long the API-server frontend waits for the EngineCore to finish loading before it gives up
# and kills itself. vLLM's default is 600s, which is too short for DeepSeek-V4-Pro: loading its
# 806 GiB checkpoint plus CUDA-graph memory profiling across 16 GPUs / 4 nodes takes ~10 min, so
# the frontend died seconds before the engine became ready (ApiServer_0 "died with exit code
# None"). 1800s gives ~3x margin. Harmless for the small models (ready in <60s, so it never fires).
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. It looks like a clean
# fix for the high-concurrency fragmentation OOM, but it makes vLLM's EngineCore initialisation
# crash during the determine_available_memory profiling step ("Engine core initialization failed
# ... Failed core proc(s): {}") — a known incompatibility (vllm-project/vllm#29544). Leave the
# CUDA allocator at its default.

# Pin NCCL to the node's high-speed inter-node NIC. The multi-node DeepSeek-V4 serve (TP4×PP4
# across 4 nodes) drives its tensor/pipeline collectives over NCCL; left to auto-select, NCCL
# bound the wrong interface here, so cross-node broadcasts/all-gathers never completed and the
# 600s watchdog killed the run (WorkNCCL ... BROADCAST/_ALLGATHER_BASE timeouts in the worker
# logs). eno2np0 is the per-node fabric interface on this cluster. Single-node serves ignore it.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno2np0}"

# IMPORTANT: forward the above into the container with --env. sbatch's env does NOT reliably reach
# inside apptainer for the worker jobs, so relying on inheritance silently drops HF_HUB_OFFLINE /
# VLLM_CACHE_ROOT (vLLM then falls back to ~/.cache/vllm and still pings the Hub). --env is the
# guaranteed mechanism. (To authenticate instead of going offline, add --env HF_TOKEN=$HF_TOKEN.)
CONTAINER_EXEC="${CONTAINER_EXEC:-apptainer exec --nv --bind /path/to/shared --env HF_HOME=$HF_HOME,HF_HUB_OFFLINE=$HF_HUB_OFFLINE,VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT,VLLM_ENGINE_READY_TIMEOUT_S=$VLLM_ENGINE_READY_TIMEOUT_S,NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME $REPO_ROOT/containers/latest.sif}"
