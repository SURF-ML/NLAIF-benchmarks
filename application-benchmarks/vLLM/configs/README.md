# Cluster & model configs

Each cluster has a directory `configs/<cluster>/` (e.g. `snellius/h100`)
containing:

```
configs/<cluster>/
├── system.sh                 # cluster environment (partition, container, GPUS_PER_NODE, HF_HOME)
├── llama-70b-fp8/
│   ├── serve-params.json     # vllm serve sweep (parallelism × max_num_seqs)
│   └── bench-params.json     # vllm bench sweep (max_concurrency × 3 workload shapes)
├── gpt-oss-120b/
│   └── …
└── deepseek-v4-pro/
    └── …
```

`./run.sh <cluster>` sources `system.sh` and runs every `<model>/` dir in one sweep.

The `snellius/h100` config (NVIDIA, 4 GPUs/node) is the worked reference; copy it and adjust for
your platform (on ROCm, use `--rocm` in `CONTAINER_EXEC`).

## system.sh

Defines the cluster. Every value falls back to a pre-set environment variable, so users
override without editing the file:

| variable          | meaning                                                          |
|-------------------|------------------------------------------------------------------|
| `GPUS_PER_NODE`   | GPUs per node. Serve configs needing more span nodes via Ray.    |
| `SBATCH_ARGS`     | sbatch flags for worker jobs (partition, account, QOS).          |
| `GPU_REQUEST_TMPL`| GPU-per-node request template; `{n}` = per-node count. Default `--gpus-per-node={n}`; use `--gres=gpu:{n}` for gres-based schedulers. |
| `HF_HOME`         | HuggingFace cache (model weights), shared across nodes.          |
| `CONTAINER_EXEC`  | container prefix, e.g. `apptainer exec --nv --bind /path/to/shared …sif`. |

To add a cluster: copy an existing `system.sh`, adjust the values, then generate
its model configs (below).

## serve-params.json / bench-params.json

Plain JSON lists. The orchestrator cross-joins them with
`--link-vars max_num_seqs=max_concurrency,model=model`, so one serve config pairs with the
benches that share its model and concurrency. Any key becomes a `vllm` flag
(`tensor-parallel-size` → `--tensor-parallel-size`, `enable-expert-parallel: true` →
`--enable-expert-parallel`), so model-specific serve flags live right in the JSON. Keys
starting with `_` are **meta-keys**, not flags: `_extra_serve_args` is appended to
`vllm serve` verbatim — used for flags the `--key=value` builder can't express, e.g.
DeepSeek-V4's dotted `-cc.*` compilation flags.

**serve** entry — parallelism layout + GPU-side concurrency cap:
```json
{"model": "nvidia/Llama-3.3-70B-Instruct-FP8",
 "tensor-parallel-size": 2, "max_num_seqs": 16}
```
**bench** entry — client load + workload shape (dataset on `/dev/shm`):
```json
{"model": "nvidia/Llama-3.3-70B-Instruct-FP8", "max_concurrency": 16, "num_prompts": 32,
 "dataset-path": "/dev/shm/llama70b_i4608_o4608_n1024.jsonl", "custom-output-len": 4608}
```

## Generating configs

These files are mechanical, so author them with the generator instead of by hand:

```bash
python scripts/gen_configs.py                      # all clusters/models
python scripts/gen_configs.py --cluster snellius/h100
```

Edit the `MODELS` / `CLUSTERS` tables at the top of
[`scripts/gen_configs.py`](../scripts/gen_configs.py) to change the model checkpoint,
precision, parallelism sweep, or per-cluster overrides (e.g. DeepSeek uses TP4×PP4 on
4-GPU nodes but TP8 single-node on 8-GPU nodes), then re-run.

## Precision policy

One fixed precision per model (encoded in the checkpoint). See
[README §3](../README.md#3-models--precision-policy) for the authoritative model and precision
table — do not duplicate it here.
