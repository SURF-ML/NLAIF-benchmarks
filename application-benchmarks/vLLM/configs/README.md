# Cluster & model configs

Each cluster has a directory `configs/<cluster>/` (e.g. `snellius/h100`)
containing:

```
configs/<cluster>/
├── system.sh                 # cluster environment (partition, container, GPUS_PER_NODE, HF_HOME)
└── deepseek-v4-pro/
    ├── serve-params.json     # one entry per parallelism layout, each carrying its max batch size N
    └── bench-params.json     # one entry per (N, operating point, workload shape)
```

`./run.sh <cluster>` sources `system.sh` and runs every `<model>/` dir.

The `snellius/h100` config (NVIDIA, 4 GPUs/node) is the worked reference; copy it and adjust for
your platform (on ROCm, use `--rocm` in `CONTAINER_EXEC`).

Only DeepSeek-V4-Pro is scored. `gen_configs.py` also knows two unscored models (Llama-70B,
GPT-OSS-120B) that exist purely to smoke-test the harness on a single node — generate them with
`--only-model`; they are never written by default.

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

Plain JSON lists. The orchestrator pairs them with `--link-vars model=model,max_num_seqs=_n`,
so a layout is matched only with the operating points derived from **its own** max batch size N —
otherwise a small layout would also be benched at a large layout's concurrencies. Each matched
pair becomes **one SLURM job**, so every measurement runs against a freshly started engine.
Any key becomes a `vllm` flag
(`tensor-parallel-size` → `--tensor-parallel-size`, `enable-expert-parallel: true` →
`--enable-expert-parallel`), so model-specific serve flags live right in the JSON. Keys
starting with `_` are **meta-keys**, not flags: `_extra_serve_args` is appended to
`vllm serve` verbatim — used for flags the `--key=value` builder can't express, e.g.
DeepSeek-V4's dotted `-cc.*` compilation flags.

**serve** entry — parallelism layout + that layout's max batch size N:
```json
{"model": "deepseek-ai/DeepSeek-V4-Pro", "tensor-parallel-size": 4,
 "pipeline-parallel-size": 4, "max_num_seqs": 32}
```
**bench** entry — one operating point × workload shape (dataset on `/dev/shm`). `_point` labels
the point and `_n` records the N it was derived from, which is what pairs it with its layout:
```json
{"model": "deepseek-ai/DeepSeek-V4-Pro", "_point": "high", "_n": 32,
 "max_concurrency": 32, "num_prompts": 64,
 "dataset-path": "/dev/shm/dsv4_i1024_o8192_n1024.jsonl", "custom-output-len": 8192}
```

## Generating configs

These files are mechanical, so author them with the generator instead of by hand:

```bash
python scripts/gen_configs.py                      # all clusters/models
python scripts/gen_configs.py --cluster snellius/h100
```

Edit the `MODELS` / `CLUSTERS` / `SHAPES` / `POINTS` tables at the top of
[`scripts/gen_configs.py`](../scripts/gen_configs.py) to change the checkpoint, the workload
shapes, the operating points, or per-cluster overrides — the parallelism layouts and each
layout's `max_num_seqs` (N) live in `CLUSTERS` (e.g. DeepSeek uses TP4×PP4 on 4-GPU nodes but
TP8 single-node on 8-GPU nodes). Add entries to `parallels` to submit several layouts. Then
re-run.

## Precision policy

One fixed precision, encoded in the checkpoint. See
[README §3](../README.md#3-models--precision-policy) for the authoritative model and precision
table — do not duplicate it here.
