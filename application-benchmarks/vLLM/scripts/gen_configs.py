#!/usr/bin/env python3
"""
Generate the per-(cluster, model) serve-params.json / bench-params.json sweep files.

The benchmark sweeps two axes to trace a Pareto frontier of tok/s/GPU vs tok/s/user:
  * serve side:  parallelism layout  ×  max_num_seqs (the GPU-side concurrency cap)
  * bench side:  max_concurrency (client load)  ×  the 3 fixed workload shapes

Every model uses a single, fixed precision (the checkpoint encodes it):
  * Llama-3.3-70B  -> FP8        (nvidia/Llama-3.3-70B-Instruct-FP8)
  * GPT-OSS-120B   -> MXFP4      (openai/gpt-oss-120b, native)
  * DeepSeek-V4-Pro-> FP4+FP8    (deepseek-ai/DeepSeek-V4-Pro, native mixed)

Parallelism is per cluster because it depends on GPUs/node and GPU memory:
  * dense / small-MoE models (Llama, GPT-OSS) sweep tensor-parallel-size {1,2,4}.
  * DeepSeek-V4-Pro (~1.6T) needs ~16 GPUs. On 4-GPU nodes (the Snellius H100 example)
    that is TP=4 within a node + PP=4 across nodes (served via Ray). On 8-GPU nodes it is
    TP=8 on a single node, matching the upstream recipe.

This script IS the canonical way to (re)author configs — edit MODELS / CLUSTERS below
and re-run. Output paths: configs/<cluster>/<model>/{serve,bench}-params.json

Usage:
    python scripts/gen_configs.py            # write all clusters/models
    python scripts/gen_configs.py --cluster snellius/h100
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed workload shapes (input_len, output_len) and the dataset on /dev/shm.
SHAPES = [(1024, 8192), (4608, 4608), (8192, 1024)]
# Concurrency / max_num_seqs sweep (powers of two).
CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Reduced sweeps for pipeline sanity-checks. `parallels` is a slice applied to each model's
# full parallels list — debug/small pick the largest single-node layout (TP=4 for the dense
# models, much faster than TP=1); dsv4 has only one entry anyway, so every slice keeps its
# single TP4×PP4. Non-`full` profiles write to a suffixed cluster dir.
PROFILES = {
    "full":  {"concurrency": CONCURRENCY, "shapes": SHAPES,     "parallels": slice(None)},
    "debug": {"concurrency": [1],         "shapes": SHAPES[:1], "parallels": slice(-1, None)},
    "debug2":{"concurrency": [2],         "shapes": SHAPES,     "parallels": slice(-1, None)},
    "small": {"concurrency": [4, 32],     "shapes": SHAPES,     "parallels": slice(-1, None)},
    # DeepSeek-V4 concurrency curve: one fixed 16-GPU layout (TP4×PP4), max_num_seqs swept over
    # powers of two up to 32. Past ~32 concurrent requests a 1.6T model on 16 GPUs is KV-bound,
    # so the curve's useful range is 1..32. Pair with --only-model deepseek-v4-pro.
    "dsv4":  {"concurrency": [1, 2, 4, 8, 16, 32], "shapes": SHAPES, "parallels": slice(None)},
}


def _serve_entries(model: str, extra: dict, parallels: list[dict],
                   concurrency: list[int]) -> list[dict]:
    out = []
    for par in parallels:
        for mns in concurrency:
            out.append({"model": model, **extra, **par, "max_num_seqs": mns})
    return out


def _bench_entries(model: str, dataset_prefix: str,
                   concurrency: list[int], shapes: list[tuple[int, int]]) -> list[dict]:
    out = []
    for mns in concurrency:
        for in_len, out_len in shapes:
            out.append({
                "model": model,
                "max_concurrency": mns,
                "num_prompts": 2 * mns,
                "dataset-path": f"/dev/shm/{dataset_prefix}_i{in_len}_o{out_len}_n1024.jsonl",
                "custom-output-len": out_len,
            })
    return out


# Per-model definitions shared across clusters. `parallels` may be overridden per
# cluster (see CLUSTERS); `extra` are constant serve flags appended to every entry.
MODELS = {
    "llama-70b-fp8": {
        "model": "nvidia/Llama-3.3-70B-Instruct-FP8",
        "dataset_prefix": "llama70b",
        "extra": {},  # FP8 precision is encoded in the checkpoint; vLLM auto-detects it.
        "parallels": [{"tensor-parallel-size": 1},
                      {"tensor-parallel-size": 2},
                      {"tensor-parallel-size": 4}],
    },
    "gpt-oss-120b": {
        "model": "openai/gpt-oss-120b",
        "dataset_prefix": "gpt120b",
        "extra": {},  # native MXFP4 weights; no quantization flag
        "parallels": [{"tensor-parallel-size": 1},
                      {"tensor-parallel-size": 2},
                      {"tensor-parallel-size": 4}],
    },
    "deepseek-v4-pro": {
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "dataset_prefix": "dsv4",
        # Native FP4+FP8 mixed checkpoint. trust-remote-code + tokenizer-mode are required to
        # load it; expert-parallel is the MoE layout over the TP×PP world. These are universal;
        # per-cluster tunables (parallelism layout, and any compilation flags) live in CLUSTERS.
        "extra": {"trust-remote-code": True, "tokenizer-mode": "deepseek_v4",
                  "enable-expert-parallel": True},
        # `parallels` is cluster-specific (GPU memory / GPUs-per-node) — see CLUSTERS.
        "parallels": None,
    },
}

# Cluster-specific overrides (per-platform: GPU count/memory, and any compilation tunables).
# `extra` here is *merged into* the model's base extra; other keys replace. An 8-GPU single-node
# DeepSeek layout (TP8) is documented in the README; uncomment the example to author such a config.
CLUSTERS = {
    "snellius/h100": {           # 4 × 94GiB H100 per node
        "deepseek-v4-pro": {
            "parallels": [{"tensor-parallel-size": 4, "pipeline-parallel-size": 4}],
            # _extra_serve_args is appended verbatim to `vllm serve` (keys starting with `_` are
            # skipped by apply_params_to_cmd / combo_subdir): the dotted -cc.* compilation flags
            # can't go through the --key=value builder, and DeepSeek-V4's default compilation
            # crashes on this TP4×PP4 layout — full torch.compile (mode=3) with a decode-only
            # CUDA-graph capture is required here. Per-platform tunable, so it lives per-cluster.
            "extra": {"_extra_serve_args": "-cc.mode=3 -cc.cudagraph_mode=FULL_DECODE_ONLY"},
        },
    },
}


def write_model(cluster: str, model_key: str, profile: str) -> None:
    spec = dict(MODELS[model_key])
    override = CLUSTERS[cluster].get(model_key, {})
    # Merge `extra` (cluster extra augments the model's base extra); other keys replace.
    merged_extra = {**spec.get("extra", {}), **override.get("extra", {})}
    spec.update(override)
    spec["extra"] = merged_extra
    if spec["parallels"] is None:
        raise ValueError(f"{model_key} has no parallels for cluster {cluster}")

    p = PROFILES[profile]
    parallels = spec["parallels"][p["parallels"]]

    # Non-`full` profiles go to a suffixed cluster dir (e.g. snellius/h100-debug) so the
    # orchestrator's */{serve,bench}-params.json glob sees only the reduced sweep.
    out_cluster = cluster if profile == "full" else f"{cluster}-{profile}"
    out_dir = REPO_ROOT / "configs" / out_cluster / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    serve = _serve_entries(spec["model"], spec["extra"], parallels, p["concurrency"])
    bench = _bench_entries(spec["model"], spec["dataset_prefix"], p["concurrency"], p["shapes"])

    (out_dir / "serve-params.json").write_text(json.dumps(serve, indent=2) + "\n")
    (out_dir / "bench-params.json").write_text(json.dumps(bench, indent=2) + "\n")
    print(f"  {out_cluster}/{model_key}: {len(serve)} serve × {len(bench)} bench entries")

    # Suffixed clusters inherit the base cluster's system.sh (partition, container, GPUs/node).
    if profile != "full":
        sys_link = REPO_ROOT / "configs" / out_cluster / "system.sh"
        if not sys_link.is_symlink() and not sys_link.exists():
            sys_link.symlink_to(Path("..") / Path(cluster).name / "system.sh")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster", default=None,
                    help="Only this cluster (e.g. snellius/h100). Default: all.")
    ap.add_argument("--profile", default="full", choices=list(PROFILES),
                    help="full = real sweep; debug/small = reduced sanity sweeps written to "
                         "configs/<cluster>-<profile>/ (default: full).")
    ap.add_argument("--only-model", default=None, choices=list(MODELS),
                    help="Generate just this one model (default: all). Useful for a "
                         "model-specific sweep dir, e.g. --profile dsv4 --only-model deepseek-v4-pro.")
    args = ap.parse_args()

    clusters = [args.cluster] if args.cluster else list(CLUSTERS)
    model_keys = [args.only_model] if args.only_model else list(MODELS)
    for cluster in clusters:
        if cluster not in CLUSTERS:
            raise SystemExit(f"Unknown cluster {cluster}; known: {list(CLUSTERS)}")
        print(f"{cluster} (profile={args.profile}):")
        for model_key in model_keys:
            write_model(cluster, model_key, args.profile)


if __name__ == "__main__":
    main()
