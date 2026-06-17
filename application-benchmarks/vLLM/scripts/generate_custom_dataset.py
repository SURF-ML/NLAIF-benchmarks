#!/usr/bin/env python3
"""
Generate a random dataset in vLLM CustomDataset JSONL format.

The output file can be passed directly to vLLM's benchmark suite via:
    --dataset-name custom --dataset-path <output>

Each line in the output is:
    {"prompt": "<raw text>"}

The prompts are random token sequences decoded to text, mirroring the
strategy used by vLLM's RandomDataset. When loaded by CustomDataset,
the chat template is applied on top, so the final token count will be
slightly larger than --input-len. Pass --skip-chat-template to vLLM's
bench serve command to skip that.

Usage:
    python generate_custom_dataset.py --output dataset.jsonl
    python generate_custom_dataset.py \\
        --num-requests 4096 \\
        --input-len 2048 \\
        --output-len 512 \\
        --range-ratio 0.1 \\
        --tokenizer nvidia/Llama-3.3-70B-Instruct-FP8 \\
        --output my_dataset.jsonl
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a random CustomDataset-compatible JSONL file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="nvidia/Llama-3.3-70B-Instruct-FP8",
        help="HuggingFace tokenizer name or local path",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=131072,
        help="Number of prompts to generate",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Target input length in tokens (before chat template)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Expected output length stored as metadata (not consumed by CustomDataset)",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Length of a shared random prefix prepended to every prompt",
    )
    parser.add_argument(
        "--range-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction by which to vary lengths around the target. "
            "0.0 = all prompts have exactly --input-len tokens. "
            "Must be in [0, 1)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading tokenizers that ship custom code (e.g. DeepSeek-V4-Pro)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="custom_datasets",
        help="Directory to write the output JSONL file into",
    )
    parser.add_argument(
        "--model-short-name",
        type=str,
        default=None,
        help="Short name used in the output filename (e.g. 'llama70b'). "
             "Defaults to the last component of --tokenizer.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print progress every N requests",
    )
    return parser.parse_args()


def make_output_path(args: argparse.Namespace) -> str:
    """Build a descriptive output filename from the run parameters."""
    short_name = args.model_short_name or args.tokenizer.rstrip("/").split("/")[-1]

    name = (
        f"{short_name}"
        f"_i{args.input_len}"
        f"_o{args.output_len}"
        f"_n{args.num_requests}"
    )
    if args.prefix_len:
        name += f"_pfx{args.prefix_len}"
    if args.range_ratio:
        name += f"_rng{args.range_ratio}"
    name += ".jsonl"

    os.makedirs(args.output_dir, exist_ok=True)
    return os.path.join(args.output_dir, name)


def build_allowed_tokens(tokenizer: AutoTokenizer) -> np.ndarray:
    """Return an array of non-special token IDs."""
    all_ids = np.arange(tokenizer.vocab_size)
    prohibited = set(tokenizer.all_special_ids)
    return np.array([t for t in all_ids if t not in prohibited], dtype=np.int64)


def sample_lengths(
    rng: np.random.Generator,
    num_requests: int,
    target_len: int,
    range_ratio: float,
    min_len: int = 1,
) -> np.ndarray:
    """Sample integer lengths uniformly around target_len."""
    low = math.floor(target_len * (1.0 - range_ratio))
    high = math.ceil(target_len * (1.0 + range_ratio))
    low = max(low, min_len)
    high = max(high, min_len)
    return rng.integers(low, high + 1, size=num_requests)


def generate_prompt(
    tokenizer: AutoTokenizer,
    allowed_tokens: np.ndarray,
    prefix_ids: list,
    input_len: int,
    rng: np.random.Generator,
) -> tuple:
    """
    Build a random prompt that tokenizes to exactly input_len tokens
    INCLUDING special tokens (BOS etc.), matching what CustomDataset measures via
    `len(tokenizer(prompt).input_ids)`.

    Strategy: maintain a pool of content token IDs (no special tokens), then:
      - decode pool → text
      - encode with add_special_tokens=True → check total length
      - if too long:  strip specials, keep first (input_len - n_special) content IDs
      - if too short: strip specials, append more random tokens
    Repeat until exact match.
    """
    n_special = int(tokenizer.num_special_tokens_to_add())
    target_len = len(prefix_ids) + input_len          # total tokens incl. special
    content_target = target_len - n_special            # content tokens we control

    # Seed pool with content tokens + 10% buffer
    pool = list(prefix_ids) + rng.choice(
        allowed_tokens, size=content_target + max(64, content_target // 10), replace=True
    ).tolist()

    for _ in range(50):
        text = tokenizer.decode(pool, skip_special_tokens=True)
        re_ids = tokenizer.encode(text, add_special_tokens=True)

        if len(re_ids) == target_len:
            return text, target_len
        elif len(re_ids) > target_len:
            # Strip leading special tokens, keep exactly content_target content IDs
            pool = re_ids[n_special:target_len]
        else:
            # Too few content tokens — append more
            deficit = target_len - len(re_ids) + 16
            pool = re_ids[n_special:] + rng.choice(allowed_tokens, size=deficit, replace=True).tolist()

    # Fallback (should be extremely rare)
    return text, len(re_ids)


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.range_ratio < 1.0):
        print("Error: --range-ratio must be in [0, 1).", file=sys.stderr)
        sys.exit(1)

    output_path = make_output_path(args)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )

    rng = np.random.default_rng(args.seed)
    allowed_tokens = build_allowed_tokens(tokenizer)

    input_lens = sample_lengths(rng, args.num_requests, args.input_len, args.range_ratio)
    output_lens = sample_lengths(rng, args.num_requests, args.output_len, args.range_ratio)

    # Build shared prefix (empty if prefix_len == 0)
    prefix_ids = (
        rng.choice(allowed_tokens, size=args.prefix_len, replace=True).tolist()
        if args.prefix_len > 0
        else []
    )

    print(
        f"Generating {args.num_requests:,} prompts "
        f"(input_len≈{args.input_len}, output_len≈{args.output_len}, "
        f"prefix_len={args.prefix_len}, range_ratio={args.range_ratio}) …"
    )

    records = []
    actual_input_lens = []
    width = len(str(args.num_requests))

    for i in range(args.num_requests):
        prompt, actual_len = generate_prompt(
            tokenizer=tokenizer,
            allowed_tokens=allowed_tokens,
            prefix_ids=prefix_ids,
            input_len=int(input_lens[i]),
            rng=rng,
        )
        # output_len is metadata only; CustomDataset ignores unknown columns
        records.append({"prompt": prompt, "output_len": int(output_lens[i])})
        actual_input_lens.append(actual_len)

        if (i + 1) % args.log_every == 0 or (i + 1) == args.num_requests:
            print(f"  {i + 1:{width}}/{args.num_requests}", end="\r")

    print()

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    mean_in = sum(actual_input_lens) / len(actual_input_lens)
    mean_out = sum(r["output_len"] for r in records) / len(records)
    print(f"\nWrote {len(records):,} records to: {output_path}")
    print(f"  Actual input  tokens — min: {min(actual_input_lens)}, "
          f"max: {max(actual_input_lens)}, mean: {mean_in:.1f}")
    print(f"  Expected output tokens — min: {min(r['output_len'] for r in records)}, "
          f"max: {max(r['output_len'] for r in records)}, mean: {mean_out:.1f}")
    print(
        "\nNote: CustomDataset applies the chat template on top of each prompt.\n"
        "      Use --skip-chat-template with 'vllm bench serve' to skip it."
    )


if __name__ == "__main__":
    main()