#!/usr/bin/env python3
"""Prepare ImageNet dataset by dumping extra metadata."""

import argparse
import os
from dinov3_src.dinov3.data.datasets import ImageNet


def prepare_imagenet(root: str, extra: str) -> None:
    """Prepare ImageNet dataset for all splits.
    
    Args:
        root: Path to the ImageNet root directory
        extra: Path to store extra metadata
    """
    for split in ImageNet.Split:
        print(f"Processing {split.name} split...")
        dataset = ImageNet(split=split, root=root, extra=extra)
        dataset.dump_extra()
        print(f"  Completed {split.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ImageNet dataset by dumping extra metadata"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.getenv("ROOT_PATH", "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder"),
        help="Path to the ImageNet root directory (default: $ROOT_PATH or hardcoded path)",
    )
    parser.add_argument(
        "--extra",
        type=str,
        default=os.getenv("EXTRA_PATH", "/home/larsve/MLPerfTraining/data/imagenet1k/extra"),
        help="Path to store extra metadata (default: $EXTRA_PATH or hardcoded path)",
    )
    
    args = parser.parse_args()
    
    print(f"Root: {args.root}")
    print(f"Extra: {args.extra}")
    print()
    
    prepare_imagenet(args.root, args.extra)
    
    print("\nDone!")


if __name__ == "__main__":
    main()