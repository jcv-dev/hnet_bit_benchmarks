#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic dataset preparation script for HNetBit models.

Downloads any HuggingFace text dataset, converts it to the byte-level
format used by the training pipeline, and saves train/val/test splits.

Examples:
    # TinyStories (text column auto-detected)
    python -m hnet_bit.scripts.prepare_dataset \
        --dataset roneneldan/TinyStories \
        --output_dir data/tinystories

    # WikiText-103
    python -m hnet_bit.scripts.prepare_dataset \
        --dataset wikitext --dataset_config wikitext-103-raw-v1 \
        --output_dir data/wikitext103

    # Alpaca (instruction + output columns)
    python -m hnet_bit.scripts.prepare_dataset \
        --dataset tatsu-lab/alpaca \
        --text_columns instruction output \
        --output_dir data/alpaca

    # Limit to 10k samples for quick testing
    python -m hnet_bit.scripts.prepare_dataset \
        --dataset roneneldan/TinyStories \
        --max_samples 10000 \
        --output_dir data/tinystories_small

    # Custom text column
    python -m hnet_bit.scripts.prepare_dataset \
        --dataset bookcorpus --text_column text \
        --output_dir data/bookcorpus
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# Add parent to path so we can run as script or module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hnet_bit.training.dataset_loader import (
    DatasetConfig,
    HuggingFaceDataset,
)


def prepare(args: argparse.Namespace) -> None:
    """Download, convert, and save a dataset."""

    os.makedirs(args.output_dir, exist_ok=True)

    # Build split list -------------------------------------------------------
    # Default: try train / validation / test.  User can override.
    splits = {
        "train": args.train_split,
        "val": args.val_split,
        "test": args.test_split,
    }

    for label, split_name in splits.items():
        if split_name is None:
            continue

        print(f"\n{'='*60}")
        print(f"Preparing {label} split  ({split_name})")
        print(f"{'='*60}")

        ds_cfg = DatasetConfig(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=split_name,
            text_column=args.text_column,
            text_columns=args.text_columns,
            separator=args.separator,
            max_seq_length=args.max_seq_length,
            max_samples=args.max_samples,
            streaming=args.streaming,
            cache_dir=os.path.join(args.output_dir, "hf_cache"),
        )

        t0 = time.time()
        try:
            hf_ds = HuggingFaceDataset(ds_cfg)  # loads automatically
        except Exception as exc:
            print(f"  Skipping '{split_name}' — {exc}")
            continue

        elapsed = time.time() - t0
        print(f"  Loaded in {elapsed:.1f}s — {len(hf_ds)} samples")

        # Save byte tensor
        out_path = os.path.join(args.output_dir, f"{label}_bytes.pt")
        torch.save(hf_ds.raw_bytes, out_path)
        print(f"  Saved → {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")

        # Stats
        stats = hf_ds.statistics
        if stats:
            print(f"  Samples: {stats.num_samples:,}")
            print(f"  Total bytes: {stats.total_bytes:,}")
            if stats.avg_bytes_per_sample:
                print(f"  Mean length: {stats.avg_bytes_per_sample:.0f} bytes/sample")

    # Save a metadata file so train.py can reload easily
    meta = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "text_column": args.text_column,
        "text_columns": args.text_columns,
        "max_seq_length": args.max_seq_length,
    }
    import json
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone.  Output directory: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare any HuggingFace dataset for HNetBit training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset name (e.g. 'roneneldan/TinyStories')")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save prepared data")

    # Optional — dataset
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="HF dataset config / subset name")
    parser.add_argument("--text_column", type=str, default="auto",
                        help="Text column (default: auto-detect)")
    parser.add_argument("--text_columns", type=str, nargs="+", default=None,
                        help="Multiple text columns to join")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator when joining multiple columns")

    # Splits
    parser.add_argument("--train_split", type=str, default="train",
                        help="Training split name (default: 'train')")
    parser.add_argument("--val_split", type=str, default="validation",
                        help="Validation split name (default: 'validation', 'none' to skip)")
    parser.add_argument("--test_split", type=str, default=None,
                        help="Test split name (default: skip)")

    # Preprocessing
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence length in bytes")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples per split")
    parser.add_argument("--streaming", action="store_true",
                        help="Stream dataset (for very large datasets)")

    args = parser.parse_args()

    # 'none' means skip
    if args.val_split and args.val_split.lower() == "none":
        args.val_split = None
    if args.test_split and args.test_split.lower() == "none":
        args.test_split = None

    prepare(args)


if __name__ == "__main__":
    main()
