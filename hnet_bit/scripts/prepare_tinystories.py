#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TinyStories dataset preparation script.

Downloads TinyStories from HuggingFace and prepares it for byte-level training.

Usage:
    python scripts/prepare_tinystories.py --output_dir ./data/tinystories
    python scripts/prepare_tinystories.py --max_samples 10000 --output_dir ./data/tinystories_small
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hnet_bit.training import (
    DatasetConfig,
    TinyStoriesDataset,
    print_dataset_stats,
)
from hnet_bit.training.data import ByteLevelDataset


def prepare_tinystories(
    output_dir: str,
    max_seq_length: int = 512,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_samples: int = None,
    seed: int = 42,
):
    """
    Download and prepare TinyStories dataset.
    
    Args:
        output_dir: Directory to save processed data
        max_seq_length: Sequence length for training
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        max_samples: Optional limit on samples (for debugging)
        seed: Random seed for reproducibility
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TinyStories Dataset Preparation")
    print("=" * 60)
    
    # Load dataset
    print("\n[Step 1/4] Loading TinyStories from HuggingFace...")
    dataset = TinyStoriesDataset(
        split="train",
        max_seq_length=max_seq_length,
        cache_dir=str(output_path / "cache"),
        max_samples=max_samples,
    )
    
    # Print statistics
    print_dataset_stats(dataset)
    
    # Split data
    print("\n[Step 2/4] Splitting into train/val/test...")
    raw_bytes = dataset.raw_bytes
    total_len = len(raw_bytes)
    
    torch.manual_seed(seed)
    
    test_len = int(total_len * test_ratio)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len - test_len
    
    train_data = raw_bytes[:train_len]
    val_data = raw_bytes[train_len:train_len + val_len]
    test_data = raw_bytes[train_len + val_len:]
    
    print(f"  Train: {train_len:,} bytes ({100 * train_len / total_len:.1f}%)")
    print(f"  Val: {val_len:,} bytes ({100 * val_len / total_len:.1f}%)")
    print(f"  Test: {test_len:,} bytes ({100 * test_len / total_len:.1f}%)")
    
    # Create datasets
    print("\n[Step 3/4] Creating ByteLevelDatasets...")
    splits = {}
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        ds = ByteLevelDataset(
            data=data,
            max_seq_length=max_seq_length,
            stride=max_seq_length,  # No overlap
        )
        splits[name] = ds
        print(f"  {name}: {len(ds):,} sequences")
    
    # Save processed data
    print("\n[Step 4/4] Saving processed data...")
    
    # Save raw byte tensors
    torch.save({
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "config": {
            "max_seq_length": max_seq_length,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "total_bytes": total_len,
        },
    }, output_path / "tinystories_bytes.pt")
    
    # Save statistics
    stats = dataset.statistics.to_dict()
    stats.update({
        "train_bytes": train_len,
        "val_bytes": val_len,
        "test_bytes": test_len,
        "train_sequences": len(splits["train"]),
        "val_sequences": len(splits["val"]),
        "test_sequences": len(splits["test"]),
    })
    
    with open(output_path / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSaved to {output_path}:")
    print(f"  - tinystories_bytes.pt (raw byte tensors)")
    print(f"  - statistics.json (dataset statistics)")
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    
    return splits


def load_prepared_dataset(
    data_dir: str,
    max_seq_length: int = None,
) -> dict:
    """
    Load a previously prepared dataset.
    
    Args:
        data_dir: Directory with prepared data
        max_seq_length: Override sequence length (optional)
        
    Returns:
        Dict with 'train', 'val', 'test' ByteLevelDatasets
    """
    data_path = Path(data_dir)
    data_file = data_path / "tinystories_bytes.pt"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Prepared data not found at {data_file}")
    
    data = torch.load(data_file, weights_only=True)
    config = data["config"]
    
    seq_len = max_seq_length or config["max_seq_length"]
    
    splits = {}
    for name in ["train", "val", "test"]:
        splits[name] = ByteLevelDataset(
            data=data[name],
            max_seq_length=seq_len,
        )
    
    return splits


def main():
    parser = argparse.ArgumentParser(
        description='Prepare TinyStories dataset for byte-level training'
    )
    parser.add_argument('--output_dir', type=str, default='./data/tinystories',
                        help='Output directory for processed data')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Sequence length for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction for validation (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Fraction for test (default: 0.1)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of stories (for debugging)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    prepare_tinystories(
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
