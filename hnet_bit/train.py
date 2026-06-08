#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training entry point for HNetBit models.

Usage:
    # Train hierarchical model with defaults
    python -m hnet_bit.train --output_dir runs/test

    # Train hierarchical model with config
    python -m hnet_bit.train \
        --model_config configs/hnet_bit_1stage.json \
        --training_config configs/training_small.json

    # Resume from checkpoint
    python -m hnet_bit.train \
        --resume_from runs/experiment/checkpoint_step_1000.pt
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from hnet_bit.training.config import TrainingConfig
from hnet_bit.training.data import ByteLevelDataset, TextFileDataset, create_synthetic_data
from hnet_bit.training.dataset_loader import DatasetConfig, HuggingFaceDataset
from hnet_bit.training.trainer import Trainer
from hnet_bit.training.logger import TrainingLogger


def build_model(model_config_path: str | None, device: torch.device):
    """Build HNetBit model from config."""
    from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

    if model_config_path:
        with open(model_config_path, 'r') as f:
            cfg = json.load(f)
        config = HNetBitConfig(**cfg)
    else:
        config = HNetBitConfig.small_1stage()
    
    model = HNetBitForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: HNetBit, Parameters: {n_params:,}")
    return model.to(device)


def build_datasets(config: TrainingConfig, max_seq_length: int):
    """Build training and optional validation dataset.

    Supports three data sources (checked in order):
      1. ``config.dataset`` — any HuggingFace dataset
      2. ``config.data_path`` — a local text file
      3. Synthetic data (fallback for quick testing)
    """

    # -----------------------------------------------------------------
    # 1. HuggingFace dataset
    # -----------------------------------------------------------------
    if config.dataset:
        print(f"Loading HuggingFace dataset: {config.dataset}")

        def _make_ds(split: str) -> HuggingFaceDataset:
            ds_cfg = DatasetConfig(
                dataset_name=config.dataset,
                dataset_config=config.dataset_config,
                split=split,
                text_column=config.text_column,
                text_columns=config.text_columns,
                separator=config.separator,
                max_seq_length=max_seq_length,
                max_samples=config.max_samples,
                streaming=config.streaming,
                seed=config.seed,
            )
            return HuggingFaceDataset(ds_cfg)  # loads automatically

        train_ds = _make_ds(config.train_split)

        try:
            # Use 10% of max_samples for validation (if limited)
            saved_max = config.max_samples
            if config.max_samples:
                config.max_samples = max(1, config.max_samples // 10)
            val_ds = _make_ds(config.val_split)
            config.max_samples = saved_max
        except Exception as exc:
            # Many datasets don't have a 'validation' split — fall back
            # to a random subset of the training data.
            print(f"  No '{config.val_split}' split ({exc}); "
                  "carving 10% of training data for validation.")
            n = len(train_ds)
            n_val = max(1, n // 10)
            n_train = n - n_val
            val_ds = torch.utils.data.Subset(train_ds, range(n_train, n))
            train_ds = torch.utils.data.Subset(train_ds, range(n_train))

        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        return train_ds, val_ds

    # -----------------------------------------------------------------
    # 2. Local text file
    # -----------------------------------------------------------------
    if config.data_path and os.path.exists(config.data_path):
        print(f"Loading data from {config.data_path}")
        full_ds = TextFileDataset(config.data_path, max_seq_length=max_seq_length)
        
        # 90/10 split
        n = len(full_ds)
        n_val = max(1, n // 10)
        n_train = n - n_val
        
        train_ds = torch.utils.data.Subset(full_ds, range(n_train))
        val_ds = torch.utils.data.Subset(full_ds, range(n_train, n))
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        return train_ds, val_ds

    # -----------------------------------------------------------------
    # 3. Synthetic fallback
    # -----------------------------------------------------------------
    print("No data_path or dataset found — using synthetic data for testing")
    data = create_synthetic_data(num_bytes=max_seq_length * 500)
    train_tokens = data[:int(len(data) * 0.9)]
    val_tokens = data[int(len(data) * 0.9):]
    
    train_ds = ByteLevelDataset(train_tokens, max_seq_length=max_seq_length)
    val_ds = ByteLevelDataset(val_tokens, max_seq_length=max_seq_length)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="Train HNetBit models")
    
    # Model
    parser.add_argument("--model_config", type=str, default=None,
                        help="Path to model config JSON")
    
    # Training config
    parser.add_argument("--training_config", type=str, default=None,
                        help="Path to training config JSON")
    
    # Override common training params via CLI
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    
    # HuggingFace dataset ------------------------------------------------
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name (e.g. 'roneneldan/TinyStories')")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="HF dataset config / subset name")
    parser.add_argument("--text_column", type=str, default=None,
                        help="Text column name (default: auto-detect)")
    parser.add_argument("--text_columns", type=str, nargs="+", default=None,
                        help="Multiple text columns to join (e.g. --text_columns instruction output)")
    parser.add_argument("--streaming", action="store_true", default=False,
                        help="Stream dataset instead of downloading entirely")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of dataset samples (for debugging)")
    parser.add_argument("--train_split", type=str, default=None,
                        help="Training split name (default: 'train')")
    parser.add_argument("--val_split", type=str, default=None,
                        help="Validation split name (default: 'validation')")
    
    args = parser.parse_args()
    
    # Load or create training config
    if args.training_config:
        config = TrainingConfig.load(args.training_config)
    else:
        config = TrainingConfig()
    
    # CLI overrides
    for key in ['output_dir', 'data_path', 'max_steps', 'batch_size',
                'max_seq_length', 'resume_from', 'seed', 'bf16', 'fp16',
                'dataset', 'dataset_config', 'text_column', 'text_columns',
                'streaming', 'max_samples', 'train_split', 'val_split']:
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)
    # Map --lr to learning_rate
    if args.lr is not None:
        config.learning_rate = args.lr
    
    # Defaults
    if config.output_dir == "./checkpoints":
        config.output_dir = f"runs/hnetbit_experiment"
    
    # Seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        config.fp16 = False
        config.bf16 = False
    
    # Build model
    model = build_model(args.model_config, device)
    
    # Apply per-stage learning rate multipliers
    if config.lr_multipliers is not None:
        print(f"Applying LR multipliers: {config.lr_multipliers}")
        model.backbone._apply_lr_multiplier(config.lr_multipliers)
    
    # Build datasets
    max_seq_length = config.max_seq_length
    train_ds, val_ds = build_datasets(config, max_seq_length)
    
    # Logger
    logger = TrainingLogger(config.output_dir)
    
    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        logger=logger,
    )
    trainer.train()


if __name__ == "__main__":
    main()
