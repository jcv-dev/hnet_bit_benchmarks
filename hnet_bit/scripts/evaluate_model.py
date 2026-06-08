#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model evaluation script.

Evaluates a trained model on test data with comprehensive metrics.

Usage:
    python scripts/evaluate_model.py --model_path checkpoints/best.pt --data_dir ./data/tinystories
    python scripts/evaluate_model.py --model_path checkpoints/best.pt --model_config configs/hnet_bit_2stage.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM
from hnet_bit.utils import ByteTokenizer
from hnet_bit.training import collate_fn
from hnet_bit.training.evaluator import Evaluator, EvaluatorConfig
from hnet_bit.scripts.prepare_tinystories import load_prepared_dataset


def load_model(model_path: str, config_path: str = None, device: str = 'cuda'):
    """Load model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get config
    if 'config' in checkpoint:
        cfg_dict = checkpoint['config']
    elif config_path:
        with open(config_path, 'r') as f:
            cfg_dict = json.load(f)
    else:
        raise ValueError("No config in checkpoint and no config_path provided")
    
    # Create model - always use HNetBit
    model_type = cfg_dict.get('model_type', 'hnet_bit')
    config = HNetBitConfig(**cfg_dict)
    model = HNetBitForCausalLM(config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    
    # Model options
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_config', type=str, default=None,
                        help='Path to model config (if not in checkpoint)')
    
    # Data options
    parser.add_argument('--data_dir', type=str, default='./data/tinystories',
                        help='Directory with prepared data')
    parser.add_argument('--max_seq_length', type=int, default=None,
                        help='Override sequence length')
    
    # Evaluation options
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit batches for speed (None = all)')
    parser.add_argument('--num_generation_samples', type=int, default=10,
                        help='Number of generation samples')
    parser.add_argument('--generation_max_tokens', type=int, default=100,
                        help='Max tokens per generation')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    
    # Hardware options
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = load_model(args.model_path, args.model_config, args.device)
    tokenizer = ByteTokenizer()
    
    print(f"  Model: {type(model).__name__}")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    datasets = load_prepared_dataset(args.data_dir, args.max_seq_length)
    
    test_loader = DataLoader(
        datasets['test'],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        datasets['val'],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"  Test sequences: {len(datasets['test']):,}")
    print(f"  Val sequences: {len(datasets['val']):,}")
    
    # Create evaluator
    config = EvaluatorConfig(
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        num_generation_samples=args.num_generation_samples,
        generation_max_tokens=args.generation_max_tokens,
        output_dir=args.output_dir,
    )
    
    evaluator = Evaluator(model, tokenizer, config, args.device)
    
    # Run evaluation
    results = evaluator.run_full_evaluation(
        test_dataloader=test_loader,
        val_dataloader=val_loader,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Test Loss: {results['test']['loss']:.4f}")
    print(f"Test BPB: {results['test']['bpb']:.4f}")
    print(f"Test Perplexity: {results['test']['perplexity']:.2f}")
    print(f"Test Accuracy: {results['test']['accuracy'] * 100:.2f}%")
    print(f"UTF-8 Validity: {results['generation']['utf8_validity'] * 100:.2f}%")
    print(f"Model Size: {results['efficiency']['total_memory_mb']:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
