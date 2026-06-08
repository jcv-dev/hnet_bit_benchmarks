#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-end experiment runner.

Runs the complete pipeline: data preparation → training → evaluation → analysis.

Usage:
    python scripts/run_experiment.py --config configs/training_2stage_tinystories.json
    python scripts/run_experiment.py --config configs/training_2stage_tinystories.json --skip_data_prep
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM
from hnet_bit.utils import ByteTokenizer
from hnet_bit.training import (
    TrainingConfig,
    Trainer,
    TrainingLogger,
)
from hnet_bit.training.data import ByteLevelDataset
from hnet_bit.training.evaluator import Evaluator, EvaluatorConfig


def load_experiment_config(config_path: str) -> dict:
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def prepare_data(config: dict) -> dict:
    """Prepare dataset based on config."""
    from hnet_bit.scripts.prepare_tinystories import (
        prepare_tinystories,
        load_prepared_dataset,
    )
    
    data_config = config["data"]
    data_dir = data_config["data_dir"]
    
    # Check if data already exists
    data_file = Path(data_dir) / "tinystories_bytes.pt"
    if data_file.exists():
        print(f"Loading existing data from {data_dir}")
        return load_prepared_dataset(data_dir, data_config.get("max_seq_length"))
    
    # Prepare data
    print("Preparing dataset...")
    return prepare_tinystories(
        output_dir=data_dir,
        max_seq_length=data_config.get("max_seq_length", 512),
        val_ratio=data_config.get("val_ratio", 0.1),
        test_ratio=data_config.get("test_ratio", 0.1),
        seed=config.get("seed", 42),
    )


def build_model(config: dict, device: str = 'cuda'):
    """Build model from config."""
    model_config = config["model"]
    
    # Load model config file
    config_file = model_config.get("config_file")
    if config_file:
        # Handle relative path
        config_path = Path(__file__).parent.parent / config_file
        with open(config_path, 'r') as f:
            model_cfg_dict = json.load(f)
    else:
        model_cfg_dict = model_config.get("params", {})
    
    # Create model
    model_type = model_config.get("type", "hnet_bit")
    if model_type == "hnet_bit" or model_cfg_dict.get("model_type") == "hnet_bit":
        cfg = HNetBitConfig(**model_cfg_dict)
        model = HNetBitForCausalLM(cfg)
    else:
        # Default to HNetBit
        cfg = HNetBitConfig(**model_cfg_dict)
        model = HNetBitForCausalLM(cfg)
    
    model = model.to(device)
    return model, model_cfg_dict


def setup_training(config: dict, model, train_dataset, val_dataset, device: str):
    """Setup training components."""
    train_config = config["training"]
    output_config = config["output"]
    
    # Parse mixed_precision string into bf16/fp16 bools
    mixed_precision = train_config.get("mixed_precision", "bf16")
    use_bf16 = (mixed_precision == "bf16")
    use_fp16 = (mixed_precision == "fp16")
    
    # Training config (field names must match TrainingConfig dataclass)
    training_config = TrainingConfig(
        learning_rate=train_config.get("learning_rate", 3e-4),
        weight_decay=train_config.get("weight_decay", 0.01),
        batch_size=train_config.get("batch_size", 16),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
        max_steps=train_config.get("max_steps", 10000),
        warmup_steps=train_config.get("warmup_steps", 500),
        lr_scheduler=train_config.get("scheduler", "cosine"),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        eval_interval=train_config.get("eval_interval", 500),
        save_interval=train_config.get("save_interval", 1000),
        log_interval=train_config.get("log_interval", 50),
        bf16=use_bf16,
        fp16=use_fp16,
        output_dir=output_config.get("checkpoint_dir", "./checkpoints"),
        # Hierarchical training params (optional)
        lr_multipliers=train_config.get("lr_multipliers", None),
        lambda_lb=train_config.get("lambda_lb", 0.0),
        downsampling_factor=train_config.get("downsampling_factor", 2.5),
    )
    
    # Create directories
    Path(training_config.output_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(output_config.get("log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    logger = TrainingLogger(
        log_dir=str(log_dir),
        experiment_name=config.get("experiment_name", "experiment"),
        use_wandb=False,  # Set to True if you want wandb logging
    )
    
    # Trainer builds its own optimizer/scheduler from config internally
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        logger=logger,
    )
    
    return trainer, training_config


def run_training(trainer: Trainer, config: dict) -> str:
    """Run training and return best checkpoint path."""
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    print(f"\nTraining completed in {train_time / 3600:.2f} hours")
    
    # Return best checkpoint
    ckpt_dir = Path(config["output"]["checkpoint_dir"])
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        return str(best_ckpt)
    
    # Find latest checkpoint
    checkpoints = sorted(ckpt_dir.glob("*.pt"))
    return str(checkpoints[-1]) if checkpoints else None


def run_evaluation(model_path: str, config: dict, test_dataset, device: str):
    """Run evaluation on trained model."""
    from torch.utils.data import DataLoader
    from hnet_bit.training.data import collate_fn
    
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg_dict = checkpoint.get("config", {})
    
    # Always use HNetBit
    model = HNetBitForCausalLM(HNetBitConfig(**cfg_dict))
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    tokenizer = ByteTokenizer()
    
    # Create test dataloader
    eval_config = config.get("evaluation", {})
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_config.get("batch_size", 32),
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Setup evaluator
    results_dir = config["output"].get("results_dir", "./results")
    evaluator_config = EvaluatorConfig(
        batch_size=eval_config.get("batch_size", 32),
        num_generation_samples=eval_config.get("num_generation_samples", 20),
        generation_max_tokens=eval_config.get("generation_max_tokens", 100),
        output_dir=results_dir,
    )
    
    evaluator = Evaluator(model, tokenizer, evaluator_config, device)
    
    # Run evaluation
    results = evaluator.run_full_evaluation(test_loader)
    
    return results


def save_experiment_summary(config: dict, results: dict, output_dir: str):
    """Save experiment summary."""
    summary = {
        "experiment": config.get("experiment_name", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nExperiment summary saved to {output_path / 'experiment_summary.json'}")


def main():
    parser = argparse.ArgumentParser(
        description='Run complete experiment pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full experiment
    python scripts/run_experiment.py --config configs/training_2stage_tinystories.json
    
    # Skip data preparation (if already done)
    python scripts/run_experiment.py --config configs/training_2stage_tinystories.json --skip_data_prep
    
    # Only evaluate existing checkpoint
    python scripts/run_experiment.py --config configs/training_2stage_tinystories.json --eval_only --model_path checkpoints/best.pt
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment config JSON')
    parser.add_argument('--skip_data_prep', action='store_true',
                        help='Skip dataset preparation')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training (use existing checkpoint)')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (for skip_training or eval_only)')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device from config')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')
    
    args = parser.parse_args()
    
    # Load config
    config = load_experiment_config(args.config)
    
    # Override from args
    if args.device:
        config.setdefault("hardware", {})["device"] = args.device
    if args.seed:
        config["seed"] = args.seed
    
    device = config.get("hardware", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Set seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("=" * 60)
    print(f"Experiment: {config.get('experiment_name', 'unknown')}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    # Step 1: Data preparation
    datasets = None
    if not args.skip_data_prep and not args.eval_only:
        print("\n[Step 1] Preparing data...")
        datasets = prepare_data(config)
    else:
        print("\n[Step 1] Loading existing data...")
        from hnet_bit.scripts.prepare_tinystories import load_prepared_dataset
        datasets = load_prepared_dataset(
            config["data"]["data_dir"],
            config["data"].get("max_seq_length"),
        )
    
    print(f"  Train: {len(datasets['train']):,} sequences")
    print(f"  Val: {len(datasets['val']):,} sequences")
    print(f"  Test: {len(datasets['test']):,} sequences")
    
    # Step 2: Training
    model_path = args.model_path
    
    if not args.skip_training and not args.eval_only:
        print("\n[Step 2] Building model...")
        model, model_cfg = build_model(config, device)
        print(f"  Model: {type(model).__name__}")
        print(f"  Parameters: {model.count_parameters():,}")
        
        print("\n[Step 3] Setting up training...")
        trainer, train_config = setup_training(
            config, model, datasets["train"], datasets["val"], device
        )
        
        print("\n[Step 4] Training...")
        model_path = run_training(trainer, config)
    else:
        print("\n[Step 2-4] Skipping training")
        if not model_path:
            ckpt_dir = Path(config["output"]["checkpoint_dir"])
            best_ckpt = ckpt_dir / "best.pt"
            model_path = str(best_ckpt) if best_ckpt.exists() else None
    
    # Step 3: Evaluation
    results = None
    if not args.skip_evaluation and model_path:
        print(f"\n[Step 5] Running evaluation on {model_path}...")
        results = run_evaluation(model_path, config, datasets["test"], device)
    else:
        print("\n[Step 5] Skipping evaluation")
    
    # Save summary
    if results:
        save_experiment_summary(
            config, 
            results, 
            config["output"].get("results_dir", "./results")
        )
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    
    if results:
        print(f"\nFinal Test Results:")
        print(f"  Loss: {results['test']['loss']:.4f}")
        print(f"  BPB: {results['test']['bpb']:.4f}")
        print(f"  Perplexity: {results['test']['perplexity']:.2f}")
        print(f"  Accuracy: {results['test']['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
