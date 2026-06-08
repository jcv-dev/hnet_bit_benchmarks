#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training analysis and visualization script.

Analyzes training logs and generates visualizations.

Usage:
    python scripts/analyze_training.py --log_dir ./logs/experiment_1
    python scripts/analyze_training.py --checkpoint_dir ./checkpoints
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_training_log(log_file: str) -> List[Dict]:
    """Load training log from CSV file."""
    records = []
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            record = {}
            for key, value in row.items():
                try:
                    record[key] = float(value)
                except (ValueError, TypeError):
                    record[key] = value
            records.append(record)
    return records


def analyze_training_logs(log_dir: str) -> Dict:
    """
    Analyze training logs from a directory.
    
    Args:
        log_dir: Directory containing training logs
        
    Returns:
        Dict with analysis results
    """
    log_path = Path(log_dir)
    
    results = {
        "log_dir": str(log_path),
        "training": None,
        "evaluation": None,
    }
    
    # Find log files
    train_log = log_path / "training_log.csv"
    if train_log.exists():
        train_records = load_training_log(str(train_log))
        results["training"] = analyze_train_records(train_records)
    
    return results


def analyze_train_records(records: List[Dict]) -> Dict:
    """Analyze training records."""
    if not records:
        return {}
    
    # Extract metrics
    steps = [r.get('step', i) for i, r in enumerate(records)]
    losses = [r.get('loss', 0) for r in records]
    bpbs = [r.get('bpb', 0) for r in records]
    lrs = [r.get('learning_rate', r.get('lr', 0)) for r in records]
    
    # Basic statistics
    analysis = {
        "total_steps": len(records),
        "final_loss": losses[-1] if losses else 0,
        "final_bpb": bpbs[-1] if bpbs else 0,
        "best_loss": min(losses) if losses else 0,
        "best_bpb": min(bpbs) if bpbs else 0,
        "loss_reduction": (losses[0] - losses[-1]) / losses[0] if losses and losses[0] > 0 else 0,
    }
    
    # Find best checkpoint
    if bpbs:
        best_idx = bpbs.index(min(bpbs))
        analysis["best_step"] = int(steps[best_idx])
    
    return analysis


def analyze_checkpoints(checkpoint_dir: str) -> List[Dict]:
    """
    Analyze checkpoints in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        List of checkpoint info dicts
    """
    import torch
    
    ckpt_path = Path(checkpoint_dir)
    checkpoints = []
    
    for ckpt_file in sorted(ckpt_path.glob("*.pt")):
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            info = {
                "name": ckpt_file.name,
                "path": str(ckpt_file),
                "step": ckpt.get('step', 0),
                "loss": ckpt.get('loss', 0),
            }
            if 'config' in ckpt:
                info["model_type"] = ckpt['config'].get('model_type', 'unknown')
            checkpoints.append(info)
        except Exception as e:
            print(f"Warning: Could not load {ckpt_file}: {e}")
    
    return checkpoints


def plot_training_curves(
    log_dir: str,
    output_file: str = None,
    show: bool = True,
):
    """
    Plot training curves from logs.
    
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    log_path = Path(log_dir)
    train_log = log_path / "training_log.csv"
    
    if not train_log.exists():
        print(f"Training log not found at {train_log}")
        return
    
    records = load_training_log(str(train_log))
    
    # Extract data
    steps = [r.get('step', i) for i, r in enumerate(records)]
    losses = [r.get('loss', 0) for r in records]
    bpbs = [r.get('bpb', 0) for r in records]
    lrs = [r.get('learning_rate', r.get('lr', 0)) for r in records]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Analysis', fontsize=14)
    
    # Loss curve
    ax = axes[0, 0]
    ax.plot(steps, losses, 'b-', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # BPB curve
    ax = axes[0, 1]
    ax.plot(steps, bpbs, 'g-', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Bits per Byte')
    ax.set_title('Bits per Byte (BPB)')
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 0]
    ax.plot(steps, lrs, 'r-', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    # Loss histogram
    ax = axes[1, 1]
    ax.hist(losses, bins=50, alpha=0.7, color='blue')
    ax.set_xlabel('Loss')
    ax.set_ylabel('Frequency')
    ax.set_title('Loss Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    if show:
        plt.show()
    
    plt.close()


def analyze_ternary_weights(model) -> Dict:
    """
    Analyze ternary weight distribution in a model.
    
    Args:
        model: Model with BitLinear layers
        
    Returns:
        Dict with weight statistics
    """
    import torch
    
    total_weights = 0
    ternary_weights = 0
    weight_counts = {-1: 0, 0: 0, 1: 0}
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            total_weights += param.numel()
            
            # Quantize weights
            scale = 1.0 / param.abs().mean().clamp(min=1e-5)
            quantized = (param * scale).round().clamp(-1, 1)
            
            # Count
            for val in [-1, 0, 1]:
                count = (quantized == val).sum().item()
                weight_counts[val] += count
                ternary_weights += count
    
    return {
        "total_weights": total_weights,
        "weight_counts": weight_counts,
        "percentages": {
            str(k): 100 * v / max(total_weights, 1)
            for k, v in weight_counts.items()
        },
    }


def print_analysis_report(results: Dict) -> None:
    """Print formatted analysis report."""
    print("\n" + "=" * 60)
    print("Training Analysis Report")
    print("=" * 60)
    
    if results.get("training"):
        tr = results["training"]
        print("\nTraining Statistics:")
        print(f"  Total Steps: {tr.get('total_steps', 0):,}")
        print(f"  Final Loss: {tr.get('final_loss', 0):.4f}")
        print(f"  Final BPB: {tr.get('final_bpb', 0):.4f}")
        print(f"  Best Loss: {tr.get('best_loss', 0):.4f}")
        print(f"  Best BPB: {tr.get('best_bpb', 0):.4f}")
        print(f"  Loss Reduction: {tr.get('loss_reduction', 0) * 100:.1f}%")
        if 'best_step' in tr:
            print(f"  Best Step: {tr['best_step']:,}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze training logs and visualize results'
    )
    
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory containing training logs')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing checkpoints')
    parser.add_argument('--plot', action='store_true',
                        help='Generate training curve plots')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model checkpoint for weight analysis')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Analyze training logs
    print("Analyzing training logs...")
    results = analyze_training_logs(args.log_dir)
    
    # Print report
    print_analysis_report(results)
    
    # Analyze checkpoints if provided
    if args.checkpoint_dir:
        print("\nAnalyzing checkpoints...")
        checkpoints = analyze_checkpoints(args.checkpoint_dir)
        results["checkpoints"] = checkpoints
        
        print(f"Found {len(checkpoints)} checkpoints:")
        for ckpt in checkpoints[:5]:  # Show first 5
            print(f"  - {ckpt['name']}: step={ckpt.get('step', '?')}, loss={ckpt.get('loss', '?'):.4f}")
    
    # Analyze model weights if provided
    if args.model_path:
        import torch
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM
        
        print("\nAnalyzing ternary weights...")
        ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
        
        model = HNetBitForCausalLM(HNetBitConfig(**cfg))
        
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        
        weight_analysis = analyze_ternary_weights(model)
        results["weight_analysis"] = weight_analysis
        
        print(f"Ternary Weight Distribution:")
        for val, pct in weight_analysis["percentages"].items():
            print(f"  {val}: {pct:.1f}%")
    
    # Save results
    with open(output_path / "analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path / 'analysis_results.json'}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating training curve plots...")
        plot_training_curves(
            args.log_dir,
            output_file=str(output_path / "training_curves.png"),
            show=False,
        )


if __name__ == "__main__":
    # Handle missing imports gracefully
    try:
        from hnet_bit.models.hnet_bit import HNetBitConfig
    except ImportError:
        pass
    
    main()
