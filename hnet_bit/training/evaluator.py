# -*- coding: utf-8 -*-

"""
Comprehensive evaluation pipeline for byte-level language models.

Provides:
- Full model evaluation on test sets
- Generation quality assessment
- Efficiency profiling
- Checkpoint comparison
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import collate_fn
from .metrics import (
    EvaluationMetrics,
    compute_bpb,
    compute_perplexity,
    compute_accuracy,
    compute_utf8_validity,
    compute_model_flops,
    compute_memory_footprint,
    evaluate_model,
    evaluate_generation_quality,
)


@dataclass
class EvaluatorConfig:
    """Configuration for model evaluation."""
    
    # Evaluation settings
    batch_size: int = 16
    max_batches: Optional[int] = None  # Limit for speed (None = all)
    
    # Generation evaluation
    num_generation_samples: int = 10
    generation_max_tokens: int = 100
    generation_prompts: Optional[List[str]] = None
    
    # Efficiency profiling
    profile_memory: bool = True
    profile_latency: bool = True
    latency_warmup_runs: int = 5
    latency_measure_runs: int = 20
    
    # Output
    output_dir: str = "./evaluation_results"
    save_generations: bool = True


class Evaluator:
    """
    Comprehensive model evaluator.
    
    Performs:
    - Language modeling evaluation (loss, BPB, perplexity, accuracy)
    - Generation quality evaluation (UTF-8 validity, fluency)
    - Efficiency profiling (memory, latency, throughput)
    
    Args:
        model: The model to evaluate
        tokenizer: ByteTokenizer for encoding/decoding
        config: Evaluation configuration
        device: Device to run evaluation on
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[EvaluatorConfig] = None,
        device: str = 'cuda',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EvaluatorConfig()
        self.device = device
        self.model.eval()
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        name: str = "test",
    ) -> EvaluationMetrics:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader with evaluation data
            name: Name for this evaluation (e.g., 'test', 'validation')
            
        Returns:
            EvaluationMetrics with all computed metrics
        """
        print(f"\nEvaluating on {name} set...")
        
        metrics = evaluate_model(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
            max_batches=self.config.max_batches,
        )
        
        print(f"  Loss: {metrics.loss:.4f}")
        print(f"  BPB: {metrics.bpb:.4f}")
        print(f"  Perplexity: {metrics.perplexity:.2f}")
        print(f"  Accuracy: {metrics.accuracy * 100:.2f}%")
        print(f"  Top-5 Accuracy: {metrics.top5_accuracy * 100:.2f}%")
        
        return metrics
    
    def evaluate_generation(
        self,
        prompts: Optional[List[str]] = None,
    ) -> Tuple[List[str], EvaluationMetrics]:
        """
        Evaluate generation quality.
        
        Args:
            prompts: List of prompts to generate from
            
        Returns:
            Tuple of (generated_texts, metrics)
        """
        if prompts is None:
            prompts = self.config.generation_prompts or self._get_default_prompts()
        
        print(f"\nEvaluating generation quality ({len(prompts)} prompts)...")
        
        generated_texts, metrics = evaluate_generation_quality(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=self.config.generation_max_tokens,
            device=self.device,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
        
        print(f"  UTF-8 Validity: {metrics.utf8_validity * 100:.2f}%")
        print(f"  Avg Generation Length: {metrics.avg_generation_length:.1f} tokens")
        print(f"  Throughput: {metrics.tokens_per_second:.1f} tokens/sec")
        
        # Save generations if configured
        if self.config.save_generations:
            self._save_generations(prompts, generated_texts)
        
        return generated_texts, metrics
    
    def profile_efficiency(self) -> Dict[str, Any]:
        """
        Profile model efficiency (memory, latency, FLOPs).
        
        Returns:
            Dict with efficiency metrics
        """
        print("\nProfiling model efficiency...")
        
        results = {}
        
        # Memory footprint
        if self.config.profile_memory:
            memory = compute_memory_footprint(self.model)
            results.update({
                "total_params": memory["total_params"],
                "trainable_params": memory["trainable_params"],
                "param_memory_mb": memory["param_memory_mb"],
                "total_memory_mb": memory["total_memory_mb"],
            })
            print(f"  Parameters: {memory['total_params']:,}")
            print(f"  Memory: {memory['total_memory_mb']:.2f} MB")
        
        # Latency profiling
        if self.config.profile_latency:
            latency_results = self._profile_latency()
            results.update(latency_results)
            print(f"  Latency (forward): {latency_results['forward_latency_ms']:.2f} ms")
            print(f"  Latency (generation per token): {latency_results['generation_latency_per_token_ms']:.2f} ms")
        
        return results
    
    def _profile_latency(self) -> Dict[str, float]:
        """Profile forward and generation latency."""
        # Create sample input
        sample_len = 128
        input_ids = torch.randint(0, 256, (1, sample_len), device=self.device)
        
        # Warmup
        for _ in range(self.config.latency_warmup_runs):
            with torch.no_grad():
                _ = self.model(input_ids)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure forward pass
        start = time.time()
        for _ in range(self.config.latency_measure_runs):
            with torch.no_grad():
                _ = self.model(input_ids)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        forward_time = (time.time() - start) / self.config.latency_measure_runs
        
        # Measure generation
        gen_tokens = 50
        start = time.time()
        with torch.no_grad():
            _ = self.model.generate(
                input_ids[:, :20],  # Short prompt
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=0,
            )
        if self.device == 'cuda':
            torch.cuda.synchronize()
        gen_time = time.time() - start
        
        return {
            "forward_latency_ms": forward_time * 1000,
            "forward_tokens_per_sec": sample_len / forward_time,
            "generation_latency_per_token_ms": (gen_time / gen_tokens) * 1000,
            "generation_tokens_per_sec": gen_tokens / gen_time,
        }
    
    def _get_default_prompts(self) -> List[str]:
        """Get default prompts for generation evaluation."""
        return [
            "Once upon a time",
            "The quick brown fox",
            "In a galaxy far away",
            "Hello, my name is",
            "The most important thing",
            "Yesterday, I went to",
            "The weather today is",
            "Scientists have discovered",
            "In conclusion, we can say",
            "The story begins with",
        ]
    
    def _save_generations(self, prompts: List[str], texts: List[str]) -> None:
        """Save generated texts to file."""
        output_file = self.output_dir / "generations.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (prompt, text) in enumerate(zip(prompts, texts)):
                f.write(f"=== Sample {i + 1} ===\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Generated:\n{text}\n\n")
        print(f"  Generations saved to {output_file}")
    
    def run_full_evaluation(
        self,
        test_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            test_dataloader: Test set DataLoader
            val_dataloader: Optional validation set DataLoader
            
        Returns:
            Dict with all evaluation results
        """
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": type(self.model).__name__,
        }
        
        # Evaluate on test set
        test_metrics = self.evaluate_dataset(test_dataloader, "test")
        results["test"] = test_metrics.to_dict()
        
        # Evaluate on validation set if provided
        if val_dataloader is not None:
            val_metrics = self.evaluate_dataset(val_dataloader, "validation")
            results["validation"] = val_metrics.to_dict()
        
        # Evaluate generation quality
        generated_texts, gen_metrics = self.evaluate_generation()
        results["generation"] = gen_metrics.to_dict()
        results["sample_generations"] = generated_texts[:5]  # First 5
        
        # Profile efficiency
        efficiency = self.profile_efficiency()
        results["efficiency"] = efficiency
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to JSON."""
        output_file = self.output_dir / "evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")
    
    def compare_checkpoints(
        self,
        checkpoint_paths: List[str],
        test_dataloader: DataLoader,
        load_fn,  # Function to load model from checkpoint
    ) -> Dict[str, Dict]:
        """
        Compare multiple checkpoints on the same test set.
        
        Args:
            checkpoint_paths: List of checkpoint file paths
            test_dataloader: Test set DataLoader
            load_fn: Function(path) -> model to load checkpoints
            
        Returns:
            Dict mapping checkpoint name to metrics
        """
        results = {}
        
        for path in checkpoint_paths:
            name = Path(path).stem
            print(f"\n{'=' * 50}")
            print(f"Evaluating: {name}")
            print('=' * 50)
            
            # Load model
            model = load_fn(path)
            model = model.to(self.device)
            model.eval()
            
            # Temporarily swap model
            original_model = self.model
            self.model = model
            
            # Evaluate
            metrics = self.evaluate_dataset(test_dataloader, name)
            results[name] = metrics.to_dict()
            
            # Restore original model
            self.model = original_model
        
        # Print comparison table
        self._print_comparison_table(results)
        
        return results
    
    def _print_comparison_table(self, results: Dict[str, Dict]) -> None:
        """Print comparison table of checkpoints."""
        print("\n" + "=" * 70)
        print("Checkpoint Comparison")
        print("=" * 70)
        print(f"{'Checkpoint':<25} {'Loss':<10} {'BPB':<10} {'PPL':<10} {'Acc':<10}")
        print("-" * 70)
        
        for name, metrics in results.items():
            print(f"{name:<25} "
                  f"{metrics['loss']:<10.4f} "
                  f"{metrics['bpb']:<10.4f} "
                  f"{metrics['perplexity']:<10.2f} "
                  f"{metrics['accuracy'] * 100:<10.2f}%")
        
        print("=" * 70)


def create_evaluator(
    model: nn.Module,
    tokenizer,
    output_dir: str = "./evaluation_results",
    device: str = 'cuda',
    **kwargs,
) -> Evaluator:
    """
    Factory function to create an Evaluator.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for encoding/decoding
        output_dir: Directory to save results
        device: Device to use
        **kwargs: Additional EvaluatorConfig parameters
        
    Returns:
        Configured Evaluator instance
    """
    config = EvaluatorConfig(output_dir=output_dir, **kwargs)
    return Evaluator(model, tokenizer, config, device)
