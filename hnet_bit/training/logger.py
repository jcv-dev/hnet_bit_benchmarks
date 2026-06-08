# -*- coding: utf-8 -*-

"""
Training logger with console, CSV, TensorBoard, and optional WandB support.

Enhanced features:
- TensorBoard logging
- Ternary weight statistics tracking
- Gradient norm logging
- Generation samples during training
- Multiple logging levels (minimal, standard, verbose)
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn


class TrainingLogger:
    """
    Logger for training metrics.
    
    Outputs to console and CSV file. Optionally logs to TensorBoard and WandB.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name for this experiment
        use_wandb: Whether to use WandB logging
        use_tensorboard: Whether to use TensorBoard logging
        wandb_project: WandB project name
        log_level: 'minimal', 'standard', or 'verbose'
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "experiment",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: str = "simplified-slm",
        log_level: str = "standard",
    ):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_level = log_level
        os.makedirs(log_dir, exist_ok=True)
        
        # CSV logging
        self.csv_path = os.path.join(log_dir, "training_log.csv")
        self.csv_file = None
        self.csv_writer = None
        self._csv_initialized = False
        
        # Evaluation CSV
        self.eval_csv_path = os.path.join(log_dir, "eval_log.csv")
        self.eval_csv_file = None
        self.eval_csv_writer = None
        self._eval_csv_initialized = False
        
        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_log_dir = os.path.join(log_dir, "tensorboard")
                self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError:
                print("WARNING: tensorboard not installed, skipping TensorBoard logging")
        
        # WandB
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    dir=log_dir,
                )
            except ImportError:
                print("WARNING: wandb not installed, skipping WandB logging")
        
        self.start_time = time.time()
        self._step_start = time.time()
        self._last_log_step = 0

    def _init_csv(self, keys: List[str]) -> None:
        """Initialize CSV writer with headers."""
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=keys)
        self.csv_writer.writeheader()
        self._csv_initialized = True

    def _init_eval_csv(self, keys: List[str]) -> None:
        """Initialize evaluation CSV writer."""
        self.eval_csv_file = open(self.eval_csv_path, 'w', newline='')
        self.eval_csv_writer = csv.DictWriter(self.eval_csv_file, fieldnames=keys)
        self.eval_csv_writer.writeheader()
        self._eval_csv_initialized = True

    def log(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "train",
    ) -> None:
        """
        Log metrics for a training step.
        
        Args:
            metrics: Dict of metric name → value
            step: Current training step
            prefix: Prefix for metric names (default: 'train')
        """
        metrics = dict(metrics)  # Copy
        metrics['step'] = step
        metrics['elapsed_seconds'] = time.time() - self.start_time
        
        # Compute BPB (bits per byte) from loss
        if 'loss' in metrics:
            metrics['bpb'] = metrics['loss'] / math.log(2)
            metrics['perplexity'] = math.exp(min(metrics['loss'], 20))
        
        # Console output
        if self.log_level != 'minimal':
            self._print_metrics(metrics, step)
        
        # CSV
        if not self._csv_initialized:
            self._init_csv(list(metrics.keys()))
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()
        
        # TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if key in ('step', 'elapsed_seconds'):
                    continue
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"{prefix}/{key}", value, step)
        
        # WandB
        if self.wandb_run is not None:
            import wandb
            prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb.log(prefixed, step=step)
        
        self._last_log_step = step

    def _print_metrics(self, metrics: Dict, step: int) -> None:
        """Print metrics to console."""
        parts = [f"step={step:>6d}"]
        
        # Order of metrics to display
        priority_keys = ['loss', 'bpb', 'lr', 'learning_rate', 'grad_norm']
        
        for k in priority_keys:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, float):
                    if k in ('lr', 'learning_rate'):
                        parts.append(f"{k}={v:.2e}")
                    else:
                        parts.append(f"{k}={v:.4f}")
        
        # Add other metrics if verbose
        if self.log_level == 'verbose':
            for k, v in metrics.items():
                if k not in priority_keys and k not in ('step', 'elapsed_seconds'):
                    if isinstance(v, float):
                        parts.append(f"{k}={v:.4f}")
        
        print(" | ".join(parts))

    def log_eval(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log evaluation metrics.
        
        Args:
            metrics: Evaluation metrics
            step: Current training step
        """
        print(f"  [EVAL] step={step} | " + " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
            if k != 'step'
        ))
        
        # CSV
        eval_metrics = dict(metrics)
        eval_metrics['step'] = step
        if not self._eval_csv_initialized:
            self._init_eval_csv(list(eval_metrics.keys()))
        self.eval_csv_writer.writerow(eval_metrics)
        self.eval_csv_file.flush()
        
        # TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"eval/{key}", value, step)
        
        # WandB
        if self.wandb_run is not None:
            import wandb
            eval_metrics_wb = {f"eval/{k}": v for k, v in metrics.items()}
            wandb.log(eval_metrics_wb, step=step)

    def log_gradients(
        self,
        model: nn.Module,
        step: int,
        log_histogram: bool = False,
    ) -> Dict[str, float]:
        """
        Log gradient statistics.
        
        Args:
            model: Model to analyze gradients
            step: Current training step
            log_histogram: Whether to log gradient histograms (expensive)
            
        Returns:
            Dict with gradient statistics
        """
        total_norm = 0.0
        grad_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                if self.log_level == 'verbose':
                    grad_norms[f"grad_norm/{name}"] = param_norm
                
                if log_histogram and self.tb_writer is not None:
                    self.tb_writer.add_histogram(f"gradients/{name}", param.grad, step)
        
        total_norm = total_norm ** 0.5
        
        # Log to TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("gradients/total_norm", total_norm, step)
        
        return {"grad_norm": total_norm}

    def log_ternary_weights(
        self,
        model: nn.Module,
        step: int,
    ) -> Dict[str, float]:
        """
        Log ternary weight statistics for BitLinear layers.
        
        Args:
            model: Model with BitLinear layers
            step: Current training step
            
        Returns:
            Dict with weight distribution statistics
        """
        total_weights = 0
        weight_counts = {-1: 0, 0: 0, 1: 0}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                total_weights += param.numel()
                
                # Simulate quantization
                scale = 1.0 / param.abs().mean().clamp(min=1e-5)
                quantized = (param * scale).round().clamp(-1, 1)
                
                for val in [-1, 0, 1]:
                    weight_counts[val] += (quantized == val).sum().item()
        
        if total_weights == 0:
            return {}
        
        # Compute percentages
        stats = {
            "ternary_weights/pct_neg1": 100 * weight_counts[-1] / total_weights,
            "ternary_weights/pct_zero": 100 * weight_counts[0] / total_weights,
            "ternary_weights/pct_pos1": 100 * weight_counts[1] / total_weights,
        }
        
        # Log to TensorBoard
        if self.tb_writer is not None:
            for key, value in stats.items():
                self.tb_writer.add_scalar(key, value, step)
        
        return stats

    def log_generation_sample(
        self,
        model: nn.Module,
        tokenizer,
        step: int,
        prompt: str = "Once upon a time",
        max_new_tokens: int = 50,
        device: str = 'cuda',
    ) -> str:
        """
        Generate and log a text sample.
        
        Args:
            model: Language model
            tokenizer: ByteTokenizer
            step: Current training step
            prompt: Generation prompt
            max_new_tokens: Tokens to generate
            device: Device for generation
            
        Returns:
            Generated text
        """
        model.eval()
        
        # Encode prompt
        encoded = tokenizer.encode([prompt])[0]['input_ids']
        input_ids = torch.tensor([encoded.tolist()], dtype=torch.long, device=device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=0,
            )
        
        # Decode
        output_tokens = output_ids[0].tolist()
        generated_text = tokenizer.decode(output_tokens, errors='replace')
        
        # Log
        print(f"\n  [GEN SAMPLE @ step {step}]")
        print(f"  Prompt: {prompt}")
        print(f"  Output: {generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")
        print()
        
        # TensorBoard text
        if self.tb_writer is not None:
            self.tb_writer.add_text("generations", f"**Prompt:** {prompt}\n\n**Output:** {generated_text}", step)
        
        model.train()
        return generated_text

    def log_learning_rate(self, lr: float, step: int) -> None:
        """Log learning rate."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("train/learning_rate", lr, step)

    def log_memory(self, step: int) -> Dict[str, float]:
        """Log GPU memory usage."""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        stats = {
            "memory/allocated_mb": allocated,
            "memory/reserved_mb": reserved,
        }
        
        if self.tb_writer is not None:
            for key, value in stats.items():
                self.tb_writer.add_scalar(key, value, step)
        
        return stats

    def log_hparams(self, hparams: Dict, metrics: Dict) -> None:
        """Log hyperparameters and final metrics."""
        if self.tb_writer is not None:
            self.tb_writer.add_hparams(hparams, metrics)
        
        if self.wandb_run is not None:
            import wandb
            wandb.config.update(hparams)

    def close(self) -> None:
        """Clean up resources."""
        if self.csv_file is not None:
            self.csv_file.close()
        if self.eval_csv_file is not None:
            self.eval_csv_file.close()
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
        
        print(f"\nLogs saved to: {self.log_dir}")


class MetricsAggregator:
    """
    Aggregate metrics over multiple steps for logging.
    
    Useful for accumulating metrics over gradient accumulation steps.
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}
    
    def add(self, key: str, value: float, count: int = 1) -> None:
        """Add a metric value."""
        if key not in self.metrics:
            self.metrics[key] = []
            self.counts[key] = 0
        self.metrics[key].append(value * count)
        self.counts[key] += count
    
    def add_dict(self, metrics: Dict[str, float], count: int = 1) -> None:
        """Add multiple metrics."""
        for key, value in metrics.items():
            self.add(key, value, count)
    
    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics."""
        result = {}
        for key in self.metrics:
            total = sum(self.metrics[key])
            count = self.counts[key]
            result[key] = total / max(count, 1)
        return result
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()
