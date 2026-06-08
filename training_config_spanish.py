# -*- coding: utf-8 -*-

"""
Training configuration for the Spanish Billion Words benchmark.

Provides:
    - WSD (Warmup–Stable–Decay) learning‑rate scheduler
    - Per‑model training configurations
    - Gradient checkpointing helpers
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ============================================================================
# WSD scheduler (Warmup → Stable → Decay)
# ============================================================================

def build_wsd_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Warmup–Stable–Decay scheduler as described in the H‑Net paper.

    LR ramps linearly during [0, warmup_steps],
    stays constant during (warmup_steps, warmup_steps + stable_steps],
    then decays via cosine to ``min_lr_ratio * peak_lr`` during the rest.

    Args:
        optimizer: The optimizer.
        warmup_steps: Number of linear warmup steps.
        stable_steps: Number of steps at peak LR.
        decay_steps: Number of cosine decay steps.
        min_lr_ratio: Minimum LR as a fraction of peak at end of decay.

    Returns:
        A LambdaLR scheduler.
    """
    total_steps = warmup_steps + stable_steps + decay_steps

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        elif step < warmup_steps + stable_steps:
            # Stable
            return 1.0
        else:
            # Cosine decay
            decay_progress = (step - warmup_steps - stable_steps) / float(max(1, decay_steps))
            decay_progress = min(decay_progress, 1.0)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    return LambdaLR(optimizer, _lr_lambda)


# ============================================================================
# Training configuration dataclass
# ============================================================================

@dataclass
class SpanishTrainingConfig:
    """Configuration for training on Spanish Billion Words."""

    # Model identity
    model_name: str = "hybrid"      # 'transformer', 'matmulfree', 'hybrid'
    model_size: str = "350M"        # '150M', '350M', '750M'

    # Paths
    output_dir: str = "./runs/spanish"
    cache_dir: str = "./data/spanish"

    # Sequence lengths (bytes for byte‑level, tokens for BPE)
    byte_seq_length: int = 8192
    token_seq_length: int = 1792    # approx 8192 bytes / 4.57 bytes‑per‑token

    # Training schedule (total_training_bytes = bytes of underlying text, same for all models)
    total_training_bytes: int = 100_000_000_000   # 100B
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Optimizer
    learning_rate: float = 0.0       # 0 = auto‑select per model/size
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # WSD scheduler fractions (of total steps)
    warmup_fraction: float = 0.01    # 1% warmup
    stable_fraction: float = 0.79    # 79% stable
    decay_fraction: float = 0.20     # 20% cosine decay
    min_lr_ratio: float = 0.0

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Gradient checkpointing
    gradient_checkpointing: bool = True

    # Evaluation
    eval_interval_steps: int = 1000     # BPB every 1k steps
    save_interval_steps: int = 5000
    log_interval_steps: int = 10

    # Checkpointing milestones (in bytes of training text) for final metrics
    checkpoint_milestones: List[int] = field(
        default_factory=lambda: [25_000_000_000, 50_000_000_000, 100_000_000_000]
    )

    # Inference memory measurement
    inference_seq_length: int = 2048

    # Reproducibility
    seed: int = 42

    # Override for debugging (set via CLI --max_steps)
    max_steps_override: Optional[int] = None

    # Average bytes per BPE token (used for fair text budget across model types)
    avg_bytes_per_token: float = 1.0

    # Hierarchical model specific
    lr_multipliers: Optional[List[float]] = None
    lambda_lb: float = 0.01        # load‑balancing loss weight for hybrid
    downsampling_factor: float = 5.0  # Spanish avg word length ~5 chars

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def bytes_per_step(self) -> int:
        """Bytes of underlying text consumed per optimizer step (same unit for all model types)."""
        if self.model_name == "transformer":
            return int(self.effective_batch_size * self.token_seq_length * self.avg_bytes_per_token)
        return self.effective_batch_size * self.byte_seq_length

    @property
    def total_steps(self) -> int:
        return self.total_training_bytes // self.bytes_per_step

    @property
    def warmup_steps(self) -> int:
        return int(self.total_steps * self.warmup_fraction)

    @property
    def stable_steps(self) -> int:
        return int(self.total_steps * self.stable_fraction)

    @property
    def decay_steps(self) -> int:
        return self.total_steps - self.warmup_steps - self.stable_steps

    def resolve_learning_rate(self) -> float:
        """Auto‑select LR based on model and size if not explicitly set."""
        if self.learning_rate > 0:
            return self.learning_rate

        if self.model_name == "transformer":
            return 3e-4

        # MatMul‑free and Hybrid use higher LR per paper
        lr_map = {"150M": 4e-3, "350M": 2.5e-3, "750M": 1.5e-3}
        return lr_map.get(self.model_size, 2.5e-3)


# ============================================================================
# Optimizer builder
# ============================================================================

def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: SpanishTrainingConfig,
) -> tuple:
    """
    Build AdamW optimizer with weight‑decay grouping + WSD scheduler.

    Returns (optimizer, scheduler).
    """
    lr = config.resolve_learning_rate()

    if getattr(config, 'lr_multipliers', None) and hasattr(model, 'backbone') and hasattr(model.backbone, '_apply_lr_multiplier'):
        model.backbone._apply_lr_multiplier(config.lr_multipliers)

    param_groups_dict = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        is_no_decay = (param.dim() < 2 or any(nd in name for nd in ("bias", "norm", "pad_dimension", "embeddings")))
        wd = 0.0 if is_no_decay else config.weight_decay
        
        optim_config = getattr(param, "_optim", {})
        lr_mult = optim_config.get("lr_multiplier", 1.0)
        
        key = (wd, lr_mult)
        if key not in param_groups_dict:
            param_groups_dict[key] = []
        param_groups_dict[key].append(param)

    param_groups = [
        {"params": params, "weight_decay": wd, "lr": lr * lr_mult}
        for (wd, lr_mult), params in param_groups_dict.items()
    ]

    optimizer = AdamW(
        param_groups,
        lr=lr,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )

    scheduler = build_wsd_scheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        stable_steps=config.stable_steps,
        decay_steps=config.decay_steps,
        min_lr_ratio=config.min_lr_ratio,
    )

    print(f"[Optimizer] AdamW lr={lr:.2e}  total_steps={config.total_steps:,}")
    print(f"  warmup={config.warmup_steps:,}  stable={config.stable_steps:,}  "
          f"decay={config.decay_steps:,}")

    return optimizer, scheduler


# ============================================================================
# Gradient checkpointing helper
# ============================================================================

def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """
    Enable gradient checkpointing on the model to reduce memory usage.
    Handles different model types (LlamaForCausalLM, HGRNBitForCausalLM,
    HNetBitForCausalLM).
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("[GradCkpt] Enabled via model.gradient_checkpointing_enable()")
            return
        except Exception as e:
            print(f"[GradCkpt] Native HF enable failed: {e}")

    if hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing"):
        model.model.gradient_checkpointing = True
        print("[GradCkpt] Enabled via model.model.gradient_checkpointing = True")
    elif hasattr(model, "backbone"):
        # HNetBitForCausalLM: set on the backbone module
        model.backbone._gradient_checkpointing = True
        for module in model.backbone.modules():
            if hasattr(module, "_gradient_checkpointing"):
                module._gradient_checkpointing = True
        print("[GradCkpt] Enabled via HNetBit backbone")
    elif hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = True
        print("[GradCkpt] Enabled via model.gradient_checkpointing = True")
    else:
        print("[GradCkpt] WARNING: Could not enable gradient checkpointing")

