# -*- coding: utf-8 -*-

"""
Metrics for the Spanish Billion Words benchmark.

Two metrics:
    1. Bits‑Per‑Byte (BPB)
    2. Inference Memory (peak GPU MB)
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================================
# Metric 1: Bits-Per-Byte (BPB)
# ============================================================================

@torch.no_grad()
def compute_bpb(
    model: nn.Module,
    dataloader: DataLoader,
    is_byte_level: bool,
    avg_bytes_per_token: float = 1.0,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute Bits‑Per‑Byte on a validation set.

    For byte‑level models:
        BPB = mean_NLL / ln(2)

    For BPE models:
        Token‑level NLL is converted to BPB by accounting for the average
        number of bytes each token represents:
        BPB = mean_NLL / (avg_bytes_per_token × ln(2))

    Args:
        model: The language model in eval mode.
        dataloader: Validation DataLoader.
        is_byte_level: True for byte/char models, False for BPE.
        avg_bytes_per_token: Average bytes per BPE token.
        device: Torch device.
        max_batches: Cap the number of batches (for speed).

    Returns:
        dict with 'bpb', 'loss', 'num_tokens', 'num_bytes'.
    """
    model.eval()

    total_nll = 0.0       # sum of per‑token NLL (nats)
    total_tokens = 0      # number of valid prediction positions
    total_bytes = 0       # effective bytes counted

    amp_dtype = torch.bfloat16 if next(model.parameters()).dtype == torch.bfloat16 else torch.float16

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

        # outputs.loss is already averaged per token by CrossEntropyLoss
        mask = labels != -100
        n_tok = mask.sum().item()

        total_nll += outputs.loss.item() * n_tok
        total_tokens += n_tok

    if total_tokens == 0:
        return {"bpb": float("inf"), "loss": float("inf"),
                "num_tokens": 0, "num_bytes": 0}

    mean_nll = total_nll / total_tokens

    if is_byte_level:
        bpb = mean_nll / math.log(2)
        total_bytes = total_tokens  # 1 byte = 1 token
    else:
        bpb = mean_nll / (avg_bytes_per_token * math.log(2))
        total_bytes = int(total_tokens * avg_bytes_per_token)

    return {
        "bpb": bpb,
        "loss": mean_nll,
        "num_tokens": total_tokens,
        "num_bytes": total_bytes,
    }


# ============================================================================
# Metric 2: Inference Memory
# ============================================================================

@torch.no_grad()
def measure_inference_memory(
    model: nn.Module,
    seq_length: int = 2048,
    is_byte_level: bool = True,
    vocab_size: int = 256,
    device: str = "cuda",
    warmup_runs: int = 3,
) -> Dict[str, float]:
    """
    Measure peak GPU memory during a single forward pass.

    Sets batch_size=1, runs a forward pass with the given sequence length,
    and reports ``torch.cuda.max_memory_allocated()``.

    Args:
        model: The model to profile.
        seq_length: Sequence length (bytes for byte‑level, tokens for BPE).
        is_byte_level: Whether the model is byte‑level.
        vocab_size: Vocabulary size for generating input ids.
        device: CUDA device.
        warmup_runs: Number of warmup forward passes to stabilise CUDA context.

    Returns:
        dict with 'peak_memory_mb', 'allocated_memory_mb'.
    """
    if not torch.cuda.is_available():
        return {"peak_memory_mb": 0.0, "allocated_memory_mb": 0.0}

    model.eval()
    model = model.to(device)

    input_ids = torch.randint(0, vocab_size, (1, seq_length), device=device)

    # Warmup passes (to prime CUDA allocator / JIT caches)
    for _ in range(warmup_runs):
        _ = model(input_ids=input_ids)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Measured forward pass
    _ = model(input_ids=input_ids)
    torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)

    return {
        "peak_memory_mb": peak_mb,
        "allocated_memory_mb": alloc_mb,
    }


# ============================================================================
# Combined evaluation
# ============================================================================

def run_full_eval(
    model: nn.Module,
    val_loader: DataLoader,
    is_byte_level: bool,
    avg_bytes_per_token: float = 1.0,
    vocab_size: int = 256,
    inference_seq_length: int = 2048,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run all metrics (BPB + inference memory) and return a combined dict.
    """
    bpb_results = compute_bpb(
        model, val_loader, is_byte_level,
        avg_bytes_per_token=avg_bytes_per_token,
        device=device, max_batches=max_batches,
    )

    mem_results = measure_inference_memory(
        model, seq_length=inference_seq_length,
        is_byte_level=is_byte_level, vocab_size=vocab_size,
        device=device,
    )

    return {**bpb_results, **mem_results}
