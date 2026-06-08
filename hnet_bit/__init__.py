# -*- coding: utf-8 -*-

"""
Simplified MatMul-Free Small Language Model with Ternary Weights

A minimal implementation combining:
- BitLinear ternary quantization ({-1, 0, +1}) from matmulfreellm
- HGRN (Hierarchical Gated Recurrent Network) recurrent layers
- Chunk-parallel recurrence for efficient training

Architecture:
    Embedding → N×(RMSNorm→HGRN→Residual, RMSNorm→MLP→Residual) → RMSNorm → LM Head

Configuration:
    - vocab_size: 256 (byte-level tokenization)
    - hidden_size: 512
    - num_layers: 6
    - num_heads: 1
    - use_short_conv: False
    - use_lower_bound: False
"""

__version__ = "0.1.0"

from hnet_bit.ops._triton import triton_available, triton_failure_reason

_ok = triton_available()
_reason = triton_failure_reason()
if _ok:
    pass  # Triton is working
elif _reason:
    import warnings
    warnings.warn(f"Triton unavailable: {_reason}")

from hnet_bit.models import HNetBitConfig, HNetBitForCausalLM

__all__ = [
    "HNetBitConfig",
    "HNetBitForCausalLM",
]
