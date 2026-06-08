# -*- coding: utf-8 -*-

"""
Model factory for the three architectures benchmarked on Spanish Billion Words.

Model A – Transformer BPE  (LlamaForCausalLM from HuggingFace)
Model B – MatMul‑free LM   (HGRNBitForCausalLM from matmulfreellm)
Model C – Hybrid            (HNetBitForCausalLM from hnet_bit)

Each factory returns:
    (model, is_byte_level, vocab_size)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure local repos are importable
# ---------------------------------------------------------------------------
_TESIS = Path(__file__).resolve().parent
_MATMULFREE = _TESIS / "matmulfreellm"
_HNET_BIT = _TESIS / "hnet_bit"

for _p in [str(_TESIS), str(_MATMULFREE), str(_HNET_BIT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ============================================================================
# Size configurations
# ============================================================================

TRANSFORMER_CONFIGS = {
    "tiny": dict(hidden_size=64,  num_hidden_layers=2, num_attention_heads=2,
                 intermediate_size=128,  max_position_embeddings=256),
    "150M": dict(hidden_size=768,  num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072,  max_position_embeddings=2048),
    "350M": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                 intermediate_size=4096,  max_position_embeddings=2048),
    "750M": dict(hidden_size=1536, num_hidden_layers=24, num_attention_heads=16,
                 intermediate_size=6144,  max_position_embeddings=2048),
}

MATMULFREE_CONFIGS = {
    "tiny": dict(hidden_size=64,  num_hidden_layers=2, vocab_size=256),
    "150M": dict(hidden_size=768,  num_hidden_layers=16, vocab_size=256),
    "350M": dict(hidden_size=1024, num_hidden_layers=24, vocab_size=256),
    "750M": dict(hidden_size=1536, num_hidden_layers=28, vocab_size=256),
}

HYBRID_CONFIGS = {
    # Smoke test: 1‑stage hierarchy, tiny
    "tiny": dict(
        d_model=[48, 64],
        num_blocks=[[1, 0, 1], [2]],
        num_heads=1, expand_ratio=1, hidden_ratio=2,
    ),
    # 1‑stage hierarchy — d_model=[outer, inner]
    "150M": dict(
        d_model=[576, 768],
        num_blocks=[[4, 0, 4], [10]],
        num_heads=4, expand_ratio=2, hidden_ratio=4,
    ),
    # 2‑stage hierarchy — d_model=[outer, mid, inner]
    "350M": dict(
        d_model=[640, 896, 1152],
        num_blocks=[[4, 0, 4], [4, 0, 4], [12]],
        num_heads=4, expand_ratio=2, hidden_ratio=4,
    ),
    "750M": dict(
        d_model=[896, 1152, 1536],
        num_blocks=[[6, 0, 6], [6, 0, 6], [16]],
        num_heads=4, expand_ratio=2, hidden_ratio=4,
    ),
}


# ============================================================================
# Factory functions
# ============================================================================

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_transformer(size: str, tokenizer_name: str = "gpt2") -> Tuple[nn.Module, bool, int]:
    """
    Build a Llama‑style Transformer baseline.

    Uses GPT‑2 tokenizer for BPE (change via --tokenizer_name).
    Returns (model, is_byte_level=False, vocab_size).
    """
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size

    cfg_overrides = TRANSFORMER_CONFIGS[size]
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        use_cache=True,
        **cfg_overrides,
    )

    model = LlamaForCausalLM(config)
    print(f"[Transformer {size}] Parameters: {_count_params(model):,}  vocab={vocab_size}")
    return model, False, vocab_size


def build_matmulfree(size: str) -> Tuple[nn.Module, bool, int]:
    """
    Build a MatMul‑free LM (HGRNBit) at byte level.

    Returns (model, is_byte_level=True, vocab_size=256).
    """
    from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
    from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM

    cfg_overrides = MATMULFREE_CONFIGS[size]
    config = HGRNBitConfig(
        attn_mode="fused_recurrent",
        num_heads=1,
        expand_ratio=1,
        hidden_ratio=4,
        hidden_act="swish",
        use_short_conv=False,
        use_lower_bound=True,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        use_cache=True,
        fuse_cross_entropy=False,
        bos_token_id=254,
        eos_token_id=255,
        **cfg_overrides,
    )

    model = HGRNBitForCausalLM(config)
    print(f"[MatMulFree {size}] Parameters: {_count_params(model):,}  vocab=256")
    return model, True, 256


def build_hybrid(size: str) -> Tuple[nn.Module, bool, int]:
    """
    Build the Hybrid model (HNetBit = H‑Net chunking + MLGRU backbone).

    Returns (model, is_byte_level=True, vocab_size=256).
    """
    from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

    cfg_overrides = HYBRID_CONFIGS[size]
    config = HNetBitConfig(
        vocab_size=256,
        attn_mode="fused_recurrent",
        hidden_act="swish",
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        use_cache=True,
        use_fused_bitlinear=False,
        use_short_conv=True,
        conv_size=4,
        share_conv_kernel=True,
        use_lower_bound=False,
        bos_token_id=254,
        eos_token_id=255,
        **cfg_overrides,
    )

    model = HNetBitForCausalLM(config)
    print(f"[Hybrid {size}] Parameters: {_count_params(model):,}  vocab=256")
    return model, True, 256


def build_hybrid_attn(size: str) -> Tuple[nn.Module, bool, int]:
    """
    Build the Hybrid model with attention in the innermost stage.

    Same architecture as build_hybrid, but interleaves CausalMHABit
    with HGRN blocks at the deepest hierarchy level. Pattern is
    "xaxa" (HGRN-Attention-HGRN-Attention...) with sliding window
    size 64 and RoPE positional embeddings.

    Returns (model, is_byte_level=True, vocab_size=256).
    """
    from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

    cfg_overrides = HYBRID_CONFIGS[size]
    config = HNetBitConfig(
        vocab_size=256,
        attn_mode="fused_recurrent",
        hidden_act="swish",
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        use_cache=True,
        use_fused_bitlinear=False,
        use_short_conv=True,
        conv_size=4,
        share_conv_kernel=True,
        use_lower_bound=False,
        bos_token_id=254,
        eos_token_id=255,
        innermost_use_attention=True,
        attention_layers_pattern="xaxa",
        attention_window_size=64,
        **cfg_overrides,
    )

    model = HNetBitForCausalLM(config)
    print(f"[Hybrid-Attn {size}] Parameters: {_count_params(model):,}  vocab=256")
    return model, True, 256


# ============================================================================
# Dispatcher
# ============================================================================

MODEL_BUILDERS = {
    "transformer": build_transformer,
    "matmulfree": build_matmulfree,
    "hybrid": build_hybrid,
    "hybrid_attn": build_hybrid_attn,
}


def build_model(model_name: str, size: str, **kwargs) -> Tuple[nn.Module, bool, int]:
    """
    Build a model by name and size.

    Args:
        model_name: 'transformer', 'matmulfree', or 'hybrid'
        size: '150M', '350M', or '750M'

    Returns:
        (model, is_byte_level, vocab_size)
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_BUILDERS)}")
    if size not in ("tiny", "150M", "350M", "750M"):
        raise ValueError(f"Unknown size: {size}. Choose from tiny, 150M, 350M, 750M")
    return MODEL_BUILDERS[model_name](size, **kwargs)
