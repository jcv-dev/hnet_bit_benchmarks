# -*- coding: utf-8 -*-

"""
Causal Multi-Head Attention with sliding window for HNetBit innermost stage.

Uses standard nn.Linear (full-precision) for Q/K/V/O projections to ensure
stable attention score computation, unlike the rest of hnet_bit which uses
ternary BitLinear.

Supports:
- Sliding-window causal attention (configurable window_size)
- RoPE rotary positional embeddings
- KV cache for single-token generation
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet_bit.ops.bitnet import RMSNorm
from hnet_bit.ops.activations import swiglu
from hnet_bit.ops.fusedbitnet import FusedBitLinear as BitLinear


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE)."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        if offset + seq_len > self.cos_cached.shape[0]:
            self._build_cache(offset + seq_len)
        return (
            self.cos_cached[offset:offset + seq_len],
            self.sin_cached[offset:offset + seq_len],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensors."""
    return (x * cos) + (_rotate_half(x) * sin)


class CausalMHABit(nn.Module):
    """
    Causal Multi-Head Attention with sliding window for HNetBit innermost stage.
    
    Uses full-precision nn.Linear for Q/K/V/O projections (not BitLinear) for
    attention score stability. Supports sliding-window causal masking and RoPE.
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        window_size: Sliding window size for attention (0 = full causal)
        max_position_embeddings: Maximum sequence length for RoPE
        rms_norm_eps: Epsilon for RMSNorm
        layer_idx: Index of this layer (for KV cache)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        window_size: int = 64,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.layer_idx = layer_idx

        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        # Full-precision projections for stable attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
        )

        self.scale = self.head_dim ** -0.5

    def _build_sliding_window_mask(
        self, seq_len: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a sliding-window causal attention mask."""
        # Causal mask: position i can attend to positions [max(0, i-window+1), i]
        row = torch.arange(seq_len, device=device)
        col = torch.arange(seq_len, device=device)
        mask = (col[None, :] <= row[:, None])  # Causal
        if self.window_size > 0:
            mask = mask & (row[:, None] - col[None, :] < self.window_size)  # Window
        # Convert to float mask: 0 for attend, -inf for don't attend
        attn_mask = torch.where(mask, 0.0, float('-inf')).to(dtype)
        return attn_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[object]]:
        """
        Forward pass.
        
        Args:
            hidden_states: (B, L, D) input tensor
            attention_mask: Optional (B, L) boolean mask
            past_key_values: Cache wrapper for KV cache
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output, attn_weights_or_none, updated_cache)
        """
        B, L, D = hidden_states.shape

        # Get position offset from cache
        offset = 0
        if past_key_values is not None and use_cache:
            kv_state = past_key_values[self.layer_idx]
            if kv_state is not None:
                offset = kv_state[0].shape[2]  # (B, H, cached_len, head_dim)

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(L, offset=offset)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # Concatenate with cached K, V
        if past_key_values is not None and use_cache:
            kv_state = past_key_values[self.layer_idx]
            if kv_state is not None:
                cached_k, cached_v = kv_state
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
            # Update cache
            past_key_values.update((k, v), self.layer_idx, L)

        # Compute attention
        total_len = k.shape[2]

        if L == 1:
            # Single-token generation: no mask needed, just attend to all cached + current
            # But apply window constraint
            if self.window_size > 0 and total_len > self.window_size:
                k = k[:, :, -self.window_size:]
                v = v[:, :, -self.window_size:]
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # Build sliding window causal mask
            attn_mask = self._build_sliding_window_mask(total_len, hidden_states.device, q.dtype)
            # If we have cached tokens, we only need the last L rows of the mask
            if total_len > L:
                attn_mask = attn_mask[-L:]
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.o_proj(attn_output)

        attn_weights = None  # Not returning weights by default

        return output, attn_weights, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize empty KV cache for generation."""
        param = next(self.parameters())
        k_cache = param.new_zeros(batch_size, self.num_heads, 0, self.head_dim)
        v_cache = param.new_zeros(batch_size, self.num_heads, 0, self.head_dim)
        return (k_cache, v_cache)


class CausalMHABlock(nn.Module):
    """
    Attention block with pre-norm residual, matching HGRNBitBlock interface.
    
    Architecture:
        residual = x
        x = attn_norm(x)
        x = attn(x) + residual
        residual = x
        x = mlp_norm(x)
        x = mlp(x) + residual
    
    Args:
        config: Stage config object
        layer_idx: Index of this layer
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Determine attention params from config
        num_heads = getattr(config, 'attention_num_heads', None) or config.num_heads
        window_size = getattr(config, 'attention_window_size', 64)
        max_pos = getattr(config, 'max_position_embeddings', 2048)

        # Attention with pre-normalization
        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalMHABit(
            hidden_size=config.hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            max_position_embeddings=max_pos,
            rms_norm_eps=config.rms_norm_eps,
            layer_idx=layer_idx,
        )

        # MLP with pre-normalization (reuse HGRNBitMLP for consistency)
        from hnet_bit.layers.hgrn_bit import HGRNBitMLP
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HGRNBitMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[object]]:
        """Forward pass matching HGRNBitBlock interface."""
        # Attention with pre-norm
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attn_weights, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights, past_key_values
