# -*- coding: utf-8 -*-

"""
HGRN-Bit layers for simplified SLM.

Implements:
- HGRNBitAttention: Gated recurrent attention with ternary weights
- HGRNBitMLP: Feed-forward network with SwiGLU and ternary weights
- HGRNBitBlock: Combined attention + MLP block

Simplified configuration:
- use_short_conv=False by default (enable for local inductive bias)
- use_lower_bound=False (no learned forget gate bounds)

Supports multi-head HGRN (num_heads > 1), state expansion (expand_ratio > 1),
and optional short convolution (use_short_conv=True) for local pattern capture.

Adapted from matmulfreellm.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import Cache

from hnet_bit.ops.fusedbitnet import FusedBitLinear as BitLinear
from hnet_bit.ops.bitnet import RMSNorm
from hnet_bit.ops.activations import swiglu
from hnet_bit.ops.fused_norm_gate import FusedRMSNormSwishGate
from hnet_bit.ops.hgrn import fused_recurrent_hgrn

try:
    from hnet_bit.ops.short_conv import ShortConvolution
except ImportError:
    ShortConvolution = None


class HGRNBitAttention(nn.Module):
    """
    HGRN (Hierarchical Gated Recurrent Network) Attention with BitLinear.
    
    Implements a gated recurrent mechanism:
    - i = swiglu(i_proj(x), 1 - f)  # input with complement gate
    - f = sigmoid(f_proj(x))        # forget gate
    - h = f * h_prev + i            # state update
    - o = g_norm(g_proj(x), h)      # output with gate normalization
    
    Uses ternary-quantized BitLinear for all projections.
    
    Args:
        mode: Recurrence mode ('fused_recurrent' supported)
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads (default 1)
        expand_ratio: State expansion ratio (default 1)
        use_short_conv: Whether to use short convolution for local inductive bias
        conv_size: Convolution kernel size (default 4)
        conv_bias: Whether conv has bias
        share_conv_kernel: Whether to share conv kernel (True = one conv, False = three separate)
        layernorm_eps: Epsilon for layer normalization
        layer_idx: Index of this layer in the model
    """

    def __init__(
        self,
        mode: str = 'fused_recurrent',
        hidden_size: int = 1024,
        num_heads: Optional[int] = 1,
        expand_ratio: Optional[int] = 1,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        share_conv_kernel: bool = True,
        layernorm_eps: float = 1e-5,
        layer_idx: int = None
    ) -> None:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.input_dim = int(hidden_size * expand_ratio)
        self.head_dim = self.input_dim // self.num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.share_conv_kernel = share_conv_kernel
        self.layer_idx = layer_idx

        assert mode in ['fused_recurrent'], f"Not supported mode `{mode}`."
        assert self.input_dim % num_heads == 0, f"input_dim ({self.input_dim}) must be divisible by num_heads ({num_heads})"

        # Projections with ternary quantization
        self.i_proj = BitLinear(hidden_size, self.input_dim, bias=False)
        self.f_proj = BitLinear(hidden_size, self.input_dim, bias=False)
        self.g_proj = BitLinear(hidden_size, self.input_dim, bias=False)

        # Short convolution for local inductive bias
        if use_short_conv:
            if ShortConvolution is None:
                raise ImportError(
                    "ShortConvolution could not be imported from hnet_bit.ops.short_conv. "
                    "Check that the module exists, or set use_short_conv=False."
                )
            if share_conv_kernel:
                self.h_conv1d = ShortConvolution(hidden_size, conv_size, activation='silu')
            else:
                self.q_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')
                self.f_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')
                self.i_conv1d = ShortConvolution(self.input_dim, conv_size, activation='silu')

        # Output normalization with gating
        self.g_norm = FusedRMSNormSwishGate(self.input_dim, layernorm_eps)
        self.o_proj = BitLinear(self.input_dim, hidden_size, bias=False)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, (nn.Linear, BitLinear)):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """
        Forward pass of HGRN attention.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            past_key_values: Optional cache for generation
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights (not supported)
            lower_bound: Optional lower bound for forget gate (disabled)
            
        Returns:
            Tuple of (output, None, updated_cache)
        """
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        last_state = past_key_values[self.layer_idx] if use_cache and past_key_values is not None else None
        
        # Compute projections (with optional short convolution)
        if self.use_short_conv:
            conv_state = last_state[0] if last_state is not None else None
            if self.share_conv_kernel:
                # Shared conv: convolve hidden_states first, then project
                hidden_states = self.h_conv1d(hidden_states, conv_state)
                i = self.i_proj(hidden_states)
                f = self.f_proj(hidden_states)
            else:
                conv_state_i = last_state[2] if last_state is not None else None
                conv_state_f = last_state[1] if last_state is not None else None
                # Separate conv: project first, then convolve each stream
                i = self.i_conv1d(self.i_proj(hidden_states), conv_state_i)
                f = self.f_conv1d(self.f_proj(hidden_states), conv_state_f)
        else:
            i = self.i_proj(hidden_states)
            f = self.f_proj(hidden_states)

        # Forget gate with sigmoid
        f = f.sigmoid()
        
        # Input with complement gating: swiglu(i, 1 - f)
        i = swiglu(i, 1 - f)
        
        # Apply attention mask if provided (for left-padding)
        if attention_mask is not None:
            i = i.mul_(attention_mask.unsqueeze(-1))
        
        # Reshape for multi-head processing: [b, l, (h*d)] -> [b, h, l, d]
        i, f = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (i, f))

        # Get recurrent state from cache
        recurrent_state = last_state[-1] if last_state is not None else None
        
        # Run fused recurrent HGRN
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_hgrn(
                i, f, 
                initial_state=recurrent_state, 
                output_final_state=use_cache
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # Update cache
        if past_key_values is not None and use_cache:
            if self.use_short_conv:
                if self.share_conv_kernel:
                    last_state = (conv_state, recurrent_state)
                else:
                    last_state = (conv_state_i, conv_state_f, recurrent_state)
            else:
                last_state = (recurrent_state,)
            past_key_values.update(last_state, self.layer_idx, i.shape[2])

        # Apply gate normalization and output projection
        o = self.g_norm(self.g_proj(hidden_states), rearrange(o, 'b h l d -> b l (h d)'))
        o = self.o_proj(o)

        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        """Initialize recurrent state (and optional conv state) for generation."""
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            if self.share_conv_kernel:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
            else:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),
                          param.new_zeros(batch_size, self.hidden_size, self.conv_size),
                          param.new_zeros(batch_size, self.hidden_size, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        """Return the size of the recurrent state (+ conv states if enabled)."""
        state_size = self.hidden_size
        if self.use_short_conv:
            for module in self.children():
                if ShortConvolution is not None and isinstance(module, ShortConvolution):
                    state_size += module.state_size
        return state_size


class HGRNBitMLP(nn.Module):
    """
    Feed-forward MLP with SwiGLU activation and BitLinear layers.
    
    Architecture: gate_proj -> SwiGLU -> down_proj
    
    The intermediate size is computed as:
    - 2/3 * hidden_size * hidden_ratio, rounded to multiple of 256
    
    Args:
        hidden_size: Model hidden dimension
        hidden_ratio: Ratio for intermediate size computation
        intermediate_size: Override for intermediate size
        hidden_act: Activation function name (unused, always SwiGLU)
    """

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        
        # Compute intermediate size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        # Projections with ternary quantization
        # gate_proj outputs 2x for SwiGLU split
        self.gate_proj = BitLinear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = BitLinear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLP.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        z = self.down_proj(swiglu(gate, y))
        return z


class HGRNBitBlock(nn.Module):
    """
    Transformer-like block with HGRN attention and MLP.
    
    Architecture (prenorm residual stream, matching original matmulfreellm):
        residual = x
        x = attn_norm(x)
        x = attn(x)
        x, residual = mlp_norm(x, residual, prenorm=True)  # fused add + norm
        x = mlp(x) + residual
    
    Args:
        config: Model configuration object
        layer_idx: Index of this layer in the model
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Attention with pre-normalization
        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.attn = HGRNBitAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_ratio=config.expand_ratio,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            share_conv_kernel=config.share_conv_kernel,
            layernorm_eps=config.rms_norm_eps,
            layer_idx=layer_idx
        )
        
        # MLP with pre-normalization
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HGRNBitMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        """
        Forward pass of the block.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            past_key_values: Optional cache
            use_cache: Whether to use cache
            output_attentions: Whether to output attention weights
            lower_bound: Optional forget gate lower bound (disabled)
            
        Returns:
            Tuple of (hidden_states, attentions, cache)
        """
        # Attention with pre-normalization
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            lower_bound=lower_bound
        )
        # MLP with fused residual prenorm (matches original matmulfreellm):
        # mlp_norm adds attn output to residual, normalizes, returns (normed, sum)
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)
        return outputs
