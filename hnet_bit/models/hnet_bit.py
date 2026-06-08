# -*- coding: utf-8 -*-

"""
HNetBit: Hierarchical H-Net with ternary BitLinear weights.

Combines H-Net's multi-stage dynamic chunking architecture with MatmulFree's
ternary weight quantization. Replaces H-Net's Isotropic (Mamba2/Attention)
blocks with HGRNBitBlocks from hnet_bit.

Architecture (2-stage example):
    Input bytes → Embedding(256, d_model[0])
    Stage 0:
        Encoder: N × HGRNBitBlock(d_model[0])
        RoutingModuleBit → ChunkLayer (adaptive downsampling)
        Stage 1 (main_network):
            Encoder: M × HGRNBitBlock(d_model[1])
            RoutingModuleBit → ChunkLayer
            Stage 2 (innermost):
                K × HGRNBitBlock(d_model[2])
            DeChunkLayer
            Decoder: M × HGRNBitBlock(d_model[1])
        DeChunkLayer
        Decoder: N × HGRNBitBlock(d_model[0])
    LM Head → 256 (byte logits)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from hnet_bit.layers.hgrn_bit import HGRNBitBlock
from hnet_bit.layers.attention import CausalMHABlock
from hnet_bit.ops.fusedbitnet import FusedBitLinear as BitLinear
from hnet_bit.ops.bitnet import RMSNorm
from hnet_bit.ops.dynamic_chunking import (
    RoutingModuleBit,
    RoutingModuleOutput,
    ChunkLayer,
    DeChunkLayer,
)
from hnet_bit.utils.hnet_cache import HGRNBlockCache, HNetBitCache

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom output with router_outputs
# ---------------------------------------------------------------------------
@dataclass
class HNetBitCausalLMOutput(CausalLMOutputWithPast):
    """
    Extends CausalLMOutputWithPast with a dedicated field for routing outputs
    from the hierarchical boundary prediction modules.
    
    Attributes:
        router_outputs: List of RoutingModuleOutput from each hierarchy stage,
                        used for computing load-balancing loss during training.
    """
    router_outputs: Optional[List[RoutingModuleOutput]] = None


# ---------------------------------------------------------------------------
# STE for residual gating (from H-Net)
# ---------------------------------------------------------------------------
class _STE(torch.autograd.Function):
    """Straight-through estimator that returns ones in forward, passes gradient in backward."""
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def _ste_func(x):
    return _STE.apply(x)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class HNetBitConfig(PretrainedConfig):
    """
    Configuration for Hierarchical HNetBit model.
    
    Multi-stage architecture with dynamic chunking and MatMulFree ternary weights.
    
    Args:
        vocab_size: Vocabulary size (256 for byte-level)
        d_model: List of hidden dimensions per stage, e.g. [512, 768, 1024]
        num_blocks: List of [encoder_blocks, innermost_blocks, decoder_blocks] per stage.
                    e.g. [[4, 0, 4], [4, 0, 4], [8]] where last entry = innermost.
                    For each non-innermost stage: [n_encoder, unused, n_decoder].
                    For innermost stage: [n_blocks].
        num_heads: Number of HGRN attention heads (default 1)
        expand_ratio: HGRN state expansion ratio (default 1)
        hidden_ratio: MLP hidden size ratio (default 4)
        intermediate_size: Override MLP intermediate size (None = auto-compute)
        attn_mode: Recurrence mode ('fused_recurrent')
        max_position_embeddings: Max sequence length
        rms_norm_eps: RMSNorm epsilon
        use_cache: Whether to use cache for generation
        initializer_range: Weight initialization std
        tie_word_embeddings: Whether to tie embeddings and LM head
        use_fused_bitlinear: Use FusedBitLinear (Triton) instead of BitLinear
    """
    model_type = 'hnet_bit'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: List[int] = None,
        num_blocks: List[List[int]] = None,
        num_heads: int = 1,
        expand_ratio: int = 1,
        hidden_ratio: int = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        attn_mode: str = "fused_recurrent",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 254,
        eos_token_id: int = 255,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        use_fused_bitlinear: bool = False,
        # For compatibility with HGRNBitBlock config interface
        use_short_conv: bool = False,
        conv_size: int = 4,
        share_conv_kernel: bool = True,
        use_lower_bound: bool = False,
        # Innermost attention configuration
        innermost_use_attention: bool = False,
        attention_window_size: int = 64,
        attention_num_heads: Optional[int] = None,
        attention_layers_pattern: Optional[str] = None,
        **kwargs,
    ):
        if d_model is None:
            d_model = [512, 768]
        if num_blocks is None:
            # Default: 2-stage with 4 enc/dec blocks, 8 innermost blocks
            num_blocks = [[4, 0, 4], [8]]

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_stages = len(d_model)
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.attn_mode = attn_mode
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.use_fused_bitlinear = use_fused_bitlinear
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.share_conv_kernel = share_conv_kernel
        self.use_lower_bound = use_lower_bound
        self.innermost_use_attention = innermost_use_attention
        self.attention_window_size = attention_window_size
        self.attention_num_heads = attention_num_heads
        self.attention_layers_pattern = attention_layers_pattern

        # Total layer count for transformers compatibility (e.g. DynamicCache)
        self.num_hidden_layers = sum(
            (b[0] + b[2]) if len(b) == 3 else b[0]
            for b in num_blocks
        )
        # Also expose hidden_size as the outermost d_model for transformers
        self.hidden_size = d_model[0]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Validate configuration consistency
        if len(self.d_model) != len(self.num_blocks):
            raise ValueError(
                f"d_model has {len(self.d_model)} stages but num_blocks has "
                f"{len(self.num_blocks)} entries. They must match."
            )
        for i, blocks in enumerate(self.num_blocks):
            is_innermost = (i == len(self.d_model) - 1)
            if is_innermost and len(blocks) not in (1, 3):
                raise ValueError(
                    f"Innermost stage {i} num_blocks must have 1 or 3 elements, "
                    f"got {len(blocks)}."
                )
            if not is_innermost and len(blocks) != 3:
                raise ValueError(
                    f"Non-innermost stage {i} num_blocks must have 3 elements "
                    f"[encoder, unused, decoder], got {len(blocks)}."
                )

    def _make_stage_config(self, stage_idx: int):
        """Create a config-like object for HGRNBitBlock at a given stage."""
        class _StageConfig:
            pass
        cfg = _StageConfig()
        cfg.hidden_size = self.d_model[stage_idx]
        cfg.num_heads = self.num_heads
        cfg.expand_ratio = self.expand_ratio
        cfg.hidden_ratio = self.hidden_ratio
        cfg.intermediate_size = self.intermediate_size
        cfg.hidden_act = self.hidden_act
        cfg.attn_mode = self.attn_mode
        cfg.rms_norm_eps = self.rms_norm_eps
        cfg.use_short_conv = self.use_short_conv
        cfg.conv_size = self.conv_size
        cfg.share_conv_kernel = self.share_conv_kernel
        cfg.max_position_embeddings = self.max_position_embeddings
        cfg.attention_window_size = self.attention_window_size
        cfg.attention_num_heads = self.attention_num_heads
        return cfg

    @classmethod
    def small_1stage(cls, **kwargs):
        """Small 1-stage model (~15M params): 4 enc + 8 inner + 4 dec."""
        return cls(
            d_model=[256, 384],
            num_blocks=[[4, 0, 4], [8]],
            **kwargs,
        )

    @classmethod
    def base_2stage(cls, **kwargs):
        """Base 2-stage model (~50M params)."""
        return cls(
            d_model=[512, 768, 1024],
            num_blocks=[[4, 0, 4], [4, 0, 4], [8]],
            **kwargs,
        )

    @classmethod
    def large_2stage(cls, **kwargs):
        """Large 2-stage model (~100M params)."""
        return cls(
            d_model=[768, 1024, 1536],
            num_blocks=[[4, 0, 4], [4, 0, 4], [12]],
            **kwargs,
        )


# ---------------------------------------------------------------------------
# HGRNBitStack: stack of HGRNBitBlocks at one dimension
# ---------------------------------------------------------------------------
class HGRNBitStack(nn.Module):
    """
    A stack of HGRNBitBlock (and optionally CausalMHABlock) layers.
    
    When is_innermost=True and config.innermost_use_attention=True, interleaves
    HGRN and attention blocks according to config.attention_layers_pattern.
    Pattern is a string of 'x' (HGRN) and 'a' (attention), e.g. "xaxaxaxa".
    
    Args:
        config: HNetBitConfig
        stage_idx: Which hierarchy stage this stack belongs to
        num_layers: Number of blocks in this stack
        layer_idx_offset: Starting layer_idx for cache indexing
        is_innermost: Whether this is the innermost hierarchy stage
    """

    def __init__(
        self,
        config: HNetBitConfig,
        stage_idx: int,
        num_layers: int,
        layer_idx_offset: int = 0,
        is_innermost: bool = False,
    ):
        super().__init__()
        stage_cfg = config._make_stage_config(stage_idx)

        # Determine per-layer block types
        use_attention = is_innermost and getattr(config, 'innermost_use_attention', False)
        if use_attention:
            pattern = getattr(config, 'attention_layers_pattern', None)
            if pattern is None:
                # Default: alternate starting with HGRN
                pattern = ''.join('xa'[i % 2] for i in range(num_layers))
            if len(pattern) < num_layers:
                # Extend pattern cyclically
                pattern = (pattern * ((num_layers // len(pattern)) + 1))[:num_layers]
        else:
            pattern = 'x' * num_layers

        layers = []
        for i in range(num_layers):
            lid = layer_idx_offset + i
            if pattern[i] == 'a':
                layers.append(CausalMHABlock(stage_cfg, layer_idx=lid))
            else:
                layers.append(HGRNBitBlock(stage_cfg, layer_idx=lid))

        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(config.d_model[stage_idx], eps=config.rms_norm_eps)
        self.num_layers = num_layers
        self.d_model = config.d_model[stage_idx]

    @property
    def height(self) -> int:
        """Number of residual additions (2 per block: attn + mlp)."""
        return self.num_layers * 2

    def init_cache(self, batch_size: int) -> HGRNBlockCache:
        """Initialize HGRNBlockCache for generation."""
        cache = HGRNBlockCache()
        for layer in self.layers:
            state = layer.attn.init_state(batch_size)
            cache.states.append(state)
        return cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[HGRNBlockCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[HGRNBlockCache]]:
        """
        Forward through all blocks.
        
        Args:
            hidden_states: (B, L, D)
            mask: (B, L) attention mask
            cache: Optional HGRNBlockCache
            use_cache: Whether to update cache
            
        Returns:
            (hidden_states, updated_cache)
        """
        # Build a lightweight cache adapter for HGRNBitBlock's interface
        _cache_wrapper = _HGRNBlockCacheWrapper(cache) if cache is not None else None

        for i, layer in enumerate(self.layers):
            hidden_states, _, _cache_wrapper = layer(
                hidden_states,
                attention_mask=mask,
                past_key_values=_cache_wrapper,
                use_cache=use_cache,
            )

        hidden_states = self.norm(hidden_states)

        out_cache = _cache_wrapper.inner if _cache_wrapper is not None else None
        return hidden_states, out_cache

    def step(
        self,
        hidden_states: torch.Tensor,
        cache: HGRNBlockCache,
    ) -> Tuple[torch.Tensor, HGRNBlockCache]:
        """
        Step-by-step forward for generation. hidden_states is (B, 1, D).
        """
        _cache_wrapper = _HGRNBlockCacheWrapper(cache)

        for i, layer in enumerate(self.layers):
            hidden_states, _, _cache_wrapper = layer(
                hidden_states,
                past_key_values=_cache_wrapper,
                use_cache=True,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, _cache_wrapper.inner


class _HGRNBlockCacheWrapper:
    """
    Adapter making HGRNBlockCache compatible with the Cache-like interface
    expected by HGRNBitBlock (which calls past_key_values[layer_idx] and .update()).
    """

    def __init__(self, inner: HGRNBlockCache):
        self.inner = inner

    def __getitem__(self, layer_idx: int):
        return self.inner[layer_idx]

    def update(self, state, layer_idx: int, seq_len=1):
        self.inner.update(state, layer_idx, seq_len)
        return state


# ---------------------------------------------------------------------------
# HNetBit: recursive hierarchical module
# ---------------------------------------------------------------------------
class HNetBit(nn.Module):
    """
    Recursive hierarchical module with ternary weights.
    
    Non-innermost stage:
        encoder → routing → chunk → main_network → dechunk → residual → decoder
    
    Innermost stage:
        main_network (just a stack of HGRNBitBlocks)
    
    Args:
        config: HNetBitConfig
        stage_idx: Current stage in the hierarchy (0 = outermost)
    """

    def __init__(self, config: HNetBitConfig, stage_idx: int = 0):
        super().__init__()
        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]
        self.config = config

        num_stages = len(config.d_model)
        is_innermost = (stage_idx == num_stages - 1)
        self.is_innermost = is_innermost

        if is_innermost:
            # Innermost: stack of HGRN blocks (optionally interleaved with attention)
            n_blocks = config.num_blocks[stage_idx][0]
            self.main_network = HGRNBitStack(
                config, stage_idx, n_blocks, layer_idx_offset=0,
                is_innermost=True,
            )
        else:
            # Non-innermost: encoder + routing + chunk + main + dechunk + decoder
            enc_blocks = config.num_blocks[stage_idx][0]
            dec_blocks = config.num_blocks[stage_idx][2]

            self.encoder = HGRNBitStack(
                config, stage_idx, enc_blocks, layer_idx_offset=0,
            )
            self.decoder = HGRNBitStack(
                config, stage_idx, dec_blocks, layer_idx_offset=enc_blocks,
            )

            # Dynamic chunking
            self.routing_module = RoutingModuleBit(self.d_model)
            self.chunk_layer = ChunkLayer()
            self.dechunk_layer = DeChunkLayer(self.d_model)

            # Residual projection (FP32 for precision, following H-Net)
            self.residual_proj = nn.Linear(
                self.d_model, self.d_model, dtype=torch.float32,
            )
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True

            self.residual_func = lambda out, residual, p: \
                out * _ste_func(p) + residual

            # Recursive main_network at next stage
            self.main_network = HNetBit(config, stage_idx + 1)

        # Dimension padding if this stage has a larger d_model than its parent
        if stage_idx > 0 and config.d_model[stage_idx] - config.d_model[stage_idx - 1] > 0:
            self.pad_dimension = nn.Parameter(
                torch.zeros(config.d_model[stage_idx] - config.d_model[stage_idx - 1])
            )
        else:
            self.pad_dimension = None

    def _apply_lr_multiplier(self, lr_multipliers: List[float]) -> None:
        """
        Apply stage-specific learning rate multipliers to parameters.
        
        Follows H-Net's pattern of using param._optim attribute to store
        per-parameter optimization settings that are later read by the
        optimizer builder.
        
        Args:
            lr_multipliers: List of LR multipliers per stage [stage_0, stage_1, ...]
                           e.g., [2.0, 1.5, 1.0] means stage 0 gets 2x base LR,
                           stage 1 gets 1.5x, stage 2 gets 1x
        
        Example:
            >>> model = HNetBit(config)
            >>> model._apply_lr_multiplier([2.0, 1.5, 1.0])
            >>> # Now outer stage params have _optim={'lr_multiplier': 2.0}
        """
        from hnet_bit.utils.helpers import apply_optimization_params
        
        # Apply LR multiplier to all parameters at this stage
        for param in self.parameters():
            apply_optimization_params(param, lr_multiplier=lr_multipliers[self.stage_idx])
        
        # Recursively apply to nested stages (if hierarchical)
        if not self.is_innermost and hasattr(self.main_network, '_apply_lr_multiplier'):
            self.main_network._apply_lr_multiplier(lr_multipliers)

    def _count_residuals(self) -> int:
        """Count total residual additions for weight initialization scaling."""
        if self.is_innermost:
            return self.main_network.height
        else:
            n = self.encoder.height + self.decoder.height
            if isinstance(self.main_network, HNetBit):
                n += self.main_network._count_residuals()
            else:
                n += self.main_network.height
            return n

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=None,
    ) -> HNetBitCache:
        """Allocate nested cache for generation."""
        device = next(self.parameters()).device

        if self.is_innermost:
            return HNetBitCache(
                main_network_cache=self.main_network.init_cache(batch_size),
                is_innermost=True,
            )
        else:
            return HNetBitCache(
                encoder_cache=self.encoder.init_cache(batch_size),
                routing_state=self.routing_module.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype,
                ),
                main_network_cache=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype,
                ),
                dechunk_state=self.dechunk_layer.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype,
                ),
                decoder_cache=self.decoder.init_cache(batch_size),
                is_innermost=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inference_params: Optional[HNetBitCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, List[RoutingModuleOutput]]:
        """
        Forward pass through this hierarchy stage.
        
        Args:
            hidden_states: (B, L, D_parent) or (B, L, D_self)
            mask: (B, L) boolean mask, True = valid
            inference_params: Optional HNetBitCache for generation
            use_cache: Whether to update cache
            
        Returns:
            (output_hidden_states, boundary_predictions_list)
        """
        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]

        # Pad to this stage's d_model if needed
        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_dimension.expand(EARLY_DIMS + (-1,))),
                dim=-1,
            )

        if self.is_innermost:
            enc_cache = inference_params.main_network_cache if inference_params is not None else None
            hidden_states, updated_cache = self.main_network(
                hidden_states, mask=mask, cache=enc_cache, use_cache=use_cache,
            )
            if inference_params is not None:
                inference_params.main_network_cache = updated_cache
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        # --- Non-innermost stage ---
        enc_cache = inference_params.encoder_cache if inference_params is not None else None
        hidden_states, enc_cache_out = self.encoder(
            hidden_states, mask=mask, cache=enc_cache, use_cache=use_cache,
        )
        if inference_params is not None:
            inference_params.encoder_cache = enc_cache_out

        # Residual in FP32
        hidden_states_fp32 = hidden_states.to(dtype=self.residual_proj.weight.dtype)
        residual = self.residual_proj(hidden_states_fp32)

        # Routing: predict chunk boundaries
        routing_state = inference_params.routing_state if inference_params is not None else None
        bpred_output = self.routing_module(
            hidden_states, mask=mask, inference_params=routing_state,
        )

        # Chunk: select boundary tokens
        next_hidden_states, next_cu_seqlens, next_mask = self.chunk_layer(
            hidden_states, bpred_output.boundary_mask, mask=mask,
        )

        # Recursive: process chunks at next stage
        main_cache = inference_params.main_network_cache if inference_params is not None else None
        next_hidden_states, prev_boundary_predictions = self.main_network(
            next_hidden_states, mask=next_mask,
            inference_params=main_cache, use_cache=use_cache,
        )

        # DeChunk: reconstruct full sequence from chunks
        dechunk_state = inference_params.dechunk_state if inference_params is not None else None
        hidden_states = self.dechunk_layer(
            next_hidden_states,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            mask=mask,
            inference_params=dechunk_state,
        )

        # Ensure shapes match for residual
        if hidden_states.shape != residual.shape:
            # This shouldn't happen if dechunk is correct, but add safety check
            raise RuntimeError(
                f"Shape mismatch in residual: hidden_states={hidden_states.shape}, "
                f"residual={residual.shape}"
            )

        # Residual connection with STE
        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype),
            residual,
            bpred_output.selected_probs,
        ).to(next_hidden_states.dtype if next_hidden_states.shape[0] > 0 else hidden_states.dtype)

        # Decoder
        dec_cache = inference_params.decoder_cache if inference_params is not None else None
        hidden_states, dec_cache_out = self.decoder(
            hidden_states, mask=mask, cache=dec_cache, use_cache=use_cache,
        )
        if inference_params is not None:
            inference_params.decoder_cache = dec_cache_out

        hidden_states = hidden_states[..., :D]
        return hidden_states, [bpred_output, *prev_boundary_predictions]

    def step(
        self,
        hidden_states: torch.Tensor,
        inference_params: HNetBitCache,
    ) -> Tuple[torch.Tensor, List[RoutingModuleOutput]]:
        """
        Step-by-step forward for generation. hidden_states is (B, 1, D).
        """
        D = hidden_states.shape[-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_dimension.expand(hidden_states.shape[:-1] + (-1,))),
                dim=-1,
            )

        if self.is_innermost:
            hidden_states, cache = self.main_network.step(
                hidden_states, inference_params.main_network_cache,
            )
            inference_params.main_network_cache = cache
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        # Encoder step
        hidden_states, enc_cache = self.encoder.step(
            hidden_states, inference_params.encoder_cache,
        )
        inference_params.encoder_cache = enc_cache

        # Residual
        hidden_states_fp32 = hidden_states.to(dtype=self.residual_proj.weight.dtype)
        residual = self.residual_proj(hidden_states_fp32)

        # Routing step
        bpred_output = self.routing_module.step(
            hidden_states, inference_params.routing_state,
        )

        # Chunk step: select tokens at boundaries
        hidden_states_inner = self.chunk_layer.step(
            hidden_states, bpred_output.boundary_mask,
        )

        # Process selected tokens through inner stage
        if hidden_states_inner.shape[0] > 0:
            hidden_states_inner, prev_boundary_predictions = self.main_network.step(
                hidden_states_inner, inference_params.main_network_cache,
            )
        else:
            prev_boundary_predictions = []

        # Dechunk step
        hidden_states = self.dechunk_layer.step(
            hidden_states_inner,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            inference_params.dechunk_state,
        )

        # Residual
        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype),
            residual,
            bpred_output.selected_probs,
        ).to(hidden_states_inner.dtype if hidden_states_inner.shape[0] > 0 else hidden_states.dtype)

        # Decoder step
        hidden_states, dec_cache = self.decoder.step(
            hidden_states, inference_params.decoder_cache,
        )
        inference_params.decoder_cache = dec_cache

        hidden_states = hidden_states[..., :D]
        return hidden_states, [bpred_output, *prev_boundary_predictions]


# ---------------------------------------------------------------------------
# HNetBitPreTrainedModel
# ---------------------------------------------------------------------------
class HNetBitPreTrainedModel(PreTrainedModel):
    """Base class for HNetBit models."""

    config_class = HNetBitConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['HGRNBitBlock', 'HGRNBitStack', 'HNetBit']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with H-Net-style scaling and prenorm residual rescaling."""
        if isinstance(module, (nn.Linear, BitLinear)):
            if not getattr(module.weight, '_no_reinit', False):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        # Prenorm residual rescaling (GPT-2 paper, matching matmulfreellm):
        # Scale o_proj and down_proj weights by 1/sqrt(2 * num_hidden_layers)
        for name, p in module.named_parameters():
            if name in ["o_proj.weight", "down_proj.weight"]:
                with torch.no_grad():
                    p /= math.sqrt(2 * self.config.num_hidden_layers)


# ---------------------------------------------------------------------------
# HNetBitForCausalLM
# ---------------------------------------------------------------------------
class HNetBitForCausalLM(HNetBitPreTrainedModel, GenerationMixin):
    """
    Hierarchical HNetBit model with Language Modeling head.
    
    Architecture:
        Embedding(256, d_model[0]) → HNetBit(hierarchical) → BitLinear LM Head(d_model[0], 256)
    
    All linear layers use ternary quantization ({-1, 0, +1}).
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: HNetBitConfig):
        super().__init__(config)
        self.config = config
        d_embed = config.d_model[0]

        self.embeddings = nn.Embedding(config.vocab_size, d_embed)
        self.backbone = HNetBit(config, stage_idx=0)
        self.lm_head = BitLinear(d_embed, config.vocab_size, bias=False)

        self.post_init()

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate hierarchical cache for generation."""
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        """Override to use HNetBitCache instead of DynamicCache."""
        if model_kwargs.get('past_key_values') is None:
            # Allocate our hierarchical cache
            batch_size = model_kwargs['input_ids'].shape[0] if 'input_ids' in model_kwargs else 1
            model_kwargs['past_key_values'] = self.allocate_inference_cache(
                batch_size,
                self.config.max_position_embeddings,
                dtype=model_kwargs['input_ids'].dtype if 'input_ids' in model_kwargs else None,
            )

    def generate(self, *args, **kwargs):
        """Generate with error handling for unsupported strategies."""
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with an unsupported decoding strategy for {self.__class__.__name__}."
                )
            raise

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Prepare inputs for generation step."""
        # If we have past_key_values (HNetBitCache), only use last token
        if past_key_values is not None and isinstance(past_key_values, HNetBitCache):
            if past_key_values.get_seq_length() > 0:
                input_ids = input_ids[:, -1:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids.contiguous()}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        })
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Byte token IDs [batch, seq_len]
            attention_mask: Boolean mask [batch, seq_len], True = valid
            inputs_embeds: Optional pre-computed embeddings
            past_key_values: HNetBitCache for generation
            labels: Target labels for loss computation
            use_cache: Whether to return updated cache
            
        Returns:
            CausalLMOutputWithPast with logits, loss, and cache
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        # Embed
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must specify either input_ids or inputs_embeds")
            hidden_states = self.embeddings(input_ids)
        else:
            hidden_states = inputs_embeds

        B, L, D = hidden_states.shape

        # Construct mask (True = valid token)
        if attention_mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=hidden_states.device)
        else:
            mask = attention_mask.bool()

        # Initialize cache for generation
        inference_params = past_key_values
        if use_cache:
            if inference_params is None:
                inference_params = self.backbone.allocate_inference_cache(
                    B, L, dtype=hidden_states.dtype,
                )
            elif not isinstance(inference_params, HNetBitCache):
                # Handle transformers DynamicCache or other cache types
                inference_params = self.backbone.allocate_inference_cache(
                    B, L, dtype=hidden_states.dtype,
                )

        # Forward through hierarchical backbone
        if use_cache and inference_params is not None and inference_params.get_seq_length() > 0:
            # Step mode (single token generation)
            hidden_states, bpred_outputs = self.backbone.step(
                hidden_states, inference_params,
            )
        else:
            # Full forward (training or prefill)
            hidden_states, bpred_outputs = self.backbone(
                hidden_states, mask=mask,
                inference_params=inference_params,
                use_cache=use_cache,
            )

        # LM head
        logits = self.lm_head(hidden_states)

        # Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            labels = labels.to(logits.device)
            # Shift for next-token prediction
            labels = torch.cat(
                (labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)),
                dim=1,
            )
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits, inference_params)
            return (loss,) + output if loss is not None else output

        return HNetBitCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=inference_params,
            router_outputs=bpred_outputs if output_hidden_states else None,
        )

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_ternary_weight_stats(self) -> dict:
        """Get statistics about ternary weight distribution across the model."""
        stats = {'total_params': 0, 'ternary_params': 0, 'distribution': {-1: 0, 0: 0, 1: 0}}

        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                scale = 1.0 / param.data.abs().mean().clamp_(min=1e-5)
                w_ternary = (param.data * scale).round().clamp_(-1, 1)

                stats['total_params'] += param.numel()
                stats['ternary_params'] += param.numel()
                stats['distribution'][-1] += (w_ternary == -1).sum().item()
                stats['distribution'][0] += (w_ternary == 0).sum().item()
                stats['distribution'][1] += (w_ternary == 1).sum().item()

        total = sum(stats['distribution'].values())
        if total > 0:
            stats['distribution_pct'] = {
                k: v / total * 100 for k, v in stats['distribution'].items()
            }
        return stats
