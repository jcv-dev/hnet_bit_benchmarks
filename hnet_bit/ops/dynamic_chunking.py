# -*- coding: utf-8 -*-

"""
Dynamic Chunking module for hierarchical sequence processing.

Adapted from H-Net's dc.py.  Routing uses full-precision nn.Linear projections
(matching the original) since cosine-similarity boundary prediction is sensitive
to quantization noise.  DeChunkLayer EMA uses the Triton-accelerated
fused_recurrent_hgrn kernel (with CPU fallback) instead of a sequential loop.

Components:
- RoutingModuleBit: Predicts chunk boundaries via cosine similarity (nn.Linear projections)
- ChunkLayer: Groups tokens into variable-length chunks based on boundaries
- DeChunkLayer: Reconstructs full sequence from chunk representations using parallel EMA
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet_bit.ops.hgrn import fused_recurrent_hgrn


@dataclass
class RoutingModuleOutput:
    """Output of the routing module.
    
    Attributes:
        boundary_prob: Probability of boundary at each position, shape [..., 2]
                       (prob_no_boundary, prob_boundary)
        boundary_mask: Binary mask of selected boundaries, shape [...]
        selected_probs: Probability of the selected action, shape [..., 1]
    """
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


@dataclass
class RoutingModuleState:
    """Inference state for the routing module.
    
    Attributes:
        has_seen_tokens: Whether each batch element has processed any tokens yet
        last_hidden_state: Last hidden state per batch element for boundary prediction
    """
    has_seen_tokens: torch.Tensor   # (batch_size,)
    last_hidden_state: torch.Tensor  # (batch_size, d_model)


@dataclass
class DeChunkState:
    """Inference state for the dechunk layer.
    
    Attributes:
        last_value: Last output value per batch element for EMA
    """
    last_value: torch.Tensor  # (batch_size, d_model)


class RoutingModuleBit(nn.Module):
    """
    Predicts chunk boundaries using cosine similarity with full-precision projections.
    
    Computes boundary probability between consecutive tokens:
        boundary_prob = (1 - cos_sim(q_proj(h_t), k_proj(h_{t+1}))) / 2
    
    High boundary probability indicates semantic dissimilarity → chunk boundary.
    Uses nn.Linear (not BitLinear) to avoid quantization noise in boundary
    decisions, matching the original H-Net implementation.
    
    Args:
        d_model: Hidden dimension
        device: Device for parameter allocation
        dtype: Dtype for parameter allocation
    """

    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Full-precision projections for cosine similarity (matching original H-Net dc.py)
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

        # Initialize to near-identity for stable training start
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, device, dtype=None
    ) -> RoutingModuleState:
        """Allocate inference state for step-by-step generation."""
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        inference_params: Optional[RoutingModuleState] = None,
    ) -> RoutingModuleOutput:
        """
        Compute chunk boundaries for a sequence.
        
        Args:
            hidden_states: Input tensor (B, L, D) or packed (T, D) if cu_seqlens given
            mask: Valid token mask (B, L), True for valid positions
            cu_seqlens: Cumulative sequence lengths for packed mode (N+1,)
            inference_params: Optional state for prefill phase
            
        Returns:
            RoutingModuleOutput with boundary predictions
        """
        assert (mask is not None) or (cu_seqlens is not None), \
            "Either mask or cu_seqlens must be provided"

        if inference_params is not None:
            assert mask is not None, "Mask must be provided if inference_params is provided"
            assert (~inference_params.has_seen_tokens).all(), \
                "Cannot have seen tokens when running forward (use step for incremental)"

        packed_mode = cu_seqlens is not None
        if packed_mode:
            # Packed mode: hidden_states is (T, D), make it (1, T, D)
            hidden_states = hidden_states.unsqueeze(0)

        # Cosine similarity between consecutive tokens (computed in FP32 for stability)
        q = F.normalize(self.q_proj_layer(hidden_states[:, :-1]).float(), dim=-1)
        k = F.normalize(self.k_proj_layer(hidden_states[:, 1:]).float(), dim=-1)
        cos_sim = torch.einsum("b l d, b l d -> b l", q, k)
        
        # Boundary probability: high when consecutive tokens are dissimilar
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

        # Force first token to always be a boundary
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        if packed_mode:
            # Force boundary at start of each packed sequence
            boundary_prob = boundary_prob.squeeze(0)  # (T,)
            boundary_prob[cu_seqlens[:-1]] = PAD_PROB

        # Stack as [prob_no_boundary, prob_boundary]
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        # Select boundary or not (argmax → hard decision)
        selected_idx = torch.argmax(boundary_prob, dim=-1)
        boundary_mask = selected_idx == 1  # True where boundary

        # Mask out invalid positions
        if mask is not None:
            boundary_mask = boundary_mask & mask

        # Update inference state during prefill
        if inference_params is not None:
            has_mask = mask.any(dim=-1)
            inference_params.has_seen_tokens.copy_(
                has_mask | inference_params.has_seen_tokens
            )
            last_mask = torch.clamp(mask.sum(dim=-1) - 1, min=0)
            inference_params.last_hidden_state.copy_(
                torch.where(
                    has_mask.unsqueeze(-1).expand_as(inference_params.last_hidden_state),
                    hidden_states[
                        torch.arange(hidden_states.shape[0], device=hidden_states.device),
                        last_mask,
                    ],
                    inference_params.last_hidden_state,
                )
            )

        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )

    def step(
        self,
        hidden_states: torch.Tensor,
        inference_params: RoutingModuleState,
    ) -> RoutingModuleOutput:
        """
        Step-by-step boundary prediction for generation.
        
        Args:
            hidden_states: Current token hidden state (B, 1, D)
            inference_params: Current routing state
            
        Returns:
            RoutingModuleOutput for current step
        """
        hidden_states_squeezed = hidden_states.squeeze(1)  # (B, D)
        
        # Cosine similarity with last token (FP32 for stability)
        q = F.normalize(self.q_proj_layer(inference_params.last_hidden_state).float(), dim=-1)
        k = F.normalize(self.k_proj_layer(hidden_states_squeezed).float(), dim=-1)
        cos_sim = torch.einsum("b d, b d -> b", q, k)
        
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        
        # Update state
        inference_params.last_hidden_state.copy_(hidden_states_squeezed)

        # First token is always a boundary
        boundary_prob = torch.where(
            inference_params.has_seen_tokens,
            boundary_prob,
            torch.ones_like(boundary_prob),
        )
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        inference_params.has_seen_tokens.copy_(
            torch.ones_like(inference_params.has_seen_tokens)
        )

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_prob[..., 1] > 0.5,
            selected_probs=boundary_prob.max(dim=-1).values.unsqueeze(-1),
        )


class ChunkLayer(nn.Module):
    """
    Groups tokens into variable-length chunks based on boundary mask.
    
    No learnable parameters — pure selection/gathering operation.
    Works in unpacked mode (B, L, D) with mask.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Select boundary tokens to form chunks.
        
        Args:
            hidden_states: Input tensor (B, L, D) or packed (T, D)
            boundary_mask: Boolean boundary mask (B, L) or (T,)
            cu_seqlens: Cumulative sequence lengths for packed mode (N+1,)
            mask: Valid token mask (B, L)
            
        Returns:
            Tuple of (next_hidden_states, next_cu_seqlens, next_mask):
                In packed mode: next_hidden_states (M, D), next_cu_seqlens (N+1,), next_mask=None
                In padded mode: next_hidden_states (B, M, D), next_cu_seqlens=None, next_mask (B, M)
        """
        assert (mask is not None) or (cu_seqlens is not None), \
            "Either mask or cu_seqlens must be provided"

        if cu_seqlens is not None:
            # Packed mode: hidden_states is (T, D), boundary_mask is (T,)
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = F.pad(
                boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0)
            )
            return next_hidden_states, next_cu_seqlens, None
        
        # Padded mode: hidden_states is (B, L, D)
        num_tokens = boundary_mask.sum(dim=-1)  # (B,)
        next_max_seqlen = int(num_tokens.max())

        device = hidden_states.device
        L = hidden_states.shape[1]
        
        # Sort: boundary tokens first, then non-boundary
        token_idx = (
            torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)

        # Gather the first next_max_seqlen tokens (the boundary tokens)
        next_hidden_states = torch.gather(
            hidden_states,
            dim=1,
            index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                -1, -1, hidden_states.shape[-1]
            ),
        )

        next_mask = (
            torch.arange(next_max_seqlen, device=device)[None, :]
            < num_tokens[:, None]
        )

        return next_hidden_states, None, next_mask

    def step(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step-by-step chunk selection for generation.
        
        Args:
            hidden_states: Current hidden states (B, 1, D)
            boundary_mask: Boundary decisions (B,)
            
        Returns:
            Selected hidden states (B', 1, D) where B' = sum(boundary_mask)
        """
        return hidden_states[boundary_mask]


class DeChunkLayer(nn.Module):
    """
    Reconstructs full sequence from chunk representations using EMA.
    
    Uses exponential moving average to propagate chunk information to all
    positions between boundaries. The EMA is computed via fused_recurrent_hgrn
    (Triton-accelerated parallel scan on GPU, sequential fallback on CPU),
    which maps the EMA recurrence to an equivalent HGRN recurrence:
        EMA:  out_t = p_t * x_t + (1 - p_t) * out_{t-1}
        HGRN: h_t   = g_t * h_{t-1} + x_t   with g = (1-p), x = p * hidden
    
    Args:
        d_model: Hidden dimension
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, device, dtype=None,
    ) -> DeChunkState:
        """Allocate inference state for generation."""
        return DeChunkState(
            last_value=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        inference_params: Optional[DeChunkState] = None,
    ) -> torch.Tensor:
        """
        Reconstruct full sequence from chunked representations.
        
        Runs EMA on the M boundary tokens only, then broadcasts back to all
        positions via plug_back_idx (matching H-Net dc.py semantics).
        
        EMA: out_t = p_t * chunk_t + (1 - p_t) * out_{t-1}
        where p_t is the boundary probability at each chunk position.
        
        Args:
            hidden_states: Chunked representations (B, M, D) or packed (M, D)
            boundary_mask: Original boundary mask (B, L) or (T,)
            boundary_prob: Boundary probabilities (B, L, 2) or (T, 2)
            cu_seqlens: Cumulative sequence lengths for packed mode (N+1,)
            mask: Valid token mask (B, L)
            inference_params: Optional state for prefill
            
        Returns:
            Reconstructed sequence (B, L, D) or packed (T, D)
        """
        original_dtype = hidden_states.dtype
        packed_mode = cu_seqlens is not None

        if packed_mode:
            # Packed mode: hidden_states is (M, D), boundary_mask is (T,)
            T = boundary_mask.shape[0]
            M = hidden_states.shape[0]
            D = hidden_states.shape[-1]
            device = hidden_states.device

            p_all = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - 1e-4)  # (T,)
            p = p_all[boundary_mask]  # (M,)

            # Build seq_idx for packed sequences so EMA resets at sequence boundaries
            seq_idx = torch.zeros(M, dtype=torch.long, device=device)
            next_cu = F.pad(boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0))
            for i in range(len(next_cu) - 1):
                seq_idx[next_cu[i]:next_cu[i+1]] = i

            if M > 0:
                p_expanded = p.unsqueeze(-1)  # (M, 1)
                x_hgrn = (p_expanded * hidden_states.float()).unsqueeze(0).unsqueeze(0)  # (1, 1, M, D)
                g_hgrn = (1 - p_expanded).expand(-1, D).unsqueeze(0).unsqueeze(0)  # (1, 1, M, D)

                # Process each sequence segment separately via seq_idx
                # Group by sequence and run EMA per sequence
                ema_parts = []
                for i in range(len(next_cu) - 1):
                    start, end = int(next_cu[i]), int(next_cu[i+1])
                    if end > start:
                        seg_x = x_hgrn[:, :, start:end, :]
                        seg_g = g_hgrn[:, :, start:end, :]
                        seg_out, _ = fused_recurrent_hgrn(seg_x, seg_g)
                        ema_parts.append(seg_out.squeeze(0).squeeze(0))
                if ema_parts:
                    ema_out = torch.cat(ema_parts, dim=0)  # (M, D)
                else:
                    ema_out = hidden_states.float()
            else:
                ema_out = hidden_states.float()

            # Broadcast back to all T positions
            plug_back_idx = boundary_mask.cumsum(dim=0) - 1  # (T,)
            out = torch.gather(
                ema_out, dim=0,
                index=plug_back_idx.unsqueeze(-1).expand(-1, D),
            )  # (T, D)

            return out.to(original_dtype)

        # Padded mode: original implementation
        B, L = boundary_mask.shape
        M = hidden_states.shape[1]
        D = hidden_states.shape[-1]
        device = hidden_states.device

        # Extract boundary probabilities for the M boundary tokens only.
        # Use the same argsort trick as ChunkLayer to select boundary probs.
        p_all = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - 1e-4)  # (B, L)

        token_idx = (
            torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)

        p = torch.gather(p_all, dim=1, index=seq_sorted_indices[:, :M])  # (B, M)

        # Parallel EMA via fused_recurrent_hgrn (Triton on GPU, naive loop on CPU).
        # EMA: out_t = p_t * x_t + (1-p_t) * out_{t-1}
        # Maps to HGRN: h_t = g_t * h_{t-1} + x_t  with  g=(1-p), x=p*hidden.
        if M > 0:
            p_expanded = p.unsqueeze(-1)  # (B, M, 1)
            x_hgrn = (p_expanded * hidden_states.float()).unsqueeze(1)  # (B, 1, M, D)
            g_hgrn = (1 - p_expanded).expand(-1, -1, D).unsqueeze(1)  # (B, 1, M, D)
            ema_out, _ = fused_recurrent_hgrn(x_hgrn, g_hgrn)  # (B, 1, M, D)
            ema_out = ema_out.squeeze(1)  # (B, M, D)
        else:
            ema_out = hidden_states.float()

        # Broadcast EMA output back to all L positions via plug_back_idx.
        plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
        out = torch.gather(
            ema_out,
            dim=1,
            index=plug_back_idx.unsqueeze(-1).expand(-1, -1, D),
        )  # (B, L, D)

        if inference_params is not None:
            inference_params.last_value.copy_(out[:, -1].to(original_dtype))

        return out.to(original_dtype)

    def step(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        inference_params: DeChunkState,
    ) -> torch.Tensor:
        """
        Step-by-step dechunking for generation.
        
        Args:
            hidden_states: Chunk hidden states (B', 1, D) where B' = sum(boundary_mask)
            boundary_mask: Boundary decisions (B,)
            boundary_prob: Boundary probabilities (B, 2)
            inference_params: Current EMA state
            
        Returns:
            Reconstructed output (B, 1, D)
        """
        B = boundary_mask.shape[0]
        D = inference_params.last_value.shape[-1]
        device = inference_params.last_value.device
        dtype = inference_params.last_value.dtype

        p = torch.zeros(B, device=device, dtype=dtype)
        p[boundary_mask] = boundary_prob[boundary_mask, -1].clamp(min=1e-4, max=1 - 1e-4)

        current_hidden_states = torch.zeros(B, D, device=device, dtype=dtype)
        if hidden_states.shape[0] > 0:
            current_hidden_states[boundary_mask] = hidden_states.squeeze(1)

        result = p.unsqueeze(-1) * current_hidden_states + \
                 (1 - p.unsqueeze(-1)) * inference_params.last_value
        inference_params.last_value.copy_(result)

        return result.unsqueeze(1)
