# -*- coding: utf-8 -*-

"""
Fused recurrent HGRN implementation.

Sequential recurrence optimized with Triton kernels.
Faster than naive Python loops, but slower than chunk-parallel for long sequences.
Used during inference (single token generation).

Adapted from matmulfreellm.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

from hnet_bit.utils.helpers import contiguous


@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    x,
    g,
    o,
    h0,
    ht,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_x = x + i_bh * T * D + o_d
    p_g = g + i_bh * T * D + o_d
    p_o = o + i_bh * T * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
    for _ in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_h = b_g * b_h + b_x
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask)

        p_x += D
        p_g += D
        p_o += D

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * D + o_d
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    g,
    o,
    dx,
    dg,
    do,
    h0,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_g = g + (i_bh * T + T - 1) * D + o_d
    p_o = o + (i_bh * T + T - 2) * D + o_d
    p_dx = dx + (i_bh * T + T - 1) * D + o_d
    p_dg = dg + (i_bh * T + T - 1) * D + o_d
    p_do = do + (i_bh * T + T - 1) * D + o_d

    b_dh = tl.zeros([BD], dtype=tl.float32)
    for i in range(T - 1, -1, -1):
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
        if i > 0:
            b_o = tl.load(p_o, mask=mask, other=0).to(tl.float32)
        elif USE_INITIAL_STATE:
            b_o = tl.load(h0 + i_bh * D + o_d, mask=mask, other=0).to(tl.float32)
        else:
            b_o = tl.zeros([BD], dtype=tl.float32)

        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dg = b_dh * b_o
        b_dh = b_dh * b_g
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=mask)

        p_g -= D
        p_o -= D
        p_dx -= D
        p_dg -= D
        p_do -= D


class FusedRecurrentHGRNFunction(torch.autograd.Function):
    """Fused recurrent HGRN forward and backward passes."""

    @staticmethod
    @contiguous
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        B, H, T, D = x.shape

        final_state = None
        if output_final_state:
            final_state = x.new_empty(B, H, D)

        o = torch.empty_like(x)
        
        def grid(meta): 
            return (triton.cdiv(D, meta['BD']), B * H)
        
        fused_recurrent_hgrn_fwd_kernel[grid](
            x, g, o, initial_state, final_state,
            T, D,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None
        )
        ctx.save_for_backward(g, o, initial_state)
        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht=None):
        g, o, initial_state = ctx.saved_tensors
        B, H, T, D = do.shape

        dx = torch.empty_like(o)
        dg = torch.empty_like(g)
        
        def grid(meta): 
            return (triton.cdiv(D, meta['BD']), B * H)
        
        fused_recurrent_hgrn_bwd_kernel[grid](
            g, o, dx, dg, do, initial_state,
            T, D,
            USE_INITIAL_STATE=initial_state is not None,
        )

        return dx, dg, None, None


def _naive_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Naive (pure-PyTorch) recurrent HGRN for CPU fallback.
    
    Computes: h_t = g_t * h_{t-1} + x_t
    """
    B, H, T, D = x.shape
    dtype = x.dtype

    h = initial_state.float() if initial_state is not None else x.new_zeros(B, H, D, dtype=torch.float32)
    outputs = []
    for t in range(T):
        h = g[:, :, t].float() * h + x[:, :, t].float()
        outputs.append(h)
    o = torch.stack(outputs, dim=2).to(dtype)
    final_state = h.to(dtype) if output_final_state else None
    return o, final_state


def fused_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused recurrent HGRN.
    
    Computes: h_t = g_t * h_{t-1} + x_t
    
    Note: Unlike chunk_hgrn which uses exp(g), this uses g directly.
    The caller should pass g after applying sigmoid or appropriate transform.
    
    Uses Triton kernels on CUDA, falls back to naive PyTorch loop on CPU.
    
    Args:
        x: Input tensor [B, H, T, D]
        g: Gate tensor [B, H, T, D]
        initial_state: Optional initial hidden state [B, H, D]
        output_final_state: Whether to return final hidden state
        
    Returns:
        Tuple of (output [B, H, T, D], final_state [B, H, D] or None)
    """
    if not x.is_cuda:
        return _naive_recurrent_hgrn(x, g, initial_state, output_final_state)
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = FusedRecurrentHGRNFunction.apply(x, g, initial_state, output_final_state)
    return o, final_state
