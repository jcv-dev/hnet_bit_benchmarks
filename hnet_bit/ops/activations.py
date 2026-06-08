# -*- coding: utf-8 -*-

"""
Activation functions for HNet-Bit.

Includes SwiGLU and other gated activations used in HGRN.
"""

import torch
import torch.nn.functional as F


@torch.jit.script
def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation: x * sigmoid(x)"""
    return F.silu(x)


@torch.jit.script
def swiglu_simple(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU activation: x * sigmoid(x) * y
    
    A gated activation combining swish with a gate signal.
    
    Args:
        x: Input tensor (will be passed through swish)
        y: Gate tensor
        
    Returns:
        Gated output
    """
    return F.silu(x) * y


# Try to use optimized CUDA jiterator if available
try:
    swiglu_fwd_codestring = """
    template <typename T> T swiglu_fwd(T x, T y) {
        return float(x) * float(y) / (1.0f + ::exp(-float(x)));
    }
    """
    swiglu_bwd_codestring = """
    template <typename T> T swiglu_bwd(T x, T y, T g, T& dx, T& dy) {
        float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
        dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
        dy = float(x) * x_sigmoid * float(g);
    }
    """
    
    _swiglu_fwd = torch.cuda.jiterator._create_jit_fn(swiglu_fwd_codestring)
    _swiglu_bwd = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_bwd_codestring, num_outputs=2)
    
    class SwiGLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            ctx.save_for_backward(x, y)
            if x.is_cuda:
                return _swiglu_fwd(x, y)
            return F.silu(x) * y

        @staticmethod
        def backward(ctx, dout):
            x, y = ctx.saved_tensors
            if x.is_cuda:
                return _swiglu_bwd(x, y, dout)
            # CPU fallback for backward
            x_sigmoid = torch.sigmoid(x.float())
            dx = x_sigmoid * (1 + x.float() * (1.0 - x_sigmoid)) * dout.float() * y.float()
            dy = x.float() * x_sigmoid * dout.float()
            return dx.to(x.dtype), dy.to(y.dtype)
    
    swiglu = SwiGLUFunction.apply
    _CUDA_SWIGLU_AVAILABLE = True

except Exception:
    # Fallback to simple implementation
    def swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return swiglu_simple(x, y)
    _CUDA_SWIGLU_AVAILABLE = False


# Activation function mapping
ACT2FN = {
    'silu': swish,
    'swish': swish,
}
