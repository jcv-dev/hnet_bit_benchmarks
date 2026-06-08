# -*- coding: utf-8 -*-

"""
BitLinear layer with ternary weight quantization.

Implements ternary weights {-1, 0, +1} with Straight-Through Estimator (STE)
for gradient computation during training.

Adapted from matmulfreellm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """
    Per-token quantization to 8 bits. No grouping is needed for quantization.

    Args:
        x: An activation tensor with shape [n, d].

    Returns:
        A quantized activation tensor with shape [n, d].
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    Per-tensor quantization to 1.58 bits (ternary: {-1, 0, +1}).
    
    Uses mean absolute value for scaling, then rounds to nearest of {-1, 0, +1}.

    Args:
        w: A weight tensor with shape [d, k].

    Returns:
        A quantized weight tensor with shape [d, k].
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm - no mean centering, just variance normalization.
    
    Args:
        hidden_size: Size of the hidden dimension
        eps: Small constant for numerical stability
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(
        self, 
        x: torch.Tensor, 
        residual: torch.Tensor = None, 
        prenorm: bool = False
    ) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor
            residual: Optional residual to add before normalization
            prenorm: If True, return both normalized output and (x + residual)
            
        Returns:
            Normalized tensor, or tuple of (normalized, residual_sum) if prenorm=True
        """
        if residual is not None:
            x = x + residual

        # Save pre-norm value for prenorm residual stream
        input_for_residual = x

        # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        output = self.weight * x

        if prenorm:
            return output, input_for_residual
        return output


class BitLinear(nn.Linear):
    """
    A custom linear layer that applies quantization on both activations and weights.
    
    Uses:
    - Ternary weight quantization: weights → {-1, 0, +1}
    - 8-bit activation quantization: per-token scaling to [-128, 127]
    - Straight-Through Estimator (STE) for gradient flow through quantization
    - Built-in RMS normalization before quantization
    
    This is primarily for training; kernel optimization is needed for deployment.
    
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If False, the layer will not learn an additive bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(BitLinear, self).__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features, eps=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantization.

        Args:
            x: An input tensor with shape [..., in_features].

        Returns:
            An output tensor with shape [..., out_features].
        """
        w = self.weight

        # Apply RMS normalization to the input
        x_norm = self.norm(x)

        # Apply quantization with STE trick:
        # Forward uses quantized values, backward flows through full precision
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()

        # Perform linear operation with quantized values
        y = F.linear(x_quant, w_quant)
        
        if self.bias is not None:
            y = y + self.bias
            
        return y
