# -*- coding: utf-8 -*-

"""
Short convolution module for local inductive bias in HGRN.

Copied from matmulfreellm (mmfreelm/modules/convolution.py) to avoid
external dependency. Original source:
https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/convolution.py
"""

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None


# Minimal activation lookup (only silu/swish needed for HGRN usage)
ACT2FN = {
    'silu': F.silu,
    'swish': F.silu,
}


class ShortConvolution(nn.Conv1d):
    """
    Simple wrapper around ``nn.Conv1d`` that accepts dimension last.

    Uses depthwise convolution (groups=hidden_size) with optional causal_conv1d
    acceleration when available.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: Optional[str] = 'silu',
        use_causal_conv: Optional[bool] = True,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )

        self.hidden_size = hidden_size
        self.activation = None
        if activation is not None:
            assert activation in ['silu', 'swish'], \
                f"Activation `{activation}` not supported yet."
            self.activation = activation

        if use_causal_conv:
            if causal_conv1d_fn is None:
                warnings.warn(
                    "Please install `causal-conv1d` to use causal convolutions, "
                    "setting `use_causal_conv` to False."
                )
                use_causal_conv = False
        self.use_causal_conv = use_causal_conv

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.activation is not None:
            s += ', activation={activation}'
        if not self.use_causal_conv:
            s += ', use_causal_conv={use_causal_conv}'
        return s.format(**self.__dict__)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape ``[batch_size, seq_len, hidden_size]``
            cache: Previous cache tensor of shape ``[batch_size, hidden_size, kernel_size]``

        Returns:
            Tensor of shape ``[batch_size, seq_len, hidden_size]``.
            The ``cache`` (if provided) is updated in-place.
        """
        if not next(self.parameters()).is_cuda:
            warnings.warn(
                "CUDA is required for using causal convolutions, "
                "setting `use_causal_conv` to False."
            )
            self.use_causal_conv = False

        if cache is not None and x.shape[1] == 1:
            return self.step(x, cache)

        x = rearrange(x, "b l d -> b d l")
        if self.use_causal_conv:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
        else:
            x = self._conv_forward(x, self.weight, self.bias)[..., :x.shape[-1]]
            if self.activation is not None:
                x = ACT2FN[self.activation](x)
        return rearrange(x, "b d l -> b l d")

    def step(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
    ):
        """Single-token step for generation with cache."""
        assert x.shape[1] == 1, \
            "Only support decoding with 1 token at a time for now"

        x = x.squeeze(1)
        if self.use_causal_conv:
            x = causal_conv1d_update(
                x=x,
                conv_state=cache,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
        else:
            dtype = x.dtype
            cache.copy_(torch.roll(cache, shifts=-1, dims=-1))
            cache[:, :, -1] = x
            x = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
            if self.bias is not None:
                x = x + self.bias
            if self.activation is not None:
                x = ACT2FN[self.activation](x).to(dtype=dtype)
        return x.unsqueeze(1)

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size
