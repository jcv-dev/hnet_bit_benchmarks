# -*- coding: utf-8 -*-

"""Utility functions and decorators."""

import functools
from typing import Callable

import torch


def apply_optimization_params(param: torch.Tensor, **kwargs) -> None:
    """
    Annotate a parameter with optimization hyperparameters.
    
    Stores settings like lr_multiplier or weight_decay in param._optim attribute.
    Used by the optimizer builder to create per-parameter groups with different
    learning rates or weight decay values.
    
    This pattern is used by H-Net for stage-specific learning rates in hierarchical models.
    
    Args:
        param: PyTorch parameter to annotate
        **kwargs: Optimization settings (e.g., lr_multiplier=2.0, weight_decay=0.0)
    
    Example:
        >>> # Apply 2x learning rate to encoder parameters
        >>> apply_optimization_params(model.encoder.weight, lr_multiplier=2.0)
        >>> # Disable weight decay for bias parameters
        >>> apply_optimization_params(model.bias, weight_decay=0.0)
    """
    if hasattr(param, "_optim"):
        param._optim.update(kwargs)
    else:
        param._optim = kwargs


def contiguous(fn: Callable) -> Callable:
    """
    Decorator that ensures all tensor arguments are contiguous before calling the function.
    
    Args:
        fn: Function to wrap
        
    Returns:
        Wrapped function with contiguous tensor arguments
    """
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(
            ctx,
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}
        )
    return wrapper
