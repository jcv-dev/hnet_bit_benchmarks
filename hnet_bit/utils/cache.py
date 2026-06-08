# -*- coding: utf-8 -*-

"""
Recurrent cache for storing hidden states during generation.

Adapted from matmulfreellm for HGRN models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache


class RecurrentCache(Cache):
    """
    A cache used for storing hidden states produced by recurrent models like HGRN.

    It stores the states of each layer as a tuple of tensors.
    """

    def __init__(
        self,
        seen_tokens: int = 0
    ) -> None:
        self.states: List[Tuple[torch.Tensor, ...]] = []
        self._seen_tokens = seen_tokens

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states:
            yield state

    def __len__(self):
        return len(self.states)

    def update(
        self,
        state: Tuple[torch.Tensor],
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Updates the cache with the new `state` for the layer `layer_idx`.

        Parameters:
            state: The new state to cache.
            layer_idx: The index of the layer to cache the states for.
            cache_kwargs: Additional arguments (unused, for compatibility).

        Return:
            The updated state.
        """
        if isinstance(state, torch.Tensor):
            state = (state,)
        if len(self.states) <= layer_idx:
            self.states.append(state)
        else:
            # In-place update for existing tensors to save memory
            updated_state = []
            for i, s in enumerate(state):
                if i < len(self.states[layer_idx]) and self.states[layer_idx][i] is not None:
                    self.states[layer_idx][i].copy_(s)
                    updated_state.append(self.states[layer_idx][i])
                else:
                    updated_state.append(s)
            self.states[layer_idx] = tuple(updated_state)
            # Update seen tokens once we achieve the last layer
            if layer_idx == len(self) - 1:
                self._seen_tokens += 1

        return state

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states."""
        if len(self.states) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length. RecurrentCache has no maximum."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search."""
        for layer_idx in range(len(self.states)):
            device = self.states[layer_idx][0].device
            self.states[layer_idx] = tuple(
                s.index_select(0, beam_idx.to(device)) for s in self.states[layer_idx]
            )

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        return tuple(self.states)

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        seen_tokens: int = 0
    ) -> RecurrentCache:
        """Converts a legacy cache format into RecurrentCache."""
        cache = cls(seen_tokens)
        if past_key_values is not None:
            # If it's already a RecurrentCache, return it
            if isinstance(past_key_values, RecurrentCache):
                return past_key_values
            
            # If it's a Cache object (like DynamicCache from transformers),
            # it's incompatible with recurrent models - return empty cache
            if isinstance(past_key_values, Cache):
                # DynamicCache is for attention-based models, not recurrent models
                # Return empty cache to start fresh
                return cache
            
            # Now handle as tuple/list
            for layer_idx in range(len(past_key_values)):
                cache.update(past_key_values[layer_idx], layer_idx)
        return cache
