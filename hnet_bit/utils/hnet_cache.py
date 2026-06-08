# -*- coding: utf-8 -*-

"""
Hierarchical cache for HNetBit generation.

Nested cache structure matching the recursive HNetBit architecture:
- Each non-innermost stage has: encoder_cache, routing_state, main_network_cache, dechunk_state, decoder_cache
- Innermost stage has: main_network_cache only (flat HGRN block states)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch

from hnet_bit.ops.dynamic_chunking import RoutingModuleState, DeChunkState


@dataclass
class HGRNBlockCache:
    """
    Cache for a stack of HGRNBitBlocks (encoder, decoder, or innermost blocks).
    
    Stores recurrent states per layer as tuples of tensors.
    
    Attributes:
        states: List of per-layer state tuples, indexed by layer_idx
        seen_tokens: Number of tokens processed so far
    """
    states: List[Optional[Tuple[torch.Tensor, ...]]] = field(default_factory=list)
    seen_tokens: int = 0

    def __getitem__(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, ...]]:
        if layer_idx < len(self.states):
            return self.states[layer_idx]
        return None

    def update(
        self,
        state: Tuple[torch.Tensor, ...],
        layer_idx: int,
        seq_len: int = 1,
    ) -> None:
        """Update cache for a specific layer."""
        if isinstance(state, torch.Tensor):
            state = (state,)
        while len(self.states) <= layer_idx:
            self.states.append(None)
        
        if self.states[layer_idx] is not None:
            # In-place update to save memory when shapes match,
            # otherwise replace (needed for growing KV caches in attention)
            updated = []
            for i, s in enumerate(state):
                old = self.states[layer_idx]
                if i < len(old) and old[i] is not None and old[i].shape == s.shape:
                    old[i].copy_(s)
                    updated.append(old[i])
                else:
                    updated.append(s)
            self.states[layer_idx] = tuple(updated)
        else:
            self.states[layer_idx] = state
        
        # Update seen_tokens once we update the last layer
        if layer_idx == len(self.states) - 1:
            self.seen_tokens += seq_len

    @property
    def num_layers(self) -> int:
        return len(self.states)


@dataclass
class HNetBitCache:
    """
    Nested hierarchical cache for HNetBit model.
    
    Matches the recursive structure of HNetBit:
    - Non-innermost: encoder → routing → main → dechunk → decoder
    - Innermost: main only (stack of HGRN blocks)
    
    The main_network_cache is either another HNetBitCache (recursive)
    or an HGRNBlockCache (innermost level).
    
    Attributes:
        encoder_cache: Cache for encoder HGRN blocks (None if innermost)
        routing_state: Routing module state for boundary prediction (None if innermost)
        main_network_cache: Recursive HNetBitCache or HGRNBlockCache
        dechunk_state: DeChunk EMA state (None if innermost)
        decoder_cache: Cache for decoder HGRN blocks (None if innermost)
        is_innermost: Whether this is the innermost hierarchy level
    """
    encoder_cache: Optional[HGRNBlockCache] = None
    routing_state: Optional[RoutingModuleState] = None
    main_network_cache: Optional[Union['HNetBitCache', HGRNBlockCache]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_cache: Optional[HGRNBlockCache] = None
    is_innermost: bool = False

    def get_seq_length(self) -> int:
        """Get the number of tokens processed at the outermost stage."""
        if self.is_innermost:
            if self.main_network_cache is not None:
                return self.main_network_cache.seen_tokens
            return 0
        
        # Always check encoder cache for non-innermost (it's at the input level)
        if self.encoder_cache is not None:
            return self.encoder_cache.seen_tokens
        
        # Fallback to dechunk state if available
        if self.dechunk_state is not None and hasattr(self.dechunk_state, 'tokens_processed'):
            return self.dechunk_state.tokens_processed
        
        return 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cache for beam search."""
        device = beam_idx.device
        
        if self.encoder_cache is not None:
            for i, state in enumerate(self.encoder_cache.states):
                if state is not None:
                    self.encoder_cache.states[i] = tuple(
                        s.index_select(0, beam_idx.to(s.device)) for s in state
                    )
        
        if self.routing_state is not None:
            self.routing_state.has_seen_tokens = self.routing_state.has_seen_tokens.index_select(
                0, beam_idx.to(self.routing_state.has_seen_tokens.device)
            )
            self.routing_state.last_hidden_state = self.routing_state.last_hidden_state.index_select(
                0, beam_idx.to(self.routing_state.last_hidden_state.device)
            )
        
        if self.main_network_cache is not None:
            if isinstance(self.main_network_cache, HNetBitCache):
                self.main_network_cache.reorder_cache(beam_idx)
            else:
                for i, state in enumerate(self.main_network_cache.states):
                    if state is not None:
                        self.main_network_cache.states[i] = tuple(
                            s.index_select(0, beam_idx.to(s.device)) for s in state
                        )
        
        if self.dechunk_state is not None:
            self.dechunk_state.last_value = self.dechunk_state.last_value.index_select(
                0, beam_idx.to(self.dechunk_state.last_value.device)
            )
        
        if self.decoder_cache is not None:
            for i, state in enumerate(self.decoder_cache.states):
                if state is not None:
                    self.decoder_cache.states[i] = tuple(
                        s.index_select(0, beam_idx.to(s.device)) for s in state
                    )
