# -*- coding: utf-8 -*-

"""
Unit tests for CausalMHABit, CausalMHABlock, mixed HGRNBitStack, and model-level
attention integration.

Tests:
1. CausalMHABit: output shape, causal masking, sliding window, RoPE, KV cache, gradients
2. CausalMHABlock: interface compatibility with HGRNBitBlock
3. Mixed HGRNBitStack: correct block types, forward/backward, step with cache
4. Model integration: config round-trip, generation with attention
"""

import sys
import os
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn

try:
    import pytest
except ImportError:
    pytest = None


def _skip_if_no_cuda(func):
    """Decorator to skip test if CUDA not available."""
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            print(f"WARNING Skipping {func.__name__}: CUDA not available")
            return
        return func(*args, **kwargs)
    return wrapper


def _make_small_config(**overrides):
    """Create a minimal config with innermost attention enabled."""
    from hnet_bit.models.hnet_bit import HNetBitConfig
    defaults = dict(
        vocab_size=256,
        d_model=[64, 96],
        num_blocks=[[2, 0, 2], [4]],
        num_heads=2,
        expand_ratio=1,
        hidden_ratio=2,
        max_position_embeddings=128,
        use_cache=True,
        innermost_use_attention=True,
        attention_window_size=32,
        attention_num_heads=4,
        attention_layers_pattern="xaxa",
    )
    defaults.update(overrides)
    return HNetBitConfig(**defaults)


# ---------------------------------------------------------------------------
# Test CausalMHABit (raw attention module)
# ---------------------------------------------------------------------------

class TestCausalMHABit(unittest.TestCase):
    """Tests for the raw CausalMHABit attention layer."""

    def setUp(self):
        from hnet_bit.layers.attention import CausalMHABit
        self.hidden_size = 64
        self.num_heads = 4
        self.window_size = 16
        self.attn = CausalMHABit(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            window_size=self.window_size,
            max_position_embeddings=128,
        )

    def test_output_shape(self):
        """Output matches input shape (B, L, D)."""
        x = torch.randn(2, 32, self.hidden_size)
        out, _, _ = self.attn(x)
        self.assertEqual(out.shape, (2, 32, self.hidden_size))

    def test_causal_masking(self):
        """Verify mask is lower-triangular with sliding window."""
        from hnet_bit.layers.attention import CausalMHABit
        attn = CausalMHABit(hidden_size=32, num_heads=2, window_size=4)
        mask = attn._build_sliding_window_mask(8, torch.device('cpu'), torch.float32)
        # Check causality: position 0 can only attend to position 0
        self.assertEqual(mask[0, 0].item(), 0.0)
        self.assertEqual(mask[0, 1].item(), float('-inf'))
        # Check window: position 7 can attend to positions 4-7 but not 3
        self.assertEqual(mask[7, 4].item(), 0.0)
        self.assertEqual(mask[7, 7].item(), 0.0)
        self.assertEqual(mask[7, 3].item(), float('-inf'))

    def test_full_causal_when_window_zero(self):
        """window_size=0 means full causal attention (no window constraint)."""
        from hnet_bit.layers.attention import CausalMHABit
        attn = CausalMHABit(hidden_size=32, num_heads=2, window_size=0)
        mask = attn._build_sliding_window_mask(8, torch.device('cpu'), torch.float32)
        # Position 7 should attend to all previous positions
        for j in range(8):
            self.assertEqual(mask[7, j].item(), 0.0)
        # Position 0 should only attend to itself
        self.assertEqual(mask[0, 0].item(), 0.0)
        for j in range(1, 8):
            self.assertEqual(mask[0, j].item(), float('-inf'))

    def test_gradient_flow(self):
        """Gradients flow through attention layer."""
        x = torch.randn(1, 16, self.hidden_size, requires_grad=True)
        out, _, _ = self.attn(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))

    def test_init_state_empty_cache(self):
        """init_state returns zero-length KV cache tensors."""
        k_cache, v_cache = self.attn.init_state(batch_size=3)
        self.assertEqual(k_cache.shape, (3, self.num_heads, 0, self.hidden_size // self.num_heads))
        self.assertEqual(v_cache.shape, (3, self.num_heads, 0, self.hidden_size // self.num_heads))

    def test_kv_cache_step_by_step(self):
        """Test single-token generation with KV cache accumulation."""
        from hnet_bit.models.hnet_bit import HGRNBlockCache

        class _SimpleCache:
            def __init__(self, states):
                self.states = states
            def __getitem__(self, idx):
                return self.states[idx]
            def update(self, state, idx, seq_len=1):
                self.states[idx] = state

        B = 2
        state = self.attn.init_state(B)
        cache = _SimpleCache([state])
        self.attn.layer_idx = 0  # Ensure layer_idx is 0

        # First token
        x1 = torch.randn(B, 1, self.hidden_size)
        out1, _, cache = self.attn(x1, past_key_values=cache, use_cache=True)
        self.assertEqual(out1.shape, (B, 1, self.hidden_size))
        self.assertEqual(cache.states[0][0].shape[2], 1)  # 1 token cached

        # Second token
        x2 = torch.randn(B, 1, self.hidden_size)
        out2, _, cache = self.attn(x2, past_key_values=cache, use_cache=True)
        self.assertEqual(out2.shape, (B, 1, self.hidden_size))
        self.assertEqual(cache.states[0][0].shape[2], 2)  # 2 tokens cached

    def test_head_dim_assertion(self):
        """Should raise assertion when hidden_size not divisible by num_heads."""
        from hnet_bit.layers.attention import CausalMHABit
        with self.assertRaises(AssertionError):
            CausalMHABit(hidden_size=65, num_heads=4)


# ---------------------------------------------------------------------------
# Test CausalMHABlock (block wrapper)
# ---------------------------------------------------------------------------

class TestCausalMHABlock(unittest.TestCase):
    """Tests that CausalMHABlock matches HGRNBitBlock interface."""

    def setUp(self):
        self.config = _make_small_config()

    def test_forward_output_shape(self):
        """Forward output shape matches input."""
        from hnet_bit.layers.attention import CausalMHABlock
        stage_cfg = self.config._make_stage_config(1)  # innermost stage
        block = CausalMHABlock(stage_cfg, layer_idx=0)

        B, L, D = 2, 16, self.config.d_model[1]
        x = torch.randn(B, L, D)
        out, attn_w, cache = block(x)
        self.assertEqual(out.shape, (B, L, D))
        self.assertIsNone(attn_w)

    def test_interface_matches_hgrn_block(self):
        """CausalMHABlock has same interface as HGRNBitBlock."""
        from hnet_bit.layers.attention import CausalMHABlock
        from hnet_bit.layers.hgrn_bit import HGRNBitBlock

        stage_cfg = self.config._make_stage_config(1)
        attn_block = CausalMHABlock(stage_cfg, layer_idx=0)
        hgrn_block = HGRNBitBlock(stage_cfg, layer_idx=1)

        B, L, D = 2, 16, self.config.d_model[1]
        x = torch.randn(B, L, D)

        # Both should accept same kwargs
        out_a, _, _ = attn_block(x, use_cache=False)
        out_h, _, _ = hgrn_block(x, use_cache=False)
        self.assertEqual(out_a.shape, out_h.shape)

    def test_has_attn_with_init_state(self):
        """Block exposes .attn with init_state() for cache setup."""
        from hnet_bit.layers.attention import CausalMHABlock
        stage_cfg = self.config._make_stage_config(1)
        block = CausalMHABlock(stage_cfg, layer_idx=0)
        self.assertTrue(hasattr(block, 'attn'))
        self.assertTrue(hasattr(block.attn, 'init_state'))
        state = block.attn.init_state(batch_size=2)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 2)


# ---------------------------------------------------------------------------
# Test mixed HGRNBitStack with attention blocks
# ---------------------------------------------------------------------------

class TestMixedHGRNBitStack(unittest.TestCase):
    """Tests for HGRNBitStack with mixed HGRN+attention blocks."""

    def test_pattern_creates_correct_block_types(self):
        """Pattern 'xaxa' creates alternating HGRN and attention blocks."""
        from hnet_bit.models.hnet_bit import HGRNBitStack
        from hnet_bit.layers.attention import CausalMHABlock
        from hnet_bit.layers.hgrn_bit import HGRNBitBlock

        config = _make_small_config(attention_layers_pattern="xaxa")
        stack = HGRNBitStack(config, stage_idx=1, num_layers=4, is_innermost=True)

        self.assertIsInstance(stack.layers[0], HGRNBitBlock)
        self.assertIsInstance(stack.layers[1], CausalMHABlock)
        self.assertIsInstance(stack.layers[2], HGRNBitBlock)
        self.assertIsInstance(stack.layers[3], CausalMHABlock)

    def test_all_hgrn_when_not_innermost(self):
        """Non-innermost stack should use only HGRN blocks regardless of config."""
        from hnet_bit.models.hnet_bit import HGRNBitStack
        from hnet_bit.layers.hgrn_bit import HGRNBitBlock

        config = _make_small_config(attention_layers_pattern="aaaa")
        # is_innermost=False, so all blocks should be HGRN
        stack = HGRNBitStack(config, stage_idx=0, num_layers=4, is_innermost=False)

        for layer in stack.layers:
            self.assertIsInstance(layer, HGRNBitBlock)

    def test_all_hgrn_when_attention_disabled(self):
        """When innermost_use_attention=False, all blocks are HGRN."""
        from hnet_bit.models.hnet_bit import HGRNBitStack
        from hnet_bit.layers.hgrn_bit import HGRNBitBlock

        config = _make_small_config(innermost_use_attention=False)
        stack = HGRNBitStack(config, stage_idx=1, num_layers=4, is_innermost=True)

        for layer in stack.layers:
            self.assertIsInstance(layer, HGRNBitBlock)

    def test_pattern_cyclic_extension(self):
        """Short pattern is extended cyclically to match num_layers."""
        from hnet_bit.models.hnet_bit import HGRNBitStack
        from hnet_bit.layers.attention import CausalMHABlock
        from hnet_bit.layers.hgrn_bit import HGRNBitBlock

        config = _make_small_config(attention_layers_pattern="xa")
        stack = HGRNBitStack(config, stage_idx=1, num_layers=4, is_innermost=True)

        # Pattern "xa" extended to "xaxa"
        self.assertIsInstance(stack.layers[0], HGRNBitBlock)
        self.assertIsInstance(stack.layers[1], CausalMHABlock)
        self.assertIsInstance(stack.layers[2], HGRNBitBlock)
        self.assertIsInstance(stack.layers[3], CausalMHABlock)

    def test_forward_backward(self):
        """Mixed stack forward and backward pass works."""
        config = _make_small_config()
        from hnet_bit.models.hnet_bit import HGRNBitStack
        stack = HGRNBitStack(config, stage_idx=1, num_layers=4, is_innermost=True)

        B, L, D = 2, 16, config.d_model[1]
        x = torch.randn(B, L, D, requires_grad=True)
        out, cache = stack(x)
        self.assertEqual(out.shape, (B, L, D))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_init_cache_mixed(self):
        """init_cache works for mixed HGRN+attention blocks."""
        config = _make_small_config()
        from hnet_bit.models.hnet_bit import HGRNBitStack
        stack = HGRNBitStack(config, stage_idx=1, num_layers=4, is_innermost=True)
        cache = stack.init_cache(batch_size=2)

        # Should have 4 states (one per layer)
        self.assertEqual(len(cache.states), 4)
        # HGRN layers (0, 2) return tuple of tensors (recurrent state)
        self.assertIsInstance(cache.states[0], tuple)
        self.assertIsInstance(cache.states[0][-1], torch.Tensor)
        # Attention layers (1, 3) return (k_cache, v_cache) tuples
        self.assertIsInstance(cache.states[1], tuple)
        self.assertEqual(len(cache.states[1]), 2)

    def test_step_with_mixed_cache(self):
        """Step (single token generation) works with mixed blocks."""
        config = _make_small_config()
        from hnet_bit.models.hnet_bit import HGRNBitStack
        stack = HGRNBitStack(config, stage_idx=1, num_layers=4, is_innermost=True)
        cache = stack.init_cache(batch_size=2)

        B, D = 2, config.d_model[1]
        x = torch.randn(B, 1, D)
        out, cache = stack.step(x, cache)
        self.assertEqual(out.shape, (B, 1, D))

        # Step again
        x2 = torch.randn(B, 1, D)
        out2, cache = stack.step(x2, cache)
        self.assertEqual(out2.shape, (B, 1, D))


# ---------------------------------------------------------------------------
# Test model-level integration
# ---------------------------------------------------------------------------

class TestModelWithAttention(unittest.TestCase):
    """End-to-end tests with attention enabled in the model config."""

    def test_config_serialization_roundtrip(self):
        """Attention config fields survive to_dict / from_dict."""
        config = _make_small_config()
        d = config.to_dict()
        self.assertTrue(d['innermost_use_attention'])
        self.assertEqual(d['attention_window_size'], 32)
        self.assertEqual(d['attention_num_heads'], 4)
        self.assertEqual(d['attention_layers_pattern'], "xaxa")

        from hnet_bit.models.hnet_bit import HNetBitConfig
        config2 = HNetBitConfig(**{k: v for k, v in d.items() if k != 'model_type'})
        self.assertTrue(config2.innermost_use_attention)
        self.assertEqual(config2.attention_layers_pattern, "xaxa")

    def test_model_forward_with_attention(self):
        """Full model forward pass with innermost attention."""
        from hnet_bit.models.hnet_bit import HNetBitForCausalLM
        config = _make_small_config()
        model = HNetBitForCausalLM(config)
        model.eval()

        B, L = 2, 32
        input_ids = torch.randint(0, 256, (B, L))
        with torch.no_grad():
            output = model(input_ids)
        self.assertEqual(output.logits.shape, (B, L, 256))

    def test_model_generate_with_attention(self):
        """Model generate() works with mixed attention+HGRN cache."""
        from hnet_bit.models.hnet_bit import HNetBitForCausalLM
        config = _make_small_config()
        model = HNetBitForCausalLM(config)
        model.eval()

        prompt = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=4, temperature=1.0)
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 12)  # 8 prompt + 4 generated


# ---------------------------------------------------------------------------
# Test RotaryEmbedding
# ---------------------------------------------------------------------------

class TestRotaryEmbedding(unittest.TestCase):
    """Tests for RoPE implementation."""

    def test_output_shape(self):
        from hnet_bit.layers.attention import RotaryEmbedding
        rope = RotaryEmbedding(dim=32, max_position_embeddings=64)
        cos, sin = rope(seq_len=16)
        # dim=32 -> inv_freq has 16 entries -> freqs is (16, 16) -> cat gives (16, 32)
        self.assertEqual(cos.shape, (16, 32))
        self.assertEqual(sin.shape, (16, 32))

    def test_with_offset(self):
        from hnet_bit.layers.attention import RotaryEmbedding
        rope = RotaryEmbedding(dim=32, max_position_embeddings=64)
        cos_full, sin_full = rope(seq_len=16, offset=0)
        cos_off, sin_off = rope(seq_len=1, offset=15)
        # Position 15 from full should match offset=15, length=1
        self.assertTrue(torch.allclose(cos_full[15:16], cos_off))
        self.assertTrue(torch.allclose(sin_full[15:16], sin_off))

    def test_auto_extend_cache(self):
        """Cache auto-extends when seq_len exceeds max_position_embeddings."""
        from hnet_bit.layers.attention import RotaryEmbedding
        rope = RotaryEmbedding(dim=32, max_position_embeddings=16)
        cos, sin = rope(seq_len=32)  # Should auto-extend
        self.assertEqual(cos.shape[0], 32)


if __name__ == '__main__':
    unittest.main()
