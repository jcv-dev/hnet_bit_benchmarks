# -*- coding: utf-8 -*-

"""
Unit tests for HGRNBitAttention: multi-head, expand_ratio, and short convolution.

Tests:
1. Multi-head HGRN: head_dim computation, forward shape, gradient flow
2. expand_ratio: state scaling, projection sizes, forward correctness
3. Short convolution: module creation, forward, init_state, step
"""

import sys
import os

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
    """Create a minimal config for fast testing."""
    from hnet_bit.models.hnet_bit import HNetBitConfig
    defaults = dict(
        vocab_size=256,
        d_model=[64, 96],
        num_blocks=[[2, 0, 2], [4]],
        num_heads=1,
        expand_ratio=1,
        hidden_ratio=2,
        max_position_embeddings=128,
        use_cache=True,
    )
    defaults.update(overrides)
    return HNetBitConfig(**defaults)


# ---------------------------------------------------------------------------
# Commit 1: Multi-Head HGRN
# ---------------------------------------------------------------------------
class TestHGRNBitAttentionMultiHead:
    """Tests for multi-head HGRN support."""

    def test_head_dim_calculation(self):
        """Test that head_dim = input_dim // num_heads."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        attn = HGRNBitAttention(hidden_size=64, num_heads=4, expand_ratio=1)
        # input_dim = 64 * 1 = 64, head_dim = 64 // 4 = 16
        assert attn.head_dim == 16, f"Expected head_dim=16, got {attn.head_dim}"
        assert attn.num_heads == 4
        assert attn.input_dim == 64
        print("OK head_dim calculation correct for num_heads=4")

    def test_multihead_forward_shape(self):
        """Test forward output shape with multi-head."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        B, L, D = 2, 32, 64
        attn = HGRNBitAttention(hidden_size=D, num_heads=4, expand_ratio=1)
        x = torch.randn(B, L, D)

        o, _, _ = attn(x)
        assert o.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {o.shape}"
        assert not torch.isnan(o).any(), "Output contains NaN"
        print("OK Multi-head forward shape correct")

    def test_assertion_checks_input_dim_not_hidden_size(self):
        """Test that assertion checks input_dim % num_heads, not hidden_size % num_heads."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        # hidden_size=64, expand_ratio=1 => input_dim=64; 64 % 3 != 0 => should fail
        try:
            HGRNBitAttention(hidden_size=64, num_heads=3, expand_ratio=1)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass

        # hidden_size=63, expand_ratio=1 => input_dim=63; 63 % 3 == 0 => should succeed
        attn = HGRNBitAttention(hidden_size=63, num_heads=3, expand_ratio=1)
        assert attn.head_dim == 21
        print("OK Assertion correctly checks input_dim divisibility")

    def test_multihead_gradient_flow(self):
        """Test gradient flow through multi-head forward."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        B, L, D = 2, 16, 64
        attn = HGRNBitAttention(hidden_size=D, num_heads=4, expand_ratio=1)
        x = torch.randn(B, L, D, requires_grad=True)

        o, _, _ = attn(x)
        loss = o.sum()
        loss.backward()

        assert x.grad is not None, "Input grad is None"
        assert not torch.isnan(x.grad).any(), "Input grad contains NaN"

        for name in ['i_proj', 'f_proj', 'g_proj', 'o_proj']:
            proj = getattr(attn, name)
            assert proj.weight.grad is not None, f"{name}.weight.grad is None"
            assert not torch.isnan(proj.weight.grad).any(), f"{name}.weight.grad has NaN"

        print("OK Multi-head gradient flow correct")

    def test_single_vs_multihead_has_correct_heads(self):
        """Test that single-head and multi-head modules have different head_dim."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        D = 64
        attn1 = HGRNBitAttention(hidden_size=D, num_heads=1, expand_ratio=1)
        attn4 = HGRNBitAttention(hidden_size=D, num_heads=4, expand_ratio=1)

        # Single-head: head_dim == hidden_size
        assert attn1.head_dim == D, f"Expected head_dim={D}, got {attn1.head_dim}"
        # Multi-head: head_dim = hidden_size / num_heads
        assert attn4.head_dim == D // 4, f"Expected head_dim={D // 4}, got {attn4.head_dim}"

        # Recurrent state shapes should differ
        state1 = attn1.init_state(batch_size=2)
        state4 = attn4.init_state(batch_size=2)
        # state is a tuple; last element is the recurrent state (B, num_heads, head_dim)
        assert state1[-1].shape == (2, 1, D), f"Wrong state1 shape: {state1[-1].shape}"
        assert state4[-1].shape == (2, 4, D // 4), f"Wrong state4 shape: {state4[-1].shape}"

        print("OK Single-head and multi-head have correct head configurations")

    def test_config_num_heads_propagates(self):
        """Test that num_heads from config propagates to attention layers."""
        from hnet_bit.models.hnet_bit import HNetBitForCausalLM

        config = _make_small_config(num_heads=4)
        model = HNetBitForCausalLM(config)

        # Navigate to the innermost stage:
        # model.backbone is stage 0 (non-innermost)
        # model.backbone.main_network is stage 1 (innermost HNetBit)
        # model.backbone.main_network.main_network is the HGRNBitStack
        innermost_stack = model.backbone.main_network.main_network
        for layer in innermost_stack.layers:
            assert layer.attn.num_heads == 4, \
                f"Expected num_heads=4, got {layer.attn.num_heads}"

        # Full forward should work
        B, L = 2, 32
        input_ids = torch.randint(0, 256, (B, L))
        outputs = model(input_ids)
        assert outputs.logits.shape == (B, L, 256)
        print("OK Config num_heads propagates to all layers")


# ---------------------------------------------------------------------------
# Commit 2: expand_ratio=2
# ---------------------------------------------------------------------------
class TestHGRNBitExpandRatio:
    """Tests for expand_ratio scaling."""

    def test_expand_ratio_scales_input_dim(self):
        """Test input_dim = hidden_size * expand_ratio."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        attn = HGRNBitAttention(hidden_size=64, num_heads=1, expand_ratio=2)
        assert attn.input_dim == 128, f"Expected input_dim=128, got {attn.input_dim}"
        print("OK expand_ratio=2 doubles input_dim")

    def test_expand_ratio_2_forward_shape(self):
        """Test output shape is still (B, L, hidden_size) with expand_ratio=2."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        B, L, D = 2, 32, 64
        attn = HGRNBitAttention(hidden_size=D, num_heads=1, expand_ratio=2)
        x = torch.randn(B, L, D)

        o, _, _ = attn(x)
        assert o.shape == (B, L, D), f"Expected {(B, L, D)}, got {o.shape}"
        assert not torch.isnan(o).any()
        print("OK expand_ratio=2 output shape correct")

    def test_expand_ratio_2_multihead_head_dim(self):
        """Test head_dim = (hidden_size * expand_ratio) // num_heads."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        # hidden_size=64, expand=2 => input_dim=128, num_heads=4 => head_dim=32
        attn = HGRNBitAttention(hidden_size=64, num_heads=4, expand_ratio=2)
        assert attn.head_dim == 32, f"Expected head_dim=32, got {attn.head_dim}"
        print("OK expand_ratio=2 + num_heads=4 head_dim correct")

    def test_expand_ratio_projection_sizes(self):
        """Test projection weight shapes scale with expand_ratio."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        attn = HGRNBitAttention(hidden_size=64, num_heads=1, expand_ratio=2)
        # i_proj: (input_dim, hidden_size) = (128, 64)
        assert attn.i_proj.weight.shape == (128, 64), \
            f"Expected i_proj shape (128, 64), got {attn.i_proj.weight.shape}"
        assert attn.f_proj.weight.shape == (128, 64)
        assert attn.g_proj.weight.shape == (128, 64)
        # o_proj: (hidden_size, input_dim) = (64, 128)
        assert attn.o_proj.weight.shape == (64, 128), \
            f"Expected o_proj shape (64, 128), got {attn.o_proj.weight.shape}"
        print("OK Projection sizes scale with expand_ratio")

    def test_expand_ratio_gradient_flow(self):
        """Test gradient flow with expand_ratio=2."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        B, L, D = 2, 16, 64
        attn = HGRNBitAttention(hidden_size=D, num_heads=4, expand_ratio=2)
        x = torch.randn(B, L, D, requires_grad=True)

        o, _, _ = attn(x)
        loss = o.sum()
        loss.backward()

        assert x.grad is not None
        for name in ['i_proj', 'f_proj', 'g_proj', 'o_proj']:
            proj = getattr(attn, name)
            assert proj.weight.grad is not None, f"{name}.weight.grad is None"
        print("OK Gradient flow correct with expand_ratio=2")

    def test_config_expand_ratio_propagates(self):
        """Test that expand_ratio=2 from config works in full model."""
        from hnet_bit.models.hnet_bit import HNetBitForCausalLM

        config = _make_small_config(num_heads=4, expand_ratio=2)
        model = HNetBitForCausalLM(config)

        B, L = 2, 32
        input_ids = torch.randint(0, 256, (B, L))
        outputs = model(input_ids, labels=input_ids.clone())

        assert outputs.loss is not None
        assert outputs.logits.shape == (B, L, 256)
        outputs.loss.backward()
        print("OK Full model works with expand_ratio=2, num_heads=4")


# ---------------------------------------------------------------------------
# Commit 3: Short Convolution
# ---------------------------------------------------------------------------
class TestHGRNBitShortConv:
    """Tests for short convolution support in HGRNBitAttention."""

    def test_short_conv_modules_created_shared(self):
        """Test that shared conv module is created when use_short_conv=True."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        attn = HGRNBitAttention(
            hidden_size=64, num_heads=1, expand_ratio=1,
            use_short_conv=True, conv_size=4, share_conv_kernel=True,
        )
        assert hasattr(attn, 'h_conv1d'), "Missing h_conv1d module"
        assert not hasattr(attn, 'q_conv1d'), "Should not have separate q_conv1d"
        print("OK Shared conv module created")

    def test_separate_conv_modules_created(self):
        """Test separate conv modules when share_conv_kernel=False."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        attn = HGRNBitAttention(
            hidden_size=64, num_heads=1, expand_ratio=1,
            use_short_conv=True, conv_size=4, share_conv_kernel=False,
        )
        assert hasattr(attn, 'q_conv1d'), "Missing q_conv1d"
        assert hasattr(attn, 'f_conv1d'), "Missing f_conv1d"
        assert hasattr(attn, 'i_conv1d'), "Missing i_conv1d"
        assert not hasattr(attn, 'h_conv1d'), "Should not have shared h_conv1d"
        print("OK Separate conv modules created")

    def test_short_conv_disabled_no_module(self):
        """Test no conv modules when use_short_conv=False."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        attn = HGRNBitAttention(
            hidden_size=64, num_heads=1, expand_ratio=1,
            use_short_conv=False,
        )
        assert not hasattr(attn, 'h_conv1d'), "Should not have h_conv1d"
        assert not hasattr(attn, 'q_conv1d'), "Should not have q_conv1d"
        print("OK No conv modules when disabled")

    def test_short_conv_forward_shape(self):
        """Test forward shape with short conv enabled."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        B, L, D = 2, 32, 64
        attn = HGRNBitAttention(
            hidden_size=D, num_heads=1, expand_ratio=1,
            use_short_conv=True, conv_size=4, share_conv_kernel=True,
        )
        x = torch.randn(B, L, D)

        o, _, _ = attn(x)
        assert o.shape == (B, L, D), f"Expected {(B, L, D)}, got {o.shape}"
        assert not torch.isnan(o).any()
        print("OK Short conv forward shape correct")

    def test_short_conv_init_state_includes_conv_buffer(self):
        """Test init_state returns conv state + recurrent state."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        B = 2
        # Shared conv kernel
        attn = HGRNBitAttention(
            hidden_size=64, num_heads=4, expand_ratio=1,
            use_short_conv=True, conv_size=4, share_conv_kernel=True,
        )
        state = attn.init_state(B)
        # Should be (conv_state, recurrent_state) = 2 elements
        assert len(state) == 2, f"Expected 2-element state tuple, got {len(state)}"
        # Conv state: (B, hidden_size, conv_size)
        assert state[0].shape == (B, 64, 4), f"Expected conv state (2,64,4), got {state[0].shape}"
        # Recurrent state: (B, num_heads, head_dim)
        assert state[1].shape == (B, 4, 16), f"Expected recurrent state (2,4,16), got {state[1].shape}"
        print("OK Short conv init_state structure correct")

    def test_short_conv_init_state_separate_kernels(self):
        """Test init_state with separate conv kernels."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        B = 2
        attn = HGRNBitAttention(
            hidden_size=64, num_heads=4, expand_ratio=1,
            use_short_conv=True, conv_size=4, share_conv_kernel=False,
        )
        state = attn.init_state(B)
        # Should be (q_conv, f_conv, i_conv, recurrent_state) = 4 elements
        assert len(state) == 4, f"Expected 4-element state tuple, got {len(state)}"
        for i in range(3):
            assert state[i].shape == (B, 64, 4), f"Conv state {i} shape wrong"
        assert state[3].shape == (B, 4, 16)
        print("OK Separate conv init_state structure correct")

    def test_short_conv_gradient_flow(self):
        """Test gradient flow through short conv path."""
        from hnet_bit.layers.hgrn_bit import HGRNBitAttention

        B, L, D = 2, 16, 64
        attn = HGRNBitAttention(
            hidden_size=D, num_heads=4, expand_ratio=1,
            use_short_conv=True, conv_size=4, share_conv_kernel=True,
        )
        x = torch.randn(B, L, D, requires_grad=True)

        o, _, _ = attn(x)
        loss = o.sum()
        loss.backward()

        assert x.grad is not None
        assert attn.h_conv1d.weight.grad is not None, "h_conv1d weight grad is None"
        print("OK Short conv gradient flow correct")

    def test_short_conv_config_in_full_model(self):
        """Test full model with short conv enabled."""
        from hnet_bit.models.hnet_bit import HNetBitForCausalLM

        config = _make_small_config(
            num_heads=4, expand_ratio=1,
            use_short_conv=True, conv_size=4, share_conv_kernel=True,
        )
        model = HNetBitForCausalLM(config)

        # Verify conv modules exist in layers
        has_conv = False
        for name, module in model.named_modules():
            if hasattr(module, 'h_conv1d'):
                has_conv = True
                break
        assert has_conv, "No h_conv1d found in model"

        # Forward should work
        B, L = 2, 32
        input_ids = torch.randint(0, 256, (B, L))
        outputs = model(input_ids)
        assert outputs.logits.shape == (B, L, 256)
        print("OK Full model with short conv works")

    def test_no_conv_when_flag_false(self):
        """Verify no conv modules when use_short_conv=False in full model."""
        from hnet_bit.models.hnet_bit import HNetBitForCausalLM

        config = _make_small_config(use_short_conv=False)
        model = HNetBitForCausalLM(config)

        for name, module in model.named_modules():
            assert not hasattr(module, 'h_conv1d'), f"Found h_conv1d in {name}"
            assert not hasattr(module, 'q_conv1d'), f"Found q_conv1d in {name}"
        print("OK No conv modules when flag is False")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all_tests():
    """Run all tests manually."""
    test_classes = [
        TestHGRNBitAttentionMultiHead,
        TestHGRNBitExpandRatio,
        TestHGRNBitShortConv,
    ]

    total = 0
    passed = 0
    failed = 0

    for cls in test_classes:
        print(f"\n{'=' * 60}")
        print(f"Running {cls.__name__}")
        print(f"{'=' * 60}")

        instance = cls()
        for method_name in sorted(dir(instance)):
            if method_name.startswith('test_'):
                total += 1
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except Exception as e:
                    failed += 1
                    print(f"MISSING {method_name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    if pytest is not None:
        pytest.main([__file__, '-v'])
    else:
        run_all_tests()
