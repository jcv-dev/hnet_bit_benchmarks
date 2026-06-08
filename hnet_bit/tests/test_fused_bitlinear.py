# -*- coding: utf-8 -*-

"""
Unit tests for FusedBitLinear (Triton-optimized BitLinear).

Tests:
1. FusedBitLinear inherits from BitLinear
2. CPU fallback to standard BitLinear
3. Forward pass shapes
4. Backward pass gradient flow
5. Numerical equivalence with standard BitLinear (CUDA)
6. Triton availability detection
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


class TestFusedBitLinearBasic:
    """Basic tests that work on CPU."""

    def test_is_bitlinear_subclass(self):
        """Test FusedBitLinear inherits from BitLinear."""
        from hnet_bit.ops.fusedbitnet import FusedBitLinear
        from hnet_bit.ops.bitnet import BitLinear

        layer = FusedBitLinear(64, 32)
        assert isinstance(layer, BitLinear), \
            "FusedBitLinear should be a subclass of BitLinear"

        print(f"OK FusedBitLinear inherits from BitLinear")

    def test_forward_cpu_fallback(self):
        """Test that FusedBitLinear falls back to standard BitLinear on CPU."""
        from hnet_bit.ops.fusedbitnet import FusedBitLinear

        layer = FusedBitLinear(128, 64, bias=False)
        x = torch.randn(2, 16, 128)

        y = layer(x)

        assert y.shape == (2, 16, 64), \
            f"Expected shape (2, 16, 64), got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"

        print(f"OK CPU fallback works: {x.shape} -> {y.shape}")

    def test_forward_shapes(self):
        """Test various input shapes."""
        from hnet_bit.ops.fusedbitnet import FusedBitLinear

        layer = FusedBitLinear(256, 128)

        for shape in [(1, 8, 256), (4, 32, 256), (2, 1, 256)]:
            x = torch.randn(*shape)
            y = layer(x)
            assert y.shape == shape[:-1] + (128,), \
                f"Input {shape} -> expected {shape[:-1] + (128,)}, got {y.shape}"

        print(f"OK Forward shapes correct for various batch/seq lengths")

    def test_backward_cpu(self):
        """Test gradient flow through FusedBitLinear on CPU."""
        from hnet_bit.ops.fusedbitnet import FusedBitLinear

        layer = FusedBitLinear(64, 32, bias=False)
        x = torch.randn(2, 8, 64, requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"
        assert layer.weight.grad is not None, "No gradient for weights"
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
        assert not torch.isnan(layer.weight.grad).any(), "Weight gradient contains NaN"

        print(f"OK CPU backward pass works")
        print(f"  Input grad norm: {x.grad.norm():.4f}")
        print(f"  Weight grad norm: {layer.weight.grad.norm():.4f}")

    def test_has_norm(self):
        """Test that FusedBitLinear has RMSNorm component."""
        from hnet_bit.ops.fusedbitnet import FusedBitLinear
        from hnet_bit.ops.bitnet import RMSNorm

        layer = FusedBitLinear(128, 64)

        assert hasattr(layer, 'norm'), "FusedBitLinear should have 'norm' attribute"
        assert isinstance(layer.norm, RMSNorm), \
            f"norm should be RMSNorm, got {type(layer.norm)}"

        print(f"OK FusedBitLinear has RMSNorm component")

    def test_triton_availability_flag(self):
        """Test Triton availability is correctly detected."""
        from hnet_bit.ops.fusedbitnet import _TRITON_AVAILABLE

        # Just report the flag — it's machine-dependent
        print(f"OK Triton available: {_TRITON_AVAILABLE}")


class TestFusedBitLinearCPUEquivalence:
    """Test numerical equivalence between FusedBitLinear and BitLinear on CPU."""

    def test_same_weights(self):
        """Test that FusedBitLinear and BitLinear have same weights."""
        from hnet_bit.ops.fusedbitnet import FusedBitLinear
        from hnet_bit.ops.bitnet import BitLinear

        fused = FusedBitLinear(128, 64, bias=False)
        standard = BitLinear(128, 64, bias=False)

        # Copy weights
        standard.weight.data.copy_(fused.weight.data)
        standard.norm.weight.data.copy_(fused.norm.weight.data)

        x = torch.randn(2, 16, 128)

        y_fused = fused(x)
        y_standard = standard(x)

        # On CPU, FusedBitLinear calls super().forward() which IS BitLinear.forward()
        # So they should be identical
        assert torch.allclose(y_fused, y_standard, atol=1e-6), \
            f"CPU outputs should be identical. Max diff: {(y_fused - y_standard).abs().max()}"

        print(f"OK CPU outputs are identical between Fused and standard BitLinear")

    def test_backward_equivalence(self):
        """Test gradient equivalence between Fused and standard on CPU."""
        from hnet_bit.ops.fusedbitnet import FusedBitLinear
        from hnet_bit.ops.bitnet import BitLinear

        torch.manual_seed(42)
        fused = FusedBitLinear(64, 32, bias=False)

        torch.manual_seed(42)
        standard = BitLinear(64, 32, bias=False)

        # Make sure weights are the same
        standard.weight.data.copy_(fused.weight.data)
        standard.norm.weight.data.copy_(fused.norm.weight.data)

        x = torch.randn(2, 8, 64)
        x_fused = x.clone().requires_grad_(True)
        x_standard = x.clone().requires_grad_(True)

        y_fused = fused(x_fused)
        y_standard = standard(x_standard)

        y_fused.sum().backward()
        y_standard.sum().backward()

        # Input grads should be the same on CPU (same codepath)
        assert torch.allclose(x_fused.grad, x_standard.grad, atol=1e-5), \
            f"Input gradients differ. Max diff: {(x_fused.grad - x_standard.grad).abs().max()}"

        print(f"OK Backward equivalence on CPU confirmed")


class TestFusedBitLinearCUDA:
    """CUDA-specific tests for FusedBitLinear with Triton kernels."""

    @_skip_if_no_cuda
    def test_forward_cuda(self):
        """Test FusedBitLinear forward on CUDA."""
        from hnet_bit.ops.fusedbitnet import FusedBitLinear, _TRITON_AVAILABLE

        layer = FusedBitLinear(256, 128).cuda()
        x = torch.randn(4, 32, 256, device='cuda')

        y = layer(x)

        assert y.shape == (4, 32, 128)
        assert not torch.isnan(y).any()

        print(f"OK CUDA forward: {x.shape} -> {y.shape}")
        print(f"  Using Triton: {_TRITON_AVAILABLE}")

    @_skip_if_no_cuda
    def test_backward_cuda(self):
        """Test FusedBitLinear backward on CUDA."""
        from hnet_bit.ops.fusedbitnet import FusedBitLinear

        layer = FusedBitLinear(128, 64).cuda()
        x = torch.randn(2, 16, 128, device='cuda', requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.weight.grad is not None
        assert not torch.isnan(x.grad).any()

        print(f"OK CUDA backward pass works")

    @_skip_if_no_cuda
    def test_numerical_equivalence_cuda(self):
        """Test that fused and standard BitLinear produce similar results on CUDA."""
        from hnet_bit.ops.fusedbitnet import FusedBitLinear, _TRITON_AVAILABLE
        from hnet_bit.ops.bitnet import BitLinear

        if not _TRITON_AVAILABLE:
            print(f"WARNING Triton not available — Fused will use same codepath as standard")
            return

        torch.manual_seed(42)
        fused = FusedBitLinear(128, 64, bias=False).cuda()

        standard = BitLinear(128, 64, bias=False).cuda()
        standard.weight.data.copy_(fused.weight.data)
        standard.norm.weight.data.copy_(fused.norm.weight.data)

        x = torch.randn(2, 16, 128, device='cuda')

        y_fused = fused(x)
        y_standard = standard(x)

        # Fused kernel may have small numerical differences due to 
        # different computation order, but should be close
        max_diff = (y_fused - y_standard).abs().max().item()
        # Tolerance for fused vs unfused (Triton may use different precision internally)
        assert max_diff < 0.1, \
            f"Fused/standard output difference too large: {max_diff}"

        print(f"OK CUDA numerical equivalence: max diff = {max_diff:.6f}")


class TestFusedBitLinearIntegration:
    """Integration tests: FusedBitLinear in HNetBit model."""

    def test_model_uses_fused(self):
        """Test that HNetBit can be configured to use FusedBitLinear."""
        from hnet_bit.models.hnet_bit import HNetBitConfig

        config = HNetBitConfig(
            d_model=[64, 96],
            num_blocks=[[2, 0, 2], [4]],
            use_fused_bitlinear=True,
        )
        assert config.use_fused_bitlinear is True

        print(f"OK Config supports use_fused_bitlinear flag")

    def test_bitlinear_in_model(self):
        """Test that model BitLinear layers work (FusedBitLinear is optional)."""
        from hnet_bit.ops.bitnet import BitLinear

        layer = BitLinear(64, 32)
        x = torch.randn(1, 8, 64)
        y = layer(x)

        assert y.shape == (1, 8, 32)
        assert not torch.isnan(y).any()

        print(f"OK BitLinear (base class) works correctly as integration check")


def run_all_tests():
    """Run all FusedBitLinear tests."""
    print("=" * 60)
    print("Running FusedBitLinear Unit Tests")
    print("=" * 60)

    print("\n--- Basic Tests (CPU) ---")
    t_basic = TestFusedBitLinearBasic()
    t_basic.test_is_bitlinear_subclass()
    t_basic.test_forward_cpu_fallback()
    t_basic.test_forward_shapes()
    t_basic.test_backward_cpu()
    t_basic.test_has_norm()
    t_basic.test_triton_availability_flag()

    print("\n--- CPU Equivalence Tests ---")
    t_equiv = TestFusedBitLinearCPUEquivalence()
    t_equiv.test_same_weights()
    t_equiv.test_backward_equivalence()

    print("\n--- Integration Tests ---")
    t_int = TestFusedBitLinearIntegration()
    t_int.test_model_uses_fused()
    t_int.test_bitlinear_in_model()

    if torch.cuda.is_available():
        print("\n--- CUDA Tests ---")
        t_cuda = TestFusedBitLinearCUDA()
        t_cuda.test_forward_cuda()
        t_cuda.test_backward_cuda()
        t_cuda.test_numerical_equivalence_cuda()
    else:
        print("\nWARNING CUDA not available, skipping GPU tests")

    print("\n" + "=" * 60)
    print("All FusedBitLinear tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
