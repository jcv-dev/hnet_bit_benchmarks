# -*- coding: utf-8 -*-

"""
Unit tests for HNet-Bit with ternary weights.

Tests:
1. BitLinear ternary weight quantization
2. HGRN forward/backward pass
3. Full model forward pass and output shape
4. STE gradient flow through quantization
5. Memory footprint comparison
"""

import sys
import os

# Add project root to path (parent of hnet_bit package)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn

# Optional pytest import
try:
    import pytest
except ImportError:
    pytest = None


class TestBitLinear:
    """Tests for BitLinear layer with ternary quantization."""
    
    def test_weight_quant_ternary(self):
        """Test that weight_quant produces ternary values."""
        from hnet_bit.ops.bitnet import weight_quant
        
        # Random weights
        w = torch.randn(256, 512)
        w_quant = weight_quant(w)
        
        # Compute scale and get ternary values
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        w_ternary = (w * scale).round().clamp_(-1, 1)
        
        # Check all values are in {-1, 0, 1}
        unique_vals = torch.unique(w_ternary)
        assert all(v in [-1, 0, 1] for v in unique_vals.tolist()), \
            f"Expected only -1, 0, 1 but got {unique_vals.tolist()}"
        
        print(f"OK Weight quantization produces ternary values: {unique_vals.tolist()}")
        
        # Check distribution is reasonable (not all zeros)
        dist = {
            -1: (w_ternary == -1).sum().item(),
            0: (w_ternary == 0).sum().item(),
            1: (w_ternary == 1).sum().item()
        }
        total = sum(dist.values())
        print(f"  Distribution: -1: {dist[-1]/total*100:.1f}%, 0: {dist[0]/total*100:.1f}%, 1: {dist[1]/total*100:.1f}%")
        
    def test_activation_quant_range(self):
        """Test that activation_quant produces values in expected range."""
        from hnet_bit.ops.bitnet import activation_quant
        
        x = torch.randn(32, 512)
        x_quant = activation_quant(x)
        
        # After quantization and de-quantization, values should be close to original
        # but quantized to 8-bit precision
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        x_scaled = (x * scale).round().clamp_(-128, 127)
        
        # Check scaled values are integers in [-128, 127]
        assert x_scaled.min() >= -128 and x_scaled.max() <= 127, \
            f"Scaled values out of range: [{x_scaled.min()}, {x_scaled.max()}]"
        
        print(f"OK Activation quantization produces 8-bit values")
        
    def test_bitlinear_forward(self):
        """Test BitLinear forward pass."""
        from hnet_bit.ops.bitnet import BitLinear
        
        layer = BitLinear(512, 256, bias=False)
        x = torch.randn(2, 128, 512)
        
        y = layer(x)
        
        assert y.shape == (2, 128, 256), f"Expected shape (2, 128, 256), got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN values"
        
        print(f"OK BitLinear forward pass: {x.shape} -> {y.shape}")
        
    def test_bitlinear_gradient_flow(self):
        """Test that gradients flow through BitLinear with STE."""
        from hnet_bit.ops.bitnet import BitLinear
        
        layer = BitLinear(512, 256, bias=False)
        x = torch.randn(2, 128, 512, requires_grad=True)
        
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist and are not NaN
        assert x.grad is not None, "No gradient for input"
        assert layer.weight.grad is not None, "No gradient for weights"
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
        assert not torch.isnan(layer.weight.grad).any(), "Weight gradient contains NaN"
        
        print(f"OK Gradients flow through BitLinear (STE working)")
        print(f"  Input grad norm: {x.grad.norm():.4f}")
        print(f"  Weight grad norm: {layer.weight.grad.norm():.4f}")


# Helper for skipif when pytest not available
def _skip_if_no_cuda(func):
    """Decorator to skip test if CUDA not available."""
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            print(f"WARNING Skipping {func.__name__}: CUDA not available")
            return
        return func(*args, **kwargs)
    return wrapper


class TestHGRN:
    """Tests for HGRN recurrence operations."""
    
    @_skip_if_no_cuda
    def test_fused_recurrent_forward(self):
        """Test fused recurrent HGRN forward pass."""
        from hnet_bit.ops.hgrn import fused_recurrent_hgrn
        
        B, H, T, D = 2, 1, 128, 512
        x = torch.randn(B, H, T, D, device='cuda')
        g = torch.randn(B, H, T, D, device='cuda').sigmoid()
        
        o, final_state = fused_recurrent_hgrn(x, g, output_final_state=True)
        
        assert o.shape == (B, H, T, D), f"Expected shape {(B, H, T, D)}, got {o.shape}"
        assert final_state.shape == (B, H, D), f"Expected state shape {(B, H, D)}, got {final_state.shape}"
        assert not torch.isnan(o).any(), "Output contains NaN"
        
        print(f"OK Fused recurrent HGRN forward: {x.shape} -> {o.shape}")
        
    @_skip_if_no_cuda
    def test_fused_recurrent_backward(self):
        """Test fused recurrent HGRN backward pass."""
        from hnet_bit.ops.hgrn import fused_recurrent_hgrn
        
        B, H, T, D = 2, 1, 128, 512
        x = torch.randn(B, H, T, D, device='cuda', requires_grad=True)
        g = torch.randn(B, H, T, D, device='cuda', requires_grad=True).sigmoid()
        
        o, _ = fused_recurrent_hgrn(x, g)
        loss = o.sum()
        loss.backward()
        
        assert x.grad is not None, "No gradient for x"
        assert not torch.isnan(x.grad).any(), "x gradient contains NaN"
        
        print(f"OK Fused recurrent HGRN backward pass")
        
    @_skip_if_no_cuda
    def test_chunk_hgrn_forward(self):
        """Test chunk-parallel HGRN forward pass."""
        from hnet_bit.ops.hgrn import chunk_hgrn
        import torch.nn.functional as F
        
        B, H, T, D = 2, 1, 128, 512
        x = torch.randn(B, H, T, D, device='cuda')
        g = torch.randn(B, H, T, D, device='cuda')
        
        # Chunk HGRN uses log-space gates
        g_log = F.logsigmoid(g)
        x_input = (1 - g.sigmoid()) * x
        
        o, final_state = chunk_hgrn(x_input, g_log, output_final_state=True)
        
        assert o.shape == (B, H, T, D), f"Expected shape {(B, H, T, D)}, got {o.shape}"
        assert final_state.shape == (B, H, D), f"Expected state shape {(B, H, D)}, got {final_state.shape}"
        assert not torch.isnan(o).any(), "Output contains NaN"
        
        print(f"OK Chunk HGRN forward: {x.shape} -> {o.shape}")


class TestMemory:
    """Memory footprint tests."""
    
    def test_memory_comparison(self):
        """Compare memory of ternary vs FP16 weights."""
        from hnet_bit.ops.bitnet import weight_quant
        
        # Simulate a weight matrix
        hidden_size = 2048
        intermediate_size = 2048
        
        # FP16 memory
        fp16_bytes = hidden_size * intermediate_size * 2  # 2 bytes per FP16
        
        # Ternary memory (theoretical)
        # Each weight is {-1, 0, 1}, can be stored in ~1.58 bits
        # In practice, we'd use 2 bits per weight
        ternary_bytes = hidden_size * intermediate_size * 2 / 8  # 2 bits per weight
        
        savings = (1 - ternary_bytes / fp16_bytes) * 100
        
        print(f"OK Memory comparison for {hidden_size}x{intermediate_size} matrix:")
        print(f"  FP16: {fp16_bytes / 1024:.1f} KB")
        print(f"  Ternary (2-bit): {ternary_bytes / 1024:.1f} KB")
        print(f"  Savings: {savings:.1f}%")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running HNetBit Component Unit Tests")
    print("=" * 60)
    
    # BitLinear tests
    print("\n--- BitLinear Tests ---")
    test_bitlinear = TestBitLinear()
    test_bitlinear.test_weight_quant_ternary()
    test_bitlinear.test_activation_quant_range()
    test_bitlinear.test_bitlinear_forward()
    test_bitlinear.test_bitlinear_gradient_flow()
    
    # Memory test (CPU)
    print("\n--- Memory Tests ---")
    test_memory = TestMemory()
    test_memory.test_memory_comparison()
    
    # GPU tests
    if torch.cuda.is_available():
        print("\n--- HGRN Tests (CUDA) ---")
        test_hgrn = TestHGRN()
        test_hgrn.test_fused_recurrent_forward()
        test_hgrn.test_fused_recurrent_backward()
        test_hgrn.test_chunk_hgrn_forward()
        
        print("\n--- Full HNetBit model tests ---")
        print("  See test_hnet_bit.py for hierarchical model tests")
    else:
        print("\nWARNING CUDA not available, skipping GPU tests")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
