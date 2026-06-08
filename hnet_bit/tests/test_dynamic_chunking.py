# -*- coding: utf-8 -*-

"""
Unit tests for dynamic chunking with ternary weights.

Tests:
1. RoutingModuleBit forward pass and output shapes
2. RoutingModuleBit ternary weight properties
3. Boundary probability range and distribution
4. ChunkLayer token selection
5. DeChunkLayer EMA reconstruction
6. End-to-end routing → chunk → dechunk pipeline
7. Step-by-step inference mode
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


class TestRoutingModuleBit:
    """Tests for RoutingModuleBit (boundary prediction with ternary projections)."""

    def test_forward_shape(self):
        """Test output shapes of RoutingModuleBit forward pass."""
        from hnet_bit.ops.dynamic_chunking import RoutingModuleBit

        d_model = 128
        module = RoutingModuleBit(d_model)

        B, L = 2, 32
        x = torch.randn(B, L, d_model)
        mask = torch.ones(B, L, dtype=torch.bool)

        out = module(x, mask=mask)

        assert out.boundary_prob.shape == (B, L, 2), \
            f"Expected boundary_prob shape {(B, L, 2)}, got {out.boundary_prob.shape}"
        assert out.boundary_mask.shape == (B, L), \
            f"Expected boundary_mask shape {(B, L)}, got {out.boundary_mask.shape}"
        assert out.selected_probs.shape == (B, L, 1), \
            f"Expected selected_probs shape {(B, L, 1)}, got {out.selected_probs.shape}"

        print(f"OK RoutingModuleBit forward shapes correct")

    def test_boundary_prob_range(self):
        """Test that boundary probabilities are in [0, 1] and sum to ~1."""
        from hnet_bit.ops.dynamic_chunking import RoutingModuleBit

        d_model = 128
        module = RoutingModuleBit(d_model)

        B, L = 4, 64
        x = torch.randn(B, L, d_model)
        mask = torch.ones(B, L, dtype=torch.bool)

        out = module(x, mask=mask)

        # All probabilities in [0, 1]
        assert out.boundary_prob.min() >= 0.0, \
            f"boundary_prob min < 0: {out.boundary_prob.min()}"
        assert out.boundary_prob.max() <= 1.0, \
            f"boundary_prob max > 1: {out.boundary_prob.max()}"

        # Each position's [p_no_boundary, p_boundary] should sum to 1
        sums = out.boundary_prob.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            f"Boundary probs don't sum to 1: {sums.min()}, {sums.max()}"

        print(f"OK Boundary probabilities in valid range and sum to 1")

    def test_first_token_always_boundary(self):
        """Test that the first token is always predicted as a boundary."""
        from hnet_bit.ops.dynamic_chunking import RoutingModuleBit

        d_model = 128
        module = RoutingModuleBit(d_model)

        B, L = 4, 32
        x = torch.randn(B, L, d_model)
        mask = torch.ones(B, L, dtype=torch.bool)

        out = module(x, mask=mask)

        # First position should always be a boundary
        assert out.boundary_mask[:, 0].all(), \
            "First token should always be a boundary"

        print(f"OK First token is always a boundary")

    def test_uses_nn_linear(self):
        """Test that projections use nn.Linear (full-precision for routing stability)."""
        from hnet_bit.ops.dynamic_chunking import RoutingModuleBit

        module = RoutingModuleBit(128)

        assert isinstance(module.q_proj_layer, nn.Linear), \
            "q_proj_layer should be nn.Linear"
        assert isinstance(module.k_proj_layer, nn.Linear), \
            "k_proj_layer should be nn.Linear"

        print(f"OK RoutingModuleBit uses nn.Linear projections")

    def test_identity_init(self):
        """Test that projections are initialized to near-identity."""
        from hnet_bit.ops.dynamic_chunking import RoutingModuleBit

        d_model = 64
        module = RoutingModuleBit(d_model)

        # After init, weights should be identity
        eye = torch.eye(d_model)
        assert torch.allclose(module.q_proj_layer.weight.data, eye), \
            "q_proj should be initialized to identity"
        assert torch.allclose(module.k_proj_layer.weight.data, eye), \
            "k_proj should be initialized to identity"

        print(f"OK Projections initialized to identity")

    def test_mask_respected(self):
        """Test that invalid positions (mask=False) are not boundaries."""
        from hnet_bit.ops.dynamic_chunking import RoutingModuleBit

        d_model = 128
        module = RoutingModuleBit(d_model)

        B, L = 2, 32
        x = torch.randn(B, L, d_model)
        mask = torch.ones(B, L, dtype=torch.bool)
        mask[:, 16:] = False  # Last 16 tokens invalid

        out = module(x, mask=mask)

        # No boundaries in masked-out region
        assert not out.boundary_mask[:, 16:].any(), \
            "Masked-out positions should not be boundaries"

        print(f"OK Mask is respected: no boundaries in invalid region")

    def test_gradient_flow(self):
        """Test gradients flow through RoutingModuleBit."""
        from hnet_bit.ops.dynamic_chunking import RoutingModuleBit

        d_model = 64
        module = RoutingModuleBit(d_model)

        x = torch.randn(2, 16, d_model, requires_grad=True)
        mask = torch.ones(2, 16, dtype=torch.bool)

        out = module(x, mask=mask)
        # Use selected_probs for backward (differentiable path)
        loss = out.selected_probs.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"

        has_grad = sum(1 for p in module.parameters() if p.grad is not None)
        assert has_grad > 0, "No parameter gradients"

        print(f"OK Gradients flow through RoutingModuleBit ({has_grad} params with grad)")

    def test_step_mode(self):
        """Test step-by-step inference mode."""
        from hnet_bit.ops.dynamic_chunking import RoutingModuleBit

        d_model = 64
        module = RoutingModuleBit(d_model)

        B = 2
        state = module.allocate_inference_cache(B, 128, device=torch.device('cpu'))

        # First step: should always be boundary (hasn't seen tokens)
        h0 = torch.randn(B, 1, d_model)
        out0 = module.step(h0, state)
        assert out0.boundary_mask.all(), "First token step should always be boundary"

        # Second step: may or may not be boundary
        h1 = torch.randn(B, 1, d_model)
        out1 = module.step(h1, state)
        assert out1.boundary_prob.shape == (B, 2), \
            f"Expected step boundary_prob shape {(B, 2)}, got {out1.boundary_prob.shape}"

        print(f"OK Step-by-step inference mode works correctly")


class TestChunkLayer:
    """Tests for ChunkLayer token selection."""

    def test_forward_shapes(self):
        """Test ChunkLayer output shapes."""
        from hnet_bit.ops.dynamic_chunking import ChunkLayer

        layer = ChunkLayer()
        B, L, D = 2, 32, 64
        x = torch.randn(B, L, D)
        mask = torch.ones(B, L, dtype=torch.bool)

        # Create a boundary mask: roughly every 4 tokens
        boundary_mask = torch.zeros(B, L, dtype=torch.bool)
        boundary_mask[:, ::4] = True

        chunked, _, chunk_mask = layer(x, boundary_mask, mask=mask)

        expected_M = boundary_mask.sum(dim=-1).max().item()
        assert chunked.shape[0] == B, "Batch size should match"
        assert chunked.shape[1] == expected_M, \
            f"Expected {expected_M} chunks, got {chunked.shape[1]}"
        assert chunked.shape[2] == D, "Dimension should match"
        assert chunk_mask.shape == (B, expected_M), \
            f"Chunk mask shape mismatch"

        print(f"OK ChunkLayer: ({B}, {L}, {D}) -> ({B}, {expected_M}, {D})")

    def test_selects_boundary_tokens(self):
        """Test that ChunkLayer selects exactly the boundary tokens."""
        from hnet_bit.ops.dynamic_chunking import ChunkLayer

        layer = ChunkLayer()
        B, L, D = 1, 8, 4

        # Make x with distinct values per position
        x = torch.arange(L).float().view(1, L, 1).expand(B, L, D)
        mask = torch.ones(B, L, dtype=torch.bool)

        # Boundaries at positions 0, 3, 7
        boundary_mask = torch.zeros(B, L, dtype=torch.bool)
        boundary_mask[0, [0, 3, 7]] = True

        chunked, _, chunk_mask = layer(x, boundary_mask, mask=mask)

        # Should get 3 chunks with values 0, 3, 7
        assert chunked.shape == (1, 3, D), f"Expected (1, 3, {D}), got {chunked.shape}"
        expected_vals = torch.tensor([0.0, 3.0, 7.0])
        actual_vals = chunked[0, :, 0]
        assert torch.allclose(actual_vals, expected_vals), \
            f"Expected values {expected_vals.tolist()}, got {actual_vals.tolist()}"

        print(f"OK ChunkLayer selects correct boundary tokens")

    def test_variable_boundaries_per_batch(self):
        """Test ChunkLayer with different number of boundaries per batch element."""
        from hnet_bit.ops.dynamic_chunking import ChunkLayer

        layer = ChunkLayer()
        B, L, D = 2, 16, 8
        x = torch.randn(B, L, D)
        mask = torch.ones(B, L, dtype=torch.bool)

        boundary_mask = torch.zeros(B, L, dtype=torch.bool)
        boundary_mask[0, [0, 4, 8]] = True      # 3 boundaries
        boundary_mask[1, [0, 2, 5, 10, 15]] = True  # 5 boundaries

        chunked, _, chunk_mask = layer(x, boundary_mask, mask=mask)

        # Max chunks = 5
        assert chunked.shape[1] == 5
        # First batch has 3 valid, second has 5
        assert chunk_mask[0, :3].all() and not chunk_mask[0, 3:].any()
        assert chunk_mask[1, :5].all()

        print(f"OK Variable boundaries per batch handled correctly")

    def test_step_mode(self):
        """Test ChunkLayer step mode (single token)."""
        from hnet_bit.ops.dynamic_chunking import ChunkLayer

        layer = ChunkLayer()
        B = 4
        D = 32
        h = torch.randn(B, 1, D)

        # Some are boundaries, some are not
        boundary_mask = torch.tensor([True, False, True, False])

        selected = layer.step(h, boundary_mask)

        assert selected.shape[0] == boundary_mask.sum().item(), \
            f"Expected {boundary_mask.sum().item()} selected, got {selected.shape[0]}"

        print(f"OK ChunkLayer step mode selects {selected.shape[0]} of {B}")


class TestDeChunkLayer:
    """Tests for DeChunkLayer EMA reconstruction."""

    def test_forward_shape(self):
        """Test DeChunkLayer output shape matches original sequence length."""
        from hnet_bit.ops.dynamic_chunking import DeChunkLayer

        d_model = 64
        layer = DeChunkLayer(d_model)

        B, L = 2, 32
        M = 8  # Number of chunks
        D = d_model

        # Simulated chunked representations
        hidden_states = torch.randn(B, M, D)
        boundary_mask = torch.zeros(B, L, dtype=torch.bool)
        boundary_mask[:, ::4] = True  # Boundary every 4 tokens
        boundary_prob = torch.stack([
            torch.full((B, L), 0.5),
            torch.full((B, L), 0.5),
        ], dim=-1)

        out = layer(hidden_states, boundary_mask, boundary_prob)

        assert out.shape == (B, L, D), \
            f"Expected shape {(B, L, D)}, got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"

        print(f"OK DeChunkLayer: ({B}, {M}, {D}) -> ({B}, {L}, {D})")

    def test_boundary_positions_get_chunk_values(self):
        """Test that at boundary positions, output is dominated by chunk values."""
        from hnet_bit.ops.dynamic_chunking import DeChunkLayer

        d_model = 16
        layer = DeChunkLayer(d_model)

        B, L = 1, 8
        M = 2

        chunk_vals = torch.ones(B, M, d_model) * 10.0
        boundary_mask = torch.zeros(B, L, dtype=torch.bool)
        boundary_mask[0, [0, 4]] = True

        # High boundary prob = chunk value dominates
        boundary_prob = torch.zeros(B, L, 2)
        boundary_prob[:, :, 0] = 0.01  # Low no-boundary prob
        boundary_prob[:, :, 1] = 0.99  # High boundary prob

        out = layer(chunk_vals, boundary_mask, boundary_prob)

        # First two values should be very close to 10 (p=0.99 * chunk + 0.01 * prev)
        assert (out[0, 0] - 10.0).abs().max() < 0.5, \
            "First position should be close to chunk value"

        print(f"OK Boundary positions receive chunk values")

    def test_ema_smoothing(self):
        """Test that EMA provides smooth interpolation between chunks."""
        from hnet_bit.ops.dynamic_chunking import DeChunkLayer

        d_model = 8
        layer = DeChunkLayer(d_model)

        B, L = 1, 8
        M = 2

        # Chunk 0 = all 1s, Chunk 1 = all 10s
        chunk_vals = torch.zeros(B, M, d_model)
        chunk_vals[0, 0] = 1.0
        chunk_vals[0, 1] = 10.0

        boundary_mask = torch.zeros(B, L, dtype=torch.bool)
        boundary_mask[0, [0, 4]] = True

        # Low boundary prob between boundaries = smooth EMA
        boundary_prob = torch.zeros(B, L, 2)
        boundary_prob[:, :, 0] = 0.7  # Most prob on no-boundary
        boundary_prob[:, :, 1] = 0.3  # Some on boundary
        # But force position 0 and 4 to be strong boundaries
        boundary_prob[:, 0, 0] = 0.01
        boundary_prob[:, 0, 1] = 0.99
        boundary_prob[:, 4, 0] = 0.01
        boundary_prob[:, 4, 1] = 0.99

        out = layer(chunk_vals, boundary_mask, boundary_prob)

        # Between boundaries (positions 1-3), values should smoothly interpolate from chunk 0
        # Position 4+ should jump back toward chunk 1 (=10.0)
        assert out[0, 4, 0] > out[0, 3, 0], \
            "Value should jump at boundary position"

        print(f"OK EMA provides smooth interpolation between chunks")

    def test_step_mode(self):
        """Test DeChunkLayer step-by-step inference."""
        from hnet_bit.ops.dynamic_chunking import DeChunkLayer

        d_model = 32
        layer = DeChunkLayer(d_model)

        B = 2
        state = layer.allocate_inference_cache(B, 128, device=torch.device('cpu'))

        # Step with boundary
        h = torch.randn(2, 1, d_model)  # All are boundaries
        boundary_mask = torch.ones(B, dtype=torch.bool)
        boundary_prob = torch.zeros(B, 2)
        boundary_prob[:, 0] = 0.1
        boundary_prob[:, 1] = 0.9

        out = layer.step(h, boundary_mask, boundary_prob, state)

        assert out.shape == (B, 1, d_model), \
            f"Expected shape {(B, 1, d_model)}, got {out.shape}"
        assert not torch.isnan(out).any(), "Step output contains NaN"

        print(f"OK DeChunkLayer step mode works")


class TestEndToEndPipeline:
    """End-to-end tests: routing → chunk → process → dechunk."""

    def test_full_pipeline(self):
        """Test complete dynamic chunking pipeline."""
        from hnet_bit.ops.dynamic_chunking import (
            RoutingModuleBit, ChunkLayer, DeChunkLayer,
        )

        d_model = 64
        B, L = 2, 32

        routing = RoutingModuleBit(d_model)
        chunk_layer = ChunkLayer()
        dechunk_layer = DeChunkLayer(d_model)

        x = torch.randn(B, L, d_model)
        mask = torch.ones(B, L, dtype=torch.bool)

        # 1. Routing: predict boundaries
        routing_out = routing(x, mask=mask)

        # 2. Chunk: select boundary tokens
        chunked, _, chunk_mask = chunk_layer(x, routing_out.boundary_mask, mask=mask)

        # 3. "Process" chunked (identity for test)
        processed = chunked

        # 4. Dechunk: reconstruct sequence
        reconstructed = dechunk_layer(
            processed,
            routing_out.boundary_mask,
            routing_out.boundary_prob,
            mask=mask,
        )

        assert reconstructed.shape == (B, L, d_model), \
            f"Reconstructed shape should match input: {reconstructed.shape}"
        assert not torch.isnan(reconstructed).any(), "Reconstructed contains NaN"

        n_boundaries = routing_out.boundary_mask.sum().item()
        print(f"OK Full pipeline: ({B}, {L}, {d_model}) -> {n_boundaries} chunks -> ({B}, {L}, {d_model})")

    def test_pipeline_gradient_flow(self):
        """Test gradients flow through the full pipeline."""
        from hnet_bit.ops.dynamic_chunking import (
            RoutingModuleBit, ChunkLayer, DeChunkLayer,
        )

        d_model = 64
        B, L = 2, 16

        routing = RoutingModuleBit(d_model)
        chunk_layer = ChunkLayer()
        dechunk_layer = DeChunkLayer(d_model)

        x = torch.randn(B, L, d_model, requires_grad=True)
        mask = torch.ones(B, L, dtype=torch.bool)

        routing_out = routing(x, mask=mask)
        chunked, _, chunk_mask = chunk_layer(x, routing_out.boundary_mask, mask=mask)
        reconstructed = dechunk_layer(
            chunked,
            routing_out.boundary_mask,
            routing_out.boundary_prob,
            mask=mask,
        )

        loss = reconstructed.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"

        print(f"OK Pipeline gradients flow correctly")


def run_all_tests():
    """Run all dynamic chunking tests."""
    print("=" * 60)
    print("Running Dynamic Chunking Unit Tests")
    print("=" * 60)

    print("\n--- RoutingModuleBit Tests ---")
    t_routing = TestRoutingModuleBit()
    t_routing.test_forward_shape()
    t_routing.test_boundary_prob_range()
    t_routing.test_first_token_always_boundary()
    t_routing.test_uses_bitlinear()
    t_routing.test_identity_init()
    t_routing.test_mask_respected()
    t_routing.test_gradient_flow()
    t_routing.test_step_mode()

    print("\n--- ChunkLayer Tests ---")
    t_chunk = TestChunkLayer()
    t_chunk.test_forward_shapes()
    t_chunk.test_selects_boundary_tokens()
    t_chunk.test_variable_boundaries_per_batch()
    t_chunk.test_step_mode()

    print("\n--- DeChunkLayer Tests ---")
    t_dechunk = TestDeChunkLayer()
    t_dechunk.test_forward_shape()
    t_dechunk.test_boundary_positions_get_chunk_values()
    t_dechunk.test_ema_smoothing()
    t_dechunk.test_step_mode()

    print("\n--- End-to-End Pipeline Tests ---")
    t_pipe = TestEndToEndPipeline()
    t_pipe.test_full_pipeline()
    t_pipe.test_pipeline_gradient_flow()

    print("\n" + "=" * 60)
    print("All dynamic chunking tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
