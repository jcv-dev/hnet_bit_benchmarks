# -*- coding: utf-8 -*-

"""
Unit tests for packed sequences (cu_seqlens) support.

Tests:
1. RoutingModuleBit with cu_seqlens
2. ChunkLayer with cu_seqlens
3. DeChunkLayer with cu_seqlens
4. pack_sequences utility
5. End-to-end packed pipeline
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


class TestPackedRoutingModule:
    """Tests for RoutingModuleBit with cu_seqlens (packed mode)."""

    def test_cu_seqlens_boundaries_forced(self):
        """Test that boundary_prob is forced to 1.0 at sequence starts."""
        from hnet_bit.ops.dynamic_chunking import RoutingModuleBit

        d_model = 64
        module = RoutingModuleBit(d_model)

        # Two sequences of length 30 each, packed into T=60
        T = 60
        x = torch.randn(T, d_model)
        cu_seqlens = torch.tensor([0, 30, 60], dtype=torch.long)

        out = module(x, cu_seqlens=cu_seqlens)

        # First tokens of each sequence should be boundaries
        assert out.boundary_mask[0].item(), "First token of seq 0 should be boundary"
        assert out.boundary_mask[30].item(), "First token of seq 1 should be boundary"
        # boundary_prob at these positions should be ~1.0
        assert out.boundary_prob[0, 1].item() >= 0.99, "boundary_prob at pos 0 should be ~1.0"
        assert out.boundary_prob[30, 1].item() >= 0.99, "boundary_prob at pos 30 should be ~1.0"
        print("OK cu_seqlens boundaries forced correctly")

    def test_cu_seqlens_output_is_flat(self):
        """Test that packed mode output is flat (T, ...)."""
        from hnet_bit.ops.dynamic_chunking import RoutingModuleBit

        d_model = 64
        module = RoutingModuleBit(d_model)

        T = 40
        x = torch.randn(T, d_model)
        cu_seqlens = torch.tensor([0, 20, 40], dtype=torch.long)

        out = module(x, cu_seqlens=cu_seqlens)

        assert out.boundary_prob.shape == (T, 2), \
            f"Expected boundary_prob shape (40, 2), got {out.boundary_prob.shape}"
        assert out.boundary_mask.shape == (T,), \
            f"Expected boundary_mask shape (40,), got {out.boundary_mask.shape}"
        assert out.selected_probs.shape == (T, 1), \
            f"Expected selected_probs shape (40, 1), got {out.selected_probs.shape}"
        print("OK Packed output shapes are flat")

    def test_packed_vs_padded_boundary_probs(self):
        """Test that packed and padded modes produce similar boundary probs."""
        from hnet_bit.ops.dynamic_chunking import RoutingModuleBit

        d_model = 64
        torch.manual_seed(42)
        module = RoutingModuleBit(d_model)

        L = 16
        x1 = torch.randn(1, L, d_model)
        x2 = torch.randn(1, L, d_model)

        # Padded mode
        mask = torch.ones(1, L, dtype=torch.bool)
        out1 = module(x1, mask=mask)
        out2 = module(x2, mask=mask)

        # Packed mode
        x_packed = torch.cat([x1.squeeze(0), x2.squeeze(0)], dim=0)  # (2L, D)
        cu_seqlens = torch.tensor([0, L, 2 * L], dtype=torch.long)
        out_packed = module(x_packed, cu_seqlens=cu_seqlens)

        # Boundary probs for seq 1 should match between packed and padded
        bp_padded_seq1 = out1.boundary_prob[0]  # (L, 2)
        bp_packed_seq1 = out_packed.boundary_prob[:L]  # (L, 2)
        assert torch.allclose(bp_padded_seq1, bp_packed_seq1, atol=1e-5), \
            "Boundary probs for seq 1 differ between packed and padded"
        print("OK Packed vs padded boundary probs match")


class TestPackedChunkLayer:
    """Tests for ChunkLayer with cu_seqlens (packed mode)."""

    def test_packed_chunk_output_shape(self):
        """Test packed ChunkLayer returns flat (M, D) and cu_seqlens."""
        from hnet_bit.ops.dynamic_chunking import ChunkLayer

        layer = ChunkLayer()
        T, D = 40, 32
        x = torch.randn(T, D)

        # Boundary at every 4th position
        boundary_mask = torch.zeros(T, dtype=torch.bool)
        boundary_mask[::4] = True  # 10 boundaries

        cu_seqlens = torch.tensor([0, 20, 40], dtype=torch.long)

        next_hidden_states, next_cu_seqlens, next_mask = layer(
            x, boundary_mask, cu_seqlens=cu_seqlens,
        )

        M = boundary_mask.sum().item()
        assert next_hidden_states.shape == (M, D), \
            f"Expected ({M}, {D}), got {next_hidden_states.shape}"
        assert next_cu_seqlens is not None, "next_cu_seqlens should not be None"
        assert next_mask is None, "next_mask should be None in packed mode"
        assert len(next_cu_seqlens) == 3, f"Expected 3 elements, got {len(next_cu_seqlens)}"
        assert next_cu_seqlens[0] == 0
        assert next_cu_seqlens[-1] == M
        print("OK Packed ChunkLayer output shape correct")

    def test_packed_chunk_cu_seqlens_correctness(self):
        """Test that next_cu_seqlens correctly reflects per-sequence boundary counts."""
        from hnet_bit.ops.dynamic_chunking import ChunkLayer

        layer = ChunkLayer()
        T, D = 20, 16

        x = torch.randn(T, D)
        cu_seqlens = torch.tensor([0, 10, 20], dtype=torch.long)

        # Seq 0: boundaries at 0, 3, 7 = 3 boundaries
        # Seq 1: boundaries at 10, 14 = 2 boundaries
        boundary_mask = torch.zeros(T, dtype=torch.bool)
        boundary_mask[[0, 3, 7, 10, 14]] = True

        _, next_cu_seqlens, _ = layer(x, boundary_mask, cu_seqlens=cu_seqlens)

        assert next_cu_seqlens[0] == 0
        assert next_cu_seqlens[1] == 3, f"Seq 0 should have 3 boundaries, got {next_cu_seqlens[1]}"
        assert next_cu_seqlens[2] == 5, f"Total should be 5 boundaries, got {next_cu_seqlens[2]}"
        print("OK Packed ChunkLayer cu_seqlens correctness verified")


class TestPackedDeChunkLayer:
    """Tests for DeChunkLayer with cu_seqlens (packed mode)."""

    def test_packed_dechunk_output_shape(self):
        """Test packed DeChunkLayer returns flat (T, D)."""
        from hnet_bit.ops.dynamic_chunking import DeChunkLayer

        d_model = 32
        layer = DeChunkLayer(d_model)

        T = 20
        M = 5  # 5 boundary tokens
        cu_seqlens = torch.tensor([0, 10, 20], dtype=torch.long)

        # Simulated chunked output
        hidden_states = torch.randn(M, d_model)
        boundary_mask = torch.zeros(T, dtype=torch.bool)
        boundary_mask[[0, 3, 7, 10, 15]] = True  # Same boundaries as chunk test
        boundary_prob = torch.stack([
            torch.full((T,), 0.5),
            torch.full((T,), 0.5),
        ], dim=-1)

        out = layer(
            hidden_states, boundary_mask, boundary_prob,
            cu_seqlens=cu_seqlens,
        )

        assert out.shape == (T, d_model), f"Expected ({T}, {d_model}), got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        print("OK Packed DeChunkLayer output shape correct")

    def test_packed_dechunk_boundary_values(self):
        """Test that boundary positions receive chunk-derived values."""
        from hnet_bit.ops.dynamic_chunking import DeChunkLayer

        d_model = 16
        layer = DeChunkLayer(d_model)

        T = 10
        cu_seqlens = torch.tensor([0, 5, 10], dtype=torch.long)

        # Two boundaries, one per sequence
        boundary_mask = torch.zeros(T, dtype=torch.bool)
        boundary_mask[0] = True  # Seq 0 start
        boundary_mask[5] = True  # Seq 1 start

        hidden_states = torch.ones(2, d_model) * 10.0  # Clear signal

        # High boundary prob so EMA strongly follows chunk values
        boundary_prob = torch.zeros(T, 2)
        boundary_prob[:, 0] = 0.1
        boundary_prob[:, 1] = 0.9

        out = layer(hidden_states, boundary_mask, boundary_prob, cu_seqlens=cu_seqlens)

        # At boundary positions (0, 5), values should be close to chunk values
        assert out[0].mean().item() > 5.0, "Boundary pos 0 should reflect chunk value"
        assert out[5].mean().item() > 5.0, "Boundary pos 5 should reflect chunk value"
        print("OK Packed DeChunkLayer boundary values correct")


class TestPackSequencesUtil:
    """Tests for pack_sequences utility function."""

    def test_pack_sequences_output_lengths(self):
        """Test that packed batches respect max_seq_len."""
        from hnet_bit.training.data import pack_sequences

        sequences = [torch.randint(0, 256, (20,)) for _ in range(10)]
        packed = pack_sequences(sequences, max_seq_len=50)

        assert len(packed) > 0, "Should produce at least one packed batch"
        for batch in packed:
            T = batch['input_ids'].shape[0]
            assert T <= 50, f"Packed batch exceeds max_seq_len: {T} > 50"
            assert batch['cu_seqlens'][-1] == T
            assert batch['cu_seqlens'][0] == 0
        print("OK pack_sequences respects max_seq_len")

    def test_pack_sequences_no_overflow(self):
        """Test that no packed batch exceeds max_seq_len."""
        from hnet_bit.training.data import pack_sequences

        sequences = [torch.randint(0, 256, (l,)) for l in [5, 10, 15, 20, 25, 30]]
        packed = pack_sequences(sequences, max_seq_len=30)

        for i, batch in enumerate(packed):
            T = batch['input_ids'].shape[0]
            assert T <= 30, f"Batch {i} overflow: {T} > 30"
        print("OK No overflow in pack_sequences")

    def test_pack_sequences_preserves_content(self):
        """Test that unpacking via cu_seqlens recovers original tokens."""
        from hnet_bit.training.data import pack_sequences

        seq1 = torch.tensor([1, 2, 3, 4, 5])
        seq2 = torch.tensor([10, 20, 30, 40])
        packed = pack_sequences([seq1, seq2], max_seq_len=100)

        assert len(packed) == 1, "Should fit in one batch"
        batch = packed[0]
        cu = batch['cu_seqlens']
        ids = batch['input_ids']
        labels = batch['labels']

        # Reconstruct: seq1 input is [1,2,3,4], label is [2,3,4,5]
        recon_ids_1 = ids[cu[0]:cu[1]]
        recon_labels_1 = labels[cu[0]:cu[1]]
        assert torch.equal(recon_ids_1, torch.tensor([1, 2, 3, 4]))
        assert torch.equal(recon_labels_1, torch.tensor([2, 3, 4, 5]))

        # seq2 input is [10,20,30], label is [20,30,40]
        recon_ids_2 = ids[cu[1]:cu[2]]
        recon_labels_2 = labels[cu[1]:cu[2]]
        assert torch.equal(recon_ids_2, torch.tensor([10, 20, 30]))
        assert torch.equal(recon_labels_2, torch.tensor([20, 30, 40]))
        print("OK pack_sequences preserves content")

    def test_pack_sequences_skips_short(self):
        """Test that sequences shorter than 2 tokens are skipped."""
        from hnet_bit.training.data import pack_sequences

        sequences = [torch.tensor([1]), torch.tensor([1, 2, 3]), torch.tensor([])]
        packed = pack_sequences(sequences, max_seq_len=100)

        # Only the (1,2,3) sequence should survive
        assert len(packed) == 1
        assert packed[0]['input_ids'].shape[0] == 2  # [1, 2]
        assert packed[0]['labels'].shape[0] == 2  # [2, 3]
        print("OK pack_sequences skips short sequences")


class TestPackedEndToEnd:
    """End-to-end tests for packed pipeline."""

    def test_packed_pipeline_no_error(self):
        """Test full packed routing→chunk→dechunk pipeline."""
        from hnet_bit.ops.dynamic_chunking import (
            RoutingModuleBit, ChunkLayer, DeChunkLayer,
        )

        d_model = 64
        T = 40
        cu_seqlens = torch.tensor([0, 20, 40], dtype=torch.long)

        routing = RoutingModuleBit(d_model)
        chunk_layer = ChunkLayer()
        dechunk_layer = DeChunkLayer(d_model)

        x = torch.randn(T, d_model)

        # Route
        routing_out = routing(x, cu_seqlens=cu_seqlens)

        # Chunk
        chunked, next_cu_seqlens, _ = chunk_layer(
            x, routing_out.boundary_mask, cu_seqlens=cu_seqlens,
        )

        # Dechunk
        reconstructed = dechunk_layer(
            chunked, routing_out.boundary_mask, routing_out.boundary_prob,
            cu_seqlens=cu_seqlens,
        )

        assert reconstructed.shape == (T, d_model), \
            f"Expected ({T}, {d_model}), got {reconstructed.shape}"
        assert not torch.isnan(reconstructed).any()
        print("OK Packed end-to-end pipeline works")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all_tests():
    """Run all tests manually."""
    test_classes = [
        TestPackedRoutingModule,
        TestPackedChunkLayer,
        TestPackedDeChunkLayer,
        TestPackSequencesUtil,
        TestPackedEndToEnd,
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
