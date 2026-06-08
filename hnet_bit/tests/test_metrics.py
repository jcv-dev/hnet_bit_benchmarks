# -*- coding: utf-8 -*-

"""
Tests for evaluation metrics module.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from hnet_bit.training.metrics import (
    EvaluationMetrics,
    compute_bpb,
    compute_perplexity,
    compute_accuracy,
    compute_utf8_validity,
    compute_utf8_char_validity,
    compute_memory_footprint,
    MetricsTracker,
)


class TestComputeBPB:
    """Tests for bits per byte computation."""
    
    def test_zero_loss(self):
        """Test BPB with zero loss."""
        assert compute_bpb(0.0) == 0.0
    
    def test_positive_loss(self):
        """Test BPB with positive loss."""
        import math
        loss = 1.0
        expected = loss / math.log(2)
        assert abs(compute_bpb(loss) - expected) < 1e-6
    
    def test_bpb_greater_than_loss(self):
        """BPB should be greater than loss (since ln(2) < 1)."""
        loss = 2.0
        assert compute_bpb(loss) > loss


class TestComputePerplexity:
    """Tests for perplexity computation."""
    
    def test_zero_loss(self):
        """Test perplexity with zero loss."""
        assert compute_perplexity(0.0) == 1.0
    
    def test_positive_loss(self):
        """Test perplexity with positive loss."""
        import math
        loss = 2.0
        expected = math.exp(loss)
        assert abs(compute_perplexity(loss) - expected) < 1e-6
    
    def test_clamp_large_loss(self):
        """Test that large losses are clamped."""
        # Very large loss would overflow exp()
        ppl = compute_perplexity(100.0, clamp_max=1000.0)
        assert ppl <= 1000.0


class TestComputeAccuracy:
    """Tests for accuracy computation."""
    
    def test_perfect_accuracy(self):
        """Test with perfect predictions."""
        logits = torch.zeros(2, 5, 10)  # batch=2, seq=5, vocab=10
        labels = torch.zeros(2, 5, dtype=torch.long)
        
        # Set logit for class 0 to be highest
        logits[:, :, 0] = 10.0
        
        top1, top5 = compute_accuracy(logits, labels)
        assert top1 == 1.0
        assert top5 == 1.0
    
    def test_zero_accuracy(self):
        """Test with completely wrong predictions."""
        logits = torch.zeros(2, 5, 10)
        labels = torch.zeros(2, 5, dtype=torch.long)
        
        # Set logit for class 1 to be highest (but labels are 0)
        logits[:, :, 1] = 10.0
        
        top1, top5 = compute_accuracy(logits, labels)
        assert top1 == 0.0
        # Top-5 should still be 0 since 0 is not in top 5
    
    def test_top5_accuracy(self):
        """Test top-5 accuracy with correct in top 5."""
        logits = torch.zeros(1, 1, 10)
        labels = torch.tensor([[4]], dtype=torch.long)
        
        # Make classes 0-4 highest (correct answer 4 is in top 5)
        logits[0, 0, :5] = torch.tensor([5, 4, 3, 2, 1], dtype=torch.float)
        
        top1, top5 = compute_accuracy(logits, labels)
        assert top1 == 0.0  # Argmax is 0, not 4
        assert top5 == 1.0  # 4 is in top 5
    
    def test_ignore_index(self):
        """Test that ignore_index is properly handled."""
        logits = torch.zeros(1, 5, 10)
        labels = torch.tensor([[0, -100, 0, -100, 0]], dtype=torch.long)
        
        logits[:, :, 0] = 10.0
        
        top1, top5 = compute_accuracy(logits, labels, ignore_index=-100)
        # Only 3 valid positions, all correct
        assert top1 == 1.0


class TestComputeUTF8Validity:
    """Tests for UTF-8 validity computation."""
    
    def test_valid_ascii(self):
        """Test with valid ASCII bytes."""
        sequences = [
            [72, 101, 108, 108, 111],  # "Hello"
            [87, 111, 114, 108, 100],  # "World"
        ]
        assert compute_utf8_validity(sequences) == 1.0
    
    def test_invalid_bytes(self):
        """Test with invalid UTF-8 bytes."""
        sequences = [
            [0xFF, 0xFE],  # Invalid UTF-8
        ]
        assert compute_utf8_validity(sequences) == 0.0
    
    def test_mixed_validity(self):
        """Test with mix of valid and invalid."""
        sequences = [
            [72, 101, 108, 108, 111],  # Valid "Hello"
            [0xFF, 0xFE],  # Invalid
        ]
        assert compute_utf8_validity(sequences) == 0.5
    
    def test_empty_input(self):
        """Test with empty input."""
        assert compute_utf8_validity([]) == 0.0


class TestComputeUTF8CharValidity:
    """Tests for character-level UTF-8 validity."""
    
    def test_all_valid(self):
        """Test all valid ASCII."""
        seq = [72, 101, 108, 108, 111]  # "Hello"
        assert compute_utf8_char_validity(seq) == 1.0
    
    def test_empty(self):
        """Test empty sequence."""
        assert compute_utf8_char_validity([]) == 0.0


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""
    
    def test_default_values(self):
        """Test default initialization."""
        metrics = EvaluationMetrics()
        assert metrics.loss == 0.0
        assert metrics.bpb == 0.0
        assert metrics.accuracy == 0.0
    
    def test_to_dict(self):
        """Test serialization to dict."""
        metrics = EvaluationMetrics(loss=1.5, bpb=2.16, accuracy=0.8)
        d = metrics.to_dict()
        
        assert d["loss"] == 1.5
        assert d["bpb"] == 2.16
        assert d["accuracy"] == 0.8
    
    def test_str_representation(self):
        """Test string representation."""
        metrics = EvaluationMetrics(loss=1.5, bpb=2.16, accuracy=0.8)
        s = str(metrics)
        
        assert "Loss" in s
        assert "BPB" in s
        assert "Accuracy" in s


class TestComputeMemoryFootprint:
    """Tests for memory footprint computation."""
    
    def test_simple_model(self):
        """Test with a simple model."""
        model = nn.Linear(100, 50)
        memory = compute_memory_footprint(model)
        
        assert "total_params" in memory
        assert "trainable_params" in memory
        assert "param_memory_mb" in memory
        
        # Linear(100, 50) has 100*50 + 50 = 5050 params
        assert memory["total_params"] == 5050
    
    def test_frozen_params(self):
        """Test with frozen parameters."""
        model = nn.Linear(100, 50)
        for param in model.parameters():
            param.requires_grad = False
        
        memory = compute_memory_footprint(model)
        assert memory["trainable_params"] == 0


class TestMetricsTracker:
    """Tests for MetricsTracker."""
    
    def test_update_single(self):
        """Test adding single metrics via update."""
        tracker = MetricsTracker()
        tracker.update(1, {"loss": 1.0})
        tracker.update(2, {"loss": 2.0})
        
        assert tracker.get_running_average("loss") == 1.5
    
    def test_update_multiple_metrics(self):
        """Test adding dict of metrics."""
        tracker = MetricsTracker()
        tracker.update(1, {"loss": 1.0, "acc": 0.8})
        tracker.update(2, {"loss": 2.0, "acc": 0.9})
        
        assert tracker.get_running_average("loss") == pytest.approx(1.5)
        assert tracker.get_running_average("acc") == pytest.approx(0.85)
    
    def test_summary(self):
        """Test summary functionality."""
        tracker = MetricsTracker()
        tracker.update(1, {"loss": 1.0})
        tracker.update(2, {"loss": 2.0})
        
        summary = tracker.summary()
        assert "loss" in summary
        assert summary["loss"]["mean"] == 1.5
    
    def test_get_latest(self):
        """Test getting latest value."""
        tracker = MetricsTracker(window_size=10)
        tracker.update(1, {"loss": 1.0})
        tracker.update(2, {"loss": 2.0})
        tracker.update(3, {"loss": 3.0})
        
        assert tracker.get_latest("loss") == 3.0
    
    def test_get_best(self):
        """Test getting best value."""
        tracker = MetricsTracker()
        tracker.update(1, {"loss": 2.0})
        tracker.update(2, {"loss": 1.0})
        tracker.update(3, {"loss": 3.0})
        
        step, value = tracker.get_best("loss", mode='min')
        assert step == 2
        assert value == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
