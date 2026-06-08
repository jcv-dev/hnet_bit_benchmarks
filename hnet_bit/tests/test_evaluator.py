# -*- coding: utf-8 -*-

"""
Tests for evaluation pipeline.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from hnet_bit.training.evaluator import Evaluator, EvaluatorConfig, create_evaluator
from hnet_bit.training.data import ByteLevelDataset, collate_fn
from hnet_bit.utils import ByteTokenizer


class MockModel(nn.Module):
    """Simple mock model for testing."""
    
    def __init__(self, vocab_size=256, hidden_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, labels=None, attention_mask=None):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )
        
        # Return mock output
        class Output:
            pass
        output = Output()
        output.logits = logits
        output.loss = loss
        return output
    
    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        # Simple generation: just append random tokens
        batch_size = input_ids.size(0)
        new_tokens = torch.randint(0, self.vocab_size, (batch_size, max_new_tokens))
        return torch.cat([input_ids, new_tokens.to(input_ids.device)], dim=1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class TestEvaluatorConfig:
    """Tests for EvaluatorConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = EvaluatorConfig()
        assert config.batch_size == 16
        assert config.num_generation_samples == 10
        assert config.profile_memory is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EvaluatorConfig(
            batch_size=32,
            num_generation_samples=5,
            profile_memory=False,
        )
        assert config.batch_size == 32
        assert config.num_generation_samples == 5
        assert config.profile_memory is False


class TestEvaluator:
    """Tests for Evaluator class."""
    
    @pytest.fixture
    def setup(self):
        """Setup model, tokenizer, and data for tests."""
        model = MockModel(vocab_size=256, hidden_size=64)
        tokenizer = ByteTokenizer()
        
        # Create simple test data
        data = torch.randint(0, 256, (1000,), dtype=torch.long)
        dataset = ByteLevelDataset(data, max_seq_length=32)
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_fn,
        )
        
        return model, tokenizer, dataloader
    
    def test_evaluator_creation(self, setup):
        """Test creating an evaluator."""
        model, tokenizer, dataloader = setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvaluatorConfig(output_dir=tmpdir)
            evaluator = Evaluator(model, tokenizer, config, device='cpu')
            
            assert evaluator.model is model
            assert evaluator.tokenizer is tokenizer
    
    def test_evaluate_dataset(self, setup):
        """Test dataset evaluation."""
        model, tokenizer, dataloader = setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvaluatorConfig(output_dir=tmpdir, max_batches=5)
            evaluator = Evaluator(model, tokenizer, config, device='cpu')
            
            metrics = evaluator.evaluate_dataset(dataloader, "test")
            
            assert metrics.loss >= 0
            assert metrics.bpb >= 0
            assert 0 <= metrics.accuracy <= 1
    
    def test_evaluate_generation(self, setup):
        """Test generation evaluation."""
        model, tokenizer, _ = setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvaluatorConfig(
                output_dir=tmpdir,
                num_generation_samples=3,
                generation_max_tokens=20,
            )
            evaluator = Evaluator(model, tokenizer, config, device='cpu')
            
            prompts = ["Hello", "World", "Test"]
            texts, metrics = evaluator.evaluate_generation(prompts)
            
            assert len(texts) == 3
            assert metrics.num_samples == 3
    
    def test_profile_efficiency(self, setup):
        """Test efficiency profiling."""
        model, tokenizer, _ = setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvaluatorConfig(
                output_dir=tmpdir,
                profile_memory=True,
                profile_latency=True,
                latency_warmup_runs=2,
                latency_measure_runs=3,
            )
            evaluator = Evaluator(model, tokenizer, config, device='cpu')
            
            results = evaluator.profile_efficiency()
            
            assert "total_params" in results
            assert "param_memory_mb" in results
            assert "forward_latency_ms" in results
    
    def test_run_full_evaluation(self, setup):
        """Test full evaluation pipeline."""
        model, tokenizer, dataloader = setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvaluatorConfig(
                output_dir=tmpdir,
                max_batches=2,
                num_generation_samples=2,
                latency_warmup_runs=1,
                latency_measure_runs=2,
            )
            evaluator = Evaluator(model, tokenizer, config, device='cpu')
            
            results = evaluator.run_full_evaluation(dataloader)
            
            assert "test" in results
            assert "generation" in results
            assert "efficiency" in results
            
            # Check files were created
            assert (Path(tmpdir) / "evaluation_results.json").exists()


class TestCreateEvaluator:
    """Tests for create_evaluator factory function."""
    
    def test_factory_function(self):
        """Test creating evaluator with factory function."""
        model = MockModel()
        tokenizer = ByteTokenizer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = create_evaluator(
                model=model,
                tokenizer=tokenizer,
                output_dir=tmpdir,
                device='cpu',
                batch_size=8,
            )
            
            assert evaluator.config.batch_size == 8
            assert str(evaluator.output_dir) == tmpdir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
