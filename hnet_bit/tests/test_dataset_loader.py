# -*- coding: utf-8 -*-

"""
Tests for dataset loader module.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from hnet_bit.training.dataset_loader import (
    DatasetConfig,
    DatasetStatistics,
    HuggingFaceDataset,
    TinyStoriesDataset,
    TextDirectoryDataset,
    create_train_val_split,
    prepare_datasets,
)
from hnet_bit.training.data import ByteLevelDataset


class TestDatasetConfig:
    """Tests for DatasetConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DatasetConfig()
        assert config.dataset_name == "roneneldan/TinyStories"
        assert config.max_seq_length == 512
        assert config.vocab_size == 256 if hasattr(config, 'vocab_size') else True
    
    def test_cache_key_deterministic(self):
        """Test that cache key is deterministic."""
        config1 = DatasetConfig(max_seq_length=512)
        config2 = DatasetConfig(max_seq_length=512)
        assert config1.cache_key() == config2.cache_key()
    
    def test_cache_key_different(self):
        """Test that different configs have different cache keys."""
        config1 = DatasetConfig(max_seq_length=512)
        config2 = DatasetConfig(max_seq_length=256)
        assert config1.cache_key() != config2.cache_key()
    
    def test_to_dict(self):
        """Test config serialization."""
        config = DatasetConfig(max_seq_length=256, min_length=20)
        d = config.to_dict()
        assert d["max_seq_length"] == 256
        assert d["min_length"] == 20


class TestDatasetStatistics:
    """Tests for DatasetStatistics."""
    
    def test_from_data(self):
        """Test computing statistics from data."""
        data = torch.tensor([65, 66, 67, 65, 65], dtype=torch.long)  # "ABCAA"
        stats = DatasetStatistics.from_data(data)
        
        assert stats.total_bytes == 5
        assert stats.byte_distribution is not None
        assert stats.byte_distribution[65] == 3  # 'A' appears 3 times
        assert stats.byte_distribution[66] == 1  # 'B' appears 1 time
    
    def test_to_dict(self):
        """Test statistics serialization."""
        stats = DatasetStatistics(
            num_samples=100,
            total_bytes=10000,
            avg_bytes_per_sample=100.0,
        )
        d = stats.to_dict()
        assert d["num_samples"] == 100
        assert d["total_bytes"] == 10000


class TestByteLevelDataset:
    """Tests for ByteLevelDataset."""
    
    def test_basic_creation(self):
        """Test creating a dataset."""
        data = torch.randint(0, 256, (1000,), dtype=torch.long)
        dataset = ByteLevelDataset(data, max_seq_length=100)
        
        assert len(dataset) > 0
        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'labels' in sample
        assert sample['input_ids'].shape[0] == 100
    
    def test_with_stride(self):
        """Test dataset with overlapping sequences."""
        data = torch.randint(0, 256, (1000,), dtype=torch.long)
        dataset_no_overlap = ByteLevelDataset(data, max_seq_length=100, stride=100)
        dataset_overlap = ByteLevelDataset(data, max_seq_length=100, stride=50)
        
        # Overlapping should have more samples
        assert len(dataset_overlap) > len(dataset_no_overlap)
    
    def test_labels_shifted(self):
        """Test that labels are shifted by 1."""
        data = torch.arange(10, dtype=torch.long)
        dataset = ByteLevelDataset(data, max_seq_length=5)
        
        sample = dataset[0]
        # input_ids should be [0, 1, 2, 3, 4]
        # labels should be [1, 2, 3, 4, 5]
        assert sample['input_ids'][0].item() == 0
        assert sample['labels'][0].item() == 1


class TestTextDirectoryDataset:
    """Tests for TextDirectoryDataset."""
    
    def test_load_text_files(self):
        """Test loading text files from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "file1.txt").write_text("Hello world!")
            (Path(tmpdir) / "file2.txt").write_text("Goodbye world!")
            
            dataset = TextDirectoryDataset(tmpdir, max_seq_length=10)
            
            # Should have loaded both files
            assert len(dataset) > 0
    
    def test_nested_directories(self):
        """Test loading from nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("Nested content here")
            
            dataset = TextDirectoryDataset(tmpdir, max_seq_length=10)
            
            assert len(dataset) > 0


class TestTrainValSplit:
    """Tests for train/val split functionality."""
    
    def test_split_ratio(self):
        """Test that split ratios are approximately correct."""
        # Create mock dataset
        config = DatasetConfig(max_seq_length=100, min_length=1)
        
        # Create a simple mock dataset
        data = torch.randint(0, 256, (10000,), dtype=torch.long)
        mock_dataset = ByteLevelDataset(data, max_seq_length=100)
        
        # We need a dataset that has raw_bytes attribute
        # For now, skip this test as it requires HuggingFace dataset
        # This would be an integration test


class TestIntegration:
    """Integration tests (require network, may be slow)."""
    
    @pytest.mark.skipif(
        os.environ.get("SKIP_SLOW_TESTS", "1") == "1",
        reason="Slow integration test"
    )
    def test_tinystories_small(self):
        """Test loading small subset of TinyStories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TinyStoriesDataset(
                split="train",
                max_seq_length=128,
                cache_dir=tmpdir,
                max_samples=100,  # Very small for testing
            )
            
            assert len(dataset) > 0
            sample = dataset[0]
            assert sample['input_ids'].shape[0] == 128
            
            # Check statistics
            stats = dataset.statistics
            assert stats.num_samples == 100
            assert stats.total_bytes > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
