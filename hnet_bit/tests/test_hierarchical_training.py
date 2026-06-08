# -*- coding: utf-8 -*-

"""Tests for hierarchical training features (per-stage LR and load balancing loss)."""

import pytest
import torch
import torch.nn as nn

from hnet_bit.models.hnet_bit import HNetBit, HNetBitConfig
from hnet_bit.training.config import TrainingConfig
from hnet_bit.training.optimizer import build_optimizer_hierarchical
from hnet_bit.training.trainer import Trainer
from hnet_bit.ops.dynamic_chunking import RoutingModuleOutput
from hnet_bit.utils.helpers import apply_optimization_params


class TestApplyOptimizationParams:
    """Test apply_optimization_params utility."""
    
    def test_apply_new_params(self):
        """Test applying params to tensor without existing _optim."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        apply_optimization_params(param, lr_multiplier=2.0, weight_decay=0.01)
        
        assert hasattr(param, '_optim')
        assert param._optim['lr_multiplier'] == 2.0
        assert param._optim['weight_decay'] == 0.01
    
    def test_update_existing_params(self):
        """Test updating existing _optim attribute."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        param._optim = {'lr_multiplier': 1.0}
        
        apply_optimization_params(param, lr_multiplier=2.0, weight_decay=0.01)
        
        assert param._optim['lr_multiplier'] == 2.0
        assert param._optim['weight_decay'] == 0.01


class TestHNetBitLRMultiplier:
    """Test _apply_lr_multiplier on HNetBit model."""
    
    def test_apply_lr_multiplier_2stage(self):
        """Test applying LR multipliers to 2-stage model."""
        config = HNetBitConfig(
            vocab_size=256,
            d_model=[256, 384, 512],
            num_blocks=[[2, 0, 2], [2, 0, 2], [4]],
        )
        model = HNetBit(config, stage_idx=0)
        
        lr_multipliers = [2.0, 1.5, 1.0]
        model._apply_lr_multiplier(lr_multipliers)
        
        # Check that parameters have _optim attribute
        for name, param in model.named_parameters():
            assert hasattr(param, '_optim'), f"Parameter {name} missing _optim"
            assert 'lr_multiplier' in param._optim
            
            # Outer stage should have higher LR
            lr_mult = param._optim['lr_multiplier']
            assert lr_mult in lr_multipliers
    
    def test_apply_lr_multiplier_innermost(self):
        """Test applying LR multipliers to innermost stage."""
        config = HNetBitConfig(
            vocab_size=256,
            d_model=[256],
            num_blocks=[[4]],
        )
        model = HNetBit(config, stage_idx=0)
        
        lr_multipliers = [1.0]
        model._apply_lr_multiplier(lr_multipliers)
        
        for param in model.parameters():
            assert hasattr(param, '_optim')
            assert param._optim['lr_multiplier'] == 1.0


class TestHierarchicalOptimizer:
    """Test hierarchical optimizer builder."""
    
    def test_build_hierarchical_optimizer(self):
        """Test building optimizer with per-stage LRs."""
        config = HNetBitConfig(
            vocab_size=256,
            d_model=[128, 192],
            num_blocks=[[2, 0, 2], [4]],
        )
        model = HNetBit(config, stage_idx=0)
        
        training_config = TrainingConfig(
            learning_rate=1e-3,
            lr_multipliers=[2.0, 1.0],
            weight_decay=0.01,
        )
        
        # Create a mock model with backbone
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = model
                self.lm_head = nn.Linear(128, 256)
        
        mock_model = MockModel()
        optimizer = build_optimizer_hierarchical(mock_model, training_config)
        
        # Should have multiple param groups
        assert len(optimizer.param_groups) > 1
        
        # Check that different LRs exist
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        assert len(set(lrs)) > 1, "Should have different learning rates"
        
        # Max LR should be base_lr * max_lr_multiplier
        max_lr = max(lrs)
        assert max_lr == pytest.approx(training_config.learning_rate * 2.0)
    
    def test_fallback_to_standard_optimizer(self):
        """Test fallback when no lr_multipliers specified."""
        config = HNetBitConfig(
            vocab_size=256,
            d_model=[128],
            num_blocks=[[4]],
        )
        model = HNetBit(config, stage_idx=0)
        
        training_config = TrainingConfig(
            learning_rate=1e-3,
            weight_decay=0.01,
        )
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = model
                self.lm_head = nn.Linear(128, 256)
        
        mock_model = MockModel()
        optimizer = build_optimizer_hierarchical(mock_model, training_config)
        
        # Should still create valid optimizer
        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) >= 1


class TestLoadBalancingLoss:
    """Test load balancing loss computation."""
    
    def test_compute_lb_loss_single_stage(self):
        """Test load balancing loss for single routing output."""
        from hnet_bit.training.trainer import Trainer
        from hnet_bit.training.data import ByteLevelDataset
        
        # Create minimal trainer with proper dataset
        config = HNetBitConfig(vocab_size=256, d_model=[128], num_blocks=[[4]])
        model = HNetBit(config, stage_idx=0)
        
        training_config = TrainingConfig(
            output_dir="./test_checkpoints",
            data_path="./dummy",
            lambda_lb=0.05,
            downsampling_factor=2.5,
            max_seq_length=64,  # Shorter for tests
        )
        
        # Create dataset with enough data
        data = torch.randint(0, 256, (10000,), dtype=torch.long)
        dataset = ByteLevelDataset(data, max_seq_length=64)
        trainer = Trainer(model, training_config, dataset)
        
        # Create mock router output
        batch_size, seq_len = 2, 64
        router_output = RoutingModuleOutput(
            boundary_prob=torch.rand(batch_size, seq_len),
            boundary_mask=torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool),
            selected_probs=torch.rand(batch_size, seq_len),
        )
        
        lb_loss = trainer.compute_load_balancing_loss([router_output])
        
        assert isinstance(lb_loss, (float, torch.Tensor))
        if isinstance(lb_loss, torch.Tensor):
            assert lb_loss.ndim == 0  # Scalar
            assert lb_loss >= 0.0
    
    def test_compute_lb_loss_empty_outputs(self):
        """Test load balancing loss with empty router outputs."""
        from hnet_bit.training.trainer import Trainer
        from hnet_bit.training.data import ByteLevelDataset
        
        config = HNetBitConfig(vocab_size=256, d_model=[128], num_blocks=[[4]])
        model = HNetBit(config, stage_idx=0)
        
        training_config = TrainingConfig(
            output_dir="./test_checkpoints",
            data_path="./dummy",
            lambda_lb=0.05,
            max_seq_length=64,
        )
        
        # Create dataset with enough data
        data = torch.randint(0, 256, (10000,), dtype=torch.long)
        dataset = ByteLevelDataset(data, max_seq_length=64)
        trainer = Trainer(model, training_config, dataset)
        
        lb_loss = trainer.compute_load_balancing_loss([])
        assert lb_loss == 0.0
    
    def test_compute_lb_loss_multiple_stages(self):
        """Test load balancing loss for multiple routing outputs."""
        from hnet_bit.training.trainer import Trainer
        from hnet_bit.training.data import ByteLevelDataset
        
        config = HNetBitConfig(vocab_size=256, d_model=[128], num_blocks=[[4]])
        model = HNetBit(config, stage_idx=0)
        
        training_config = TrainingConfig(
            output_dir="./test_checkpoints",
            data_path="./dummy",
            lambda_lb=0.05,
            max_seq_length=64,
        )
        
        # Create dataset with enough data
        data = torch.randint(0, 256, (10000,), dtype=torch.long)
        dataset = ByteLevelDataset(data, max_seq_length=64)
        trainer = Trainer(model, training_config, dataset)
        
        # Create multiple router outputs
        batch_size, seq_len = 2, 64
        router_outputs = [
            RoutingModuleOutput(
                boundary_prob=torch.rand(batch_size, seq_len),
                boundary_mask=torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool),
                selected_probs=torch.rand(batch_size, seq_len),
            )
            for _ in range(2)
        ]
        
        lb_loss = trainer.compute_load_balancing_loss(router_outputs)
        
        assert isinstance(lb_loss, (float, torch.Tensor))
        if isinstance(lb_loss, torch.Tensor):
            assert lb_loss >= 0.0


class TestTrainingConfigHierarchical:
    """Test TrainingConfig with hierarchical parameters."""
    
    def test_config_with_lr_multipliers(self):
        """Test config accepts lr_multipliers."""
        config = TrainingConfig(
            output_dir="./test",
            data_path="./data",
            lr_multipliers=[2.0, 1.5, 1.0],
            lambda_lb=0.05,
        )
        
        assert config.lr_multipliers == [2.0, 1.5, 1.0]
        assert config.lambda_lb == 0.05
        assert config.downsampling_factor == 2.5  # Default
    
    def test_config_without_hierarchical_params(self):
        """Test config works without hierarchical parameters."""
        config = TrainingConfig(
            output_dir="./test",
            data_path="./data",
        )
        
        assert config.lr_multipliers is None
        assert config.lambda_lb == 0.0
        assert config.downsampling_factor == 2.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
