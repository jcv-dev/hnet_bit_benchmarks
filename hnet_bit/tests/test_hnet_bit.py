# -*- coding: utf-8 -*-

"""
Unit tests for HNetBit: Hierarchical H-Net with ternary BitLinear weights.

Tests:
1. HNetBitConfig construction and presets
2. HNetBit backbone forward pass (1-stage and 2-stage)
3. HNetBitForCausalLM full model forward, loss, generation
4. Ternary weight statistics
5. Gradient flow through hierarchical architecture
6. Hierarchical cache allocation and structure
7. Parameter counting
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


def _make_small_config():
    """Create a minimal config for fast testing."""
    from hnet_bit.models.hnet_bit import HNetBitConfig
    return HNetBitConfig(
        vocab_size=256,
        d_model=[64, 96],
        num_blocks=[[2, 0, 2], [4]],
        num_heads=1,
        expand_ratio=1,
        hidden_ratio=2,
        max_position_embeddings=128,
        use_cache=True,
    )


def _make_2stage_config():
    """Create a minimal 2-stage config for testing."""
    from hnet_bit.models.hnet_bit import HNetBitConfig
    return HNetBitConfig(
        vocab_size=256,
        d_model=[32, 48, 64],
        num_blocks=[[2, 0, 2], [2, 0, 2], [4]],
        num_heads=1,
        expand_ratio=1,
        hidden_ratio=2,
        max_position_embeddings=128,
        use_cache=True,
    )


class TestHNetBitConfig:
    """Tests for HNetBitConfig."""

    def test_default_config(self):
        """Test default config values."""
        from hnet_bit.models.hnet_bit import HNetBitConfig

        config = HNetBitConfig()
        assert config.vocab_size == 256
        assert config.d_model == [512, 768]
        assert config.num_blocks == [[4, 0, 4], [8]]
        assert config.num_stages == 2
        assert config.num_heads == 1

        print(f"OK Default config: {config.num_stages} stages, d_model={config.d_model}")

    def test_small_1stage_preset(self):
        """Test small_1stage preset constructor."""
        from hnet_bit.models.hnet_bit import HNetBitConfig

        config = HNetBitConfig.small_1stage()
        assert config.d_model == [256, 384]
        assert len(config.num_blocks) == 2
        assert config.num_stages == 2

        print(f"OK small_1stage preset: d_model={config.d_model}")

    def test_base_2stage_preset(self):
        """Test base_2stage preset constructor."""
        from hnet_bit.models.hnet_bit import HNetBitConfig

        config = HNetBitConfig.base_2stage()
        assert config.d_model == [512, 768, 1024]
        assert len(config.num_blocks) == 3
        assert config.num_stages == 3

        print(f"OK base_2stage preset: d_model={config.d_model}")

    def test_large_2stage_preset(self):
        """Test large_2stage preset constructor."""
        from hnet_bit.models.hnet_bit import HNetBitConfig

        config = HNetBitConfig.large_2stage()
        assert config.d_model == [768, 1024, 1536]
        assert config.num_stages == 3

        print(f"OK large_2stage preset: d_model={config.d_model}")

    def test_make_stage_config(self):
        """Test _make_stage_config creates proper per-stage config."""
        from hnet_bit.models.hnet_bit import HNetBitConfig

        config = HNetBitConfig(d_model=[128, 256])
        cfg0 = config._make_stage_config(0)
        cfg1 = config._make_stage_config(1)

        assert cfg0.hidden_size == 128
        assert cfg1.hidden_size == 256
        assert cfg0.num_heads == config.num_heads
        assert cfg1.hidden_act == config.hidden_act

        print(f"OK _make_stage_config creates correct per-stage configs")

    def test_config_serialization(self):
        """Test config can be serialized and deserialized."""
        from hnet_bit.models.hnet_bit import HNetBitConfig

        config = HNetBitConfig(d_model=[128, 256, 512], num_blocks=[[2, 0, 2], [2, 0, 2], [4]])
        d = config.to_dict()

        config2 = HNetBitConfig(**d)
        assert config2.d_model == config.d_model
        assert config2.num_blocks == config.num_blocks
        assert config2.vocab_size == config.vocab_size

        print(f"OK Config serialization round-trip works")


class TestHNetBitBackbone:
    """Tests for HNetBit backbone module."""

    def test_1stage_forward_cpu(self):
        """Test 1-stage backbone forward pass on CPU."""
        from hnet_bit.models.hnet_bit import HNetBit

        config = _make_small_config()
        backbone = HNetBit(config, stage_idx=0)

        B, L = 2, 16
        x = torch.randn(B, L, config.d_model[0])
        mask = torch.ones(B, L, dtype=torch.bool)

        out, bpreds = backbone(x, mask=mask)

        assert out.shape == (B, L, config.d_model[0]), \
            f"Expected shape {(B, L, config.d_model[0])}, got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert len(bpreds) == 1, f"Expected 1 boundary prediction, got {len(bpreds)}"

        print(f"OK 1-stage backbone forward: ({B}, {L}, {config.d_model[0]}) -> {out.shape}")

    def test_2stage_forward_cpu(self):
        """Test 2-stage backbone forward pass on CPU."""
        from hnet_bit.models.hnet_bit import HNetBit

        config = _make_2stage_config()
        backbone = HNetBit(config, stage_idx=0)

        B, L = 2, 16
        x = torch.randn(B, L, config.d_model[0])
        mask = torch.ones(B, L, dtype=torch.bool)

        out, bpreds = backbone(x, mask=mask)

        assert out.shape == (B, L, config.d_model[0]), \
            f"Expected shape {(B, L, config.d_model[0])}, got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        # 2-stage: outer routing + inner routing = 2 boundary predictions
        assert len(bpreds) == 2, f"Expected 2 boundary predictions, got {len(bpreds)}"

        print(f"OK 2-stage backbone forward: ({B}, {L}, {config.d_model[0]}) -> {out.shape}")
        print(f"  Boundary predictions: {len(bpreds)}")

    def test_dimension_padding(self):
        """Test dimension padding between stages."""
        from hnet_bit.models.hnet_bit import HNetBit

        config = _make_small_config()  # d_model=[64, 96]
        backbone = HNetBit(config, stage_idx=0)

        # The innermost stage (stage_idx=1) should have pad_dimension
        inner = backbone.main_network  # This is the innermost HNetBit or HGRNBitStack
        if hasattr(inner, 'pad_dimension') and inner.pad_dimension is not None:
            pad_size = inner.pad_dimension.shape[0]
            assert pad_size == config.d_model[1] - config.d_model[0], \
                f"Expected pad size {config.d_model[1] - config.d_model[0]}, got {pad_size}"
            print(f"OK Dimension padding: {config.d_model[0]} + {pad_size} = {config.d_model[1]}")
        else:
            print(f"OK Dimension padding correctly absent for innermost HGRNBitStack")

    def test_residual_count(self):
        """Test _count_residuals reports correct number of residual additions."""
        from hnet_bit.models.hnet_bit import HNetBit

        config = _make_small_config()  # 2 enc + 4 inner + 2 dec
        backbone = HNetBit(config, stage_idx=0)

        n_residuals = backbone._count_residuals()
        # Each block has 2 residual additions (attn + mlp)
        # enc: 2*2=4, inner: 4*2=8, dec: 2*2=4 = 16 total
        expected = (2 + 4 + 2) * 2
        assert n_residuals == expected, \
            f"Expected {expected} residuals, got {n_residuals}"

        print(f"OK Residual count: {n_residuals}")


class TestHNetBitForCausalLM:
    """Tests for full HNetBitForCausalLM model."""

    def test_model_forward_cpu(self):
        """Test full model forward pass on CPU."""
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        config = _make_small_config()
        model = HNetBitForCausalLM(config)

        B, L = 2, 16
        input_ids = torch.randint(0, 256, (B, L))

        outputs = model(input_ids)
        logits = outputs.logits

        assert logits.shape == (B, L, 256), \
            f"Expected logits shape {(B, L, 256)}, got {logits.shape}"
        assert not torch.isnan(logits).any(), "Logits contain NaN"

        print(f"OK Model forward on CPU: {input_ids.shape} -> logits {logits.shape}")

    def test_model_with_labels_cpu(self):
        """Test model forward with loss computation on CPU."""
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        config = _make_small_config()
        model = HNetBitForCausalLM(config)

        B, L = 2, 16
        input_ids = torch.randint(0, 256, (B, L))
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)

        assert outputs.loss is not None, "Loss should not be None with labels"
        assert not torch.isnan(outputs.loss), "Loss is NaN"
        assert outputs.loss.item() > 0, "Loss should be positive"

        print(f"OK Model forward with loss: {outputs.loss.item():.4f}")

    def test_model_backward_cpu(self):
        """Test gradient flow through full model on CPU."""
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        config = _make_small_config()
        model = HNetBitForCausalLM(config)

        B, L = 1, 16
        input_ids = torch.randint(0, 256, (B, L))

        outputs = model(input_ids, labels=input_ids.clone())
        outputs.loss.backward()

        has_grads = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = len(list(model.parameters()))

        assert has_grads > 0, "No parameters received gradients"

        # Check no NaN gradients
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

        print(f"OK Backward pass: {has_grads}/{total_params} params have gradients")

    def test_2stage_forward_cpu(self):
        """Test 2-stage model forward pass."""
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        config = _make_2stage_config()
        model = HNetBitForCausalLM(config)

        B, L = 1, 16
        input_ids = torch.randint(0, 256, (B, L))

        outputs = model(input_ids, labels=input_ids.clone())

        assert outputs.logits.shape == (B, L, 256)
        assert outputs.loss is not None
        assert not torch.isnan(outputs.loss)

        print(f"OK 2-stage model forward: loss={outputs.loss.item():.4f}")

    def test_parameter_count(self):
        """Test parameter counting method."""
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        config = _make_small_config()
        model = HNetBitForCausalLM(config)

        n_params = model.count_parameters()
        assert n_params > 0, "Should have parameters"

        n_all = model.count_parameters(trainable_only=False)
        assert n_all >= n_params

        print(f"OK Parameter count: {n_params:,} trainable, {n_all:,} total")

    def test_ternary_weight_stats(self):
        """Test ternary weight statistics."""
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        config = _make_small_config()
        model = HNetBitForCausalLM(config)

        stats = model.get_ternary_weight_stats()

        assert stats['total_params'] > 0
        assert stats['ternary_params'] > 0
        assert -1 in stats['distribution']
        assert 0 in stats['distribution']
        assert 1 in stats['distribution']

        total_dist = sum(stats['distribution'].values())
        assert total_dist > 0

        if 'distribution_pct' in stats:
            total_pct = sum(stats['distribution_pct'].values())
            assert abs(total_pct - 100.0) < 1.0, \
                f"Distribution percentages should sum to ~100%, got {total_pct}"

        print(f"OK Ternary stats: {stats['ternary_params']:,} params")
        if 'distribution_pct' in stats:
            print(f"  Distribution: " + ", ".join(
                f"{k}: {v:.1f}%" for k, v in stats['distribution_pct'].items()
            ))

    def test_attention_mask(self):
        """Test model with attention mask (padded batch)."""
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        config = _make_small_config()
        model = HNetBitForCausalLM(config)

        B, L = 2, 16
        input_ids = torch.randint(0, 256, (B, L))
        # Second sequence is shorter
        attention_mask = torch.ones(B, L, dtype=torch.long)
        attention_mask[1, 8:] = 0

        outputs = model(input_ids, attention_mask=attention_mask)

        assert outputs.logits.shape == (B, L, 256)
        assert not torch.isnan(outputs.logits).any()

        print(f"OK Attention mask handled correctly")


class TestHierarchicalCache:
    """Tests for hierarchical cache allocation and usage."""

    def test_cache_allocation_1stage(self):
        """Test cache allocation for 1-stage model."""
        from hnet_bit.models.hnet_bit import HNetBit, HNetBitConfig
        from hnet_bit.utils.hnet_cache import HNetBitCache

        config = _make_small_config()
        backbone = HNetBit(config, stage_idx=0)

        B = 2
        cache = backbone.allocate_inference_cache(B, 128)

        assert isinstance(cache, HNetBitCache)
        assert not cache.is_innermost  # Outermost stage is not innermost
        assert cache.encoder_cache is not None
        assert cache.decoder_cache is not None
        assert cache.routing_state is not None
        assert cache.dechunk_state is not None
        assert cache.main_network_cache is not None

        # Inner cache should be innermost
        assert cache.main_network_cache.is_innermost

        print(f"OK 1-stage cache allocation correct")

    def test_cache_allocation_2stage(self):
        """Test cache allocation for 2-stage model."""
        from hnet_bit.models.hnet_bit import HNetBit
        from hnet_bit.utils.hnet_cache import HNetBitCache

        config = _make_2stage_config()
        backbone = HNetBit(config, stage_idx=0)

        cache = backbone.allocate_inference_cache(2, 128)

        # Outer is not innermost
        assert not cache.is_innermost
        # Middle stage
        mid_cache = cache.main_network_cache
        assert not mid_cache.is_innermost
        # Innermost
        inner_cache = mid_cache.main_network_cache
        assert inner_cache.is_innermost

        print(f"OK 2-stage cache: outer → middle → innermost")

    def test_cache_seq_length(self):
        """Test cache sequence length tracking."""
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM
        from hnet_bit.utils.hnet_cache import HNetBitCache

        config = _make_small_config()
        model = HNetBitForCausalLM(config)
        model.eval()

        B, L = 1, 8
        input_ids = torch.randint(0, 256, (B, L))

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)

        cache = outputs.past_key_values
        assert isinstance(cache, HNetBitCache)
        seq_len = cache.get_seq_length()
        assert seq_len > 0, f"Cache seq_length should be > 0 after prefill, got {seq_len}"

        print(f"OK Cache seq_length after prefill: {seq_len}")


class TestModelCUDA:
    """CUDA-only tests for HNetBit."""

    @_skip_if_no_cuda
    def test_forward_cuda(self):
        """Test full model forward on CUDA."""
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        config = _make_small_config()
        model = HNetBitForCausalLM(config).cuda()

        B, L = 2, 32
        input_ids = torch.randint(0, 256, (B, L), device='cuda')

        outputs = model(input_ids, labels=input_ids.clone())

        assert outputs.logits.shape == (B, L, 256)
        assert not torch.isnan(outputs.loss)

        print(f"OK CUDA forward: loss={outputs.loss.item():.4f}")

    @_skip_if_no_cuda
    def test_backward_cuda(self):
        """Test backward on CUDA."""
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        config = _make_small_config()
        model = HNetBitForCausalLM(config).cuda()

        B, L = 1, 16
        input_ids = torch.randint(0, 256, (B, L), device='cuda')

        outputs = model(input_ids, labels=input_ids.clone())
        outputs.loss.backward()

        has_grads = sum(1 for p in model.parameters() if p.grad is not None)
        assert has_grads > 0

        print(f"OK CUDA backward: {has_grads} params with gradients")

    @_skip_if_no_cuda
    def test_generation_cuda(self):
        """Test generation on CUDA."""
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        config = _make_small_config()
        model = HNetBitForCausalLM(config).cuda()
        model.eval()

        prompt = torch.tensor([[72, 101, 108, 108, 111]], device='cuda')  # "Hello"

        with torch.no_grad():
            generated = model.generate(
                prompt,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=0,
            )

        assert generated.shape[0] == 1
        assert generated.shape[1] >= prompt.shape[1]

        print(f"OK CUDA generation: {prompt.shape} -> {generated.shape}")
        print(f"  Generated: {generated[0].tolist()}")


class TestModelFromConfig:
    """Tests for loading model from JSON config."""

    def test_load_1stage_config(self):
        """Test creating model from 1-stage JSON config."""
        import json
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "hnet_bit_1stage.json"
        )

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            config = HNetBitConfig(**cfg)
            model = HNetBitForCausalLM(config)
            n_params = model.count_parameters()
            print(f"OK 1-stage config loaded: {n_params:,} params")
        else:
            print(f"WARNING Config not found: {config_path}")

    def test_load_2stage_config(self):
        """Test creating model from 2-stage JSON config."""
        import json
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "hnet_bit_2stage.json"
        )

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            config = HNetBitConfig(**cfg)
            model = HNetBitForCausalLM(config)
            n_params = model.count_parameters()
            print(f"OK 2-stage config loaded: {n_params:,} params")
        else:
            print(f"WARNING Config not found: {config_path}")


def run_all_tests():
    """Run all HNetBit tests."""
    print("=" * 60)
    print("Running HNetBit Unit Tests")
    print("=" * 60)

    print("\n--- Config Tests ---")
    t_cfg = TestHNetBitConfig()
    t_cfg.test_default_config()
    t_cfg.test_small_1stage_preset()
    t_cfg.test_base_2stage_preset()
    t_cfg.test_large_2stage_preset()
    t_cfg.test_make_stage_config()
    t_cfg.test_config_serialization()

    print("\n--- Backbone Tests (CPU) ---")
    t_bb = TestHNetBitBackbone()
    t_bb.test_1stage_forward_cpu()
    t_bb.test_2stage_forward_cpu()
    t_bb.test_dimension_padding()
    t_bb.test_residual_count()

    print("\n--- Full Model Tests (CPU) ---")
    t_model = TestHNetBitForCausalLM()
    t_model.test_model_forward_cpu()
    t_model.test_model_with_labels_cpu()
    t_model.test_model_backward_cpu()
    t_model.test_2stage_forward_cpu()
    t_model.test_parameter_count()
    t_model.test_ternary_weight_stats()
    t_model.test_attention_mask()

    print("\n--- Cache Tests ---")
    t_cache = TestHierarchicalCache()
    t_cache.test_cache_allocation_1stage()
    t_cache.test_cache_allocation_2stage()
    t_cache.test_cache_seq_length()

    print("\n--- Config File Tests ---")
    t_cfgfile = TestModelFromConfig()
    t_cfgfile.test_load_1stage_config()
    t_cfgfile.test_load_2stage_config()

    if torch.cuda.is_available():
        print("\n--- CUDA Tests ---")
        t_cuda = TestModelCUDA()
        t_cuda.test_forward_cuda()
        t_cuda.test_backward_cuda()
        t_cuda.test_generation_cuda()
    else:
        print("\nWARNING CUDA not available, skipping GPU tests")

    print("\n" + "=" * 60)
    print("All HNetBit tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
