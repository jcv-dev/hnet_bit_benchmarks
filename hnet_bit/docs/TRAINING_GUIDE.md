# Training Guide

This guide covers standalone training with the `hnet_bit` package (`python -m hnet_bit.train`). For the Spanish Billion Words benchmark that compares this model against transformer and MatMulFree baselines, see the repository root `README.md` and the `train_spanish.py` script.

This guide covers how to train the Simplified SLM models on your own data.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Configuration](#model-configuration)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Generation](#generation)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Installation

```bash
# Install the package
pip install -e .

# Install training dependencies
pip install datasets wandb tensorboard matplotlib seaborn

# For CUDA support
pip install triton>=2.2
```

### Hardware Requirements

- **Minimum**: 8GB GPU VRAM (for flat model with batch_size=8)
- **Recommended**: 24GB GPU VRAM (for 2-stage hierarchical model)
- **CPU Training**: Supported but significantly slower

---

## Quick Start

The fastest way to start training:

```bash
# 1. Prepare TinyStories dataset
python scripts/prepare_tinystories.py --output_dir ./data/tinystories

# 2. Run training experiment
python scripts/run_experiment.py --config configs/training_flat_tinystories.json
```

This will:
1. Download and prepare the TinyStories dataset
2. Train the flat SimplifiedSLM model
3. Evaluate on the test set
4. Save checkpoints and logs

---

## Dataset Preparation

### TinyStories (Recommended for Testing)

```bash
# Full dataset (~200MB)
python scripts/prepare_tinystories.py \
    --output_dir ./data/tinystories \
    --max_seq_length 512

# Small subset for debugging
python scripts/prepare_tinystories.py \
    --output_dir ./data/tinystories_small \
    --max_seq_length 256 \
    --max_samples 10000
```

### Custom Text Files

```python
from hnet_bit.training import TextDirectoryDataset, collate_fn
from torch.utils.data import DataLoader

# Load from directory of .txt files
dataset = TextDirectoryDataset(
    directory="./my_texts",
    max_seq_length=512,
)

dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
```

### HuggingFace Datasets

```python
from hnet_bit.training import DatasetConfig, HuggingFaceDataset

# WikiText
config = DatasetConfig(
    dataset_name="wikitext",
    dataset_config="wikitext-103-v1",
    split="train",
    max_seq_length=512,
)
dataset = HuggingFaceDataset(config)

# Any text dataset
config = DatasetConfig(
    dataset_name="your/dataset",
    text_column="text",  # Column containing text
    max_seq_length=512,
)
```

### Dataset Statistics

After preparation, check statistics:

```python
from hnet_bit.training import print_dataset_stats

print_dataset_stats(dataset)
# Output:
# Dataset Statistics
# ==================
# Number of original samples: 100,000
# Total bytes: 45,678,901
# Training sequences: 89,215
# ...
```

---

## Model Configuration

### Available Models

| Model | Config File | Parameters | Description |
|-------|-------------|------------|-------------|
| Flat (Base) | `slm_base.json` | ~25M | SimplifiedSLM, 6 layers, 512 hidden |
| 1-Stage | `hnet_bit_1stage.json` | ~35M | Hierarchical, 256→384 |
| 2-Stage | `hnet_bit_2stage.json` | ~80M | Hierarchical, 512→768→1024 |

### Creating Custom Config

```json
{
    "model_type": "hnet_bit",
    "vocab_size": 256,
    "d_model": [384, 512, 768],
    "num_blocks": [[3, 0, 3], [3, 0, 3], [6]],
    "num_heads": 1,
    "expand_ratio": 1,
    "hidden_ratio": 4,
    "attn_mode": "fused_recurrent",
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-6
}
```

Key parameters:
- `d_model`: List of hidden dimensions per stage
- `num_blocks`: Triplets of [encoder_blocks, 0, decoder_blocks] per stage, final stage is list of single int
- `hidden_ratio`: MLP expansion ratio (4 = 4x hidden size)

---

## Training

### Using Experiment Config (Recommended)

Create an experiment config file:

```json
{
    "experiment_name": "my_experiment",
    "model": {
        "config_file": "configs/slm_base.json",
        "type": "flat"
    },
    "data": {
        "data_dir": "./data/tinystories",
        "max_seq_length": 512
    },
    "training": {
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "learning_rate": 3e-4,
        "max_steps": 10000,
        "warmup_steps": 500,
        "scheduler": "cosine",
        "mixed_precision": "bf16"
    },
    "output": {
        "checkpoint_dir": "./checkpoints/my_experiment",
        "log_dir": "./logs/my_experiment"
    }
}
```

Run training:

```bash
python scripts/run_experiment.py --config my_experiment.json
```

### Using train.py Directly

```bash
python train.py \
    --model_config configs/slm_base.json \
    --data_path ./data/tinystories \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --max_steps 10000 \
    --output_dir ./checkpoints/my_run
```

### Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Batch size per device |
| `gradient_accumulation_steps` | 4 | Accumulation steps |
| `learning_rate` | 3e-4 | Initial learning rate |
| `weight_decay` | 0.01 | Weight decay |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `warmup_steps` | 500 | LR warmup steps |
| `scheduler` | cosine | LR schedule (cosine/linear) |
| `mixed_precision` | bf16 | FP16/BF16/none |

### Monitoring Training

Training logs are saved to:
- `logs/<experiment>/training_log.csv` - Metrics CSV
- `logs/<experiment>/tensorboard/` - TensorBoard logs

View TensorBoard:

```bash
tensorboard --logdir logs/
```

Key metrics to monitor:
- **loss**: Cross-entropy loss (should decrease)
- **bpb**: Bits per byte (primary metric, lower is better)
- **learning_rate**: Should follow expected schedule
- **grad_norm**: Gradient norm (should be stable)

### Checkpoints

Checkpoints are saved to:
- `checkpoints/<experiment>/step_XXXXX.pt` - Periodic checkpoints
- `checkpoints/<experiment>/best.pt` - Best validation loss

Resume training from checkpoint:

```bash
python train.py --resume_from ./checkpoints/my_run/step_5000.pt
```

---

## Evaluation

### Using the Evaluation Script

```bash
python scripts/evaluate_model.py \
    --model_path ./checkpoints/best.pt \
    --data_dir ./data/tinystories \
    --output_dir ./evaluation_results
```

### Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| BPB (Bits per Byte) | Primary metric | < 1.5 |
| Perplexity | Exponential of loss | < 20 |
| Accuracy | Next-byte accuracy | > 30% |
| Top-5 Accuracy | In top 5 predictions | > 60% |
| UTF-8 Validity | Valid generated UTF-8 | > 95% |

### Comparing Checkpoints

```python
from hnet_bit.training.evaluator import Evaluator

evaluator = Evaluator(model, tokenizer, config)
results = evaluator.compare_checkpoints(
    checkpoint_paths=["step_5000.pt", "step_10000.pt", "best.pt"],
    test_dataloader=test_loader,
    load_fn=load_model,
)
# Prints comparison table
```

---

## Generation

### Basic Generation

```bash
python generate.py \
    --model_path ./checkpoints/best.pt \
    --prompt "Once upon a time" \
    --max_tokens 200 \
    --temperature 0.8
```

### Interactive Mode

```bash
python generate.py --model_path ./checkpoints/best.pt --interactive
# Enter prompts interactively
# Commands: /quit, /temp <v>, /topk <v>, /topp <v>
```

### Batch Generation

```bash
# From file
python generate.py \
    --model_path ./checkpoints/best.pt \
    --prompt_file prompts.txt \
    --output_file outputs.txt

# Multiple samples from one prompt
python generate.py \
    --model_path ./checkpoints/best.pt \
    --prompt "The story begins" \
    --num_samples 5
```

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 1.0 | Sampling temperature |
| `top_k` | 50 | Top-k sampling |
| `top_p` | 0.9 | Nucleus sampling |
| `repetition_penalty` | 1.0 | Repetition penalty |
| `max_tokens` | 100 | Max new tokens |

---

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size
2. Increase gradient accumulation
3. Use smaller model (flat instead of hierarchical)
4. Reduce sequence length

```json
{
    "training": {
        "batch_size": 8,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 256
    }
}
```

### Loss Not Decreasing

1. Check learning rate (try 1e-4 or 5e-4)
2. Increase warmup steps
3. Check data loading (is data shuffled?)
4. Verify model is in training mode

### NaN Loss

1. Reduce learning rate
2. Enable gradient clipping (max_grad_norm=1.0)
3. Check for data issues (invalid bytes)
4. Try different initialization seed

### Slow Training

1. Use mixed precision (bf16 or fp16)
2. Increase batch size
3. Use more data workers
4. Enable compile_model if PyTorch 2.0+

### Poor Generation Quality

1. Train longer
2. Try different sampling parameters
3. Check if model converged (validation loss)
4. Use nucleus sampling (top_p=0.9)

---

## Best Practices

### Training Tips

1. **Start small**: Test with subset before full dataset
2. **Monitor early**: Check loss after 100-500 steps
3. **Save often**: Use save_interval for recovery
4. **Log everything**: Enable TensorBoard/WandB

### Hyperparameter Recommendations

| Model Size | Batch Size | LR | Steps |
|------------|------------|-----|-------|
| Small (<30M) | 32-64 | 3e-4 | 10-20K |
| Medium (30-100M) | 16-32 | 2e-4 | 20-50K |
| Large (>100M) | 8-16 | 1e-4 | 50-100K |

### Evaluation Best Practices

1. Always evaluate on held-out test set
2. Compare against baselines (random, flat model)
3. Check generation quality qualitatively
4. Report BPB instead of perplexity for byte-level

---

## Reference

### File Structure After Training

```
my_experiment/
├── checkpoints/
│   ├── step_1000.pt
│   ├── step_2000.pt
│   └── best.pt
├── logs/
│   ├── training_log.csv
│   ├── eval_log.csv
│   └── tensorboard/
├── results/
│   ├── evaluation_results.json
│   └── generations.txt
└── experiment_summary.json
```

### Key Classes

- `SimplifiedSLMForCausalLM`: Flat model
- `HNetBitForCausalLM`: Hierarchical model
- `Trainer`: Training loop
- `Evaluator`: Evaluation pipeline
- `ByteTokenizer`: Byte-level tokenizer
