# HNetBit: Hierarchical MatMul-Free Language Model

A hierarchical language model combining H-Net's multi-stage architecture with MatMul-free ternary weights:
- **BitLinear** layers with ternary weight quantization ({-1, 0, +1})
- **HGRN** (Hierarchical Gated Recurrent Network) for efficient sequence modeling
- **Multi-stage hierarchical** architecture with dynamic chunking
- **Chunk-parallel** recurrence for fast training
- **Byte-level tokenization** (vocab_size=256)

## Architecture

```
Embedding (256 → d_model[0])
    ↓
Stage 0 Encoder (N × HGRNBitBlock)
    ↓
ChunkLayer (dynamic downsampling)
    ↓
Stage 1 Encoder (M × HGRNBitBlock)
    ↓
ChunkLayer
    ↓
Innermost Stage (K × HGRNBitBlock)
    ↓
DeChunkLayer
    ↓
Stage 1 Decoder (M × HGRNBitBlock)
    ↓
DeChunkLayer
    ↓
Stage 0 Decoder (N × HGRNBitBlock)
    ↓
BitLinear LM Head (d_model[0] → 256)
```

### Key Features

- **Hierarchical Processing**: Multi-stage architecture with learned compression
- **Dynamic Chunking**: Cosine-similarity based adaptive downsampling
- **Ternary Weights**: All linear layers use weights quantized to {-1, 0, +1}
- **No MatMul**: Ternary weights enable efficient addition-based computation
- **Linear Complexity**: HGRN has O(n) complexity vs O(n²) for attention
- **Memory Efficient**: ~8x memory reduction vs FP16 weights (theoretical)

## Installation

```bash
# Requirements
pip install torch transformers triton einops

# Install package
cd hnet_bit
pip install -e .
```

## Quick Start

```python
from hnet_bit.models import HNetBitConfig, HNetBitForCausalLM
from hnet_bit.utils import ByteTokenizer

# Create hierarchical model
config = HNetBitConfig.small_1stage()  # or .small_2stage()
model = HNetBitForCausalLM(config).cuda()

# Tokenize
tokenizer = ByteTokenizer()
text = "Hello world"
encoded = tokenizer.encode([text])[0]['input_ids']
input_ids = torch.tensor([encoded.tolist()]).cuda()

# Generate
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0].tolist()))
```

## Model Configurations

| Config | Stages | d_model | Blocks | Params (approx.) |
|--------|--------|---------|--------|-----------|
| Tiny (smoke test) | 1 | [48, 64] | [[1, 0, 1], [2]] | 0.3M |
| Small 1-stage  | 1 | [256, 384] | [[4, 0, 4], [8]] | ~21M |
| Base 2-stage   | 2 | [512, 768, 1024] | [[4, 0, 4], [4, 0, 4], [8]] | ~50M |
| Large 2-stage  | 2 | [768, 1024, 1536] | [[4, 0, 4], [4, 0, 4], [12]] | ~100M |

Parameter counts are approximate. Exact counts depend on hidden_ratio, expand_ratio,
and the number of non-quantized components (embedding, residual projections, routing).

```python
# Use preset configs
config = HNetBitConfig.small_1stage()  # or .small_2stage()
```

## Project Structure

```
hnet_bit/
├── models/
│   └── hnet_bit.py         # HNetBit hierarchical model
├── layers/
│   └── hgrn_bit.py         # HGRNBitAttention, HGRNBitMLP, HGRNBitBlock
├── ops/
│   ├── bitnet.py           # BitLinear, weight_quant, activation_quant
│   ├── activations.py      # SwiGLU, swish
│   ├── dynamic_chunking.py # Cosine-similarity based chunking
│   ├── fused_norm_gate.py  # FusedRMSNormSwishGate
│   └── hgrn/
│       ├── chunk.py          # Chunk-parallel HGRN
│       └── recurrent_fuse.py # Fused recurrent HGRN
├── training/
│   ├── trainer.py          # Training loop
│   ├── config.py           # TrainingConfig
│   ├── data.py             # ByteLevelDataset
│   ├── dataset_loader.py   # HuggingFace dataset integration
│   ├── metrics.py          # BPB, perplexity, evaluation metrics
│   ├── evaluator.py        # Comprehensive evaluation pipeline
│   ├── logger.py           # TensorBoard, W&B, CSV logging
│   └── optimizer.py        # Optimizer factory
├── utils/
│   ├── tokenizers.py       # ByteTokenizer
│   ├── cache.py            # RecurrentCache
│   ├── hnet_cache.py       # HNetBit cache
│   └── helpers.py          # Utility functions
├── scripts/
│   ├── prepare_tinystories.py  # Dataset preparation
│   ├── evaluate_model.py       # Model evaluation
│   ├── analyze_training.py     # Training analysis
│   └── run_experiment.py       # End-to-end pipeline
├── configs/
│   ├── hnet_bit_1stage.json            # 1-stage hierarchical
│   ├── hnet_bit_2stage.json            # 2-stage hierarchical
│   ├── training_1stage_tinystories.json # 1-stage training config
│   └── training_2stage_tinystories.json # 2-stage training config
├── tests/
│   ├── test_model.py           # Model tests
│   ├── test_dataset_loader.py  # Dataset loader tests
│   ├── test_metrics.py         # Metrics tests
│   └── test_evaluator.py       # Evaluator tests
├── docs/
│   └── TRAINING_GUIDE.md       # Comprehensive training guide
├── train.py                    # Training entry point
└── generate.py                 # Text generation script
```

## Technical Details

### Ternary Weight Quantization

```python
def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    return (w * scale).round().clamp_(-1, 1) / scale
```

Uses Straight-Through Estimator (STE) for gradient flow:
```python
w_quant = w + (weight_quant(w) - w).detach()
```

### HGRN Recurrence

```
f_t = sigmoid(f_proj(x_t))           # Forget gate
i_t = swiglu(i_proj(x_t), 1 - f_t)   # Input with complement gate
h_t = f_t * h_{t-1} + i_t            # State update
o_t = g_norm(g_proj(x_t), h_t)       # Gated output
```

### Chunk-Parallel Training

Uses Triton kernels for parallelized recurrence computation:
- Forward: O(T/chunk_size) parallel chunks
- Backward: Gradient accumulation across chunks
- Much faster than sequential for long sequences

## Running Tests

```bash
cd hnet_bit
python -m pytest tests/ -v

# Or run directly
python tests/test_model.py
```

## Generation

```bash
# Basic generation
python generate.py --prompt "Hello" --max_tokens 100 --temperature 0.8
python generate.py --model_path ./checkpoint.pt --prompt "Once upon a time" --greedy

# Interactive mode
python generate.py --model_path ./checkpoint.pt --interactive

# Batch generation from file
python generate.py --model_path ./checkpoint.pt --prompt_file prompts.txt --output_file outputs.txt
```

## Training

Full training infrastructure is now available! See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for comprehensive documentation.

### Quick Start

```bash
# 1. Prepare TinyStories dataset
python scripts/prepare_tinystories.py --output_dir ./data/tinystories

# 2. Train 2-stage hierarchical model
python train.py \
    --model_config configs/hnet_bit_2stage.json \
    --training_config configs/training_2stage_tinystories.json

# 3. Evaluate model
python scripts/evaluate_model.py --checkpoint ./checkpoints/best_model.pt --dataset tinystories

# 4. Generate text
python generate.py --model_path ./checkpoints/best_model.pt --prompt "Once upon a time"
```

### End-to-End Pipeline

```bash
# Run complete experiment (data → train → evaluate → analyze)
python scripts/run_experiment.py --config configs/training_2stage_tinystories.json
```

### Training Features

- **Datasets**: TinyStories, WikiText-2/103, custom text directories
- **Mixed Precision**: BF16 training with PyTorch AMP
- **Logging**: TensorBoard + Weights & Biases + CSV export
- **Metrics**: BPB (bits per byte), perplexity, accuracy, UTF-8 validity
- **Monitoring**: Ternary weight stats, gradient norms, generation samples
- **Analysis**: Training curves, loss visualization, checkpoint comparison

### Model Configurations

| Model Type | Config File | Description |
|------------|-------------|-------------|
| 2-Stage Hierarchical | `training_2stage_tinystories.json` | Full hierarchical with dynamic chunking |
| 1-Stage Hierarchical | `training_1stage_tinystories.json` | Single hierarchy level |

## Benchmark pipeline

This model is one of three architectures benchmarked in a thesis comparing transformer, flat MatMulFree, and hierarchical hybrid models on the Spanish Billion Words dataset. The benchmark pipeline is at the repository root:

- `train_spanish.py` — unified training script for all three architectures
- `generate_results.py` — aggregate per-run results into a comparison table
- `test_smoke.sh` — smoke test (add `--gpu` for all models)
- `model_factory.py` — builds each architecture at configurable sizes

See the root `README.md` for usage instructions.

## References

- [H-Net Paper](https://arxiv.org/abs/2410.06687) - Hierarchical dynamic chunking architecture
- [MatMul-Free LM](https://github.com/ridgerchu/matmulfreellm) - Original BitLinear and HGRN implementation
- [HGRN2 Paper](https://arxiv.org/abs/2404.07904) - Gated Linear RNNs with State Expansion
- [BitNet Paper](https://arxiv.org/abs/2310.11453) - Scaling 1-bit Transformers

## License

MIT License
