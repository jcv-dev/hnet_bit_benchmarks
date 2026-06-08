# HNetBit: Hierarchical MatMul-Free Language Model Benchmark

This repository benchmarks three language model architectures on the Spanish Billion Words dataset under a fixed compute budget of 100 billion bytes of training text.

Architecture C (hybrid) is the primary contribution: it combines HNet's dynamic hierarchical chunking with MatMulFreeLM's ternary weight quantization. Architectures A (Transformer BPE) and B (flat MatMulFreeLM) are baselines.

## Architectures

| Label | Architecture | Tokenization | Weights | Model File |
|---|---|---|---|---|---|
| transformer | Llama-style Transformer (HF) | BPE (gpt2) | FP16 | HuggingFace `transformers` |
| matmulfree | HGRN recurrent LM (flat) | Byte-level (256 vocab) | Ternary {-1, 0, +1} | `mmfreelm/models/hgrn_bit/` |
| hybrid | HNetBit hierarchical recurrent LM | Byte-level (256 vocab) | Ternary {-1, 0, +1} | `hnet_bit/models/hnet_bit.py` |
| hybrid_attn | Hybrid with sliding-window attention in the innermost stage | Byte-level (256 vocab) | Ternary + FP16 QKV | `hnet_bit/models/hnet_bit.py` |

Architecture D (hybrid_attn) is an ablation variant of the hybrid model that replaces the innermost HGRN blocks with an alternating pattern of HGRN and sliding-window attention (`"xaxa"`). The attention layers use full-precision QKV projections with RoPE positional embeddings and a window size of 64. This variant tests whether attention at the deepest hierarchy level improves quality over pure-HGRN processing.

### Key features of the hybrid architecture

- **Hierarchical processing**: Multi-stage architecture with learned dynamic chunking. The model learns to segment byte sequences into variable-length chunks via cosine-similarity boundary detection, processes them at progressively higher dimensions, and reconstructs the original length.
- **Ternary weights**: All linear projections use weights quantized to {-1, 0, +1} via straight-through estimator, reducing storage ~8x versus FP16 and enabling addition-based computation.
- **HGRN recurrence**: Gated linear recurrence with O(n) training and O(1) per-step inference replaces the standard O(n^2) self-attention.
- **Chunk-parallel training**: Exponential moving average for dechunking is computed via the same HGRN kernel (parallel scan), avoiding the sequential loop of the original HNet.

## Setup

```bash
bash scripts/setup_cloud.sh   # Install dependencies (run on cloud GPU)
```

## Quick start: smoke test

```bash
# Test hybrid model on CPU (takes ~30 seconds)
bash test_smoke.sh

# Test all three models on GPU (requires CUDA + HuggingFace login)
bash test_smoke.sh --gpu
```

The smoke test trains each model at "tiny" size for 15 steps on a synthetic 100KB Spanish corpus, then aggregates results. Output goes to `./runs/smoke_test/`.

## Full benchmark

### 1. Build the dataset

```bash
# For byte-level models (matmulfree, hybrid):
python train_spanish.py --model hybrid --size 150M --skip_data_build

# For BPE model (transformer) — builds both byte and BPE data:
python train_spanish.py --model transformer --size 150M
```

The dataset downloads from HuggingFace once and caches to `./data/spanish/`. If you only need byte-level, use `--skip_data_build` after the first run.

### 2. Train

```bash
# Minimal test run (10 steps, for verification):
python train_spanish.py --model hybrid --size tiny   --max_steps 10 --batch_size 2

# Production runs:
python train_spanish.py --model matmulfree  --size 350M
python train_spanish.py --model transformer --size 750M
python train_spanish.py --model hybrid      --size 150M
```

### 3. Aggregate results

```bash
# Collect existing results from completed runs:
python generate_results.py --runs_dir ./runs/spanish --output results.csv

# Re-evaluate all final checkpoints (computes BPB + memory fresh):
python generate_results.py --runs_dir ./runs/spanish --reeval --output results.csv
```

## Model configurations

### Sizes available

| Size | transformer (params) | matmulfree (params) | hybrid (params) | hybrid_attn (params) | Use |
|---|---|---|---|---|---|
| tiny | ~10K | 166K | 259K | 259K | Smoke test / CI |
| 150M | ~150M | ~150M | ~150M | ~126M | Lightweight baseline |
| 350M | ~350M | ~350M | ~350M | ~350M | Primary comparison |
| 750M | ~750M | ~750M | ~750M | ~750M | Scale test |

The `hybrid_attn` model uses the same architecture configs as `hybrid` but enables sliding-window attention in the innermost stage (pattern `"xaxa"`, window size 64). The attention QKV projections use full-precision `nn.Linear` instead of `BitLinear` for stability. All configs are defined in `model_factory.py`.

### Hybrid architecture details

| Size | d_model | Stages | Blocks | Innermost pattern |
|---|---|---|---|---|
| tiny | [48, 64] | 1 (2 levels) | enc:1, inner:2, dec:1 | HGRN only |
| 150M | [576, 768] | 1 (2 levels) | enc:4, inner:10, dec:4 | HGRN only (or `xaxa` for hybrid_attn) |
| 350M | [640, 896, 1152] | 2 (3 levels) | enc:4, inner:12, dec:4 each | HGRN only |
| 750M | [896, 1152, 1536] | 2 (3 levels) | enc:6, inner:16, dec:6 each | HGRN only |

The `hybrid_attn` model enables `innermost_use_attention=True` with `attention_layers_pattern="xaxa"`, which alternates HGRN and attention blocks at the innermost stage. The pattern is automatically extended cyclically for any number of innermost blocks. Sliding window size defaults to 64 with RoPE positional embeddings.

## Metrics collected

The benchmark saves the following per run and in the aggregated CSV:

| Column | Description |
|---|---|
| Model | Architecture label (transformer, matmulfree, hybrid) |
| Size | Configuration label (tiny, 150M, 350M, 750M) |
| BPB / Best_Val_BPB / Final_Val_BPB | Bits per byte (lower is better). Main quality metric. |
| Val_BPB_at_25B/50B/100B | BPB at training milestones (bytes processed) |
| Best_Val_Loss / Final_Val_Loss | Cross-entropy loss on validation set |
| Inference_Memory_MB | Peak GPU memory during batch=1 forward pass |
| Param_Count_M | Total model parameters in millions |
| Disk_Size_MB | Theoretical model disk size (BF16, params * 2 bytes) |
| Peak_Training_Memory_MB | Peak GPU memory during training (allocated) |
| Peak_Reserved_Memory_MB | Peak GPU memory reserved (includes cache) |
| Overall_Compression_Ratio | (Hybrid only) Product of per-stage boundary ratios |
| Best_Train_Loss / Final_Train_Loss | Training cross-entropy loss |
| Peak_Tok_Per_Sec / Avg_Tok_Per_Sec | Training throughput |
| Training_Time_Hours | Wall-clock training time |
| LR, Batch_Size, Grad_Accum, Total_Bytes, Seq_Length | Hyperparameters |

### Per-run output files

Each run creates a directory `runs/spanish/{model}_{size}/` containing:

| File | Contents |
|---|---|
| `config.json` | Training hyperparameters |
| `training_stats.json` | Training time, parameter count, peak memory, compression ratio |
| `training_steps_log.csv` | Step-by-step loss, learning rate, gradient norm, throughput |
| `validation_log.csv` | Periodic validation loss and BPB |
| `checkpoint_final.pt` | Final model checkpoint |
| `checkpoint_best.pt` | Checkpoint with lowest validation BPB |

The per-model results CSV is saved at `runs/spanish/results_{model}_{size}.csv`.

## Output file structure

```
runs/spanish/
├── results_hybrid_350M.csv            # Per-model final results
├── results_matmulfree_350M.csv
├── results_transformer_350M.csv
├── hybrid_350M/                       # Per-run directory
│   ├── config.json
│   ├── training_stats.json
│   ├── training_steps_log.csv
│   ├── validation_log.csv
│   ├── checkpoint_final.pt
│   ├── checkpoint_best.pt
│   └── tensorboard/
├── matmulfree_350M/
│   └── ...
├── transformer_350M/
│   └── ...
└── results.csv                        # Aggregated results (generate_results.py)
```

## Tokenization

**Byte-level models** (matmulfree, hybrid): operate directly on raw UTF-8 bytes with a fixed vocabulary of 256. No tokenizer training or download needed.

**Transformer baseline** (transformer): uses the GPT-2 BPE tokenizer with 50,257 tokens. The `--tokenizer_name` flag can switch to any HuggingFace tokenizer (e.g., `--tokenizer_name meta-llama/Llama-3.2-1B`). The GPT-2 tokenizer is used by default because it does not require authentication.

## GPU recommendations

For thesis-scale benchmarking with personal funds, a single NVIDIA A100 80GB (cloud rental at approximately $0.60-$1.80/hour) covers all three model sizes including the 1B parameter expansion. See the table below for approximate training times per model at 100 billion bytes of training text.

| Size | Batch | Steps | Approx. time (A100 80GB) | Memory |
|---|---|---|---|---|
| 150M | 4 x 8 ga | 381K | ~20 hours | ~3 GB |
| 350M | 4 x 8 ga | 381K | ~35 hours | ~7 GB |
| 750M | 4 x 8 ga | 381K | ~55 hours | ~14 GB |
| 1B | 4 x 8 ga | 381K | ~70 hours | ~20 GB |

Cloud providers with good spot pricing: RunPod, Lambda Labs, Vast.ai. See `DEPLOYMENT.md` for a step-by-step guide on setting up a cloud machine, installing dependencies, and running the full benchmark.

## Repository structure

```
tesis/
├── train_spanish.py           # Main benchmark training script
├── generate_results.py         # Result aggregation
├── test_smoke.sh              # Smoke test script
├── model_factory.py           # Builds all three model architectures
├── training_config_spanish.py # Training configuration and WSD scheduler
├── data_spanish.py            # Dataset loading (byte and BPE)
├── metrics_spanish.py         # BPB and memory measurement
├── matmulfreellm/             # MatMulFreeLM reference implementation
│   └── mmfreelm/
│       ├── ops/hgrn/          # HGRN Triton kernels
│       ├── ops/bitnet.py      # BitLinear quantization
│       ├── models/hgrn_bit/   # HGRNBitForCausalLM
│       └── modules/           # FusedNormGate, ShortConv, activations
├── hnet-main/                 # HNet reference implementation
│   └── hnet/
│       ├── models/hnet.py     # Recursive hierarchical backbone
│       └── modules/dc.py      # Dynamic chunking (routing, chunk, dechunk)
└── hnet_bit/                  # This project's model implementation
    ├── models/hnet_bit.py     # HNetBitForCausalLM (hierarchical + ternary)
    ├── layers/hgrn_bit.py     # HGRNBitBlock (attention + MLP)
    ├── ops/
    │   ├── bitnet.py          # BitLinear with inline RMSNorm
    │   ├── fusedbitnet.py     # Triton-fused BitLinear
    │   ├── dynamic_chunking.py# RoutingModuleBit, ChunkLayer, DeChunkLayer
    │   ├── activations.py     # SwiGLU (CUDA jiterator + CPU fallback)
    │   ├── fused_norm_gate.py # FusedRMSNormSwishGate
    │   ├── short_conv.py      # Optional short convolution
    │   └── hgrn/              # HGRN recurrence (chunk + fused recurrent)
    ├── utils/hnet_cache.py    # Nested cache for generation
    ├── training/              # Alternative training pipeline
    ├── tests/                 # 48+ unit tests
    ├── configs/               # JSON configuration files
    └── docs/TRAINING_GUIDE.md # Standalone training documentation

## References

- MatMul-Free LM: https://github.com/ridgerchu/matmulfreellm
- HNet (Dynamic Chunking): https://github.com/voidism/HNet
- HGRN2: Gated Linear RNNs with State Expansion, arXiv:2404.07904
- BitNet: Scaling 1-bit Transformers, arXiv:2310.11453
