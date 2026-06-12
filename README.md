# HNetBit: Hierarchical MatMul-Free Language Model Benchmark

This repository benchmarks four language model architectures on the Spanish Billion Words dataset under a fixed training budget of 25 billion bytes of underlying text.

The hybrid model (`hybrid`) is the primary contribution: it combines H-Net's dynamic hierarchical chunking with ternary weight quantization (MatMul-free). A fourth variant (`hybrid_attn`) ablates the innermost stage with sliding-window attention. The transformer (BPE Llama-style) and flat matmulfree (HGRN recurrent) models serve as baselines.

## Architectures

| Label | Architecture | Tokenization | Weights | Source |
|---|---|---|---|---|
| `transformer` | Llama-style Transformer (HF) | BPE (gpt2, 50K vocab) | FP16 | HuggingFace `transformers` |
| `matmulfree` | HGRN recurrent LM (flat) | Byte-level (256 vocab) | Ternary {-1, 0, +1} | `matmulfreellm/` |
| `hybrid` | HNetBit hierarchical recurrent LM | Byte-level (256 vocab) | Ternary {-1, 0, +1} | `hnet_bit/` |
| `hybrid_attn` | Hybrid + sliding-window attention in innermost stage | Byte-level (256 vocab) | Ternary + FP16 QKV | `hnet_bit/` |

The `hybrid_attn` variant replaces half the innermost HGRN blocks with alternating sliding-window attention blocks (pattern `"xaxa"`, window size 64, RoPE), using full-precision Q/K/V/O projections for attention score stability.

### Key features

- **Ternary weights**: All linear projections quantized to {-1, 0, +1} via straight-through estimator, achieving ~7-8× disk compression vs FP16
- **Hierarchical processing**: Multi-stage learned dynamic chunking compresses byte sequences for efficient inner-stage processing
- **HGRN recurrence**: Gated linear recurrence with O(L) training and O(1) per-step inference replaces O(L²) self-attention

## Setup

```bash
bash scripts/setup_cloud.sh   # Install dependencies (cloud GPU)
```

If inside an unprivileged Docker container, Triton JIT may fail. Force CPU fallback:
```bash
export HNETBIT_DISABLE_TRITON=1
```

## Quick start: smoke test

```bash
bash test_smoke.sh           # hybrid only (CPU)
bash test_smoke.sh --gpu     # all models (GPU)
```

Trains all models at "tiny" size for 15 steps, then aggregates results to `runs/smoke_test/`.

## Full benchmark

### 1. Build the dataset

```bash
# Downloads ~8.7 GB Spanish text from HuggingFace, caches to data/spanish/
python train_spanish.py --model hybrid --size 150M --max_steps 1 --batch_size 1
# Subsequent runs use --skip_data_build
```

### 2. Train

```bash
python train_spanish.py --model hybrid --size 150M                    # full run
python train_spanish.py --model hybrid --size 150M --max_steps 100    # debug run
python train_spanish.py --model transformer --size 350M
python train_spanish.py --model matmulfree --size 750M
python train_spanish.py --model hybrid_attn --size 150M
```

### 3. Aggregate results

```bash
# Collect results from completed runs:
python generate_results.py --runs_dir ./runs/spanish --output results.csv

# Re-evaluate all final checkpoints:
python generate_results.py --runs_dir ./runs/spanish --reeval --cache_dir ./data/spanish
```

### 4. Deploy and profile

```bash
# Export compact ternary-weight deployment:
python export_deployment.py --checkpoint runs/spanish/hybrid_150M/checkpoint_best.pt

# Profile inference throughput:
python profile_inference.py --checkpoint runs/spanish/hybrid_150M/checkpoint_best.pt
```

## Architecture details

| Size | transformer | matmulfree | hybrid | hybrid_attn |
|---|---|---|---|---|
| 150M | 190M (113M non-emb) | 114M | 138M | ~126M |
| 350M | 505M (403M non-emb) | 309M | 419M | ~407M |
| 750M | 1060M (906M non-emb) | 794M | 1026M | ~1010M |

Non-embedding params exclude lookup tables (embedding + LM head). The transformer incurs ~77-154M embedding overhead from the 50K BPE vocab. Byte-level models use a 256 vocab, making their embedding tables negligible. Report both in the thesis.

### Hybrid architecture layouts

| Size | d_model | Stages | Encoder | Innermost | Decoder |
|---|---|---|---|---|---|
| 150M | [576, 768] | 1-stage (2 levels) | 4 blocks × d=576 | 10 blocks × d=768 | 4 blocks × d=576 |
| 350M | [640, 896, 1152] | 2-stage (3 levels) | 4+4 blocks | 12 blocks × d=1152 | 4+4 blocks |
| 750M | [896, 1152, 1536] | 2-stage (3 levels) | 6+6 blocks | 16 blocks × d=1536 | 6+6 blocks |

For `hybrid_attn`, innermost stage blocks alternate HGRN and sliding-window attention (window=64, RoPE). The pattern `"xaxa"` cycles automatically for any number of blocks.

## Training methodology

All models consume the same 25B bytes of underlying Spanish text. A `bytes_per_step` formula equalizes the budget across tokenization schemes. Effective batch size is 32 (batch_size=4 × gradient_accumulation=8). Training uses AdamW with WSD learning rate schedule (1% warmup, 79% stable, 20% cosine decay) under bf16 mixed precision. Full details and fairness analysis in `AGENTS.md`.

## Repository structure

```
├── train_spanish.py           # Main benchmark training script
├── model_factory.py           # Builds all four architectures at configurable sizes
├── training_config_spanish.py # SpanishTrainingConfig, WSD scheduler, optimizer
├── data_spanish.py            # Dataset handling (byte-level + BPE)
├── metrics_spanish.py         # BPB computation, inference memory
├── generate_results.py        # Result aggregation + auto-export deploy models
├── export_deployment.py       # Compact ternary-weight deployment export
├── profile_inference.py       # Prefill latency + decode throughput profiler
├── test_smoke.sh              # Smoke test script
├── AGENTS.md                  # Complete thesis source-of-truth documentation
├── architecture.md            # Detailed HNetBit architecture reference
├── DEPLOYMENT.md              # Cloud GPU deployment guide
├── hnet_bit/                  # Hybrid model implementation (layers, ops, models)
├── matmulfreellm/             # Reference MatMulFree repo (used by matmulfree)
├── hnet-main/                 # Reference HNet repo (not directly imported)
└── scripts/setup_cloud.sh     # Cloud GPU dependency installer
```

## References

- H-Net (Dynamic Chunking): https://github.com/voidism/HNet
- MatMul-Free LM: https://github.com/ridgerchu/matmulfreellm
- HGRN2 (Gated Linear RNNs): arXiv:2404.07904
- BitNet (Scaling 1-bit Transformers): arXiv:2310.11453
