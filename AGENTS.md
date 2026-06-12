# AGENTS.md

## Project overview

Thesis benchmark comparing three language model architectures on Spanish Billion Words (25B-byte training budget):

- `transformer` — Llama-style, BPE tokenizer (gpt2), FP16 weights, HuggingFace
- `matmulfree` — flat HGRN recurrent LM, byte-level, ternary {-1,0,+1} weights (from `matmulfreellm/`)
- `hybrid` / `hybrid_attn` — hierarchical HNetBit with dynamic chunking + ternary weights (from `hnet_bit/`)
- `hybrid_attn` — same as hybrid, but innermost layer alternates HGRN with sliding-window attention

## Key files

| File | Purpose |
|---|---|
| `train_spanish.py` | Main training script (CLI: `--model`, `--size`) |
| `model_factory.py` | Builds all 3 architectures at configurable sizes (tiny, 150M, 350M, 750M) |
| `training_config_spanish.py` | SpanishTrainingConfig dataclass, WSD scheduler, optimizer builder |
| `data_spanish.py` | Dataset handling (byte-level + BPE), tokenization |
| `metrics_spanish.py` | BPB computation, inference memory measurement |
| `generate_results.py` | Aggregates per-run results into a comparison CSV (auto-exports deploy models) |
| `export_deployment.py` | Converts training checkpoint → compact ternary-weight deployment (CLI: `--checkpoint`) |
| `profile_inference.py` | Measures prefill latency + decode throughput for all architectures (CLI: `--checkpoint` or `--export`) |
| `hnet_bit/` | The hybrid model implementation (layers, ops, models) |
| `matmulfreellm/` | Reference MatMulFree repo (used by matmulfree model) |
| `hnet-main/` | Reference HNet repo (not directly imported) |
| `scripts/setup_cloud.sh` | Cloud GPU dependency installer |

## Essential commands

### Smoke test
```bash
bash test_smoke.sh           # hybrid only (CPU)
bash test_smoke.sh --gpu     # all models (GPU)
```

### Dataset (one-time)
```bash
# Downloads ~8.7 GB Spanish text from HuggingFace, caches to data/spanish/
python train_spanish.py --model hybrid --size 150M --max_steps 1 --batch_size 1
# Subsequent runs use --skip_data_build
```

### Training
```bash
python train_spanish.py --model hybrid --size 150M                    # full run
python train_spanish.py --model hybrid --size 150M --max_steps 100    # debug run
```

### Results
```bash
python generate_results.py --runs_dir ./runs/spanish --output results.csv
```

### Deployment export
```bash
python export_deployment.py --checkpoint runs/spanish/hybrid_150M/checkpoint_best.pt
# Produces model_deploy.pt with frozen ternary weights (7-8x smaller than fp16)
```

### Inference profiling
```bash
python profile_inference.py --checkpoint runs/spanish/hybrid_150M/checkpoint_best.pt
python profile_inference.py --export runs/spanish/hybrid_150M/model_deploy.pt   # from compact export
```

## Architecture details

| Size | transformer params | matmulfree params | hybrid params | hybrid_attn params |
|---|---|---|---|---|
| 150M | 190M (113M non-emb) | 114M | 138M | ~126M |
| 350M | 505M (403M non-emb) | 309M | 419M | ~407M |
| 750M | 1060M (906M non-emb) | 794M | 1026M | ~1010M |

Non-embedding params exclude lookup tables (embedding + LM head). Transformer has ~77-154M embedding overhead from 50K BPE vocab. Byte-level models use 256 vocab. Report both in thesis.

**Important context window asymmetry**: The transformer sees ~5,760 bytes of context (1,280 BPE tokens × ~4.5 bytes/token) while byte-level models see 4,096 bytes. The `bytes_per_step` formula equalizes the text budget, but the per-sample context window differs — giving the transformer a potential advantage on long-range dependencies. Document this in the thesis.

## Framework quirks

- **Triton JIT compilation**: required for fast training. Blocked in unprivileged Docker containers. Set `HNETBIT_DISABLE_TRITON=1` to fall back to naive PyTorch loops (3-10x slower). Use VM templates on Vast.ai or RunPod to avoid this.
- **HuggingFace token**: the `huggingface-cli` shell command may not exist in cloud templates. Use `python3 -c "from huggingface_hub import login; login(token='...')"` or write token to `HF_HOME/token` directly.
- **BPE tokenization RAM (fixed)**: tokenizing 8.7 GB corpus with gpt2 produces ~2.9B tokens (~88 GB as int32). Both `_write_byte_corpus` and `_write_bpe_corpus` run in `multiprocessing.Process` children to prevent CPython's pymalloc heap accumulation. The BPE worker is recycled every 50 chunks (`CHUNKS_PER_WORKER=50` in `data_spanish.py:195`) to cap child RSS. Parent process RSS stays flat (~5-10 GB). Do NOT use `open_memmap` — OS page cache balloons.
- **max_position_embeddings**: transformer tiny config had 256 which conflicted with DataLoader's 1792-token sequences. Fixed to 2048. Real sizes (150M+) already use 2048.
- **Transformer embedding overhead**: 50K vocab x hidden_size x 2 (embed + lm_head) is ~77M params at 150M that contribute zero FLOPs. Track `Non_Emb_Params_M` for fair comparison.
- **Gradient checkpointing**: HNetBitForCausalLM is not HF-compatible. GC is functional for matmulfree and transformer, but the hybrid/hybrid_attn forward methods do not wrap layers with `torch.utils.checkpoint`. The `enable_gradient_checkpointing()` call sets the flag but it is never read by HNetBit's forward pass. Hybrid models train without activation recomputation — document this in training methodology comparisons.
- **Training time implications of no gradient checkpointing**: Hybrid 150M takes ~80 hours (3.4 days) on a single GPU. The transformer will be faster because (1) it has fewer total steps (smaller training budget per byte×step due to BPE packing), (2) gradient checkpointing works, reducing per-step computation by allowing larger batch sizes and (3) its forward pass is purely attention+MLP without hierarchical chunking overhead. Peak training memory for hybrid 150M is ~15.5 GB (no GC + hierarchical chunking buffers for each stage). The transformer at similar size should be ~6-8 GB.

## Methodology

### Training budget

All models consume the same 25B bytes of underlying Spanish text. The `bytes_per_step` formula equalizes the budget across tokenization schemes:

- Byte-level (matmulfree, hybrid, hybrid_attn): `effective_batch × byte_seq_length` = 32 × 4,096 = 131,072 bytes/step
- BPE (transformer): `effective_batch × token_seq_length × avg_bytes_per_token` = 32 × 1,280 × ~4.5 = 184,320 bytes/step

Total steps differ, but total bytes are identical.

### Equalized across all models

| Axis | Value |
|---|---|
| Training budget | 25B underlying text bytes |
| Effective batch size | 32 (4 × 8 gradient accumulation) |
| Optimizer | AdamW (β1=0.9, β2=0.95, ε=1e-8) |
| Weight decay | 0.01 (no decay on bias, norm, embeddings, dim<2) |
| Max gradient norm | 1.0 |
| Mixed precision | bf16 |
| LR schedule | WSD (1% warmup, 79% stable, 20% cosine decay) |
| Data | `jhonparra18/spanish_billion_words_clean` via HuggingFace |
| Train/val split | 95%/5% contiguous (last 5% as validation, no shuffle) |
| Seed | 42 |

### Differences between models

| Axis | Transformer | Matmulfree / Hybrid |
|---|---|---|
| Learning rate | 3e-4 | 4e-3 (150M), 2.5e-3 (350M), 1.5e-3 (750M) |
| Per-sample context | ~5,760 bytes (1,280 BPE tokens) | 4,096 bytes |
| Gradient checkpointing | ✓ functional | ✓ matmulfree, ✗ hybrid (not implemented in forward) |
| Tokenization | BPE (gpt2, 50K vocab) | Byte-level (256 vocab) |
| Weight representation | FP16 | Ternary {-1,0,+1} (BitLinear STE) |

The LR difference follows the original MatMul-free and H-Net papers — ternary models require higher learning rates to converge. The context window asymmetry is inherent to byte-level tokenization (byte models get more tokens per sample for the same byte budget) and is conservative — the transformer has a potential advantage on long-range dependencies.

### Metrics

**BPB (Bits-Per-Byte)** — primary quality metric. Computed from cross-entropy loss:

- Byte-level: `BPB = mean_NLL / ln(2)`  (nats per byte → bits per byte)
- BPE: `BPB = mean_NLL / (avg_bytes_per_token × ln(2))`  (nats per token → bits per byte)

Lower is better. Validation BPB is computed every 1,000 steps over 50 batches (~0.8 MB of text). Final evaluation uses 200 batches (~3.3 MB of text). BPB is preferred over perplexity because it is vocabulary-independent.

**Deployment size** — primary efficiency metric. The compact export (`model_deploy.pt`) stores frozen ternary weights at ~2.1 bits/parameter. Compared to FP16 (16 bits/param), this achieves 7-8× compression for ternary models. The transformer's export stores FP16 weights (no compression).

**Inference throughput** — secondary metric. Measured via `profile_inference.py`:
- Prefill (Time-To-First-Token): single forward pass at seq lengths 512/1024/2048/4096, 3 warmup + 5 timed runs, CUDA event timing
- Decode: greedy autoregressive generation of 256 tokens from a 256-token prompt, at batch sizes 1/4/8
- Memory: `torch.cuda.max_memory_allocated()` during generation; uses `torch.no_grad()` to avoid building autograd graphs (critical — without it, all intermediate activations are retained, inflating memory 3-5× for hierarchical models)

### Inference memory behaviour

- **Prefill memory**: Stays near-constant across context lengths (~600-800 MB for 150M hybrid) — the recurrent architecture has no KV cache, so memory does not grow with sequence length. The small variation is from activation buffer sizes.
- **Decode memory**: Nearly flat across batch sizes (e.g. 573 MB at batch 1 → 652 MB at batch 8). Compare to a transformer where every new batch entry adds a separate KV cache.
- **Why the numbers are low**:
  1. Ternary weights pack to ~2 bits/param, so the model's weight memory in the deploy export is ~40 MB for 150M params (unpacked to FP32 during inference, still only ~550 MB)
  2. No autograd graph (`torch.no_grad()`) — intermediate buffers freed immediately
  3. RNN-like recurrence means no quadratic attention memory

### Understanding the compression columns in results.csv

The CSV has two compression-related columns that are often confused:

| Column | What it measures | Hybrid 150M value |
|---|---|---|
| `Overall_Compression_Ratio` | **Training sparsity** — fraction of ternary weights that are non-zero during training (stored in `training_stats.json`). This is NOT the compression you get on disk. A value of ~0.31 means ~31% of ternary weights are non-zero on average. | ~0.31 |
| `Deploy_Size_MB` / `Bits_Per_Param` | **Actual deployment compression** — the size of `model_deploy.pt` on disk after 2-bit packing. The real compression ratio is `FP16_equivalent / Deploy_Size_MB` (e.g. 263.9 MB / 37.9 MB = 7.0×). | 37.9 MB / 2.29 bits/param |

The deployment compression ratio is shown in the export CLI output as `Compression ratio : 7.0x vs fp16` but is not stored in a dedicated CSV column. Compute it by dividing `Disk_Size_MB` (FP16 equivalent) by `Deploy_Size_MB`.

### Known limitations

- **Single run per model**: All runs use seed=42. No statistical error bars or variance estimates.
- **No hyperparameter sweep**: Optimizer, LR, weight decay, and WSD fractions are fixed. Architecture-specific hyperparameters (lambda_lb, lr_multipliers) use defaults rather than tuned values.
- **Spanish only**: Results may not generalize to other languages, scripts, or domains.
- **Contiguous val split**: The last 5% of the corpus is used for validation without shuffling. If the corpus has topical or chronological drift, validation metrics may not represent the full training distribution.
- **Not FLOPs-matched**: Unlike the original H-Net paper, models are compared at equal training bytes, not equal compute FLOPs. Parameter counts differ across architectures.
- **Context window asymmetry**: The transformer sees ~5,760 equivalent bytes of context vs 4,096 for byte-level models — a potential advantage on long-range dependencies.
- **Gradient checkpointing gap**: Transformer and matmulfree use activation recomputation; hybrid models do not. This affects training memory consumption but not final model quality.

## Output files per run

Each run at `runs/spanish/{model}_{size}/` produces:

| File | Contents |
|---|---|
| `training_stats.json` | param_count, non_embedding_param_count, training_time_hours, training_time_seconds, peak_training_memory_mb, disk_size_mb, compression_ratio (hybrid) |
| `training_steps_log.csv` | step, bytes_seen, loss, lr, grad_norm, tok_per_sec, peak_mem_mb, stage compression ratios (hybrid) |
| `validation_log.csv` | step, bytes_seen, val_loss, val_bpb |
| `config.json` | Full training config |
| `checkpoint_best.pt` | Best validation BPB checkpoint (~2.2 GB, full training state) |
| `checkpoint_final.pt` | Final checkpoint at end of training |
| `checkpoint_milestone_<N>B.pt` | Checkpoint at each milestone (6.25B, 12.5B, 18.75B, 25B bytes) |
| `model_deploy.pt` | Compact ternary-weight export (~36-40 MB for 150M, auto-generated by `generate_results.py`) |
| `*_inference_profile.json` | Prefill latency + decode throughput (generated by `profile_inference.py`) |

Intermediate step checkpoints are automatically deleted at end — only final, best, and milestone survive.

Results CSVs are at `runs/spanish/results_<model>_<size>.csv` (parent directory, not inside per-run subdirectory).

## Run order (10 runs)

By size tier, sequentially (one at a time to avoid GPU contention):
1. 150M: hybrid, transformer, matmulfree, hybrid_attn
2. 350M: hybrid, transformer, matmulfree
3. 750M: hybrid, transformer, matmulfree

Use `tmux new -s name -d 'cmd'` to keep runs alive after SSH disconnect. Use `python generate_results.py --output results_tier.csv` after each tier.
