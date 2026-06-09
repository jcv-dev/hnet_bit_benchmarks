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
| `generate_results.py` | Aggregates per-run results into a comparison CSV |
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

## Architecture details

| Size | transformer params | matmulfree params | hybrid params |
|---|---|---|---|
| 150M | 190M (113M non-emb) | 114M | 138M |
| 350M | 505M (403M non-emb) | 309M | 419M |
| 750M | 1060M (906M non-emb) | 794M | 1026M |

Non-embedding params exclude lookup tables (embedding + LM head). Transformer has ~77-154M embedding overhead from 50K BPE vocab. Byte-level models use 256 vocab. Report both in thesis.

## Framework quirks

- **Triton JIT compilation**: required for fast training. Blocked in unprivileged Docker containers. Set `HNETBIT_DISABLE_TRITON=1` to fall back to naive PyTorch loops (3-10x slower). Use VM templates on Vast.ai or RunPod to avoid this.
- **HuggingFace token**: the `huggingface-cli` shell command may not exist in cloud templates. Use `python3 -c "from huggingface_hub import login; login(token='...')"` or write token to `HF_HOME/token` directly.
- **BPE tokenization RAM (fixed)**: tokenizing 8.7 GB corpus with gpt2 produces ~2.9B tokens (~88 GB as int32). Both `_write_byte_corpus` and `_write_bpe_corpus` run in `multiprocessing.Process` children to prevent CPython's pymalloc heap accumulation. The BPE worker is recycled every 50 chunks (`CHUNKS_PER_WORKER=50` in `data_spanish.py:195`) to cap child RSS. Parent process RSS stays flat (~5-10 GB). Do NOT use `open_memmap` — OS page cache balloons.
- **max_position_embeddings**: transformer tiny config had 256 which conflicted with DataLoader's 1792-token sequences. Fixed to 2048. Real sizes (150M+) already use 2048.
- **Transformer embedding overhead**: 50K vocab x hidden_size x 2 (embed + lm_head) is ~77M params at 150M that contribute zero FLOPs. Track `Non_Emb_Params_M` for fair comparison.
- **Gradient checkpointing**: HNetBitForCausalLM is not HF-compatible. Falls back to `model.backbone._gradient_checkpointing = True`.

## Output files per run

Each run at `runs/spanish/{model}_{size}/` produces:

| File | Contents |
|---|---|
| `training_stats.json` | param_count, non_embedding_param_count, training_time, peak_memory, disk_size_mb, compression_ratio (hybrid) |
| `training_steps_log.csv` | step, bytes_seen, loss, lr, grad_norm, tok_per_sec |
| `validation_log.csv` | step, bytes_seen, val_loss, val_bpb |
| `config.json` | Full training config |

Intermediate step checkpoints are automatically deleted at end — only final, best, and milestone survive.

## Run order (10 runs)

By size tier, sequentially (one at a time to avoid GPU contention):
1. 150M: hybrid, transformer, matmulfree, hybrid_attn
2. 350M: hybrid, transformer, matmulfree
3. 750M: hybrid, transformer, matmulfree

Use `tmux new -s name -d 'cmd'` to keep runs alive after SSH disconnect. Use `python generate_results.py --output results_tier.csv` after each tier.
