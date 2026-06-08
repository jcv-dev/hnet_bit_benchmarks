# HNetBit — Training to Generation Guide

This document covers standalone training with the `hnet_bit` package (using `python -m hnet_bit.train`). For the Spanish Billion Words benchmark that compares the hybrid model against transformer and MatMulFree baselines, see the parent directory's `README.md` and the `train_spanish.py` script. The benchmark pipeline is the primary training system for thesis experiments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Sources](#data-sources)
3. [Training](#training)
4. [Generation](#generation)
5. [Config File Reference](#config-file-reference)

---

## Quick Start

All commands assume you are in the repository root (`hnet_bit/`) with the
virtual environment activated.

```bash
cd /path/to/hnet_bit
source venv/bin/activate
```

### Minimal run (synthetic data, no downloads)

```bash
# Train a small hierarchical model for 100 steps on random bytes
python -m hnet_bit.train \
    --model_config configs/hnet_bit_1stage.json \
    --max_steps 100 \
    --batch_size 4 \
    --max_seq_length 128 \
    --output_dir runs/quick_test

# Generate from the last checkpoint
python -m hnet_bit.generate \
    --model_path runs/quick_test/checkpoint_latest.pt \
    --prompt "Hello world" \
    --max_tokens 50
```

---

## Data Sources

The training pipeline supports three data sources, checked in this order:

| Priority | Source | Flag | Description |
|----------|--------|------|-------------|
| 1 | **HuggingFace dataset** | `--dataset` | Any text dataset on the HF Hub |
| 2 | **Local text file** | `--data_path` | A `.txt` file on disk |
| 3 | **Synthetic** | *(none)* | Random bytes, used automatically as fallback |

### Using a HuggingFace dataset

Pass `--dataset <name>` and the pipeline downloads, tokenises to bytes, and
caches the result automatically.  The text column is **auto-detected** for
common datasets.

```bash
# TinyStories — "text" column detected automatically
python -m hnet_bit.train \
    --dataset roneneldan/TinyStories \
    --model_config configs/hnet_bit_1stage.json \
    --output_dir runs/tinystories_1stage

# WikiText-103 — needs a config/subset name
python -m hnet_bit.train \
    --dataset wikitext \
    --dataset_config wikitext-103-raw-v1 \
    --model_config configs/hnet_bit_2stage.json \
    --output_dir runs/wikitext_2stage

# Alpaca — join instruction + output columns
python -m hnet_bit.train \
    --dataset tatsu-lab/alpaca \
    --text_columns instruction output \
    --model_config configs/hnet_bit_1stage.json \
    --output_dir runs/alpaca_1stage

# Explicit text column (when auto-detect doesn't match)
python -m hnet_bit.train \
    --dataset bookcorpus \
    --text_column text \
    --output_dir runs/bookcorpus
```

**Useful flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | — | HuggingFace dataset identifier |
| `--dataset_config` | `None` | Subset / config name (e.g. `wikitext-103-raw-v1`) |
| `--text_column` | `auto` | Column with text. `auto` tries common names |
| `--text_columns` | `None` | Multiple columns to concatenate (space-separated) |
| `--streaming` | `False` | Stream instead of downloading all at once |
| `--max_samples` | `None` | Cap number of samples (useful for debugging) |
| `--train_split` | `train` | Name of the training split |
| `--val_split` | `validation` | Name of the validation split |

**Auto-detected column names** (checked in order):
`text`, `content`, `story`, `document`, `passage`, `sentence`, `paragraph`,
`article`, `body`.
Multi-column pairs: `instruction`+`output`, `instruction`+`response`,
`prompt`+`completion`, `prompt`+`response`, `input`+`output`,
`question`+`answer`, `context`+`response`, `human`+`assistant`.

If none match, the first string-valued column is used as a last resort.

### Pre-downloading with prepare_dataset.py

For large datasets or repeated experiments it's faster to download once:

```bash
# Download and save to data/tinystories/
python -m hnet_bit.scripts.prepare_dataset \
    --dataset roneneldan/TinyStories \
    --output_dir data/tinystories

# Then train from the local file
python -m hnet_bit.train \
    --data_path data/tinystories/train_bytes.pt \
    --output_dir runs/tinystories_1stage
```

`prepare_dataset.py` flags:

```
--dataset           (required) HF dataset name
--output_dir        (required) Where to save .pt files
--dataset_config    HF subset name
--text_column       Column name or "auto"
--text_columns      Multiple columns to join
--separator         Separator between joined columns (default: "\n\n")
--train_split       Training split (default: "train")
--val_split         Validation split (default: "validation", "none" to skip)
--test_split        Test split (default: skip)
--max_seq_length    Byte sequence length (default: 512)
--max_samples       Limit samples per split
--streaming         Stream the dataset
```

### Using a local text file

Any plain text file works:

```bash
python -m hnet_bit.train \
    --data_path /path/to/corpus.txt \
    --model_config configs/hnet_bit_1stage.json \
    --output_dir runs/local_corpus
```

The file is read as raw bytes, split into chunks of `max_seq_length`, and
automatically divided 90/10 into train/val.

---

## Training

### Hierarchical HNetBit Models

Dynamic chunking + multi-resolution stages from HNet, with MatMulFree
ternary blocks.

**1-stage** (faster, smaller):

```bash
python -m hnet_bit.train \
    --model_config configs/hnet_bit_1stage.json \
    --dataset roneneldan/TinyStories \
    --max_steps 15000 \
    --batch_size 32 \
    --lr 3e-4 \
    --bf16 \
    --output_dir runs/hier_1stage
```

**2-stage** (larger, more compression):

```bash
python -m hnet_bit.train \
    --model_config configs/hnet_bit_2stage.json \
    --dataset roneneldan/TinyStories \
    --max_steps 20000 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr 3e-4 \
    --bf16 \
    --output_dir runs/hier_2stage
```

### Common training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model_config` | — | Path to model config JSON (required) |
| `--training_config` | — | Path to training config JSON |
| `--output_dir` | `runs/hnetbit_experiment` | Checkpoint & log directory |
| `--max_steps` | `10000` | Total training steps |
| `--batch_size` | `32` | Per-device batch size |
| `--lr` | `3e-4` | Peak learning rate |
| `--max_seq_length` | `512` | Sequence length in bytes |
| `--bf16` | `False` | Enable BF16 mixed precision |
| `--fp16` | `False` | Enable FP16 mixed precision |
| `--resume_from` | — | Resume from a checkpoint file |
| `--seed` | `42` | Random seed |

CLI flags override values from `--training_config`.

### Resuming training

```bash
python -m hnet_bit.train \
    --model_config configs/hnet_bit_1stage.json \
    --resume_from runs/hier_1stage/checkpoint_step_5000.pt \
    --output_dir runs/hier_1stage
```

---

## Generation

### Single prompt

```bash
python -m hnet_bit.generate \
    --model_path runs/tinystories_1stage/checkpoint_latest.pt \
    --prompt "Once upon a time" \
    --max_tokens 200 \
    --temperature 0.8
```

### Interactive mode

```bash
python -m hnet_bit.generate \
    --model_path runs/hier_1stage/checkpoint_latest.pt \
    --interactive
```

### Batch generation from file

```bash
# prompts.txt: one prompt per line
python -m hnet_bit.generate \
    --model_path runs/tinystories_1stage/checkpoint_latest.pt \
    --prompt_file prompts.txt \
    --output_file outputs.txt \
    --max_tokens 150
```

### Multiple samples per prompt

```bash
python -m hnet_bit.generate \
    --model_path runs/tinystories_1stage/checkpoint_latest.pt \
    --prompt "The cat" \
    --num_samples 5 \
    --temperature 1.0
```

### Generation flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | — | Path to checkpoint |
| `--model_config` | — | Path to model config JSON |
| `--prompt` | — | Input prompt text |
| `--prompt_file` | — | File with one prompt per line |
| `--interactive` | — | Enter prompts interactively |
| `--output_file` | — | Save outputs to file |
| `--max_tokens` | `100` | Maximum tokens to generate |
| `--temperature` | `1.0` | Sampling temperature (lower = more deterministic) |
| `--top_k` | `50` | Top-k sampling |
| `--top_p` | `0.9` | Nucleus sampling threshold |
| `--repetition_penalty` | `1.0` | Penalise repeated tokens (1.0 = off) |
| `--greedy` | — | Greedy decoding (deterministic) |
| `--num_samples` | `1` | Samples to generate per prompt |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--seed` | — | Seed for reproducibility |

---

## Config File Reference

Config files live in `configs/`.  There are two kinds:

### Model configs

Define the architecture.  Passed via `--model_config`.

#### `hnet_bit_1stage.json` — 1-stage hierarchical

```
d_model             [512, 768]          Per-stage dimensions
num_blocks          [[4, 0, 4], [8]]    Blocks per stage: [pre-chunk, routing, post-chunk]
                                        Stage 0: 4 encoder + 4 decoder (with routing/chunking)
                                        Stage 1: 8 blocks (innermost, no routing)
```

The list structure of `num_blocks` encodes the hierarchical layout.  Each
inner list is `[encoder_blocks, unused, decoder_blocks]`.  The
final stage (innermost) has a single number (no chunking at the top).

#### `hnet_bit_2stage.json` — 2-stage hierarchical

```
d_model             [512, 768, 1024]
num_blocks          [[4, 0, 4], [4, 0, 4], [8]]
```

Three resolution levels: byte-level (512d) → chunk-level (768d) → top-level
(1024d).


### Training configs

Define hyperparameters.  Passed via `--training_config`.  CLI flags override
any value set here.

#### `training_small.json` — Quick test

```
output_dir                  "runs/small_test"
max_steps                   100
batch_size                  4
gradient_accumulation_steps 1
learning_rate               1e-3
warmup_steps                10
lr_scheduler                "cosine"
max_seq_length              128
eval_interval               50
save_interval               50
log_interval                10
```

#### `training_hierarchical.json` — Full hierarchical run

```
output_dir                  "runs/hierarchical"
max_steps                   10000
batch_size                  8
gradient_accumulation_steps 4       → effective batch = 32
learning_rate               6e-4
weight_decay                0.1
warmup_steps                500
bf16                        true
```

### Experiment configs (reference)

The `training_*_tinystories.json` files are full experiment descriptors with
model + data + training + evaluation sections.  They are **not** directly
consumed by `train.py` (which takes separate `--model_config` and
`--training_config`), but serve as documentation of recommended
hyperparameters.

#### Key hierarchical-specific fields

| Field | Recommended | Description |
|-------|-------------|-------------|
| `lr_multipliers` | `[1.5, 1.0]` (1-stage) / `[2.0, 1.5, 1.0]` (2-stage) | Per-stage LR multipliers. Higher = faster learning at lower stages. |
| `lambda_lb` | `0.03`–`0.05` | Load-balancing loss weight for the routing module. `0.0` disables it. |
| `downsampling_factor` | `2.5` | Expected token compression ratio per stage. |

---

## End-to-End Examples

### Example 1: HNetBit on TinyStories (HF)

```bash
# 1. Train 1-stage model
python -m hnet_bit.train \
    --model_config configs/hnet_bit_1stage.json \
    --dataset roneneldan/TinyStories \
    --max_steps 10000 \
    --batch_size 32 \
    --lr 3e-4 \
    --bf16 \
    --output_dir runs/hnet_ts

# 2. Generate
python -m hnet_bit.generate \
    --model_path runs/hnet_ts/checkpoint_latest.pt \
    --prompt "Once upon a time there was a little" \
    --max_tokens 200 \
    --temperature 0.8 \
    --top_p 0.9
```

### Example 2: 2-stage model on WikiText (HF)

```bash
# 1. Train 2-stage hierarchical model
python -m hnet_bit.train \
    --model_config configs/hnet_bit_2stage.json \
    --dataset wikitext \
    --dataset_config wikitext-103-raw-v1 \
    --max_steps 15000 \
    --batch_size 16 \
    --lr 3e-4 \
    --bf16 \
    --output_dir runs/hier_wiki

# 2. Generate
python -m hnet_bit.generate \
    --model_path runs/hier_wiki/checkpoint_latest.pt \
    --prompt "The history of" \
    --max_tokens 300
```

### Example 3: Local text file

```bash
# 1. Train on local corpus
python -m hnet_bit.train \
    --model_config configs/hnet_bit_1stage.json \
    --data_path /path/to/my_corpus.txt \
    --max_steps 5000 \
    --batch_size 16 \
    --output_dir runs/local

# 2. Generate
python -m hnet_bit.generate \
    --model_path runs/local/checkpoint_latest.pt \
    --interactive
```

### Example 4: Instruction dataset (multi-column)

```bash
# 1. Train on Alpaca (instruction + output joined)
python -m hnet_bit.train \
    --model_config configs/hnet_bit_1stage.json \
    --dataset tatsu-lab/alpaca \
    --text_columns instruction output \
    --max_steps 8000 \
    --batch_size 32 \
    --output_dir runs/alpaca

# 2. Generate
python -m hnet_bit.generate \
    --model_path runs/alpaca/checkpoint_latest.pt \
    --prompt "Explain what a neural network is" \
    --max_tokens 200
```

### Example 5: Quick debug run (limited samples)

```bash
python -m hnet_bit.train \
    --model_config configs/hnet_bit_1stage.json \
    --dataset roneneldan/TinyStories \
    --max_samples 500 \
    --max_steps 50 \
    --batch_size 4 \
    --max_seq_length 128 \
    --output_dir runs/debug
```
