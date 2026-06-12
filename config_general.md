# HNetBit General-Purpose Configurations

Two tiers. One works today with the current codebase on 8 GPUs. One requires engineering.

---

## Tier 1: Practical Best (~3.1B params)

Works with data-parallel training on 8×H100/A100 80GB.  
Add the entry to `model_factory.py:57-81` and wire in `DistributedDataParallel` (see below).

### Architecture

```
d_model:              [1280, 1792, 2560, 3328]    # 4-stage hierarchy
num_blocks:           [[4,0,4], [4,0,4], [6,0,6], [24]]
num_heads:            8
expand_ratio:         2
hidden_ratio:         4
innermost_use_attention:  True
attention_window_size:    256
attention_layers_pattern: "xaxa"    # 12 attn + 12 HGRN in innermost
max_position_embeddings: 32768
use_short_conv:       True
conv_size:            4
share_conv_kernel:    True
```

### Training

| Parameter | Value |
|---|---|
| `byte_seq_length` | 16384 |
| `total_training_bytes` | 500 000 000 000 (500B) |
| `batch_size` (per GPU) | 1 |
| `gradient_accumulation_steps` | 32 |
| Effective batch | 8 GPUs × 1 × 32 = 256 |
| `learning_rate` | 5e-4 |
| `warmup_fraction` | 0.02 |
| `stable_fraction` | 0.78 |
| `decay_fraction` | 0.20 |
| `min_lr_ratio` | 0.0 |
| `weight_decay` | 0.01 |
| `max_grad_norm` | 1.0 |
| `bf16` | True |
| `gradient_checkpointing` | True |

### Hardware

| Metric | Value |
|---|---|
| GPUs | 8× H100 80GB (or A100 80GB) |
| VRAM per GPU | ~62 GB |
| Total steps | 119 000 |
| Estimated time | 3-4 weeks |

### Multi-GPU (required for 8-GPU training)

`train_spanish.py` currently has **zero multi-GPU support** — no `DistributedDataParallel`, no `accelerate`. Running it on a multi-GPU machine uses one GPU.

Add `DistributedDataParallel` before running Tier 1. The changes needed:

**1. Wrap the model** in `train_spanish.py:82`:
```python
if torch.distributed.is_initialized():
    self.model = torch.nn.parallel.DistributedDataParallel(
        self.model, device_ids=[local_rank], output_device=local_rank
    )
```

**2. Use `DistributedSampler`** in `create_dataloaders` (`data_spanish.py`):
```python
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset, shuffle=True, seed=config.seed)
```

**3. Gate checkpointing, logging, and eval to rank 0**:
```python
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    self._save_checkpoint(...)
    self._evaluate_and_log(...)
```

**4. Launch with `torchrun`**:
```bash
torchrun --nproc_per_node=8 train_spanish.py --model hybrid_attn --size 3B \
  --total_tokens 500000000000 --batch_size 1 --grad_accum 32 \
  --lr 5e-4
```

Once DDP is wired in, Tier 1 fits comfortably (`model + optimizer = 43 GB + ~15 GB activations = ~58 GB`, well within 80 GB).

Requires adding to `HYBRID_CONFIGS` in `model_factory.py`:

```python
"3B": dict(
    d_model=[1280, 1792, 2560, 3328],
    num_blocks=[[4, 0, 4], [4, 0, 4], [6, 0, 6], [24]],
    num_heads=8,
    expand_ratio=2,
    hidden_ratio=4,
),
```

---

## Tier 2: Aspirational Best (~7B+ params)

Requires pipeline parallelism engineering.  
Split outer stages across GPU groups, replicate innermost.

### Architecture

```
d_model:              [2048, 2816, 3584, 4096]    # 4-stage hierarchy
num_blocks:           [[6,0,6], [6,0,6], [8,0,8], [32]]
num_heads:            16
expand_ratio:         2
hidden_ratio:         4
innermost_use_attention:  True
attention_window_size:    512
attention_layers_pattern: "xaxa"    # 16 attn + 16 HGRN in innermost
max_position_embeddings: 65536
use_short_conv:       True
conv_size:            4
share_conv_kernel:    True
```

### Training

| Parameter | Value |
|---|---|
| `byte_seq_length` | 32768 |
| `total_training_bytes` | 1 000 000 000 000 (1T) |
| `batch_size` (per GPU) | 1 |
| `gradient_accumulation_steps` | 64 |
| `learning_rate` | 3e-4 |
| `warmup_fraction` | 0.02 |
| `stable_fraction` | 0.78 |
| `decay_fraction` | 0.20 |

### Hardware

| Metric | Value |
|---|---|
| GPUs | 16-32× H100 (pipeline parallel) |
| Total steps | 119 000 |
| Estimated time | 6-8 weeks |
| Engineering | Pipeline parallelism, custom data pipeline |

---

## Why 4 stages?

With dynamic chunking targeting ~5× compression per stage:

```
16384 → ~3277 → ~655 → ~131  (positions at each stage)
```

At 131 positions, `window_size=256` is effectively **full attention** — every token attends to every other token at the innermost level, at a tiny fraction of the O(L²) cost. This is the key advantage of the hierarchical architecture for long-context models.

---

## Training data

No configuration change matters as much as this. Spanish Billion Words is 8.7 GB of monologue — it will never produce a conversational model regardless of architecture. For a generally capable model you need diverse byte-level data:

| Data type | Minimum size | Sources |
|---|---|---|
| Web text (multilingual) | 200B+ bytes | mC4, CulturaX, HPLT |
| Code | 50B+ bytes | The Stack v2, StarCoder (raw) |
| Books / academic | 20B+ bytes | Project Gutenberg, SciELO |
| Chat / conversation | 10B+ bytes | OpenAssistant, UltraChat (translated) |

Byte-level means no tokenizer artifacts. Every language, every format, every encoding just works as raw bytes. This is the architecture's superpower for data mixing: you concatenate files as-is and they become training data.

**Recommended path**: Train Tier 1 on a large multilingual web corpus (~400 GB raw), then fine-tune on conversational data. This yields an actually good generative model in ~4-5 weeks.

---

## Divisibility verification

All stage dimensions satisfy the two codebase constraints:

- HGRN: `d_model × expand_ratio % num_heads == 0` (`hgrn_bit.py:96`)
- Attention: `d_model % num_heads == 0` (`attention.py:102`)

### Tier 1 (3B)

| Stage | d_model | `d×2/8` (HGRN) | `d/8` (Attention) |
|---|---|---|---|
| 0 | 1280 | 320 | 160 |
| 1 | 1792 | 448 | 224 |
| 2 | 2560 | 640 | 320 |
| 3 | 3328 | 832 | 416 |

### Tier 2 (7B)

| Stage | d_model | `d×2/16` (HGRN) | `d/16` (Attention) |
|---|---|---|---|
| 0 | 2048 | 256 | 128 |
| 1 | 2816 | 352 | 176 |
| 2 | 3584 | 448 | 224 |
| 3 | 4096 | 512 | 256 |
