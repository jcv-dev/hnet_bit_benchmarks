# HNetBit Architecture — Detailed Technical Reference

> **Hierarchical H-Net with Ternary BitLinear Weights**
> A MatMul-free small language model that combines H-Net's multi-stage dynamic chunking with ternary weight quantization ({-1, 0, +1}).

---

## Table of Contents

1. [Model Classification](#1-model-classification)
2. [High-Level Overview](#2-high-level-overview)
3. [Input Representation — Byte-Level Tokenization](#3-input-representation--byte-level-tokenization)
4. [Top-Level Wrapper — HNetBitForCausalLM](#4-top-level-wrapper--hnetbitforcausallm)
5. [The Core Primitive — BitLinear (Ternary Quantization)](#5-the-core-primitive--bitlinear-ternary-quantization)
6. [The Recurrence Engine — HGRN (Hierarchically Gated Recurrent Network)](#6-the-recurrence-engine--hgrn)
7. [Activations — SwiGLU and FusedRMSNormSwishGate](#7-activations--swiglu-and-fusedrmsnormswishgate)
8. [The Block — HGRNBitBlock](#8-the-block--hgrnbitblock)
9. [The Stack — HGRNBitStack](#9-the-stack--hgrnbitstack)
10. [Dynamic Chunking — RoutingModuleBit, ChunkLayer, DeChunkLayer](#10-dynamic-chunking--routingmodulebit-chunklayer-dechunklayer)
11. [The Recursive Backbone — HNetBit](#11-the-recursive-backbone--hnetbit)
12. [End-to-End Forward Pass (Training)](#12-end-to-end-forward-pass-training)
13. [Autoregressive Generation (Inference)](#13-autoregressive-generation-inference)
14. [Configuration Presets](#14-configuration-presets)
15. [Parameter Efficiency and the MatMul-Free Property](#15-parameter-efficiency-and-the-matmul-free-property)
16. [File Map](#16-file-map)

---

## 1. Model Classification

### Architecture Type
**HNetBit is a decoder-only, autoregressive language model with recurrent (non-transformer) architecture.**

| Property | Classification |
|----------|----------------|
| **Model Type** | Decoder-only causal language model |
| **Sequence Mixing** | Recurrent (HGRN gated recurrence), **NOT transformer/attention-based** |
| **Weight Precision** | Ternary quantized ({-1, 0, +1}) — MatMul-free |
| **Hierarchy** | Multi-stage recursive with dynamic chunking |
| **Tokenization** | Byte-level (vocabulary size = 256) |
| **Training Objective** | Next-token prediction (cross-entropy) |

### Key Distinctions from Standard Architectures

- **Not a Transformer**: No self-attention mechanism. Uses HGRN (Hierarchically Gated Recurrent Network) for sequence mixing with O(L) complexity but O(1) memory per position.
  
- **Not an Encoder-Decoder**: Pure decoder architecture for autoregressive generation. The "encoder" and "decoder" components at each hierarchy stage are both decoder blocks that process the same causal sequence — terminology inherited from H-Net but both operate in a left-to-right causal manner.

- **Not RNN/LSTM**: While recurrent, it uses modern gated linear recurrences (HGRN) with better parallelization properties than traditional RNNs. Training uses chunk-wise parallel processing via recurrent_fuse kernels.

- **MatMul-Free**: All matrix multiplications replaced with ternary weight operations (additions/subtractions only), enabling extreme efficiency on specialized hardware.

### Comparable Models
- **Similar to**: RetNet, RWKV, Mamba (linear recurrent alternatives to transformers)
- **Unique aspect**: Combines hierarchical dynamic chunking (variable-length compression) with ternary quantization

---

## 2. High-Level Overview

HNetBit is a **hierarchical, recurrent language model** that operates on raw bytes (vocab = 256). It draws from two research lines:

| Concept | Source | What it contributes |
|---------|--------|---------------------|
| **H-Net** | Hwang et al. | Multi-stage hierarchy via dynamic chunking — the model learns to segment a sequence into variable-length chunks, processes them at progressively higher dimensions, and reconstructs the original length |
| **MatMulFree LM** | Zhu et al. | Ternary weight quantization (`{-1, 0, +1}`) and HGRN-based recurrence — all linear projections use `BitLinear`, replacing dense float multiply-accumulate with additions/subtractions |

The merger produces a model where:
- **Every linear layer** uses ternary weights (1.58 bits per weight).
- **Sequence mixing** uses HGRN gated recurrence (O(1) state per step) instead of attention.
- **Sequence length reduction** happens adaptively through learned cosine-similarity boundary detection — not fixed pooling.
- **Multiple stages** nest recursively: each stage encodes, identifies boundaries, chunks, recurses into a deeper stage at higher dimension, then reconstructs and decodes.

Schematically (2-stage example, 3 dimension levels):

```
Input bytes (B, L)
  │
  ▼
Embedding(256, 512)
  │
  ▼
┌─── Stage 0 (d=512) ─────────────────────────────────────┐
│ Encoder: 4 × HGRNBitBlock(512)                          │
│   │                                                      │
│   ├──► residual_proj (FP32) ──────────────────────────┐  │
│   │                                                   │  │
│   ├──► RoutingModuleBit → boundaries                  │  │
│   │                                                   │  │
│   ├──► ChunkLayer → (B, M, 512)                      │  │
│   │         │                                         │  │
│   │    ┌─── Stage 1 (d=768) ────────────────────┐    │  │
│   │    │ Pad 512→768                             │    │  │
│   │    │ Encoder: 4 × HGRNBitBlock(768)          │    │  │
│   │    │ RoutingModuleBit → ChunkLayer           │    │  │
│   │    │      │                                  │    │  │
│   │    │ ┌─── Stage 2 (d=1024) [innermost] ──┐  │    │  │
│   │    │ │ Pad 768→1024                       │  │    │  │
│   │    │ │ 8 × HGRNBitBlock(1024)             │  │    │  │
│   │    │ │ RMSNorm                            │  │    │  │
│   │    │ │ Unpad → 768                        │  │    │  │
│   │    │ └────────────────────────────────────┘  │    │  │
│   │    │ DeChunkLayer (EMA) → (B, M, 768)        │    │  │
│   │    │ out · STE(p) + residual                 │    │  │
│   │    │ Decoder: 4 × HGRNBitBlock(768)          │    │  │
│   │    │ Unpad → 512                             │    │  │
│   │    └─────────────────────────────────────────┘    │  │
│   │                                                   │  │
│   ◄──── DeChunkLayer (EMA) → (B, L, 512) ◄───────────┘  │
│         out · STE(p) + residual                          │
│                                                          │
│ Decoder: 4 × HGRNBitBlock(512)                          │
└──────────────────────────────────────────────────────────┘
  │
  ▼
BitLinear LM Head (512 → 256)
  │
  ▼
Byte logits (B, L, 256)
```

---

## 3. Input Representation — Byte-Level Tokenization

The model operates on **byte-level tokens** — each token is one byte from the range `[0, 255]`.

- **Vocabulary size**: 256 (fixed)
- **Tokenization**: UTF-8 bytes (one byte = one token)
- **Special tokens**: `bos_token_id = 254`, `eos_token_id = 255` (configurable)

**Byte-level tokenization** means:
- Each byte in the UTF-8 encoding becomes a discrete token
- Multi-byte characters (emoji, Chinese, etc.) span multiple tokens
- Example: `"Hello :D"` → `[72, 101, 108, 108, 111, 32, 240, 159, 152, 128]` (10 tokens)
  - ASCII characters: 1 character ≈ 1 token
  - Unicode characters: 1 character = 1-4 tokens (depending on encoding)

An `nn.Embedding(256, d_model[0])` maps each byte ID to a dense vector at the outermost dimension. This embedding uses **full-precision** (float32/bfloat16) — it is the only non-ternary projection touching the input.

### Relationship to H-Net's "End-to-End" Design

**Important clarification**: After examining the original H-Net codebase (`hnet-main/hnet/utils/tokenizers.py`), **H-Net also uses byte-level tokenization** with the same approach as this implementation. The "end-to-end" claim in the paper refers specifically to:

1. **End-to-end hierarchical segmentation**: Boundaries learned via gradient descent (not fixed chunk sizes)
2. **End-to-end optimization**: All hierarchy stages jointly trained
3. **Eliminating learned tokenizers**: Using bytes (256 vocab) instead of BPE/WordPiece (50k vocab)

#### What H-Net Replaces

The innovation is replacing **this pipeline**:
```python
# Traditional LM with BPE tokenization:
text → BPE tokenizer (50k+ vocab, language-specific) → Transformer → BPE detokenizer → text
```

With **this pipeline**:
```python
# H-Net with byte tokenization:
text → bytes (256 vocab, universal) → dynamic chunking → hierarchical processing → dechunking → bytes → text
```

#### The Key Innovation: Learned Segmentation vs. Fixed Tokens

| Aspect | BPE/WordPiece Tokenization | H-Net Byte + Chunking |
|--------|---------------------------|----------------------|
| **Vocabulary** | 30k-50k learned tokens | 256 bytes (fixed) |
| **Segmentation** | Fixed (determined during tokenizer training) | Learned (dynamic boundaries) |
| **Language Coverage** | Language-specific (artifacts in other languages) | Universal (any UTF-8 text) |
| **OOV Handling** | UNK tokens for unknown sequences | No OOV (all bytes represented) |
| **Hierarchy** | Flat (single level) | Multi-stage (learned compression) |

#### What "End-to-End" Does NOT Mean

Based on the original H-Net codebase:
- MISSING **NOT** eliminating all discrete tokenization (bytes are still discrete symbols)
- MISSING **NOT** learning from continuous signals (audio waveforms, pixels)
- MISSING **NOT** learning what "bytes" represent (UTF-8 encoding is predetermined)

Instead, "end-to-end" means:
- OK Learning **how to segment** bytes into meaningful chunks
- OK Learning **multiple levels** of abstraction hierarchically
- OK Optimizing **all stages jointly** without intermediate supervision

#### This Implementation Matches Original H-Net OK

**Your implementation is architecturally correct** and follows the same design as the original H-Net:
- Same byte tokenization (vocab_size=256)
- Same dynamic boundary learning
- Same hierarchical chunking/dechunking
- **Difference**: Uses HGRN+BitLinear instead of Attention+Dense layers

#### Advantages Over BPE Tokenization

Byte-level operation with learned chunking provides:
- OK **Universal vocabulary**: Works on any language, code, or UTF-8 data
- OK **No vocabulary artifacts**: No tokenizer-specific biases or boundaries
- OK **No OOV tokens**: All possible byte sequences are valid
- OK **Learned segmentation**: Boundaries adapt to content and context
- OK **Better cross-lingual transfer**: Especially for languages poorly covered by BPE training

#### The Tradeoff

- WARNING️ **Longer sequences**: Byte-level is 3-4× longer than subword tokenization
- OK **Mitigated by hierarchy**: Dynamic chunking compresses effectively at deeper stages

---

## 4. Top-Level Wrapper — HNetBitForCausalLM

**File**: `models/hnet_bit.py` — class `HNetBitForCausalLM`

This is the full causal language model, compatible with HuggingFace `transformers`:

```
HNetBitForCausalLM
├── embeddings: nn.Embedding(256, d_model[0])
├── backbone:   HNetBit(config, stage_idx=0)   ← recursive hierarchy
└── lm_head:    BitLinear(d_model[0], 256)      ← ternary output projection
```

**Forward pass** (simplified):
1. `hidden_states = embeddings(input_ids)`  →  `(B, L, d_model[0])`
2. `hidden_states, boundary_preds = backbone(hidden_states, mask)`
3. `logits = lm_head(hidden_states)`  →  `(B, L, 256)`
4. If labels provided: shift-by-one cross-entropy loss.

**Key compatibility features**:
- Extends `PreTrainedModel` + `GenerationMixin` for `model.generate()` support.
- Overrides `_prepare_cache_for_generation` to allocate `HNetBitCache` instead of `DynamicCache`.
- Exposes `num_hidden_layers` (sum of all blocks across stages) and `hidden_size` (d_model[0]) for transformers internals.

---

## 5. The Core Primitive — BitLinear (Ternary Quantization)

**File**: `ops/bitnet.py` — class `BitLinear`

Every linear projection in the model (except `nn.Embedding` and `residual_proj`) is a `BitLinear` layer. This is an `nn.Linear` subclass that applies **ternary weight quantization** and **8-bit activation quantization** during the forward pass.

### 5.1 Weight Quantization (1.58-bit)

```python
def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u
```

Steps:
1. Compute the per-tensor mean of absolute values: `α = mean(|W|)`.
2. Scale: `W' = W / α`.
3. Round to nearest integer and clamp to `{-1, 0, +1}`.
4. Rescale back by `α`.

The **effective weights** at inference are always one of `{-α, 0, +α}`, meaning multiplication by a weight element becomes at most an addition or subtraction. This is the **MatMul-free** property.

### 5.2 Activation Quantization (8-bit)

```python
def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y
```

Per-token scaling to the `[-128, 127]` integer range, then immediately de-scaled back to floating point. This constrains the dynamic range of activations before the ternary matmul.

### 5.3 Straight-Through Estimator (STE)

During training, the full-precision weights and activations are maintained. Quantization is applied **only in the forward pass**, and gradients flow through the quantization as if it were an identity function:

```python
x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
w_quant = w + (weight_quant(w) - w).detach()
y = F.linear(x_quant, w_quant)
```

The `.detach()` on the quantization difference means the gradient of the loss w.r.t. `w` is computed as if the quantization did not happen, but the forward value uses the quantized version.

### 5.4 Built-in RMSNorm

Each `BitLinear` contains an `RMSNorm` applied to the input **before** quantization:

```python
x_norm = self.norm(x)      # RMSNorm
x_quant = activation_quant(x_norm)  # quantize
w_quant = weight_quant(self.weight)  # quantize
y = F.linear(x_quant, w_quant)
```

This stabilizes the input distribution before the harsh quantization.

---

## 6. The Recurrence Engine — HGRN

**Files**: `ops/hgrn/recurrent_fuse.py`, `ops/hgrn/chunk.py`

HGRN (Hierarchically Gated Recurrent Network) replaces self-attention as the sequence mixing mechanism. It is a **linear recurrence** with a forget gate:

$$h_t = f_t \cdot h_{t-1} + i_t$$

where:
- $h_t \in \mathbb{R}^d$ is the hidden state at time $t$
- $f_t \in (0, 1)^d$ is the **forget gate** (element-wise)
- $i_t \in \mathbb{R}^d$ is the **input** (already gated — see Section 7)

### 6.1 Why Not Attention?

Standard self-attention requires $O(L^2)$ computation and $O(L)$ memory per head. HGRN requires:
- **Training**: $O(L)$ computation via chunked parallel scan (`chunk_hgrn`)
- **Inference**: $O(1)$ state per step — just `h_prev` of shape `(B, H, d_head)`

This makes generation constant-time per step regardless of context length.

### 6.2 Implementation Variants

| Kernel | When used | Implementation |
|--------|-----------|----------------|
| `fused_recurrent_hgrn` (Triton) | CUDA, forward+backward | Triton kernel with autotuned block sizes. Processes the recurrence sequentially but parallelizes across the head dimension |
| `chunk_hgrn` (Triton) | CUDA, forward+backward | Chunked parallel scan — breaks the sequence into chunks of 64, processes intra-chunk in parallel, then combines inter-chunk states |
| `_naive_recurrent_hgrn` (PyTorch) | CPU fallback | Pure Python loop: `for t in range(T): h = g[:,:,t] * h + x[:,:,t]` |
| `_naive_chunk_hgrn` (PyTorch) | CPU fallback | Pure Python loop with `exp(g)` parameterization for chunk mode |

The dispatcher checks `x.is_cuda` and selects the appropriate backend automatically.

---

## 7. Activations — SwiGLU and FusedRMSNormSwishGate

### 7.1 SwiGLU

**File**: `ops/activations.py`

SwiGLU is a gated activation function:

$$\text{SwiGLU}(x, y) = \text{SiLU}(x) \cdot y = \frac{x}{1 + e^{-x}} \cdot y$$

It is used in two places:
1. **HGRNBitAttention**: `i = SwiGLU(i_proj(x), 1 - f)` — the input is gated by the *complement* of the forget gate.
2. **HGRNBitMLP**: `out = SwiGLU(gate, y)` — the MLP uses a gated feed-forward pattern.

Implementation has two backends:
- **CUDA**: Custom jiterator with fused forward/backward kernels (avoids materializing intermediate sigmoid).
- **CPU**: `F.silu(x) * y` with manual backward.

### 7.2 FusedRMSNormSwishGate

**File**: `ops/fused_norm_gate.py`

Used as the output gate normalization in `HGRNBitAttention`:

$$\text{FusedRMSNormSwishGate}(x, \text{gate}) = \text{RMSNorm}(x) \cdot \text{gate} \cdot \sigma(\text{gate})$$

This fuses three operations — normalization, swish activation, and gating — into a single module. The `gate` signal comes from the recurrence output, and `x` comes from a separate `g_proj` linear projection.

---

## 8. The Block — HGRNBitBlock

**File**: `layers/hgrn_bit.py` — class `HGRNBitBlock`

Each block follows a **pre-norm residual** pattern with two sub-layers:

```
Input x
  │
  ├──────── residual ────────────┐
  ▼                              │
RMSNorm (attn_norm)              │
  │                              │
  ▼                              │
HGRNBitAttention                 │
  │                              │
  ▼                              ▼
  + ◄──────── addition ──────────┘
  │
  ├──────── residual ────────────┐
  ▼                              │
RMSNorm (mlp_norm)               │
  │                              │
  ▼                              │
HGRNBitMLP                       │
  │                              │
  ▼                              ▼
  + ◄──────── addition ──────────┘
  │
  ▼
Output
```

### 8.1 HGRNBitAttention — Step by Step

Given input `x` of shape `(B, L, D)`:

1. **Input projection**: `i = BitLinear_i(x)` → `(B, L, D)`
2. **Forget projection**: `f = σ(BitLinear_f(x))` → `(B, L, D)`, values in `(0, 1)`
3. **Input gating**: `i = SwiGLU(i, 1 - f)` — the input is modulated by the *complement* of the forget gate. When `f` is high (remember), `1-f` is low, suppressing new input. When `f` is low (forget), `1-f` is high, allowing new input through.
4. **Reshape for heads**: `i, f: (B, L, D) → (B, H, L, d_head)` where `H = num_heads` and `d_head = D / H`
5. **Gated recurrence**: `o = fused_recurrent_hgrn(i, f, h_prev)` — applies $h_t = f_t \cdot h_{t-1} + i_t$ across the sequence.
6. **Reshape back**: `o: (B, H, L, d_head) → (B, L, D)`
7. **Output gate**: `g = BitLinear_g(x)` — a separate projection of the *original* input `x`
8. **Gate normalization**: `o = FusedRMSNormSwishGate(g, o)` — normalizes `g`, gates with `o`
9. **Output projection**: `o = BitLinear_o(o)` → `(B, L, D)`

All four projections (`i_proj`, `f_proj`, `g_proj`, `o_proj`) use `BitLinear` (ternary).

### 8.2 HGRNBitMLP — Step by Step

Given input `x` of shape `(B, L, D)`:

1. **Up-projection**: `y = BitLinear_gate(x)` → `(B, L, 2·I)` where `I = intermediate_size`
   - `I` is computed as `round_up(⅔ · D · hidden_ratio, 256)` — by default with `hidden_ratio=4`, `I ≈ 2.67·D` rounded to the nearest multiple of 256.
2. **Split**: `gate, y = y.chunk(2, dim=-1)` → both `(B, L, I)`
3. **Gated activation**: `z = SwiGLU(gate, y)` → `(B, L, I)`
4. **Down-projection**: `out = BitLinear_down(z)` → `(B, L, D)`

Two `BitLinear` projections: `gate_proj` (up, 2×) and `down_proj` (down).

---

## 9. The Stack — HGRNBitStack

**File**: `models/hnet_bit.py` — class `HGRNBitStack`

A `HGRNBitStack` is simply:
- `N` sequential `HGRNBitBlock` modules
- A final `RMSNorm`

It is used as the **encoder**, **decoder**, or **innermost** network at each stage. Each stack operates at a single dimension `d_model[stage_idx]`.

When `config.innermost_use_attention=True`, the stack interleaves HGRN and attention blocks according to `config.attention_layers_pattern`. The pattern is a string of characters where `'x'` = HGRN and `'a'` = attention; for example `"xaxa"` produces HGRN-Attention-HGRN-Attention. If the pattern is shorter than the number of layers, it is extended cyclically. This is used by the `hybrid_attn` benchmark model.

The stack also manages the `HGRNBlockCache` — a flat list of per-layer recurrent states:
```python
cache.states[i] = (h_state,)  # one tensor per layer, shape (B, H, d_head)
```

---

## 10. Dynamic Chunking — RoutingModuleBit, ChunkLayer, DeChunkLayer

**File**: `ops/dynamic_chunking.py`

This is the mechanism that creates the hierarchy — it adaptively **reduces sequence length** between stages and **reconstructs** it afterward.

### 10.1 RoutingModuleBit — Boundary Detection

**Goal**: Decide where to split the sequence into chunks. Boundaries are placed where the semantic content changes significantly.

**Method**: Cosine similarity between consecutive tokens after projection.

Given encoder output `h` of shape `(B, L, D)`:

1. **Project**: `q = normalize(BitLinear_q(h[:-1]))`, `k = normalize(BitLinear_k(h[1:]))` — project consecutive pairs into a comparison space, then L2-normalize.
2. **Cosine similarity**: `cos_sim(t) = q_t · k_{t+1}` — dot product of normalized vectors.
3. **Boundary probability**: `p(t) = (1 - cos_sim(t)) / 2` — ranges from 0 (identical, no boundary) to 1 (opposite, strong boundary). Clamped to `[0, 1]`.
4. **Force first boundary**: `p(0) = 1.0` — the first token is always a boundary to ensure at least one chunk.
5. **Hard decision**: `boundary_mask = argmax([1-p, p])` — a binary mask. During training, this is a hard argmax (non-differentiable), but gradients flow through the `selected_probs` via the STE in the residual connection.

**Initialization**: `q_proj` and `k_proj` weights start as identity matrices → initial cosine similarity is high → few boundaries → the model starts with conservative chunking and learns to segment.

### 10.2 ChunkLayer — Sequence Compression

**Goal**: Extract only the boundary tokens, reducing `(B, L, D)` to `(B, M, D)` where `M ≪ L`.

**Method**: Pure gather operation (no learnable parameters).

1. Compute `num_tokens = boundary_mask.sum(dim=-1)` per batch element.
2. `M = max(num_tokens)` — the output length.
3. Sort token indices so boundary tokens come first, then gather the first `M`.
4. Return `(next_hidden_states, next_mask)` — the compressed sequence and its validity mask.

**Compression ratio**: Depends on boundary detection. Early in training with identity-initialized projections, `M ≈ 1` (only the forced first boundary). As training progresses, the model learns meaningful segmentation, and `M` grows to reflect the actual semantic structure of the input.

### 10.3 DeChunkLayer — Sequence Reconstruction

**Goal**: Expand `(B, M, D)` back to `(B, L, D)` using the boundary structure and an Exponential Moving Average (EMA).

**Method**:

1. **Index mapping**: `plug_back_idx = cumsum(boundary_mask) - 1` — maps every position to its corresponding chunk index.
2. **Gather**: For each position `t`, look up the chunk representation it belongs to: `expanded[t] = chunks[plug_back_idx[t]]`.
3. **EMA smoothing**: Sequential pass from left to right:
   $$\text{out}_t = p_t \cdot \text{expanded}_t + (1 - p_t) \cdot \text{out}_{t-1}$$
   where $p_t$ is the boundary probability at position $t$.
   - At boundaries ($p_t \approx 1$): output directly uses the new chunk value.
   - Between boundaries ($p_t \approx 0$): output carries forward the previous value with slight blending.

4. **Implementation detail**: The EMA loop uses list accumulation + `torch.stack()` to avoid in-place tensor modification that would break autograd.

The EMA provides a **smooth transition** between chunk representations rather than hard assignment, which helps gradient flow and produces better results.

---

## 11. The Recursive Backbone — HNetBit

**File**: `models/hnet_bit.py` — class `HNetBit`

`HNetBit` is a **recursive** `nn.Module`. Each instance represents one stage of the hierarchy. Non-innermost stages construct a child `HNetBit` at the next stage as their `main_network`.

### 11.1 Non-Innermost Stage

```
Input (B, L, D_parent)
  │
  ▼
pad_dimension: concat learnable zeros → (B, L, D_self)   [if D_self > D_parent]
  │
  ▼
encoder: HGRNBitStack → processes at D_self
  │
  ├──► residual_proj (nn.Linear, FP32, zero-init) → FP32 residual
  │
  ├──► RoutingModuleBit → boundary_mask, boundary_prob, selected_probs
  │
  ├──► ChunkLayer(hidden_states, boundary_mask) → (B, M, D_self)
  │         │
  │         ▼
  │    main_network: HNetBit(stage_idx + 1)  ← recursive call
  │         │
  │         ▼
  │    (B, M, D_self)
  │
  ▼
DeChunkLayer(chunks, boundary_mask, boundary_prob) → (B, L, D_self)
  │
  ▼
residual_func: out · STE(selected_probs) + residual
  │
  ▼
decoder: HGRNBitStack → processes at D_self
  │
  ▼
truncate: [..., :D_parent]  → (B, L, D_parent)
  │
  ▼
Output
```

### 11.2 Innermost Stage

```
Input (B, M', D_parent)
  │
  ▼
pad_dimension → (B, M', D_self)
  │
  ▼
main_network: HGRNBitStack → N blocks of HGRNBitBlock + RMSNorm
  │
  ▼
truncate → (B, M', D_parent)
  │
  ▼
Output
```

### 11.3 Dimension Padding

When transitioning from a parent stage (smaller `D`) to a child stage (larger `D`), the extra dimensions are filled with a **learnable parameter** `pad_dimension` of shape `(D_self - D_parent,)` that is broadcast across all positions. This is initialized to zeros, so at the start of training, the extra dimensions contribute nothing. The model gradually learns what information to place there.

On the return path, the output is simply **truncated** to the parent dimension: `hidden_states[..., :D_parent]`.

### 11.4 Residual Connection with STE Gating

The residual skip connection across the chunking roundtrip uses:

```python
out = out * STE(selected_probs) + residual
```

where:
- `STE(p)` returns `ones_like(p)` in the forward pass (multiplicative identity, so the output is unchanged) but passes the gradient of `p` in the backward pass.
- `residual` comes from `residual_proj`, which is a **full-precision** `nn.Linear` initialized to **zeros**. This means at initialization, the residual contributes nothing, and the model is free to gradually learn to use it.

The purpose is to provide gradient signal to the routing module (which makes hard argmax decisions). Without STE, the boundary_mask has zero gradient. With STE on `selected_probs`, the gradients from the loss flow back through the probability the routing module assigned to its decision, encouraging better boundary placement.

---

## 12. End-to-End Forward Pass (Training)

For a 2-stage model with `d_model = [512, 768, 1024]`, `num_blocks = [[4,0,4], [4,0,4], [8]]`:

```
Step 1:  input_ids (B, L)           ← raw bytes
Step 2:  embeddings(input_ids)      → (B, L, 512)

─── Enter Stage 0 (d=512) ───
Step 3:  encoder[0..3]              → 4 × HGRNBitBlock(512)
Step 4:  residual_proj              → FP32 residual (B, L, 512)
Step 5:  RoutingModuleBit           → boundary_mask_0, prob_0
Step 6:  ChunkLayer                 → (B, M, 512)

  ─── Enter Stage 1 (d=768) ───
  Step 7:  pad 512→768              → (B, M, 768)
  Step 8:  encoder[0..3]            → 4 × HGRNBitBlock(768)
  Step 9:  residual_proj            → FP32 residual (B, M, 768)
  Step 10: RoutingModuleBit         → boundary_mask_1, prob_1
  Step 11: ChunkLayer               → (B, M', 768)

    ─── Enter Stage 2 (d=1024) [innermost] ───
    Step 12: pad 768→1024           → (B, M', 1024)
    Step 13: main_network[0..7]     → 8 × HGRNBitBlock(1024)
    Step 14: RMSNorm                → normalized (B, M', 1024)
    Step 15: truncate               → (B, M', 768)
    ─── Exit Stage 2 ───

  Step 16: DeChunkLayer (EMA, prob_1) → (B, M, 768)
  Step 17: out·STE(p_1) + residual   → (B, M, 768)
  Step 18: decoder[0..3]              → 4 × HGRNBitBlock(768)
  Step 19: truncate                   → (B, M, 512)
  ─── Exit Stage 1 ───

Step 20: DeChunkLayer (EMA, prob_0) → (B, L, 512)
Step 21: out·STE(p_0) + residual    → (B, L, 512)
Step 22: decoder[0..3]              → 4 × HGRNBitBlock(512)
─── Exit Stage 0 ───

Step 23: lm_head (BitLinear)        → (B, L, 256)
Step 24: CrossEntropyLoss(shifted)
```

### Sequence Length Flow

If `L = 1024` and the routing modules select ~25% boundaries each time:
- Stage 0: `L = 1024` → `M ≈ 256`
- Stage 1: `M = 256` → `M' ≈ 64`
- Stage 2 (innermost): processes only 64 tokens at the highest dimension (1024)

This aggressive reduction means the expensive inner stages process very short sequences, keeping overall compute manageable despite the increasing dimension.

---

## 13. Autoregressive Generation (Inference)

During generation, the model processes **one token at a time** using cached recurrent states.

### 13.1 Cache Structure — HNetBitCache

```python
@dataclass
class HNetBitCache:
    # For non-innermost stages:
    encoder_cache:       HGRNBlockCache    # list of (h_state,) per encoder layer
    routing_state:       RoutingModuleState # last_hidden_state + has_seen_tokens
    main_network_cache:  HNetBitCache      # recursive child cache
    dechunk_state:       DeChunkState       # last EMA value per batch element
    decoder_cache:       HGRNBlockCache    # list of (h_state,) per decoder layer
    
    # For innermost:
    main_network_cache:  HGRNBlockCache    # list of (h_state,) per layer
```

The cache is **recursive** — mirroring the model's recursive structure.

### 13.2 Step-by-Step Generation

For each new token:

1. **Embed**: `hidden = embedding(new_token)` → `(B, 1, D)`
2. **backbone.step(hidden)**:
   - **Encoder step**: run each block on `(B, 1, D)`, updating `h_state` in cache
   - **Routing step**: cosine similarity between `last_hidden_state` (cached) and current token → boundary decision
   - **Chunk step**: if `boundary_mask` is True for this batch element, pass the token to the inner stage; otherwise, skip
   - **Inner step**: if a token was selected, `main_network.step(token)` recursively
   - **DeChunk step**: EMA update: `out = p·chunk_value + (1-p)·last_ema_value`
   - **Residual**: add FP32 residual with STE
   - **Decoder step**: run each block on `(B, 1, D)`, updating cache
3. **LM head**: `logits = lm_head(hidden)` → `(B, 1, 256)`
4. **Sample** next token from logits.

**Key insight**: When the routing module decides the current token is *not* a boundary, the inner stages are **completely skipped** — only the EMA carry-forward runs. This means that for most tokens (especially between semantic boundaries), generation is very fast, using only the outer encoder + decoder.

---

## 14. Configuration Presets

### `tiny` (~259K parameters)
```json
{
    "d_model": [48, 64],
    "num_blocks": [[1, 0, 1], [2]],
    "num_heads": 1, "expand_ratio": 1, "hidden_ratio": 2
}
```
- 2 dimension levels (1 stage of chunking)
- 1 encoder block, 2 innermost blocks, 1 decoder block
- Used for smoke test and verification

### `small_1stage` (~21M parameters)
```json
{
    "d_model": [256, 384],
    "num_blocks": [[4, 0, 4], [8]],
    "num_heads": 1, "hidden_ratio": 4
}
```
- 2 dimension levels (1 stage of chunking)
- 4 encoder blocks (d=256), 8 innermost blocks (d=384), 4 decoder blocks (d=256)
- Suitable for debugging and small experiments

### `base_2stage` (~50M parameters)
```json
{
    "d_model": [512, 768, 1024],
    "num_blocks": [[4, 0, 4], [4, 0, 4], [8]],
    "num_heads": 1, "hidden_ratio": 4
}
```
- 3 dimension levels (2 stages of chunking)
- Outer: 4+4 blocks at d=512, Middle: 4+4 blocks at d=768, Inner: 8 blocks at d=1024
- Recommended starting point for real training

### `large_2stage` (~100M parameters)
```json
{
    "d_model": [768, 1024, 1536],
    "num_blocks": [[4, 0, 4], [4, 0, 4], [12]],
    "num_heads": 1, "hidden_ratio": 4
}
```
- Same structure as base, but larger dimensions and more inner blocks

### Benchmark presets (defined in `model_factory.py`)
The Spanish benchmark uses three size tiers for each architecture:

| Size | transformer | matmulfree | hybrid (d_model) | hybrid (blocks) | Notes |
|---|---|---|---|---|---|
| tiny | 2 layers, 64d | 2 layers, 64d | [48, 64] | [[1,0,1], [2]] | Smoke test |
| 150M | 12 layers, 768d | 16 layers, 768d | [576, 768] | [[4,0,4], [10]] | Default |
| 350M | 24 layers, 1024d | 24 layers, 1024d | [640, 896, 1152] | [[4,0,4], [4,0,4], [12]] | 2-stage |
| 750M | 24 layers, 1536d | 28 layers, 1536d | [896, 1152, 1536] | [[6,0,6], [6,0,6], [16]] | 2-stage |

An additional `hybrid_attn` model type is available that uses the same configs as `hybrid` but enables `innermost_use_attention=True` with `attention_layers_pattern="xaxa"` (alternating HGRN and sliding-window attention blocks, window size 64, RoPE). This is an ablation to test whether attention at the deepest hierarchy level improves over pure-HGRN processing. Run with `python train_spanish.py --model hybrid_attn --size 150M`.

---

## 15. Parameter Efficiency and the MatMul-Free Property

### 15.1 What is "MatMul-Free"?

In a standard transformer, every linear projection `y = Wx` requires a full floating-point matrix multiplication. With ternary weights `W ∈ {-α, 0, +α}`:

$$y_j = \sum_i W_{ji} \cdot x_i = \alpha \sum_{i: W_{ji}=+1} x_i - \alpha \sum_{i: W_{ji}=-1} x_i$$

This reduces to **accumulation** (additions and subtractions only). On hardware that supports this natively (e.g., custom accelerators), this can be significantly faster and more energy-efficient than float multiply-accumulate.

During training, `F.linear()` still uses float matmul for gradient computation (via STE). The efficiency gain is primarily at **inference** time, and the parameter storage is minimal at 1.58 bits per weight.

### 15.2 Where FP32 is Still Used

Not everything is ternary. The following components use full precision:
- `nn.Embedding` — the byte embedding table
- `residual_proj` — the FP32 residual projection (zero-initialized)
- `RMSNorm` weights — the learnable scale parameters
- `pad_dimension` — the learnable padding vectors
- `FusedRMSNormSwishGate` weight — normalization scale
- Cosine similarity in `RoutingModuleBit` — computed in FP32 for numerical stability
- EMA in `DeChunkLayer` — computed in FP32

All of these are either small (1D parameters) or critical for numerical stability.

### 15.3 What Replaces Self-Attention

| Standard Transformer | HNetBit Equivalent |
|---|---|
| QKV projection → attention scores → softmax → weighted sum | i_proj, f_proj → sigmoid → SwiGLU gating → HGRN recurrence |
| O(L²) per layer | O(L) per layer |
| KV cache grows with sequence | Fixed-size recurrent state h ∈ ℝ^d |
| Dense float matmul | Ternary BitLinear (addition/subtraction) |

---

## 16. File Map

### HNetBit model implementation (`hnet_bit/`)

| File | Purpose |
|------|---------|
| `models/hnet_bit.py` | `HNetBitConfig`, `HGRNBitStack`, `HNetBit` (recursive backbone), `HNetBitForCausalLM` (full model) |
| `layers/hgrn_bit.py` | `HGRNBitAttention`, `HGRNBitMLP`, `HGRNBitBlock` |
| `ops/bitnet.py` | `BitLinear`, `RMSNorm`, `weight_quant`, `activation_quant` |
| `ops/fusedbitnet.py` | `FusedBitLinear` — Triton-optimized BitLinear with CPU fallback |
| `ops/dynamic_chunking.py` | `RoutingModuleBit`, `ChunkLayer`, `DeChunkLayer`, dataclasses |
| `ops/activations.py` | `SwiGLU` (CUDA jiterator + CPU fallback) |
| `ops/fused_norm_gate.py` | `FusedRMSNormSwishGate` with Triton + CPU fallback |
| `ops/short_conv.py` | Optional short convolution for local inductive bias |
| `ops/hgrn/recurrent_fuse.py` | `fused_recurrent_hgrn` — Triton kernel + naive CPU fallback |
| `ops/hgrn/chunk.py` | `chunk_hgrn` — chunked parallel scan + naive CPU fallback |
| `utils/hnet_cache.py` | `HGRNBlockCache`, `HNetBitCache` — recursive cache for generation |
| `utils/tokenizers.py` | Byte-level tokenizer utilities |
| `utils/helpers.py` | `contiguous` decorator, `apply_optimization_params` |
| `configs/hnet_bit_1stage.json` | Small 1-stage config |
| `configs/hnet_bit_2stage.json` | Base 2-stage config |
| `training/` | Standalone training infrastructure (trainer, optimizer, data, logger) |
| `scripts/` | Dataset prep, evaluation, experiment pipeline |
| `tests/` | Test suites (hnet_bit, dynamic_chunking, fused_bitlinear, etc.) |
| `docs/TRAINING_GUIDE.md` | Standalone training documentation |

### Benchmark pipeline (root level)

| File | Purpose |
|------|---------|
| `train_spanish.py` | Unified training script for three architectures on Spanish Billion Words |
| `model_factory.py` | Builds transformer, matmulfree, and hybrid models at configurable sizes |
| `training_config_spanish.py` | SpanishTrainingConfig dataclass and WSD scheduler |
| `data_spanish.py` | Dataset loading (byte and BPE), memory-mapped datasets |
| `metrics_spanish.py` | BPB computation and inference memory measurement |
| `generate_results.py` | Aggregates per-run results into a single CSV table |
| `test_smoke.sh` | Smoke test script (hybrid on CPU, all three with `--gpu` flag) |
| `matmulfreellm/` | Reference MatMulFree repository (HGRN kernels, BitLinear) |
| `hnet-main/` | Reference HNet repository (dynamic chunking, Isotropic blocks) |
