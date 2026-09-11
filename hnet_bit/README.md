# HNetBit

**Hierarchical H-Net with ternary (MatMul-free) weights** — a byte-level, decoder-only autoregressive language model that combines H-Net's multi-stage dynamic chunking with MatMulFree LM's ternary weight quantization.

HNetBit is built from three ideas:

- **Dynamic chunking hierarchy** (H-Net): the model learns *where* to split a byte sequence into variable-length chunks, processes only the boundary tokens at a deeper, wider stage, then reconstructs the full sequence with an exponential moving average.
- **Gated linear recurrence** (HGRN): sequence mixing is a first-order recurrence `h_t = f_t ⊙ h_{t-1} + i_t` with O(L) training and O(1) state per generated token — no self-attention, no growing KV cache.
- **Ternary weights** (BitLinear): every linear projection is quantized to `{-1, 0, +1}` during the forward pass with a straight-through estimator, so weights cost ~1.58 bits and a matmul becomes a sum of additions/subtractions.

The result is a hierarchical, recurrence-based LM that operates directly on bytes (vocab 256), learns its own segmentation instead of relying on a fixed tokenizer, and keeps a constant-size inference state regardless of context length.

---

## 1. Model classification

| Property | Classification |
|---|---|
| **Model type** | Decoder-only causal language model |
| **Sequence mixing** | Recurrent (HGRN gated linear recurrence), **not attention** (except the optional innermost variant) |
| **Weight precision** | Ternary `{-1, 0, +1}` — MatMul-free, ~1.58 bits/weight |
| **Hierarchy** | Multi-stage recursive with learned dynamic chunking |
| **Tokenization** | Byte-level (vocab = 256) |
| **Objective** | Next-token prediction (shifted cross-entropy) |
| **Inference state** | Fixed-size recurrent state (O(1) per token), plus a bounded KV cache in the attention variant |

### Key distinctions

- **Not a Transformer**: no self-attention by default. Sequence mixing uses HGRN, giving O(L) training compute and a fixed `(B, H, d_head)` state per layer at inference. An optional variant interleaves sliding-window attention into the innermost stage only.
- **Not an encoder-decoder**: the `encoder` and `decoder` stacks at each hierarchy stage are both causal left-to-right blocks — the terminology is inherited from H-Net.
- **Not a classic RNN/LSTM**: HGRN is a modern gated linear recurrence with straightforward parallelization over the head dimension and a Triton backward kernel, not a tanh/sigmoid RNN.
- **MatMul-free**: all dense float matmuls are replaced by ternary weight operations. During training `F.linear` is still a float matmul; the efficiency gain (1.58-bit storage, addition-only inference) materializes in deployment.

### Comparable models

RetNet, RWKV, Mamba (linear recurrent alternatives to transformers). The distinguishing combination here is **hierarchical dynamic chunking + ternary quantization**.

---

## 2. High-level overview

HNetBit draws from two research lines:

| Concept | Source | Contribution |
|---|---|---|
| **H-Net** | Hwang et al. | Multi-stage hierarchy via dynamic chunking: learn to segment a sequence into variable-length chunks, process them at progressively higher dimensions, then reconstruct the original length |
| **MatMul-Free LM** | Zhu et al. | Ternary weight quantization (`{-1, 0, +1}`) via `BitLinear` and HGRN-based recurrence instead of attention |

Schematic for a 2-stage model (`d_model = [512, 768, 1024]`, `num_blocks = [[4,0,4], [4,0,4], [8]]`):

```
Input bytes (B, L)
  │
  ▼
Embedding(256, 512)
  │
  ▼
┌─── Stage 0 (d = 512) ────────────────────────────────────┐
│ Encoder: 4 × HGRNBitBlock(512)                           │
│   │                                                      │
│   ├──► residual_proj (FP32, zero-init) ───────────────┐  │
│   │                                                   │  │
│   ├──► RoutingModuleBit → boundary mask               │  │
│   │                                                   │  │
│   ├──► ChunkLayer → (B, M, 512)                       │  │
│   │         │                                         │  │
│   │    ┌─── Stage 1 (d = 768) ────────────────────┐   │  │
│   │    │ Pad 512→768                             │   │  │
│   │    │ Encoder: 4 × HGRNBitBlock(768)          │   │  │
│   │    │ RoutingModuleBit → ChunkLayer           │   │  │
│   │    │      │                                  │   │  │
│   │    │ ┌─── Stage 2 (d = 1024) [innermost] ─┐  │   │  │
│   │    │ │ 8 × HGRNBitBlock(1024) + RMSNorm    │  │   │  │
│   │    │ │ Unpad → 768                        │  │   │  │
│   │    │ └────────────────────────────────────┘  │   │  │
│   │    │ DeChunkLayer (EMA) → (B, M, 768)        │   │  │
│   │    │ out · STE(p) + residual                 │   │  │
│   │    │ Decoder: 4 × HGRNBitBlock(768)          │   │  │
│   │    │ Unpad → 512                             │   │  │
│   │    └─────────────────────────────────────────┘   │  │
│   │                                                   │  │
│   ◄──── DeChunkLayer (EMA) → (B, L, 512) ◄────────────┘  │
│         out · STE(p) + residual                          │
│                                                          │
│ Decoder: 4 × HGRNBitBlock(512)                           │
└──────────────────────────────────────────────────────────┘
  │
  ▼
BitLinear LM Head (512 → 256)
  │
  ▼
Byte logits (B, L, 256)
```

If each router keeps ~25% of positions as boundaries, a 1024-byte input is processed as ~256 tokens at stage 1 and ~64 tokens at the widest innermost stage — the expensive high-dimensional layers see very short sequences.

---

## 3. Input representation — byte-level

Each token is one byte in `[0, 255]`:

- **Vocabulary**: 256, fixed.
- **Tokenization**: UTF-8 bytes, one byte per token. Multi-byte characters span multiple tokens.
- **Special tokens**: `bos_token_id = 254`, `eos_token_id = 255` (configurable).
- **Embedding**: `nn.Embedding(256, d_model[0])`, kept in full precision — the only non-ternary projection touching the input.

Example: `"Hello 😀"` → `[72, 101, 108, 108, 111, 32, 240, 159, 152, 128]` (six ASCII bytes + a four-byte UTF-8 emoji).

Byte-level operation contrasts with fixed subword tokenization:

| Aspect | BPE/WordPiece | HNetBit bytes + chunking |
|---|---|---|
| Vocabulary | 30k–50k learned tokens | 256 bytes (fixed) |
| Segmentation | Fixed after tokenizer training | Learned, content-dependent boundaries |
| Language coverage | Language-specific | Universal (any UTF-8 input) |
| OOV handling | UNK tokens | None — every byte is representable |
| Hierarchy | Flat | Multi-stage learned compression |

The tradeoff is longer base sequences (roughly 3–4× more symbols), which the hierarchy is designed to absorb: deeper stages operate on aggressively shortened sequences.

---

## 4. The core primitive — BitLinear

**File**: `ops/bitnet.py`, `ops/fusedbitnet.py`

Every linear projection in the model — HGRN projections, MLP projections, and the LM head — is a `BitLinear` layer. It is an `nn.Linear` subclass that quantizes both weights and activations in the forward pass while keeping a full-precision copy for gradient updates.

### 4.1 Weight quantization (ternary, ~1.58 bits)

```python
def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u
```

1. `α = mean(|W|)` (per tensor).
2. Scale `W/α`, round to the nearest of `{-1, 0, +1}`, clamp.
3. Rescale by `α`.

Effective weights at inference are always `{-α, 0, +α}`, so `y_j = Σ_i W_ji x_i = α·Σ_{W=+1} x_i − α·Σ_{W=−1} x_i` — accumulation only.

Frozen ternary tensors can be stored at 2 bits/weight with `pack_ternary_tensor` / `unpack_ternary_tensor` in `ops/bitnet.py`.

### 4.2 Activation quantization (8-bit)

```python
def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y
```

Per-token scaling to `[-128, 127]`, immediately de-scaled back to float. This bounds the activation dynamic range entering the ternary matmul.

### 4.3 Straight-through estimator

Quantization is applied in the forward pass only; gradients bypass it:

```python
x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
w_quant = w + (weight_quant(w) - w).detach()
y = F.linear(x_quant, w_quant)
```

The `.detach()` on the quantization residual means `∂L/∂w` is computed as if quantization were an identity.

### 4.4 Built-in normalization

Each `BitLinear` owns an `RMSNorm` applied to its input *before* quantization:

```python
x_norm = self.norm(x)
x_quant = activation_quant(x_norm)
w_quant = weight_quant(self.weight)
y = F.linear(x_quant, w_quant)
```

### 4.5 FusedBitLinear

`FusedBitLinear` (`ops/fusedbitnet.py`, imported as `BitLinear` throughout `models/` and `layers/`) fuses RMSNorm → quantization → linear into Triton kernels for a 2–3× speedup. When Triton is unavailable or the tensor is on CPU it transparently falls back to the pure-PyTorch `BitLinear` path, so the same model code runs everywhere.

Triton availability is detected in `ops/_triton.py` and can be forced off with:

```bash
export HNETBIT_DISABLE_TRITON=1
```

which routes HGRN recurrence and fused normalization to naive PyTorch implementations (slower, but works in unprivileged containers without a usable GPU runtime).

### 4.6 What stays in full precision

| Component | Reason |
|---|---|
| `nn.Embedding` | Input/output representation |
| `residual_proj` (FP32, zero-init) | Precision of the skip path across chunking; starts at zero so the model learns its use |
| `RoutingModuleBit.q_proj` / `k_proj` | Cosine-similarity boundary decisions are sensitive to quantization noise; identity-initialized |
| `CausalMHABit` Q/K/V/O (attention variant) | Stable attention scores |
| `RMSNorm` / `FusedRMSNormSwishGate` weights | 1-D scale parameters |
| `pad_dimension` | Learnable dimension padding vector |
| Cosine similarity and DeChunk EMA math | Computed in FP32 internally for numerical stability |

---

## 5. Sequence mixing — HGRN recurrence

**Files**: `ops/hgrn/recurrent_fuse.py`, `ops/hgrn/chunk.py`

HGRN (Hierarchically Gated Recurrent Network) replaces self-attention with a linear recurrence with a forget gate:

```
h_t = f_t ⊙ h_{t-1} + i_t        h_t, f_t, i_t ∈ ℝ^d
```

- `f_t = σ(W_f x_t)` — forget gate in `(0, 1)`, element-wise.
- `i_t = SwiGLU(W_i x_t, 1 − f_t)` — input, gated by the *complement* of the forget gate: when the model remembers (`f` high), new input is suppressed; when it forgets (`f` low), input flows through.
- `h_t` is the recurrent state, carried across time steps.

### 5.1 Complexity

| | Self-attention | HGRN |
|---|---|---|
| Training compute | O(L²) | O(L) |
| Inference state per layer | KV cache growing with L | Fixed `(B, H, d_head)` |
| Per-step decode cost | O(L) | O(1) |

Training is parallelized by mapping the recurrence to a Triton scan kernel; inference uses a single-token step that reads and writes the state.

### 5.2 Kernel backends

| Kernel | Backend | Notes |
|---|---|---|
| `fused_recurrent_hgrn` | Triton (CUDA) | Forward + custom backward kernel; parallelizes across the head dimension while scanning over time. Used by `HGRNBitAttention` and `DeChunkLayer`. |
| `chunk_hgrn` | Triton (CUDA) | Chunked parallel scan (intra-chunk parallel, inter-chunk combine). Available as an alternative kernel. |
| `_naive_recurrent_hgrn` | PyTorch | Sequential loop fallback for CPU or when Triton is disabled. |
| `_naive_chunk_hgrn` | PyTorch | Sequential fallback for the chunked variant. |

The dispatcher chooses automatically: `fused_recurrent_hgrn` falls back to the naive loop whenever `x.is_cuda` is false or `triton_available()` fails.

### 5.3 Multi-head and expansion

`HGRNBitAttention` supports `num_heads > 1` (recurrence per head, `d_head = hidden_size · expand_ratio / num_heads`) and `expand_ratio > 1` (state width larger than the model dimension).

### 5.4 Optional short convolution

With `use_short_conv=True`, a depthwise causal `ShortConvolution` (kernel size `conv_size`, default 4) is inserted before the projections to inject a local inductive bias. `share_conv_kernel=True` applies one shared convolution to the input; `False` convolves the `i` and `f` streams separately. Conv state is part of the recurrent cache.

---

## 6. Activations

### 6.1 SwiGLU

**File**: `ops/activations.py`

```
SwiGLU(x, y) = SiLU(x) · y
```

Used in two places:

1. `HGRNBitAttention`: `i = SwiGLU(i_proj(x), 1 − f)`.
2. `HGRNBitMLP`: gated feed-forward hidden layer.

CUDA uses a fused jiterator implementation; CPU falls back to `F.silu(x) * y`.

### 6.2 FusedRMSNormSwishGate

**File**: `ops/fused_norm_gate.py`

Output-gate normalization in `HGRNBitAttention`:

```
FusedRMSNormSwishGate(x, gate) = RMSNorm(x) ⊙ gate ⊙ σ(gate)
```

where `x` is a separate `g_proj` projection of the original block input and `gate` is the recurrence output. Fuses normalization, swish activation, and gating; Triton-accelerated on CUDA with a pure-PyTorch reference fallback.

---

## 7. The block and the stack

### 7.1 HGRNBitBlock

**File**: `layers/hgrn_bit.py`

Pre-norm residual layout with two sub-layers:

```
x
├──────────────────────────────┐
▼                              │
RMSNorm (attn_norm)            │
▼                              │
HGRNBitAttention               │
▼                              │
+ ◄────────────────────────────┘
│
├──────────────────────────────┐
▼                              │
RMSNorm (mlp_norm)             │
▼                              │
HGRNBitMLP                     │
▼                              │
+ ◄────────────────────────────┘
▼
out
```

The add + norm between sub-layers is fused: `mlp_norm(attn_out, residual, prenorm=True)` returns both the normalized sum and the full-precision residual used for the second skip.

### 7.2 HGRNBitAttention

Given `x ∈ ℝ^{B×L×D}`:

1. `i = i_proj(x)` (ternary), `f = σ(f_proj(x))` (ternary). With short conv enabled, the convolution is applied either to `x` before projection (shared kernel) or to `i` and `f` separately.
2. `i = SwiGLU(i, 1 − f)`.
3. Reshape to `(B, H, L, d_head)`.
4. `o, state = fused_recurrent_hgrn(i, f, h_prev)` — applies `h_t = f_t ⊙ h_{t-1} + i_t`.
5. `o = FusedRMSNormSwishGate(g_proj(x), o)`.
6. `o = o_proj(o)` (ternary).

All four projections are `BitLinear`; the recurrence state is `(B, H, d_head)` (plus conv state if enabled). `init_state` / `state_size` handle cache allocation.

### 7.3 HGRNBitMLP

Given `x ∈ ℝ^{B×L×D}`:

1. `y = gate_proj(x)` → `(B, L, 2I)` (ternary).
2. `gate, y = y.chunk(2, dim=-1)`.
3. `z = SwiGLU(gate, y)`.
4. `out = down_proj(z)` (ternary).

The intermediate size defaults to `I = 256 · ⌈(⅔ · D · hidden_ratio) / 256⌉` (with `hidden_ratio = 4`, `I ≈ 2.67·D`, rounded up to a multiple of 256), or is set explicitly via `intermediate_size`.

### 7.4 HGRNBitStack

**File**: `models/hnet_bit.py`

`N` sequential blocks at one stage dimension followed by a final `RMSNorm`. Used as the encoder, decoder, or innermost network. Each stack owns an `HGRNBlockCache` — a flat list of per-layer recurrent states indexed by a global `layer_idx` (the decoder's indices continue after the encoder's).

When `innermost_use_attention=True`, the innermost stack interleaves HGRN and attention blocks according to `attention_layers_pattern`, a string of `'x'` (HGRN) and `'a'` (attention). The pattern is extended cyclically if shorter than the layer count; the default alternates starting with HGRN (`"xaxa..."`).

---

## 8. Sliding-window attention variant

**File**: `layers/attention.py`

An optional ablation replaces every other innermost block with causal multi-head attention while keeping the rest of the hierarchy unchanged.

`CausalMHABit`:

- Full-precision `nn.Linear` Q/K/V/O projections (ternary attention projections destabilize attention scores).
- **Sliding-window** causal masking: position `i` attends to `[i − window + 1, i]`, window default 64. `window_size = 0` gives full causal attention.
- **RoPE** rotary position embeddings, cached up to `max_position_embeddings`.
- KV cache appended on the fly; during single-token decoding only the last `window_size` keys/values are kept.

`CausalMHABlock` mirrors `HGRNBitBlock`'s pre-norm interface so the two block types can be mixed freely inside a stack. Its MLP is the same ternary `HGRNBitMLP`.

Enable with:

```json
{
    "innermost_use_attention": true,
    "attention_window_size": 64,
    "attention_layers_pattern": "xaxa",
    "attention_num_heads": 4
}
```

---

## 9. Dynamic chunking

**File**: `ops/dynamic_chunking.py`

Dynamic chunking is what creates the hierarchy: it shortens the sequence between stages and reconstructs it afterward. All three components operate on causal, left-to-right sequences.

### 9.1 RoutingModuleBit — boundary detection

Given encoder output `h ∈ ℝ^{B×L×D}`:

1. Project consecutive positions: `q = normalize(W_q h_{t})`, `k = normalize(W_k h_{t+1})` where `W_q, W_k` are **full-precision, identity-initialized** `nn.Linear` projections.
2. Cosine similarity `cos_sim(t) = q_t · k_{t+1}` (computed in FP32).
3. Boundary probability `p_t = clamp((1 − cos_sim(t)) / 2, 0, 1)` — 0 means "identical, same chunk", 1 means "dissimilar, new chunk".
4. Force the first position to be a boundary: `p_0 = 1`.
5. Hard decision: `boundary_mask = p > 0.5` (implemented as `argmax([1−p, p])`). Gradients do not flow through the argmax; they reach the router through the STE-gated residual instead (Section 9.4).

Identity initialization means initial cosine similarities are high, so the model starts with very few boundaries and learns to segment during training.

Every stage's `RoutingModuleOutput` bubbles up through the backbone and is exposed as `router_outputs` on the model output when `output_hidden_states=True`. The training loop can use it to compute H-Net's auxiliary load-balancing loss over boundary probabilities, ensuring the router does not collapse to "all boundaries" or "no boundaries". It is disabled by default in the standalone trainer (`lambda_lb = 0.0`).

### 9.2 ChunkLayer — compression

A parameterless gather that keeps only boundary tokens:

- Padded mode: sort boundary tokens to the front with an index trick, gather the first `M = max_t(Σ boundary_mask)` positions, and return the new validity mask.
- Packed mode: boolean-index the concatenated tokens and recompute `cu_seqlens`.

Output: `(B, M, D)` with `M ≪ L`.

### 9.3 DeChunkLayer — reconstruction with EMA

Expands `(B, M, D)` back to `(B, L, D)`:

1. `plug_back_idx = cumsum(boundary_mask) − 1` maps every position to its chunk.
2. Each position looks up its chunk representation.
3. An exponential moving average smooths transitions:
   ```
   out_t = p_t · chunk_t + (1 − p_t) · out_{t−1}
   ```
   with `p_t` clamped to `[1e-4, 1−1e-4]`. At a boundary the new chunk value dominates; between boundaries the previous value is carried forward.

The EMA is not a Python loop: it is algebraically mapped onto the HGRN recurrence `h_t = g_t ⊙ h_{t-1} + x_t` by setting `x = p·chunk` and `g = 1 − p`, so it runs on the same fused scan kernel. The math is done in FP32 and cast back to the model dtype.

### 9.4 Residual with STE gating

The skip connection across the chunking round-trip is:

```python
out = dechunk_out * STE(selected_probs) + residual
```

- `STE(p)` returns `ones_like(p)` in the forward pass (so the value is unchanged) and passes the gradient of `p` in the backward pass. This is the only gradient path to the router's hard decisions.
- `residual` comes from `residual_proj`, a full-precision `nn.Linear` initialized to **zero**, so the model starts without a skip contribution and learns to use it.

### 9.5 Dimension padding

Because deeper stages have larger `d_model`, the child stage receives its input appended with a learnable `pad_dimension` vector (initialized to zero) broadcast over positions. On the way back, the output is truncated to the parent dimension: `hidden_states[..., :D_parent]`.

---

## 10. The recursive backbone — HNetBit

**File**: `models/hnet_bit.py`

`HNetBit` is a recursive `nn.Module`: each instance is one stage, and every non-innermost stage owns a child `HNetBit` at the next stage as its `main_network`.

### 10.1 Non-innermost stage

```
Input (B, L, D_parent)
  │
  ▼
pad_dimension → (B, L, D_self)                    [if D_self > D_parent]
  │
  ▼
encoder: HGRNBitStack
  │
  ├──► residual_proj (FP32, zero-init) → residual
  │
  ├──► RoutingModuleBit → boundary_mask, boundary_prob, selected_probs
  │
  ├──► ChunkLayer → (B, M, D_self)
  │         │
  │    main_network: HNetBit(stage + 1) → (B, M, D_self)
  │         │
  ▼         ▼
DeChunkLayer (EMA) → (B, L, D_self)
  │
  ▼
out · STE(selected_probs) + residual
  │
  ▼
decoder: HGRNBitStack
  │
  ▼
truncate → (B, L, D_parent)
```

### 10.2 Innermost stage

```
Input (B, M, D_parent)
  ▼
pad_dimension → (B, M, D_self)
  ▼
main_network: HGRNBitStack (N blocks + final RMSNorm)
  ▼
truncate → (B, M, D_parent)
```

### 10.3 End-to-end forward pass

For `d_model = [512, 768, 1024]`, `num_blocks = [[4,0,4], [4,0,4], [8]]`:

```
 1. input_ids (B, L)                    raw bytes
 2. embeddings                          → (B, L, 512)
    ── Stage 0 (d = 512) ──
 3. encoder: 4 × HGRNBitBlock(512)
 4. residual_proj                       FP32 residual (B, L, 512)
 5. RoutingModuleBit                    boundary_mask₀, probs₀
 6. ChunkLayer                          → (B, M, 512)
      ── Stage 1 (d = 768) ──
 7. pad 512 → 768                       → (B, M, 768)
 8. encoder: 4 × HGRNBitBlock(768)
 9. residual_proj                       FP32 residual (B, M, 768)
10. RoutingModuleBit                    boundary_mask₁, probs₁
11. ChunkLayer                          → (B, M′, 768)
        ── Stage 2 (d = 1024, innermost) ──
12. pad 768 → 1024                      → (B, M′, 1024)
13. main_network: 8 × HGRNBitBlock(1024) + RMSNorm
14. truncate → (B, M′, 768)
      ── back to Stage 1 ──
15. DeChunkLayer (EMA, probs₁)          → (B, M, 768)
16. out · STE(p₁) + residual            → (B, M, 768)
17. decoder: 4 × HGRNBitBlock(768)
18. truncate → (B, M, 512)
    ── back to Stage 0 ──
19. DeChunkLayer (EMA, probs₀)          → (B, L, 512)
20. out · STE(p₀) + residual            → (B, L, 512)
21. decoder: 4 × HGRNBitBlock(512)
22. lm_head (BitLinear)                 → (B, L, 256)
23. shifted cross-entropy loss
```

The recursion itself is implemented by the child `HNetBit` being called from `main_network`; each level appends its `RoutingModuleOutput` to a list that bubbles up for load-balancing / analysis hooks.

### 10.4 Top-level wrapper — HNetBitForCausalLM

```
HNetBitForCausalLM
├── embeddings: nn.Embedding(256, d_model[0])
├── backbone:   HNetBit(config, stage_idx=0)
└── lm_head:    BitLinear(d_model[0], 256)
```

- Extends `PreTrainedModel` + `GenerationMixin` for `model.generate()` support.
- Overrides `_prepare_cache_for_generation` to allocate the hierarchical `HNetBitCache` instead of `DynamicCache`.
- Exposes `num_hidden_layers` (sum of all blocks across stages) and `hidden_size` (`d_model[0]`) for transformers internals.
- Accepts optional `tie_word_embeddings`.
- Next-token targets are built by shifting labels by one position; the final position is filled with `ignore_index` before cross-entropy.

### 10.5 Initialization

- Embeddings: normal, `std = 1.0`.
- Linear/BitLinear weights: normal, `std = initializer_range` (0.02), except parameters marked `_no_reinit` (identity router projections, zero residual projection).
- Prenorm residual rescaling (GPT-2 style): `o_proj` and `down_proj` weights divided by `√(2 · num_hidden_layers)`.
- `pad_dimension` initialized to zeros.

---

## 11. Autoregressive generation

### 11.1 Recursive cache

Generation state mirrors the architecture and is captured by `HNetBitCache` (`utils/hnet_cache.py`). For a non-innermost stage:

```python
encoder_cache:      HGRNBlockCache       # (h_state,) per encoder layer
routing_state:      RoutingModuleState   # last_hidden_state + has_seen_tokens
main_network_cache: HNetBitCache         # recursive child cache
dechunk_state:      DeChunkState         # last EMA value per batch element
decoder_cache:      HGRNBlockCache       # (h_state,) per decoder layer
is_innermost:       bool
```

The innermost stage has only `main_network_cache: HGRNBlockCache` (a flat list of `(h_state,)` tuples, one per block). All cache fields are optional and `None` on stages that do not use them.

`HGRNBlockCache.states` is a list of per-layer tuples (recurrent state, optional conv state, or KV pairs in the attention variant). In-place updates are used when shapes match.

### 11.2 Step-by-step decoding

For each new token:

1. **Embed** → `(B, 1, D)`.
2. **Encoder step** — each encoder block updates its recurrent state.
3. **Routing step** — cosine similarity between the cached `last_hidden_state` and the current token produces a boundary decision.
4. **Chunk step** — the token is passed to the inner stage only if it is a boundary.
5. **Inner step** — recurse into the child stage for selected tokens, slicing and merging the child cache by the boundary mask.
6. **DeChunk step** — `out = p · chunk_value + (1 − p) · last_ema_value`.
7. **Residual + decoder step** — STE-gated skip, then decoder blocks update their states.
8. **LM head** → `(B, 1, 256)` logits; sample the next byte.

When the router decides a token is *not* a boundary, the inner stages are skipped entirely — only the EMA carry-forward runs. Most tokens between semantic boundaries cost only the outer encoder + decoder passes.

---

## 12. Configuration

`HNetBitConfig` (`models/hnet_bit.py`) extends `PretrainedConfig`:

| Parameter | Default | Meaning |
|---|---|---|
| `vocab_size` | 256 | Byte vocabulary |
| `d_model` | `[512, 768]` | Hidden width per stage; length = number of hierarchy stages |
| `num_blocks` | `[[4, 0, 4], [8]]` | Per stage: `[encoder, unused, decoder]` for non-innermost, `[n]` for the innermost stage |
| `num_heads` | 1 | HGRN heads (recurrence is computed per head) |
| `expand_ratio` | 1 | Recurrent state width multiplier |
| `hidden_ratio` | 4 | MLP width ratio; `I = 256·⌈(⅔·d·ratio)/256⌉` |
| `intermediate_size` | `None` | Explicit MLP width override |
| `hidden_act` | `"swish"` | Accepted for interface compatibility; the MLP always uses SwiGLU |
| `attn_mode` | `"fused_recurrent"` | Recurrence mode |
| `max_position_embeddings` | 2048 | Max sequence length; RoPE cache size in the attention variant |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `use_cache` | `True` | Enable inference cache |
| `pad_token_id` / `bos_token_id` / `eos_token_id` | `None` / 254 / 255 | Special token IDs |
| `tie_word_embeddings` | `False` | Tie embedding and LM head weights |
| `initializer_range` | 0.02 | Weight init std |
| `use_fused_bitlinear` | `False` | Stored for API compatibility; `FusedBitLinear` is always instantiated and falls back internally |
| `use_short_conv` | `False` | Enable short convolution in HGRN attention |
| `conv_size` / `share_conv_kernel` | 4 / `True` | Short conv kernel size and sharing |
| `use_lower_bound` | `False` | Reserved — no learned forget-gate bound |
| `innermost_use_attention` | `False` | Interleave attention into the innermost stack |
| `attention_window_size` | 64 | Sliding window (`0` = full causal) |
| `attention_num_heads` | `None` | Attention heads (defaults to `num_heads`) |
| `attention_layers_pattern` | `None` | `'x'`/`'a'` pattern; default alternates from HGRN |

Derived attributes: `num_stages = len(d_model)`, `num_hidden_layers` (all blocks across stages), `hidden_size = d_model[0]`. The config validates that `d_model` and `num_blocks` align and that non-innermost stages use `[enc, unused, dec]` triples.

### 12.1 Built-in presets (classmethods)

| Preset | `d_model` | `num_blocks` | Structure |
|---|---|---|---|
| `small_1stage` | `[256, 384]` | `[[4,0,4], [8]]` | 2 levels, ~21M params |
| `base_2stage` | `[512, 768, 1024]` | `[[4,0,4], [4,0,4], [8]]` | 3 levels, ~50M params |
| `large_2stage` | `[768, 1024, 1536]` | `[[4,0,4], [4,0,4], [12]]` | 3 levels, ~100M params |

### 12.2 JSON configs (`configs/`)

| File | `d_model` | `num_blocks` | `num_heads` | `expand_ratio` | `use_short_conv` |
|---|---|---|---|---|---|
| `hnet_bit_1stage.json` | `[256, 384]` | `[[4,0,4], [8]]` | 4 | 2 | true |
| `hnet_bit_100M.json` | `[576, 768]` | `[[4,0,4], [10]]` | 4 | 2 | true |
| `hnet_bit_2stage.json` | `[512, 768, 1024]` | `[[4,0,4], [4,0,4], [8]]` | 4 | 2 | true |
| `hnet_bit_350M.json` | `[640, 896, 1152]` | `[[4,0,4], [4,0,4], [12]]` | 4 | 2 | true |

The `training_*.json` files are experiment configs consumed by the standalone training pipeline.

---

## 13. The MatMul-free property

### 13.1 Why ternary equals MatMul-free

With `W ∈ {−α, 0, +α}`:

```
y_j = Σ_i W_ji · x_i = α · Σ_{W_ji = +1} x_i − α · Σ_{W_ji = −1} x_i
```

A multiply-accumulate per weight becomes a signed addition (or nothing for zeros). On hardware that supports it natively, this is faster and more energy-efficient than float matmul. Storage drops to ~1.58 bits/weight — `pack_ternary_tensor` stores 4 values per byte (2 bits each) plus a float scale.

During training, gradients flow through the STE and `F.linear` still uses float matmul; the savings are realized at inference and in model storage.

### 13.2 What replaces self-attention

| Standard Transformer | HNetBit |
|---|---|
| QKV projection → attention scores → softmax → weighted sum | `i_proj`, `f_proj` → sigmoid → SwiGLU gating → HGRN recurrence |
| O(L²) per layer | O(L) per layer |
| KV cache grows with sequence length | Fixed recurrent state `h ∈ ℝ^d` per layer |
| Dense float matmul | Ternary BitLinear (accumulation) |
| Fixed tokenizer vocabulary | Learned dynamic chunk boundaries over raw bytes |

### 13.3 Sequence-length flow

Chunking is the compute lever: a 1024-byte sequence with ~25% boundary rates becomes ~256 tokens at stage 1 and ~64 at stage 2, so the widest (and most expensive) layers process a small fraction of the original positions. The returned `RoutingModuleOutput.selected_probs` and boundary masks make this compression measurable during training and inference.

---

## 14. File map

| File | Purpose |
|---|---|
| `models/hnet_bit.py` | `HNetBitConfig`, `HGRNBitStack`, `HNetBit` (recursive backbone), `HNetBitForCausalLM` |
| `layers/hgrn_bit.py` | `HGRNBitAttention`, `HGRNBitMLP`, `HGRNBitBlock` |
| `layers/attention.py` | `CausalMHABit`, `CausalMHABlock`, RoPE, sliding-window masking |
| `ops/bitnet.py` | `BitLinear`, `RMSNorm`, `weight_quant`, `activation_quant`, ternary pack/unpack |
| `ops/fusedbitnet.py` | `FusedBitLinear` — Triton-fused norm + quantization + linear, pure-PyTorch fallback |
| `ops/dynamic_chunking.py` | `RoutingModuleBit`, `ChunkLayer`, `DeChunkLayer`, routing/dechunk states |
| `ops/activations.py` | `SwiGLU` (CUDA jiterator + CPU fallback) |
| `ops/fused_norm_gate.py` | `FusedRMSNormSwishGate` (Triton + reference fallback) |
| `ops/short_conv.py` | `ShortConvolution` — depthwise causal conv for local inductive bias |
| `ops/hgrn/recurrent_fuse.py` | `fused_recurrent_hgrn` — Triton scan kernel + naive fallback |
| `ops/hgrn/chunk.py` | `chunk_hgrn` — chunked parallel scan + naive fallback |
| `ops/_triton.py` | Triton availability detection / `HNETBIT_DISABLE_TRITON` handling |
| `utils/hnet_cache.py` | `HGRNBlockCache`, `HNetBitCache` — recursive generation cache |
| `utils/tokenizers.py` | Byte-level tokenizer utilities |
| `utils/helpers.py` | `contiguous` decorator, optimizer parameter helpers |
| `configs/` | JSON model and experiment configs |
| `training/` | Standalone training infrastructure (trainer, optimizer, data, logger, evaluator) |
| `train.py`, `generate.py` | Standalone training and generation entry points |
| `tests/` | Test suites for blocks, chunking, fused kernels, model, training |

---

## 15. References

- H-Net (Dynamic Chunking): https://github.com/voidism/HNet
- MatMul-Free LM: https://github.com/ridgerchu/matmulfreellm
- HGRN2 (Gated Linear RNNs): arXiv:2404.07904
- BitNet (Scaling 1-bit Transformers): arXiv:2310.11453
