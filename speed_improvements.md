# Hybrid Model Speed Improvements

## Current baseline (150M, A100)

| Scenario | Tok/s | Memory |
|---|---|---|
| Prefill 4096 | 108 ms | 742 MB |
| Decode batch 1 | 49 tok/s | 573 MB |
| Decode batch 8 | 377 tok/s | 652 MB |

## CPU (desktop, no GPU)

| Scenario | Tok/s | Memory |
|---|---|---|
| Prefill 4096 | 3.2 s | 0 MB (RAM) |
| Decode batch 1 | 4 tok/s | ~500 MB RAM |

---

## 1. Low effort (hours)

### FP16/BF16 inference
Load the model in half precision instead of FP32. Currently the deploy export unpacks ternary weights to float32. Converting to BF16 halves weight memory and improves throughput.

```
model = model.half()  # or model.bfloat16()
```

**Potential:** ~1.5-2× tok/s, halves memory.

### Torch.compile
Apply `torch.compile` to the model's forward pass. This fuses operations, reduces Python overhead, and improves cache locality. Works best on GPU.

```python
model.forward = torch.compile(model.forward, mode="reduce-overhead")
```

**Potential:** ~1.3-2× on GPU, minimal on CPU.

### Pin CPU threads
Set environment variables for CPU inference:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_INDUCTOR_CPP_WRAPPER=1
```

**Potential:** ~1.5-2× on CPU.

---

## 2. Medium effort (days)

### Key-value cache for encoder/decoder blocks
The outer stage's encoder and decoder blocks are HGRNBitBlocks — recurrent, no KV cache. A custom cache for the HGRN state (like the existing `HGRNBlockCache` but optimized to avoid redundant computation) would skip blocks where the hidden state hasn't meaningfully changed.

**Potential:** ~2-3× on long contexts.

### Skip inner stage when no boundaries
Most decode steps have 0 boundary tokens. Currently the routing/chunk/dechunk pipeline still runs even when the inner stage is skipped. Optimizing this path — early-exit after the encoder when no boundaries are detected — saves the routing step overhead.

**Potential:** ~1.2× (small but free).

### Batch the inner stage across multiple timesteps
Instead of processing one token at a time through the hierarchy, buffer several tokens and process them as a batch. The inner stage's sequential bottleneck (processing B' ≪ B boundary tokens) becomes efficient when B' grows.

Requires modifying the step loop to accumulate tokens before flushing through the hierarchy.

**Potential:** ~2-4× on decode (diminishing returns with latency).

### Speculative decoding
Train a small auxiliary model (or use a smaller stage-1-only version) to propose candidate tokens cheaply. Verify in batch through the full hierarchy. Common technique — 2-4× speedup on transformers, applicable here too.

Requires a draft model + verification logic.

**Potential:** ~2-4×.

---

## 3. High effort (weeks)

### Fused hierarchical step kernel
The entire per-token path (encoder → routing → chunk → inner → dechunk → decoder) can be fused into a single Triton kernel. Currently each block launches separate kernels (norm, quant, linear, conv, HGRN). A fused kernel keeps all intermediates in SRAM.

This is the single largest improvement available but requires significant CUDA/Triton engineering.

**Potential:** ~3-5× on GPU.

### Pipeline across hierarchy stages
On multi-GPU: assign each hierarchy stage to a separate GPU. Stage 0 on GPU 0, stage 1 on GPU 1, etc. Each GPU communicates boundaries and hidden states. This eliminates the sequential bottleneck of the current single-GPU pipeline.

Requires DDP or pipeline parallelism for inference — non-trivial.

**Potential:** ~2-4× with 4 GPUs.

### Asynchronous decode
Run decoder blocks on the current token while the encoder/routing stage processes the next token. The hierarchy naturally suggests a pipeline: stage i processes token t while stage i-1 processes token t+1.

Requires thread-level parallelism and careful state management.

**Potential:** ~2×.

---

## 4. Architectural changes (long-term research)

### Convert to sliding-window attention throughout
Replace all HGRN blocks with sliding-window attention (as `hybrid_attn` already does for the innermost stage). Significantly faster on GPU (attention matmuls are highly optimized), possibly slower on CPU.

**Tradeoff:** Loses the O(1) memory property — KV cache grows with window size.

### Skip quantization at inference time
The BitLinear quantization (activation_quant + weight_quant) is needed only during training. At inference, pre-quantize weights once and skip activation_quant. This removes 2 of 3 operations per BitLinear forward.

Requires separating the training and inference forward paths.

**Potential:** ~2-3×.

---

## Summary

| Improvement | Effort | Speedup | Notes |
|---|---|---|---|
| FP16 inference | Hours | 1.5-2× | Trivial, always worth doing |
| torch.compile | Hours | 1.3-2× | Free on GPU |
| Skip quantization at inference | Days | 2-3× | Clean engineering, no architecture change |
| Fused step kernel | Weeks | 3-5× | Best single improvement |
| Speculative decoding | Weeks | 2-4× | Works for any autoregressive model |
| Pipeline across GPUs | Weeks | 2-4× | Only if multi-GPU available |

**Realistic target with 1-2 weeks of engineering (FP16 + skip quant + torch.compile + fused kernel):** ~300-500 tok/s on A100, ~30-50 tok/s on CPU.
