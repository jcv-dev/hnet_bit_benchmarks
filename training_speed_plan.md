# Training Speed & Memory Improvement Plan (hnet_bit / hybrid)

Status: PLAN — not yet implemented.
Target: hybrid / hybrid_attn training (hnet_bit), applies to **future runs only**.

> **Decision (2026-08-12):** The 350M campaign is already in flight on 3 Vast.ai
> A100-80GB machines and the methodology is strict — **no changes are applied to
> the running 350M tier**. Everything below is future work (restarts, a revived
> 750M tier, or a fresh re-run), documented as a methodology note in the thesis.

---

## 1. Baseline (completed 150M tier + live 350M observations)

### 150M tier (completed)

| Model | s/step | Peak mem (alloc/reserved) | Throughput | Total time | Final BPB |
|---|---|---|---|---|---|
| hybrid | 1.52 | 15.4 / 18.1 GB | 86 KB/s | 80.6 h | 1.6221 |
| matmulfree | 2.27 | 3.4 / 3.8 GB | 58 KB/s | 120.3 h | 1.4743 |
| transformer | **~0.75** | 6.7 / 8.5 GB | ~171 KB/s | 40.3 h | 1.4837 |

> Correction (2026-08-12): an earlier analysis quoted transformer 150M at 1.38 MB/s
> / 0.094 s/step. That number came from the *reconstructed* training log whose first
> rows contain resume fast-forward artifacts (tok/s spikes ~1e9). Actual steady state
> is ~0.75 s/step from `training_stats.json` (40.3 h ÷ 193,247 steps). The transformer
> is ~2× faster per byte than hybrid at 150M, **not** ~16×.

### 350M tier (live, ~1–2% through)

| Model | s/step | Peak mem | Projected total | Notes |
|---|---|---|---|---|
| hybrid | 2.32 | 27.9 GB | ~123 h | matches process.md (100–125 h) |
| matmulfree | 4.33 | 7.5 GB | ~229 h | **faster than process.md's 300–340 h estimate** |
| transformer | ~1.8 (early, still warmup) | 12.1 GB | ~115–130 h | GC genuinely active ("Enabled via gradient_checkpointing_enable()") |

## 2. Findings (what is wrong today)

1. **Training uses the sequential HGRN scan.** `HGRNBitAttention` calls
   `fused_recurrent_hgrn` (hnet_bit/layers/hgrn_bit.py:196) — a serial loop over
   T=4096. The repo already contains a tested chunk-parallel `chunk_hgrn`
   (hnet_bit/ops/hgrn/chunk.py) that is exported and unit-tested but never used.
   `DeChunkLayer.forward` also uses the fused scan (hnet_bit/ops/dynamic_chunking.py:440,482).
2. **Gradient checkpointing is a silent no-op for hybrid.**
   `training_config_spanish.py:253` calls HF `gradient_checkpointing_enable()`, but
   `HGRNBitBlock.forward` never wraps with `torch.utils.checkpoint`. Only ~1.1 GB of
   the 15.4 GB (150M) is weights/grads/optimizer; ~13 GB is activations. Live proof
   of the asymmetry: at 350M, hybrid uses 27.9 GB vs transformer's 12.1 GB.
3. **i/f/g projections are 3 separate BitLinear calls** on the same input
   (hgrn_bit.py:99-101): 3× RMSNorm + 3× activation quant + 3× weight quant + 3 GEMMs.
   They are all linear in the same `hidden_states` and can merge into one
   `hidden → 3·input_dim` projection (the MLP already uses this trick).
4. **Per-micro-step CUDA syncs.** train_spanish.py:272 calls
   `boundary_mask.float().mean().item()` every micro-step (8× per optimizer step);
   `accum_loss += loss.item()` (line 281) adds another.
5. **Duplicate argsort.** ChunkLayer (dynamic_chunking.py:316) and DeChunkLayer
   (dynamic_chunking.py:471) each compute the identical argsort over the same
   `boundary_mask`.
6. **Allocator fragmentation.** 18.1 GB reserved vs 15.4 GB allocated (17% headroom).
7. **No real ternary GEMM.** `F.linear` runs on dense bf16 copies of weights that are
   ~69% zeros; no kernel exploits the ternary structure.

## 3. Revised bottleneck hypothesis (2026-08-12, from 350M scaling data)

The original plan hypothesized a launch-bound step with ~1 s fixed overhead. The new
350M evidence **contradicts** that:

- Hybrid 150M→350M: FLOPs/step ratio 1.55×, observed time ratio 1.53× — **exactly
  FLOP-proportional**. If a large fixed overhead existed, the time ratio would be
  below the FLOP ratio. The implied fixed overhead is tiny (~0.07 s).
- Both tiers run at a constant **~10% of A100 bf16 peak** (28–29 TF/s effective).
- Transformer runs at ~17–20%, matmulfree ~12% — each stable across tiers.

Conclusion: the model is **efficiency-bound** — the per-FLOP cost is bad (~10% of
peak) and stable with scale. The cause is the kernel mix (small norm/quant/fp32-
intermediate kernels interleaved with GEMMs, serial scan dependency chains), not
CPU launch rate. CPU at ~4% is consistent with either regime (one GIL-bound core,
syncs make the thread sleep); it is not a discriminator.

Consequence for priorities: fixes that attack the ~10% constant (fusion, chunk scan,
fewer fp32 passes) matter more than launch-rate fixes; the ternary kernel's ceiling
is **bounded** because the problem is kernel shape/occupancy, not GEMM FLOP count.

## 4. Step 0 — Diagnostic gate (run BEFORE implementation, zero cost)

```bash
nvidia-smi dmon -d 5      # watch sm% (GPU util) on the hybrid machine
```

- **sm% in 40–60 band** (predicted given the scaling data): small-kernel stall /
  efficiency-bound regime → priorities as ranked in §5.
- **sm% > 90**: GEMM-efficiency-bound → move item 7 (ternary kernel) up; the rest
  still applies.
- **sm% < 20**: host-bound after all → sync removal moves back up.

If available, run `torch.profiler` (or `nsys`) for ~20 steps on a 150M checkpoint
to calibrate the per-item estimates.

## 5. Planned changes (re-prioritized) — with impact estimates

Estimates assume Triton available, A100, hybrid 150M shape. Direction is confident,
magnitude needs the Step-0 gate.

| Prio | Change | Files | Speed | Memory | Risk/Effort |
|---|---|---|---|---|---|
| **1** | **Fuse i/f/g into one BitLinear** (chunk output after GEMM) | hnet_bit/layers/hgrn_bit.py | **+15–30%** — direct attack on the ~10% efficiency constant (fewer norm/quant passes, bigger GEMMs) | −0.5 GB | Medium / ~1 day |
| 2 | Chunk-parallel HGRN for training (`chunk_recurrent_hgrn`: plain `torch.logsigmoid(z)` wrapper — no custom autograd needed, derivative is 1−sigmoid(z), division-free) | hnet_bit/ops/hgrn/chunk.py, hgrn_bit.py, dynamic_chunking.py | +5–15% (removes serial scan stalls between GEMMs) | −0.5–1 GB | Medium / 1–2 days |
| 3 | Real gradient checkpointing per block (honor existing `gradient_checkpointing` / `_gradient_checkpointing` flags) | hnet_bit/models/hnet_bit.py (HGRNBitStack) | −10–20% at same batch; **+10–25% net if batch 4→8** | **15.4 → 6–8 GB (batch 4) or 11–13 GB (batch 8)** | Medium / 1–2 days |
| 4 | Remove `.item()` syncs (boundary ratios + accum_loss as GPU tensors, flush at log time) | train_spanish.py, metrics_spanish.py | **+5–15%** (downgraded from +15–35%: the fixed-overhead model it attacked was disproven) | — | Trivial / hours |
| 5 | Compute argsort once in HNetBit.forward, pass to Chunk+DeChunk | hnet_bit/ops/dynamic_chunking.py, models/hnet_bit.py | +1–3% | — | Trivial |
| 6 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | scripts/setup_cloud.sh, process.md | — | reserved 18.1 → ~16 GB | Trivial |
| 7 | Triton ternary-GEMM kernel (2-bit packed weights) — **deprioritized**: ceiling bounded while GEMMs run at ~10% efficiency; only pursue after 1–6 validated AND profiling shows GEMMs dominate the remainder | hnet_bit/ops/ (new, flag-gated off) | +15–30% at best in current regime; **+1.5–2.5× only if 1–6 first lift GEMM efficiency** | −1–2 GB | High / ~2–4 weeks |

Combined realistic target (items 1–6, batch 8 + GC): 1.52 → ~0.5–0.9 s/step,
80.6 h → ~27–45 h (**2–3×**), peak memory 15.4 → 11–13 GB at batch 8 (6–8 GB at
batch 4). At 350M: 2.32 → ~1.1–1.6 s/step → ~58–85 h instead of ~123 h.

## 6. Implementation notes (resolved during planning)

- **Chunk-HGRN wiring**: new `chunk_recurrent_hgrn(x, z, ...)` = `chunk_hgrn(x,
  torch.logsigmoid(z), ...)`; gradient chains through `logsigmoid` automatically
  (backward `dz = dg·(1−sigmoid(z))`, no division → safe against bf16 gate
  underflow). New `attn_mode='chunk_recurrent'` for training/prefill; seq=1
  generation auto-falls back to `fused_recurrent` (existing `mode` selection in
  hgrn_bit.py:156). DeChunkLayer EMA: `g = log(1−p)`, p already clamped to
  [1e-4, 1−1e-4] → always finite.
- **Fusion scope**: valid in the `share_conv_kernel` branch (production config).
  The separate-conv branch keeps per-stream projections. Old checkpoints
  (i_proj/f_proj/g_proj) load via a `_load_from_state_dict` hook that rebuilds
  the fused weight by concatenation.
- **GC flag**: `enable_gradient_checkpointing()` (HF) sets `gradient_checkpointing`
  on submodules; honor both that and `_gradient_checkpointing`. Training only
  (`self.training`, `use_cache=False`, no cache).
- **Ternary kernel**: off by default (`use_ternary_gemm=False` config flag); pack
  layout = 4 K-consecutive weights per byte (distinct from deployment's
  interleaved `pack_ternary_tensor`); decode via `tl.join`; dense STE backward
  (no custom autograd inside the fused norm/quant Function — the outer backward
  already computes dense `dy`/`dW`).

## 7. Validation

- Equivalence: fused vs chunk HGRN on identical weights/inputs (rtol 1e-3);
  reuse existing tests (`test_model.py::TestHGRN`, `test_hgrn_bit.py`,
  `test_dynamic_chunking.py`, `test_fused_bitlinear.py`).
- `bash test_smoke.sh --gpu` (all models).
- 100-step debug run (seed 42) comparing loss trajectory vs the saved 150M logs.
- 5k-step stability run measuring s/step, peak memory, chunk ratios, val BPB
  vs baseline at the equivalent bytes-seen point.
- Re-run `nvidia-smi dmon` during the debug run — sm% should rise toward 80–90%
  if the efficiency-bound diagnosis is right.

## 8. Timing & methodology notes

- 350M tier runs unmodified (decision above). Changes apply to restarts, the
  750M tier, or a fresh re-run — documented as a methodology note.
- Numerics shift ~1e-7 (chunk-HGRN exp(logsigmoid) vs direct sigmoid; fusion
  shares one activation-quant scale where the original used three) — document
  in thesis + AGENTS.md.
- After item 3 lands, update AGENTS.md's "Gradient checkpointing gap" section
  (the asymmetry vs transformer/matmulfree disappears for future runs).

## Appendix A — Scaling analysis (supporting material for the thesis)

Training-cost scaling (25B-byte budget fixed), power-law fits `t ∝ P^α` from the
two observed tiers:

| Model | α | 350M | 750M | 1B | 2B |
|---|---|---|---|---|---|
| transformer | 1.12 (≈ linear) | ~120 h | ~187 h | ~258 h | ~560 h |
| hybrid | 0.38 (sublinear) | ~123 h | ~153 h | ~171 h | ~223 h |

- **Crossover at ~573M params** — squarely in the dropped 750M tier; at 2B params
  the hybrid is ~2.5× cheaper to train.
- **Caveat (must be stated)**: α=0.38 is partly a recipe artifact — the 150M→350M
  jump added a hierarchy level (1-stage → 2-stage), buying params in stages that
  see only ~10% of tokens. If a 2B recipe scales all stages uniformly, α → ~1 and
  the crossover disappears. Single seed, 2 points per model → scenario, not
  prediction. The 750M tier is the direct crossover test (~6–8 days/model).
- Quality at 150M: hybrid is *worst* (1.6221 vs 1.4743 matmulfree / 1.4837
  transformer) — the scaling story is about training cost, not quality.

## Appendix B — Recommended-budget analysis (from the 150M validation log)

| Budget | Steps | Time @ 1.52 s/step | val BPB | vs full budget |
|---|---|---|---|---|
| Chinchilla 20 tok/param (2.76B bytes) | 21,000 | ~8.9 h | 1.6885 | +0.072 BPB (4.4% worse) |
| Chinchilla 50 tok/param (6.9B bytes) | 53,000 | ~22.4 h | 1.6386 | +0.022 BPB (1.3% worse) |
| Thesis budget (25B bytes) | 190,734 | 80.6 h | 1.6169 | baseline |

~96% of final quality (1/BPB framing) at ~11% of wall time — a property of the
equal-bytes protocol (all models are over-trained 4–9× past Chinchilla by design),
not of any architecture.

## Appendix C — Fairness framing (for the thesis)

The benchmark equalizes protocol (bytes, optimizer, WSD, seed, batch, paper-default
LRs, no sweeps) — it measures protocol-constrained behavior, not per-architecture
optimum. Cite: AGENTS.md methodology + "Known limitations".
