## **OBJECTIVE 1: Implement Simplified MatMul-Free SLM with Ternary Weights**

### **A1. Research State of the Art (1-2 weeks)**

**Steps:**
1. **Study MatmulFree-LM architecture**
   - Read the paper and analyze hgrn_bit
   - Understand BitLinear layers in bitnet.py
   - Study HGRN (Hierarchical Gated Recurrent Network) in hgrn_bit.py

2. **Study H-Net architecture (Dynamic Chunking for Hierarchical Sequence Modeling)**
   - Analyze hnet.py - hierarchical nested architecture:
     * Multi-stage processing (encoder → main_network → decoder) at each hierarchy level
     * Recursive nesting: main_network at stage N is another HNet at stage N+1
     * Isotropic (non-hierarchical) blocks at the innermost level
   - Understand the mixer architecture in mixer_seq.py:
     * HNetForCausalLM: Full model with embedding and LM head
     * Handles packed sequences with cu_seqlens for efficiency
   - **Study Dynamic Chunking (dc.py) - KEY INNOVATION:**
     * **RoutingModule**: Predicts chunk boundaries using cosine similarity
       - Computes boundary_prob = (1 - cosine_similarity) / 2 between consecutive tokens
       - Adaptive segmentation based on semantic similarity
     * **ChunkLayer**: Groups tokens into variable-length chunks
     * **DeChunkLayer**: Reconstructs full sequence from chunk representations
   - Analyze Isotropic blocks (isotropic.py):
     * Uses Mamba2 SSM (via mamba_ssm) for efficient sequence modeling
     * Supports Multi-head Attention as alternative

3. **Document key findings**
   - Create a comparison table: MatMul operations vs. alternatives
   - List ternary quantization techniques {-1, 0, +1}
   - Identify computational savings (memory, FLOPs)
   - **Compare sequence modeling approaches:**
     | Feature | HGRN (MatmulFree) | H-Net | Simplified SLM |
     |---------|-------------------|-------|----------------|
     | Quantization | Ternary | FP16/FP32 | Ternary |
     | Sequence Model | Gated Recurrence | Mamba2 SSM | Gated Recurrence |
     | Chunking | Fixed | Dynamic (content-aware) | TBD |
     | Hierarchy | None | Multi-stage | TBD |
     | Complexity | O(n) | O(n) | O(n) |

### **A2. Design Simplified Architecture (1 week)**

**Steps:**
1. **Define your hybrid architecture combining H-Net + MatmulFree concepts**
   
   **Core Design Principles:**
   - **No traditional tokenization**: Use byte-level input (vocab_size=256) - already implemented
   - **Ternary weights**: BitLinear with {-1, 0, +1} quantization
   - **Efficient sequence modeling**: Gated recurrence (HGRN) with O(n) complexity
   
   **Architecture Options (choose based on complexity/performance trade-off):**
   
   **Option A: Simplified (current simplified_slm)** - Recommended for initial implementation
   ```
   Input: Raw bytes (vocab_size=256, no tokenizer needed)
   Embedding layer (256 → hidden_size)
   N × HGRNBitBlock with:
     * RMSNorm → HGRNBitAttention (ternary) → Residual
     * RMSNorm → HGRNBitMLP (SwiGLU, ternary) → Residual
   Output: BitLinear LM head (hidden_size → 256)
   ```
   
   **Option B: With H-Net Dynamic Chunking** - Advanced, higher performance potential
   ```
   Input: Raw bytes (vocab_size=256)
   Embedding layer (256 → hidden_size)
   N × blocks with:
     * HGRNBitBlock (ternary, local context)
     * Dynamic Chunking Module (adapted from H-Net):
       - RoutingModule: cosine similarity boundary prediction
       - ChunkLayer: group tokens by semantic similarity
       - Process chunks at lower resolution
       - DeChunkLayer: reconstruct full sequence
   Output: BitLinear LM head (hidden_size → 256)
   ```
   
   **Option C: Hierarchical (H-Net inspired)** - Most complex, best long-range modeling
   ```
   Input: Raw bytes (vocab_size=256)
   Embedding layer (256 → d_model[0])
   HNetBit (hierarchical with ternary layers):
     Stage 0 (finest): Encoder → Main → Decoder
       Main → Stage 1 (coarser): Encoder → Main → Decoder
         Main → Stage 2 (coarsest): Isotropic blocks
     * All linear layers use BitLinear (ternary)
     * Dynamic chunking at each stage boundary
   Output: BitLinear LM head (d_model[0] → 256)
   ```

2. **Create architecture diagram**
   - Draw block diagram showing data flow
   - Specify dimensions (hidden_size, num_layers, etc.)
   - Choose model size: Start with ~10M-100M parameters
   - Document ternary weight placement

3. **Write configuration file**
   - Create `configs/slm_ternary_simplified.json` based on simplified_slm config
   - **Add optional H-Net inspired parameters:**
     ```json
     {
       "vocab_size": 256,
       "hidden_size": 512,
       "num_hidden_layers": 6,
       "use_dynamic_chunking": false,
       "num_stages": 1,
       "d_model": [512],
       "enable_hierarchical": false
     }
     ```
   - Define hyperparameters: layers, hidden_dim, etc.

### **A3. Implement Key Components (2-3 weeks)**

**Steps:**
1. **Create project structure** (already exists in simplified-slm)
   ```bash
   # Current structure
   simplified_slm/
   ├── models/        # Model classes and configs
   ├── layers/        # HGRNBitBlock, attention layers
   ├── ops/           # BitLinear, activations, HGRN ops
   ├── utils/         # ByteTokenizer, cache, helpers
   └── configs/       # Model configurations
   ```

2. **Verify BitLinear layer** (already implemented from MatmulFree)
   - Location: `ops/bitnet.py`
   - Ensure ternary quantization: `sign(W) ∈ {-1, 0, +1}`
   - Test with random tensors
   - Verify gradient flow through STE (Straight-Through Estimator)

3. **Implement/verify HGRN recurrence**
   - Location: `layers/hgrn_bit.py`
   - Uses fused recurrent operations from `ops/hgrn/`
   - Gated recurrence: h_t = f_t * h_{t-1} + i_t
   - All projections use BitLinear (ternary)

4. **[OPTIONAL] Implement Dynamic Chunking Module (from H-Net)**
   If Option B or C is chosen, adapt from H-Net:
   
   a. **Create `ops/dynamic_chunking.py`**:
   ```python
   class RoutingModuleBit(nn.Module):
       """Predict chunk boundaries using cosine similarity with ternary weights."""
       def __init__(self, d_model):
           self.q_proj = BitLinear(d_model, d_model, bias=False)
           self.k_proj = BitLinear(d_model, d_model, bias=False)
       
       def forward(self, hidden_states):
           # Cosine similarity between consecutive tokens
           cos_sim = F.normalize(self.q_proj(hidden_states[:, :-1])) @ \
                     F.normalize(self.k_proj(hidden_states[:, 1:]))
           boundary_prob = (1 - cos_sim) / 2
           return boundary_prob
   ```
   
   b. **Adapt ChunkLayer and DeChunkLayer**:
   - Copy from `hnet/modules/dc.py`
   - Ensure compatibility with ternary weight layers
   
   c. **Create `layers/hierarchical_block.py`**:
   - Combine HGRNBitBlock with dynamic chunking
   - Multi-stage processing at different resolutions

5. **Implement normalization layers** (already done)
   - RMSNorm in `ops/bitnet.py`
   - FusedRMSNormSwishGate in `ops/fused_norm_gate.py`

6. **Verify model class** (already exists)
   - `SimplifiedSLMForCausalLM` in `models/modeling.py`
   - Stack blocks: `[Embedding → N×HGRNBitBlock → Norm → LM_Head]`
   - Check `forward()` and `generate()` methods
   - All linear operations use BitLinear (ternary)

7. **[OPTIONAL] Implement hierarchical model class**
   If Option C is chosen, create `models/hnet_bit.py`:
   - Adapt HNet architecture with BitLinear layers
   - Multi-stage nested structure
   - Dynamic chunking between stages

### **A4. Integration and Unit Testing (1 week)**

**Steps:**
1. **Unit tests for each component**
   ```python
   # Test BitLinear quantization (verify ternary)
   def test_bitlinear_ternary():
       layer = BitLinear(256, 512)
       qweight = weight_quant(layer.weight)
       assert set(qweight.unique().tolist()).issubset({-1, 0, 1})
   
   # Test HGRN forward pass
   def test_hgrn_attention():
       attn = HGRNBitAttention(hidden_size=512)
       x = torch.randn(2, 32, 512)
       out, _, _ = attn(x)
       assert out.shape == x.shape
   
   # Test full model forward pass
   def test_model_forward():
       config = SimplifiedSLMConfig(vocab_size=256)
       model = SimplifiedSLMForCausalLM(config)
       input_ids = torch.randint(0, 256, (2, 64))
       output = model(input_ids)
       assert output.logits.shape == (2, 64, 256)
   
   # Test gradient flow
   def test_gradient_flow():
       model = SimplifiedSLMForCausalLM(config)
       loss = model(input_ids, labels=input_ids).loss
       loss.backward()
       for p in model.parameters():
           assert p.grad is not None
   ```

2. **Integration test**
   - Create dummy input: `torch.randint(0, 256, (batch, seq_len))` (raw bytes)
   - Run forward pass
   - Verify output shape: `(batch, seq_len, 256)` for byte vocabulary
   - Check memory footprint

3. **Verify ternary weights**
   - After initialization, check all BitLinear weights quantize to {-1, 0, +1}
   - Measure actual memory vs. FP16 baseline
   - Compute theoretical compression ratio

4. **[If implementing dynamic chunking] Test chunking module**
   ```python
   def test_routing_module():
       router = RoutingModuleBit(d_model=512)
       x = torch.randn(2, 64, 512)
       probs = router(x)
       assert probs.shape == (2, 63)  # boundaries between consecutive tokens
       assert (probs >= 0).all() and (probs <= 1).all()
   ```

5. **Documentation**
   - Write docstrings for all classes/functions
   - Update `README.md` with architecture description
   - Document design decisions and H-Net vs MatmulFree trade-offs

---

## **OBJECTIVE 2: Training and Inference Experiments**

### **A5. Dataset Selection and Preprocessing (1 week)**

**Steps:**
1. **Choose dataset** (pick one for scope control):
   - **Spanish**: OSCAR Spanish subset (10-50M bytes)
   - **English**: WikiText-103 or TinyStories
   - **Bilingual**: CC-100 Spanish/English subset

2. **Download and prepare**
   ```python
   from datasets import load_dataset
   dataset = load_dataset("wikipedia", "20220301.es", split="train[:10%]")
   ```

3. **Preprocessing - BYTE-LEVEL (No Tokenization)**
   ```python
   # Direct byte encoding - no tokenizer training needed!
   from simplified_slm.utils import ByteTokenizer
   
   tokenizer = ByteTokenizer()  # vocab_size=256
   
   def encode_bytes(example):
       text = example['text']
       encoded = tokenizer.encode([text])[0]
       return {'input_ids': encoded['input_ids'].tolist()}
   
   dataset = dataset.map(encode_bytes)
   ```
   - **Key advantage**: No vocabulary learning, works for any language/script
   - Create train/val/test splits (80/10/10)
   - Chunk into sequences (e.g., 512 or 1024 bytes)
   - Save preprocessed data: `.arrow` or `.bin` format

4. **Data statistics**
   - Document: sequence count, average length, byte distribution
   - Note: vocab_size is always 256 (all possible bytes)

### **A6. Experiment Configuration (3-4 days)**

**Steps:**
1. **Create training config**
   ```json
   {
     "model": "simplified_slm_ternary",
     "vocab_size": 256,
     "hidden_size": 512,
     "num_layers": 6,
     "num_heads": 1,
     "batch_size": 32,
     "learning_rate": 3e-4,
     "max_steps": 10000,
     "warmup_steps": 500,
     "eval_interval": 500,
     "gradient_accumulation_steps": 4,
     "max_seq_length": 512,
     "weight_decay": 0.01,
     "use_dynamic_chunking": false
   }
   ```

2. **Set up training script**
   - Adapt existing training utilities from simplified_slm
   - Use PyTorch or Hugging Face Trainer
   - Implement learning rate scheduler (cosine or linear warmup)
   - Add gradient clipping (1.0)
   - **For byte-level**: Ensure UTF-8 handling is correct

3. **Configure hardware**
   - Determine GPU availability (single GPU, multi-GPU, CPU-only)
   - Set mixed precision training (FP16) if GPU supports it
   - Estimate training time based on hardware
   - **Note**: Ternary weights reduce compute but still need FP16/FP32 for training

4. **Set up logging**
   - Use Weights & Biases, TensorBoard, or simple CSV logging
   - Log: loss, perplexity (per byte), learning rate, memory usage
   - **Track ternary weight statistics** during training

### **A7. Execute Training Experiments (2-3 weeks)**

**Steps:**
1. **Baseline experiment**
   - Train for 5-10K steps
   - Monitor loss convergence
   - Save checkpoints every 1K steps
   - **Note**: Byte-level perplexity is different from token-level; expect higher values

2. **Hyperparameter variations** (optional but recommended):
   - Experiment 1: Vary learning rate (1e-4, 3e-4, 1e-3)
   - Experiment 2: Vary model size (small: 4 layers, base: 6 layers, large: 12 layers)
   - Experiment 3: Different quantization strategies (ternary vs binary)
   - **Experiment 4: Enable dynamic chunking (if implemented)**
   - **Experiment 5: Test hierarchical architecture (if implemented)**

3. **Monitor training**
   - Check for NaN losses or gradient explosions
   - Verify GPU/memory utilization
   - Adjust batch size if OOM errors occur
   - **Monitor ternary weight distribution** during training

4. **Save artifacts**
   - Best checkpoint (lowest validation loss)
   - Training logs and curves
   - Configuration files used

### **A8. Execute Inference Experiments (1 week)**

**Steps:**
1. **Set up generation script**
   - Use existing `generate.py` in simplified_slm
   - Implement sampling strategies: greedy, top-k, top-p (nucleus)
   - Add temperature control

2. **Qualitative evaluation (byte-level generation)**
   - Generate text with various prompts (10-20 examples)
   - Example prompts (raw bytes): 
     ```python
     prompt = "Once upon a time".encode('utf-8')  # Direct bytes
     # Or using ByteTokenizer
     tokenizer = ByteTokenizer()
     input_ids = tokenizer.encode(["The capital of Spain is"])
     generated = model.generate(input_ids, max_new_tokens=100)
     text = tokenizer.decode(generated[0].tolist())
     ```
   - Document generated outputs
   - **Check UTF-8 validity** of generated byte sequences

3. **Quantitative inference**
   - Measure generation speed (bytes/second)
   - Test on different sequence lengths (128, 256, 512 bytes)
   - Compare with baseline FP16 model (if available)
   - Measure **ternary weight efficiency** (ops without MatMul)

---

## **OBJECTIVE 3: Evaluation and Analysis**

### **A9. Design Evaluation Framework (3-4 days)**

**Steps:**
1. **Select performance metrics**:
   - **Bits per byte (BPB)**: Primary metric for byte-level LM (instead of perplexity)
     ```python
     # BPB = cross_entropy_loss / ln(2)
     bpb = loss / math.log(2)
     ```
   - **Perplexity**: exp(loss), but interpret as per-byte
   - **Accuracy**: Next-byte prediction accuracy
   - **UTF-8 validity rate**: Percentage of valid UTF-8 in generated text

2. **Select efficiency metrics**:
   - **FLOPs**: Floating-point operations per forward pass
   - **Memory**: Model size (MB), peak GPU memory
   - **Latency**: Inference time (ms per byte)
   - **Throughput**: Bytes processed per second
   - **Ternary efficiency**: Ratio of additions vs multiplications

3. **Create evaluation script**
   - Implement metric computation functions
   - Set up test dataset iterator
   - **Prepare comparison baselines:**
     - Standard Transformer/FP16 version
     - Original MatmulFree-LM (if comparable size)
     - H-Net (for architecture comparison, different quantization)

### **A10. Measure Performance Metrics (3-4 days)**

**Steps:**
1. **Compute perplexity**
   ```python
   loss = model.evaluate(test_dataloader)
   perplexity = torch.exp(loss)
   ```

2. **Compute accuracy**
   ```python
   correct = 0
   total = 0
   for batch in test_dataloader:
       predictions = model(batch.input_ids).argmax(dim=-1)
       correct += (predictions == batch.labels).sum()
       total += batch.labels.numel()
   accuracy = correct / total
   ```

3. **Statistical analysis**
   - Run multiple inference passes (5-10 runs)
   - Compute mean and standard deviation
   - Create comparison tables

### **A11. Measure Efficiency Metrics (3-4 days)**

**Steps:**
1. **FLOPs calculation**
   ```python
   from fvcore.nn import FlopCountAnalysis
   flops = FlopCountAnalysis(model, inputs)
   print(f"FLOPs: {flops.total()}")
   ```

2. **Memory measurement**
   ```python
   # Model size
   model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
   
   # Peak GPU memory
   import torch
   torch.cuda.reset_peak_memory_stats()
   output = model(input_ids)
   peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
   ```

3. **Latency measurement**
   ```python
   import time
   # Warmup
   for _ in range(10):
       model(input_ids)
   
   # Measure
   start = time.time()
   for _ in range(100):
       output = model(input_ids)
   latency_ms = (time.time() - start) / 100 * 1000
   ```

4. **Create comparison table**
   | Metric | Ternary SLM | FP16 Baseline | Improvement |
   |--------|-------------|---------------|-------------|
   | Model Size | X MB | Y MB | Z% reduction |
   | FLOPs | ... | ... | ... |
   | Latency | ... | ... | ... |

### **A12. Analysis and Conclusions (1 week)**

**Steps:**
1. **Analyze results**
   - What is the bits-per-byte vs. efficiency trade-off?
   - Did ternary quantization maintain reasonable performance?
   - Where are the bottlenecks? (memory bandwidth, computation, etc.)
   - **Compare architecture choices:**
     - HGRN (flat) vs H-Net-inspired (hierarchical)
     - With vs without dynamic chunking
     - Impact of byte-level vs token-level modeling

2. **Identify limitations**
   - Training instability issues?
   - Quality degradation in specific tasks?
   - Hardware limitations encountered?
   - **Byte-level specific**: UTF-8 handling, long-range dependencies

3. **Propose improvements**
   - Better quantization schemes
   - Architectural modifications (adopt more H-Net features?)
   - Training recipe improvements
   - **Dynamic chunking benefits**: When is it worth the complexity?

4. **Validate hypotheses**
   - Did MatMul-Free reduce computation as expected?
   - Is ternary quantization viable for SLMs?
   - **Does tokenization-free (byte-level) approach scale?**
   - **Would H-Net hierarchical structure improve long-range modeling?**

### **A13. Write Final Document (2-3 weeks)**

**Steps:**
1. **Structure**
   ```
   1. Introduction
   2. Background
      - MatMul-Free Language Models (BitLinear, HGRN)
      - H-Net: Dynamic Chunking for Hierarchical Sequence Modeling
      - Ternary Weight Quantization
      - Byte-Level (Tokenization-Free) Modeling
   3. Methodology
      - Architecture Design (combining MatmulFree + H-Net insights)
      - Implementation Details
      - Design Decisions and Trade-offs
   4. Experiments
      - Setup (byte-level preprocessing, training config)
      - Training Results
      - Evaluation Protocol
   5. Results
      - Performance Metrics (BPB, accuracy)
      - Efficiency Metrics (FLOPs, memory, latency)
      - Architecture Comparison (flat vs hierarchical, with/without chunking)
   6. Analysis and Discussion
      - Ternary Quantization Impact
      - Byte-Level vs Token-Level Trade-offs
      - H-Net Features: When Are They Worth It?
   7. Conclusions and Future Work
   8. References
   9. Appendices (Code, Detailed Results, H-Net Adaptation Details)
   ```

2. **Key sections to emphasize**
   - Clear architectural diagrams (show H-Net influence)
   - Detailed experimental setup (reproducibility)
   - Comprehensive results tables and plots
   - Honest discussion of limitations
   - **Comparison with existing approaches (MatmulFree, H-Net)**

3. **Figures and tables**
   - Training curves (loss, BPB over time)
   - Generated text examples (show UTF-8 handling)
   - Efficiency comparison charts
   - Architecture diagram (highlight ternary layers)
   - **Architecture comparison table:**
     | Feature | MatmulFree | H-Net | SimplifiedSLM |
     |---------|------------|-------|---------------|
     | Weights | Ternary | FP16 | Ternary |
     | Tokenization | BPE | BPE | Byte-level |
     | Sequence Model | HGRN | Mamba2 | HGRN |
     | Chunking | None | Dynamic | Optional |
     | Hierarchy | Flat | Multi-stage | Flat (w/ option) |

---

## **SUMMARY: KEY ARCHITECTURAL DECISIONS**

### Finalized Design: Simplified SLM with Ternary Weights

| Component | Source | Description |
|-----------|--------|-------------|
| **Embedding** | Custom | 256 → hidden_size (byte-level, no tokenizer) |
| **BitLinear** | MatmulFree | Ternary weights {-1, 0, +1} via STE |
| **Sequence Model** | MatmulFree | HGRNBitAttention with gated recurrence |
| **MLP** | MatmulFree | HGRNBitMLP with SwiGLU, BitLinear |
| **Normalization** | MatmulFree | RMSNorm, FusedRMSNormSwishGate |
| **Dynamic Chunking** | H-Net (opt.) | RoutingModule for adaptive segmentation |
| **Hierarchical** | H-Net (opt.) | Multi-stage processing at coarse/fine |
| **LM Head** | Custom | BitLinear (hidden_size → 256) |

### Why This Design?
1. **No tokenization**: Byte-level input eliminates vocabulary learning, works for any language
2. **Ternary weights**: ~8x memory reduction, addition-based computation
3. **HGRN**: O(n) complexity, efficient training with chunk-parallel
4. **H-Net options**: Dynamic chunking and hierarchy available for future enhancement

---

## **RECOMMENDED TIMELINE**

**Weeks 1-4**: Objective 1 (Implementation)
- Week 1-2: Research (A1) - Deep dive into MatmulFree and H-Net
- Week 2-3: Design (A2) - Choose architecture option
- Week 3-4: Implementation (A3) - Core components + optional H-Net features
- Week 4: Testing (A4)

**Weeks 5-8**: Objective 2 (Experiments)
- Week 5: Dataset prep (A5) - Byte-level preprocessing
- Week 5-6: Training setup (A6-A7)
- Week 7-8: Inference experiments (A8)

**Weeks 9-11**: Objective 3 (Evaluation)
**Weeks 12-14**: Final document writing

## **TOOLS & LIBRARIES TO INSTALL**

```bash
pip install torch transformers datasets accelerate
pip install triton>=2.2 einops  # Required for HGRN ops
pip install wandb tensorboard  # for logging
pip install fvcore  # for FLOPs calculation
pip install pytest  # for unit tests
pip install matplotlib seaborn pandas  # for visualization
# pip install mamba-ssm  # Optional: if implementing H-Net Mamba2 blocks
```

## **IMMEDIATE NEXT STEPS**

1. ✅ Project structure exists (simplified_slm)
2. ✅ Core components implemented (BitLinear, HGRNBitBlock, ByteTokenizer)
3. **Next actions:**
   - [ ] Run unit tests to verify implementation: `pytest tests/`
   - [ ] Test generation with dummy model: `python generate.py`
   - [ ] Decide on architecture Option A/B/C based on research
   - [ ] If Option B/C: Implement dynamic chunking module from H-Net
   - [ ] Prepare byte-level dataset for training
   - [ ] Set up training script with logging

## **REFERENCES**

- **MatMul-Free LM**: https://arxiv.org/abs/2406.02528
- **H-Net (Dynamic Chunking)**: https://arxiv.org/abs/2507.07955  
- **BitNet**: https://arxiv.org/abs/2310.11453
- **HGRN**: https://arxiv.org/abs/2311.04823