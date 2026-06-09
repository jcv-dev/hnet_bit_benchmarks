#!/usr/bin/env bash
set -euo pipefail

# Smoke test for the Spanish benchmark pipeline.
# Without --gpu: tests hybrid only (works on CPU).
# With    --gpu: also tests matmulfree + transformer (requires CUDA + HF login).

ROOT="$(cd "$(dirname "$0")" && pwd)"
SMOKE_DIR="$ROOT/runs/smoke_test"
DATA_DIR="$ROOT/data/smoke_test"
GPU_MODE=false
FAIL=

# Parse args
for arg in "$@"; do
    case "$arg" in
        --gpu) GPU_MODE=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

MODELS=("hybrid")
if $GPU_MODE; then
    MODELS+=("matmulfree" "hybrid_attn")
    # transformer requires HuggingFace login for gpt2 tokenizer
    HF_OK=false
    if command -v huggingface-cli &>/dev/null && huggingface-cli whoami &>/dev/null 2>&1; then
        HF_OK=true
    elif [ -f ~/.cache/huggingface/token ] && python3 -c "from huggingface_hub import whoami; whoami()" &>/dev/null 2>&1; then
        HF_OK=true
    fi
    if $HF_OK; then
        MODELS+=("transformer")
    else
        echo "  [--gpu] Skipping transformer: not logged into HuggingFace"
        echo "    Login with:  python3 -c \"from huggingface_hub import login; login()\""
    fi
fi

echo "=== Smoke Test: Spanish Benchmark Pipeline ==="
echo "  Models: ${MODELS[*]}"
echo "  GPU:    $GPU_MODE"
echo ""

# ------------------------------------------------------------------
# 1. Create synthetic corpus
# ------------------------------------------------------------------
echo "--- Creating synthetic corpus ---"
mkdir -p "$DATA_DIR"
# ~500KB of repeated Spanish text (validation set needs >=8192 bytes for 1 sample)
python3 -c "
text = 'el gato y el perro caminan por la calle oscura y silenciosa '
text += 'la casa es grande y tiene muchas ventanas azules '
text += 'los nios juegan en el parque mientras el sol se pone '
content = (text * 3500)[:500000]
with open('$DATA_DIR/corpus.bin', 'wb') as f:
    f.write(content.encode('utf-8'))
print(f'  Wrote {len(content)} bytes -> corpus.bin')
"

# For transformer: delete stale BPE data and rebuild from current corpus
if [[ " ${MODELS[*]} " == *" transformer "* ]]; then
    echo "--- Building BPE corpus for transformer ---"
    rm -f "$DATA_DIR"/corpus_bpe.npy "$DATA_DIR"/corpus_meta.npz
    python3 -c "
import numpy as np
from pathlib import Path
from data_spanish import SpanishCorpusBuilder
cache = Path('$DATA_DIR')
# Build BPE data from existing corpus.bin (skip HF download)
builder = SpanishCorpusBuilder(
    cache_dir=str(cache),
    tokenizer_name='gpt2',
    max_samples=10,
)
total_tokens, avg = builder.build_bpe(force=False)
meta = dict(total_bytes=builder.byte_path.stat().st_size,
            total_tokens=total_tokens,
            avg_bytes_per_token=avg)
np.savez(builder.meta_path, **meta)
print(f'  BPE done: {total_tokens} tokens, {avg:.2f} bytes/token')
" 2>&1 | tail -3
fi

# ------------------------------------------------------------------
# 2. Run each model
# ------------------------------------------------------------------
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "--- Running $MODEL tiny (15 steps) ---"

    EXTRA_ARGS=""
    if [ "$MODEL" = "transformer" ]; then
        # For transformer, use gpt2 tokenizer (open, no login needed for tiny test)
        EXTRA_ARGS="--tokenizer_name gpt2"
    fi

    python3 "$ROOT/train_spanish.py" \
        --model "$MODEL" \
        --size tiny \
        --max_steps 15 \
        --batch_size 2 \
        --grad_accum 1 \
        --total_tokens 100000 \
        --output_dir "$SMOKE_DIR" \
        --cache_dir "$DATA_DIR" \
        --lr 1e-3 \
        --no_bf16 \
        --skip_data_build \
        $EXTRA_ARGS 2>&1 | tail -10

    # Check output files
    RUN_DIR="$SMOKE_DIR/${MODEL}_tiny"
    RESULTS_CSV="$SMOKE_DIR/results_${MODEL}_tiny.csv"
    echo "  Checking output files ..."
    for f in "config.json" "training_stats.json" "validation_log.csv"; do
        if [ -f "$RUN_DIR/$f" ]; then
            echo "    OK: $f"
        else
            echo "    MISSING: $RUN_DIR/$f"
            FAIL=1
        fi
    done
    if [ -f "$RESULTS_CSV" ]; then
        echo "    OK: $RESULTS_CSV"
    else
        echo "    MISSING: $RESULTS_CSV"
        FAIL=1
    fi
done

# ------------------------------------------------------------------
# 3. Generate aggregated results
# ------------------------------------------------------------------
echo ""
echo "--- Aggregating results ---"
python3 "$ROOT/generate_results.py" \
    --runs_dir "$SMOKE_DIR" \
    --output "$SMOKE_DIR/results.csv" 2>&1 | tail -15

echo ""
echo "=== Smoke test complete ==="
if [ -n "${FAIL:-}" ]; then
    echo "WARNING: Some files were missing!"
fi
exit ${FAIL:-0}
