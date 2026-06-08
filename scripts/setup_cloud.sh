#!/usr/bin/env bash
# Cloud GPU setup script for the Spanish benchmark.
# Run this on the cloud machine after copying the code.
set -euo pipefail

echo "=== Setting up HNetBit benchmark environment ==="
echo ""

# Detect CUDA
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: No NVIDIA driver found. This benchmark requires a GPU."
    echo "Make sure you are renting a GPU instance with CUDA drivers installed."
    exit 1
fi
echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || nvidia-smi -L
echo ""

# Python version check
PYTHON_VERSION=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
echo "Python: $PYTHON_VERSION"
if ! python3 -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo "ERROR: Python 3.10+ required."
    exit 1
fi

# Create virtual environment (optional, skip if using container)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 12.4)
echo ""
echo "Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install core ML dependencies
echo ""
echo "Installing ML dependencies..."
pip install \
    transformers \
    datasets \
    tokenizers \
    accelerate \
    huggingface_hub \
    safetensors \
    einops \
    triton \
    sentencepiece \
    protobuf

# Install causal-conv1d (optional, for ShortConvolution)
echo ""
echo "Installing causal-conv1d..."
pip install causal-conv1d 2>/dev/null || {
    echo "WARNING: causal-conv1d installation failed."
    echo "The benchmark will use the fallback conv implementation."
    echo "This is slightly slower but functionally equivalent."
}

# Development / debugging
echo ""
echo "Installing dev tools..."
pip install tensorboard matplotlib pandas pytest tqdm

# Verify torch + CUDA
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Verify Triton
python3 -c "
import triton
print(f'Triton: {triton.__version__}')
" 2>/dev/null && echo "Triton: OK" || echo "WARNING: Triton not available"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. (Optional) Log in to HuggingFace for transformer baseline:"
echo "     huggingface-cli login"
echo ""
echo "  2. Run smoke test:"
echo "     bash test_smoke.sh --gpu"
echo ""
echo "  3. Run benchmark:"
echo "     python train_spanish.py --model hybrid --size 150M"
echo ""
echo "  To run detached (no hangup):"
echo "     nohup python train_spanish.py --model hybrid --size 150M > hybrid_150M.log 2>&1 &"
