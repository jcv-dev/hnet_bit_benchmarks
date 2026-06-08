#!/usr/bin/env bash
# Cloud GPU setup script for the Spanish benchmark.
# Works with Vast.ai PyTorch template (detects /venv/main/ automatically).
# Also works as a PROVISIONING_SCRIPT on Vast.ai.
set -euo pipefail

echo "=== Setting up HNetBit benchmark environment ==="
echo ""

# Detect CUDA
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: No NVIDIA driver found. This benchmark requires a GPU."
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

# Use the template's pre-existing venv if available, else create a local one
VENV_PATH="/venv/main"
if [ -d "$VENV_PATH" ]; then
    echo "Using template venv at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    if [ ! -d "venv" ]; then
        echo "Creating local virtual environment..."
        python3 -m venv venv
    fi
    source venv/bin/activate
fi

# Prefer uv if available (much faster), fall back to pip
if command -v uv &>/dev/null; then
    PKG_MANAGER="uv pip"
else
    PKG_MANAGER="pip"
fi
echo "Package manager: $PKG_MANAGER"

# Check if PyTorch already has CUDA working; skip reinstall if so
echo ""
echo "Checking PyTorch..."
PYTORCH_OK=$(python3 -c "
import torch
print('ok' if torch.cuda.is_available() else 'no_cuda')
" 2>/dev/null || echo "missing")

if [ "$PYTORCH_OK" = "ok" ]; then
    echo "PyTorch with CUDA already present -- skipping reinstall."
    python3 -c "import torch; print(f'  Version: {torch.__version__}, CUDA: {torch.version.cuda}')"
else
    echo "Installing PyTorch with CUDA 12.4..."
    $PKG_MANAGER install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

# Install core ML dependencies
echo ""
echo "Installing ML dependencies..."
$PKG_MANAGER install \
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

# Install causal-conv1d (needs ninja for parallel CUDA compilation)
echo ""
echo "Installing causal-conv1d..."
$PKG_MANAGER install ninja 2>/dev/null || true
pip install causal-conv1d 2>/dev/null || {
    echo "WARNING: causal-conv1d installation failed."
    echo "The benchmark will use the fallback conv implementation."
    echo "This is slightly slower but functionally equivalent."
}

# Dev / debugging tools
echo ""
echo "Installing dev tools..."
$PKG_MANAGER install tensorboard matplotlib pandas pytest tqdm

# Verify huggingface-cli is available (installed by huggingface_hub)
echo ""
echo "Checking CLI tools..."
if command -v huggingface-cli &>/dev/null; then
    echo "  huggingface-cli: OK"
else
    echo "  WARNING: huggingface-cli not found. Run: source /venv/main/bin/activate"
fi

# Verification
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch:        {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:   {torch.version.cuda}')
    print(f'GPU:            {torch.cuda.get_device_name(0)}')
    print(f'Memory:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

python3 -c "import triton; print(f'Triton:         {triton.__version__}')" \
    && echo "Triton:         OK" \
    || echo "WARNING: Triton not available"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. (Optional) Log in to HuggingFace:"
echo "     huggingface-cli login"
echo ""
echo "  2. Run smoke test:"
echo "     bash test_smoke.sh --gpu"
echo ""
echo "  3. Run benchmark:"
echo "     python train_spanish.py --model hybrid --size 150M"
echo ""
echo "  Using tmux (recommended — survives SSH disconnect):"
echo "     tmux new -s hybrid_150M"
echo "     # Inside session: cd ~/tesis && source /venv/main/bin/activate"
echo "     python train_spanish.py --model hybrid --size 150M"
echo "     # Detach: Ctrl+B then D. Reattach: tmux attach -t hybrid_150M"
echo ""
echo "  Using nohup (alternative):"
echo "     nohup python train_spanish.py --model hybrid --size 150M > hybrid_150M.log 2>&1 &"
echo "     tail -f hybrid_150M.log"
