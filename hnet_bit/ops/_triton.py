# Runtime Triton availability check.
# Triton's JIT compiler may be installed but fail to compile kernels
# in restricted environments (unprivileged Docker containers).
# This module provides a unified availability check with:
#   - Import-time kernel compilation test
#   - Environment variable override (HNETBIT_DISABLE_TRITON=1)

import os
import logging

_TRITON_AVAILABLE = False
_TRITON_FAILURE_REASON = None

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    # Test that Triton can actually compile a trivial kernel
    @triton.jit
    def _probe_kernel(x_ptr, n: tl.constexpr):
        idx = tl.arange(0, n)
        tl.store(x_ptr + idx, tl.full([n], 1.0, dtype=tl.float32))

    def _probe():
        import torch
        x = torch.zeros(8, device="cuda", dtype=torch.float32)
        _probe_kernel[(1,)](x, 8)
        return True

    if os.environ.get("HNETBIT_DISABLE_TRITON", "").lower() in ("1", "true", "yes"):
        _TRITON_FAILURE_REASON = "disabled by HNETBIT_DISABLE_TRITON env var"
        logger.info("Triton disabled by HNETBIT_DISABLE_TRITON")
    elif torch.cuda.is_available():
        try:
            _TRITON_AVAILABLE = _probe()
        except Exception as e:
            _TRITON_FAILURE_REASON = (
                f"Triton kernel compilation failed: {e}. "
                "This is common in unprivileged Docker containers that block JIT compilation. "
                "To fix: rent a VM template instead of a Docker template, "
                "or use RunPod (privileged containers), "
                "or set HNETBIT_DISABLE_TRITON=1 to use naive PyTorch loops (slower but works)."
            )
        if not _TRITON_AVAILABLE and not _TRITON_FAILURE_REASON:
            _TRITON_FAILURE_REASON = "kernel compilation test returned False"
    else:
        _TRITON_FAILURE_REASON = "CUDA not available"

except Exception as e:
    _TRITON_FAILURE_REASON = str(e)
    logger.info(f"Triton not available: {e}")


def triton_available() -> bool:
    return _TRITON_AVAILABLE


def triton_failure_reason() -> str:
    return _TRITON_FAILURE_REASON
