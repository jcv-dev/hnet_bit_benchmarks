#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compact ternary-weight model export for all architectures.

Converts a training checkpoint (FP32 master weights + optimizer + scheduler)
into a lightweight deployment file containing only frozen ternary {-1,0,+1}
weights and necessary metadata.

Works for: transformer, matmulfree, hybrid, hybrid_attn

Usage:
    python export_deployment.py --checkpoint runs/spanish/hybrid_150M/checkpoint_best.pt
    python export_deployment.py --checkpoint runs/spanish/hybrid_150M/checkpoint_best.pt --output model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from hnet_bit.ops.bitnet import pack_ternary_tensor

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_factory import build_model


def _find_bitlinear_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find all BitLinear / FusedBitLinear modules in the model."""
    matches = []
    for name, module in model.named_modules():
        clsname = type(module).__name__
        if clsname in ("BitLinear", "FusedBitLinear"):
            matches.append((name, module))
    return matches


def freeze_ternary_weights(model: nn.Module) -> int:
    """
    Freeze all BitLinear weights to hard ternary {-1, 0, +1} in-place.

    Calls weight_quant() on every BitLinear/FusedBitLinear .weight and replaces
    the tensor. No STE — weights are now permanently ternary and cannot be trained.

    Returns the number of BitLinear layers frozen.
    """
    from hnet_bit.ops.bitnet import weight_quant

    frozen = 0
    for _name, module in _find_bitlinear_modules(model):
        with torch.no_grad():
            module.weight.data = weight_quant(module.weight.data)
        frozen += 1
    return frozen


def _build_model_from_checkpoint(checkpoint: dict) -> Tuple[nn.Module, bool, int, str, str]:
    """
    Rebuild a model from a checkpoint dict.

    Handles both Spanish benchmark checkpoints (SpanishTrainingConfig)
    and hnet_bit internal checkpoints (HNetBitConfig or training_config).
    """
    cfg = checkpoint.get("config", {})

    if "model_name" in cfg and "model_size" in cfg:
        model, is_byte_level, vocab_size = build_model(cfg["model_name"], cfg["model_size"])
        return model, is_byte_level, vocab_size, cfg["model_name"], cfg["model_size"]

    if "model_type" in cfg and cfg["model_type"] == "hnet_bit":
        from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM
        hnet_cfg = HNetBitConfig(**{k: v for k, v in cfg.items()
                                     if k in HNetBitConfig().__dict__})
        model = HNetBitForCausalLM(hnet_cfg)
        return model, True, 256, "hybrid", "unknown"

    raise ValueError(
        "Cannot determine model type from checkpoint config. "
        "Expected keys: model_name+model_size (Spanish benchmark) "
        f"or model_type (hnet_bit). Found: {list(cfg.keys())[:10]}"
    )


def export_model(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    verify: bool = True,
) -> dict:
    """
    Export a training checkpoint to a compact ternary-weight deployment file.

    Args:
        checkpoint_path: Path to checkpoint_best.pt or checkpoint_final.pt
        output_path: Output path for the export. Auto-generated if None.
        verify: If True, run a forward pass to verify logit equality.

    Returns:
        dict with stats: checkpoint_size_mb, fp16_size_mb, export_size_mb,
                         bits_per_param, compression_ratio, frozen_layers,
                         param_count, model_name, model_size.
    """
    checkpoint_path = str(checkpoint_path)
    checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)

    print(f"Loading checkpoint: {checkpoint_path}  ({checkpoint_size_mb:.1f} MB)")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model, is_byte_level, vocab_size, model_name, model_size = _build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    param_count = sum(p.numel() for p in model.parameters())
    non_emb_count = sum(
        p.numel() for name, p in model.named_parameters()
        if "embed" not in name and "lm_head" not in name
    )
    fp16_size_mb = param_count * 2 / (1024 ** 2)

    # Determine device for verification
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save pre-freeze state on CPU before any device move
    if verify:
        pre_freeze_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Move to GPU so short_conv (causal conv) works correctly
    if device == "cuda":
        model = model.to(device)

    frozen = freeze_ternary_weights(model)

    if frozen == 0 and model_name not in ("transformer",):
        print(f"  WARNING: No BitLinear layers found in {model_name} model. "
              "Weights remain full-precision.")

    # Count actual storage bits after freezing
    total_bits = 0
    for name, param in model.named_parameters():
        n = param.numel()
        module = None
        for m_name, m_module in _find_bitlinear_modules(model):
            if f"{m_name}.weight" == name:
                module = m_module
                break
        if module is not None:
            total_bits += n * 2 + 32
        else:
            total_bits += n * 32

    bits_per_param = total_bits / max(param_count, 1)

    # Auto-generate output path
    if output_path is None:
        output_dir = os.path.dirname(checkpoint_path)
        output_path = os.path.join(output_dir, "model_deploy.pt")

    # Verify forward pass matches
    if verify:
        print("  Verifying forward pass...")
        model.eval()
        x = torch.randint(0, min(vocab_size, 256), (1, 64), device=device)
        with torch.no_grad():
            out_frozen = model(input_ids=x)
            frozen_logits = out_frozen.logits if hasattr(out_frozen, "logits") else out_frozen[0]

        model.load_state_dict(pre_freeze_state)
        model.eval()
        with torch.no_grad():
            out_orig = model(input_ids=x)
            orig_logits = out_orig.logits if hasattr(out_orig, "logits") else out_orig[0]

        max_diff = (frozen_logits - orig_logits).abs().max().item()
        if max_diff < 1e-5:
            print(f"  VERIFIED: max logit diff = {max_diff:.2e} (OK)")
        else:
            print(f"  WARNING: max logit diff = {max_diff:.2e} (may indicate issue)")

        freeze_ternary_weights(model)

    # Move back to CPU for packing and saving
    model = model.cpu()

    # Build packed state dict: separate BitLinear weights → packed format
    full_sd = model.state_dict()
    packed_weights = {}
    for module_name, module in _find_bitlinear_modules(model):
        weight_key = f"{module_name}.weight"
        if weight_key in full_sd:
            packed_weights[weight_key] = pack_ternary_tensor(full_sd.pop(weight_key))

    # Save once without stats to measure exact file size, then resave with stats
    export_state = {
        "model_state_dict": full_sd,
        "packed_weights": packed_weights,
        "model_config": {
            "model_name": model_name,
            "model_size": model_size,
            "param_count": param_count,
            "non_embedding_param_count": non_emb_count,
            "vocab_size": vocab_size,
            "is_byte_level": is_byte_level,
            "frozen_layers": frozen,
        },
    }
    torch.save(export_state, output_path)
    export_size_mb = os.path.getsize(output_path) / (1024 ** 2)

    export_state["export_stats"] = {
        "checkpoint_size_mb": round(checkpoint_size_mb, 1),
        "fp16_size_mb": round(fp16_size_mb, 1),
        "export_size_mb": round(export_size_mb, 1),
        "bits_per_param": round(bits_per_param, 2),
        "compression_ratio": round(fp16_size_mb / max(export_size_mb, 0.01), 1),
    }
    torch.save(export_state, output_path)

    stats = {
        "model_name": model_name,
        "model_size": model_size,
        "param_count": param_count,
        "non_embedding_param_count": non_emb_count,
        "checkpoint_size_mb": round(checkpoint_size_mb, 1),
        "fp16_size_mb": round(fp16_size_mb, 1),
        "export_size_mb": round(export_size_mb, 1),
        "bits_per_param": round(bits_per_param, 2),
        "compression_ratio": round(fp16_size_mb / max(export_size_mb, 0.01), 1),
        "frozen_layers": frozen,
        "output_path": output_path,
    }
    return stats


def load_deploy_model(export_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a model from a compact deployment export file.

    Automatically unpacks ternary weights (packed at ~2 bits/param).
    Handles both the new packed format and legacy unpacked exports.

    Args:
        export_path: Path to the export file (model_deploy.pt).
        device: Device to place the model on ("cpu" or "cuda").

    Returns:
        Model with all weights loaded (ternary weights unpacked to float32).
    """
    from model_factory import build_model
    from hnet_bit.ops.bitnet import unpack_ternary_tensor

    data = torch.load(export_path, map_location="cpu", weights_only=False)
    cfg = data["model_config"]
    model, _, _ = build_model(cfg["model_name"], cfg["model_size"])

    full_sd = dict(data["model_state_dict"])
    if "packed_weights" in data:
        for key, packed in data["packed_weights"].items():
            full_sd[key] = unpack_ternary_tensor(packed)

    model.load_state_dict(full_sd)
    model = model.to(device)
    model.eval()
    return model


def print_report(stats: dict) -> None:
    """Print a human-readable export report."""
    print()
    print("=" * 60)
    print(f"  Export: {stats['model_name']}_{stats['model_size']}")
    print("=" * 60)
    print(f"  Parameters          : {stats['param_count']:,}  "
          f"({stats['non_embedding_param_count']:,} non-emb)")
    print(f"  Training checkpoint : {stats['checkpoint_size_mb']:,.1f} MB")
    print(f"  FP16 equivalent     : {stats['fp16_size_mb']:,.1f} MB")
    print(f"  Compact export      : {stats['export_size_mb']:,.1f} MB")
    print(f"  Compression ratio   : {stats['compression_ratio']:.1f}x vs fp16")
    print(f"  Bits per parameter  : {stats['bits_per_param']:.2f}")
    print(f"  Frozen layers       : {stats['frozen_layers']}")
    print(f"  Saved to            : {stats['output_path']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Export training checkpoint to compact ternary-weight deployment file"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint_best.pt or checkpoint_final.pt")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for compact model (default: <checkpoint_dir>/model_deploy.pt)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip forward-pass verification")
    parser.add_argument("--json", action="store_true",
                        help="Output stats as JSON instead of human-readable")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    stats = export_model(
        args.checkpoint,
        args.output,
        verify=not args.no_verify,
    )

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_report(stats)


if __name__ == "__main__":
    main()
