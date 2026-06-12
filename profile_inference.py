#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference throughput profiler for all model architectures.

Measures:
    - Prefill latency (time-to-first-token) at multiple context lengths
    - Decode throughput (tokens/second) at multiple batch sizes
    - Peak GPU memory during inference

Works for: transformer, matmulfree, hybrid, hybrid_attn

Usage:
    python profile_inference.py --checkpoint runs/spanish/hybrid_150M/checkpoint_best.pt
    python profile_inference.py --export export/hybrid_150M_deploy.pt
    python profile_inference.py --checkpoint checkpoint.pt --prefill-only
    python profile_inference.py --checkpoint checkpoint.pt --decode-only --batch_sizes 1,4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_model_from_source(
    checkpoint_path: Optional[str] = None,
    export_path: Optional[str] = None,
    device: str = "cuda",
) -> tuple:
    """Load a model from a training checkpoint or compact export."""
    from model_factory import build_model

    if export_path and os.path.exists(export_path):
        print(f"Loading from compact export: {export_path}")
        data = torch.load(export_path, map_location="cpu", weights_only=False)
        cfg = data.get("model_config", {})
        model_name = cfg.get("model_name", "hybrid")
        model_size = cfg.get("model_size", "150M")
        model, is_byte_level, vocab_size = build_model(model_name, model_size)
        if "packed_weights" in data:
            from hnet_bit.ops.bitnet import unpack_ternary_tensor
            full_sd = dict(data["model_state_dict"])
            for key, packed in data["packed_weights"].items():
                full_sd[key] = unpack_ternary_tensor(packed)
            model.load_state_dict(full_sd)
        else:
            model.load_state_dict(data["model_state_dict"])
    elif checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config", {})
        if "model_name" in cfg and "model_size" in cfg:
            model, is_byte_level, vocab_size = build_model(cfg["model_name"], cfg["model_size"])
        elif cfg.get("model_type") == "hnet_bit":
            from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM
            hnet_cfg = HNetBitConfig(**{k: v for k, v in cfg.items()
                                         if k in HNetBitConfig().__dict__})
            model = HNetBitForCausalLM(hnet_cfg)
            is_byte_level = True
            vocab_size = 256
        else:
            raise ValueError(f"Cannot determine model type from config: {list(cfg.keys())[:10]}")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise FileNotFoundError("No checkpoint or export path provided")

    model = model.to(device)
    model.eval()
    return model, is_byte_level, vocab_size


def _timed_forward(model, input_ids, attention_mask=None, warmup=3, runs=5):
    """Time a single forward pass using CUDA events."""
    use_cuda = input_ids.device.type == "cuda"

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    if use_cuda:
        torch.cuda.synchronize()

    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs):
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / runs
    else:
        t0 = time.time()
        for _ in range(runs):
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        elapsed_ms = (time.time() - t0) / runs * 1000

    return elapsed_ms


def measure_prefill(
    model, vocab_size: int, device: str,
    seq_lengths=(512, 1024, 2048, 4096),
) -> dict:
    """Measure time-to-first-token (prefill latency) at multiple sequence lengths."""
    results = {}
    print("\n--- Prefill (Time-to-First-Token) ---")

    for sl in seq_lengths:
        input_ids = torch.randint(0, min(vocab_size, 256), (1, sl), device=device)
        attention_mask = torch.ones(1, sl, device=device)

        torch.cuda.reset_peak_memory_stats(device) if device == "cuda" else None
        torch.cuda.empty_cache() if device == "cuda" else None

        ms = _timed_forward(model, input_ids, attention_mask, warmup=3, runs=5)

        peak_mem = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    if device == "cuda" else 0.0)

        results[str(sl)] = {"ttft_ms": round(ms, 1), "peak_mem_mb": round(peak_mem, 0)}
        print(f"  seq_len={sl:>5}  ttft={ms:>7.1f} ms  mem={peak_mem:>6.0f} MB")

    return results


def measure_decode(
    model, vocab_size: int, device: str,
    batch_sizes=(1, 4, 8),
    prompt_len: int = 256,
    max_new_tokens: int = 256,
) -> dict:
    """Measure autoregressive decode throughput at multiple batch sizes."""
    results = {}
    print("\n--- Decode Throughput ---")

    for bs in batch_sizes:
        try:
            input_ids = torch.randint(0, min(vocab_size, 256), (bs, prompt_len), device=device)
            attention_mask = torch.ones(bs, prompt_len, device=device)

            torch.cuda.reset_peak_memory_stats(device) if device == "cuda" else None
            torch.cuda.empty_cache() if device == "cuda" else None

            # Warmup
            _ = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=16, do_sample=False, pad_token_id=0,
                eos_token_id=None, use_cache=True,
            )
            if device == "cuda":
                torch.cuda.synchronize()

            # Timed generation
            if device == "cuda":
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
            else:
                t0 = time.time()

            output_ids = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=0,
                eos_token_id=None, use_cache=True,
            )

            if device == "cuda":
                end_ev.record()
                torch.cuda.synchronize()
                elapsed_s = start_ev.elapsed_time(end_ev) / 1000
            else:
                elapsed_s = time.time() - t0

            new_tokens = output_ids.shape[1] - prompt_len
            total_tokens = new_tokens * bs
            tok_per_sec = total_tokens / max(elapsed_s, 0.001)

            peak_mem = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                        if device == "cuda" else 0.0)

            results[f"batch_{bs}"] = {
                "tok_per_sec": round(tok_per_sec, 0),
                "total_tokens": total_tokens,
                "elapsed_s": round(elapsed_s, 2),
                "peak_mem_mb": round(peak_mem, 0),
            }
            print(f"  batch={bs:>2}  tok/s={tok_per_sec:>8.0f}  "
                  f"tokens={total_tokens:>6}  time={elapsed_s:>6.1f}s  mem={peak_mem:>6.0f} MB")

        except Exception as e:
            print(f"  batch={bs:>2}  ERROR: {e}")
            results[f"batch_{bs}"] = {"tok_per_sec": 0, "error": str(e)}

    return results


def profile_model(
    checkpoint_path: Optional[str] = None,
    export_path: Optional[str] = None,
    device: str = "cuda",
    prefill_seq_lengths: tuple = (512, 1024, 2048, 4096),
    decode_batch_sizes: tuple = (1, 4, 8),
    prefill_only: bool = False,
    decode_only: bool = False,
) -> dict:
    """Run full inference profile and return results dict."""
    model, is_byte_level, vocab_size = load_model_from_source(checkpoint_path, export_path, device)
    param_count = sum(p.numel() for p in model.parameters())

    result = {
        "model_type": type(model).__name__,
        "is_byte_level": is_byte_level,
        "vocab_size": vocab_size,
        "param_count": param_count,
        "device": device,
    }

    if not decode_only:
        result["prefill"] = measure_prefill(model, vocab_size, device, prefill_seq_lengths)

    if not prefill_only:
        result["decode"] = measure_decode(model, vocab_size, device, decode_batch_sizes)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Profile inference throughput for all model architectures"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to training checkpoint (.pt)")
    parser.add_argument("--export", type=str, default=None,
                        help="Path to compact export file (.pt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for profile JSON (default: auto-named)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for inference")
    parser.add_argument("--prefill_lengths", type=str, default="512,1024,2048,4096",
                        help="Comma-separated prefill sequence lengths")
    parser.add_argument("--batch_sizes", type=str, default="1,4,8",
                        help="Comma-separated decode batch sizes")
    parser.add_argument("--prefill-only", action="store_true",
                        help="Only run prefill benchmark")
    parser.add_argument("--decode-only", action="store_true",
                        help="Only run decode benchmark")
    args = parser.parse_args()

    if not args.checkpoint and not args.export:
        print("ERROR: Must provide --checkpoint or --export")
        sys.exit(1)

    prefill_lengths = tuple(int(x) for x in args.prefill_lengths.split(","))
    batch_sizes = tuple(int(x) for x in args.batch_sizes.split(","))

    result = profile_model(
        checkpoint_path=args.checkpoint,
        export_path=args.export,
        device=args.device,
        prefill_seq_lengths=prefill_lengths,
        decode_batch_sizes=batch_sizes,
        prefill_only=args.prefill_only,
        decode_only=args.decode_only,
    )

    # Auto-generate output path
    if args.output:
        output_path = args.output
    elif args.checkpoint:
        base = os.path.splitext(args.checkpoint)[0]
        output_path = f"{base}_inference_profile.json"
    elif args.export:
        base = os.path.splitext(args.export)[0]
        output_path = f"{base}_inference_profile.json"
    else:
        output_path = "inference_profile.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nProfile saved to: {output_path}")


if __name__ == "__main__":
    main()
