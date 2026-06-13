#!/usr/bin/env python3
"""
Reconstruct validation_log.csv and training metrics from surviving step checkpoints.

Usage:
  python reconstruct_logs.py --run_dir runs/spanish/transformer_150M
"""

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

import torch
import numpy as np

from model_factory import build_model
from data_spanish import create_dataloaders
from metrics_spanish import compute_bpb
from training_config_spanish import SpanishTrainingConfig


def parse_step(filename: str) -> int:
    m = re.search(r"step_(\d+)", filename)
    return int(m.group(1)) if m else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to run directory with .pt checkpoints")
    parser.add_argument("--cache_dir", default="./data/spanish")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Eval batches per checkpoint (default: 200)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} not found")
        return

    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        return
    with open(config_path) as f:
        config_data = json.load(f)

    model_name = config_data["model_name"]
    model_size = config_data["model_size"]
    avg_bytes_per_token = float(config_data.get("avg_bytes_per_token", 1.0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model: {model_name} {model_size}, device={device}")
    print(f"avg_bytes_per_token={avg_bytes_per_token}")

    model, is_byte_level, vocab_size = build_model(model_name, model_size)
    model.to(device)

    _, val_loader = create_dataloaders(
        model_name=model_name,
        cache_dir=args.cache_dir,
        byte_seq_length=config_data.get("byte_seq_length", 4096),
        token_seq_length=config_data.get("token_seq_length", 1280),
        batch_size=config_data.get("batch_size", 4),
    )

    checkpoints = sorted(glob.glob(str(run_dir / "checkpoint_step_*.pt")), key=parse_step)
    if not checkpoints:
        print("ERROR: no checkpoint_step_*.pt files found")
        return

    print(f"\nFound {len(checkpoints)} checkpoints")

    records = []

    for ckpt_path in checkpoints:
        step = parse_step(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        bytes_seen = ckpt.get("bytes_seen", 0)

        result = compute_bpb(
            model, val_loader,
            is_byte_level=is_byte_level,
            avg_bytes_per_token=avg_bytes_per_token,
            device=device,
            max_batches=args.max_batches or 200,
        )

        print(f"  step={step:>7,}  bytes={bytes_seen:>15,}  loss={result['loss']:.4f}  "
              f"bpb={result['bpb']:.4f}")

        records.append({
            "step": step,
            "bytes_seen": bytes_seen,
            "val_loss": round(result["loss"], 6),
            "val_bpb": round(result["bpb"], 6),
        })

        del ckpt
        torch.cuda.empty_cache()

    out_csv = run_dir / "validation_log_reconstructed.csv"
    if records:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"\nSaved {len(records)} records to {out_csv}")

    train_csv = run_dir / "training_steps_log_reconstructed.csv"
    config_lr = float(config_data.get("learning_rate", 0))
    total_steps = int(config_data.get("total_steps", 0))
    warmup_steps = int(config_data.get("warmup_steps", 0))
    stable_steps = int(config_data.get("stable_steps", 0))
    log_interval = int(config_data.get("log_interval_steps", 10))

    train_records = []
    for ckpt_path in checkpoints:
        step = parse_step(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        bytes_seen = ckpt.get("bytes_seen", 0)
        del ckpt

        if step <= warmup_steps:
            lr = config_lr * step / max(warmup_steps, 1)
        elif step <= warmup_steps + stable_steps:
            lr = config_lr
        else:
            decay_steps_left = total_steps - (warmup_steps + stable_steps)
            steps_in_decay = step - (warmup_steps + stable_steps)
            lr = config_lr * (0.5 * (1 + np.cos(np.pi * steps_in_decay / max(decay_steps_left, 1))))

        train_records.append({
            "step": step,
            "bytes_seen": bytes_seen,
            "lr": round(lr, 8),
            "loss": "",
            "grad_norm": "",
            "tok_per_sec": "",
            "peak_mem_mb": "",
        })

    if train_records:
        with open(train_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=train_records[0].keys())
            writer.writeheader()
            writer.writerows(train_records)
        print(f"Saved {len(train_records)} skeleton records to {train_csv}")
        print("  (loss, grad_norm, tok/s, mem: unrecoverable from checkpoints alone)")


if __name__ == "__main__":
    main()
