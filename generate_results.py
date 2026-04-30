#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post‑training script to aggregate results from all model × size runs
into a single ``results.csv``.

Usage:
    python generate_results.py --runs_dir ./runs/spanish --output results.csv

    # Or evaluate all final checkpoints fresh:
    python generate_results.py --runs_dir ./runs/spanish --reeval --cache_dir ./data/spanish
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_factory import build_model
from data_spanish import create_dataloaders
from metrics_spanish import compute_bpb, measure_inference_memory


def collect_existing_results(runs_dir: str) -> list[dict]:
    """
    Scan *runs_dir* for per-run result files and merge them.

    Collects:
    - BPB and Inference_Memory_MB from ``results_*.csv``
    - Peak Tokens/sec and Best Train Loss from ``training_steps_log.csv``
    - Best BPB from ``validation_log.csv``
    """
    runs = Path(runs_dir)
    all_results = []

    for csv_file in sorted(runs.rglob("results_*.csv")):
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["BPB"] = float(row["BPB"])
                row["Inference_Memory_MB"] = float(row["Inference_Memory_MB"])

                # Try to enrich with throughput from training_steps_log.csv
                steps_log = csv_file.parent / "training_steps_log.csv"
                row["Peak_Tok_Per_Sec"] = ""
                row["Best_Train_Loss"] = ""
                if steps_log.exists():
                    try:
                        with open(steps_log) as sf:
                            steps = list(csv.DictReader(sf))
                        if steps:
                            row["Peak_Tok_Per_Sec"] = round(
                                max(float(s["tok_per_sec"]) for s in steps), 1
                            )
                            row["Best_Train_Loss"] = round(
                                min(float(s["loss"]) for s in steps), 4
                            )
                    except Exception:
                        pass

                # Try to enrich with best BPB from validation_log.csv
                val_log = csv_file.parent / "validation_log.csv"
                row["Best_Val_BPB"] = row["BPB"]  # fallback to final
                if val_log.exists():
                    try:
                        with open(val_log) as vf:
                            evals = list(csv.DictReader(vf))
                        if evals:
                            row["Best_Val_BPB"] = round(
                                min(float(e["val_bpb"]) for e in evals), 4
                            )
                    except Exception:
                        pass

                all_results.append(row)

    return all_results


def reevaluate_checkpoints(
    runs_dir: str,
    cache_dir: str = "./data/spanish",
    device: str = "cuda",
) -> list[dict]:
    """
    Find all ``checkpoint_final.pt`` files under *runs_dir*, rebuild the
    corresponding model, load weights, and compute metrics fresh.
    """
    runs = Path(runs_dir)
    all_results = []

    for ckpt_path in sorted(runs.rglob("checkpoint_final.pt")):
        # Infer model_name and size from directory name <model>_<size>
        dir_name = ckpt_path.parent.name  # e.g. "hybrid_350M"
        parts = dir_name.rsplit("_", 1)
        if len(parts) != 2:
            print(f"  Skipping {ckpt_path} (cannot parse dir name)")
            continue
        model_name, size = parts

        print(f"\n=== Re‑evaluating: {model_name} {size} ===")
        print(f"    Checkpoint: {ckpt_path}")

        try:
            model, is_byte_level, vocab_size = build_model(model_name, size)
        except Exception as e:
            print(f"  ERROR building model: {e}")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        # Determine avg_bytes_per_token
        avg_bytes_per_token = 1.0
        if model_name == "transformer":
            meta_path = Path(cache_dir) / "corpus_meta.npz"
            if meta_path.exists():
                meta = dict(np.load(meta_path))
                avg_bytes_per_token = float(meta.get("avg_bytes_per_token", 4.5))
            else:
                avg_bytes_per_token = 4.5

        # Data loaders
        _, val_loader = create_dataloaders(
            model_name=model_name,
            cache_dir=cache_dir,
            batch_size=4,
        )

        # BPB
        bpb_results = compute_bpb(
            model, val_loader, is_byte_level,
            avg_bytes_per_token=avg_bytes_per_token,
            device=device,
        )

        # Inference memory
        mem_results = measure_inference_memory(
            model, seq_length=2048,
            is_byte_level=is_byte_level,
            vocab_size=vocab_size,
            device=device,
        )

        result = {
            "Model": model_name,
            "Size": size,
            "BPB": round(bpb_results["bpb"], 4),
            "Best_Val_BPB": round(bpb_results["bpb"], 4),
            "Inference_Memory_MB": round(mem_results["peak_memory_mb"], 1),
            "Peak_Tok_Per_Sec": "",
            "Best_Train_Loss": "",
        }
        all_results.append(result)

        print(f"  BPB              : {result['BPB']:.4f}")
        print(f"  Inference Memory : {result['Inference_Memory_MB']:.1f} MB")

        # Free memory
        del model
        torch.cuda.empty_cache()

    return all_results


def save_results(results: list[dict], output_path: str) -> None:
    """Write aggregated results to CSV and print formatted table."""
    if not results:
        print("No results to save.")
        return

    keys = ["Model", "Size", "BPB", "Best_Val_BPB", "Inference_Memory_MB",
            "Peak_Tok_Per_Sec", "Best_Train_Loss"]
    # Only include keys that exist in results
    present_keys = [k for k in keys if any(k in r for r in results)]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=present_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Print table
    print(f"\n{'='*80}")
    print(f"{'Model':<15} {'Size':<8} {'BPB':<10} {'Best BPB':<10} "
          f"{'Memory (MB)':<14} {'Tok/s':<12} {'Best Loss':<10}")
    print(f"{'-'*80}")
    for r in results:
        print(
            f"{r.get('Model',''):<15} {r.get('Size',''):<8} "
            f"{float(r.get('BPB', 0)):<10.4f} "
            f"{float(r.get('Best_Val_BPB', r.get('BPB', 0))):<10.4f} "
            f"{float(r.get('Inference_Memory_MB', 0)):<14.1f} "
            f"{str(r.get('Peak_Tok_Per_Sec','')):<12} "
            f"{str(r.get('Best_Train_Loss','')):<10}"
        )
    print(f"{'='*80}")
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from Spanish training runs"
    )
    parser.add_argument("--runs_dir", type=str, default="./runs/spanish",
                        help="Directory containing training run subdirectories")
    parser.add_argument("--output", type=str, default="results.csv",
                        help="Output CSV file path")
    parser.add_argument("--reeval", action="store_true",
                        help="Re-evaluate all final checkpoints instead of collecting existing CSVs")
    parser.add_argument("--cache_dir", type=str, default="./data/spanish",
                        help="Data cache directory (needed for --reeval)")
    args = parser.parse_args()

    if args.reeval:
        results = reevaluate_checkpoints(args.runs_dir, args.cache_dir)
    else:
        results = collect_existing_results(args.runs_dir)

    save_results(results, args.output)


if __name__ == "__main__":
    main()
