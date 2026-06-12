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
from export_deployment import export_model


def _safe_float(val, default=""):
    try:
        return round(float(val), 4)
    except (ValueError, TypeError):
        return default


def _get_deploy_stats(checkpoint_path: str) -> dict:
    """
    Auto-export a checkpoint to compact ternary weights if not already done.

    Returns a dict with keys Deploy_Size_MB and Bits_Per_Param, or empty
    values on failure.
    """
    ckpt_path = Path(checkpoint_path)
    export_path = ckpt_path.parent / "model_deploy.pt"

    if export_path.exists():
        size_mb = export_path.stat().st_size / (1024 ** 2)
        bpp = ""
        try:
            data = torch.load(export_path, map_location="cpu", weights_only=False)
            exp_stats = data.get("export_stats", {})
            bpp = exp_stats.get("bits_per_param", "")
        except Exception:
            pass
        return {"Deploy_Size_MB": round(size_mb, 1), "Bits_Per_Param": bpp}

    try:
        stats = export_model(str(ckpt_path), str(export_path), verify=False)
        return {
            "Deploy_Size_MB": stats.get("export_size_mb", ""),
            "Bits_Per_Param": stats.get("bits_per_param", ""),
        }
    except Exception as e:
        print(f"  Export skipped for {ckpt_path.name}: {e}")
        return {"Deploy_Size_MB": "", "Bits_Per_Param": ""}


def collect_existing_results(runs_dir: str) -> list[dict]:
    """
    Scan *runs_dir* for per-run result files and merge ALL available info
    into one row per model x size.
    """
    runs = Path(runs_dir)
    all_results = []

    for csv_file in sorted(runs.rglob("results_*.csv")):
        run_dir = csv_file.parent
        # Derive model-specific run subdirectory from CSV filename (e.g. hybrid_tiny)
        parts = csv_file.stem.replace("results_", "", 1).rsplit("_", 1)
        model_run_subdir = "_".join(parts) if len(parts) == 2 else None
        model_run_dir = (run_dir / model_run_subdir) if model_run_subdir else run_dir
        if not model_run_dir.exists():
            model_run_dir = run_dir  # fallback to parent

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for base_row in reader:
                row = dict(base_row)
                row["BPB_Final"] = _safe_float(row.pop("BPB", ""))
                row["Inference_Memory_MB"] = _safe_float(row.get("Inference_Memory_MB", ""))

                # ── training_steps_log.csv ──────────────────────────────
                row.update({
                    "Train_Steps": "", "Best_Train_Loss": "",
                    "Final_Train_Loss": "", "Peak_Tok_Per_Sec": "",
                    "Avg_Tok_Per_Sec": "",
                })
                steps_log = model_run_dir / "training_steps_log.csv"
                if steps_log.exists():
                    try:
                        with open(steps_log) as sf:
                            steps = list(csv.DictReader(sf))
                        if steps:
                            losses = [float(s["loss"]) for s in steps]
                            toks = [float(s["tok_per_sec"]) for s in steps]
                            row["Train_Steps"] = steps[-1]["step"]
                            row["Best_Train_Loss"] = round(min(losses), 4)
                            row["Final_Train_Loss"] = round(losses[-1], 4)
                            row["Peak_Tok_Per_Sec"] = round(max(toks), 1)
                            row["Avg_Tok_Per_Sec"] = round(sum(toks) / len(toks), 1)
                    except Exception:
                        pass

                # ── validation_log.csv ──────────────────────────────────
                row.update({
                    "Best_Val_BPB": row["BPB_Final"],
                    "Final_Val_BPB": row["BPB_Final"],
                    "Best_Val_Loss": "", "Final_Val_Loss": "",
                    "Val_BPB_at_25B": "", "Val_BPB_at_50B": "",
                    "Val_BPB_at_100B": "",
                })
                val_log = model_run_dir / "validation_log.csv"
                if val_log.exists():
                    try:
                        with open(val_log) as vf:
                            evals = list(csv.DictReader(vf))
                        if evals:
                            bpbs = [float(e["val_bpb"]) for e in evals]
                            losses = [float(e["val_loss"]) for e in evals]
                            bytes_seen = [int(e.get("bytes_seen", e.get("tokens_seen", 0))) for e in evals]
                            row["Best_Val_BPB"] = round(min(bpbs), 4)
                            row["Final_Val_BPB"] = round(bpbs[-1], 4)
                            row["Best_Val_Loss"] = round(min(losses), 4)
                            row["Final_Val_Loss"] = round(losses[-1], 4)
                            for label, target in [
                                ("Val_BPB_at_25B",  25_000_000_000),
                                ("Val_BPB_at_50B",  50_000_000_000),
                                ("Val_BPB_at_100B", 100_000_000_000),
                            ]:
                                dists = [abs(t - target) for t in bytes_seen]
                                idx = dists.index(min(dists))
                                if min(dists) < 5_000_000_000:
                                    row[label] = round(bpbs[idx], 4)
                    except Exception:
                        pass

                # ── config.json ─────────────────────────────────────────
                row.update({
                    "LR": "", "Batch_Size": "", "Grad_Accum": "",
                    "Total_Bytes": "", "Seq_Length": "",
                })
                config_file = model_run_dir / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file) as cf:
                            cfg = json.load(cf)
                        row["LR"] = cfg.get("learning_rate", "")
                        row["Batch_Size"] = cfg.get("batch_size", "")
                        row["Grad_Accum"] = cfg.get("gradient_accumulation_steps", "")
                        row["Total_Bytes"] = cfg.get("total_training_bytes", "")
                        seq = (cfg.get("token_seq_length")
                               if cfg.get("model_name") == "transformer"
                               else cfg.get("byte_seq_length"))
                        row["Seq_Length"] = seq or ""
                    except Exception:
                        pass

                # ── training_stats.json ──────────────────────────────────
                stats_file = model_run_dir / "training_stats.json"
                if stats_file.exists():
                    try:
                        with open(stats_file) as sf:
                            stats = json.load(sf)
                        row["Training_Time_Hours"] = round(stats.get("training_time_hours", 0), 2)
                        row["Training_Time_Seconds"] = round(stats.get("training_time_seconds", 0), 0)
                        row["Param_Count_M"] = round(stats.get("param_count", 0) / 1_000_000, 1)
                        row["Non_Emb_Params_M"] = round(stats.get("non_embedding_param_count", 0) / 1_000_000, 1)
                        row["Disk_Size_MB"] = round(stats.get("disk_size_mb", 0), 1)
                        row["Peak_Training_Memory_MB"] = round(stats.get("peak_training_memory_mb", 0), 0)
                        row["Peak_Reserved_Memory_MB"] = round(stats.get("peak_reserved_memory_mb", 0), 0)
                        compression = stats.get("overall_compression_ratio")
                        if compression is not None:
                            row["Overall_Compression_Ratio"] = round(compression, 4)
                    except Exception:
                        pass

                # ── deployment export ───────────────────────────────────
                row.update({"Deploy_Size_MB": "", "Bits_Per_Param": ""})
                ckpt_path = model_run_dir / "checkpoint_final.pt"
                ckpt_path_best = model_run_dir / "checkpoint_best.pt"
                source_ckpt = ckpt_path if ckpt_path.exists() else ckpt_path_best
                if source_ckpt.exists():
                    deploy = _get_deploy_stats(str(source_ckpt))
                    row.update(deploy)

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

        # Read training stats
        stats_file = ckpt_path.parent / "training_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file) as sf:
                    stats = json.load(sf)
                result["Training_Time_Hours"] = round(stats.get("training_time_hours", 0), 2)
                result["Param_Count_M"] = round(stats.get("param_count", 0) / 1_000_000, 1)
                result["Non_Emb_Params_M"] = round(stats.get("non_embedding_param_count", 0) / 1_000_000, 1)
                result["Disk_Size_MB"] = round(stats.get("disk_size_mb", 0), 1)
                result["Peak_Training_Memory_MB"] = round(stats.get("peak_training_memory_mb", 0), 0)
                compression = stats.get("overall_compression_ratio")
                if compression is not None:
                    result["Overall_Compression_Ratio"] = round(compression, 4)
            except Exception:
                pass

        # Deployment export
        deploy = _get_deploy_stats(str(ckpt_path))
        result.update({k: v for k, v in deploy.items() if k not in result or not result.get(k)})

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

    # Ordered columns — all possible fields
    all_keys = [
        "Model", "Size",
        # Core metrics
        "BPB_Final", "Best_Val_BPB", "Final_Val_BPB",
        "Best_Val_Loss", "Final_Val_Loss",
        "Val_BPB_at_25B", "Val_BPB_at_50B", "Val_BPB_at_100B",
        "Inference_Memory_MB",
        # Model stats
        "Param_Count_M", "Non_Emb_Params_M", "Disk_Size_MB",
        "Deploy_Size_MB", "Bits_Per_Param",
        "Peak_Training_Memory_MB", "Peak_Reserved_Memory_MB",
        "Overall_Compression_Ratio",
        # Training stats
        "Best_Train_Loss", "Final_Train_Loss", "Train_Steps",
        "Peak_Tok_Per_Sec", "Avg_Tok_Per_Sec",
        "Training_Time_Hours",
        # Hyperparameters
        "LR", "Batch_Size", "Grad_Accum", "Total_Bytes", "Seq_Length",
    ]
    # Only write columns that have at least one non-empty value
    present_keys = [
        k for k in all_keys
        if any(str(r.get(k, "")) not in ("", "None") for r in results)
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=present_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Print concise summary table
    print(f"\n{'='*170}")
    print(f"{'Model':<15} {'Size':<8} {'Params M':<9} {'NonEmb M':<9} {'BPB':<8} {'Loss':<8} "
          f"{'Train Mem':<10} {'Disk MB':<8} {'Deploy MB':<10} {'Bits/par':<9} {'Compress':<9} {'Tok/s':<8} {'Hours':<8}")
    print(f"{'-'*170}")
    for r in results:
        print(
            f"{str(r.get('Model','')):<15} {str(r.get('Size','')):<8} "
            f"{str(r.get('Param_Count_M', '')):<9} "
            f"{str(r.get('Non_Emb_Params_M', '')):<9} "
            f"{str(r.get('Best_Val_BPB', r.get('BPB_Final', ''))):<8} "
            f"{str(r.get('Best_Val_Loss', '')):<8} "
            f"{str(r.get('Peak_Training_Memory_MB','')):<10} "
            f"{str(r.get('Disk_Size_MB','')):<8} "
            f"{str(r.get('Deploy_Size_MB','')):<10} "
            f"{str(r.get('Bits_Per_Param','')):<9} "
            f"{str(r.get('Overall_Compression_Ratio','')):<9} "
            f"{str(r.get('Peak_Tok_Per_Sec','')):<8} "
            f"{str(r.get('Training_Time_Hours','')):<8}"
        )
    print(f"{'='*170}")
    print(f"\nFull results saved to: {output_path}")
    print(f"Columns: {', '.join(present_keys)}")


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
