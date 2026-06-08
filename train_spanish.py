#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified training script for benchmarking three architectures on Spanish
Billion Words.

Usage:
    python train_spanish.py --model hybrid     --size 350M
    python train_spanish.py --model transformer --size 150M
    python train_spanish.py --model matmulfree  --size 750M

    # Quick smoke test (100 steps, tiny data)
    python train_spanish.py --model hybrid --size 150M --max_steps 100 --batch_size 2

    # Override any config value
    python train_spanish.py --model hybrid --size 350M --lr 1e-3 --bf16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Make sure local modules are on the path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_factory import build_model
from data_spanish import SpanishCorpusBuilder, create_dataloaders
from training_config_spanish import (
    SpanishTrainingConfig,
    build_optimizer_and_scheduler,
    enable_gradient_checkpointing,
)
from metrics_spanish import compute_bpb, measure_inference_memory


# ============================================================================
# Trainer
# ============================================================================

class SpanishTrainer:
    """
    Training loop for the Spanish Billion Words benchmark.

    Handles gradient accumulation, mixed precision, periodic BPB evaluation,
    and milestone checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SpanishTrainingConfig,
        train_loader,
        val_loader,
        is_byte_level: bool,
        vocab_size: int,
        avg_bytes_per_token: float = 1.0,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_byte_level = is_byte_level
        self.vocab_size = vocab_size
        self.avg_bytes_per_token = avg_bytes_per_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Gradient checkpointing
        if config.gradient_checkpointing:
            enable_gradient_checkpointing(self.model)

        # Optimizer & scheduler
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(model, config)

        # Mixed precision
        self.use_amp = config.bf16 or config.fp16
        self.amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.fp16)

        # State
        self.global_step = 0
        self.tokens_seen = 0
        self.best_val_bpb = float("inf")
        self.training_start_time = None

        # Parameter count
        self.param_count = sum(p.numel() for p in self.model.parameters())
        self.trainable_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Reset CUDA peak memory stats to measure training peak later
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Chunking stats accumulator (hybrid only)
        self._chunking_accum: list[dict] = []

        # Results log
        self.results_log: list[dict] = []
        self.train_log: list[dict] = []

        # Milestone tracking
        self._milestone_set = set(config.checkpoint_milestones)
        self._passed_milestones: set[int] = set()

        # Output dirs
        self.output_dir = Path(config.output_dir) / f"{config.model_name}_{config.model_size}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load balancing
    # ------------------------------------------------------------------

    def compute_load_balancing_loss(self, router_outputs) -> torch.Tensor:
        """Compute load balancing loss for hierarchical models."""
        if not router_outputs:
            return 0.0
        
        total_lb_loss = 0.0
        N = getattr(self.config, 'downsampling_factor', 5.0)
        
        for router_output in router_outputs:
            if router_output is None:
                continue
                
            boundary_prob = router_output.boundary_prob
            tokenized_prob = boundary_prob[..., -1]
            boundary_mask = router_output.boundary_mask
            
            true_ratio = boundary_mask.float().mean()
            average_prob = tokenized_prob.float().mean()
            
            stage_lb_loss = (
                (1 - true_ratio) * (1 - average_prob) +
                (true_ratio) * (average_prob) * (N - 1)
            ) * N / (N - 1)
            
            total_lb_loss += stage_lb_loss
            
        return total_lb_loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        self.model.train()
        cfg = self.config
        max_steps = cfg.max_steps_override if cfg.max_steps_override is not None else cfg.total_steps

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump({k: v for k, v in cfg.__dict__.items()
                       if not k.startswith("_") and not callable(v)}, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"Training {cfg.model_name} {cfg.model_size}")
        print(f"  Max steps      : {max_steps:,}")
        print(f"  Bytes/step     : {cfg.bytes_per_step:,}")
        print(f"  Total bytes    : {cfg.total_training_bytes:,}")
        print(f"  Batch size     : {cfg.batch_size} × {cfg.gradient_accumulation_steps} = {cfg.effective_batch_size}")
        print(f"  LR             : {cfg.resolve_learning_rate():.2e}")
        print(f"  Output         : {self.output_dir}")
        print(f"{'='*60}\n")

        accum_loss = 0.0
        micro_step = 0
        data_iter = iter(self.train_loader)
        t_start = time.time()
        self.training_start_time = t_start

        while self.global_step < max_steps:
            # Get batch (cycle)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch.get("attention_mask"),
                    output_hidden_states=True if cfg.model_name in ("hybrid", "hybrid_attn") else False,
                )
                
                ce_loss = outputs.loss
                
                if cfg.model_name in ("hybrid", "hybrid_attn") and (router_outputs := getattr(outputs, "router_outputs", None)):
                    lb_loss = self.compute_load_balancing_loss(router_outputs)
                    total_loss = ce_loss + cfg.lambda_lb * lb_loss
                    # Accumulate chunking stats
                    chunk_entry = {}
                    for s, r in enumerate(router_outputs):
                        if r is not None and r.boundary_mask is not None:
                            chunk_entry[f"stage_{s}_compression_ratio"] = r.boundary_mask.float().mean().item()
                    self._chunking_accum.append(chunk_entry)
                else:
                    total_loss = ce_loss

                loss = total_loss / cfg.gradient_accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()
            accum_loss += loss.item()
            micro_step += 1

            # Optimizer step
            if micro_step % cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

                self.global_step += 1
                self.tokens_seen += cfg.bytes_per_step

                # Log
                if self.global_step % cfg.log_interval_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - t_start
                    tok_per_sec = self.tokens_seen / max(elapsed, 1)
                    
                    train_entry = {
                        "step": self.global_step,
                        "bytes_seen": self.tokens_seen,
                        "loss": accum_loss,
                        "lr": lr,
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "tok_per_sec": tok_per_sec
                    }
                    # Average and log chunking stats for hybrid
                    if self._chunking_accum:
                        ratios = []
                        merged = {}
                        for e in self._chunking_accum:
                            for k, v in e.items():
                                merged.setdefault(k, []).append(v)
                        for k, vals in merged.items():
                            avg = sum(vals) / len(vals)
                            train_entry[k] = avg
                            ratios.append(avg)
                        if ratios:
                            overall = 1.0
                            for r in ratios:
                                overall *= r
                            train_entry["overall_compression_ratio"] = overall
                        self._chunking_accum = []
                    self.train_log.append(train_entry)
                    
                    # Save train log periodically
                    if self.global_step % (cfg.log_interval_steps * 10) == 0:
                        self._save_train_log_csv()

                    print(f"step={self.global_step:>8,}  loss={accum_loss:.4f}  "
                          f"lr={lr:.2e}  grad_norm={grad_norm:.3f}  "
                          f"bytes={self.tokens_seen:>14,}  tok/s={tok_per_sec:,.0f}")

                accum_loss = 0.0

                # Periodic BPB evaluation
                if self.global_step % cfg.eval_interval_steps == 0:
                    self._evaluate_and_log()
                    self.model.train()

                # Save checkpoint
                if self.global_step % cfg.save_interval_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

                # Milestone checkpoints
                for milestone in list(self._milestone_set - self._passed_milestones):
                    if self.tokens_seen >= milestone:
                        self._passed_milestones.add(milestone)
                        ms_label = f"{milestone // 1_000_000_000}B"
                        print(f"\n*** Milestone: {ms_label} bytes reached ***")
                        self._evaluate_and_log()
                        self._save_checkpoint(f"milestone_{ms_label}")
                        self.model.train()

        # Final evaluation & save
        self._evaluate_and_log()

        # Training time & hardware stats
        training_elapsed = time.time() - self.training_start_time
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        disk_size_mb = sum(p.numel() for p in self.model.parameters()) * 2 / (1024 ** 2)  # bf16

        # Average compression ratio from training log (hybrid only)
        ratios = [e.get("overall_compression_ratio")
                  for e in self.train_log if "overall_compression_ratio" in e]
        avg_compression = sum(ratios) / len(ratios) if ratios else None

        stats = {
            "training_time_seconds": training_elapsed,
            "training_time_hours": training_elapsed / 3600,
            "total_steps": self.global_step,
            "bytes_seen": self.tokens_seen,
            "param_count": self.param_count,
            "trainable_param_count": self.trainable_param_count,
            "disk_size_mb": round(disk_size_mb, 1),
            "peak_training_memory_mb": round(peak_memory_mb, 0),
            "peak_reserved_memory_mb": round(peak_reserved_mb, 0),
        }
        if avg_compression is not None:
            stats["overall_compression_ratio"] = avg_compression
        with open(self.output_dir / "training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Training time  : {training_elapsed / 3600:.2f} hours ({training_elapsed:.0f} seconds)")
        print(f"  Peak GPU mem   : {peak_memory_mb:.0f} MB allocated / {peak_reserved_mb:.0f} MB reserved")
        print(f"  Model params   : {self.param_count:,} ({self.param_count / 1e6:.1f}M)  disk ~{disk_size_mb:.0f} MB")

        self._save_checkpoint("final")
        self._save_results_csv()
        self._save_train_log_csv()

        print(f"\nTraining complete. Best val BPB: {self.best_val_bpb:.4f}")
        print(f"Results saved to {self.output_dir / 'results.csv'}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_and_log(self) -> None:
        """Compute validation BPB and log it."""
        bpb_results = compute_bpb(
            self.model, self.val_loader, self.is_byte_level,
            avg_bytes_per_token=self.avg_bytes_per_token,
            device=str(self.device),
            max_batches=50,  # cap to keep eval fast
        )

        bpb = bpb_results["bpb"]
        val_loss = bpb_results["loss"]

        if bpb < self.best_val_bpb:
            self.best_val_bpb = bpb
            self._save_checkpoint("best")

        entry = {
            "step": self.global_step,
            "bytes_seen": self.tokens_seen,
            "val_loss": val_loss,
            "val_bpb": bpb,
        }
        self.results_log.append(entry)

        print(f"  [eval] step={self.global_step:,}  val_loss={val_loss:.4f}  "
              f"val_bpb={bpb:.4f}  best_bpb={self.best_val_bpb:.4f}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, name: str) -> None:
        path = self.output_dir / f"checkpoint_{name}.pt"
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_val_bpb": self.best_val_bpb,
            "config": {k: v for k, v in self.config.__dict__.items()
                       if not k.startswith("_") and not callable(v)},
        }
        torch.save(state, path)
        print(f"  Saved checkpoint: {path}")

    # ------------------------------------------------------------------
    # Results output
    # ------------------------------------------------------------------

    def _save_train_log_csv(self) -> None:
        """Save step-by-step training log as CSV."""
        csv_path = self.output_dir / "training_steps_log.csv"
        if not self.train_log:
            return
        keys = self.train_log[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.train_log)

    def _save_results_csv(self) -> None:
        """Save validation evaluation log as CSV."""
        csv_path = self.output_dir / "validation_log.csv"
        if not self.results_log:
            return
        keys = self.results_log[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results_log)
        print(f"  Training log saved: {csv_path}")


# ============================================================================
# Post-training: final results table
# ============================================================================

def generate_final_results(
    model: nn.Module,
    val_loader,
    config: SpanishTrainingConfig,
    is_byte_level: bool,
    vocab_size: int,
    avg_bytes_per_token: float,
    device: str = "cuda",
    output_dir: str = None,
) -> dict:
    """
    Generate the final results dict for one model × size combination.

    Returns a dict with:
        Model, Size, BPB, Inference_Memory_MB, Training_Time_Hours
    """
    print(f"\n--- Final Evaluation: {config.model_name} {config.model_size} ---")

    # BPB
    bpb_results = compute_bpb(
        model, val_loader, is_byte_level,
        avg_bytes_per_token=avg_bytes_per_token,
        device=device,
    )

    # Inference memory
    mem_results = measure_inference_memory(
        model,
        seq_length=config.inference_seq_length,
        is_byte_level=is_byte_level,
        vocab_size=vocab_size,
        device=device,
    )

    val_loss = bpb_results["loss"]
    val_perplexity = math.exp(min(val_loss, 20))

    result = {
        "Model": config.model_name,
        "Size": config.model_size,
        "BPB": round(bpb_results["bpb"], 4),
        "Val_Loss": round(val_loss, 4),
        "Val_Perplexity": round(val_perplexity, 2),
        "Inference_Memory_MB": round(mem_results["peak_memory_mb"], 1),
    }

    # Read training-time stats from training_stats.json if available
    if output_dir is not None:
        stats_path = Path(output_dir) / "training_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            result["Training_Time_Hours"] = round(stats.get("training_time_hours", 0), 2)
            result["Training_Time_Seconds"] = round(stats.get("training_time_seconds", 0), 0)
            result["Param_Count_M"] = round(stats.get("param_count", 0) / 1_000_000, 1)
            result["Param_Count"] = stats.get("param_count", 0)
            result["Disk_Size_MB"] = round(stats.get("disk_size_mb", 0), 1)
            result["Peak_Training_Memory_MB"] = round(stats.get("peak_training_memory_mb", 0), 0)
            result["Peak_Reserved_Memory_MB"] = round(stats.get("peak_reserved_memory_mb", 0), 0)
            compression = stats.get("overall_compression_ratio")
            if compression is not None:
                result["Overall_Compression_Ratio"] = round(compression, 4)
        else:
            result["Training_Time_Hours"] = ""
            # Compute param count from model directly if no stats file
            p = sum(p.numel() for p in model.parameters())
            result["Param_Count_M"] = round(p / 1_000_000, 1)

    print(f"  BPB               : {result['BPB']:.4f}")
    print(f"  Val Loss          : {result['Val_Loss']:.4f}")
    print(f"  Perplexity        : {result['Val_Perplexity']:.2f}")
    print(f"  Params            : {result.get('Param_Count_M', '?')}M")
    if "Peak_Training_Memory_MB" in result and result["Peak_Training_Memory_MB"]:
        print(f"  Peak Train Mem   : {result['Peak_Training_Memory_MB']:.0f} MB")
    print(f"  Inference Memory  : {result['Inference_Memory_MB']:.1f} MB")
    if result.get("Training_Time_Hours"):
        print(f"  Training Time     : {result['Training_Time_Hours']:.2f} hours")

    return result


def save_results_csv(results: list[dict], output_path: str) -> None:
    """Write the final results table to CSV."""
    if not results:
        return
    keys = results[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and benchmark language models on Spanish Billion Words"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["transformer", "matmulfree", "hybrid", "hybrid_attn"],
                        help="Model architecture")
    parser.add_argument("--size", type=str, required=True,
                        choices=["tiny", "150M", "350M", "750M"],
                        help="Model size tier")

    # Overrides
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps (for debugging)")
    parser.add_argument("--total_tokens", type=int, default=None,
                        help="Total tokens to train on")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--no_bf16", action="store_true", default=False,
                        help="Disable bf16 (use fp32)")
    parser.add_argument("--output_dir", type=str, default="./runs/spanish")
    parser.add_argument("--cache_dir", type=str, default="./data/spanish")
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of HF dataset samples (for debugging)")
    parser.add_argument("--skip_data_build", action="store_true",
                        help="Skip data downloading/preprocessing")

    # Tokenizer (overrides default for transformer)
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Tokenizer name for transformer model (default: gpt2)")

    # Eval only
    parser.add_argument("--eval_only", action="store_true",
                        help="Run only evaluation on a checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for eval_only mode")

    return parser.parse_args()


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Build config
    config = SpanishTrainingConfig(
        model_name=args.model,
        model_size=args.size,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )

    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.grad_accum is not None:
        config.gradient_accumulation_steps = args.grad_accum
    if args.total_tokens is not None:
        config.total_training_bytes = args.total_tokens
    if args.no_bf16:
        config.bf16 = False
    if args.fp16:
        config.fp16 = True
        config.bf16 = False

    # Handle --max_steps override (for debugging)
    if args.max_steps is not None:
        config.max_steps_override = args.max_steps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        config.bf16 = False
        config.fp16 = False

    # ------------------------------------------------------------------
    # 1. Build data corpus
    # ------------------------------------------------------------------
    if not args.skip_data_build:
        builder = SpanishCorpusBuilder(
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
        )
        if args.model == "transformer":
            builder.build()
        else:
            builder.build_bytes_only()

    # Compute avg_bytes_per_token for BPE models and set on config
    avg_bytes_per_token = 1.0
    if args.model == "transformer":
        meta_path = Path(args.cache_dir) / "corpus_meta.npz"
        if meta_path.exists():
            meta = dict(np.load(meta_path))
            avg_bytes_per_token = float(meta.get("avg_bytes_per_token", 4.5))
        else:
            avg_bytes_per_token = 4.5  # reasonable default for Llama-3 on Spanish
        print(f"[BPE] avg_bytes_per_token = {avg_bytes_per_token:.2f}")
    config.avg_bytes_per_token = avg_bytes_per_token

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    build_kwargs = {}
    if args.tokenizer_name is not None:
        build_kwargs["tokenizer_name"] = args.tokenizer_name
    model, is_byte_level, vocab_size = build_model(args.model, args.size, **build_kwargs)

    # ------------------------------------------------------------------
    # 3. Build data loaders
    # ------------------------------------------------------------------
    train_loader, val_loader = create_dataloaders(
        model_name=args.model,
        cache_dir=args.cache_dir,
        byte_seq_length=config.byte_seq_length,
        token_seq_length=config.token_seq_length,
        batch_size=config.batch_size,
    )

    # ------------------------------------------------------------------
    # 4. Eval-only mode
    # ------------------------------------------------------------------
    if args.eval_only:
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint: {args.checkpoint}")

        model.to(device)
        result = generate_final_results(
            model, val_loader, config, is_byte_level, vocab_size,
            avg_bytes_per_token, device=device,
            output_dir=str(Path(config.output_dir) / f"{args.model}_{args.size}"),
        )
        # Save single result as CSV
        out_csv = Path(config.output_dir) / f"results_{args.model}_{args.size}.csv"
        save_results_csv([result], str(out_csv))
        return

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    trainer = SpanishTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        is_byte_level=is_byte_level,
        vocab_size=vocab_size,
        avg_bytes_per_token=avg_bytes_per_token,
    )
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Final results
    # ------------------------------------------------------------------
    model.to(device)
    result = generate_final_results(
        model, val_loader, config, is_byte_level, vocab_size,
        avg_bytes_per_token, device=device,
        output_dir=str(Path(config.output_dir) / f"{args.model}_{args.size}"),
    )
    out_csv = Path(config.output_dir) / f"results_{args.model}_{args.size}.csv"
    save_results_csv([result], str(out_csv))


if __name__ == "__main__":
    main()
