#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text generation script for Simplified SLM.

Enhanced features:
- Batch generation support
- Interactive mode
- Prompt file input
- Output file writing
- Multiple sampling strategies
- Generation quality metrics

Usage:
    # Single prompt
    python generate.py --prompt "Hello world" --max_tokens 100
    
    # From checkpoint
    python generate.py --model_path ./checkpoints/model.pt --prompt "Once upon a time"
    
    # Batch from file
    python generate.py --model_path ./checkpoints/model.pt --prompt_file prompts.txt --output_file outputs.txt
    
    # Interactive mode
    python generate.py --model_path ./checkpoints/model.pt --interactive
    
    # Batch generation
    python generate.py --prompt "The story begins" --batch_size 4 --num_samples 10
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hnet_bit.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM
from hnet_bit.utils import ByteTokenizer


def load_model(
    model_path: str = None,
    config=None,
    device: str = 'cuda',
):
    """
    Load or create an HNetBit model.
    
    Args:
        model_path: Path to saved model checkpoint
        config: Model configuration (used if no checkpoint)
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'config' in checkpoint:
            cfg_dict = checkpoint['config']
            config = HNetBitConfig(**cfg_dict)
        
        config = config or HNetBitConfig()
        model = HNetBitForCausalLM(config)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        config = config or HNetBitConfig()
        print(f"Creating new HNetBit model: d_model={config.d_model}, stages={config.num_stages}")
        model = HNetBitForCausalLM(config)
    
    model = model.to(device)
    model.eval()
    return model


def generate_text(
    model,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = 'cuda',
    repetition_penalty: float = 1.0,
) -> Tuple[str, Dict]:
    """
    Generate text from a prompt.
    
    Args:
        model: The language model
        tokenizer: Byte tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        do_sample: Whether to sample (False = greedy)
        device: Device for generation
        repetition_penalty: Penalty for repeating tokens
        
    Returns:
        Tuple of (generated_text, stats_dict)
    """
    # Encode prompt
    encoded = tokenizer.encode([prompt])[0]['input_ids']
    input_ids = torch.tensor([encoded.tolist()], dtype=torch.long, device=device)
    prompt_len = input_ids.size(1)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        if do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            })
            if repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = repetition_penalty
        else:
            gen_kwargs["do_sample"] = False
        
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )
    
    gen_time = time.time() - start_time
    
    # Decode output
    output_tokens = output_ids[0].tolist()
    new_tokens = output_tokens[prompt_len:]
    
    generated_text = tokenizer.decode(output_tokens, errors='replace')
    
    # Compute stats
    stats = {
        "prompt_tokens": prompt_len,
        "generated_tokens": len(new_tokens),
        "total_tokens": len(output_tokens),
        "generation_time": gen_time,
        "tokens_per_second": len(new_tokens) / max(gen_time, 1e-6),
    }
    
    # Check UTF-8 validity
    try:
        bytes(new_tokens).decode('utf-8')
        stats["utf8_valid"] = True
    except (UnicodeDecodeError, ValueError):
        stats["utf8_valid"] = False
    
    return generated_text, stats


def batch_generate(
    model,
    tokenizer: ByteTokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = 'cuda',
    show_progress: bool = True,
) -> List[Tuple[str, Dict]]:
    """
    Generate text for multiple prompts.
    
    Args:
        model: The language model
        tokenizer: Byte tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate per prompt
        Other args: Same as generate_text
        
    Returns:
        List of (generated_text, stats) tuples
    """
    results = []
    total = len(prompts)
    
    for i, prompt in enumerate(prompts):
        if show_progress:
            print(f"\rGenerating {i + 1}/{total}...", end="", flush=True)
        
        text, stats = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            device=device,
        )
        results.append((text, stats))
    
    if show_progress:
        print()  # newline
    
    return results


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a text file (one per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def save_outputs_to_file(
    file_path: str,
    results: List[Tuple[str, Dict]],
    prompts: List[str],
    include_stats: bool = True,
) -> None:
    """Save generated outputs to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, ((text, stats), prompt) in enumerate(zip(results, prompts)):
            f.write(f"=== Sample {i + 1} ===\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated:\n{text}\n")
            if include_stats:
                f.write(f"Stats: {stats}\n")
            f.write("\n")


def interactive_mode(
    model,
    tokenizer: ByteTokenizer,
    device: str = 'cuda',
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> None:
    """
    Interactive generation mode.
    
    Commands:
        /quit or /exit - Exit interactive mode
        /temp <value> - Set temperature
        /topk <value> - Set top-k
        /topp <value> - Set top-p
        /max <value> - Set max tokens
        /greedy - Use greedy decoding
        /sample - Use sampling
    """
    print("\n" + "=" * 60)
    print("Interactive Generation Mode")
    print("=" * 60)
    print("Commands: /quit, /temp <v>, /topk <v>, /topp <v>, /max <v>, /greedy, /sample")
    print("Enter a prompt to generate text.")
    print("=" * 60 + "\n")
    
    do_sample = True
    
    while True:
        try:
            user_input = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith('/'):
            parts = user_input.split()
            cmd = parts[0].lower()
            
            if cmd in ['/quit', '/exit', '/q']:
                print("Exiting...")
                break
            elif cmd == '/temp' and len(parts) > 1:
                temperature = float(parts[1])
                print(f"Temperature set to {temperature}")
            elif cmd == '/topk' and len(parts) > 1:
                top_k = int(parts[1])
                print(f"Top-k set to {top_k}")
            elif cmd == '/topp' and len(parts) > 1:
                top_p = float(parts[1])
                print(f"Top-p set to {top_p}")
            elif cmd == '/max' and len(parts) > 1:
                max_new_tokens = int(parts[1])
                print(f"Max tokens set to {max_new_tokens}")
            elif cmd == '/greedy':
                do_sample = False
                print("Using greedy decoding")
            elif cmd == '/sample':
                do_sample = True
                print("Using sampling")
            else:
                print(f"Unknown command: {cmd}")
            continue
        
        # Generate
        text, stats = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=user_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            device=device,
        )
        
        print(f"\n{text}")
        print(f"\n[{stats['generated_tokens']} tokens, {stats['tokens_per_second']:.1f} tok/s, UTF-8: {'OK' if stats['utf8_valid'] else 'MISSING'}]")
        print()


def print_generation_summary(results: List[Tuple[str, Dict]]) -> None:
    """Print summary statistics for a batch of generations."""
    total_tokens = sum(s['generated_tokens'] for _, s in results)
    total_time = sum(s['generation_time'] for _, s in results)
    utf8_valid = sum(1 for _, s in results if s.get('utf8_valid', False))
    
    print("\n" + "=" * 50)
    print("Generation Summary")
    print("=" * 50)
    print(f"Samples generated: {len(results)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average tokens/second: {total_tokens / max(total_time, 1e-6):.1f}")
    print(f"UTF-8 valid: {utf8_valid}/{len(results)} ({100 * utf8_valid / len(results):.1f}%)")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='Generate text with Simplified SLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single prompt
    python generate.py --prompt "Once upon a time"
    
    # From checkpoint
    python generate.py --model_path checkpoints/best.pt --prompt "Hello"
    
    # Interactive mode
    python generate.py --model_path checkpoints/best.pt --interactive
    
    # Batch from file
    python generate.py --prompt_file prompts.txt --output_file outputs.txt
    
    # Multiple samples from single prompt
    python generate.py --prompt "The quick brown" --num_samples 5
        """
    )
    
    # Model options
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--model_config', type=str, default=None,
                        help='Path to model config JSON')
    
    # Input options
    parser.add_argument('--prompt', type=str, default=None,
                        help='Input prompt')
    parser.add_argument('--prompt_file', type=str, default=None,
                        help='File with prompts (one per line)')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive generation mode')
    
    # Output options
    parser.add_argument('--output_file', type=str, default=None,
                        help='Save outputs to file')
    
    # Generation options
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling threshold')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='Repetition penalty (1.0 = no penalty)')
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy decoding')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate per prompt')
    
    # Hardware options
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load config
    if args.model_config:
        with open(args.model_config, 'r') as f:
            cfg_dict = json.load(f)
        config = HNetBitConfig(**cfg_dict)
    else:
        config = HNetBitConfig()
    
    # Load model
    model = load_model(args.model_path, config, args.device)
    tokenizer = ByteTokenizer()
    
    print(f"\nModel: {model.count_parameters():,} parameters")
    
    # Interactive mode
    if args.interactive:
        interactive_mode(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        return
    
    # Determine prompts
    prompts = []
    if args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    elif args.prompt:
        prompts = [args.prompt] * args.num_samples
    else:
        prompts = ["Hello"] * args.num_samples
    
    print(f"Generating {len(prompts)} sample(s)...")
    print("-" * 50)
    
    # Generate
    results = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        device=args.device,
    )
    
    # Print results
    for i, (text, stats) in enumerate(results):
        if len(prompts) > 1:
            print(f"\n=== Sample {i + 1} ===")
        print(f"Prompt: {prompts[i][:50]}{'...' if len(prompts[i]) > 50 else ''}")
        print(f"Generated:\n{text}")
        print(f"[{stats['generated_tokens']} tokens, {stats['tokens_per_second']:.1f} tok/s]")
    
    # Save to file if requested
    if args.output_file:
        save_outputs_to_file(args.output_file, results, prompts)
        print(f"\nOutputs saved to {args.output_file}")
    
    # Print summary for batch
    if len(results) > 1:
        print_generation_summary(results)


if __name__ == "__main__":
    main()
