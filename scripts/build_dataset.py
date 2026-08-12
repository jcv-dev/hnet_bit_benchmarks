#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-time Spanish Billion Words dataset build.

Downloads ``jhonparra18/spanish_billion_words_clean`` from HuggingFace and
writes the corpus files under *cache_dir*:

- ``corpus.bin``              — raw UTF-8 bytes (byte-level models)
- ``corpus_bpe.npy``          — BPE token ids + ``corpus_meta.npz`` (transformer)

Default builds the bytes corpus only (~8.7 GB download, fast). Use ``--bpe``
on the transformer machine to also run gpt2 tokenization (~2.9B tokens,
needs ~30-40 GB RAM, runs in a child process so parent RSS stays flat).

Usage:
    python scripts/build_dataset.py                 # bytes corpus only
    python scripts/build_dataset.py --bpe           # bytes + BPE tokenization
    python scripts/build_dataset.py --cache_dir /data/spanish --max_samples 10000
    python scripts/build_dataset.py --force         # rebuild even if cached
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_spanish import SpanishCorpusBuilder


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Spanish Billion Words corpus (bytes and/or BPE)."
    )
    parser.add_argument("--cache_dir", type=str, default="./data/spanish",
                        help="Where to write the corpus files (default: ./data/spanish)")
    parser.add_argument("--bpe", action="store_true",
                        help="Also run gpt2 BPE tokenization (transformer machine only)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of HF dataset samples (debugging)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if corpus files already exist")
    args = parser.parse_args()

    builder = SpanishCorpusBuilder(
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
    )

    if args.bpe:
        meta = builder.build(force=args.force)
        print(f"[build_dataset] Bytes corpus:  {meta['total_bytes']:,} bytes")
        print(f"[build_dataset] BPE corpus:    {meta['total_tokens']:,} tokens "
              f"({meta['avg_bytes_per_token']:.2f} bytes/token)")
    else:
        total = builder.build_bytes_only(force=args.force)
        print(f"[build_dataset] Bytes corpus:  {total:,} bytes")
        print("[build_dataset] Done. Use --bpe for the transformer BPE corpus.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
