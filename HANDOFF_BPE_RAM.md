# BPE Tokenization RAM Problem (SOLVED)

## Symptom (original)

During `_write_bpe_corpus()` in `data_spanish.py`, RAM grew steadily while tokenizing the 8.7 GB Spanish corpus with the gpt2 tokenizer. The first 500 MB of text was processed fine, but RAM kept climbing to 38+ GB and beyond. The process was killed by OOM at ~100 GB on a 128 GB machine.

The RAM started climbing during this phase:
```
[SpanishCorpus] Wrote 8,738,577,577 bytes -> data/spanish/corpus.bin
[SpanishCorpus] Tokenizing 8,738,577,577 bytes with gpt2 ...
  [6%] 500 MB — 166,663,794 tokens
```

## Root cause

**Two independent causes**, both rooted in CPython's pymalloc holding freed arenas forever:

1. **`_write_byte_corpus()`** — the `datasets` library (even with `streaming=True`) accumulates internal state while iterating 46.9M samples. ~5 GB of RSS persists after completion.

2. **`_write_bpe_corpus()`** — each 10 MB chunk produces ~3.3M Python int objects (~92 MB) from `tokenizer.encode()`. Across 870 chunks, pymalloc accumulates ~80 GB of arenas in a single process.

## Fix

Move both phases into `multiprocessing.Process` children. Each child's address space is fully reclaimed by the OS on exit.

### Byte corpus fix (`_write_byte_corpus`)

- Added module-level `_byte_corpus_worker()` that does the `load_dataset` + write in a child process
- Parent spawns the worker, receives the total byte count via `Pipe()`, then joins
- All `datasets` library memory is freed when the child exits

### BPE tokenization fix (`_write_bpe_corpus`)

- Added module-level `_tokenize_worker()` that loads the tokenizer once and processes chunks until poison pill (`None`)
- Parent spawns the worker, feeds 10 MB chunks via `Pipe()`, receives int32 bytes back
- **Worker is recycled every 50 chunks (~500 MB text) via `CHUNKS_PER_WORKER = 50`**. This caps the child's pymalloc accumulation to ~6-8 GB instead of 80+ GB.
- 18 worker spawns total (870 chunks / 50). Each spawn loads the tokenizer from disk cache (~1-2 seconds).

### Combined effect

Both the parent's and child's RSS stay bounded:
- Parent: ~15 GB idle + minimal pipe/file buffers
- Child (peak): ~20 GB COW from fork + ~6-8 GB arena accumulation → ~28 GB max
- After each worker exits → child memory fully reclaimed

## File changes

- `data_spanish.py`: added `_tokenize_worker()` (line 30), `_byte_corpus_worker()` (line 56), rewrote `_write_byte_corpus()` (line 169), rewrote `_write_bpe_corpus()` (line 188)

## How to test

```bash
rm -f data/spanish/corpus_bpe.npy data/spanish/corpus_meta.npz
python train_spanish.py --model transformer --size 150M --max_steps 1 --batch_size 1
```

Monitor system RAM with `free -h`. Expected behavior:
- Byte corpus phase: child process accumulates memory (up to ~6 GB above idle) but exits after completion
- BPE phase: child processes are recycled every 500 MB, system RAM peaks at ~28 GB but doesn't grow monotonically toward OOM
