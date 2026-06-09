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

CPython's pymalloc allocator holds freed memory in internal arenas and does not return it to the OS during the lifetime of the process. Each 10 MB text chunk produces ~3.3M Python int objects (~92 MB heap) via `tokenizer.encode()`. Even though Python frees these objects, the kernel still sees RSS climb monotonically across hundreds of iterations because pymalloc keeps the arenas.

## Fix

Move tokenization into a `multiprocessing.Process` child. The child's address space is fully reclaimed by the OS on exit, zeroing out all accumulated Python heap.

- Added module-level `_tokenize_worker()` function (pickle-safe, Linux fork-safe)
- Rewrote `_write_bpe_corpus()` to spawn the worker, feed 10 MB chunks via `multiprocessing.Pipe`, and receive raw int32 bytes back
- No new dependencies (uses stdlib `multiprocessing`)
- Worker loads the tokenizer once, processes all chunks, then exits via poison pill (`None`)

The parent process now only ever holds ~10 MB of raw input bytes + ~13 MB of token output bytes, keeping its RSS flat.

## File changes

- `data_spanish.py`: added `_tokenize_worker()` function (line 30), rewrote `_write_bpe_corpus()` (line 165)

## How to test

```bash
rm -f data/spanish/corpus_bpe.npy data/spanish/corpus_meta.npz
python train_spanish.py --model transformer --size 150M --max_steps 1 --batch_size 1
```

Monitor RAM with `htop` — parent process RSS should stay flat (5-10 GB for tokenizer model, doesn't grow).
