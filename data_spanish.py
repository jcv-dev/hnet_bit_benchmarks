# -*- coding: utf-8 -*-

"""
Data loading for Spanish Billion Words.

Provides:
    - SpanishCorpusBuilder: downloads and preprocesses the dataset
    - ByteDataset: fixed‑length byte chunks for char/byte‑level models
    - TokenDataset: fixed‑length BPE token chunks for Transformer baseline
    - AlignedBatchSampler: ensures all models see the same underlying text
"""

from __future__ import annotations

import mmap
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


# --------------------------------------------------------------------------
# BPE tokenization worker (runs in subprocess to avoid memory accumulation)
# --------------------------------------------------------------------------

def _tokenize_worker(conn, tokenizer_name):
    """Child subprocess: loads tokenizer once, processes chunks until poison pill.

    Runs in a separate process so Python heap memory from tokenization is
    returned to the OS when the process exits.
    """
    from transformers import AutoTokenizer
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    while True:
        chunk = conn.recv()
        if chunk is None:
            break

        text = chunk.decode("utf-8", errors="replace")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        result = np.array(tokens, dtype=np.int32).tobytes() if tokens else b""
        conn.send(result)


# --------------------------------------------------------------------------
# 1. Corpus builder: download, concatenate, write binary files
# --------------------------------------------------------------------------

class SpanishCorpusBuilder:
    """
    Downloads the ``crscardellino/spanish_billion_words`` dataset from
    HuggingFace and writes two persistent files:

    - ``corpus.bin``          — raw UTF‑8 bytes (flat binary)
    - ``corpus_bpe.npy``      — BPE token ids (int32 numpy memmap)
    - ``byte2token_offsets.npy`` — mapping from byte offsets to token offsets

    All files are written to *cache_dir*.
    """

    def __init__(
        self,
        cache_dir: str = "./data/spanish",
        hf_dataset: str = "jhonparra18/spanish_billion_words_clean",
        tokenizer_name: str = "gpt2",
        max_samples: Optional[int] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.hf_dataset = hf_dataset
        self.tokenizer_name = tokenizer_name
        self.max_samples = max_samples

        self.byte_path = self.cache_dir / "corpus.bin"
        self.bpe_path = self.cache_dir / "corpus_bpe.npy"
        self.offset_path = self.cache_dir / "byte2token_offsets.npy"
        self.meta_path = self.cache_dir / "corpus_meta.npz"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, force: bool = False) -> dict:
        """
        Build all corpus files.  Skips if they already exist unless *force*.

        Returns dict with keys: total_bytes, total_tokens, avg_bytes_per_token.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not force and self.byte_path.exists() and self.meta_path.exists():
            meta = dict(np.load(self.meta_path))
            print(f"[SpanishCorpus] Using cached corpus — "
                  f"{int(meta['total_bytes']):,} bytes, {int(meta['total_tokens']):,} tokens")
            return {k: int(v) if k != "avg_bytes_per_token" else float(v)
                    for k, v in meta.items()}

        return self._build_from_scratch()

    def build_bytes_only(self, force: bool = False) -> int:
        """Build only the bytes corpus (for byte‑level models). Returns total bytes."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not force and self.byte_path.exists():
            size = self.byte_path.stat().st_size
            print(f"[SpanishCorpus] Byte corpus exists — {size:,} bytes")
            return size
        return self._write_byte_corpus()

    def build_bpe(self, force: bool = False) -> Tuple[int, float]:
        """Build the BPE corpus (requires bytes corpus).  Returns (total_tokens, avg_bytes_per_token)."""
        assert self.byte_path.exists(), "Call build_bytes_only() first"
        if not force and self.bpe_path.exists():
            tokens = np.load(self.bpe_path, mmap_mode="r")
            total_bytes = self.byte_path.stat().st_size
            avg = total_bytes / max(len(tokens), 1)
            print(f"[SpanishCorpus] BPE corpus exists — {len(tokens):,} tokens, "
                  f"avg {avg:.2f} bytes/token")
            return len(tokens), avg
        return self._write_bpe_corpus()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_from_scratch(self) -> dict:
        total_bytes = self._write_byte_corpus()
        total_tokens, avg = self._write_bpe_corpus()
        meta = dict(total_bytes=total_bytes, total_tokens=total_tokens,
                    avg_bytes_per_token=avg)
        np.savez(self.meta_path, **meta)
        return meta

    def _write_byte_corpus(self) -> int:
        from datasets import load_dataset

        print(f"[SpanishCorpus] Downloading {self.hf_dataset} ...")
        ds = load_dataset(self.hf_dataset, split="train", streaming=True, trust_remote_code=True)

        total = 0
        with open(self.byte_path, "wb") as f:
            for i, example in enumerate(ds):
                if self.max_samples and i >= self.max_samples:
                    break
                text = example.get("text", "")
                if not text:
                    continue
                raw = text.encode("utf-8")
                f.write(raw)
                total += len(raw)
                if (i + 1) % 100_000 == 0:
                    print(f"  {i+1:>10,} samples — {total:>15,} bytes")

        print(f"[SpanishCorpus] Wrote {total:,} bytes → {self.byte_path}")
        return total

    def _write_bpe_corpus(self) -> Tuple[int, float]:
        from multiprocessing import Process, Pipe
        import struct

        total_bytes = self.byte_path.stat().st_size

        CHUNK = 10 * 1024 * 1024  # 10 MB
        raw_path = self.cache_dir / "corpus_bpe.raw"
        total_tokens = 0

        print(f"[SpanishCorpus] Tokenizing {total_bytes:,} bytes with {self.tokenizer_name} ...")

        # Spawn worker subprocess — its entire Python heap is reclaimed on exit
        parent_conn, child_conn = Pipe()
        worker = Process(target=_tokenize_worker, args=(child_conn, self.tokenizer_name))
        worker.start()
        child_conn.close()

        try:
            with open(raw_path, "wb") as out:
                byte_offset = 0
                with open(self.byte_path, "rb") as f:
                    while True:
                        raw = f.read(CHUNK)
                        if not raw:
                            break

                        parent_conn.send(raw)
                        result = parent_conn.recv()
                        if result:
                            out.write(result)
                            total_tokens += len(result) // 4

                        byte_offset += len(raw)

                        if byte_offset % (500 * 1024 * 1024) == 0:
                            pct = 100.0 * byte_offset / total_bytes
                            print(f"  [{pct:.0f}%] {byte_offset // (1024*1024):,} MB — "
                                  f"{total_tokens:,} tokens")
                            out.flush()
        finally:
            try:
                parent_conn.send(None)
            except (BrokenPipeError, EOFError):
                pass
            worker.join(timeout=30)
            if worker.is_alive():
                worker.kill()
                worker.join()
            parent_conn.close()

        # Build .npy header and prepend to the raw file
        print(f"[SpanishCorpus] Writing {total_tokens:,} tokens → {self.bpe_path} ...")
        header_dict = {"descr": "<i4", "fortran_order": False, "shape": (total_tokens,)}
        import io
        buf = io.BytesIO()
        np.lib.format._write_array_header(buf, header_dict)
        header_data = buf.getvalue()
        with open(self.bpe_path, "wb") as npy_file:
            npy_file.write(b"\x93NUMPY\x01\x00")
            npy_file.write(struct.pack("<H", len(header_data)))
            npy_file.write(header_data)
            with open(raw_path, "rb") as raw_f:
                while True:
                    buf = raw_f.read(64 * 1024 * 1024)
                    if not buf:
                        break
                    npy_file.write(buf)

        os.unlink(raw_path)

        avg = total_bytes / max(total_tokens, 1)
        total_bytes_on_disk = total_tokens * 4
        print(f"[SpanishCorpus] Wrote {total_tokens:,} tokens → {self.bpe_path}")
        print(f"  Disk: {total_bytes_on_disk / 1024**3:.1f} GB  "
              f"Average bytes per token: {avg:.2f}")

        return total_tokens, avg


# --------------------------------------------------------------------------
# 2. Datasets
# --------------------------------------------------------------------------

class ByteDataset(Dataset):
    """
    Memory‑mapped byte dataset.  Returns fixed‑length chunks of raw bytes
    from ``corpus.bin``.

    Each sample is a dict with 'input_ids' and 'labels' (shifted by 1).
    """

    def __init__(
        self,
        bin_path: str,
        seq_length: int = 8192,
        stride: Optional[int] = None,
        split: str = "train",
        val_ratio: float = 0.05,
    ):
        self.bin_path = Path(bin_path)
        self.seq_length = seq_length
        self.stride = stride or seq_length

        # Memory‑map the file
        self._file = open(self.bin_path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.total_bytes = len(self._mm)

        # Train / val split
        split_point = int(self.total_bytes * (1 - val_ratio))
        if split == "train":
            self.start = 0
            self.end = split_point
        else:
            self.start = split_point
            self.end = self.total_bytes

        self.effective_len = self.end - self.start
        self.num_samples = max(0, (self.effective_len - self.seq_length - 1) // self.stride + 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = self.start + idx * self.stride
        raw = self._mm[offset : offset + self.seq_length + 1]  # +1 for label shift
        byte_arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
        input_ids = torch.from_numpy(byte_arr[:-1].copy())
        labels = torch.from_numpy(byte_arr[1:].copy())
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(self.seq_length, dtype=torch.bool),
        }

    def __del__(self):
        try:
            self._mm.close()
            self._file.close()
        except Exception:
            pass


class TokenDataset(Dataset):
    """
    Memory‑mapped BPE token dataset.  Returns fixed‑length chunks from
    ``corpus_bpe.npy``.
    """

    def __init__(
        self,
        npy_path: str,
        seq_length: int = 1792,
        stride: Optional[int] = None,
        split: str = "train",
        val_ratio: float = 0.05,
    ):
        self.npy_path = Path(npy_path)
        self.seq_length = seq_length
        self.stride = stride or seq_length

        self.tokens = np.load(self.npy_path, mmap_mode="r")
        total = len(self.tokens)

        split_point = int(total * (1 - val_ratio))
        if split == "train":
            self.start = 0
            self.end = split_point
        else:
            self.start = split_point
            self.end = total

        self.effective_len = self.end - self.start
        self.num_samples = max(0, (self.effective_len - self.seq_length - 1) // self.stride + 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = self.start + idx * self.stride
        chunk = self.tokens[offset : offset + self.seq_length + 1].astype(np.int64)
        input_ids = torch.from_numpy(chunk[:-1].copy())
        labels = torch.from_numpy(chunk[1:].copy())
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(self.seq_length, dtype=torch.bool),
        }


# --------------------------------------------------------------------------
# 3. Collate & DataLoader helpers
# --------------------------------------------------------------------------

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of same‑length samples into a batch."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


def create_dataloaders(
    model_name: str,
    cache_dir: str = "./data/spanish",
    byte_seq_length: int = 8192,
    token_seq_length: int = 1792,
    batch_size: int = 4,
    num_workers: int = 2,
    val_ratio: float = 0.05,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and val DataLoaders appropriate for the model type.

    Args:
        model_name: 'transformer' (BPE) or 'matmulfree' / 'hybrid' (byte‑level)
    """
    cache = Path(cache_dir)

    if model_name == "transformer":
        npy_path = cache / "corpus_bpe.npy"
        assert npy_path.exists(), (
            f"BPE corpus not found at {npy_path}. Run SpanishCorpusBuilder.build() first."
        )
        train_ds = TokenDataset(str(npy_path), seq_length=token_seq_length,
                                split="train", val_ratio=val_ratio)
        val_ds = TokenDataset(str(npy_path), seq_length=token_seq_length,
                              split="val", val_ratio=val_ratio)
    else:
        bin_path = cache / "corpus.bin"
        assert bin_path.exists(), (
            f"Byte corpus not found at {bin_path}. Run SpanishCorpusBuilder.build_bytes_only() first."
        )
        train_ds = ByteDataset(str(bin_path), seq_length=byte_seq_length,
                               split="train", val_ratio=val_ratio)
        val_ds = ByteDataset(str(bin_path), seq_length=byte_seq_length,
                             split="val", val_ratio=val_ratio)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True, num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False, num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[Data] {model_name}: train={len(train_ds):,} samples, val={len(val_ds):,} samples")
    return train_loader, val_loader
