# -*- coding: utf-8 -*-

"""
HuggingFace Datasets integration for byte-level language modeling.

Supports TinyStories, WikiText, and other HuggingFace datasets with
automatic byte-level conversion, caching, and preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, IterableDataset

from .data import ByteLevelDataset


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing.

    Works with any HuggingFace text dataset.  The text column is
    resolved automatically when *text_column* is ``"auto"``.
    For datasets with multiple relevant columns (e.g. instruction +
    output) set *text_columns* to a list and optionally provide a
    *separator* to join them.
    """

    # Dataset source
    dataset_name: str = "roneneldan/TinyStories"
    dataset_config: Optional[str] = None
    split: str = "train"

    # Text columns -------------------------------------------------------
    # Single column: set text_column (str).  "auto" = detect automatically.
    # Multiple columns: set text_columns (list[str]).  They are joined with
    #   *separator* in order.
    text_column: str = "auto"
    text_columns: Optional[List[str]] = None
    separator: str = "\n\n"

    # Preprocessing
    max_seq_length: int = 512
    stride: Optional[int] = None  # Defaults to max_seq_length (no overlap)

    # Filtering
    min_length: int = 10  # Minimum text length in bytes
    max_length: Optional[int] = None  # Maximum text length in bytes

    # Caching
    cache_dir: str = "./data/cache"
    use_cache: bool = True

    # Subset for testing
    max_samples: Optional[int] = None  # Limit total samples (for debugging)

    # Streaming (for large datasets)
    streaming: bool = False

    # Seed for reproducibility
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "text_column": self.text_column,
            "text_columns": self.text_columns,
            "separator": self.separator,
            "max_seq_length": self.max_seq_length,
            "stride": self.stride,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "max_samples": self.max_samples,
            "streaming": self.streaming,
            "seed": self.seed,
        }

    def cache_key(self) -> str:
        """Generate a unique cache key for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


@dataclass
class DatasetStatistics:
    """Statistics about a processed dataset."""
    
    num_samples: int = 0
    total_bytes: int = 0
    avg_bytes_per_sample: float = 0.0
    min_bytes: int = 0
    max_bytes: int = 0
    byte_distribution: Optional[Dict[int, int]] = None  # byte value -> count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "total_bytes": self.total_bytes,
            "avg_bytes_per_sample": self.avg_bytes_per_sample,
            "min_bytes": self.min_bytes,
            "max_bytes": self.max_bytes,
        }
    
    @classmethod
    def from_data(cls, byte_tensor: torch.Tensor) -> "DatasetStatistics":
        """Compute statistics from a byte tensor."""
        stats = cls()
        stats.total_bytes = len(byte_tensor)
        
        # Compute byte distribution
        unique, counts = torch.unique(byte_tensor, return_counts=True)
        stats.byte_distribution = {
            int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())
        }
        
        return stats


class HuggingFaceDataset(Dataset):
    """
    Dataset that loads from HuggingFace datasets hub and converts to byte sequences.
    
    Supports TinyStories, WikiText, OSCAR, and any text dataset on the hub.
    
    Args:
        config: DatasetConfig with loading parameters
        transform: Optional transform to apply to each sample
    """

    def __init__(
        self,
        config: DatasetConfig,
        transform: Optional[Callable] = None,
    ):
        self.config = config
        self.transform = transform
        self._data: Optional[torch.Tensor] = None
        self._inner: Optional[ByteLevelDataset] = None
        self._stats: Optional[DatasetStatistics] = None
        
        # Check for cached data
        cache_path = self._get_cache_path()
        if config.use_cache and cache_path.exists():
            self._load_from_cache(cache_path)
        else:
            self._load_and_preprocess()
            if config.use_cache:
                self._save_to_cache(cache_path)
    
    def _get_cache_path(self) -> Path:
        """Get path to cached dataset."""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self.config.cache_key()
        return cache_dir / f"{cache_key}.pt"
    
    def _load_from_cache(self, cache_path: Path) -> None:
        """Load preprocessed data from cache."""
        print(f"Loading cached dataset from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        self._data = cached["data"]
        self._stats = DatasetStatistics(**cached["stats"])
        self._create_inner_dataset()
    
    def _save_to_cache(self, cache_path: Path) -> None:
        """Save preprocessed data to cache."""
        print(f"Saving dataset cache to {cache_path}")
        torch.save({
            "data": self._data,
            "stats": self._stats.to_dict(),
            "config": self.config.to_dict(),
        }, cache_path)
    
    def _load_and_preprocess(self) -> None:
        """Load dataset from HuggingFace and convert to bytes."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install the 'datasets' library: pip install datasets"
            )
        
        print(f"Loading dataset: {self.config.dataset_name}")
        
        # Load dataset
        load_kwargs = {
            "path": self.config.dataset_name,
            "split": self.config.split,
            "streaming": self.config.streaming,
        }
        if self.config.dataset_config:
            load_kwargs["name"] = self.config.dataset_config
        
        dataset = load_dataset(**load_kwargs)
        
        # Limit samples if specified
        if self.config.max_samples and not self.config.streaming:
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
        
        # Resolve text column(s)
        text_columns = self._resolve_text_columns(dataset)
        print(f"Using text column(s): {text_columns}")
        
        # Convert to bytes
        all_bytes = []
        sample_lengths = []
        
        print("Converting to byte sequences...")
        for i, example in enumerate(dataset):
            if self.config.max_samples and i >= self.config.max_samples:
                break
                
            text = self._extract_text(example, text_columns)
            if text is None:
                continue
                
            byte_seq = list(text.encode('utf-8'))
            
            # Apply length filters
            if len(byte_seq) < self.config.min_length:
                continue
            if self.config.max_length and len(byte_seq) > self.config.max_length:
                byte_seq = byte_seq[:self.config.max_length]
            
            sample_lengths.append(len(byte_seq))
            all_bytes.extend(byte_seq)
            
            # Progress indicator
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1} samples...")
        
        self._data = torch.tensor(all_bytes, dtype=torch.long)
        
        # Compute statistics
        self._stats = DatasetStatistics.from_data(self._data)
        self._stats.num_samples = len(sample_lengths)
        if sample_lengths:
            self._stats.avg_bytes_per_sample = sum(sample_lengths) / len(sample_lengths)
            self._stats.min_bytes = min(sample_lengths)
            self._stats.max_bytes = max(sample_lengths)
        
        print(f"Dataset loaded: {self._stats.num_samples} samples, {self._stats.total_bytes:,} bytes")
        
        self._create_inner_dataset()

    # ------------------------------------------------------------------
    # Text column resolution
    # ------------------------------------------------------------------

    # Common text column names across popular HF datasets
    _KNOWN_TEXT_COLUMNS = [
        "text", "content", "story", "document", "passage",
        "sentence", "paragraph", "article", "body",
    ]
    # Common instruction / chat column pairs
    _KNOWN_MULTI_COLUMNS = [
        ["instruction", "output"],
        ["instruction", "response"],
        ["prompt", "completion"],
        ["prompt", "response"],
        ["input", "output"],
        ["question", "answer"],
        ["context", "response"],
        ["human", "assistant"],
    ]

    def _resolve_text_columns(self, dataset) -> List[str]:
        """Determine which column(s) contain the text.

        Priority:
        1. Explicit ``text_columns`` list in config.
        2. Explicit ``text_column`` (not "auto") in config.
        3. Auto-detect from dataset column names.
        """
        if self.config.text_columns:
            return list(self.config.text_columns)

        if self.config.text_column != "auto":
            return [self.config.text_column]

        # Auto-detect
        try:
            columns = set(dataset.column_names)
        except AttributeError:
            # Streaming / iterable datasets – peek at first example
            first = next(iter(dataset))
            columns = set(first.keys())

        # Check single-column names first
        for col in self._KNOWN_TEXT_COLUMNS:
            if col in columns:
                return [col]

        # Check multi-column pairs
        for pair in self._KNOWN_MULTI_COLUMNS:
            if all(c in columns for c in pair):
                return list(pair)

        # Fallback: pick the first string-valued column
        try:
            first = next(iter(dataset))
        except StopIteration:
            raise ValueError("Dataset is empty – cannot auto-detect text column.")
        for col in first:
            if isinstance(first[col], str):
                print(f"  Auto-detected text column: '{col}'")
                return [col]

        raise ValueError(
            f"Could not auto-detect a text column.  Available columns: "
            f"{sorted(columns)}.  Set text_column or text_columns explicitly."
        )

    def _extract_text(self, example: dict, text_columns: List[str]) -> Optional[str]:
        """Extract and join text from one or more columns."""
        parts = []
        for col in text_columns:
            val = example.get(col)
            if val is None:
                continue
            if isinstance(val, list):
                # e.g. conversation turns stored as a list of dicts
                val = self.config.separator.join(
                    turn.get("content", turn.get("value", str(turn)))
                    if isinstance(turn, dict) else str(turn)
                    for turn in val
                )
            parts.append(str(val))
        if not parts:
            return None
        return self.config.separator.join(parts)

    def _create_inner_dataset(self) -> None:
        """Create the inner ByteLevelDataset."""
        self._inner = ByteLevelDataset(
            data=self._data,
            max_seq_length=self.config.max_seq_length,
            stride=self.config.stride,
        )

    @property
    def statistics(self) -> DatasetStatistics:
        """Get dataset statistics."""
        return self._stats
    
    @property
    def raw_bytes(self) -> torch.Tensor:
        """Get raw byte tensor."""
        return self._data

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._inner[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class TinyStoriesDataset(HuggingFaceDataset):
    """
    TinyStories dataset with sensible defaults.
    
    TinyStories is a collection of simple children's stories,
    ideal for training and evaluating small language models.
    
    Args:
        split: 'train' or 'validation'
        max_seq_length: Sequence length for training
        cache_dir: Directory for caching processed data
        max_samples: Limit number of samples (for debugging/testing)
    """

    def __init__(
        self,
        split: str = "train",
        max_seq_length: int = 512,
        cache_dir: str = "./data/cache",
        max_samples: Optional[int] = None,
        **kwargs,
    ):
        config = DatasetConfig(
            dataset_name="roneneldan/TinyStories",
            split=split,
            text_column="text",
            max_seq_length=max_seq_length,
            cache_dir=cache_dir,
            max_samples=max_samples,
            min_length=50,  # Skip very short stories
            **kwargs,
        )
        super().__init__(config)


class WikiTextDataset(HuggingFaceDataset):
    """
    WikiText dataset (WikiText-103 or WikiText-2).
    
    Args:
        version: 'wikitext-103-v1' or 'wikitext-2-v1'
        split: 'train', 'validation', or 'test'
        max_seq_length: Sequence length for training
        cache_dir: Directory for caching processed data
    """

    def __init__(
        self,
        version: str = "wikitext-103-v1",
        split: str = "train",
        max_seq_length: int = 512,
        cache_dir: str = "./data/cache",
        **kwargs,
    ):
        config = DatasetConfig(
            dataset_name="wikitext",
            dataset_config=version,
            split=split,
            text_column="text",
            max_seq_length=max_seq_length,
            cache_dir=cache_dir,
            min_length=20,
            **kwargs,
        )
        super().__init__(config)


class TextDirectoryDataset(Dataset):
    """
    Dataset that loads all text files from a directory.
    
    Recursively finds all .txt files and converts to byte sequences.
    
    Args:
        directory: Path to directory containing text files
        max_seq_length: Sequence length for training
        extensions: List of file extensions to include
    """

    def __init__(
        self,
        directory: str,
        max_seq_length: int = 512,
        stride: Optional[int] = None,
        extensions: List[str] = [".txt", ".md"],
    ):
        self.directory = Path(directory)
        self.max_seq_length = max_seq_length
        self.stride = stride or max_seq_length
        
        # Find all text files
        all_bytes = []
        file_count = 0
        
        for ext in extensions:
            for file_path in self.directory.rglob(f"*{ext}"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    all_bytes.extend(list(text.encode('utf-8')))
                    file_count += 1
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
        
        print(f"Loaded {file_count} files, {len(all_bytes):,} bytes total")
        
        self._data = torch.tensor(all_bytes, dtype=torch.long)
        self._inner = ByteLevelDataset(
            data=self._data,
            max_seq_length=max_seq_length,
            stride=self.stride,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._inner[idx]


def create_train_val_split(
    dataset: HuggingFaceDataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[ByteLevelDataset, ByteLevelDataset]:
    """
    Split a dataset into train and validation sets.
    
    Splits the raw byte data, then creates ByteLevelDatasets.
    
    Args:
        dataset: Source dataset to split
        val_ratio: Fraction for validation (0.1 = 10%)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    torch.manual_seed(seed)
    
    raw_bytes = dataset.raw_bytes
    total_len = len(raw_bytes)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len
    
    # Split at a random point (to avoid always having same validation data)
    # But for reproducibility, we use seed
    train_data = raw_bytes[:train_len]
    val_data = raw_bytes[train_len:]
    
    train_dataset = ByteLevelDataset(
        data=train_data,
        max_seq_length=dataset.config.max_seq_length,
        stride=dataset.config.stride,
    )
    
    val_dataset = ByteLevelDataset(
        data=val_data,
        max_seq_length=dataset.config.max_seq_length,
        stride=dataset.config.stride,
    )
    
    return train_dataset, val_dataset


def prepare_datasets(
    config: DatasetConfig,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, ByteLevelDataset]:
    """
    Prepare train/val/test datasets from a single source.
    
    Args:
        config: Dataset configuration
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        
    Returns:
        Dict with 'train', 'val', 'test' datasets
    """
    # Load full dataset
    full_dataset = HuggingFaceDataset(config)
    raw_bytes = full_dataset.raw_bytes
    total_len = len(raw_bytes)
    
    # Compute split points
    test_len = int(total_len * test_ratio)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len - test_len
    
    # Split data
    train_data = raw_bytes[:train_len]
    val_data = raw_bytes[train_len:train_len + val_len]
    test_data = raw_bytes[train_len + val_len:]
    
    datasets = {}
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        datasets[name] = ByteLevelDataset(
            data=data,
            max_seq_length=config.max_seq_length,
            stride=config.stride,
        )
    
    print(f"Split sizes - Train: {len(datasets['train'])}, "
          f"Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
    
    return datasets


def print_dataset_stats(dataset: HuggingFaceDataset) -> None:
    """Print detailed statistics about a dataset."""
    stats = dataset.statistics
    
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    print(f"Number of original samples: {stats.num_samples:,}")
    print(f"Total bytes: {stats.total_bytes:,}")
    print(f"Average bytes per sample: {stats.avg_bytes_per_sample:.1f}")
    print(f"Min sample length: {stats.min_bytes} bytes")
    print(f"Max sample length: {stats.max_bytes} bytes")
    print(f"Training sequences: {len(dataset):,}")
    print(f"Sequence length: {dataset.config.max_seq_length}")
    
    if stats.byte_distribution:
        # Show most common bytes
        sorted_bytes = sorted(
            stats.byte_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        print("\nTop 10 most common bytes:")
        for byte_val, count in sorted_bytes:
            char = chr(byte_val) if 32 <= byte_val < 127 else f"\\x{byte_val:02x}"
            pct = 100 * count / stats.total_bytes
            print(f"  {byte_val:3d} ('{char}'): {count:,} ({pct:.2f}%)")
    
    print("=" * 50 + "\n")
