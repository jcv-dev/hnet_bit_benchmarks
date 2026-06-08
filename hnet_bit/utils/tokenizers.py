# -*- coding: utf-8 -*-

"""
Byte-level tokenizer for simplified SLM.

Uses direct UTF-8 byte encoding with vocabulary size 256.
No tokenizer training needed - simple and universal.
"""

import numpy as np


class ByteTokenizer:
    """
    Simple byte-level tokenizer using UTF-8 encoding.
    
    Attributes:
        vocab_size: 256 (all possible byte values)
        bos_idx: 254 (beginning of sequence token)
        eos_idx: 255 (end of sequence token)
    """
    
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.pad_idx = 0  # Use null byte as padding
        self.dtype = np.uint8
    
    @property
    def pad_token_id(self):
        """Alias for compatibility with HuggingFace API."""
        return self.pad_idx
    
    @property
    def eos_token_id(self):
        """Alias for compatibility with HuggingFace API."""
        return self.eos_idx
    
    @property
    def bos_token_id(self):
        """Alias for compatibility with HuggingFace API."""
        return self.bos_idx

    def encode(
        self, 
        seqs: list[str], 
        add_bos: bool = False, 
        add_eos: bool = False, 
        **kwargs
    ) -> list[dict[str, np.ndarray]]:
        """
        Encode strings to byte sequences.
        
        Args:
            seqs: List of strings to encode
            add_bos: Whether to prepend BOS token
            add_eos: Whether to append EOS token
            
        Returns:
            List of dicts with 'input_ids' containing byte arrays
        """
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")

            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte = bytearray(text_byte)
            text_byte_ids = np.array(text_byte, dtype=self.dtype)

            total_outputs.append({"input_ids": text_byte_ids})

        return total_outputs

    def decode(self, tokens: np.ndarray | list[int], **kwargs) -> str:
        """
        Decode byte sequences back to string.
        
        Args:
            tokens: Byte sequence to decode
            
        Returns:
            Decoded string
        """
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return bytearray(tokens).decode("utf-8", **kwargs)
