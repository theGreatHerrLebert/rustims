"""
Simple peptide tokenizer for training without imspy_connector dependency.

This tokenizer handles UNIMOD-style modified peptide sequences and can be
used for training when the full Rust-based tokenizer is not available.

Based on the ProformaTokenizer design but implemented in pure Python.
"""

import re
from typing import List, Dict, Optional, Tuple, Union
import json


# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
EXTENDED_AMINO_ACIDS = ["U", "O", "B", "Z", "X", "J"]

# Special tokens
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Common PTM composites (amino acid + modification)
COMMON_COMPOSITES = [
    "C[UNIMOD:4]",    # Carbamidomethyl
    "M[UNIMOD:35]",   # Oxidation
    "S[UNIMOD:21]",   # Phospho
    "T[UNIMOD:21]",   # Phospho
    "Y[UNIMOD:21]",   # Phospho
    "K[UNIMOD:1]",    # Acetyl
    "N[UNIMOD:7]",    # Deamidation
    "Q[UNIMOD:7]",    # Deamidation
    "K[UNIMOD:121]",  # Ubiquitination (GG)
    "R[UNIMOD:267]",  # Citrullination
    "K[UNIMOD:34]",   # Methylation
    "R[UNIMOD:34]",   # Methylation
    "K[UNIMOD:36]",   # Dimethylation
    "R[UNIMOD:36]",   # Dimethylation
    "K[UNIMOD:37]",   # Trimethylation
    "[UNIMOD:1]-",    # N-term Acetyl
]

# UNIMOD pattern
UNIMOD_PATTERN = re.compile(r'\[UNIMOD:(\d+)\]')


class SimpleTokenizer:
    """
    Simple peptide tokenizer for modified sequences.

    Handles UNIMOD notation: "MC[UNIMOD:4]PEPTIDE" -> ["M", "C[UNIMOD:4]", "P", ...]

    Args:
        add_special_tokens: Whether to add [CLS] and [SEP] (default: True)
        max_length: Maximum sequence length (default: 100)

    Example:
        >>> tokenizer = SimpleTokenizer()
        >>> tokens = tokenizer.tokenize("PEPTIDE")
        >>> ids = tokenizer.encode(tokens)
        >>> decoded = tokenizer.decode(ids)
    """

    def __init__(
        self,
        add_special_tokens: bool = True,
        max_length: int = 100,
    ):
        self.add_special_tokens = add_special_tokens
        self.max_length = max_length

        # Build vocabulary
        self._build_vocab()

    def _build_vocab(self):
        """Build the token vocabulary."""
        vocab = []

        # Special tokens first
        vocab.extend(SPECIAL_TOKENS)

        # Standard amino acids
        vocab.extend(AMINO_ACIDS)

        # Extended amino acids
        vocab.extend(EXTENDED_AMINO_ACIDS)

        # Common composites
        vocab.extend(COMMON_COMPOSITES)

        # Individual UNIMOD tokens (for rare modifications)
        for i in range(1, 2001):  # Cover most UNIMOD IDs
            vocab.append(f"[UNIMOD:{i}]")

        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}

        # Special token IDs
        self.pad_token_id = self.token_to_id["[PAD]"]
        self.unk_token_id = self.token_to_id["[UNK]"]
        self.cls_token_id = self.token_to_id["[CLS]"]
        self.sep_token_id = self.token_to_id["[SEP]"]
        self.mask_token_id = self.token_to_id["[MASK]"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, sequence: str) -> List[str]:
        """
        Tokenize a peptide sequence.

        Args:
            sequence: Peptide sequence with optional UNIMOD modifications

        Returns:
            List of tokens
        """
        tokens = []

        # Handle N-terminal modification
        if sequence.startswith("[UNIMOD:"):
            match = UNIMOD_PATTERN.match(sequence)
            if match:
                end = match.end()
                if end < len(sequence) and sequence[end] == "-":
                    # N-terminal mod: [UNIMOD:X]-
                    nterm_mod = sequence[:end+1]
                    if nterm_mod in self.token_to_id:
                        tokens.append(nterm_mod)
                    else:
                        # Split into UNIMOD token and dash
                        tokens.append(sequence[:end])
                    sequence = sequence[end+1:]

        i = 0
        while i < len(sequence):
            # Check for amino acid + modification composite
            if i + 1 < len(sequence) and sequence[i+1] == "[":
                # Find closing bracket
                bracket_end = sequence.find("]", i+1)
                if bracket_end != -1:
                    composite = sequence[i:bracket_end+1]
                    if composite in self.token_to_id:
                        # Known composite
                        tokens.append(composite)
                        i = bracket_end + 1
                        continue
                    else:
                        # Unknown composite - split into AA + mod
                        tokens.append(sequence[i])
                        mod = sequence[i+1:bracket_end+1]
                        if mod in self.token_to_id:
                            tokens.append(mod)
                        else:
                            tokens.append("[UNK]")
                        i = bracket_end + 1
                        continue

            # Regular amino acid
            if sequence[i] in self.token_to_id:
                tokens.append(sequence[i])
            elif sequence[i].upper() in self.token_to_id:
                tokens.append(sequence[i].upper())
            else:
                tokens.append("[UNK]")
            i += 1

        # Add special tokens
        if self.add_special_tokens:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

        return tokens

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Convert IDs back to sequence."""
        tokens = [self.id_to_token.get(i, "[UNK]") for i in ids]
        # Remove special tokens and join
        tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return "".join(tokens)

    def tokenize_batch(self, sequences: List[str]) -> List[List[str]]:
        """Tokenize multiple sequences."""
        return [self.tokenize(seq) for seq in sequences]

    def encode_batch(
        self,
        token_lists: List[List[str]],
        padding: bool = True,
        max_length: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Encode multiple token lists with padding.

        Returns:
            Tuple of (input_ids, attention_mask)
        """
        max_len = max_length or self.max_length

        all_ids = []
        all_masks = []

        for tokens in token_lists:
            # Truncate if needed
            if len(tokens) > max_len:
                if self.add_special_tokens:
                    tokens = tokens[:max_len-1] + ["[SEP]"]
                else:
                    tokens = tokens[:max_len]

            ids = self.encode(tokens)
            mask = [1] * len(ids)

            # Pad
            if padding:
                pad_len = max_len - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len

            all_ids.append(ids)
            all_masks.append(mask)

        return all_ids, all_masks

    def __call__(
        self,
        sequences: Union[str, List[str]],
        padding: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Dict:
        """
        Tokenize and encode sequences.

        Args:
            sequences: Single sequence or list of sequences
            padding: Whether to pad to max_length
            max_length: Maximum length (default: self.max_length)
            return_tensors: "pt" for PyTorch tensors, None for lists

        Returns:
            Dict with input_ids and attention_mask
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        token_lists = self.tokenize_batch(sequences)
        ids, masks = self.encode_batch(token_lists, padding=padding, max_length=max_length)

        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

        return {
            "input_ids": ids,
            "attention_mask": masks,
        }

    def save(self, path: str):
        """Save tokenizer vocabulary to file."""
        with open(path, "w") as f:
            json.dump({
                "vocab": self.vocab,
                "add_special_tokens": self.add_special_tokens,
                "max_length": self.max_length,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """Load tokenizer from file."""
        with open(path) as f:
            data = json.load(f)

        tokenizer = cls(
            add_special_tokens=data.get("add_special_tokens", True),
            max_length=data.get("max_length", 100),
        )

        if "vocab" in data:
            tokenizer.vocab = data["vocab"]
            tokenizer.token_to_id = {t: i for i, t in enumerate(tokenizer.vocab)}
            tokenizer.id_to_token = {i: t for i, t in enumerate(tokenizer.vocab)}

        return tokenizer

    @classmethod
    def with_defaults(cls) -> "SimpleTokenizer":
        """Create tokenizer with default settings."""
        return cls(add_special_tokens=True, max_length=100)

    def __repr__(self) -> str:
        return f"SimpleTokenizer(vocab_size={self.vocab_size}, max_length={self.max_length})"

    def __len__(self) -> int:
        return self.vocab_size


# Convenience function for compatibility
def get_tokenizer() -> SimpleTokenizer:
    """Get a tokenizer instance (tries ProformaTokenizer, falls back to SimpleTokenizer)."""
    try:
        from imspy_predictors.utilities.tokenizers import ProformaTokenizer
        return ProformaTokenizer.with_defaults()
    except ImportError:
        return SimpleTokenizer.with_defaults()
