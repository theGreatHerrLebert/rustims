from typing import List, Tuple, Optional, Union
import imspy_connector

py_ml_utility = imspy_connector.py_ml_utility  # the module built with PyO3


class ProformaTokenizer:
    """
    Tokenizer for peptide sequences in PROFORMA/UNIMOD format.

    This tokenizer wraps a high-performance Rust implementation that:
    - Tokenizes peptide sequences with UNIMOD modifications
    - Uses a hybrid composite strategy for common PTMs
    - Supports extended amino acids (U, O, B, Z, X, J)
    - Provides vocabulary management and serialization

    Attributes:
        vocab_size: Number of tokens in the vocabulary
        pad_token_id: ID of the padding token
        unk_token_id: ID of the unknown token
        cls_token_id: ID of the [CLS] token
        sep_token_id: ID of the [SEP] token

    Example:
        >>> tokenizer = ProformaTokenizer.with_defaults()
        >>> tokens = tokenizer.tokenize("MC[UNIMOD:4]PEPTIDE")
        >>> print(tokens)
        ['[CLS]', 'M', 'C[UNIMOD:4]', 'P', 'E', 'P', 'T', 'I', 'D', 'E', '[SEP]']
    """

    def __init__(
        self,
        common_composites: Optional[List[str]] = None,
        terminal_mods: Optional[List[str]] = None,
        special_tokens: Optional[List[str]] = None,
        unimod_tokens: Optional[List[str]] = None,
    ):
        """
        Create a new tokenizer with custom vocabulary.

        Args:
            common_composites: AA+modification combinations to tokenize as single units.
                              If None, uses default ~30 common PTMs.
            terminal_mods: N-terminal and C-terminal modifications.
                          If None, uses default terminal modifications.
            special_tokens: Special tokens like [PAD], [CLS], [SEP], [MASK], [UNK].
                           If None, uses default special tokens.
            unimod_tokens: Individual UNIMOD tokens for fallback.
                          If None, uses [UNIMOD:1] through [UNIMOD:500].
        """
        self.__py_ptr = py_ml_utility.PyProformaTokenizer(
            common_composites=common_composites,
            terminal_mods=terminal_mods,
            special_tokens=special_tokens,
            unimod_tokens=unimod_tokens,
        )

    @classmethod
    def with_defaults(cls) -> "ProformaTokenizer":
        """
        Create a tokenizer with default vocabulary suitable for transformer models.

        This uses sensible defaults for proteomics:
        - Standard + extended amino acids (26 total: A-Y + U, O, B, Z, X, J)
        - Common PTM composites (~30 common modifications as single tokens)
        - N-term and C-term modifications
        - All PSI standard UNIMOD tokens (1-2100) for full compatibility

        The resulting vocabulary size is approximately 2200 tokens.

        Returns:
            ProformaTokenizer instance with default vocabulary
        """
        instance = cls.__new__(cls)
        instance.__py_ptr = py_ml_utility.PyProformaTokenizer.with_defaults()
        return instance

    def tokenize(self, sequence: str) -> List[str]:
        """Tokenize a single peptide sequence."""
        return self.__py_ptr.tokenize(sequence)

    def tokenize_batch(self, sequences: List[str]) -> List[List[str]]:
        """Tokenize multiple peptide sequences in parallel."""
        return self.__py_ptr.tokenize_many(sequences)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return self.__py_ptr.encode(tokens)

    def decode(self, ids: List[int]) -> List[str]:
        """Convert IDs back to tokens."""
        return self.__py_ptr.decode(ids)

    def encode_batch(
        self, token_batches: List[List[str]], pad: bool = True
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Encode multiple token sequences.

        Args:
            token_batches: List of token lists
            pad: Whether to pad sequences to the same length

        Returns:
            Tuple of (encoded_ids, attention_masks)
        """
        return self.__py_ptr.encode_many(token_batches, pad=pad)

    def decode_batch(self, id_batches: List[List[int]]) -> List[List[str]]:
        """Decode multiple ID sequences."""
        return self.__py_ptr.decode_many(id_batches)

    def save_vocab(self, path: str) -> None:
        """Save tokenizer vocabulary and metadata to JSON."""
        self.__py_ptr.save_vocab(path)

    @classmethod
    def load_vocab(cls, path: str) -> "ProformaTokenizer":
        """Load tokenizer from saved vocabulary file."""
        instance = cls.__new__(cls)
        instance.__py_ptr = py_ml_utility.PyProformaTokenizer.load_vocab(path)
        return instance

    def get_vocab(self) -> List[str]:
        """Get the complete vocabulary as a list of tokens."""
        return self.__py_ptr.get_vocab()

    def set_num_threads(self, n: int) -> None:
        """Set the number of threads for parallel tokenization."""
        self.__py_ptr.set_num_threads(n)

    def get_py_ptr(self) -> py_ml_utility.PyProformaTokenizer:
        """Get the underlying Rust tokenizer object."""
        return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, py_ptr: py_ml_utility.PyProformaTokenizer) -> "ProformaTokenizer":
        """Create a tokenizer from an existing Rust object."""
        instance = cls.__new__(cls)
        instance.__py_ptr = py_ptr
        return instance

    def to_json(self, path: str) -> None:
        """Save tokenizer vocabulary and metadata to JSON."""
        self.save_vocab(path)

    @classmethod
    def from_json(cls, path: str) -> "ProformaTokenizer":
        """Load tokenizer vocabulary and metadata from JSON."""
        return cls.load_vocab(path)

    # Properties for convenience
    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return self.__py_ptr.vocab_size()

    @property
    def pad_token_id(self) -> int:
        """ID of the padding token."""
        return self.__py_ptr.pad_token_id()

    @property
    def unk_token_id(self) -> int:
        """ID of the unknown token."""
        return self.__py_ptr.unk_token_id()

    @property
    def cls_token_id(self) -> int:
        """ID of the [CLS] token."""
        return self.__py_ptr.cls_token_id()

    @property
    def sep_token_id(self) -> int:
        """ID of the [SEP] token."""
        return self.__py_ptr.sep_token_id()

    def contains(self, token: str) -> bool:
        """Check if a token is in the vocabulary."""
        return self.__py_ptr.contains(token)

    def get_token_id(self, token: str) -> Optional[int]:
        """Get the ID for a specific token, or None if not found."""
        return self.__py_ptr.get_token_id(token)

    def __call__(
        self,
        sequences: Union[str, List[str]],
        padding: bool = True,
        return_tensors: Optional[str] = None,
    ) -> dict:
        """
        Tokenize and encode sequences in one step.

        Args:
            sequences: Single sequence or list of peptide sequences
            padding: Whether to pad sequences
            return_tensors: If 'pt', return PyTorch tensors; if 'np', return NumPy arrays

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        tokens = self.tokenize_batch(sequences)
        input_ids, attention_mask = self.encode_batch(tokens, pad=padding)
        # PyO3 converts Vec<u8> to bytes; normalize to list of ints
        attention_mask = [list(mask) for mask in attention_mask]

        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        elif return_tensors == "np":
            import numpy as np
            return {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    def __repr__(self) -> str:
        return f"ProformaTokenizer(vocab_size={self.vocab_size})"

    def __len__(self) -> int:
        return self.vocab_size
