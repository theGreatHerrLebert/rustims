from typing import List, Tuple
import imspy_connector

py_ml_utility = imspy_connector.py_ml_utility  # the module built with PyO3

class ProformaTokenizer:
    def __init__(
        self,
        common_composites: List[str],
        terminal_mods: List[str],
        special_tokens: List[str] = None,
    ):
        if special_tokens is None:
            special_tokens = ["[PAD]", "[MASK]", "[UNK]", "[CLS]", "[SEP]"]

        self.__py_ptr = py_ml_utility.PyProformaTokenizer(
            common_composites=common_composites,
            terminal_mods=terminal_mods,
            special_tokens=special_tokens,
        )

    def tokenize(self, sequence: str) -> List[str]:
        return self.__py_ptr.tokenize(sequence)

    def tokenize_batch(self, sequences: List[str]) -> List[List[str]]:
        return self.__py_ptr.tokenize_many(sequences)

    def encode(self, tokens: List[str]) -> List[int]:
        return self.__py_ptr.encode(tokens)

    def decode(self, ids: List[int]) -> List[str]:
        return self.__py_ptr.decode(ids)

    def encode_batch(
        self, token_batches: List[List[str]], pad: bool = True
    ) -> Tuple[List[List[int]], List[List[int]]]:
        return self.__py_ptr.encode_many(token_batches, pad=pad)

    def decode_batch(self, id_batches: List[List[int]]) -> List[List[str]]:
        return self.__py_ptr.decode_many(id_batches)

    def save_vocab(self, path: str) -> None:
        self.__py_ptr.save_vocab(path)

    @classmethod
    def load_vocab(cls, path: str) -> "ProformaTokenizer":
        instance = cls.__new__(cls)
        instance.__py_ptr = py_ml_utility.PyProformaTokenizer.load_vocab(path)
        return instance

    def get_vocab(self) -> List[str]:
        return self.__py_ptr.get_vocab()

    def set_num_threads(self, n: int) -> None:
        self.__py_ptr.set_num_threads(n)

    def get_py_ptr(self) -> py_ml_utility.PyProformaTokenizer:
        return self.__py_ptr

    @classmethod
    def from_py_ptr(cls, py_ptr: py_ml_utility.PyProformaTokenizer) -> "ProformaTokenizer":
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

    def __repr__(self) -> str:
        return f"ProformaTokenizer(vocab_size={len(self.get_vocab())})"
