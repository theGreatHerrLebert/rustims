"""Tests for the ProformaTokenizer."""

import pytest
import numpy as np


class TestProformaTokenizer:
    """Test suite for ProformaTokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer with default vocabulary."""
        from imspy_predictors.utilities.tokenizers import ProformaTokenizer
        return ProformaTokenizer.with_defaults()

    def test_tokenizer_creation(self, tokenizer):
        """Test tokenizer can be created."""
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0

    def test_vocab_size(self, tokenizer):
        """Test vocabulary size is reasonable."""
        # Should have ~2200 tokens (26 AA + special + composites + UNIMOD 1-2100)
        assert tokenizer.vocab_size > 2000
        assert tokenizer.vocab_size < 3000

    def test_special_tokens(self, tokenizer):
        """Test special token IDs are valid."""
        assert tokenizer.pad_token_id >= 0
        assert tokenizer.unk_token_id >= 0
        assert tokenizer.cls_token_id >= 0
        assert tokenizer.sep_token_id >= 0

    def test_tokenize_simple_sequence(self, tokenizer):
        """Test tokenizing a simple peptide sequence."""
        tokens = tokenizer.tokenize("PEPTIDE")
        assert len(tokens) > 0
        # Should have [CLS], P, E, P, T, I, D, E, [SEP]
        assert tokens[0] == "[CLS]"
        assert tokens[-1] == "[SEP]"
        assert len(tokens) == 9  # CLS + 7 AA + SEP

    def test_tokenize_with_modification(self, tokenizer):
        """Test tokenizing a modified peptide."""
        tokens = tokenizer.tokenize("MC[UNIMOD:4]PEPTIDE")
        assert len(tokens) > 0
        # C[UNIMOD:4] should be a single composite token
        assert "C[UNIMOD:4]" in tokens or "[UNIMOD:4]" in tokens

    def test_tokenize_with_rare_modification(self, tokenizer):
        """Test tokenizing with a rare modification."""
        tokens = tokenizer.tokenize("PEPTK[UNIMOD:999]IDE")
        assert len(tokens) > 0
        # Rare modification should be split: K + [UNIMOD:999]
        assert "K" in tokens

    def test_encode_decode_roundtrip(self, tokenizer):
        """Test that encode/decode is a valid roundtrip."""
        original = "PEPTIDE"
        tokens = tokenizer.tokenize(original)
        ids = tokenizer.encode(tokens)
        decoded = tokenizer.decode(ids)
        assert decoded == tokens

    def test_batch_tokenization(self, tokenizer):
        """Test batch tokenization."""
        sequences = ["PEPTIDE", "SEQUENCE", "PROTEIN"]
        tokens_batch = tokenizer.tokenize_batch(sequences)
        assert len(tokens_batch) == 3
        for tokens in tokens_batch:
            assert tokens[0] == "[CLS]"
            assert tokens[-1] == "[SEP]"

    def test_callable_interface(self, tokenizer):
        """Test the __call__ interface."""
        sequences = ["PEPTIDE", "SEQUENCE"]
        result = tokenizer(sequences, padding=True, return_tensors=None)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert len(result["input_ids"]) == 2

    def test_callable_with_pytorch(self, tokenizer):
        """Test the __call__ interface with PyTorch tensors."""
        import torch

        sequences = ["PEPTIDE", "SEQUENCE"]
        result = tokenizer(sequences, padding=True, return_tensors="pt")

        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)
        assert result["input_ids"].shape[0] == 2

    def test_callable_with_numpy(self, tokenizer):
        """Test the __call__ interface with NumPy arrays."""
        sequences = ["PEPTIDE", "SEQUENCE"]
        result = tokenizer(sequences, padding=True, return_tensors="np")

        assert isinstance(result["input_ids"], np.ndarray)
        assert isinstance(result["attention_mask"], np.ndarray)

    def test_extended_amino_acids(self, tokenizer):
        """Test that extended amino acids are in vocabulary."""
        extended_aa = ["U", "O", "B", "Z", "X", "J"]
        for aa in extended_aa:
            assert tokenizer.contains(aa), f"Extended AA {aa} not in vocabulary"

    def test_common_composites(self, tokenizer):
        """Test that common PTM composites are in vocabulary."""
        common_composites = [
            "C[UNIMOD:4]",   # Carbamidomethyl
            "M[UNIMOD:35]",  # Oxidation
            "S[UNIMOD:21]",  # Phospho
        ]
        for composite in common_composites:
            assert tokenizer.contains(composite), f"Composite {composite} not in vocabulary"

    def test_unimod_tokens(self, tokenizer):
        """Test that individual UNIMOD tokens are available."""
        # Check a few UNIMOD tokens
        for i in [1, 4, 7, 21, 35, 100, 500, 1000]:
            token = f"[UNIMOD:{i}]"
            assert tokenizer.contains(token), f"UNIMOD token {token} not in vocabulary"

    def test_get_vocab(self, tokenizer):
        """Test getting the full vocabulary."""
        vocab = tokenizer.get_vocab()
        assert len(vocab) == tokenizer.vocab_size

    def test_contains(self, tokenizer):
        """Test the contains method."""
        assert tokenizer.contains("[PAD]")
        assert tokenizer.contains("A")
        assert not tokenizer.contains("INVALID_TOKEN_123")

    def test_get_token_id(self, tokenizer):
        """Test getting token IDs."""
        pad_id = tokenizer.get_token_id("[PAD]")
        assert pad_id == tokenizer.pad_token_id

        invalid_id = tokenizer.get_token_id("INVALID_TOKEN_123")
        assert invalid_id is None

    def test_len(self, tokenizer):
        """Test __len__ returns vocab size."""
        assert len(tokenizer) == tokenizer.vocab_size

    def test_repr(self, tokenizer):
        """Test string representation."""
        repr_str = repr(tokenizer)
        assert "ProformaTokenizer" in repr_str
        assert str(tokenizer.vocab_size) in repr_str
