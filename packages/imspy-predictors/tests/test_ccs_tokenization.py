"""The chunked token-id builder must reproduce the one-shot tokenisation EXACTLY.

Why this is worth a test: `pad_token_id` is not 0, and `ProformaTokenizer.__call__(padding=True)`
pads to the longest tokenisation *in the list it is handed*, after which
`_preprocess_sequences` zero-fills the rest up to `pad_len`. So a row's tail is
`[pad_id] * (width - len) + [0] * (pad_len - width)` with `width` a property of the WHOLE input —
and the encoder reads those tokens, so a chunked tokeniser that re-derived `width` per chunk
would silently change the predicted CCS. That is the trap in the fix for the 23 GB / 10 GB-swap
`timsim-ccs` memory wall at 9M precursors, so it gets pinned here.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("imspy_connector")

PAD_LEN = 50

# "K" is the shortest (most padding); the 30-mer is the digest's max length; the UNIMOD entry
# exercises composite tokens.
SEQS = ["PEPTIDEK", "AC[UNIMOD:4]DEK", "K", "ACDEFGHIKLMNPQRSTVWY", "M" * 29 + "K"]


def _apex():
    """A predictor with only the pieces the tokenisation path touches.

    Deliberately does not go through `__init__`: that downloads and loads the CCS weights, and
    neither method under test looks at `self.model`.
    """
    from imspy_predictors.ccs.predictors import DeepPeptideIonMobilityApex
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer

    obj = DeepPeptideIonMobilityApex.__new__(DeepPeptideIonMobilityApex)
    obj.tokenizer = ProformaTokenizer.with_defaults()
    obj._device = torch.device("cpu")
    return obj


def test_pad_token_id_is_not_zero():
    """Guards every comparison below: at pad_id == 0 they would all pass trivially."""
    apex = _apex()
    assert int(apex.tokenizer.pad_token_id) != 0
    row = apex._preprocess_sequences(SEQS)[SEQS.index("K")].tolist()
    assert int(apex.tokenizer.pad_token_id) in row, "expected a real PAD region to compare"


@pytest.mark.parametrize("chunk_size", [1, 2, 3, len(SEQS), len(SEQS) + 10])
def test_chunked_token_ids_match_one_shot(chunk_size):
    apex = _apex()
    want = apex._preprocess_sequences(SEQS)
    got = torch.as_tensor(apex._token_ids(SEQS, chunk_size=chunk_size), dtype=torch.long)
    assert want.shape == (len(SEQS), PAD_LEN)
    assert torch.equal(got, want)


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_sequence_longer_than_pad_len_truncates_identically(chunk_size):
    """A tokenisation wider than `pad_len` is truncated, so it contributes no PAD region — and
    that must hold whether or not the long sequence shares a chunk with the short one."""
    seqs = ["A" * 90, "PEPTIDEK"]
    apex = _apex()
    want = apex._preprocess_sequences(seqs)
    got = torch.as_tensor(apex._token_ids(seqs, chunk_size=chunk_size), dtype=torch.long)
    assert want.shape == (2, PAD_LEN)
    assert torch.equal(got, want)


def test_empty_input():
    apex = _apex()
    assert apex._token_ids([]).shape == (0, PAD_LEN)
