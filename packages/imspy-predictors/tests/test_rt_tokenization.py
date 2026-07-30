"""The chunked token-id builder must reproduce the one-shot tokenisation EXACTLY.

Same trap as ``tests/test_ccs_tokenization.py``: `pad_token_id` is not 0, and
`ProformaTokenizer.__call__(padding=True)` pads to the longest tokenisation *in the list it is
handed*, after which `_preprocess_sequences` zero-fills the rest up to `pad_len`. So a row's tail
is `[pad_id] * (width - len) + [0] * (pad_len - width)` with `width` a property of the WHOLE
input — and the encoder reads those tokens, so a chunked tokeniser that re-derived `width` per
chunk would silently change the predicted retention time. `timsim-rt` runs on the deduplicated
peptide table and carried the exact same unbounded whole-batch call CCS had before it was fixed;
this pins the ported fix's bit-identity the same way.
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

    Deliberately does not go through `__init__`: that downloads and loads the RT weights, and
    neither method under test looks at `self.model`.
    """
    from imspy_predictors.rt.predictors import DeepChromatographyApex
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer

    obj = DeepChromatographyApex.__new__(DeepChromatographyApex)
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


# The cases below were raised by an independent review as reachable-but-unpinned. Each one is a
# distinct way the reconstructed `width` could be wrong while the cases above still passed.
@pytest.mark.parametrize("chunk_size", [1, 2, 8])
@pytest.mark.parametrize(
    "seqs, why",
    [
        (["PEPTIDEK", "ANOTHERK"], "all rows the same length: width == every len, so no PAD region"),
        (["K", "K", "K"], "all rows minimal and equal"),
        (["A" * 48, "K"], "width lands exactly on pad_len for one row, PAD-filling the other"),
        (["A" * 90, "A" * 91], "every row truncated: width == pad_len, no PAD region anywhere"),
        (["PEPTIDEK"], "a single sequence — width is that row's own length"),
    ],
)
def test_width_edge_cases_match_one_shot(seqs, why, chunk_size):
    apex = _apex()
    want = apex._preprocess_sequences(seqs)
    got = torch.as_tensor(apex._token_ids(seqs, chunk_size=chunk_size), dtype=torch.long)
    assert torch.equal(got, want), why


@pytest.mark.parametrize("pad_len", [8, 32, 64])
def test_non_default_pad_len(pad_len):
    """`pad_len` is a parameter on both methods; the reconstruction must track it, not assume 50."""
    apex = _apex()
    want = apex._preprocess_sequences(SEQS, pad_len=pad_len)
    got = torch.as_tensor(apex._token_ids(SEQS, pad_len=pad_len, chunk_size=2), dtype=torch.long)
    assert want.shape == (len(SEQS), pad_len)
    assert torch.equal(got, want)


def test_ids_fit_the_narrowed_dtype():
    """The int16 narrowing keys off `vocab_size`, which is a COUNT, not an id ceiling. If the
    tokenizer ever emits an id outside int16 the matrix must widen rather than wrap — silent
    wraparound (NumPy 1.x) would corrupt the padding region the encoder reads."""
    import numpy as np

    apex = _apex()
    ids = apex._token_ids(SEQS)
    assert ids.dtype in (np.int16, np.int32)
    assert int(ids.max()) <= np.iinfo(ids.dtype).max, "id outstripped the dtype it was stored in"
    assert int(ids.min()) >= 0, "negative token id — the hallmark of a silent int16 wrap"
