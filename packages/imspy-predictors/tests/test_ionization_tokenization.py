"""The chunked token-id builder must reproduce the one-shot tokenisation EXACTLY — ids AND mask.

Same trap as ``tests/test_ccs_tokenization.py`` for the ids: `pad_token_id` is not 0, and
`ProformaTokenizer.__call__(padding=True)` pads to the longest tokenisation *in the list it is
handed*, after which `_preprocess_sequences` zero-fills the rest up to `pad_len`. So a row's tail
is `[pad_id] * (width - len) + [0] * (pad_len - width)` with `width` a property of the WHOLE
input, and the encoder reads those tokens.

Unlike CCS/RT, this head's `_preprocess_sequences` also returns a `padding_mask`. That mask does
NOT depend on `width` — see `_token_ids`'s docstring for why (the Rust attention mask is
`[1] * len(row) + [0] * (max_len - len(row))`, i.e. purely a function of each row's own length) —
but a chunked implementation could still get it wrong independently of the ids, so it is pinned
separately rather than assumed to follow from the ids test passing.
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

    Deliberately does not go through `__init__`: that downloads and loads the charge-state
    weights, and neither method under test looks at `self.model`.
    """
    from imspy_predictors.ionization.predictors import DeepChargeStateDistribution
    from imspy_predictors.utilities.tokenizers import ProformaTokenizer

    obj = DeepChargeStateDistribution.__new__(DeepChargeStateDistribution)
    obj.tokenizer = ProformaTokenizer.with_defaults()
    obj._device = torch.device("cpu")
    return obj


def _want_ids_mask(apex, seqs, pad_len=50):
    tokens, mask = apex._preprocess_sequences(seqs, pad_len=pad_len)
    return tokens, mask


def _got_ids_mask(apex, seqs, pad_len=50, chunk_size=65536):
    ids, mask = apex._token_ids(seqs, pad_len=pad_len, chunk_size=chunk_size)
    return (
        torch.as_tensor(ids, dtype=torch.long),
        torch.as_tensor(mask, dtype=torch.bool),
    )


def test_pad_token_id_is_not_zero():
    """Guards every comparison below: at pad_id == 0 they would all pass trivially."""
    apex = _apex()
    assert int(apex.tokenizer.pad_token_id) != 0
    tokens, _ = apex._preprocess_sequences(SEQS)
    row = tokens[SEQS.index("K")].tolist()
    assert int(apex.tokenizer.pad_token_id) in row, "expected a real PAD region to compare"


@pytest.mark.parametrize("chunk_size", [1, 2, 3, len(SEQS), len(SEQS) + 10])
def test_chunked_token_ids_and_mask_match_one_shot(chunk_size):
    apex = _apex()
    want_ids, want_mask = _want_ids_mask(apex, SEQS)
    got_ids, got_mask = _got_ids_mask(apex, SEQS, chunk_size=chunk_size)
    assert want_ids.shape == (len(SEQS), PAD_LEN)
    assert torch.equal(got_ids, want_ids)
    assert torch.equal(got_mask, want_mask)


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_sequence_longer_than_pad_len_truncates_identically(chunk_size):
    """A tokenisation wider than `pad_len` is truncated, so it contributes no PAD region — and
    that must hold whether or not the long sequence shares a chunk with the short one, for both
    the ids and the mask."""
    seqs = ["A" * 90, "PEPTIDEK"]
    apex = _apex()
    want_ids, want_mask = _want_ids_mask(apex, seqs)
    got_ids, got_mask = _got_ids_mask(apex, seqs, chunk_size=chunk_size)
    assert want_ids.shape == (2, PAD_LEN)
    assert torch.equal(got_ids, want_ids)
    assert torch.equal(got_mask, want_mask)


def test_empty_input():
    apex = _apex()
    ids, mask = apex._token_ids([])
    assert ids.shape == (0, PAD_LEN)
    assert mask.shape == (0, PAD_LEN)


# The cases below mirror the CCS/RT edge cases, extended to also pin the mask.
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
    want_ids, want_mask = _want_ids_mask(apex, seqs)
    got_ids, got_mask = _got_ids_mask(apex, seqs, chunk_size=chunk_size)
    assert torch.equal(got_ids, want_ids), why
    assert torch.equal(got_mask, want_mask), why


@pytest.mark.parametrize("pad_len", [8, 32, 64])
def test_non_default_pad_len(pad_len):
    """`pad_len` is a parameter on both methods; the reconstruction must track it, not assume 50."""
    apex = _apex()
    want_ids, want_mask = _want_ids_mask(apex, SEQS, pad_len=pad_len)
    got_ids, got_mask = _got_ids_mask(apex, SEQS, pad_len=pad_len, chunk_size=2)
    assert want_ids.shape == (len(SEQS), pad_len)
    assert torch.equal(got_ids, want_ids)
    assert torch.equal(got_mask, want_mask)


def test_ids_fit_the_narrowed_dtype():
    """The int16 narrowing keys off `vocab_size`, which is a COUNT, not an id ceiling. If the
    tokenizer ever emits an id outside int16 the matrix must widen rather than wrap — silent
    wraparound (NumPy 1.x) would corrupt the padding region the encoder reads."""
    import numpy as np

    apex = _apex()
    ids, _ = apex._token_ids(SEQS)
    assert ids.dtype in (np.int16, np.int32)
    assert int(ids.max()) <= np.iinfo(ids.dtype).max, "id outstripped the dtype it was stored in"
    assert int(ids.min()) >= 0, "negative token id — the hallmark of a silent int16 wrap"


def test_mask_is_true_only_past_each_rows_own_length():
    """Pins the documented independence from `width`: the mask is a pure function of each row's
    own (clipped) length, not of the batch-wide padded width."""
    apex = _apex()
    ids, mask = apex._token_ids(SEQS)
    _, lengths_tokens = apex._preprocess_sequences(SEQS)
    for i, seq in enumerate(SEQS):
        # Recompute this row's own token length independently, via the single-sequence path.
        row_len = len(apex.tokenizer.encode(apex.tokenizer.tokenize(seq)))
        clipped = min(row_len, PAD_LEN)
        expected_row_mask = [pos >= clipped for pos in range(PAD_LEN)]
        assert mask[i].tolist() == expected_row_mask, seq
