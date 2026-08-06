"""Tests for the UNIMOD atomic-composition table."""
import numpy as np
import torch

from peptide_property_ng.modifications.composition import (
    DEFAULT_TABLE,
    ELEMENTS,
    CompositionTable,
    signed_log1p,
)


def _tokens() -> list[str]:
    """Id-indexed token strings, read straight from the built table."""
    with np.load(DEFAULT_TABLE, allow_pickle=False) as z:
        return [str(t) for t in z["tokens"]]


def test_table_loads():
    t = CompositionTable.load()
    assert t.vocab_size == 2175
    assert t.n_elements == len(ELEMENTS) == 31
    assert t.elements == ELEMENTS


def test_known_modifications():
    """Carbamidomethyl (UNIMOD:4, +C2H3NO) and Oxidation (UNIMOD:35, +O)."""
    t = CompositionTable.load()
    vocab = _tokens()
    ei = {e: i for i, e in enumerate(t.elements)}

    cam = next(i for i, v in enumerate(vocab) if "[UNIMOD:4]" in v)
    row = t.counts[cam]
    assert (row[ei["C"]], row[ei["H"]], row[ei["N"]], row[ei["O"]]) == (2, 3, 1, 1)

    ox = next(i for i, v in enumerate(vocab) if "[UNIMOD:35]" in v)
    assert t.counts[ox][ei["O"]] == 1


def test_bare_residue_is_zero():
    """A bare amino-acid token carries an all-zero composition delta."""
    t = CompositionTable.load()
    ala = _tokens().index("A")
    assert not t.counts[ala].any()


def test_signed_log1p_handles_negative_losses():
    """Neutral-loss modifications have negative counts; signed_log1p must stay finite."""
    x = torch.tensor([-40.0, -1.0, 0.0, 3.0, 120.0])
    out = signed_log1p(x)
    assert torch.isfinite(out).all()
    assert out[0] < 0 and out[2] == 0 and out[4] > 0
    # numpy path agrees
    assert np.allclose(signed_log1p(x.numpy()), out.numpy())


def test_table_has_negative_entries():
    """The table must preserve neutral-loss (negative) compositions."""
    assert (CompositionTable.load().counts < 0).any()
