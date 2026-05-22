"""Tests for the Prosit-174 -> site-indexed intensity-target conversion.

This conversion is the single place the ordinal->site remap happens, and it is
easy to get wrong — hence the explicit b/y placement tests below.
"""
import numpy as np
import pytest

from peptide_property_ng.data.fragment_targets import (
    N_ION_CHANNELS,
    PROSIT_MAX_ORDINAL,
    PROSIT_VECTOR_LEN,
    prosit174_to_sites,
)


def _prosit_vector(entries: dict[tuple[int, int], float]) -> np.ndarray:
    """Build a 174-vector; ``entries`` maps (fragment_ordinal, channel) -> value."""
    v = np.zeros((PROSIT_MAX_ORDINAL, N_ION_CHANNELS), np.float32)
    for (ordinal, channel), value in entries.items():
        v[ordinal - 1, channel] = value
    return v.reshape(-1)


def test_b_ion_site_mapping():
    """b_k (Prosit ordinal k) lands at site k-1, b channels (3:6)."""
    length = 5  # 4 cleavage sites
    t = prosit174_to_sites(_prosit_vector({(1, 3): 0.11, (4, 3): 0.44}), length)
    assert t.shape == (length - 1, N_ION_CHANNELS)
    assert t[0, 3] == 0.11   # b1 -> site 0
    assert t[3, 3] == 0.44   # b4 -> site 3


def test_y_ion_site_mapping():
    """y_o (Prosit ordinal o) lands at site L-1-o, y channels (0:3)."""
    length = 5
    t = prosit174_to_sites(_prosit_vector({(1, 0): 0.81, (4, 0): 0.84}), length)
    assert t[3, 0] == 0.81   # y1 -> site L-1-1 = 3
    assert t[0, 0] == 0.84   # y4 -> site L-1-4 = 0


def test_complementary_ions_share_a_site():
    """Site i must hold b_{i+1} and y_{L-1-i} — the complementary pair of one cleavage."""
    length = 6  # site 2 should carry b3 and y3
    t = prosit174_to_sites(_prosit_vector({(3, 3): 0.3, (3, 0): 0.9}), length)
    assert t[2, 3] == 0.3   # b3 at site 2
    assert t[2, 0] == 0.9   # y3 at site 2


def test_minus_one_mask_carries_through():
    t = prosit174_to_sites(np.full(PROSIT_VECTOR_LEN, -1.0, np.float32), 10)
    assert (t == -1.0).all()


def test_shape_and_bounds():
    v = np.zeros(PROSIT_VECTOR_LEN, np.float32)
    assert prosit174_to_sites(v, 30).shape == (29, N_ION_CHANNELS)  # max peptide length
    assert prosit174_to_sites(v, 3).shape == (2, N_ION_CHANNELS)
    with pytest.raises(ValueError):
        prosit174_to_sites(v, 31)               # peptide longer than the 174-vector
    with pytest.raises(ValueError):
        prosit174_to_sites(np.zeros(100), 10)   # wrong vector length
