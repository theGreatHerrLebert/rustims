"""Tests for fragment-indexed intensity-target construction.

Channel layout: y+1=0, y+2=1, y+3=2, b+1=3, b+2=4, b+3=5.
"""
import numpy as np

from peptide_property_ng.data.fragment_targets import N_ION_CHANNELS, build_intensity_target


def test_by_ion_site_placement():
    """b_o -> site o-1 ; y_o -> site (L-1)-o ; base-peak normalised."""
    length = 5  # 4 cleavage sites
    ftype = np.array(["b", "b", "y", "y"])
    ford = np.array([1, 4, 1, 4], dtype=np.int32)
    fch = np.array([1, 1, 1, 1], dtype=np.int32)
    finten = np.array([10.0, 40.0, 20.0, 80.0], dtype=np.float32)

    t = build_intensity_target(length, 2, ftype, ford, fch, finten)
    assert t.shape == (length - 1, N_ION_CHANNELS)
    assert t[0, 3] == 10.0 / 80.0   # b1 -> site 0, channel b+1
    assert t[3, 3] == 40.0 / 80.0   # b4 -> site 3, channel b+1
    assert t[3, 0] == 20.0 / 80.0   # y1 -> site 3, channel y+1
    assert t[0, 0] == 80.0 / 80.0   # y4 -> site 0, channel y+1 (base peak)


def test_impossible_fragment_charges_masked():
    """Precursor charge 1 -> fragment charges 2 and 3 are impossible (-1)."""
    t = build_intensity_target(
        5, 1, np.array(["b"]), np.array([1], np.int32),
        np.array([1], np.int32), np.array([5.0], np.float32),
    )
    assert (t[:, 1] == -1).all() and (t[:, 2] == -1).all()   # y+2, y+3
    assert (t[:, 4] == -1).all() and (t[:, 5] == -1).all()   # b+2, b+3
    assert (t[:, 0] >= 0).all() and (t[:, 3] >= 0).all()     # charge-1 channels valid


def test_unobserved_possible_fragment_is_zero():
    """An observable-but-unmatched fragment is 0 (a real zero peak), not masked."""
    t = build_intensity_target(
        5, 2, np.array(["b"]), np.array([1], np.int32),
        np.array([1], np.int32), np.array([5.0], np.float32),
    )
    assert t[0, 3] == 1.0                       # b1 observed -> base peak
    assert t[1, 3] == 0.0 and t[2, 3] == 0.0    # unobserved but possible -> 0


def test_out_of_range_ordinal_ignored():
    """A fragment ordinal beyond the peptide is dropped, not crashed on."""
    t = build_intensity_target(
        5, 2, np.array(["b"]), np.array([99], np.int32),
        np.array([1], np.int32), np.array([5.0], np.float32),
    )
    assert (t[t >= 0] == 0).all()  # nothing placed
