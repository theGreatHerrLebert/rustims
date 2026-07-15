"""Pin the fragment-intensity indexing. One transposed axis silently corrupts every spectrum, so
the (29,2,3) -> (ion_type, ordinal, charge) decode is verified against v1's OWN flattener, not
reasoned about.

The codebase has two incompatible flat-174 serialisations, so the *tensor* is the only unambiguous
representation. The single axis fact the decode needs — axis-2 index 0 is a y ion, index 1 is a b
ion — is taken from flatten_prosit_array, and these tests require the decode to agree with it for
every cell.
"""

import numpy as np
import pytest

from imspy_simulation.timsim.jobs.fragments import decode_tensor


def _flatten():
    return pytest.importorskip("imspy_simulation.utility").flatten_prosit_array


def test_axis2_zero_is_y_and_one_is_b_per_flatten_prosit_array():
    """flatten_prosit_array lays the flat vector out charge-major, y-block before b-block within each
    charge: [y_c1(29), b_c1(29), y_c2(29), b_c2(29), y_c3(29), b_c3(29)]. So a marker at tensor cell
    [pos, 0, c] must land in that charge's Y block, and [pos, 1, c] in its B block. This is the fact
    the decode relies on; require it to hold rather than assume it.
    """
    flatten = _flatten()
    for pos in (0, 10, 28):
        for c in range(3):
            for a2, expected_ion in ((0, "y"), (1, "b")):
                t = np.zeros((29, 2, 3), dtype=np.float32)
                t[pos, a2, c] = 1.0
                flat = np.asarray(flatten(t)).flatten()
                nz = np.nonzero(flat > 0.5)[0]
                assert len(nz) == 1, f"flatten lost the marker at {(pos, a2, c)}"
                # Which 29-block did it land in? Block order is y_c1,b_c1,y_c2,b_c2,y_c3,b_c3.
                block = nz[0] // 29
                block_ion = "y" if block % 2 == 0 else "b"
                block_charge = block // 2 + 1
                assert block_ion == expected_ion, (
                    f"tensor axis-2={a2} landed in a {block_ion} block; decode calls it "
                    f"{expected_ion}"
                )
                assert block_charge == c + 1
                assert nz[0] % 29 == pos, "ordinal (position) must be preserved"


def test_decode_tensor_matches_that_convention():
    """The decode must label every cell exactly as flatten_prosit_array's layout implies."""
    flatten = _flatten()
    t = np.full((29, 2, 3), -1.0, dtype=np.float32)  # start all-absent (Prosit's -1)
    # Light up a handful of cells with distinct values.
    cells = {(0, 0, 0): 0.9, (4, 1, 1): 0.5, (28, 0, 2): 0.2, (10, 1, 0): 0.7}
    for (k, a2, c), v in cells.items():
        t[k, a2, c] = v

    got = {(ion, ordinal, charge): inten for ion, ordinal, charge, inten in decode_tensor(t, floor=1e-3)}
    # Only the four lit cells should survive the floor (the -1 slots are dropped).
    assert len(got) == 4
    for (k, a2, c), v in cells.items():
        ion = "y" if a2 == 0 else "b"
        assert got[(ion, k + 1, c + 1)] == pytest.approx(v)

    # Cross-check the ion labelling against the flattener for one cell.
    marker = np.zeros((29, 2, 3), dtype=np.float32)
    marker[4, 1, 1] = 1.0
    flat = np.asarray(flatten(marker)).flatten()
    block = int(np.nonzero(flat > 0.5)[0][0]) // 29
    assert ("y" if block % 2 == 0 else "b") == "b"  # our decode called [.,1,.] a b ion


def test_floor_and_masking_drop_absent_and_weak_fragments():
    t = np.full((29, 2, 3), -1.0, dtype=np.float32)  # everything structurally absent
    t[0, 0, 0] = 0.0005  # below floor
    t[1, 0, 0] = 0.5     # kept
    frags = list(decode_tensor(t, floor=1e-3))
    assert frags == [("y", 2, 1, pytest.approx(0.5))]
