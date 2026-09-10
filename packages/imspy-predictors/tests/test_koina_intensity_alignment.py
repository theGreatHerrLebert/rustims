"""Tests for the Koina fragment-intensity response mapping.

Koina answers a fragment-intensity request with one row *per fragment ion*,
keyed by the index of the submitted frame, and silently drops peptides that
violate the model's requirements. Mapping that back onto the input by position
instead of by index assigns spectra to the wrong precursors -- an error nothing
downstream can detect, because a shifted spectrum is still a plausible one.

These tests build the response by hand so the mapping can be pinned without
network access; the live-server checks are collected at the bottom and skipped
by default.
"""

import numpy as np
import pandas as pd
import pytest

from imspy_predictors.intensity.predictors import Prosit2023TimsTofWrapper


VECTOR_LENGTH = Prosit2023TimsTofWrapper._PROSIT_VECTOR_LENGTH


def make_input(sequences, charges=None, collision_energies=None):
    n = len(sequences)
    return pd.DataFrame({
        "peptide_sequences": list(sequences),
        "precursor_charges": list(charges) if charges else [2] * n,
        "collision_energies": list(collision_energies) if collision_energies else [0.3] * n,
        "instrument_types": ["TIMSTOF"] * n,
    })


def make_response(input_df, rows, annotations, intensities):
    """Build a Koina-shaped response: one row per fragment, indexed by input row."""
    return pd.DataFrame(
        {
            "peptide_sequences": input_df["peptide_sequences"].to_numpy()[rows],
            "precursor_charges": input_df["precursor_charges"].to_numpy()[rows],
            "collision_energies": input_df["collision_energies"].to_numpy()[rows],
            "instrument_types": input_df["instrument_types"].to_numpy()[rows],
            "intensities": np.asarray(intensities, dtype=np.float32),
            "annotation": [a.encode() for a in annotations],
        },
        index=np.asarray(rows),
    )


class TestPrositSlotLayout:
    """The 174 flat slots are 29 ordinals x (y+1, y+2, y+3, b+1, b+2, b+3)."""

    @pytest.mark.parametrize("annotation,expected_slot", [
        ("y1+1", 0), ("y1+2", 1), ("y1+3", 2),
        ("b1+1", 3), ("b1+2", 4), ("b1+3", 5),
        ("y2+1", 6), ("y29+3", 170), ("b29+3", 173),
    ])
    def test_annotation_maps_to_expected_slot(self, annotation, expected_slot):
        inp = make_input(["PEPTIDEK"])
        res = make_response(inp, [0], [annotation], [0.5])
        out, predicted = Prosit2023TimsTofWrapper._koina_result_to_prosit_array(res, inp)
        assert out.shape == (1, VECTOR_LENGTH)
        assert out[0, expected_slot] == pytest.approx(0.5)
        assert out[0].sum() == pytest.approx(0.5), "intensity landed in more than one slot"
        assert predicted.tolist() == [True]

    def test_layout_matches_reshape_dims(self):
        """(29, 6) after reshape, y ions on 0-2 and b ions on 3-5 as mask_outofcharge assumes."""
        from imspy_predictors.intensity.utility import reshape_dims

        inp = make_input(["PEPTIDEK"])
        res = make_response(inp, [0, 0], ["y3+2", "b3+2"], [0.25, 0.75])
        out, _ = Prosit2023TimsTofWrapper._koina_result_to_prosit_array(res, inp)
        cube = reshape_dims(out)
        assert cube.shape == (1, 29, 6)
        assert cube[0, 2, 1] == pytest.approx(0.25)   # y3, charge 2
        assert cube[0, 2, 4] == pytest.approx(0.75)   # b3, charge 2


class TestFilteredPeptideAlignment:
    """Peptides the input filter drops must not shift their neighbours."""

    @pytest.mark.parametrize("dropped", [0, 1, 2])
    def test_dropped_peptide_keeps_neighbours_in_place(self, dropped):
        inp = make_input(["AAAAAK", "BBBBBK", "CCCCCK"])
        kept = [r for r in range(3) if r != dropped]
        rows = [r for r in kept for _ in range(2)]
        annotations = ["y1+1", "b1+1"] * len(kept)
        intensities = [0.1 * (r + 1) for r in kept for _ in range(2)]
        res = make_response(inp, rows, annotations, intensities)

        out, predicted = Prosit2023TimsTofWrapper._koina_result_to_prosit_array(res, inp)

        assert not out[dropped].any(), "dropped peptide should keep an all-zero spectrum"
        assert predicted[dropped] is np.False_
        for r in kept:
            assert out[r, 0] == pytest.approx(0.1 * (r + 1)), f"row {r} got another peptide's spectrum"
            assert predicted[r] is np.True_

    def test_all_peptides_dropped_yields_empty_spectra(self):
        inp = make_input(["AAAAAK", "BBBBBK"])
        out, predicted = Prosit2023TimsTofWrapper._koina_result_to_prosit_array(pd.DataFrame(), inp)
        assert out.shape == (2, VECTOR_LENGTH)
        assert not out.any()
        assert not predicted.any()


class TestResponseIntegrityGuards:
    """Every assumption about the response is checked, not trusted."""

    def test_reset_index_is_rejected(self):
        """A response re-indexed 0..n_survivors-1 stays in range but means something else."""
        inp = make_input(["AAAAAK", "BBBBBK", "CCCCCK"])
        res = make_response(inp, [1, 2], ["y1+1", "y1+1"], [0.4, 0.6])
        res.index = pd.Index([0, 1])          # what reset_index(drop=True) would produce
        with pytest.raises(ValueError, match="no longer identifies the submitted peptide"):
            Prosit2023TimsTofWrapper._koina_result_to_prosit_array(res, inp)

    def test_reordered_response_is_rejected(self):
        inp = make_input(["AAAAAK", "BBBBBK"])
        res = make_response(inp, [0, 1], ["y1+1", "y1+1"], [0.4, 0.6])
        res.index = pd.Index([1, 0])          # index no longer describes the echoed peptide
        with pytest.raises(ValueError, match="no longer identifies the submitted peptide"):
            Prosit2023TimsTofWrapper._koina_result_to_prosit_array(res, inp)

    def test_collision_energy_mismatch_is_rejected(self):
        inp = make_input(["AAAAAK"], collision_energies=[0.30])
        res = make_response(inp, [0], ["y1+1"], [0.4])
        res["collision_energies"] = 0.35
        with pytest.raises(ValueError, match="collision_energies"):
            Prosit2023TimsTofWrapper._koina_result_to_prosit_array(res, inp)

    def test_out_of_range_index_is_rejected(self):
        inp = make_input(["AAAAAK"])
        res = make_response(inp, [0], ["y1+1"], [0.4])
        res.index = pd.Index([7])
        with pytest.raises(ValueError, match="out of range"):
            Prosit2023TimsTofWrapper._koina_result_to_prosit_array(res, inp)

    def test_duplicate_slot_is_rejected(self):
        """Which duplicate wins would depend on response ordering."""
        inp = make_input(["AAAAAK"])
        res = make_response(inp, [0, 0], ["y1+1", "y1+1"], [0.4, 0.6])
        with pytest.raises(ValueError, match="duplicated"):
            Prosit2023TimsTofWrapper._koina_result_to_prosit_array(res, inp)

    def test_unmappable_annotation_is_rejected(self):
        """An annotation format change must not silently remove signal."""
        inp = make_input(["AAAAAK"])
        res = make_response(inp, [0, 0], ["y1+1", "y1-NH3+1"], [0.4, 0.6])
        with pytest.raises(ValueError, match="do not fit the Prosit b/y layout"):
            Prosit2023TimsTofWrapper._koina_result_to_prosit_array(res, inp)

    @pytest.mark.parametrize("annotation", ["y30+1", "y1+4", "y0+1"])
    def test_out_of_layout_fragment_is_rejected(self, annotation):
        inp = make_input(["AAAAAK"])
        res = make_response(inp, [0], [annotation], [0.4])
        with pytest.raises(ValueError, match="do not fit the Prosit b/y layout"):
            Prosit2023TimsTofWrapper._koina_result_to_prosit_array(res, inp)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf])
    def test_non_finite_intensity_is_rejected(self, bad_value):
        inp = make_input(["AAAAAK"])
        res = make_response(inp, [0, 0], ["y1+1", "b1+1"], [0.4, bad_value])
        with pytest.raises(ValueError, match="non-finite"):
            Prosit2023TimsTofWrapper._koina_result_to_prosit_array(res, inp)

    def test_missing_annotation_column_is_rejected(self):
        inp = make_input(["AAAAAK"])
        res = make_response(inp, [0], ["y1+1"], [0.4]).drop(columns=["annotation"])
        with pytest.raises(ValueError, match="missing column"):
            Prosit2023TimsTofWrapper._koina_result_to_prosit_array(res, inp)


class TestPrositTensorShape:
    """The Prosit path must hand consumers (29, 2, 3) tensors, not (29, 6).

    ``imspy_simulation.utility.flatten_prosit_array`` reads ``array[:, 0, c]``
    and ``array[:, 1, c]``, so a 2-D array raises a numba TypingError deep in
    frame assembly; and ``np.squeeze`` on a single-precursor batch silently
    turns 29 ordinals into 29 'precursors'.
    """

    def test_flat_vector_maps_onto_ordinal_iontype_charge(self):
        # Label every slot with its own index so each one can be traced.
        processed = np.arange(VECTOR_LENGTH, dtype=np.float32)[None, :]
        tensors = Prosit2023TimsTofWrapper._to_prosit_tensors(processed)

        assert len(tensors) == 1
        assert tensors[0].shape == (29, 2, 3)
        t = tensors[0]
        assert t[0, 0, 0] == 0      # y1+1
        assert t[0, 0, 2] == 2      # y1+3
        assert t[0, 1, 0] == 3      # b1+1
        assert t[1, 0, 0] == 6      # y2+1
        assert t[28, 1, 2] == 173   # b29+3

    @pytest.mark.parametrize("n_precursors", [1, 2, 5])
    def test_one_tensor_per_precursor(self, n_precursors):
        """A batch of one must not collapse into 29 pseudo-precursors."""
        processed = np.zeros((n_precursors, VECTOR_LENGTH), dtype=np.float32)
        tensors = Prosit2023TimsTofWrapper._to_prosit_tensors(processed)
        assert len(tensors) == n_precursors
        assert all(t.shape == (29, 2, 3) for t in tensors)

    def test_flatten_prosit_array_accepts_the_tensor(self):
        """The simulation-side consumer needs 3 dimensions and a lossless layout."""
        from imspy_predictors.lazy_imports import get_simulation_flatten_prosit

        flatten_prosit_array = get_simulation_flatten_prosit()
        processed = np.arange(VECTOR_LENGTH, dtype=np.float32)[None, :]
        tensor = Prosit2023TimsTofWrapper._to_prosit_tensors(processed)[0]

        flat = flatten_prosit_array(tensor)

        assert flat.shape == (VECTOR_LENGTH,)
        # Block layout: [y c1][b c1][y c2][b c2][y c3][b c3], 29 entries each.
        assert flat[0] == 0 and flat[1] == 6      # y1+1, y2+1
        assert flat[29] == 3 and flat[30] == 9    # b1+1, b2+1
        assert flat[58] == 1                      # y1+2
        assert flat[87] == 4                      # b1+2
        # A permutation: nothing lost, nothing duplicated.
        assert sorted(flat.tolist()) == sorted(float(i) for i in range(VECTOR_LENGTH))


@pytest.mark.skipif(
    True,  # Set to False to run network tests
    reason="Network tests disabled by default"
)
class TestKoinaIntensityAlignmentLive:
    """Same alignment contract, against the live Koina server."""

    def test_filtered_peptide_does_not_shift_the_batch(self):
        wrapper = Prosit2023TimsTofWrapper(verbose=False)
        good_a, too_long, good_b = "LGGNEQVTR", "A" * 38, "KLVSMHK"

        solo_a = wrapper._predict_with_koina([good_a], [2], [0.3])
        solo_b = wrapper._predict_with_koina([good_b], [2], [0.3])
        batched = wrapper._predict_with_koina(
            [good_a, too_long, good_b], [2, 2, 2], [0.3, 0.3, 0.3])

        # Koina serves float32 and batches server-side, so repeated calls differ
        # by ~1e-7; compare with a tolerance rather than bit-exactly.
        assert np.allclose(batched[0], solo_a[0], atol=1e-5)
        assert not batched[1].any()
        assert np.allclose(batched[2], solo_b[0], atol=1e-5)

    def test_n_terminal_acetylation_is_filtered_not_misassigned(self):
        wrapper = Prosit2023TimsTofWrapper(verbose=False)
        solo = wrapper._predict_with_koina(["KLVSMHK"], [2], [0.3])
        batched = wrapper._predict_with_koina(
            ["[UNIMOD:1]KLVSMHK", "KLVSMHK"], [2, 2], [0.3, 0.3])
        assert not batched[0].any(), "Prosit does not support N-terminal acetylation"
        assert np.allclose(batched[1], solo[0], atol=1e-5)
