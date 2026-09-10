"""Tests for the Prosit -> local-model fallback in the fragment-intensity job.

Prosit rejects N-terminal acetylation, cysteinylation and sequences over 30 aa,
which would leave those precursors with no MS2 signal at all. The fallback
predicts them with the local model instead. It splices results back by row
position, so these tests pin the position handling: a spectrum landing on the
wrong precursor is a plausible spectrum, and nothing downstream would notice.

The local predictor is stubbed so none of this needs network or model weights.
"""

import numpy as np
import pandas as pd
import pytest

from imspy_simulation.timsim.jobs import simulate_fragment_intensities as job


def tensor(fill):
    return np.full((29, 2, 3), fill, dtype=np.float32)


@pytest.fixture
def transmitted():
    return pd.DataFrame({
        "sequence": ["PEPTIDEK", "[UNIMOD:1]ACETYLK", "OTHERK"],
        "charge": [2, 2, 2],
        "collision_energy": [30.0, 30.0, 30.0],
    })


@pytest.fixture
def prosit_result(transmitted):
    """What the Prosit wrapper returns: row 1 rejected, CE already normalised."""
    out = transmitted.copy()
    out["collision_energy"] = out["collision_energy"] / 100.0
    out["intensity_predicted"] = [True, False, True]
    out["intensity"] = [tensor(0.1), tensor(0.0), tensor(0.3)]
    return out


@pytest.fixture
def stub_local(monkeypatch):
    """Stand in for the local model, recording what it was asked to predict."""
    seen = {}

    class StubPredictor:
        def __init__(self, verbose=False):
            pass

        def simulate_ion_intensities_pandas_batched(self, data, batch_size_tf_ds=None):
            seen["sequences"] = data["sequence"].tolist()
            seen["collision_energies"] = data["collision_energy"].tolist()
            out = data.copy().reset_index(drop=True)   # the real one resets the index
            out["intensity"] = [tensor(0.9) for _ in range(len(out))]
            return out

    monkeypatch.setattr(job, "DeepPeptideIntensityPredictor", StubPredictor)
    return seen


class TestPrositLocalFallback:

    def test_only_the_rejected_row_is_replaced(self, prosit_result, transmitted, stub_local):
        filled = job._fill_prosit_rejects_with_local_model(
            prosit_result, transmitted, batch_size=512, verbose=False)

        assert filled["intensity"].iloc[0].max() == pytest.approx(0.1), "Prosit row was overwritten"
        assert filled["intensity"].iloc[1].max() == pytest.approx(0.9), "fallback did not land on the rejected row"
        assert filled["intensity"].iloc[2].max() == pytest.approx(0.3), "Prosit row was overwritten"

    def test_the_rejected_peptide_is_what_gets_predicted(self, prosit_result, transmitted, stub_local):
        job._fill_prosit_rejects_with_local_model(
            prosit_result, transmitted, batch_size=512, verbose=False)
        assert stub_local["sequences"] == ["[UNIMOD:1]ACETYLK"]

    def test_collision_energy_is_not_normalised_twice(self, prosit_result, transmitted, stub_local):
        """The subset must come from the raw frame, not from the normalised result."""
        job._fill_prosit_rejects_with_local_model(
            prosit_result, transmitted, batch_size=512, verbose=False)
        assert stub_local["collision_energies"] == [30.0], "fallback was fed an already-normalised CE"

    def test_provenance_is_recorded(self, prosit_result, transmitted, stub_local):
        filled = job._fill_prosit_rejects_with_local_model(
            prosit_result, transmitted, batch_size=512, verbose=False)
        assert filled["intensity_model"].tolist() == ["prosit", "local", "prosit"]
        assert filled["intensity_predicted"].all()

    def test_nothing_to_do_when_prosit_took_everything(self, prosit_result, transmitted, stub_local):
        prosit_result["intensity_predicted"] = True
        filled = job._fill_prosit_rejects_with_local_model(
            prosit_result, transmitted, batch_size=512, verbose=False)
        assert "sequences" not in stub_local, "local model should not have been called"
        assert filled["intensity"].iloc[1].max() == pytest.approx(0.0)

    def test_reordered_frames_are_rejected(self, prosit_result, transmitted, stub_local):
        scrambled = transmitted.iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="no longer in the same order"):
            job._fill_prosit_rejects_with_local_model(
                prosit_result, scrambled, batch_size=512, verbose=False)

    def test_length_mismatch_is_rejected(self, prosit_result, transmitted, stub_local):
        with pytest.raises(ValueError, match="Row order is the only|cannot be identified safely"):
            job._fill_prosit_rejects_with_local_model(
                prosit_result, transmitted.iloc[:2], batch_size=512, verbose=False)

    def test_missing_mask_is_a_no_op(self, prosit_result, transmitted, stub_local):
        """Older predictors that do not report a mask must not break the job."""
        prosit_result = prosit_result.drop(columns=["intensity_predicted"])
        filled = job._fill_prosit_rejects_with_local_model(
            prosit_result, transmitted, batch_size=512, verbose=False)
        assert "sequences" not in stub_local
        assert "intensity_model" not in filled.columns
