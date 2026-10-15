"""Charge-state selection must stay finite when no state passes ``min_charge_contrib`` (rustims#488)."""

import numpy as np
import pandas as pd
import pytest

from imspy_predictors.ionization.predictors import (
    BinomialChargeStateDistributionModel,
    charge_state_distributions_from_sequences_rust,
    kept_charge_states,
)

# Peptide 1590 of TIMSIM-DDA-HLA-10K-G40-001: 9 basic sites, so with max_charge=4 and
# p_charge=0.5 every charge 1..4 lies below 0.25 and the old normalisation divided by zero.
HLA_488_PEPTIDE = "AVGKNKRLTKGGKKGAKKKVVD"


def _selection_before_fix(charges, min_charge_contrib, normalize):
    """The loop body of simulate_charge_state_distribution_pandas before the fix, verbatim."""
    charges = np.array(charges)
    if normalize:
        kept_prob_mass = np.sum(np.where(charges[1:] >= min_charge_contrib, charges[1:], 0))
        if kept_prob_mass > 0:
            charges[1:] = charges[1:] / kept_prob_mass
    return [(i, float(c)) for i, c in enumerate(charges[1:], start=1) if c >= min_charge_contrib]


def _kept_mass(charges, min_charge_contrib):
    c = np.asarray(charges[1:], dtype=np.float64)
    return np.sum(np.where(c >= min_charge_contrib, c, 0))


def test_matches_previous_selection_on_random_distributions():
    rng = np.random.default_rng(488)
    checked = 0
    for _ in range(5000):
        charges = rng.dirichlet(np.ones(rng.integers(3, 8)))
        threshold = float(rng.choice([0.0, 0.005, 0.1, 0.25, 0.4]))
        for normalize in (True, False):
            if normalize and _kept_mass(charges, threshold) == 0:
                continue  # the zero-mass case is the one this fix changes on purpose
            assert kept_charge_states(charges, threshold, normalize) == \
                _selection_before_fix(charges, threshold, normalize)
            checked += 1
    assert checked > 9000


def test_matches_previous_selection_on_binomial_outputs():
    sequences = ["R" * n + "AGLVSTPEDNQFYWIMAGLV"[n:] for n in range(12)] + [HLA_488_PEPTIDE]
    for p in (0.3, 0.5, 0.8):
        for charges in charge_state_distributions_from_sequences_rust(sequences, max_charge=4,
                                                                      charge_probability=p):
            for threshold in (0.005, 0.15, 0.25):
                if _kept_mass(charges, threshold) == 0:
                    continue
                assert kept_charge_states(charges, threshold) == \
                    _selection_before_fix(charges, threshold, True)


def test_rescaling_applies_before_threshold():
    # Five basic sites, p=0.5, max_charge=4: z=1 and z=4 sit below 0.25 until rescaled by 0.625.
    charges = np.array([0.03125, 0.15625, 0.3125, 0.3125, 0.15625])
    assert kept_charge_states(charges, 0.25) == [(1, 0.25), (2, 0.5), (3, 0.5), (4, 0.25)]


def test_zero_kept_mass_falls_back_to_most_probable_charge():
    # The #488 shape: all mass truncated above max_charge, every state below the threshold.
    charges = np.array([0.002, 0.018, 0.07, 0.164, 0.246])
    assert kept_charge_states(charges, 0.25) == [(4, 1.0)]


def test_zero_kept_mass_without_normalisation_is_unchanged():
    # Nothing is divided without normalisation, so there is nothing to fall back from.
    charges = np.array([0.002, 0.018, 0.07, 0.164, 0.246])
    assert kept_charge_states(charges, 0.25, normalize=False) == []


def test_all_zero_charged_mass_keeps_previous_behaviour():
    charges = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    assert kept_charge_states(charges, 0.25) == []
    # A zero threshold keeps the zero states, finite, exactly as before.
    assert kept_charge_states(charges, 0.0) == [(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0)]


def test_fallback_never_picks_a_nan_state():
    # np.argmax would return the NaN position; the fallback must pick the largest finite state.
    charges = np.array([0.0, np.nan, 0.1, 0.05, 0.2])
    assert kept_charge_states(charges, 0.25) == [(4, 1.0)]


def test_infinite_probability_yields_only_finite_abundances():
    out = kept_charge_states(np.array([0.0, 0.1, np.inf, 0.3, 0.05]), 0.25)
    assert all(np.isfinite(p) for _, p in out)


def test_preserves_input_dtype_like_before():
    charges = np.array([0.0, 0.1, 0.3, 0.3, 0.3], dtype=np.float32)
    assert kept_charge_states(charges, 0.25) == _selection_before_fix(charges, 0.25, True)


def test_does_not_mutate_input():
    charges = np.array([0.0, 0.1, 0.3, 0.3, 0.3])
    kept_charge_states(charges, 0.25)
    assert charges.tolist() == [0.0, 0.1, 0.3, 0.3, 0.3]


def test_binomial_model_on_hla_488_peptide_is_finite():
    model = BinomialChargeStateDistributionModel(charged_probability=0.5, max_charge=4)
    peptides = pd.DataFrame({"peptide_id": [1590], "sequence": [HLA_488_PEPTIDE]})

    ions = model.simulate_charge_state_distribution_pandas(peptides, min_charge_contrib=0.25)

    assert ions[["charge", "relative_abundance"]].values.tolist() == [[4, 1.0]]
