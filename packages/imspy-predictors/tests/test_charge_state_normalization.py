"""Charge-state selection must stay finite when no state passes ``min_charge_contrib`` (rustims#488)."""

import numpy as np
import pandas as pd
import pytest

from imspy_predictors.ionization.predictors import (
    BinomialChargeStateDistributionModel,
    kept_charge_states,
)

# Peptide 1590 of TIMSIM-DDA-HLA-10K-G40-001: 9 basic sites, so with max_charge=4 and
# p_charge=0.5 every charge 1..4 lies below 0.25 and the old normalisation divided by zero.
HLA_488_PEPTIDE = "AVGKNKRLTKGGKKGAKKKVVD"


def test_kept_states_are_renormalised():
    out = kept_charge_states(np.array([0.1, 0.2, 0.5, 0.2, 0.0]), min_charge_contrib=0.25)
    assert out == [(2, 1.0)]

    out = kept_charge_states(np.array([0.0, 0.1, 0.3, 0.3, 0.3]), min_charge_contrib=0.25)
    assert [c for c, _ in out] == [2, 3, 4]
    assert sum(p for _, p in out) == pytest.approx(1.0)


def test_without_normalisation_probabilities_pass_through():
    out = kept_charge_states(np.array([0.0, 0.1, 0.3, 0.6, 0.0]), 0.25, normalize=False)
    assert out == [(2, 0.3), (3, 0.6)]


def test_zero_kept_mass_falls_back_to_most_probable_charge():
    # The #488 shape: all mass truncated above max_charge, every state below the threshold.
    charges = np.array([0.002, 0.018, 0.07, 0.164, 0.246])
    out = kept_charge_states(charges, min_charge_contrib=0.25)
    assert out == [(4, 1.0)]


def test_all_zero_charged_mass_yields_no_ions():
    assert kept_charge_states(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 0.25) == []


def test_binomial_model_on_hla_488_peptide_is_finite():
    pytest.importorskip("imspy_connector")
    model = BinomialChargeStateDistributionModel(charged_probability=0.5, max_charge=4)
    peptides = pd.DataFrame({"peptide_id": [1590], "sequence": [HLA_488_PEPTIDE]})

    ions = model.simulate_charge_state_distribution_pandas(peptides, min_charge_contrib=0.25)

    assert len(ions) == 1
    assert np.isfinite(ions.relative_abundance).all()
    assert ions.relative_abundance.sum() == pytest.approx(1.0)
