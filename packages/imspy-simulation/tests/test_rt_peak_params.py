"""The normalized, identity-keyed elution-peak parameters (sigma_hat, k_hat)."""
import numpy as np
import pytest

from imspy_simulation.timsim.jobs.rt import _identity_uniform, _beta_hats, _SIGMA_ALPHA, _SIGMA_BETA, _K_ALPHA, _K_BETA


def test_identity_keyed_is_reproducible_and_order_independent():
    # Same peptide -> same draw, regardless of what else is present or the order.
    seqs = ["PEPTIDEK", "SAMPLERK", "ELVISLIVESK"]
    s1, k1 = _beta_hats(seqs)
    s2, k2 = _beta_hats(list(reversed(seqs)))
    assert s1[0] == pytest.approx(s2[2])  # PEPTIDEK's sigma_hat is the same in both orders
    assert k1[1] == pytest.approx(k2[1])  # SAMPLERK's k_hat is stable


def test_sigma_and_k_use_distinct_salts():
    # sigma and k must be independent draws for the same peptide, not the same number.
    assert _identity_uniform("PEPTIDEK", "rt_sigma") != _identity_uniform("PEPTIDEK", "rt_k")


def test_hats_are_in_unit_interval_and_match_v1_beta_distributions():
    rng_seqs = [f"PEPTIDE{i}K" for i in range(4000)]
    sigma, k = _beta_hats(rng_seqs)
    assert sigma.min() >= 0 and sigma.max() <= 1
    assert k.min() >= 0 and k.max() <= 1
    # Beta(a,b) mean = a/(a+b). Match v1's defaults (sigma Beta(4,4)=0.5; k Beta(1,20)~0.0476).
    assert sigma.mean() == pytest.approx(_SIGMA_ALPHA / (_SIGMA_ALPHA + _SIGMA_BETA), abs=0.02)
    assert k.mean() == pytest.approx(_K_ALPHA / (_K_ALPHA + _K_BETA), abs=0.01)
