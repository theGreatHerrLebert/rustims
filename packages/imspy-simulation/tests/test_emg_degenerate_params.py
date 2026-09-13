"""The EMG elution model must not emit NaN for degenerate parameter draws.

`np.random.beta` returns exactly 0.0 often enough to matter, and `k_lower_rt` defaults to 0, so a
zero sigma or lambda was reachable. `estimate_mu_from_mode_emg` divides by their product, which made
the erfcx target infinite, slipped past the positivity assert, and produced a NaN mu. That NaN
reached the Rust frame-occurrence search and panicked a worker thread mid-run.
"""
import numpy as np
import pytest

from imspy_simulation.timsim.jobs.simulate_frame_distributions_emg import (
    erfcxinv, estimate_mu_from_mode_emg, sample_sigma_k_emg, sample_sigma_lambda_emg,
)
from scipy.special import erfcx


@pytest.mark.parametrize("y", [1e-12, 1e-10, 1e-8, 1e-6, 1e-3, 0.1, 1.0, 2.0, 10.0, 1e3])
def test_erfcxinv_is_finite_and_inverts_erfcx(y):
    x = erfcxinv(np.array([y]))[0]
    assert np.isfinite(x), f"erfcxinv({y}) is not finite"
    # erfcx overflows for very negative x, so only round-trip where that is representable.
    back = erfcx(x)
    if np.isfinite(back) and back > 0:
        assert abs(back - y) / y < 1e-6, f"erfcxinv({y}) -> {x} -> {back}"


def test_erfcxinv_rejects_a_degenerate_target():
    """An infinite target means a zero sigma or lambda got through; fail loudly, never silently NaN."""
    with pytest.raises(AssertionError):
        erfcxinv(np.array([np.inf]))


@pytest.mark.parametrize("sampler,second", [(sample_sigma_k_emg, "k"), (sample_sigma_lambda_emg, "lambda")])
def test_sampled_parameters_are_strictly_positive(sampler, second):
    """The draw must never hand a zero downstream, even with a zero lower bound (k_lower_rt is 0).

    Beta parameters are chosen to make a 0.0 draw likely: a small alpha underflows to exactly zero.
    """
    np.random.seed(0)
    kw = dict(sigma_lower=0.0, sigma_upper=1.0, sigma_alpha=0.01, sigma_beta=10.0, n=50000)
    kw.update({f"{second}_lower": 0.0, f"{second}_upper": 10.0,
               f"{second}_alpha": 0.01, f"{second}_beta": 10.0})
    sigmas, others = sampler(**kw)
    assert (sigmas > 0).all(), f"{(sigmas <= 0).sum()} sigma draws were non-positive"
    assert (others > 0).all(), f"{(others <= 0).sum()} {second} draws were non-positive"
    mu = estimate_mu_from_mode_emg(np.full_like(sigmas, 10.0), sigmas, others)
    assert np.isfinite(mu).all(), f"{(~np.isfinite(mu)).sum()} non-finite mu values"
