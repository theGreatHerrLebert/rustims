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


def test_thinning_happens_before_retention_time_prediction():
    """The expensive predictor must see only the peptides we intend to keep.

    The digest produces `num_peptides_total` regardless of the requested complexity, so predicting
    retention times for the whole digest meant a small run spent most of its time on peptides it
    then discarded. Thinning first must still deliver at least the requested count, since the
    retention-time filter drops part of the early-eluting population afterwards.
    """
    import inspect
    from imspy_simulation.timsim.jobs import simulate_peptides as mod

    src = inspect.getsource(mod.simulate_peptides)
    thin = src.index("peptide_table.sample(")
    predict = src.index("simulate_separation_times_pandas")
    assert thin < predict, "thinning must precede retention-time prediction"

    sig = inspect.signature(mod.simulate_peptides).parameters
    assert "num_sample_peptides" in sig and "sample_margin" in sig
    # Headroom must be real, or the retention-time filter can leave us short of the request.
    assert sig["sample_margin"].default > 1.0
