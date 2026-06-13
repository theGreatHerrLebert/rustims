"""Phase 0 streaming: batching the EMG frame-distribution projection must not change
its output.

The per-(peptide, frame) EMG projection is deterministic and peptide-local, so
chunking the projection (to bound peak memory on dense-frame templates like the Astral
nDIA grid, ~2000 frames/peptide) must be output-equivalent to a single full-array call:

- noise OFF (the default): byte-identical occurrence + abundance.
- noise ON: also byte-identical, PROVIDED both RNGs are seeded. `add_uniform_noise`
  is Numba-jitted and draws Numba's RNG (which `np.random.seed` does NOT reset — a
  jitted seeder does). Batching preserves peptide-iteration order, so with the same
  Numba RNG state the noise draws are identical; the apparent non-determinism without
  a Numba seed is Numba's un-seeded stream, not a batching effect.
"""
import json

import numpy as np
import pandas as pd
from numba import njit

from imspy_simulation.timsim.jobs.simulate_frame_distributions_emg import (
    simulate_frame_distributions_emg,
)


@njit
def _seed_numba(s):  # Python np.random.seed does not reset Numba's nopython RNG
    np.random.seed(s)

# Dense-frame fixture mimicking the Astral nDIA template (many frames / short gradient).
_F, _T, _N = 6000, 300.0, 1500
_FRAMES = pd.DataFrame({"frame_id": np.arange(1, _F + 1), "time": np.linspace(0, _T, _F)})
_KW = dict(
    frames=_FRAMES, sigma_lower_rt=None, sigma_upper_rt=None, sigma_alpha_rt=4,
    sigma_beta_rt=4, k_lower_rt=0, k_upper_rt=10, k_alpha_rt=1, k_beta_rt=20,
    target_p=0.999, step_size=0.001, rt_cycle_length=_T / _F, n_steps=1000,
    num_threads=4, gradient_length=_T, remove_epsilon=1e-4, verbose=False,
)


def _peptides():
    rt = np.random.RandomState(7).uniform(5, _T - 5, _N)
    return pd.DataFrame({"peptide_id": np.arange(_N), "retention_time_gru_predictor": rt})


def _run(batch_size, add_noise):
    np.random.seed(123)   # fix sigma/k/mu sampling + Python-RNG draws
    _seed_numba(123)      # fix add_uniform_noise's Numba RNG so only batching differs
    out = simulate_frame_distributions_emg(
        peptides=_peptides(), add_noise=add_noise, batch_size=batch_size, **_KW
    )
    return out.sort_values("peptide_id").reset_index(drop=True)


def _assert_byte_identical(full, chunked):
    assert len(full) == len(chunked)
    for col in ("frame_occurrence", "frame_abundance",
                "frame_occurrence_start", "frame_occurrence_end"):
        assert full[col].tolist() == chunked[col].tolist(), col


def test_chunking_byte_identical_noise_off():
    _assert_byte_identical(_run(batch_size=None, add_noise=False),
                           _run(batch_size=200, add_noise=False))


def test_chunking_byte_identical_noise_on():
    # With both RNGs seeded, batching is byte-identical even with noise (peptide order
    # is preserved, so Numba's per-frame noise stream is identical).
    _assert_byte_identical(_run(batch_size=None, add_noise=True),
                           _run(batch_size=200, add_noise=True))


def test_zero_peptides_does_not_crash():
    np.random.seed(0)
    empty = pd.DataFrame({"peptide_id": pd.Series([], dtype=int),
                          "retention_time_gru_predictor": pd.Series([], dtype=float)})
    out = simulate_frame_distributions_emg(peptides=empty, add_noise=False, batch_size=200, **_KW)
    assert len(out) == 0


def test_nonpositive_batch_size_rejected():
    import pytest
    with pytest.raises(ValueError):
        simulate_frame_distributions_emg(peptides=_peptides(), add_noise=False, batch_size=0, **_KW)
