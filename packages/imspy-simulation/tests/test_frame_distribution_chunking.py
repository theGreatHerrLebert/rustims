"""Phase 0 streaming: batching the EMG frame-distribution projection must not change
its output.

The per-(peptide, frame) EMG projection is deterministic and peptide-local, so
chunking the projection (to bound peak memory on dense-frame templates like the Astral
nDIA grid, ~2000 frames/peptide) must be output-equivalent to a single full-array call:

- noise OFF (the default): byte-identical occurrence + abundance.
- noise ON: structurally identical (same occurrence, same per-peptide length, same
  normalization) but a different random noise realization — `add_uniform_noise` draws
  np.random per frame, so reordering the stream changes values (statistical
  equivalence, not byte parity — by design).
"""
import json

import numpy as np
import pandas as pd

from imspy_simulation.timsim.jobs.simulate_frame_distributions_emg import (
    simulate_frame_distributions_emg,
)

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
    np.random.seed(123)  # fix sigma/k/mu sampling so only batching differs
    out = simulate_frame_distributions_emg(
        peptides=_peptides(), add_noise=add_noise, batch_size=batch_size, **_KW
    )
    return out.sort_values("peptide_id").reset_index(drop=True)


def test_chunking_byte_identical_noise_off():
    full = _run(batch_size=None, add_noise=False)
    chunked = _run(batch_size=200, add_noise=False)
    assert len(full) == len(chunked)
    assert full["frame_occurrence"].tolist() == chunked["frame_occurrence"].tolist()
    assert full["frame_abundance"].tolist() == chunked["frame_abundance"].tolist()
    assert full["frame_occurrence_start"].tolist() == chunked["frame_occurrence_start"].tolist()
    assert full["frame_occurrence_end"].tolist() == chunked["frame_occurrence_end"].tolist()


def test_chunking_structurally_equivalent_noise_on():
    full = _run(batch_size=None, add_noise=True)
    chunked = _run(batch_size=200, add_noise=True)
    assert len(full) == len(chunked)
    # occurrence (frame membership) is deterministic -> identical even with noise
    assert full["frame_occurrence"].tolist() == chunked["frame_occurrence"].tolist()
    fa, fb = full["frame_abundance"].tolist(), chunked["frame_abundance"].tolist()
    for sa, sb in zip(fa, fb):
        va, vb = np.array(json.loads(sa)), np.array(json.loads(sb))
        assert len(va) == len(vb)                       # same structure
        # both normalized to ~1; values are different random noise realizations, so
        # they do NOT match each other tightly (statistical equivalence, by design).
        assert abs(va.sum() - 1.0) < 0.05
        assert abs(vb.sum() - 1.0) < 0.05
