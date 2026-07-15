"""CCS as an instrument-independent ion property, and 1/K0 as a per-run measurement.

These are the scientific claims of the mobility port, asserted rather than described:

1. **Lossless extraction.** timsim-ccs recovers CCS by inverting the deep model's own 1/K0 at the
   model's gas defaults. Mason-Schamp is exactly invertible, so round-tripping back to 1/K0 at those
   same defaults must reproduce the model's 1/K0 to machine precision. This is the parity guarantee:
   with the default gas, the CCS path IS v1's mobility, exactly.

2. **Cross-instrument.** The same CCS, converted with a different drift gas, gives a different 1/K0 —
   and heavier gas gives *higher* 1/K0 (lower mobility). This is the capability v1 could not express,
   because it discarded CCS and kept only N2-at-305K 1/K0.
"""

import numpy as np
import pytest

pytest.importorskip("imspy_core")
from imspy_core.chemistry import ccs_to_one_over_k0, one_over_k0_to_ccs  # noqa: E402

# The gas defaults timsim-ccs inverts at — must match jobs/ccs.py.
_GAS_N2 = 28.013
_TEMP = 31.85
_GAS_AR = 39.948

# A spread of realistic precursors: (mz, charge).
_IONS = [(400.0, 2), (500.0, 2), (350.0, 3), (700.0, 2), (280.0, 3), (900.0, 4)]


def test_ccs_extraction_is_lossless():
    """1/K0 -> CCS -> 1/K0 at the same gas is exact. This is why recovering CCS from the model's
    default-gas 1/K0 loses nothing."""
    for mz, z in _IONS:
        k0 = 0.9  # any starting mobility
        ccs = one_over_k0_to_ccs(k0, mz, z, mass_gas=_GAS_N2, temp=_TEMP)
        k0_back = ccs_to_one_over_k0(ccs, mz, z, mass_gas=_GAS_N2, temp=_TEMP)
        assert abs(k0_back - k0) < 1e-12, f"round-trip drifted at mz={mz}, z={z}"


def test_default_gas_reproduces_the_measured_mobility():
    """The parity guarantee. If the CCS came from inverting a 1/K0 at the default gas, converting it
    back at the default gas returns that exact 1/K0 — so the CCS path does not change v1's numbers
    unless the instrument changes."""
    for mz, z in _IONS:
        measured_k0 = 1.05
        ccs = one_over_k0_to_ccs(measured_k0, mz, z, mass_gas=_GAS_N2, temp=_TEMP)
        rederived = ccs_to_one_over_k0(ccs, mz, z, mass_gas=_GAS_N2, temp=_TEMP)
        assert rederived == pytest.approx(measured_k0, abs=1e-12)


def test_heavier_gas_raises_inverse_mobility():
    """Cross-instrument: the SAME CCS, a heavier gas, a different (higher) 1/K0. Reduced mass goes
    up, mobility goes down, 1/K0 goes up — and every ion must move the same way."""
    factors = []
    for mz, z in _IONS:
        ccs = one_over_k0_to_ccs(1.0, mz, z, mass_gas=_GAS_N2, temp=_TEMP)
        k0_n2 = ccs_to_one_over_k0(ccs, mz, z, mass_gas=_GAS_N2, temp=_TEMP)
        k0_ar = ccs_to_one_over_k0(ccs, mz, z, mass_gas=_GAS_AR, temp=_TEMP)
        assert k0_ar > k0_n2, f"heavier gas must raise 1/K0 (mz={mz}, z={z})"
        factors.append(k0_ar / k0_n2)
    # The shift is a coherent instrument change, not noise: every ion moves by a similar factor.
    assert np.std(factors) < 0.1 * np.mean(factors)


def test_timsim_ccs_predict_roundtrips_through_the_model():
    """End-to-end through the real deep model: predicting CCS then converting back at the default
    gas reproduces the model's own 1/K0. Skipped if the model weights aren't installed."""
    pytest.importorskip("imspy_predictors")
    pd = pytest.importorskip("pandas")
    try:
        from imspy_predictors.ccs import DeepPeptideIonMobilityApex
    except Exception:
        pytest.skip("CCS predictor unavailable")

    from imspy_simulation.timsim.jobs.ccs import predict_ccs

    seqs = ["PEPTIDEK", "SAMPLERPEPTIDEK", "GHIYSTEPKR"]
    charges = [2, 2, 3]
    mzs = [400.0, 500.0, 350.0]

    predictor = DeepPeptideIonMobilityApex(verbose=False)
    try:
        k0_model = np.asarray(predictor.simulate_ion_mobilities(seqs, charges, mzs), dtype=float).flatten()
    except Exception:
        pytest.skip("CCS model weights not available in this environment")

    precursors = pd.DataFrame({
        "precursor_id": np.arange(len(seqs), dtype=np.uint64),
        "peptide_id": np.arange(len(seqs), dtype=np.uint64),
        "charge": charges,
        "mz": mzs,
    })
    peptides = pd.DataFrame({"peptide_id": np.arange(len(seqs), dtype=np.uint64), "sequence": seqs})
    ccs = predict_ccs(precursors, peptides, verbose=False)

    k0_rederived = np.array([
        ccs_to_one_over_k0(c, m, z, mass_gas=_GAS_N2, temp=_TEMP)
        for c, m, z in zip(ccs["ccs"], mzs, charges)
    ])
    assert np.max(np.abs(k0_rederived - k0_model)) < 1e-9
