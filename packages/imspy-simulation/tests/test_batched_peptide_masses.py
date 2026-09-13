"""The batched monoisotopic mass path must match the per-peptide one exactly.

Constructing a `PeptideSequence` per peptide cost ~190 us each, which made 250 000 peptides take
48 s and was the largest remaining serial cost in the pipeline. Two changes: the lookup tables and
the UniMod regex are built once rather than per call (9.4x on its own, and it helps every caller),
and the connector gained a batched parallel entry point. Mass values must not move.
"""
import numpy as np
import pytest

from imspy_core.data.peptide import PeptideSequence

_ims = pytest.importorskip("imspy_connector")
mono_isotopic_masses = getattr(getattr(_ims, "py_peptide", None), "mono_isotopic_masses", None)
if mono_isotopic_masses is None:  # pragma: no cover
    pytest.skip("connector without mono_isotopic_masses", allow_module_level=True)


def _peptides(n, seed=0):
    rng = np.random.default_rng(seed)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    return ["".join(rng.choice(aa, int(rng.integers(7, 31)))) for _ in range(n)]


@pytest.mark.parametrize("num_threads", [1, 4, 16])
def test_batched_masses_match_the_per_peptide_path(num_threads):
    peps = _peptides(2000)
    ref = [PeptideSequence(p).mono_isotopic_mass for p in peps]
    got = mono_isotopic_masses(peps, num_threads)
    assert got == ref, "batched masses differ from the per-peptide path"


def test_batched_masses_are_thread_count_invariant():
    """Bitwise, not approximately: float addition is not associative, so a parallel sum that
    varied with scheduling would make the simulation irreproducible."""
    peps = _peptides(5000, seed=7)
    assert mono_isotopic_masses(peps, 1) == mono_isotopic_masses(peps, 16)


def test_modifications_are_still_counted():
    """The UniMod regex moved into a static; modified peptides must still pick up their delta."""
    plain, mod = "PEPTIDEK", "PEPTIDEM[UNIMOD:35]K"
    m_plain, m_mod = mono_isotopic_masses([plain, mod], 1)
    assert m_mod > m_plain
    # UNIMOD:35 is oxidation, +15.994915, on top of the extra methionine residue.
    assert abs((m_mod - m_plain) - (PeptideSequence("PEPTIDEMK").mono_isotopic_mass
                                    - PeptideSequence(plain).mono_isotopic_mass) - 15.994915) < 1e-6


def test_empty_batch():
    assert mono_isotopic_masses([], 4) == []
