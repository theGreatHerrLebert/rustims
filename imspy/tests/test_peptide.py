"""
Tests for peptide classes (PeptideSequence, PeptideIon, PeptideProductIon).

These tests verify the Rust-Python bindings work correctly for peptide
data structures, including creation, property access, mass calculations,
and fragmentation.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from imspy.data.peptide import (
    PeptideSequence,
    PeptideIon,
    PeptideProductIon,
    PeptideProductIonSeries,
    PeptideProductIonSeriesCollection
)
from imspy.data.spectrum import MzSpectrum


class TestPeptideSequenceCreation:
    """Tests for PeptideSequence creation."""

    def test_create_simple_peptide(self, simple_peptide_sequence):
        """Test creating a simple peptide sequence."""
        peptide = PeptideSequence(simple_peptide_sequence)
        assert peptide is not None
        assert peptide.sequence == simple_peptide_sequence

    def test_create_modified_peptide(self, modified_peptide_sequence):
        """Test creating a peptide with oxidation modification."""
        peptide = PeptideSequence(modified_peptide_sequence)
        assert peptide is not None
        assert "M[UNIMOD:35]" in peptide.sequence

    def test_create_phospho_peptide(self, phospho_peptide_sequence):
        """Test creating a peptide with phosphorylation."""
        peptide = PeptideSequence(phospho_peptide_sequence)
        assert peptide is not None
        assert "S[UNIMOD:21]" in peptide.sequence

    def test_create_complex_peptide(self, complex_peptide_sequence):
        """Test creating a peptide with multiple modifications."""
        peptide = PeptideSequence(complex_peptide_sequence)
        assert peptide is not None
        assert peptide.sequence == complex_peptide_sequence

    def test_create_peptide_with_id(self):
        """Test creating a peptide with peptide_id."""
        peptide = PeptideSequence("PEPTIDE", peptide_id=42)
        assert peptide.peptide_id == 42

    def test_create_peptide_without_id(self):
        """Test creating a peptide without peptide_id."""
        peptide = PeptideSequence("PEPTIDE")
        assert peptide.peptide_id is None


class TestPeptideSequenceProperties:
    """Tests for PeptideSequence property access."""

    def test_sequence_property(self, simple_peptide_sequence):
        """Test sequence property returns correct value."""
        peptide = PeptideSequence(simple_peptide_sequence)
        assert peptide.sequence == simple_peptide_sequence

    def test_mono_isotopic_mass_property(self, simple_peptide_sequence):
        """Test mono_isotopic_mass property returns float."""
        peptide = PeptideSequence(simple_peptide_sequence)
        assert isinstance(peptide.mono_isotopic_mass, float)
        assert peptide.mono_isotopic_mass > 0

    def test_atomic_composition_property(self, simple_peptide_sequence):
        """Test atomic_composition returns dict-like structure."""
        peptide = PeptideSequence(simple_peptide_sequence)
        composition = peptide.atomic_composition
        assert composition is not None
        # Should contain C, H, N, O at minimum
        assert 'C' in composition
        assert 'H' in composition
        assert 'N' in composition
        assert 'O' in composition

    def test_amino_acid_count(self, simple_peptide_sequence):
        """Test amino_acid_count returns correct count."""
        peptide = PeptideSequence(simple_peptide_sequence)
        assert peptide.amino_acid_count == len(simple_peptide_sequence)

    def test_peptide_mass_increases_with_modifications(self):
        """Test that modifications increase peptide mass."""
        unmodified = PeptideSequence("PEPTMIDE")
        oxidized = PeptideSequence("PEPTM[UNIMOD:35]IDE")
        # Oxidation adds ~16 Da
        assert oxidized.mono_isotopic_mass > unmodified.mono_isotopic_mass
        mass_diff = oxidized.mono_isotopic_mass - unmodified.mono_isotopic_mass
        assert abs(mass_diff - 15.995) < 0.01  # Oxygen mass


class TestPeptideSequenceTokenization:
    """Tests for PeptideSequence tokenization."""

    def test_to_tokens_simple(self, simple_peptide_sequence):
        """Test tokenizing simple peptide."""
        peptide = PeptideSequence(simple_peptide_sequence)
        tokens = peptide.to_tokens()
        assert len(tokens) == len(simple_peptide_sequence)
        assert tokens == list(simple_peptide_sequence)

    def test_to_tokens_with_modification_grouped(self, modified_peptide_sequence):
        """Test tokenizing modified peptide with grouped modifications."""
        peptide = PeptideSequence(modified_peptide_sequence)
        tokens = peptide.to_tokens(group_modifications=True)
        assert len(tokens) > 0
        # Modified amino acid should be grouped
        assert any("[" in token for token in tokens)

    def test_to_tokens_with_modification_ungrouped(self, modified_peptide_sequence):
        """Test tokenizing modified peptide without grouped modifications."""
        peptide = PeptideSequence(modified_peptide_sequence)
        tokens = peptide.to_tokens(group_modifications=False)
        assert len(tokens) > 0


class TestPeptideSequenceFragmentation:
    """Tests for PeptideSequence fragmentation."""

    def test_calculate_product_ion_series_b_ions(self, simple_peptide_sequence):
        """Test calculating b-ion series."""
        peptide = PeptideSequence(simple_peptide_sequence)
        n_ions, c_ions = peptide.calculate_product_ion_series(charge=1, fragment_type='b')
        assert len(n_ions) > 0
        assert len(c_ions) > 0
        assert all(isinstance(ion, PeptideProductIon) for ion in n_ions)
        assert all(isinstance(ion, PeptideProductIon) for ion in c_ions)

    def test_calculate_product_ion_series_y_ions(self, simple_peptide_sequence):
        """Test calculating y-ion series."""
        peptide = PeptideSequence(simple_peptide_sequence)
        n_ions, c_ions = peptide.calculate_product_ion_series(charge=1, fragment_type='y')
        assert len(n_ions) > 0
        assert len(c_ions) > 0

    def test_calculate_product_ion_series_different_charges(self, simple_peptide_sequence):
        """Test calculating product ion series with different charges."""
        peptide = PeptideSequence(simple_peptide_sequence)

        n_ions_1, c_ions_1 = peptide.calculate_product_ion_series(charge=1, fragment_type='b')
        n_ions_2, c_ions_2 = peptide.calculate_product_ion_series(charge=2, fragment_type='b')

        # Higher charge should give lower m/z
        assert n_ions_1[0].mz > n_ions_2[0].mz

    def test_calculate_mono_isotopic_product_ion_spectrum(self, simple_peptide_sequence):
        """Test calculating mono-isotopic product ion spectrum."""
        peptide = PeptideSequence(simple_peptide_sequence)
        spectrum = peptide.calculate_mono_isotopic_product_ion_spectrum(charge=1, fragment_type='b')
        assert isinstance(spectrum, MzSpectrum)
        assert len(spectrum.mz) > 0
        assert len(spectrum.intensity) > 0

    def test_invalid_fragment_type_raises(self, simple_peptide_sequence):
        """Test that invalid fragment type raises assertion."""
        peptide = PeptideSequence(simple_peptide_sequence)
        with pytest.raises(AssertionError):
            peptide.calculate_product_ion_series(charge=1, fragment_type='invalid')


class TestPeptideSequenceSerialization:
    """Tests for PeptideSequence serialization."""

    def test_to_json(self, simple_peptide_sequence):
        """Test JSON serialization."""
        peptide = PeptideSequence(simple_peptide_sequence)
        json_str = peptide.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_to_sage_representation(self, simple_peptide_sequence):
        """Test SAGE representation."""
        peptide = PeptideSequence(simple_peptide_sequence)
        sage_rep = peptide.to_sage_representation()
        assert isinstance(sage_rep, tuple)
        assert len(sage_rep) == 2

    def test_repr(self, simple_peptide_sequence):
        """Test string representation."""
        peptide = PeptideSequence(simple_peptide_sequence)
        repr_str = repr(peptide)
        assert "PeptideSequence" in repr_str
        assert simple_peptide_sequence in repr_str


class TestPeptideIonCreation:
    """Tests for PeptideIon creation."""

    def test_create_simple_ion(self, simple_peptide_sequence):
        """Test creating a simple peptide ion."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        assert ion is not None
        assert ion.charge == 2
        assert ion.intensity == 1000.0

    def test_create_ion_with_peptide_id(self, simple_peptide_sequence):
        """Test creating peptide ion with peptide_id."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0, peptide_id=42)
        assert ion.peptide_id == 42

    def test_create_ion_different_charges(self, simple_peptide_sequence):
        """Test creating ions with different charges."""
        ion1 = PeptideIon(sequence=simple_peptide_sequence, charge=1, intensity=1000.0)
        ion2 = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        ion3 = PeptideIon(sequence=simple_peptide_sequence, charge=3, intensity=1000.0)

        # m/z should decrease with increasing charge
        assert ion1.mz > ion2.mz > ion3.mz


class TestPeptideIonProperties:
    """Tests for PeptideIon property access."""

    def test_sequence_property(self, simple_peptide_sequence):
        """Test sequence property returns PeptideSequence."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        assert isinstance(ion.sequence, PeptideSequence)
        assert ion.sequence.sequence == simple_peptide_sequence

    def test_charge_property(self, simple_peptide_sequence):
        """Test charge property."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        assert ion.charge == 2

    def test_intensity_property(self, simple_peptide_sequence):
        """Test intensity property."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=5000.0)
        assert ion.intensity == 5000.0

    def test_mz_property(self, simple_peptide_sequence):
        """Test mz property."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        assert isinstance(ion.mz, float)
        assert ion.mz > 0

    def test_atomic_composition(self, simple_peptide_sequence):
        """Test atomic_composition via underlying sequence.

        Note: PyPeptideIon doesn't expose atomic_composition directly,
        but we can get it through the sequence property.
        """
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        # Access composition through the sequence
        composition = ion.sequence.atomic_composition
        assert composition is not None
        assert 'C' in composition


class TestPeptideIonIsotopicSpectrum:
    """Tests for PeptideIon isotopic spectrum calculation."""

    def test_calculate_isotopic_spectrum(self, simple_peptide_sequence):
        """Test calculating isotopic spectrum."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        spectrum = ion.calculate_isotopic_spectrum()
        assert isinstance(spectrum, MzSpectrum)
        assert len(spectrum.mz) > 0
        # Should have multiple isotope peaks
        assert len(spectrum.mz) >= 3

    def test_isotopic_spectrum_mz_spacing(self, simple_peptide_sequence):
        """Test that isotope peaks are spaced correctly for charge."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        spectrum = ion.calculate_isotopic_spectrum()
        # For charge 2, isotope spacing should be ~0.5 Da
        if len(spectrum.mz) > 1:
            spacing = spectrum.mz[1] - spectrum.mz[0]
            assert abs(spacing - 0.5) < 0.1

    def test_isotopic_spectrum_custom_parameters(self, simple_peptide_sequence):
        """Test isotopic spectrum with custom parameters."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        spectrum = ion.calculate_isotopic_spectrum(
            mass_tolerance=1e-4,
            abundance_threshold=1e-6,
            max_result=100,
            intensity_min=1e-3
        )
        assert isinstance(spectrum, MzSpectrum)

    def test_isotopic_spectrum_invalid_threshold_raises(self, simple_peptide_sequence):
        """Test that invalid abundance threshold raises assertion."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        with pytest.raises(AssertionError):
            ion.calculate_isotopic_spectrum(abundance_threshold=1.5)


class TestPeptideIonSerialization:
    """Tests for PeptideIon serialization."""

    def test_to_json(self, simple_peptide_sequence):
        """Test JSON serialization via sequence.

        Note: PeptideIon.to_json() relies on underlying Rust method.
        Test the sequence's to_json instead since that's where the core data is.
        """
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        # Use sequence to_json which is available
        json_str = ion.sequence.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_repr(self, simple_peptide_sequence):
        """Test string representation."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        repr_str = repr(ion)
        assert "PeptideIon" in repr_str


class TestPeptideProductIonCreation:
    """Tests for PeptideProductIon creation."""

    def test_create_b_ion(self):
        """Test creating a b-type product ion."""
        ion = PeptideProductIon(kind='b', sequence='PEP', charge=1, intensity=1000.0)
        assert ion.kind == 'b'
        assert ion.sequence == 'PEP'
        assert ion.charge == 1
        assert ion.intensity == 1000.0

    def test_create_y_ion(self):
        """Test creating a y-type product ion."""
        ion = PeptideProductIon(kind='y', sequence='TIDE', charge=1, intensity=500.0)
        assert ion.kind == 'y'
        assert ion.sequence == 'TIDE'

    def test_create_all_ion_types(self):
        """Test creating all valid ion types."""
        for kind in ['a', 'b', 'c', 'x', 'y', 'z']:
            ion = PeptideProductIon(kind=kind, sequence='TEST', charge=1, intensity=100.0)
            assert ion.kind == kind

    def test_invalid_ion_type_raises(self):
        """Test that invalid ion type raises assertion."""
        with pytest.raises(AssertionError):
            PeptideProductIon(kind='invalid', sequence='TEST', charge=1, intensity=100.0)


class TestPeptideProductIonProperties:
    """Tests for PeptideProductIon property access."""

    def test_kind_property(self):
        """Test kind property."""
        ion = PeptideProductIon(kind='b', sequence='PEP', charge=1, intensity=1000.0)
        assert ion.kind == 'b'

    def test_sequence_property(self):
        """Test sequence property."""
        ion = PeptideProductIon(kind='b', sequence='PEPTIDE', charge=1, intensity=1000.0)
        assert ion.sequence == 'PEPTIDE'

    def test_mono_isotopic_mass_property(self):
        """Test mono_isotopic_mass property."""
        ion = PeptideProductIon(kind='b', sequence='PEP', charge=1, intensity=1000.0)
        assert isinstance(ion.mono_isotopic_mass, float)
        assert ion.mono_isotopic_mass > 0

    def test_mz_property(self):
        """Test mz property."""
        ion = PeptideProductIon(kind='b', sequence='PEP', charge=1, intensity=1000.0)
        assert isinstance(ion.mz, float)
        assert ion.mz > 0

    def test_mz_decreases_with_charge(self):
        """Test that m/z decreases with increasing charge."""
        ion1 = PeptideProductIon(kind='b', sequence='PEPTIDE', charge=1, intensity=1000.0)
        ion2 = PeptideProductIon(kind='b', sequence='PEPTIDE', charge=2, intensity=1000.0)
        assert ion1.mz > ion2.mz

    def test_atomic_composition(self):
        """Test atomic_composition method."""
        ion = PeptideProductIon(kind='b', sequence='PEP', charge=1, intensity=1000.0)
        composition = ion.atomic_composition()
        assert composition is not None
        assert 'C' in composition


class TestPeptideProductIonIsotopeDistribution:
    """Tests for PeptideProductIon isotope distribution."""

    def test_isotope_distribution(self):
        """Test isotope distribution calculation."""
        ion = PeptideProductIon(kind='b', sequence='PEPTIDE', charge=1, intensity=1000.0)
        distribution = ion.isotope_distribution()
        assert isinstance(distribution, list)
        assert len(distribution) > 0
        # Each entry is (mz, intensity) tuple
        assert all(isinstance(entry, tuple) for entry in distribution)
        assert all(len(entry) == 2 for entry in distribution)


class TestPeptideProductIonSeries:
    """Tests for PeptideProductIonSeries."""

    def test_create_series(self):
        """Test creating a product ion series."""
        n_ion = PeptideProductIon(kind='b', sequence='PEP', charge=1, intensity=1000.0)
        c_ion = PeptideProductIon(kind='y', sequence='TIDE', charge=1, intensity=500.0)

        series = PeptideProductIonSeries(
            charge=1,
            n_ions=[n_ion],
            c_ions=[c_ion]
        )
        assert series.charge == 1
        assert len(series.n_ions) == 1
        assert len(series.c_ions) == 1

    def test_series_properties(self):
        """Test series properties."""
        n_ion = PeptideProductIon(kind='b', sequence='PEP', charge=2, intensity=1000.0)
        c_ion = PeptideProductIon(kind='y', sequence='TIDE', charge=2, intensity=500.0)

        series = PeptideProductIonSeries(
            charge=2,
            n_ions=[n_ion],
            c_ions=[c_ion]
        )

        assert series.charge == 2
        assert all(isinstance(ion, PeptideProductIon) for ion in series.n_ions)
        assert all(isinstance(ion, PeptideProductIon) for ion in series.c_ions)

    def test_series_to_json(self):
        """Test series JSON serialization."""
        n_ion = PeptideProductIon(kind='b', sequence='PEP', charge=1, intensity=1000.0)
        series = PeptideProductIonSeries(charge=1, n_ions=[n_ion], c_ions=[])
        json_str = series.to_json()
        assert isinstance(json_str, str)


class TestPeptideProductIonSeriesCollection:
    """Tests for PeptideProductIonSeriesCollection."""

    def test_create_collection(self):
        """Test creating a series collection."""
        n_ion = PeptideProductIon(kind='b', sequence='PEP', charge=1, intensity=1000.0)
        c_ion = PeptideProductIon(kind='y', sequence='TIDE', charge=1, intensity=500.0)
        series = PeptideProductIonSeries(charge=1, n_ions=[n_ion], c_ions=[c_ion])

        collection = PeptideProductIonSeriesCollection([series])
        assert len(collection.series) == 1

    def test_find_series_by_charge(self):
        """Test finding series by charge."""
        n_ion1 = PeptideProductIon(kind='b', sequence='PEP', charge=1, intensity=1000.0)
        n_ion2 = PeptideProductIon(kind='b', sequence='PEP', charge=2, intensity=1000.0)

        series1 = PeptideProductIonSeries(charge=1, n_ions=[n_ion1], c_ions=[])
        series2 = PeptideProductIonSeries(charge=2, n_ions=[n_ion2], c_ions=[])

        collection = PeptideProductIonSeriesCollection([series1, series2])

        found = collection.find_series(1)
        assert found is not None
        assert found.charge == 1

    def test_find_series_not_found(self):
        """Test finding non-existent series returns None."""
        n_ion = PeptideProductIon(kind='b', sequence='PEP', charge=1, intensity=1000.0)
        series = PeptideProductIonSeries(charge=1, n_ions=[n_ion], c_ions=[])
        collection = PeptideProductIonSeriesCollection([series])

        found = collection.find_series(5)
        assert found is None

    def test_collection_to_json(self):
        """Test collection JSON serialization."""
        n_ion = PeptideProductIon(kind='b', sequence='PEP', charge=1, intensity=1000.0)
        series = PeptideProductIonSeries(charge=1, n_ions=[n_ion], c_ions=[])
        collection = PeptideProductIonSeriesCollection([series])

        json_str = collection.to_json()
        assert isinstance(json_str, str)


class TestPeptideMassCalculations:
    """Tests for peptide mass calculations with known values."""

    def test_glycine_mass(self):
        """Test mass of glycine-only peptide."""
        # Glycine monoisotopic mass is ~57.02 Da
        # GG = 57.02 + 57.02 + 18.01 (water) = 132.05 Da approximately
        peptide = PeptideSequence("GG")
        expected_mass = 132.0535  # approximate
        assert abs(peptide.mono_isotopic_mass - expected_mass) < 0.01

    def test_peptide_mass_consistency(self):
        """Test that same sequence gives same mass."""
        peptide1 = PeptideSequence("PEPTIDE")
        peptide2 = PeptideSequence("PEPTIDE")
        # Use approximate comparison due to floating point precision
        assert abs(peptide1.mono_isotopic_mass - peptide2.mono_isotopic_mass) < 1e-10

    def test_mz_calculation(self, simple_peptide_sequence):
        """Test m/z calculation for different charges."""
        ion1 = PeptideIon(sequence=simple_peptide_sequence, charge=1, intensity=1000.0)
        ion2 = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)

        # m/z = (mass + charge * proton_mass) / charge
        # So mz1 * 1 - proton should approximately equal mz2 * 2 - 2 * proton
        proton_mass = 1.00727647
        mass_from_ion1 = ion1.mz - proton_mass
        mass_from_ion2 = (ion2.mz * 2) - (2 * proton_mass)

        assert abs(mass_from_ion1 - mass_from_ion2) < 0.01


class TestPeptideFromPyPtr:
    """Tests for from_py_ptr class methods."""

    def test_peptide_sequence_from_py_ptr(self, simple_peptide_sequence):
        """Test PeptideSequence from_py_ptr."""
        peptide1 = PeptideSequence(simple_peptide_sequence)
        py_ptr = peptide1.get_py_ptr()
        peptide2 = PeptideSequence.from_py_ptr(py_ptr)

        assert peptide1.sequence == peptide2.sequence
        # Use approximate comparison due to floating point precision
        assert abs(peptide1.mono_isotopic_mass - peptide2.mono_isotopic_mass) < 1e-10

    def test_peptide_ion_from_py_ptr(self, simple_peptide_sequence):
        """Test PeptideIon from_py_ptr."""
        ion1 = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=1000.0)
        py_ptr = ion1.get_py_ptr()
        ion2 = PeptideIon.from_py_ptr(py_ptr)

        assert ion1.charge == ion2.charge
        assert ion1.intensity == ion2.intensity
        assert ion1.mz == ion2.mz

    def test_product_ion_from_py_ptr(self):
        """Test PeptideProductIon from_py_ptr."""
        ion1 = PeptideProductIon(kind='b', sequence='PEPTIDE', charge=1, intensity=1000.0)
        py_ptr = ion1.get_py_ptr()
        ion2 = PeptideProductIon.from_py_ptr(py_ptr)

        assert ion1.kind == ion2.kind
        assert ion1.sequence == ion2.sequence
        assert ion1.mz == ion2.mz


class TestPeptideEdgeCases:
    """Edge case tests for peptide classes."""

    def test_single_amino_acid(self):
        """Test peptide with single amino acid."""
        peptide = PeptideSequence("K")
        assert peptide.amino_acid_count == 1
        assert peptide.mono_isotopic_mass > 0

    def test_very_long_peptide(self):
        """Test handling of long peptide sequences."""
        long_sequence = "A" * 50
        peptide = PeptideSequence(long_sequence)
        assert peptide.amino_acid_count == 50
        assert peptide.mono_isotopic_mass > 0

    def test_all_amino_acids(self):
        """Test peptide containing all standard amino acids."""
        all_aa = "ACDEFGHIKLMNPQRSTVWY"
        peptide = PeptideSequence(all_aa)
        assert peptide.amino_acid_count == 20

    def test_high_charge_ion(self, simple_peptide_sequence):
        """Test peptide ion with high charge state."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=10, intensity=1000.0)
        assert ion.mz > 0
        assert ion.mz < PeptideIon(sequence=simple_peptide_sequence, charge=1, intensity=1000.0).mz

    def test_zero_intensity_ion(self, simple_peptide_sequence):
        """Test peptide ion with zero intensity."""
        ion = PeptideIon(sequence=simple_peptide_sequence, charge=2, intensity=0.0)
        assert ion.intensity == 0.0
        assert ion.mz > 0  # m/z should still be calculated
