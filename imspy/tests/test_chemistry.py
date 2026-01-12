"""
Tests for chemistry modules (constants, elements, amino acids, sum formula, mobility).

These tests verify the Rust-Python bindings work correctly for chemistry
calculations, including mass constants, elemental data, amino acid properties,
sum formula parsing, and CCS/mobility conversions.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from imspy.chemistry.constants import (
    MASS_PROTON,
    MASS_NEUTRON,
    MASS_ELECTRON,
    MASS_WATER,
    STANDARD_TEMPERATURE,
    STANDARD_PRESSURE,
    ELEMENTARY_CHARGE,
    K_BOLTZMANN,
    AVOGADRO_NUMBER,
)
from imspy.chemistry.elements import (
    ELEMENTAL_MONO_ISOTOPIC_MASSES,
    ELEMENTAL_ISOTOPIC_MASSES,
    ELEMENTAL_ISOTOPIC_ABUNDANCES,
)
from imspy.chemistry.amino_acids import (
    AMINO_ACID_MASSES,
    AMINO_ACIDS,
    AMINO_ACID_ATOMIC_COMPOSITIONS,
)
from imspy.chemistry.sum_formula import SumFormula
from imspy.chemistry.mobility import (
    one_over_k0_to_ccs,
    ccs_to_one_over_k0,
    one_over_k0_to_ccs_par,
    ccs_to_one_over_k0_par,
)
from imspy.data.spectrum import MzSpectrum


class TestPhysicalConstants:
    """Tests for physical constants from Rust bindings."""

    def test_mass_proton(self, proton_mass):
        """Test proton mass is approximately correct."""
        # NIST value: 1.007276466621 u
        assert abs(MASS_PROTON - proton_mass) < 1e-4
        assert 1.007 < MASS_PROTON < 1.008

    def test_mass_neutron(self):
        """Test neutron mass is approximately correct."""
        # NIST value: 1.00866491595 u
        assert 1.008 < MASS_NEUTRON < 1.009

    def test_mass_electron(self):
        """Test electron mass is approximately correct."""
        # NIST value: 0.00054857990907 u
        assert 0.0005 < MASS_ELECTRON < 0.0006

    def test_mass_water(self, water_mass):
        """Test water mass is approximately correct."""
        # H2O: 2 * 1.007825 + 15.9949 ≈ 18.0106
        assert abs(MASS_WATER - water_mass) < 0.01
        assert 18.0 < MASS_WATER < 18.02

    def test_standard_temperature(self):
        """Test standard temperature is 273.15 K."""
        assert abs(STANDARD_TEMPERATURE - 273.15) < 0.01

    def test_standard_pressure(self):
        """Test standard pressure is 100000 Pa (100 kPa)."""
        # Note: Using 100 kPa instead of 101325 Pa (1 atm)
        assert abs(STANDARD_PRESSURE - 100000.0) < 1.0

    def test_elementary_charge(self):
        """Test elementary charge is approximately correct."""
        # NIST value: 1.602176634e-19 C
        assert 1.6e-19 < ELEMENTARY_CHARGE < 1.7e-19

    def test_k_boltzmann(self):
        """Test Boltzmann constant is approximately correct."""
        # NIST value: 1.380649e-23 J/K
        assert 1.38e-23 < K_BOLTZMANN < 1.39e-23

    def test_avogadro_number(self):
        """Test Avogadro's number is approximately correct."""
        # NIST value: 6.02214076e23
        assert 6.02e23 < AVOGADRO_NUMBER < 6.03e23


class TestElementalData:
    """Tests for elemental data from Rust bindings."""

    def test_elemental_masses_dict_structure(self):
        """Test that elemental masses returns a dict-like structure."""
        assert isinstance(ELEMENTAL_MONO_ISOTOPIC_MASSES, dict)
        assert len(ELEMENTAL_MONO_ISOTOPIC_MASSES) > 0

    def test_common_elements_present(self):
        """Test that common elements are present."""
        common_elements = ['H', 'C', 'N', 'O', 'S', 'P']
        for element in common_elements:
            assert element in ELEMENTAL_MONO_ISOTOPIC_MASSES

    def test_hydrogen_mass(self):
        """Test hydrogen monoisotopic mass."""
        # H: 1.00783 u approximately
        assert 1.007 < ELEMENTAL_MONO_ISOTOPIC_MASSES['H'] < 1.008

    def test_carbon_mass(self):
        """Test carbon monoisotopic mass."""
        # C: 12.0 u exactly (by definition)
        assert 11.99 < ELEMENTAL_MONO_ISOTOPIC_MASSES['C'] < 12.01

    def test_nitrogen_mass(self):
        """Test nitrogen monoisotopic mass."""
        # N: 14.003 u approximately
        assert 14.0 < ELEMENTAL_MONO_ISOTOPIC_MASSES['N'] < 14.01

    def test_oxygen_mass(self):
        """Test oxygen monoisotopic mass."""
        # O: 15.995 u approximately
        assert 15.99 < ELEMENTAL_MONO_ISOTOPIC_MASSES['O'] < 16.0

    def test_isotopic_masses_structure(self):
        """Test that isotopic masses returns proper structure."""
        assert isinstance(ELEMENTAL_ISOTOPIC_MASSES, dict)
        # Each element should map to list of isotope masses
        if 'C' in ELEMENTAL_ISOTOPIC_MASSES:
            assert isinstance(ELEMENTAL_ISOTOPIC_MASSES['C'], (list, tuple))

    def test_isotopic_abundances_structure(self):
        """Test that isotopic abundances returns proper structure."""
        assert isinstance(ELEMENTAL_ISOTOPIC_ABUNDANCES, dict)
        # Each element should map to list of abundances
        if 'C' in ELEMENTAL_ISOTOPIC_ABUNDANCES:
            abundances = ELEMENTAL_ISOTOPIC_ABUNDANCES['C']
            # Abundances should sum to approximately 1
            assert abs(sum(abundances) - 1.0) < 0.001


class TestAminoAcidData:
    """Tests for amino acid data from Rust bindings."""

    def test_amino_acid_masses_structure(self):
        """Test that amino acid masses returns proper structure."""
        assert isinstance(AMINO_ACID_MASSES, dict)
        # Should have ~20 standard amino acids
        assert len(AMINO_ACID_MASSES) >= 20

    def test_standard_amino_acids_present(self):
        """Test that all standard amino acids are present."""
        standard_aa = 'ACDEFGHIKLMNPQRSTVWY'
        for aa in standard_aa:
            assert aa in AMINO_ACID_MASSES

    def test_amino_acid_masses_reasonable(self, amino_acid_masses):
        """Test amino acid masses are in reasonable range."""
        for aa, expected_mass in amino_acid_masses.items():
            if aa in AMINO_ACID_MASSES:
                actual_mass = AMINO_ACID_MASSES[aa]
                # Allow some tolerance
                assert abs(actual_mass - expected_mass) < 0.01, f"Mass mismatch for {aa}"

    def test_glycine_is_smallest(self):
        """Test that glycine has the smallest mass."""
        glycine_mass = AMINO_ACID_MASSES['G']
        for aa, mass in AMINO_ACID_MASSES.items():
            if aa != 'G' and aa in 'ACDEFGHIKLMNPQRSTVWY':
                assert glycine_mass <= mass

    def test_tryptophan_is_largest(self):
        """Test that tryptophan has the largest mass among standard AAs."""
        trp_mass = AMINO_ACID_MASSES['W']
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            if aa in AMINO_ACID_MASSES:
                assert trp_mass >= AMINO_ACID_MASSES[aa]

    def test_amino_acid_list_structure(self):
        """Test AMINO_ACIDS returns proper structure (dict mapping name to code)."""
        assert isinstance(AMINO_ACIDS, dict)
        assert len(AMINO_ACIDS) >= 20
        # Values should be single-letter amino acid codes
        for name, code in AMINO_ACIDS.items():
            assert isinstance(name, str)
            assert isinstance(code, str)
            assert len(code) == 1

    def test_amino_acid_compositions_structure(self):
        """Test atomic compositions returns proper structure."""
        assert isinstance(AMINO_ACID_ATOMIC_COMPOSITIONS, dict)
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            if aa in AMINO_ACID_ATOMIC_COMPOSITIONS:
                composition = AMINO_ACID_ATOMIC_COMPOSITIONS[aa]
                # Each composition should have C, H, N, O at minimum
                assert 'C' in composition
                assert 'H' in composition
                assert 'N' in composition
                assert 'O' in composition


class TestSumFormulaCreation:
    """Tests for SumFormula creation."""

    def test_create_water(self):
        """Test creating water formula."""
        formula = SumFormula("H2O")
        assert formula is not None
        assert formula.formula == "H2O"

    def test_create_simple_organic(self):
        """Test creating simple organic formula."""
        formula = SumFormula("C6H12O6")  # Glucose
        assert formula is not None

    def test_create_peptide_like(self):
        """Test creating peptide-like formula."""
        formula = SumFormula("C5H9NO3")  # Glutamic acid residue-ish
        assert formula is not None


class TestSumFormulaProperties:
    """Tests for SumFormula properties."""

    def test_formula_property(self):
        """Test formula property returns string."""
        formula = SumFormula("H2O")
        assert isinstance(formula.formula, str)
        assert formula.formula == "H2O"

    def test_formula_dict_property(self):
        """Test formula_dict property returns element counts."""
        formula = SumFormula("H2O")
        elements = formula.formula_dict
        assert isinstance(elements, dict)
        assert elements.get('H') == 2
        assert elements.get('O') == 1

    def test_monoisotopic_mass_water(self):
        """Test monoisotopic mass of water."""
        formula = SumFormula("H2O")
        # H2O: 2 * 1.00783 + 15.9949 ≈ 18.0106
        assert abs(formula.monoisotopic_mass - 18.0106) < 0.001

    def test_monoisotopic_mass_glucose(self):
        """Test monoisotopic mass of glucose."""
        formula = SumFormula("C6H12O6")
        # C6H12O6: 6*12 + 12*1.00783 + 6*15.9949 ≈ 180.063
        assert abs(formula.monoisotopic_mass - 180.063) < 0.01


class TestSumFormulaIsotopeDistribution:
    """Tests for SumFormula isotope distribution."""

    def test_generate_isotope_distribution(self):
        """Test generating isotope distribution."""
        formula = SumFormula("C6H12O6")
        spectrum = formula.generate_isotope_distribution(charge=1)
        assert isinstance(spectrum, MzSpectrum)
        assert len(spectrum.mz) > 0
        assert len(spectrum.intensity) > 0

    def test_isotope_distribution_different_charges(self):
        """Test isotope distribution with different charges."""
        formula = SumFormula("C6H12O6")
        spectrum1 = formula.generate_isotope_distribution(charge=1)
        spectrum2 = formula.generate_isotope_distribution(charge=2)

        # m/z should be lower for higher charge
        assert spectrum1.mz[0] > spectrum2.mz[0]

    def test_isotope_spacing(self):
        """Test that isotope peaks are properly spaced."""
        formula = SumFormula("C10H20O5")
        spectrum = formula.generate_isotope_distribution(charge=1)

        if len(spectrum.mz) > 1:
            # For charge 1, isotope spacing should be ~1 Da
            spacing = spectrum.mz[1] - spectrum.mz[0]
            assert abs(spacing - 1.0) < 0.1


class TestSumFormulaSerialization:
    """Tests for SumFormula serialization."""

    def test_repr(self):
        """Test string representation."""
        formula = SumFormula("H2O")
        repr_str = repr(formula)
        assert "SumFormula" in repr_str
        assert "H2O" in repr_str

    def test_from_py_ptr(self):
        """Test from_py_ptr class method."""
        formula1 = SumFormula("C6H12O6")
        py_ptr = formula1.get_py_ptr()
        formula2 = SumFormula.from_py_ptr(py_ptr)

        assert formula1.formula == formula2.formula
        assert formula1.monoisotopic_mass == formula2.monoisotopic_mass


class TestMobilityConversionSingle:
    """Tests for single-value mobility conversions."""

    def test_one_over_k0_to_ccs_basic(self):
        """Test basic 1/K0 to CCS conversion."""
        one_over_k0 = 1.2  # typical value
        mz = 500.0
        charge = 2
        ccs = one_over_k0_to_ccs(one_over_k0, mz, charge)

        assert isinstance(ccs, float)
        assert ccs > 0
        # Typical CCS range is 100-1000 Å²
        assert 100 < ccs < 1500

    def test_ccs_to_one_over_k0_basic(self):
        """Test basic CCS to 1/K0 conversion."""
        ccs = 400.0  # Å²
        mz = 500.0
        charge = 2
        one_over_k0 = ccs_to_one_over_k0(ccs, mz, charge)

        assert isinstance(one_over_k0, float)
        assert one_over_k0 > 0
        # Typical 1/K0 range is 0.5-2.0 Vs/cm²
        assert 0.5 < one_over_k0 < 2.5

    def test_round_trip_conversion(self):
        """Test that CCS ↔ 1/K0 conversion is reversible."""
        original_k0 = 1.2
        mz = 500.0
        charge = 2

        ccs = one_over_k0_to_ccs(original_k0, mz, charge)
        recovered_k0 = ccs_to_one_over_k0(ccs, mz, charge)

        assert abs(original_k0 - recovered_k0) < 1e-10

    def test_mobility_decreases_with_ccs(self):
        """Test that 1/K0 (inverse mobility) increases with CCS at fixed m/z."""
        mz = 500.0
        charge = 2

        # Larger CCS should result in larger 1/K0 (lower mobility)
        k0_small_ccs = ccs_to_one_over_k0(300.0, mz, charge)
        k0_large_ccs = ccs_to_one_over_k0(600.0, mz, charge)

        assert k0_large_ccs > k0_small_ccs

    def test_mobility_increases_with_charge(self):
        """Test mobility behavior with charge."""
        ccs = 400.0
        mz = 500.0

        k0_charge1 = ccs_to_one_over_k0(ccs, mz, 1)
        k0_charge2 = ccs_to_one_over_k0(ccs, mz, 2)

        # Higher charge should give different mobility
        assert k0_charge1 != k0_charge2


class TestMobilityConversionParallel:
    """Tests for parallel mobility conversions."""

    def test_one_over_k0_to_ccs_par_basic(self):
        """Test parallel 1/K0 to CCS conversion."""
        n = 100
        one_over_k0 = np.linspace(0.8, 1.6, n)
        mz = np.linspace(300.0, 1500.0, n)
        charge = np.full(n, 2, dtype=np.int32)

        ccs = one_over_k0_to_ccs_par(one_over_k0, mz, charge)
        ccs = np.asarray(ccs)  # Convert to array if list

        assert isinstance(ccs, np.ndarray)
        assert len(ccs) == n
        assert all(ccs > 0)

    def test_ccs_to_one_over_k0_par_basic(self):
        """Test parallel CCS to 1/K0 conversion."""
        n = 100
        ccs = np.linspace(200.0, 800.0, n)
        mz = np.linspace(300.0, 1500.0, n)
        charge = np.full(n, 2, dtype=np.int32)

        one_over_k0 = ccs_to_one_over_k0_par(ccs, mz, charge)
        one_over_k0 = np.asarray(one_over_k0)  # Convert to array if list

        assert isinstance(one_over_k0, np.ndarray)
        assert len(one_over_k0) == n
        assert all(one_over_k0 > 0)

    def test_parallel_matches_sequential(self):
        """Test that parallel results match sequential."""
        n = 10
        one_over_k0 = np.linspace(0.8, 1.6, n)
        mz = np.linspace(300.0, 1500.0, n)
        charge = np.full(n, 2, dtype=np.int32)

        # Parallel computation
        ccs_par = one_over_k0_to_ccs_par(one_over_k0, mz, charge)

        # Sequential computation
        ccs_seq = np.array([
            one_over_k0_to_ccs(one_over_k0[i], mz[i], int(charge[i]))
            for i in range(n)
        ])

        assert_allclose(ccs_par, ccs_seq, rtol=1e-10)

    def test_parallel_round_trip(self):
        """Test parallel round-trip conversion."""
        n = 50
        original_k0 = np.linspace(0.8, 1.6, n)
        mz = np.linspace(300.0, 1500.0, n)
        charge = np.full(n, 2, dtype=np.int32)

        ccs = one_over_k0_to_ccs_par(original_k0, mz, charge)
        recovered_k0 = ccs_to_one_over_k0_par(ccs, mz, charge)

        assert_allclose(original_k0, recovered_k0, rtol=1e-10)

    def test_parallel_with_mixed_charges(self):
        """Test parallel conversion with mixed charge states."""
        n = 20
        one_over_k0 = np.linspace(0.8, 1.6, n)
        mz = np.linspace(300.0, 1500.0, n)
        charge = np.array([1, 2, 3, 4] * 5, dtype=np.int32)

        ccs = one_over_k0_to_ccs_par(one_over_k0, mz, charge)
        ccs = np.asarray(ccs)  # Convert to array if list

        assert len(ccs) == n
        assert all(ccs > 0)

    def test_large_array_performance(self):
        """Test parallel conversion with large arrays."""
        n = 10000
        one_over_k0 = np.random.uniform(0.6, 1.8, n)
        mz = np.random.uniform(200.0, 2000.0, n)
        charge = np.random.randint(1, 5, n).astype(np.int32)

        # Should complete without error
        ccs = one_over_k0_to_ccs_par(one_over_k0, mz, charge)
        assert len(ccs) == n


class TestMobilityEdgeCases:
    """Edge case tests for mobility conversions."""

    def test_low_mz_conversion(self):
        """Test conversion with low m/z values."""
        one_over_k0 = 1.0
        mz = 100.0  # Low m/z
        charge = 1

        ccs = one_over_k0_to_ccs(one_over_k0, mz, charge)
        assert ccs > 0

    def test_high_mz_conversion(self):
        """Test conversion with high m/z values."""
        one_over_k0 = 1.0
        mz = 5000.0  # High m/z
        charge = 5

        ccs = one_over_k0_to_ccs(one_over_k0, mz, charge)
        assert ccs > 0

    def test_single_element_array(self):
        """Test parallel conversion with single element arrays."""
        one_over_k0 = np.array([1.2])
        mz = np.array([500.0])
        charge = np.array([2], dtype=np.int32)

        ccs = one_over_k0_to_ccs_par(one_over_k0, mz, charge)
        assert len(ccs) == 1

    def test_custom_gas_mass(self):
        """Test conversion with custom drift gas mass."""
        one_over_k0 = 1.2
        mz = 500.0
        charge = 2

        # Different drift gases
        ccs_n2 = one_over_k0_to_ccs(one_over_k0, mz, charge, mass_gas=28.013)  # N2
        ccs_he = one_over_k0_to_ccs(one_over_k0, mz, charge, mass_gas=4.003)   # He

        # CCS should be different for different drift gases
        assert ccs_n2 != ccs_he

    def test_custom_temperature(self):
        """Test conversion with custom temperature."""
        one_over_k0 = 1.2
        mz = 500.0
        charge = 2

        ccs_cold = one_over_k0_to_ccs(one_over_k0, mz, charge, temp=20.0)
        ccs_hot = one_over_k0_to_ccs(one_over_k0, mz, charge, temp=50.0)

        # CCS depends on temperature
        assert ccs_cold != ccs_hot


class TestChemistryConsistency:
    """Tests for consistency across chemistry modules."""

    def test_water_mass_matches_sum_formula(self):
        """Test that water mass from constants matches sum formula."""
        formula_mass = SumFormula("H2O").monoisotopic_mass
        constant_mass = MASS_WATER

        assert abs(formula_mass - constant_mass) < 0.001

    def test_amino_acid_mass_matches_elements(self):
        """Test amino acid mass matches elemental calculation."""
        # Glycine: C2H5NO2
        if 'G' in AMINO_ACID_MASSES and 'G' in AMINO_ACID_ATOMIC_COMPOSITIONS:
            aa_mass = AMINO_ACID_MASSES['G']
            comp = AMINO_ACID_ATOMIC_COMPOSITIONS['G']

            # Calculate mass from composition
            calc_mass = (
                comp['C'] * ELEMENTAL_MONO_ISOTOPIC_MASSES['C'] +
                comp['H'] * ELEMENTAL_MONO_ISOTOPIC_MASSES['H'] +
                comp['N'] * ELEMENTAL_MONO_ISOTOPIC_MASSES['N'] +
                comp['O'] * ELEMENTAL_MONO_ISOTOPIC_MASSES['O']
            )

            assert abs(aa_mass - calc_mass) < 0.01
