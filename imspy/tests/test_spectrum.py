"""
Tests for spectrum classes (MzSpectrum, IndexedMzSpectrum, TimsSpectrum).

These tests verify the Rust-Python bindings work correctly for spectrum
data structures, including creation, property access, operations, and
numpy array interoperability.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from imspy.data.spectrum import MzSpectrum, IndexedMzSpectrum, TimsSpectrum


class TestMzSpectrum:
    """Tests for MzSpectrum class."""

    # =========================================================================
    # Creation Tests
    # =========================================================================

    def test_create_simple_spectrum(self, simple_mz_array, simple_intensity_array):
        """Test creating a simple MzSpectrum."""
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        assert spectrum is not None
        assert len(spectrum.mz) == 5
        assert len(spectrum.intensity) == 5

    def test_create_empty_spectrum(self, empty_mz_array, empty_intensity_array):
        """Test creating an empty MzSpectrum."""
        spectrum = MzSpectrum(empty_mz_array, empty_intensity_array)
        assert spectrum is not None
        assert len(spectrum.mz) == 0
        assert len(spectrum.intensity) == 0

    def test_create_single_peak_spectrum(self, single_peak_mz, single_peak_intensity):
        """Test creating a single-peak MzSpectrum."""
        spectrum = MzSpectrum(single_peak_mz, single_peak_intensity)
        assert len(spectrum.mz) == 1
        assert spectrum.mz[0] == 500.0
        assert spectrum.intensity[0] == 10000.0

    def test_create_large_spectrum(self, large_mz_array, large_intensity_array):
        """Test creating a large MzSpectrum."""
        spectrum = MzSpectrum(large_mz_array, large_intensity_array)
        assert len(spectrum.mz) == 1000
        assert len(spectrum.intensity) == 1000

    def test_create_mismatched_arrays_raises(self):
        """Test that mismatched array lengths raise an error."""
        mz = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        intensity = np.array([1000.0, 2000.0], dtype=np.float64)
        with pytest.raises(AssertionError):
            MzSpectrum(mz, intensity)

    # =========================================================================
    # Property Tests
    # =========================================================================

    def test_mz_property_returns_numpy_array(self, simple_mz_array, simple_intensity_array):
        """Test that mz property returns a numpy array."""
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        assert isinstance(spectrum.mz, np.ndarray)
        assert spectrum.mz.dtype == np.float64

    def test_intensity_property_returns_numpy_array(self, simple_mz_array, simple_intensity_array):
        """Test that intensity property returns a numpy array."""
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        assert isinstance(spectrum.intensity, np.ndarray)
        assert spectrum.intensity.dtype == np.float64

    def test_property_values_match_input(self, simple_mz_array, simple_intensity_array):
        """Test that property values match input data."""
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        assert_array_almost_equal(spectrum.mz, simple_mz_array)
        assert_array_almost_equal(spectrum.intensity, simple_intensity_array)

    def test_df_property_returns_dataframe(self, simple_mz_array, simple_intensity_array):
        """Test that df property returns a pandas DataFrame."""
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        df = spectrum.df
        assert 'mz' in df.columns
        assert 'intensity' in df.columns
        assert len(df) == 5

    # =========================================================================
    # Operator Tests
    # =========================================================================

    def test_add_spectra(self, simple_mz_array, simple_intensity_array):
        """Test adding two spectra together."""
        spectrum1 = MzSpectrum(simple_mz_array, simple_intensity_array)
        spectrum2 = MzSpectrum(simple_mz_array, simple_intensity_array * 2)
        result = spectrum1 + spectrum2
        assert result is not None
        assert isinstance(result, MzSpectrum)
        # Combined spectrum should have peaks

    def test_multiply_spectrum_by_scalar(self, simple_mz_array, simple_intensity_array):
        """Test multiplying spectrum by a scalar."""
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        result = spectrum * 2.0
        assert result is not None
        assert isinstance(result, MzSpectrum)
        # Intensities should be doubled
        assert_array_almost_equal(result.intensity, simple_intensity_array * 2.0)

    def test_multiply_spectrum_by_zero(self, simple_mz_array, simple_intensity_array):
        """Test multiplying spectrum by zero."""
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        result = spectrum * 0.0
        assert_array_almost_equal(result.intensity, np.zeros_like(simple_intensity_array))

    # =========================================================================
    # Filter Tests
    # =========================================================================

    def test_filter_by_mz_range(self, simple_mz_array, simple_intensity_array):
        """Test filtering spectrum by m/z range."""
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        filtered = spectrum.filter(mz_min=150.0, mz_max=350.0)
        assert isinstance(filtered, MzSpectrum)
        assert all(filtered.mz >= 150.0)
        assert all(filtered.mz <= 350.0)

    def test_filter_by_intensity_range(self, simple_mz_array, simple_intensity_array):
        """Test filtering spectrum by intensity range."""
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        filtered = spectrum.filter(intensity_min=1500.0, intensity_max=2500.0)
        assert isinstance(filtered, MzSpectrum)
        assert all(filtered.intensity >= 1500.0)
        assert all(filtered.intensity <= 2500.0)

    def test_filter_returns_empty_when_no_match(self, simple_mz_array, simple_intensity_array):
        """Test filter returns empty spectrum when no peaks match."""
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        filtered = spectrum.filter(mz_min=1000.0, mz_max=2000.0)
        assert len(filtered.mz) == 0

    # =========================================================================
    # Resolution Tests
    # =========================================================================

    def test_to_resolution(self, large_mz_array, large_intensity_array):
        """Test binning spectrum to resolution."""
        spectrum = MzSpectrum(large_mz_array, large_intensity_array)
        binned = spectrum.to_resolution(1)  # 0.1 Da bins
        assert isinstance(binned, MzSpectrum)
        # Binned spectrum should have fewer unique m/z values
        assert len(binned.mz) <= len(spectrum.mz)

    # =========================================================================
    # Serialization Tests
    # =========================================================================

    def test_to_jsons_and_from_jsons(self, simple_mz_array, simple_intensity_array):
        """Test JSON serialization round-trip."""
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        json_str = spectrum.to_jsons()
        restored = MzSpectrum.from_jsons(json_str)
        assert_array_almost_equal(spectrum.mz, restored.mz)
        assert_array_almost_equal(spectrum.intensity, restored.intensity)

    # =========================================================================
    # Repr Tests
    # =========================================================================

    def test_repr(self, simple_mz_array, simple_intensity_array):
        """Test string representation."""
        spectrum = MzSpectrum(simple_mz_array, simple_intensity_array)
        repr_str = repr(spectrum)
        assert "MzSpectrum" in repr_str
        assert "num_peaks=5" in repr_str


class TestIndexedMzSpectrum:
    """Tests for IndexedMzSpectrum class."""

    # =========================================================================
    # Creation Tests
    # =========================================================================

    def test_create_indexed_spectrum(self, simple_index_array, simple_mz_array, simple_intensity_array):
        """Test creating an IndexedMzSpectrum."""
        spectrum = IndexedMzSpectrum(simple_index_array, simple_mz_array, simple_intensity_array)
        assert spectrum is not None
        assert len(spectrum.index) == 5
        assert len(spectrum.mz) == 5
        assert len(spectrum.intensity) == 5

    def test_create_mismatched_arrays_raises(self, simple_mz_array, simple_intensity_array):
        """Test that mismatched array lengths raise an error."""
        index = np.array([1000, 2000], dtype=np.int32)
        with pytest.raises(AssertionError):
            IndexedMzSpectrum(index, simple_mz_array, simple_intensity_array)

    # =========================================================================
    # Property Tests
    # =========================================================================

    def test_index_property(self, simple_index_array, simple_mz_array, simple_intensity_array):
        """Test index property returns correct values."""
        spectrum = IndexedMzSpectrum(simple_index_array, simple_mz_array, simple_intensity_array)
        assert isinstance(spectrum.index, np.ndarray)
        assert_array_equal(spectrum.index, simple_index_array)

    def test_df_property(self, simple_index_array, simple_mz_array, simple_intensity_array):
        """Test df property returns DataFrame with all columns."""
        spectrum = IndexedMzSpectrum(simple_index_array, simple_mz_array, simple_intensity_array)
        df = spectrum.df
        assert 'index' in df.columns
        assert 'mz' in df.columns
        assert 'intensity' in df.columns

    # =========================================================================
    # Filter Tests
    # =========================================================================

    def test_filter_indexed_spectrum(self, simple_index_array, simple_mz_array, simple_intensity_array):
        """Test filtering IndexedMzSpectrum."""
        spectrum = IndexedMzSpectrum(simple_index_array, simple_mz_array, simple_intensity_array)
        filtered = spectrum.filter(mz_min=150.0, mz_max=350.0)
        assert isinstance(filtered, IndexedMzSpectrum)
        assert all(filtered.mz >= 150.0)
        assert all(filtered.mz <= 350.0)


class TestTimsSpectrum:
    """Tests for TimsSpectrum class."""

    # =========================================================================
    # Creation Tests
    # =========================================================================

    def test_create_tims_spectrum(self, simple_index_array, simple_mz_array, simple_intensity_array):
        """Test creating a TimsSpectrum."""
        spectrum = TimsSpectrum(
            frame_id=1,
            scan=100,
            retention_time=50.5,
            mobility=1.2,
            ms_type=0,
            index=simple_index_array,
            mz=simple_mz_array,
            intensity=simple_intensity_array
        )
        assert spectrum is not None
        assert spectrum.frame_id == 1
        assert spectrum.scan == 100
        assert spectrum.retention_time == 50.5
        assert spectrum.mobility == 1.2

    # =========================================================================
    # Property Tests
    # =========================================================================

    def test_tims_spectrum_properties(self, simple_index_array, simple_mz_array, simple_intensity_array):
        """Test all TimsSpectrum properties."""
        spectrum = TimsSpectrum(
            frame_id=42,
            scan=200,
            retention_time=100.0,
            mobility=1.5,
            ms_type=8,
            index=simple_index_array,
            mz=simple_mz_array,
            intensity=simple_intensity_array
        )
        assert spectrum.frame_id == 42
        assert spectrum.scan == 200
        assert spectrum.retention_time == 100.0
        assert spectrum.mobility == 1.5
        assert len(spectrum.mz) == 5
        assert len(spectrum.intensity) == 5

    def test_tims_spectrum_arrays(self, simple_index_array, simple_mz_array, simple_intensity_array):
        """Test TimsSpectrum array properties."""
        spectrum = TimsSpectrum(
            frame_id=1,
            scan=100,
            retention_time=50.5,
            mobility=1.2,
            ms_type=0,
            index=simple_index_array,
            mz=simple_mz_array,
            intensity=simple_intensity_array
        )
        assert isinstance(spectrum.mz, np.ndarray)
        assert isinstance(spectrum.intensity, np.ndarray)
        assert_array_almost_equal(spectrum.mz, simple_mz_array)
        assert_array_almost_equal(spectrum.intensity, simple_intensity_array)


class TestSpectrumEdgeCases:
    """Edge case tests for spectrum classes."""

    def test_very_small_intensities(self):
        """Test handling of very small intensity values."""
        mz = np.array([100.0, 200.0], dtype=np.float64)
        intensity = np.array([1e-10, 1e-15], dtype=np.float64)
        spectrum = MzSpectrum(mz, intensity)
        assert_array_almost_equal(spectrum.intensity, intensity)

    def test_very_large_intensities(self):
        """Test handling of very large intensity values."""
        mz = np.array([100.0, 200.0], dtype=np.float64)
        intensity = np.array([1e10, 1e15], dtype=np.float64)
        spectrum = MzSpectrum(mz, intensity)
        assert_array_almost_equal(spectrum.intensity, intensity)

    def test_high_precision_mz(self):
        """Test handling of high-precision m/z values."""
        mz = np.array([100.123456789, 200.987654321], dtype=np.float64)
        intensity = np.array([1000.0, 2000.0], dtype=np.float64)
        spectrum = MzSpectrum(mz, intensity)
        # Should preserve high precision
        assert abs(spectrum.mz[0] - 100.123456789) < 1e-9
        assert abs(spectrum.mz[1] - 200.987654321) < 1e-9

    def test_numpy_array_copy_independence(self):
        """Test that modifying input arrays doesn't affect spectrum."""
        mz = np.array([100.0, 200.0], dtype=np.float64)
        intensity = np.array([1000.0, 2000.0], dtype=np.float64)
        spectrum = MzSpectrum(mz.copy(), intensity.copy())
        original_mz = spectrum.mz.copy()
        mz[0] = 999.0  # Modify original
        assert_array_almost_equal(spectrum.mz, original_mz)


class TestSpectrumFromPyPtr:
    """Tests for from_py_ptr class method."""

    def test_from_py_ptr_preserves_data(self, simple_mz_array, simple_intensity_array):
        """Test that from_py_ptr correctly wraps a Rust object."""
        spectrum1 = MzSpectrum(simple_mz_array, simple_intensity_array)
        py_ptr = spectrum1.get_py_ptr()
        spectrum2 = MzSpectrum.from_py_ptr(py_ptr)
        assert_array_almost_equal(spectrum1.mz, spectrum2.mz)
        assert_array_almost_equal(spectrum1.intensity, spectrum2.intensity)
