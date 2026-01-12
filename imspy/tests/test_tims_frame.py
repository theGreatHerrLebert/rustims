"""
Tests for TimsFrame class.

These tests verify the Rust-Python bindings work correctly for TimsFrame
data structures, including creation, property access, filtering, and
numpy array interoperability.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from imspy.timstof.frame import TimsFrame
from imspy.data.spectrum import IndexedMzSpectrum


class TestTimsFrameCreation:
    """Tests for TimsFrame creation."""

    def test_create_simple_frame(self):
        """Test creating a simple TimsFrame."""
        n = 10
        frame = TimsFrame(
            frame_id=1,
            ms_type=0,
            retention_time=100.0,
            scan=np.arange(n, dtype=np.int32),
            mobility=np.linspace(0.8, 1.4, n, dtype=np.float64),
            tof=np.arange(100000, 100000 + n, dtype=np.int32),
            mz=np.linspace(400.0, 1200.0, n, dtype=np.float64),
            intensity=np.ones(n, dtype=np.float64) * 1000.0
        )
        assert frame is not None
        assert frame.frame_id == 1
        assert frame.retention_time == 100.0

    def test_create_frame_with_fixture(self, frame_data):
        """Test creating a TimsFrame with fixture data."""
        frame = TimsFrame(
            frame_id=frame_data["frame_id"],
            ms_type=frame_data["ms_type"],
            retention_time=frame_data["retention_time"],
            scan=frame_data["scan"],
            mobility=frame_data["mobility"],
            tof=frame_data["tof"],
            mz=frame_data["mz"],
            intensity=frame_data["intensity"]
        )
        assert frame is not None
        assert len(frame.mz) == 100

    def test_create_empty_frame(self):
        """Test creating an empty TimsFrame."""
        frame = TimsFrame(
            frame_id=1,
            ms_type=0,
            retention_time=100.0,
            scan=np.array([], dtype=np.int32),
            mobility=np.array([], dtype=np.float64),
            tof=np.array([], dtype=np.int32),
            mz=np.array([], dtype=np.float64),
            intensity=np.array([], dtype=np.float64)
        )
        assert frame is not None
        assert len(frame.mz) == 0

    def test_create_mismatched_arrays_raises(self):
        """Test that mismatched array lengths raise an error."""
        with pytest.raises(AssertionError):
            TimsFrame(
                frame_id=1,
                ms_type=0,
                retention_time=100.0,
                scan=np.array([1, 2, 3], dtype=np.int32),
                mobility=np.array([1.0, 1.2], dtype=np.float64),  # Wrong length
                tof=np.array([100, 200, 300], dtype=np.int32),
                mz=np.array([500.0, 600.0, 700.0], dtype=np.float64),
                intensity=np.array([1000.0, 2000.0, 3000.0], dtype=np.float64)
            )


class TestTimsFrameProperties:
    """Tests for TimsFrame property access."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for property tests."""
        n = 20
        return TimsFrame(
            frame_id=42,
            ms_type=0,
            retention_time=123.456,
            scan=np.arange(100, 100 + n, dtype=np.int32),
            mobility=np.linspace(0.9, 1.5, n, dtype=np.float64),
            tof=np.arange(150000, 150000 + n, dtype=np.int32),
            mz=np.linspace(300.0, 1500.0, n, dtype=np.float64),
            intensity=np.random.uniform(100, 10000, n).astype(np.float64)
        )

    def test_frame_id_property(self, sample_frame):
        """Test frame_id property."""
        assert sample_frame.frame_id == 42

    def test_retention_time_property(self, sample_frame):
        """Test retention_time property."""
        assert sample_frame.retention_time == 123.456

    def test_ms_type_property(self, sample_frame):
        """Test ms_type property returns numeric value."""
        assert sample_frame.ms_type == 0

    def test_ms_type_as_string_property(self, sample_frame):
        """Test ms_type_as_string property."""
        assert isinstance(sample_frame.ms_type_as_string, str)

    def test_scan_property(self, sample_frame):
        """Test scan property returns numpy array."""
        assert isinstance(sample_frame.scan, np.ndarray)
        assert sample_frame.scan.dtype == np.int32
        assert len(sample_frame.scan) == 20

    def test_mobility_property(self, sample_frame):
        """Test mobility property returns numpy array."""
        assert isinstance(sample_frame.mobility, np.ndarray)
        assert sample_frame.mobility.dtype == np.float64

    def test_tof_property(self, sample_frame):
        """Test tof property returns numpy array."""
        assert isinstance(sample_frame.tof, np.ndarray)
        assert sample_frame.tof.dtype == np.int32

    def test_mz_property(self, sample_frame):
        """Test mz property returns numpy array."""
        assert isinstance(sample_frame.mz, np.ndarray)
        assert sample_frame.mz.dtype == np.float64

    def test_intensity_property(self, sample_frame):
        """Test intensity property returns numpy array."""
        assert isinstance(sample_frame.intensity, np.ndarray)
        assert sample_frame.intensity.dtype == np.float64

    def test_df_property(self, sample_frame):
        """Test df property returns pandas DataFrame."""
        df = sample_frame.df
        assert 'frame' in df.columns
        assert 'retention_time' in df.columns
        assert 'scan' in df.columns
        assert 'mobility' in df.columns
        assert 'tof' in df.columns
        assert 'mz' in df.columns
        assert 'intensity' in df.columns
        assert len(df) == 20


class TestTimsFrameFiltering:
    """Tests for TimsFrame filtering operations."""

    @pytest.fixture
    def filterable_frame(self):
        """Create a frame with known values for filtering tests."""
        return TimsFrame(
            frame_id=1,
            ms_type=0,
            retention_time=100.0,
            scan=np.array([100, 200, 300, 400, 500], dtype=np.int32),
            mobility=np.array([0.8, 1.0, 1.2, 1.4, 1.6], dtype=np.float64),
            tof=np.array([100000, 150000, 200000, 250000, 300000], dtype=np.int32),
            mz=np.array([300.0, 500.0, 700.0, 900.0, 1100.0], dtype=np.float64),
            intensity=np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0], dtype=np.float64)
        )

    def test_filter_by_mz_range(self, filterable_frame):
        """Test filtering by m/z range."""
        filtered = filterable_frame.filter(mz_min=400.0, mz_max=800.0)
        assert isinstance(filtered, TimsFrame)
        assert all(filtered.mz >= 400.0)
        assert all(filtered.mz <= 800.0)

    def test_filter_by_scan_range(self, filterable_frame):
        """Test filtering by scan range."""
        filtered = filterable_frame.filter(scan_min=150, scan_max=350)
        assert isinstance(filtered, TimsFrame)
        assert all(filtered.scan >= 150)
        assert all(filtered.scan <= 350)

    def test_filter_by_mobility_range(self, filterable_frame):
        """Test filtering by mobility range."""
        filtered = filterable_frame.filter(mobility_min=0.9, mobility_max=1.3)
        assert isinstance(filtered, TimsFrame)
        assert all(filtered.mobility >= 0.9)
        assert all(filtered.mobility <= 1.3)

    def test_filter_by_intensity_range(self, filterable_frame):
        """Test filtering by intensity range."""
        filtered = filterable_frame.filter(intensity_min=2500.0, intensity_max=4500.0)
        assert isinstance(filtered, TimsFrame)
        assert all(filtered.intensity >= 2500.0)
        assert all(filtered.intensity <= 4500.0)

    def test_filter_combined_criteria(self, filterable_frame):
        """Test filtering with multiple criteria."""
        filtered = filterable_frame.filter(
            mz_min=400.0,
            mz_max=1000.0,
            intensity_min=1500.0
        )
        assert isinstance(filtered, TimsFrame)
        assert all(filtered.mz >= 400.0)
        assert all(filtered.mz <= 1000.0)
        assert all(filtered.intensity >= 1500.0)

    def test_filter_returns_empty_when_no_match(self, filterable_frame):
        """Test filter returns empty frame when no data matches."""
        filtered = filterable_frame.filter(mz_min=2000.0, mz_max=3000.0)
        assert len(filtered.mz) == 0


class TestTimsFrameOperations:
    """Tests for TimsFrame operations."""

    def test_add_frames(self):
        """Test adding two frames together."""
        n = 5
        frame1 = TimsFrame(
            frame_id=1,
            ms_type=0,
            retention_time=100.0,
            scan=np.arange(n, dtype=np.int32),
            mobility=np.linspace(1.0, 1.4, n, dtype=np.float64),
            tof=np.arange(100000, 100000 + n, dtype=np.int32),
            mz=np.linspace(400.0, 800.0, n, dtype=np.float64),
            intensity=np.ones(n, dtype=np.float64) * 1000.0
        )
        frame2 = TimsFrame(
            frame_id=1,
            ms_type=0,
            retention_time=100.0,
            scan=np.arange(n, dtype=np.int32),
            mobility=np.linspace(1.0, 1.4, n, dtype=np.float64),
            tof=np.arange(100000, 100000 + n, dtype=np.int32),
            mz=np.linspace(400.0, 800.0, n, dtype=np.float64),
            intensity=np.ones(n, dtype=np.float64) * 2000.0
        )
        result = frame1 + frame2
        assert isinstance(result, TimsFrame)

    def test_to_resolution(self):
        """Test converting frame to resolution."""
        n = 100
        frame = TimsFrame(
            frame_id=1,
            ms_type=0,
            retention_time=100.0,
            scan=np.random.randint(1, 100, n).astype(np.int32),
            mobility=np.random.uniform(0.8, 1.6, n).astype(np.float64),
            tof=np.random.randint(100000, 300000, n).astype(np.int32),
            mz=np.random.uniform(300.0, 1500.0, n).astype(np.float64),
            intensity=np.random.uniform(100.0, 10000.0, n).astype(np.float64)
        )
        binned = frame.to_resolution(1)
        assert isinstance(binned, TimsFrame)
        # Binned frame may have fewer unique m/z values
        assert len(binned.mz) <= len(frame.mz)

    def test_to_indexed_mz_spectrum(self):
        """Test converting frame to IndexedMzSpectrum."""
        n = 10
        frame = TimsFrame(
            frame_id=1,
            ms_type=0,
            retention_time=100.0,
            scan=np.arange(n, dtype=np.int32),
            mobility=np.linspace(1.0, 1.4, n, dtype=np.float64),
            tof=np.arange(100000, 100000 + n, dtype=np.int32),
            mz=np.linspace(400.0, 800.0, n, dtype=np.float64),
            intensity=np.ones(n, dtype=np.float64) * 1000.0
        )
        spectrum = frame.to_indexed_mz_spectrum()
        assert isinstance(spectrum, IndexedMzSpectrum)


class TestTimsFrameFromPyPtr:
    """Tests for from_py_ptr class method."""

    def test_from_py_ptr_preserves_data(self):
        """Test that from_py_ptr correctly wraps a Rust object."""
        n = 10
        scan = np.arange(n, dtype=np.int32)
        mobility = np.linspace(1.0, 1.4, n, dtype=np.float64)
        tof = np.arange(100000, 100000 + n, dtype=np.int32)
        mz = np.linspace(400.0, 800.0, n, dtype=np.float64)
        intensity = np.ones(n, dtype=np.float64) * 1000.0

        frame1 = TimsFrame(
            frame_id=1,
            ms_type=0,
            retention_time=100.0,
            scan=scan,
            mobility=mobility,
            tof=tof,
            mz=mz,
            intensity=intensity
        )
        py_ptr = frame1.get_py_ptr()
        frame2 = TimsFrame.from_py_ptr(py_ptr)

        assert frame1.frame_id == frame2.frame_id
        assert frame1.retention_time == frame2.retention_time
        assert_array_almost_equal(frame1.mz, frame2.mz)
        assert_array_almost_equal(frame1.intensity, frame2.intensity)


class TestTimsFrameEdgeCases:
    """Edge case tests for TimsFrame."""

    def test_single_point_frame(self):
        """Test frame with single data point."""
        frame = TimsFrame(
            frame_id=1,
            ms_type=0,
            retention_time=100.0,
            scan=np.array([100], dtype=np.int32),
            mobility=np.array([1.2], dtype=np.float64),
            tof=np.array([150000], dtype=np.int32),
            mz=np.array([500.0], dtype=np.float64),
            intensity=np.array([5000.0], dtype=np.float64)
        )
        assert len(frame.mz) == 1
        assert frame.mz[0] == 500.0

    def test_large_frame(self, random_seed):
        """Test handling of large frames."""
        n = 10000
        frame = TimsFrame(
            frame_id=1,
            ms_type=0,
            retention_time=100.0,
            scan=np.random.randint(1, 1000, n).astype(np.int32),
            mobility=np.random.uniform(0.6, 1.8, n).astype(np.float64),
            tof=np.random.randint(50000, 400000, n).astype(np.int32),
            mz=np.random.uniform(100.0, 2000.0, n).astype(np.float64),
            intensity=np.random.uniform(10.0, 100000.0, n).astype(np.float64)
        )
        assert len(frame.mz) == n
        # Test that operations still work on large frames
        filtered = frame.filter(mz_min=500.0, mz_max=1500.0)
        assert isinstance(filtered, TimsFrame)

    def test_frame_id_zero(self):
        """Test frame with frame_id = 0."""
        frame = TimsFrame(
            frame_id=0,
            ms_type=0,
            retention_time=0.0,
            scan=np.array([1], dtype=np.int32),
            mobility=np.array([1.0], dtype=np.float64),
            tof=np.array([100000], dtype=np.int32),
            mz=np.array([500.0], dtype=np.float64),
            intensity=np.array([1000.0], dtype=np.float64)
        )
        assert frame.frame_id == 0

    def test_negative_frame_id(self):
        """Test frame with negative frame_id (should work as int)."""
        frame = TimsFrame(
            frame_id=-1,
            ms_type=0,
            retention_time=100.0,
            scan=np.array([1], dtype=np.int32),
            mobility=np.array([1.0], dtype=np.float64),
            tof=np.array([100000], dtype=np.int32),
            mz=np.array([500.0], dtype=np.float64),
            intensity=np.array([1000.0], dtype=np.float64)
        )
        assert frame.frame_id == -1
