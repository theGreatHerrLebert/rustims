"""
Pytest configuration and fixtures for imspy tests.

These fixtures provide common test data for spectrum, peptide, and frame tests,
ensuring consistent testing of the Rust-Python bindings via imspy_connector.
"""

import pytest
import numpy as np


# =============================================================================
# Spectrum Fixtures
# =============================================================================

@pytest.fixture
def simple_mz_array():
    """Simple m/z array for basic tests."""
    return np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)


@pytest.fixture
def simple_intensity_array():
    """Simple intensity array for basic tests."""
    return np.array([1000.0, 2000.0, 3000.0, 2000.0, 1000.0], dtype=np.float64)


@pytest.fixture
def large_mz_array():
    """Larger m/z array for performance and edge case tests."""
    return np.linspace(100.0, 2000.0, 1000, dtype=np.float64)


@pytest.fixture
def large_intensity_array():
    """Larger intensity array matching large_mz_array."""
    # Gaussian-like intensity distribution
    x = np.linspace(-3, 3, 1000)
    return (np.exp(-x**2 / 2) * 10000).astype(np.float64)


@pytest.fixture
def empty_mz_array():
    """Empty m/z array for edge case tests."""
    return np.array([], dtype=np.float64)


@pytest.fixture
def empty_intensity_array():
    """Empty intensity array for edge case tests."""
    return np.array([], dtype=np.float64)


@pytest.fixture
def single_peak_mz():
    """Single peak m/z for edge case tests."""
    return np.array([500.0], dtype=np.float64)


@pytest.fixture
def single_peak_intensity():
    """Single peak intensity for edge case tests."""
    return np.array([10000.0], dtype=np.float64)


# =============================================================================
# Index Fixtures (for IndexedMzSpectrum, TimsSpectrum)
# =============================================================================

@pytest.fixture
def simple_index_array():
    """Simple index/TOF array for indexed spectrum tests."""
    return np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int32)


@pytest.fixture
def simple_scan_array():
    """Simple scan array for frame tests."""
    return np.array([100, 100, 200, 200, 300], dtype=np.int32)


@pytest.fixture
def simple_mobility_array():
    """Simple mobility array for frame tests."""
    return np.array([1.0, 1.0, 1.2, 1.2, 1.4], dtype=np.float64)


# =============================================================================
# Peptide Fixtures
# =============================================================================

@pytest.fixture
def simple_peptide_sequence():
    """Simple unmodified peptide sequence."""
    return "PEPTIDE"


@pytest.fixture
def modified_peptide_sequence():
    """Peptide with oxidation modification (UNIMOD format)."""
    return "PEPTM[UNIMOD:35]IDE"


@pytest.fixture
def phospho_peptide_sequence():
    """Peptide with phosphorylation."""
    return "PEPS[UNIMOD:21]TIDE"


@pytest.fixture
def complex_peptide_sequence():
    """Peptide with multiple modifications."""
    return "AC[UNIMOD:4]M[UNIMOD:35]PEPTIDEK"


@pytest.fixture
def peptide_sequences_batch():
    """Batch of peptide sequences for parallel processing tests."""
    return [
        "PEPTIDE",
        "SEQUENCE",
        "ANOTHER",
        "TESTPEPTIDE",
        "MYPEPTIDEK",
    ]


# =============================================================================
# Frame Fixtures
# =============================================================================

@pytest.fixture
def frame_data():
    """Complete frame data for TimsFrame tests."""
    n_points = 100
    return {
        "frame_id": 1,
        "ms_type": 0,  # Precursor
        "retention_time": 100.5,
        "scan": np.random.randint(1, 1000, n_points).astype(np.int32),
        "mobility": np.random.uniform(0.6, 1.6, n_points).astype(np.float64),
        "tof": np.random.randint(100000, 400000, n_points).astype(np.int32),
        "mz": np.random.uniform(100.0, 1700.0, n_points).astype(np.float64),
        "intensity": np.random.uniform(100.0, 10000.0, n_points).astype(np.float64),
    }


@pytest.fixture
def precursor_frame_data(frame_data):
    """Frame data specifically for precursor (MS1) frame."""
    frame_data["ms_type"] = 0
    return frame_data


@pytest.fixture
def fragment_frame_data(frame_data):
    """Frame data specifically for fragment (MS2) frame."""
    data = frame_data.copy()
    data["frame_id"] = 2
    data["ms_type"] = 8  # FragmentDda
    return data


# =============================================================================
# Chemistry Fixtures
# =============================================================================

@pytest.fixture
def amino_acid_masses():
    """Expected monoisotopic masses for standard amino acids."""
    return {
        'A': 71.03711,
        'R': 156.10111,
        'N': 114.04293,
        'D': 115.02694,
        'C': 103.00919,
        'E': 129.04259,
        'Q': 128.05858,
        'G': 57.02146,
        'H': 137.05891,
        'I': 113.08406,
        'L': 113.08406,
        'K': 128.09496,
        'M': 131.04049,
        'F': 147.06841,
        'P': 97.05276,
        'S': 87.03203,
        'T': 101.04768,
        'W': 186.07931,
        'Y': 163.06333,
        'V': 99.06841,
    }


@pytest.fixture
def water_mass():
    """Mass of water molecule."""
    return 18.01056


@pytest.fixture
def proton_mass():
    """Mass of proton."""
    return 1.00727647


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def tolerance():
    """Default tolerance for floating point comparisons."""
    return 1e-6


@pytest.fixture
def ppm_tolerance():
    """Default ppm tolerance for mass comparisons."""
    return 10.0  # 10 ppm
