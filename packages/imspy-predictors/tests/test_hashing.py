"""Tests for locality-sensitive hashing."""

import pytest
import numpy as np
import torch

# Import directly from hashing module to avoid imspy_core dependency
from imspy_predictors.hashing import CosimHasher, TimsHasher, SpectralHasher


class TestCosimHasher:
    """Test suite for CosimHasher."""

    @pytest.fixture
    def hasher(self):
        """Create a CosimHasher for testing."""
        return CosimHasher(
            target_vector_length=100,
            trials=16,
            len_trial=12,
            seed=42,
            backend='torch',
        )

    def test_hasher_creation(self, hasher):
        """Test hasher can be created."""
        assert hasher is not None
        assert hasher.trials == 16
        assert hasher.len_trial == 12
        assert hasher.target_vector_length == 100

    def test_calculate_keys_numpy(self, hasher):
        """Test hash key calculation with NumPy input."""
        n_samples = 10
        vectors = np.random.randn(n_samples, 100).astype(np.float32)

        keys = hasher.calculate_keys(vectors)

        assert keys.shape == (n_samples, 16)
        # Keys should be non-negative integers
        assert (keys >= 0).all()

    def test_calculate_keys_torch(self, hasher):
        """Test hash key calculation with PyTorch input."""
        n_samples = 10
        vectors = torch.randn(n_samples, 100)

        keys = hasher.calculate_keys(vectors)

        assert keys.shape == (n_samples, 16)

    def test_similar_vectors_similar_keys(self, hasher):
        """Test that similar vectors produce similar hash keys."""
        # Create two similar vectors
        base = np.random.randn(100).astype(np.float32)
        similar = base + np.random.randn(100).astype(np.float32) * 0.1
        different = np.random.randn(100).astype(np.float32)

        vectors = np.stack([base, similar, different])
        keys = hasher.calculate_keys(vectors)

        # Count matching keys between base and similar vs base and different
        base_keys = keys[0]
        similar_keys = keys[1]
        different_keys = keys[2]

        similar_matches = (base_keys == similar_keys).sum()
        different_matches = (base_keys == different_keys).sum()

        # Similar vectors should have more matching keys
        assert similar_matches > different_matches

    def test_to_device(self, hasher):
        """Test moving hasher to device."""
        hasher.to('cpu')
        assert hasher.hash_tensor.device.type == 'cpu'

    def test_repr(self, hasher):
        """Test string representation."""
        repr_str = repr(hasher)
        assert "CosimHasher" in repr_str
        assert "100" in repr_str  # target_vector_length


class TestTimsHasher:
    """Test suite for TimsHasher."""

    def test_tims_hasher_creation(self):
        """Test TimsHasher can be created."""
        hasher = TimsHasher(
            trials=32,
            len_trial=20,
            resolution=1,
            num_dalton=10,
            backend='torch',
        )
        assert hasher is not None
        # Vector length should be 10 * 10^1 + 1 = 101
        assert hasher.target_vector_length == 101

    def test_tims_hasher_resolution(self):
        """Test TimsHasher with different resolutions."""
        hasher_res0 = TimsHasher(resolution=0, num_dalton=10, backend='torch')
        hasher_res1 = TimsHasher(resolution=1, num_dalton=10, backend='torch')

        # resolution=0: 10 * 10^0 + 1 = 11
        assert hasher_res0.target_vector_length == 11
        # resolution=1: 10 * 10^1 + 1 = 101
        assert hasher_res1.target_vector_length == 101


class TestSpectralHasher:
    """Test suite for SpectralHasher."""

    @pytest.fixture
    def hasher(self):
        """Create a SpectralHasher for testing."""
        return SpectralHasher(trials=64, len_trial=16, backend='torch')

    def test_spectral_hasher_creation(self, hasher):
        """Test SpectralHasher can be created."""
        assert hasher is not None

    def test_spectral_hasher_prosit_dim(self, hasher):
        """Test SpectralHasher uses Prosit dimension."""
        assert SpectralHasher.PROSIT_DIM == 174

    def test_calculate_keys(self, hasher):
        """Test hash key calculation for spectral data."""
        n_samples = 10
        # Prosit-like intensity vectors
        intensities = np.random.rand(n_samples, 174).astype(np.float32)

        keys = hasher.calculate_keys(intensities)

        assert keys.shape == (n_samples, 64)

    def test_to_device(self, hasher):
        """Test moving hasher to device."""
        hasher.to('cpu')
        assert hasher._hasher.hash_tensor.device.type == 'cpu'


class TestHasherBackendCompatibility:
    """Test backend compatibility for hashers."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cuda_support(self):
        """Test that hashers work on CUDA."""
        hasher = CosimHasher(
            target_vector_length=100,
            trials=16,
            len_trial=12,
            backend='torch',
        )
        hasher.to('cuda')

        vectors = torch.randn(10, 100, device='cuda')
        keys = hasher.calculate_keys(vectors)

        assert keys.device.type == 'cuda'

    def test_len_trial_validation(self):
        """Test len_trial validation."""
        # len_trial > 64 should raise error
        with pytest.raises(ValueError):
            CosimHasher(
                target_vector_length=100,
                trials=16,
                len_trial=65,
            )

    def test_int64_dtype_for_large_len_trial(self):
        """Test int64 is used for len_trial > 32."""
        hasher = CosimHasher(
            target_vector_length=100,
            trials=16,
            len_trial=40,
            backend='torch',
        )

        assert hasher.dtype_int == 'int64'
