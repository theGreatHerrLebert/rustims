"""
Locality-Sensitive Hashing for spectral similarity.

This module provides hash-based methods for fast approximate nearest neighbor
search on spectral data. Uses random projections (SimHash) for cosine similarity.
"""

import warnings
from typing import Optional, Union

import numpy as np

# Import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class CosimHasher:
    """
    Cosine Similarity Hasher using random projections (SimHash).

    Uses random hyperplane-based locality-sensitive hashing to create
    compact hash codes that preserve cosine similarity. Similar vectors
    will have similar hash codes with high probability.

    Args:
        target_vector_length: Dimensionality of input vectors
        trials: Number of hash tables (default: 32)
        len_trial: Number of bits per hash (default: 20)
        seed: Random seed for reproducibility (default: 42)

    Example:
        >>> hasher = CosimHasher(target_vector_length=174, trials=32, len_trial=20)
        >>> vectors = torch.randn(100, 174)
        >>> hash_keys = hasher.calculate_keys(vectors)
        >>> print(hash_keys.shape)  # (100, 32)
    """

    def __init__(
        self,
        target_vector_length: int,
        trials: int = 32,
        len_trial: int = 20,
        seed: int = 42,
        backend: Optional[str] = None,  # Kept for backward compatibility, ignored
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CosimHasher. Install with: pip install torch")

        assert trials > 0, f'trials must be > 0, got: {trials}'
        assert len_trial > 0, f'len_trial must be > 0, got: {len_trial}'

        self.trials = trials
        self.len_trial = len_trial
        self.seed = seed
        self.target_vector_length = target_vector_length
        self.backend = 'torch'  # Always torch

        # Validate len_trial range
        if len_trial > 64:
            raise ValueError(f"len_trial cannot exceed 64, got: {len_trial}")
        elif len_trial > 32:
            warnings.warn(
                f"len_trial={len_trial} > 32 requires int64, may be slower."
            )
            self.dtype_int = 'int64'
        else:
            self.dtype_int = 'int32'

        # Generate random projection matrix
        np.random.seed(seed)
        size = (len_trial * trials, target_vector_length)
        random_matrix = np.random.normal(0, 1, size=size).astype(np.float32)

        # Create power-of-2 vector for binary to integer conversion
        powers = np.array([2 ** i for i in range(len_trial)])

        # Initialize PyTorch tensors
        self.hash_tensor = torch.from_numpy(random_matrix.T)  # (target_len, trials*len_trial)

        if self.dtype_int == 'int64':
            self.V = torch.from_numpy(powers.astype(np.int64)).unsqueeze(1)
        else:
            self.V = torch.from_numpy(powers.astype(np.int32)).unsqueeze(1)

    def __repr__(self) -> str:
        return (
            f"CosimHasher(target_vector_length={self.target_vector_length}, "
            f"trials={self.trials}, len_trial={self.len_trial}, "
            f"seed={self.seed}, backend='{self.backend}')"
        )

    def calculate_keys(
        self,
        W: Union["torch.Tensor", np.ndarray],
    ) -> "torch.Tensor":
        """
        Calculate hash keys for input vectors.

        Args:
            W: Input vectors of shape (n_samples, target_vector_length)

        Returns:
            Hash keys of shape (n_samples, trials) as integers
        """
        if isinstance(W, np.ndarray):
            W = torch.from_numpy(W.astype(np.float32))

        # Move tensors to same device
        device = W.device
        hash_tensor = self.hash_tensor.to(device)
        V = self.V.to(device)

        # Project and binarize: sign(W @ H) -> {-1, 1} -> {0, 1}
        S = (torch.sign(W @ hash_tensor) + 1) / 2

        # Reshape to (n_samples, trials, len_trial)
        S = S.view(S.shape[0], self.trials, self.len_trial)

        # Binary to integer conversion: S @ [1, 2, 4, 8, ...]
        # Note: Keep S as float for CUDA compatibility (integer matmul not supported on CUDA)
        # then convert result to int
        H = torch.squeeze(S @ V.float(), dim=-1)

        # Convert to integer type
        if self.dtype_int == 'int64':
            H = H.to(torch.int64)
        else:
            H = H.to(torch.int32)

        return H

    def to(self, device: str) -> "CosimHasher":
        """
        Move hasher tensors to specified device.

        Args:
            device: Device string ('cuda', 'cpu', 'cuda:0', etc.)

        Returns:
            self for chaining
        """
        self.hash_tensor = self.hash_tensor.to(device)
        self.V = self.V.to(device)
        return self


class TimsHasher(CosimHasher):
    """
    Hasher optimized for TIMS (Trapped Ion Mobility Spectrometry) spectra.

    Extends CosimHasher with parameters suitable for mass spectrometry
    data with specific m/z resolution and range.

    Args:
        trials: Number of hash tables (default: 32)
        len_trial: Number of bits per hash (default: 20)
        seed: Random seed (default: 5671)
        resolution: Decimal resolution for m/z binning (default: 1)
        num_dalton: m/z range in Daltons (default: 10)

    Example:
        >>> hasher = TimsHasher(resolution=1, num_dalton=10)
        >>> # Creates hasher for 101-dimensional vectors (10 Da * 10^1 + 1)
    """

    def __init__(
        self,
        trials: int = 32,
        len_trial: int = 20,
        seed: int = 5671,
        resolution: int = 1,
        num_dalton: int = 10,
        backend: Optional[str] = None,  # Kept for backward compatibility, ignored
    ):
        self.resolution = resolution
        self.num_dalton = num_dalton

        # Calculate vector length from resolution and mass range
        res_factor = 10 ** resolution
        target_vector_length = num_dalton * res_factor + 1

        super().__init__(
            target_vector_length=target_vector_length,
            trials=trials,
            len_trial=len_trial,
            seed=seed,
            backend=backend,
        )

    def __repr__(self) -> str:
        return (
            f"TimsHasher(trials={self.trials}, len_trial={self.len_trial}, "
            f"seed={self.seed}, resolution={self.resolution}, "
            f"num_dalton={self.num_dalton}, backend='{self.backend}')"
        )


class SpectralHasher:
    """
    Hasher for fragment ion intensity spectra (e.g., from Prosit predictions).

    Optimized for the standard 174-dimensional Prosit output format
    (29 positions * 6 ion types).

    Args:
        trials: Number of hash tables (default: 64)
        len_trial: Number of bits per hash (default: 16)
        seed: Random seed (default: 42)
    """

    PROSIT_DIM = 174  # Standard Prosit output dimension

    def __init__(
        self,
        trials: int = 64,
        len_trial: int = 16,
        seed: int = 42,
        backend: Optional[str] = None,  # Kept for backward compatibility, ignored
    ):
        self._hasher = CosimHasher(
            target_vector_length=self.PROSIT_DIM,
            trials=trials,
            len_trial=len_trial,
            seed=seed,
            backend=backend,
        )

    def __repr__(self) -> str:
        return (
            f"SpectralHasher(trials={self._hasher.trials}, "
            f"len_trial={self._hasher.len_trial}, seed={self._hasher.seed}, "
            f"backend='{self._hasher.backend}')"
        )

    def calculate_keys(self, intensities):
        """
        Calculate hash keys for fragment intensity spectra.

        Args:
            intensities: Intensity vectors of shape (n_samples, 174)

        Returns:
            Hash keys of shape (n_samples, trials)
        """
        return self._hasher.calculate_keys(intensities)

    def to(self, device: str) -> "SpectralHasher":
        """Move to device (PyTorch only)."""
        self._hasher.to(device)
        return self
