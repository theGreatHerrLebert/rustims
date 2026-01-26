"""
Gaussian Mixture Model implementations for clustering.

This module provides GMM implementations for clustering spectral data using PyTorch.
"""

from typing import Optional, Union
import numpy as np

# Import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.distributions import (
        MixtureSameFamily,
        Categorical,
        Independent,
        Normal,
    )
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


class GaussianMixtureModel:
    """
    Gaussian Mixture Model for clustering.

    Args:
        num_components: Number of Gaussian components
        data_dim: Dimensionality of the data
        prior_stddevs: Prior knowledge about cluster standard deviations (optional)
        data: Training data for initializing means by random selection (optional)
        init_means: Explicit initial means for components (optional)
        init_stds: Explicit initial standard deviations (optional)

    Example:
        >>> gmm = GaussianMixtureModel(num_components=3, data_dim=2)
        >>> gmm.fit(training_data, num_steps=200)
        >>> cluster_ids = gmm.predict(new_data)
        >>> probabilities = gmm.predict_proba(new_data)
    """

    def __init__(
        self,
        num_components: int,
        data_dim: int,
        prior_stddevs: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
        init_means: Optional[np.ndarray] = None,
        init_stds: Optional[np.ndarray] = None,
        backend: Optional[str] = None,  # Kept for backward compatibility, ignored
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GaussianMixtureModel. Install with: pip install torch")

        self.num_components = num_components
        self.data_dim = data_dim
        self.backend = 'torch'  # Always torch

        # Initialize locations (means)
        if init_means is not None:
            assert init_means.shape == (num_components, data_dim), (
                f"init_means should have shape [{num_components}, {data_dim}], "
                f"but got {init_means.shape}"
            )
            init_locs = torch.tensor(init_means, dtype=torch.float32)
        elif data is not None:
            indices = np.random.choice(data.shape[0], size=num_components, replace=True)
            init_locs = torch.tensor(data[indices], dtype=torch.float32)
        else:
            init_locs = torch.randn(num_components, data_dim)

        self.locs = nn.Parameter(init_locs)

        # Initialize scales (log of standard deviations)
        if init_stds is not None:
            init_stds_vals = np.tile(init_stds, (num_components, 1))
        else:
            init_stds_default = np.array([[3.0, 0.01, 0.01]])
            # Adjust default if data_dim != 3
            if data_dim != 3:
                init_stds_default = np.ones((1, data_dim)) * 0.1
            init_stds_vals = np.tile(init_stds_default, (num_components, 1))

        self.scales = nn.Parameter(torch.log(torch.tensor(init_stds_vals, dtype=torch.float32)))

        # Initialize mixture weights (logits)
        self.weights = nn.Parameter(torch.ones(num_components))

        # Prior standard deviations for regularization
        if prior_stddevs is not None:
            self.prior_stddevs = torch.tensor(
                np.tile(prior_stddevs, (num_components, 1)), dtype=torch.float32
            )
        else:
            self.prior_stddevs = None

        # Device tracking
        self._device = 'cpu'

    def __repr__(self) -> str:
        return (
            f"GaussianMixtureModel(num_components={self.num_components}, "
            f"data_dim={self.data_dim})"
        )

    def __call__(
        self, data: Union[np.ndarray, "torch.Tensor"]
    ) -> "torch.Tensor":
        """
        Calculate the log likelihood of the data given current GMM parameters.

        Args:
            data: Input data of shape (n_samples, data_dim)

        Returns:
            Log probabilities of shape (n_samples,)
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self._device)

        gmm = self._mixture()
        return gmm.log_prob(data)

    def _mixture(self) -> "MixtureSameFamily":
        """Create mixture distribution from current parameters."""
        # Component distribution: Independent Normal for each dimension
        component_dist = Independent(
            Normal(loc=self.locs, scale=torch.exp(self.scales)),
            reinterpreted_batch_ndims=1,
        )

        # Mixture distribution
        mix_dist = Categorical(logits=torch.log_softmax(self.weights, dim=0))

        return MixtureSameFamily(mix_dist, component_dist)

    def fit(
        self,
        data: np.ndarray,
        weights: Optional[np.ndarray] = None,
        num_steps: int = 200,
        learning_rate: float = 0.05,
        lambda_scale: float = 0.01,
        verbose: bool = True,
    ) -> None:
        """
        Fit the Gaussian Mixture Model to the data.

        Args:
            data: Input data of shape (n_samples, data_dim)
            weights: Optional sample weights
            num_steps: Number of optimization steps
            learning_rate: Learning rate for optimizer
            lambda_scale: Regularization strength for scale parameters
            verbose: Whether to print progress
        """
        data_tensor = torch.tensor(data, dtype=torch.float32, device=self._device)

        if weights is None:
            weights_tensor = torch.ones(len(data), device=self._device)
        else:
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self._device)

        # Move parameters to device
        self.locs.data = self.locs.data.to(self._device)
        self.scales.data = self.scales.data.to(self._device)
        self.weights.data = self.weights.data.to(self._device)
        if self.prior_stddevs is not None:
            self.prior_stddevs = self.prior_stddevs.to(self._device)

        optimizer = torch.optim.Adam(
            [self.locs, self.scales, self.weights],
            lr=learning_rate,
        )

        for step in range(num_steps):
            optimizer.zero_grad()

            # Calculate negative log likelihood
            log_likelihood = self(data_tensor)
            loss = -torch.sum(log_likelihood * weights_tensor)

            # Add regularization based on prior scales if provided
            if self.prior_stddevs is not None:
                scale_diff = torch.exp(self.scales) - self.prior_stddevs
                reg_loss = lambda_scale * torch.sum(scale_diff * scale_diff)
                loss = loss + reg_loss

            loss.backward()
            optimizer.step()

            if step % 100 == 0 and verbose:
                print(f"step: {step}, log-loss: {loss.item():.4f}")

    @property
    def means(self) -> np.ndarray:
        """Returns the means of the Gaussian components."""
        return self.locs.detach().cpu().numpy()

    @property
    def variances(self) -> np.ndarray:
        """Returns the variances (squared scales) of the Gaussian components."""
        return torch.exp(2 * self.scales).detach().cpu().numpy()

    @property
    def stddevs(self) -> np.ndarray:
        """Returns the standard deviations of the Gaussian components."""
        return torch.exp(self.scales).detach().cpu().numpy()

    @property
    def mixture_weights(self) -> np.ndarray:
        """Returns the normalized mixture weights."""
        return torch.softmax(self.weights, dim=0).detach().cpu().numpy()

    def predict_proba(
        self, data: Union[np.ndarray, "torch.Tensor"]
    ) -> np.ndarray:
        """
        Predict cluster membership probabilities for each sample.

        Args:
            data: Input data of shape (n_samples, data_dim)

        Returns:
            Probabilities of shape (n_samples, num_components)
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            # Get component distribution log probs for each data point
            # data shape: (n_samples, data_dim)
            # locs shape: (num_components, data_dim)

            # Expand data to (n_samples, 1, data_dim)
            data_expanded = data.unsqueeze(1)

            # Compute log probs under each component
            component_dist = Independent(
                Normal(loc=self.locs, scale=torch.exp(self.scales)),
                reinterpreted_batch_ndims=1,
            )
            log_probs = component_dist.log_prob(data_expanded)  # (n_samples, num_components)

            # Use log-sum-exp trick for numerical stability
            # softmax(log_probs) = exp(log_probs - logsumexp(log_probs))
            log_probs_max = log_probs.max(dim=-1, keepdim=True).values
            log_probs_shifted = log_probs - log_probs_max
            probs = torch.softmax(log_probs_shifted, dim=-1)

            return probs.cpu().numpy()

    def predict(self, data: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """
        Predict cluster assignments for each sample.

        Args:
            data: Input data of shape (n_samples, data_dim)

        Returns:
            Cluster indices of shape (n_samples,)
        """
        return np.argmax(self.predict_proba(data), axis=1)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample from the Gaussian Mixture Model.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Samples of shape (n_samples, data_dim)
        """
        with torch.no_grad():
            gmm = self._mixture()
            samples = gmm.sample((n_samples,))
            return samples.cpu().numpy()

    def to(self, device: str) -> "GaussianMixtureModel":
        """
        Move model to specified device.

        Args:
            device: Device string ('cuda', 'cpu', 'cuda:0', etc.)

        Returns:
            self for chaining
        """
        self._device = device
        self.locs.data = self.locs.data.to(device)
        self.scales.data = self.scales.data.to(device)
        self.weights.data = self.weights.data.to(device)
        if self.prior_stddevs is not None:
            self.prior_stddevs = self.prior_stddevs.to(device)

        return self

    def state_dict(self) -> dict:
        """
        Get model state as dictionary (for saving).

        Returns:
            Dictionary with model parameters
        """
        return {
            'locs': self.locs.detach().cpu().numpy(),
            'scales': self.scales.detach().cpu().numpy(),
            'weights': self.weights.detach().cpu().numpy(),
            'prior_stddevs': self.prior_stddevs.cpu().numpy() if self.prior_stddevs is not None else None,
            'num_components': self.num_components,
            'data_dim': self.data_dim,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Load model state from dictionary.

        Args:
            state: Dictionary with model parameters
        """
        self.locs.data = torch.tensor(state['locs'], dtype=torch.float32)
        self.scales.data = torch.tensor(state['scales'], dtype=torch.float32)
        self.weights.data = torch.tensor(state['weights'], dtype=torch.float32)
        if state['prior_stddevs'] is not None:
            self.prior_stddevs = torch.tensor(state['prior_stddevs'], dtype=torch.float32)

    @classmethod
    def from_state_dict(cls, state: dict) -> "GaussianMixtureModel":
        """
        Create a GaussianMixtureModel from saved state.

        Args:
            state: Dictionary with model parameters

        Returns:
            New GaussianMixtureModel instance
        """
        model = cls(
            num_components=state['num_components'],
            data_dim=state['data_dim'],
            init_means=state['locs'],
            init_stds=np.exp(state['scales']),
        )
        model.load_state_dict(state)
        return model
