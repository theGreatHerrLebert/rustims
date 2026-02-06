"""Tests for Gaussian Mixture Model implementations."""

import pytest
import numpy as np

# Import directly from mixture module to avoid imspy_core dependency
from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE


class TestGaussianMixtureModel:
    """Test suite for GaussianMixtureModel."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data with 3 clusters."""
        np.random.seed(42)
        # Generate 3 well-separated clusters
        cluster1 = np.random.randn(100, 2) * 0.5 + np.array([0, 0])
        cluster2 = np.random.randn(100, 2) * 0.5 + np.array([5, 5])
        cluster3 = np.random.randn(100, 2) * 0.5 + np.array([5, 0])
        return np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)

    @pytest.fixture
    def gmm(self):
        """Create a GMM for testing."""
        from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE
        return GaussianMixtureModel(
            num_components=3,
            data_dim=2,
            backend='torch',
        )

    def test_gmm_creation(self, gmm):
        """Test GMM can be created."""
        assert gmm is not None
        assert gmm.num_components == 3
        assert gmm.data_dim == 2
        assert gmm.backend == 'torch'

    def test_gmm_with_init_means(self):
        """Test GMM creation with explicit initial means."""
        from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE

        init_means = np.array([
            [0.0, 0.0],
            [5.0, 5.0],
            [5.0, 0.0],
        ], dtype=np.float32)

        gmm = GaussianMixtureModel(
            num_components=3,
            data_dim=2,
            init_means=init_means,
            backend='torch',
        )

        # Means should match initial values
        np.testing.assert_array_almost_equal(gmm.means, init_means)

    def test_gmm_with_data_init(self, sample_data):
        """Test GMM creation initializing means from data."""
        from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE

        gmm = GaussianMixtureModel(
            num_components=3,
            data_dim=2,
            data=sample_data,
            backend='torch',
        )

        # Means should be within data range
        assert gmm.means.shape == (3, 2)
        assert gmm.means.min() >= sample_data.min() - 1
        assert gmm.means.max() <= sample_data.max() + 1

    def test_log_prob(self, gmm, sample_data):
        """Test log probability calculation."""
        log_probs = gmm(sample_data)

        assert log_probs.shape == (len(sample_data),)
        # Log probs should be negative (or close to zero for high-density regions)
        assert log_probs.max() < 10  # Not unreasonably high

    def test_predict_proba(self, gmm, sample_data):
        """Test cluster membership probability prediction."""
        probs = gmm.predict_proba(sample_data)

        assert probs.shape == (len(sample_data), 3)
        # Should be valid probabilities
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        # Each row should sum to approximately 1
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(len(sample_data)), decimal=5)

    def test_predict(self, gmm, sample_data):
        """Test cluster assignment prediction."""
        predictions = gmm.predict(sample_data)

        assert predictions.shape == (len(sample_data),)
        # Should be integers in [0, num_components)
        assert predictions.dtype in [np.int32, np.int64]
        assert predictions.min() >= 0
        assert predictions.max() < 3

    def test_sample(self, gmm):
        """Test sampling from GMM."""
        n_samples = 100
        samples = gmm.sample(n_samples)

        assert samples.shape == (n_samples, 2)

    def test_fit(self, gmm, sample_data):
        """Test fitting GMM to data."""
        # Fit with reduced steps for speed
        gmm.fit(sample_data, num_steps=50, verbose=False)

        # After fitting, predictions should somewhat match the clusters
        predictions = gmm.predict(sample_data)
        assert len(np.unique(predictions)) > 1  # Should find multiple clusters

    def test_fit_with_weights(self, gmm, sample_data):
        """Test fitting GMM with sample weights."""
        weights = np.ones(len(sample_data))
        weights[:100] = 2.0  # Weight first cluster more

        gmm.fit(sample_data, weights=weights, num_steps=50, verbose=False)

        # Should still produce valid predictions
        predictions = gmm.predict(sample_data)
        assert len(predictions) == len(sample_data)

    def test_properties(self, gmm):
        """Test property accessors."""
        # Means
        means = gmm.means
        assert means.shape == (3, 2)

        # Variances
        variances = gmm.variances
        assert variances.shape == (3, 2)
        assert (variances > 0).all()

        # Standard deviations
        stddevs = gmm.stddevs
        assert stddevs.shape == (3, 2)
        assert (stddevs > 0).all()

        # Mixture weights
        weights = gmm.mixture_weights
        assert weights.shape == (3,)
        assert (weights >= 0).all()
        np.testing.assert_almost_equal(weights.sum(), 1.0)

    def test_state_dict_save_load(self, gmm, sample_data):
        """Test saving and loading model state."""
        # Fit the model first
        gmm.fit(sample_data, num_steps=50, verbose=False)

        # Get predictions before saving
        original_predictions = gmm.predict(sample_data)

        # Save state
        state = gmm.state_dict()
        assert 'locs' in state
        assert 'scales' in state
        assert 'weights' in state
        assert state['num_components'] == 3
        assert state['data_dim'] == 2

        # Create new model and load state
        from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE
        gmm2 = GaussianMixtureModel(num_components=3, data_dim=2, backend='torch')
        gmm2.load_state_dict(state)

        # Predictions should match
        loaded_predictions = gmm2.predict(sample_data)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_from_state_dict(self, gmm, sample_data):
        """Test creating GMM from saved state."""
        # Fit the model
        gmm.fit(sample_data, num_steps=50, verbose=False)

        # Save state
        state = gmm.state_dict()

        # Create new model from state
        from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE
        gmm2 = GaussianMixtureModel.from_state_dict(state)

        # Should match original
        assert gmm2.num_components == gmm.num_components
        assert gmm2.data_dim == gmm.data_dim
        np.testing.assert_array_almost_equal(gmm2.means, gmm.means)

    def test_repr(self, gmm):
        """Test string representation."""
        repr_str = repr(gmm)
        assert "GaussianMixtureModel" in repr_str
        assert "num_components=3" in repr_str
        assert "data_dim=2" in repr_str

    def test_to_device(self, gmm):
        """Test moving model to device."""
        gmm.to('cpu')
        assert gmm._device == 'cpu'

    @pytest.mark.skipif(
        True,  # Only run if CUDA available
        reason="CUDA test skipped in general testing"
    )
    def test_cuda_support(self, sample_data):
        """Test GMM on CUDA device."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE
        gmm = GaussianMixtureModel(num_components=3, data_dim=2, backend='torch')
        gmm.to('cuda')

        gmm.fit(sample_data, num_steps=10, verbose=False)
        predictions = gmm.predict(sample_data)
        assert len(predictions) == len(sample_data)


class TestGMMWithPriors:
    """Test GMM with prior standard deviations."""

    def test_gmm_with_prior_stddevs(self):
        """Test GMM with prior standard deviations for regularization."""
        from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE

        prior_stddevs = np.array([[0.5, 0.5]], dtype=np.float32)

        gmm = GaussianMixtureModel(
            num_components=3,
            data_dim=2,
            prior_stddevs=prior_stddevs,
            backend='torch',
        )

        assert gmm.prior_stddevs is not None

        # Generate some data and fit
        np.random.seed(42)
        data = np.random.randn(100, 2).astype(np.float32)
        gmm.fit(data, num_steps=50, lambda_scale=0.1, verbose=False)

        # Model should still work
        predictions = gmm.predict(data)
        assert len(predictions) == 100


class TestGMMDifferentDimensions:
    """Test GMM with different data dimensions."""

    @pytest.mark.parametrize("data_dim", [1, 3, 5, 10])
    def test_gmm_different_dims(self, data_dim):
        """Test GMM with various data dimensions."""
        from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE

        np.random.seed(42)
        data = np.random.randn(200, data_dim).astype(np.float32)

        gmm = GaussianMixtureModel(
            num_components=2,
            data_dim=data_dim,
            backend='torch',
        )

        # Should be able to fit and predict
        gmm.fit(data, num_steps=20, verbose=False)
        predictions = gmm.predict(data)
        assert predictions.shape == (200,)

    @pytest.mark.parametrize("num_components", [2, 5, 10])
    def test_gmm_different_components(self, num_components):
        """Test GMM with various numbers of components."""
        from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE

        np.random.seed(42)
        data = np.random.randn(200, 2).astype(np.float32)

        gmm = GaussianMixtureModel(
            num_components=num_components,
            data_dim=2,
            backend='torch',
        )

        gmm.fit(data, num_steps=20, verbose=False)
        probs = gmm.predict_proba(data)
        assert probs.shape == (200, num_components)


class TestGMMBackendSelection:
    """Test GMM backend selection."""

    def test_torch_backend_explicit(self):
        """Test explicit PyTorch backend selection."""
        from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE, TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        gmm = GaussianMixtureModel(
            num_components=2,
            data_dim=2,
            backend='torch',
        )
        assert gmm.backend == 'torch'

    def test_invalid_backend_ignored(self):
        """Test that backend parameter is accepted but ignored (always torch)."""
        from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE

        # backend parameter is kept for backward compatibility but ignored
        gmm = GaussianMixtureModel(
            num_components=2,
            data_dim=2,
            backend='invalid_backend',
        )
        assert gmm.backend == 'torch'

    def test_init_means_shape_validation(self):
        """Test that init_means shape is validated."""
        from imspy_predictors.mixture import GaussianMixtureModel, TORCH_AVAILABLE

        # Wrong shape should raise
        wrong_shape_means = np.array([[0, 0, 0], [1, 1, 1]])  # 2x3 instead of 2x2

        with pytest.raises(AssertionError):
            GaussianMixtureModel(
                num_components=2,
                data_dim=2,  # Expects 2D data
                init_means=wrong_shape_means,
                backend='torch',
            )
