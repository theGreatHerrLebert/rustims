"""Tests for predictor wrapper classes."""

import pytest
import numpy as np
import pandas as pd
import torch


class TestCCSPredictor:
    """Test suite for CCS/ion mobility predictors."""

    @pytest.fixture
    def ccs_model(self):
        """Create a PyTorch CCS model for testing."""
        from imspy_predictors.ccs.predictors import PyTorchCCSPredictor
        return PyTorchCCSPredictor(vocab_size=2200, architecture='gru', max_charge=5)

    def test_pytorch_ccs_predictor_creation(self, ccs_model):
        """Test PyTorch CCS predictor can be created."""
        assert ccs_model is not None

    def test_pytorch_ccs_forward(self, ccs_model):
        """Test forward pass of PyTorch CCS predictor."""
        batch_size = 4
        seq_len = 20

        mz = torch.rand(batch_size) * 1000 + 400
        charge_onehot = torch.zeros(batch_size, 5)
        charge_onehot[:, 2] = 1  # Charge 2
        sequences = torch.randint(0, 100, (batch_size, seq_len))

        ccs, residual = ccs_model(mz, charge_onehot, sequences)

        assert ccs.shape == (batch_size, 1)
        assert residual.shape == (batch_size, 1)

    def test_sqrt_slopes_intercepts(self):
        """Test fitting sqrt slopes and intercepts."""
        from imspy_predictors.ccs.predictors import get_sqrt_slopes_and_intercepts

        # Create synthetic data
        np.random.seed(42)
        n = 100
        mz = np.random.uniform(400, 2000, n)
        charge = np.random.choice([2, 3, 4], n)
        # Synthetic CCS following sqrt relationship
        ccs = 15 * np.sqrt(mz) + 50 + np.random.normal(0, 10, n)

        slopes, intercepts = get_sqrt_slopes_and_intercepts(mz, charge, ccs)

        assert len(slopes) == 4  # Charges 1-4 (with 0 for charge 1)
        assert len(intercepts) == 4


class TestRTPredictor:
    """Test suite for retention time predictors."""

    @pytest.fixture
    def rt_model(self):
        """Create a PyTorch RT model for testing."""
        from imspy_predictors.rt.predictors import PyTorchRTPredictor
        return PyTorchRTPredictor(vocab_size=2200, architecture='gru')

    def test_pytorch_rt_predictor_creation(self, rt_model):
        """Test PyTorch RT predictor can be created."""
        assert rt_model is not None

    def test_pytorch_rt_forward(self, rt_model):
        """Test forward pass of PyTorch RT predictor."""
        batch_size = 4
        seq_len = 20

        sequences = torch.randint(0, 100, (batch_size, seq_len))
        rt = rt_model(sequences)

        assert rt.shape == (batch_size, 1)


class TestChargePredictor:
    """Test suite for charge state predictors."""

    @pytest.fixture
    def charge_model(self):
        """Create a PyTorch charge model for testing."""
        from imspy_predictors.ionization.predictors import PyTorchChargePredictor
        return PyTorchChargePredictor(vocab_size=2200, max_charge=4, architecture='gru')

    def test_pytorch_charge_predictor_creation(self, charge_model):
        """Test PyTorch charge predictor can be created."""
        assert charge_model is not None

    def test_pytorch_charge_forward(self, charge_model):
        """Test forward pass of PyTorch charge predictor."""
        batch_size = 4
        seq_len = 20

        sequences = torch.randint(0, 100, (batch_size, seq_len))
        probs = charge_model(sequences)

        assert probs.shape == (batch_size, 4)
        # Should be valid probabilities
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)


class TestBinomialChargeModel:
    """Test suite for binomial charge state model."""

    def test_binomial_model_creation(self):
        """Test binomial charge model can be created."""
        from imspy_predictors.ionization.predictors import BinomialChargeStateDistributionModel
        model = BinomialChargeStateDistributionModel(
            charged_probability=0.8,
            max_charge=4,
        )
        assert model is not None

    def test_binomial_simulate_ionizations(self):
        """Test simulating ionizations with binomial model."""
        from imspy_predictors.ionization.predictors import BinomialChargeStateDistributionModel
        model = BinomialChargeStateDistributionModel(
            charged_probability=0.8,
            max_charge=4,
        )

        sequences = ["PEPTIDE", "SEQUENCE", "PROTEIN"]
        probs = model.simulate_ionizations(sequences)

        assert probs.shape[0] == 3
        assert probs.shape[1] == 5  # 0 to max_charge


class TestLinearMap:
    """Test suite for linear mapping utility."""

    def test_linear_map(self):
        """Test linear mapping function."""
        from imspy_predictors.rt.predictors import linear_map

        # Map [0, 10] to [0, 100]
        result = linear_map(5, 0, 10, 0, 100)
        assert result == 50

        # Map [0, 60] to [0, 1]
        result = linear_map(30, 0, 60, 0, 1)
        assert result == 0.5

    def test_linear_map_array(self):
        """Test linear mapping with arrays."""
        from imspy_predictors.rt.predictors import linear_map

        x = np.array([0, 5, 10])
        result = linear_map(x, 0, 10, 0, 100)

        np.testing.assert_array_equal(result, [0, 50, 100])


class TestPredictorWrapperInterface:
    """Test the unified interface of predictor wrappers."""

    def test_ccs_wrapper_interface(self):
        """Test DeepPeptideIonMobilityApex interface."""
        # Note: This test requires a trained model or mocked model
        # For now, just test the class can be imported
        from imspy_predictors.ccs.predictors import DeepPeptideIonMobilityApex
        assert DeepPeptideIonMobilityApex is not None

    def test_rt_wrapper_interface(self):
        """Test DeepChromatographyApex interface."""
        from imspy_predictors.rt.predictors import DeepChromatographyApex
        assert DeepChromatographyApex is not None

    def test_charge_wrapper_interface(self):
        """Test DeepChargeStateDistribution interface."""
        from imspy_predictors.ionization.predictors import DeepChargeStateDistribution
        assert DeepChargeStateDistribution is not None
