"""Tests for loss functions."""

import pytest
import numpy as np
import torch

from imspy_predictors.losses import (
    masked_spectral_distance,
    masked_pearson_correlation_distance,
    masked_cosine_distance,
    masked_mse_loss,
    masked_l1_loss,
    MaskedSpectralAngleLoss,
    IntensityLoss,
    GaussianNLLLossWithConstraint,
    get_intensity_loss,
)


class TestMaskedSpectralDistance:
    """Test suite for masked_spectral_distance."""

    def test_identical_vectors(self):
        """Test that identical vectors have zero loss."""
        y_true = torch.tensor([[0.5, 0.3, 0.2, 0.0, 0.0]])
        y_pred = torch.tensor([[0.5, 0.3, 0.2, 0.0, 0.0]])

        loss = masked_spectral_distance(y_true, y_pred)
        assert loss.item() < 0.01  # Near zero

    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have high loss."""
        y_true = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        y_pred = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

        loss = masked_spectral_distance(y_true, y_pred)
        assert abs(loss.item() - 1.0) < 0.1  # Should be ~1.0

    def test_masking_with_minus_one(self):
        """Test that -1 values are properly masked."""
        # Valid positions only at indices 0, 1
        y_true = torch.tensor([[0.5, 0.5, -1.0, -1.0]])
        y_pred = torch.tensor([[0.5, 0.5, 0.9, 0.8]])  # Invalid positions have different values

        loss = masked_spectral_distance(y_true, y_pred)
        # Should be near zero because only matching valid positions contribute
        assert loss.item() < 0.01

    def test_batch_processing(self):
        """Test batch processing."""
        y_true = torch.rand(32, 174)
        y_pred = torch.rand(32, 174)

        loss = masked_spectral_distance(y_true, y_pred)
        assert loss.shape == ()  # Scalar
        assert 0 <= loss.item() <= 2

    def test_reduction_none(self):
        """Test no reduction returns per-sample losses."""
        y_true = torch.rand(10, 50)
        y_pred = torch.rand(10, 50)

        loss = masked_spectral_distance(y_true, y_pred, reduction="none")
        assert loss.shape == (10,)

    def test_reduction_sum(self):
        """Test sum reduction."""
        y_true = torch.rand(10, 50)
        y_pred = torch.rand(10, 50)

        loss_mean = masked_spectral_distance(y_true, y_pred, reduction="mean")
        loss_sum = masked_spectral_distance(y_true, y_pred, reduction="sum")

        assert abs(loss_sum.item() - 10 * loss_mean.item()) < 0.01


class TestMaskedPearsonCorrelation:
    """Test suite for masked_pearson_correlation_distance."""

    def test_perfect_correlation(self):
        """Test perfectly correlated vectors."""
        y_true = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        y_pred = torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0]])  # y = 2x

        loss = masked_pearson_correlation_distance(y_true, y_pred)
        assert loss.item() < 0.01  # 1 - r ≈ 0

    def test_no_correlation(self):
        """Test uncorrelated vectors."""
        y_true = torch.tensor([[1.0, 0.0, -1.0, 0.0, 1.0]])
        y_pred = torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0]])

        loss = masked_pearson_correlation_distance(y_true, y_pred)
        # Should be close to 1 (1 - 0)
        assert abs(loss.item() - 1.0) < 0.2

    def test_masking(self):
        """Test that -1 values are masked."""
        y_true = torch.tensor([[1.0, 2.0, 3.0, -1.0, -1.0]])
        y_pred = torch.tensor([[2.0, 4.0, 6.0, 0.0, 0.0]])

        loss = masked_pearson_correlation_distance(y_true, y_pred)
        assert loss.item() < 0.01


class TestMaskedCosineDistance:
    """Test suite for masked_cosine_distance."""

    def test_identical_vectors(self):
        """Test identical vectors."""
        y_true = torch.tensor([[0.5, 0.3, 0.2]])
        y_pred = y_true.clone()

        loss = masked_cosine_distance(y_true, y_pred)
        assert loss.item() < 0.01

    def test_opposite_vectors(self):
        """Test opposite vectors."""
        y_true = torch.tensor([[1.0, 0.0, 0.0]])
        y_pred = torch.tensor([[-1.0, 0.0, 0.0]])

        loss = masked_cosine_distance(y_true, y_pred)
        assert abs(loss.item() - 2.0) < 0.1


class TestMaskedMSELoss:
    """Test suite for masked_mse_loss."""

    def test_zero_loss(self):
        """Test zero loss for identical values."""
        y_true = torch.tensor([[0.5, 0.3, 0.2]])
        y_pred = y_true.clone()

        loss = masked_mse_loss(y_true, y_pred)
        assert loss.item() == 0.0

    def test_masking(self):
        """Test that -1 values are masked."""
        y_true = torch.tensor([[0.5, 0.3, -1.0]])
        y_pred = torch.tensor([[0.5, 0.3, 1.0]])  # Different at masked position

        loss = masked_mse_loss(y_true, y_pred)
        assert loss.item() == 0.0

    def test_partial_masking(self):
        """Test partial masking."""
        y_true = torch.tensor([[0.0, 1.0, -1.0, -1.0]])
        y_pred = torch.tensor([[0.1, 1.1, 0.5, 0.5]])

        loss = masked_mse_loss(y_true, y_pred)
        # Only first two positions contribute: (0.1^2 + 0.1^2) / 2 = 0.01
        expected = 0.01
        assert abs(loss.item() - expected) < 0.001


class TestMaskedL1Loss:
    """Test suite for masked_l1_loss."""

    def test_zero_loss(self):
        """Test zero loss for identical values."""
        y_true = torch.tensor([[0.5, 0.3, 0.2]])
        y_pred = y_true.clone()

        loss = masked_l1_loss(y_true, y_pred)
        assert loss.item() == 0.0

    def test_masking(self):
        """Test that -1 values are masked."""
        y_true = torch.tensor([[0.5, 0.3, -1.0]])
        y_pred = torch.tensor([[0.5, 0.3, 1.0]])

        loss = masked_l1_loss(y_true, y_pred)
        assert loss.item() == 0.0


class TestMaskedSpectralAngleLossModule:
    """Test suite for MaskedSpectralAngleLoss nn.Module."""

    def test_forward(self):
        """Test forward pass."""
        loss_fn = MaskedSpectralAngleLoss()

        y_true = torch.rand(16, 174)
        y_pred = torch.rand(16, 174)

        loss = loss_fn(y_pred, y_true)
        assert loss.shape == ()
        assert 0 <= loss.item() <= 2

    def test_gradient_flow(self):
        """Test gradients flow correctly."""
        loss_fn = MaskedSpectralAngleLoss()

        y_true = torch.rand(8, 100)
        y_pred = torch.rand(8, 100, requires_grad=True)

        loss = loss_fn(y_pred, y_true)
        loss.backward()

        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()


class TestIntensityLoss:
    """Test suite for IntensityLoss."""

    def test_default_weights(self):
        """Test default loss (spectral only)."""
        loss_fn = IntensityLoss()

        y_true = torch.rand(8, 174)
        y_pred = torch.rand(8, 174)

        losses = loss_fn(y_pred, y_true)

        assert "total" in losses
        assert "spectral_angle" in losses
        assert losses["total"].item() == losses["spectral_angle"].item()

    def test_combined_loss(self):
        """Test combined loss."""
        loss_fn = IntensityLoss(spectral_weight=1.0, l1_weight=0.1)

        y_true = torch.rand(8, 174)
        y_pred = torch.rand(8, 174)

        losses = loss_fn(y_pred, y_true)

        assert "spectral_angle" in losses
        assert "l1" in losses
        expected_total = losses["spectral_angle"] + 0.1 * losses["l1"]
        assert abs(losses["total"].item() - expected_total.item()) < 0.001

    def test_all_losses(self):
        """Test with all loss components."""
        loss_fn = IntensityLoss(
            spectral_weight=1.0,
            pearson_weight=0.5,
            l1_weight=0.1,
            mse_weight=0.05,
        )

        y_true = torch.rand(8, 174)
        y_pred = torch.rand(8, 174)

        losses = loss_fn(y_pred, y_true)

        assert "spectral_angle" in losses
        assert "pearson" in losses
        assert "l1" in losses
        assert "mse" in losses
        assert "total" in losses


class TestGaussianNLLLossWithConstraint:
    """Test suite for GaussianNLLLossWithConstraint."""

    def test_forward(self):
        """Test forward pass."""
        loss_fn = GaussianNLLLossWithConstraint()

        mean = torch.tensor([[1.0], [2.0], [3.0]])
        std = torch.tensor([[0.1], [0.2], [0.3]])
        target = torch.tensor([[1.1], [2.1], [2.9]])

        loss = loss_fn(mean, std, target)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_variance_clamping(self):
        """Test that variance is clamped."""
        loss_fn = GaussianNLLLossWithConstraint(min_var=1e-4, max_var=1e4)

        mean = torch.tensor([[1.0]])
        std_small = torch.tensor([[1e-10]])  # Very small std
        std_large = torch.tensor([[1e10]])  # Very large std
        target = torch.tensor([[1.0]])

        # Should not raise or produce NaN
        loss_small = loss_fn(mean, std_small, target)
        loss_large = loss_fn(mean, std_large, target)

        assert not torch.isnan(loss_small)
        assert not torch.isnan(loss_large)


class TestGetIntensityLoss:
    """Test suite for get_intensity_loss factory."""

    def test_spectral_angle(self):
        """Test spectral angle loss factory."""
        loss_fn = get_intensity_loss("spectral_angle")
        assert isinstance(loss_fn, MaskedSpectralAngleLoss)

    def test_combined(self):
        """Test combined loss factory."""
        loss_fn = get_intensity_loss("combined")
        assert isinstance(loss_fn, IntensityLoss)

    def test_invalid_type(self):
        """Test invalid loss type."""
        with pytest.raises(ValueError):
            get_intensity_loss("invalid")


class TestGradientFlow:
    """Test gradient flow through loss functions."""

    def test_spectral_gradient(self):
        """Test gradient flows through spectral loss."""
        y_true = torch.rand(4, 50)
        y_pred = torch.rand(4, 50, requires_grad=True)

        loss = masked_spectral_distance(y_true, y_pred)
        loss.backward()

        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()
        assert not torch.isinf(y_pred.grad).any()

    def test_pearson_gradient(self):
        """Test gradient flows through Pearson loss."""
        y_true = torch.rand(4, 50)
        y_pred = torch.rand(4, 50, requires_grad=True)

        loss = masked_pearson_correlation_distance(y_true, y_pred)
        loss.backward()

        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()

    def test_intensity_loss_gradient(self):
        """Test gradient flows through IntensityLoss."""
        loss_fn = IntensityLoss(spectral_weight=1.0, l1_weight=0.1)

        y_true = torch.rand(4, 174)
        y_pred = torch.rand(4, 174, requires_grad=True)

        losses = loss_fn(y_pred, y_true)
        losses["total"].backward()

        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_all_masked(self):
        """Test when all values are masked."""
        y_true = torch.full((4, 50), -1.0)
        y_pred = torch.rand(4, 50)

        # Should handle gracefully (may produce nan or zero)
        loss = masked_spectral_distance(y_true, y_pred)
        # Just check it doesn't crash

    def test_single_valid_element(self):
        """Test with single valid element per sample."""
        y_true = torch.tensor([[0.5, -1.0, -1.0, -1.0]])
        y_pred = torch.tensor([[0.5, 0.1, 0.2, 0.3]])

        loss = masked_spectral_distance(y_true, y_pred)
        # Should be near zero (single element, same value)
        assert loss.item() < 0.1

    def test_zero_vectors(self):
        """Test with zero vectors."""
        y_true = torch.zeros(4, 50)
        y_pred = torch.zeros(4, 50)

        # Should handle gracefully with epsilon
        loss = masked_spectral_distance(y_true, y_pred)
        assert not torch.isnan(loss)

    def test_cuda_if_available(self):
        """Test on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        y_true = torch.rand(16, 174).cuda()
        y_pred = torch.rand(16, 174).cuda()

        loss = masked_spectral_distance(y_true, y_pred)
        assert loss.device.type == "cuda"
        assert not torch.isnan(loss)
