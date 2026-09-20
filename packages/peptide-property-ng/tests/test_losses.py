"""Tests for the multi-task losses and degenerate-target handling."""
import torch

from peptide_property_ng.losses import MultiTaskLoss, intensity_signal_mask, masked_spectral_angle


def test_spectral_angle_identical_is_zero():
    x = torch.rand(4, 6, 6)
    sa = masked_spectral_angle(x, x.clone())
    assert torch.allclose(sa, torch.zeros(4), atol=1e-3)


def test_spectral_angle_fully_masked_is_finite():
    """A target with every entry masked (-1) must not produce NaN."""
    sa = masked_spectral_angle(torch.rand(2, 5, 6), torch.full((2, 5, 6), -1.0))
    assert torch.isfinite(sa).all()


def test_signal_mask():
    target = torch.zeros(3, 4, 6)
    target[0, 0, 0] = 0.5   # sample 0 has an observed peak
    target[2] = -1.0        # sample 2 fully masked -> degenerate
    assert intensity_signal_mask(target).tolist() == [True, False, False]


def test_loss_skips_degenerate_intensity_samples():
    """A batch mixing real and degenerate intensity targets yields a finite loss."""
    target = torch.zeros(4, 5, 6)
    target[0, 1, 0] = 1.0  # only sample 0 carries signal
    total, parts = MultiTaskLoss()({"intensity": torch.rand(4, 5, 6)},
                                   {"intensity_target": target})
    assert "intensity" in parts
    assert torch.isfinite(torch.tensor(parts["intensity"]))
