"""
Loss functions for peptide property prediction.

This module provides task-specific loss functions, with particular focus
on intensity prediction which requires careful handling of invalid ions.

Key Features:
- Masked spectral angle loss for intensity prediction (dlomix-compatible)
- Masked Pearson correlation loss
- Combined loss functions
- Support for -1 marker convention (invalid peaks)

Based on implementations from dlomix (wilhelm-lab/dlomix):
https://github.com/wilhelm-lab/dlomix/blob/main/src/dlomix/losses/intensity_torch.py
"""

from typing import Optional, Dict, Literal, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Prosit output format constants
PROSIT_MAX_SEQ_LEN = 29  # Maximum peptide length - 1 (30 AA max)
PROSIT_NUM_ION_TYPES = 6  # b1+, b2+, b3+, y1+, y2+, y3+
PROSIT_NUM_OUTPUTS = PROSIT_MAX_SEQ_LEN * PROSIT_NUM_ION_TYPES  # 174


def masked_spectral_distance(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calculate masked spectral distance between true and predicted intensity vectors.

    This is the primary loss function for training intensity prediction models.
    Invalid peaks are marked with -1 in y_true and are automatically masked out.

    The spectral angle measures the angular similarity between two vectors:
    - arccos(1*1 + 0*0) = 0 -> SA = 0 -> high correlation (perfect match)
    - arccos(0*1 + 1*0) = π/2 -> SA = 1 -> low correlation (orthogonal)

    Based on dlomix implementation.

    Args:
        y_true: Target intensities of shape (batch, num_ions). Invalid positions
                should be marked with -1.
        y_pred: Predicted intensities of shape (batch, num_ions)
        reduction: "mean", "sum", or "none"

    Returns:
        Spectral distance loss in range [0, 2]. Lower is better.
    """
    # Numerical stability constant
    epsilon = 1e-7

    # Masking: multiply by (y_true + 1) to zero out positions where y_true == -1
    # This works because: -1 + 1 = 0, so invalid positions get multiplied by 0
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    # L2 normalize along last dimension
    true_norm = F.normalize(true_masked, p=2, dim=-1)
    pred_norm = F.normalize(pred_masked, p=2, dim=-1)

    # Spectral angle: arccos of dot product
    # (ions with higher intensities contribute more)
    product = (pred_norm * true_norm).sum(dim=-1)
    product = torch.clamp(product, -1.0 + epsilon, 1.0 - epsilon)
    arccos = torch.arccos(product)

    # Normalize to [0, 2] range
    batch_losses = 2 * arccos / np.pi

    if reduction == "mean":
        return batch_losses.mean()
    elif reduction == "sum":
        return batch_losses.sum()
    else:
        return batch_losses


def masked_pearson_correlation_distance(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calculate masked Pearson correlation distance between intensity vectors.

    Returns 1 - r, where r is the Pearson correlation coefficient.
    Invalid peaks (marked as -1) are excluded from the calculation.

    Based on dlomix implementation.

    Args:
        y_true: Target intensities of shape (batch, num_ions)
        y_pred: Predicted intensities of shape (batch, num_ions)
        reduction: "mean", "sum", or "none"

    Returns:
        1 - Pearson correlation. Range [0, 2]. Lower is better.
    """
    epsilon = 1e-7

    # Masking: zero out invalid positions
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    # Compute per-sample Pearson correlation
    # Center the data
    mx = true_masked.mean(dim=-1, keepdim=True)
    my = pred_masked.mean(dim=-1, keepdim=True)
    xm = true_masked - mx
    ym = pred_masked - my

    # Correlation = cov(x,y) / (std(x) * std(y))
    r_num = (xm * ym).sum(dim=-1)
    r_den = (xm ** 2).sum(dim=-1).sqrt() * (ym ** 2).sum(dim=-1).sqrt() + epsilon

    batch_correlations = r_num / r_den
    batch_losses = 1 - batch_correlations

    if reduction == "mean":
        return batch_losses.mean()
    elif reduction == "sum":
        return batch_losses.sum()
    else:
        return batch_losses


def masked_cosine_distance(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calculate masked cosine distance between intensity vectors.

    Similar to spectral distance but without the arccos transformation.
    Returns 1 - cos_sim, so 0 = perfect match, 2 = opposite vectors.

    Args:
        y_true: Target intensities of shape (batch, num_ions)
        y_pred: Predicted intensities of shape (batch, num_ions)
        reduction: "mean", "sum", or "none"

    Returns:
        1 - cosine similarity. Range [0, 2]. Lower is better.
    """
    epsilon = 1e-7

    # Masking
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    # Cosine similarity
    true_norm = F.normalize(true_masked, p=2, dim=-1)
    pred_norm = F.normalize(pred_masked, p=2, dim=-1)

    cos_sim = (pred_norm * true_norm).sum(dim=-1)
    batch_losses = 1 - cos_sim

    if reduction == "mean":
        return batch_losses.mean()
    elif reduction == "sum":
        return batch_losses.sum()
    else:
        return batch_losses


def masked_mse_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calculate masked mean squared error loss.

    Invalid positions (y_true == -1) are excluded from the loss.

    Args:
        y_true: Target intensities of shape (batch, num_ions)
        y_pred: Predicted intensities of shape (batch, num_ions)
        reduction: "mean", "sum", or "none"

    Returns:
        MSE loss value
    """
    # Create mask: True for valid positions
    mask = y_true >= 0

    # Compute squared error only at valid positions
    diff_sq = (y_pred - y_true) ** 2
    masked_diff = diff_sq * mask.float()

    if reduction == "mean":
        return masked_diff.sum() / mask.float().sum().clamp(min=1)
    elif reduction == "sum":
        return masked_diff.sum()
    else:
        return masked_diff


def masked_l1_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calculate masked L1 (absolute) loss.

    Invalid positions (y_true == -1) are excluded from the loss.

    Args:
        y_true: Target intensities of shape (batch, num_ions)
        y_pred: Predicted intensities of shape (batch, num_ions)
        reduction: "mean", "sum", or "none"

    Returns:
        L1 loss value
    """
    mask = y_true >= 0

    diff = (y_pred - y_true).abs()
    masked_diff = diff * mask.float()

    if reduction == "mean":
        return masked_diff.sum() / mask.float().sum().clamp(min=1)
    elif reduction == "sum":
        return masked_diff.sum()
    else:
        return masked_diff


class MaskedSpectralAngleLoss(nn.Module):
    """
    Masked spectral angle loss as a PyTorch module.

    Wraps masked_spectral_distance for use in training pipelines.

    Example:
        >>> loss_fn = MaskedSpectralAngleLoss()
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return masked_spectral_distance(y_true, y_pred, reduction=self.reduction)


class IntensityLoss(nn.Module):
    """
    Combined loss function for fragment intensity prediction.

    Combines spectral angle loss with optional additional loss terms.
    Uses dlomix-compatible masking (invalid peaks marked as -1).

    Args:
        spectral_weight: Weight for spectral angle loss (default: 1.0)
        pearson_weight: Weight for Pearson correlation loss (default: 0.0)
        l1_weight: Weight for L1 loss (default: 0.0)
        mse_weight: Weight for MSE loss (default: 0.0)

    Example:
        >>> loss_fn = IntensityLoss(spectral_weight=1.0, l1_weight=0.1)
        >>> losses = loss_fn(predictions, targets)
        >>> total_loss = losses["total"]
    """

    def __init__(
        self,
        spectral_weight: float = 1.0,
        pearson_weight: float = 0.0,
        l1_weight: float = 0.0,
        mse_weight: float = 0.0,
    ):
        super().__init__()
        self.spectral_weight = spectral_weight
        self.pearson_weight = pearson_weight
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute intensity loss.

        Args:
            y_pred: Predicted intensities (batch, num_ions)
            y_true: Target intensities (batch, num_ions), -1 for invalid

        Returns:
            Dictionary with 'total' and individual loss components
        """
        losses = {}
        total = torch.tensor(0.0, device=y_pred.device)

        if self.spectral_weight > 0:
            sa_loss = masked_spectral_distance(y_true, y_pred)
            losses["spectral_angle"] = sa_loss
            total = total + self.spectral_weight * sa_loss

        if self.pearson_weight > 0:
            pearson_loss = masked_pearson_correlation_distance(y_true, y_pred)
            losses["pearson"] = pearson_loss
            total = total + self.pearson_weight * pearson_loss

        if self.l1_weight > 0:
            l1_loss = masked_l1_loss(y_true, y_pred)
            losses["l1"] = l1_loss
            total = total + self.l1_weight * l1_loss

        if self.mse_weight > 0:
            mse_loss = masked_mse_loss(y_true, y_pred)
            losses["mse"] = mse_loss
            total = total + self.mse_weight * mse_loss

        losses["total"] = total
        return losses


class GaussianNLLLossWithConstraint(nn.Module):
    """
    Gaussian NLL loss with variance constraints.

    Used for uncertainty-aware predictions (e.g., CCS with std).
    Adds regularization to prevent variance collapse or explosion.

    Args:
        min_var: Minimum allowed variance (default: 1e-4)
        max_var: Maximum allowed variance (default: 1e4)
        var_reg_weight: Weight for variance regularization (default: 0.01)
    """

    def __init__(
        self,
        min_var: float = 1e-4,
        max_var: float = 1e4,
        var_reg_weight: float = 0.01,
    ):
        super().__init__()
        self.min_var = min_var
        self.max_var = max_var
        self.var_reg_weight = var_reg_weight

    def forward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL loss with variance regularization.

        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            target: Target values

        Returns:
            Loss value
        """
        var = std ** 2
        var = torch.clamp(var, self.min_var, self.max_var)

        nll = F.gaussian_nll_loss(mean, target, var)

        # Regularization on log variance to prevent extreme values
        log_var = torch.log(var)
        var_reg = (log_var ** 2).mean()

        return nll + self.var_reg_weight * var_reg


def get_intensity_loss(
    loss_type: Literal["spectral_angle", "combined", "pearson", "cosine"] = "spectral_angle",
    **kwargs,
) -> nn.Module:
    """
    Factory function to get an intensity loss function.

    Args:
        loss_type: Type of loss to use
            - "spectral_angle": Pure spectral angle loss (dlomix default)
            - "combined": Spectral angle + L1 regularization
            - "pearson": Pearson correlation distance
            - "cosine": Cosine distance (simpler than spectral angle)
        **kwargs: Additional arguments passed to IntensityLoss

    Returns:
        Loss module
    """
    if loss_type == "spectral_angle":
        return MaskedSpectralAngleLoss()
    elif loss_type == "combined":
        return IntensityLoss(spectral_weight=1.0, l1_weight=0.1, **kwargs)
    elif loss_type == "pearson":
        return IntensityLoss(spectral_weight=0.0, pearson_weight=1.0, **kwargs)
    elif loss_type == "cosine":
        class CosineLoss(nn.Module):
            def forward(self, y_pred, y_true):
                return masked_cosine_distance(y_true, y_pred)
        return CosineLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Convenience aliases for dlomix compatibility
spectral_angle_loss = masked_spectral_distance
pearson_loss = masked_pearson_correlation_distance
