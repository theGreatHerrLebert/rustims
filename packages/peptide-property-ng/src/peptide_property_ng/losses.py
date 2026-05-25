"""Multi-task losses for the unified peptide property model."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def masked_spectral_angle(
    pred: torch.Tensor,
    target: torch.Tensor,
    norm_eps: float = 1e-8,
    clamp_eps: float = 1e-4,
) -> torch.Tensor:
    """Per-sample spectral-angle distance in ``[0, 1]`` (0 = identical spectra).

    Canonical Prosit / Sage-fine-tune loss: mask ``target > 0`` (only observed
    peaks contribute). Zeros are treated as "unmatched / unknown" rather than
    real "no peak" labels — which is the correct interpretation for Sage
    ``matched_fragments`` (Sage only reports peaks it matched) and matches the
    production v4 / imspy_predictors `masked_spectral_distance` convention.

    The two eps are deliberately decoupled:
      - ``norm_eps`` (1e-8) keeps ``F.normalize`` accurate for non-degenerate
        magnitudes;
      - ``clamp_eps`` (1e-4) bounds ``arccos``'s near-boundary derivative
        (≈ -1/sqrt(2·clamp_eps), so ~70 instead of ~7000 at 1e-8). With 1e-8
        the SA-loss backward NaN-s on sparse one-positive-fragment targets
        whose normalize-to-one-hot forces cos exactly to ±1; with 1e-4 those
        same samples are absorbed by the clamp (grad 0 above 1-clamp_eps).

    Pretraining on PROSPECT-style densely-annotated targets is unaffected in
    practice — observable b/y positions almost all carry positive intensity,
    so the > 0 vs >= 0 masks coincide and cos rarely hits the boundary.
    """
    # Pooled head outputs a fixed (B, 29, n_ion) grid; site head outputs
    # (B, L-1, n_ion). Slice pred to the batch's target sites so both shapes
    # work transparently here.
    if pred.shape[1] > target.shape[1]:
        pred = pred[:, : target.shape[1], :]
    mask = (target > 0).float()
    p = (pred * mask).reshape(pred.shape[0], -1)
    t = (target * mask).reshape(target.shape[0], -1)
    p = F.normalize(p, dim=1, eps=norm_eps)
    t = F.normalize(t, dim=1, eps=norm_eps)
    cos = (p * t).sum(dim=1).clamp(-1.0 + clamp_eps, 1.0 - clamp_eps)
    return 2.0 * torch.arccos(cos) / math.pi


def intensity_signal_mask(target: torch.Tensor) -> torch.Tensor:
    """``(B,)`` bool — True where a target has at least one observed (positive) peak.

    A target with no positive b/y charge-1-3 peak is degenerate: the spectral
    angle against it is a constant with zero gradient, so such samples are
    excluded from the intensity loss and metric rather than diluting them.
    """
    return (target > 0).flatten(1).any(dim=1)


class MultiTaskLoss:
    """Weighted sum of per-task losses; tasks with no valid targets are skipped.

    - intensity : masked spectral-angle distance
    - ccs       : Gaussian negative log-likelihood (mean, std) vs 1/K0
    - rt        : L1 on normalised retention time
    - charge    : cross-entropy
    """

    DEFAULT_WEIGHTS = {"intensity": 1.0, "ccs": 1.0, "rt": 0.5, "charge": 0.3}

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(self.DEFAULT_WEIGHTS if weights is None else weights)

    def __call__(
        self,
        outputs: dict[str, object],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        parts: dict[str, torch.Tensor] = {}

        if "intensity" in outputs:
            sa = masked_spectral_angle(outputs["intensity"], batch["intensity_target"])
            signal = intensity_signal_mask(batch["intensity_target"])
            if signal.any():
                parts["intensity"] = sa[signal].mean()

        if "ccs" in outputs:
            mean, _std = outputs["ccs"]
            valid = batch["ccs_valid"]
            if valid.any():
                # L1 on the mean — stable for the prototype. Gaussian-NLL on
                # (mean, std) is the eventual target but is unbounded-below and
                # destabilises the fixed-weight multi-task sum; revisit together
                # with uncertainty-based task weighting.
                parts["ccs"] = F.l1_loss(mean[valid], batch["ccs_target"][valid])

        if "rt" in outputs:
            valid = batch["rt_valid"]
            if valid.any():
                parts["rt"] = F.l1_loss(outputs["rt"][valid], batch["rt_target"][valid])

        if "charge" in outputs:
            parts["charge"] = F.cross_entropy(outputs["charge"], batch["charge_target"])

        if not parts:
            raise ValueError("no task produced a loss for this batch")
        total = sum(self.weights.get(name, 1.0) * value for name, value in parts.items())
        return total, {name: float(value.detach()) for name, value in parts.items()}
