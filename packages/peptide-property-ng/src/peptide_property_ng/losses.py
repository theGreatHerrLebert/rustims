"""Multi-task losses for the unified peptide property model."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def masked_spectral_angle(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample spectral-angle distance in ``[0, 1]`` (0 = identical spectra).

    ``pred`` / ``target`` are ``(B, sites, channels)``. Entries where
    ``target < 0`` are masked out (impossible / padded fragment channels).
    """
    mask = (target >= 0).float()
    p = (pred * mask).reshape(pred.shape[0], -1)
    t = (target.clamp(min=0.0) * mask).reshape(target.shape[0], -1)
    p = F.normalize(p, dim=1, eps=eps)
    t = F.normalize(t, dim=1, eps=eps)
    cos = (p * t).sum(dim=1).clamp(-1.0 + eps, 1.0 - eps)
    return 2.0 * torch.arccos(cos) / math.pi


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
            parts["intensity"] = masked_spectral_angle(
                outputs["intensity"], batch["intensity_target"]
            ).mean()

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
