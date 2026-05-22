"""Ion-mobility head — a per-charge physics prior plus a learned correction.

Predicts inverse reduced ion mobility (1/K0), the raw quantity Sage reports in
its ``ion_mobility`` column — this avoids baking a Mason-Schamp CCS conversion
(with its gas/temperature constants) into the prototype.

The physics prior ``1/K0 ~ slope[z]*sqrt(m/z) + intercept[z]`` is ported from
the production ``SquareRootProjectionLayer`` (imspy_predictors); slopes and
intercepts are learnable. A Gaussian-NLL ``(mean, std)`` output keeps a
calibrated uncertainty, useful downstream for the pep-centric SNR work.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from peptide_property_ng.model.config import ModelConfig


class SqrtMzProjection(nn.Module):
    """Per-charge physics prior: ``property ~ slope[z]*sqrt(m/z) + intercept[z]``."""

    def __init__(self, max_charge: int, slope_init: float = 0.03, intercept_init: float = 0.60):
        super().__init__()
        # Index by charge state directly (row 0 unused). Init at a 1/K0 ballpark.
        self.slopes = nn.Parameter(torch.full((max_charge + 1,), float(slope_init)))
        self.intercepts = nn.Parameter(torch.full((max_charge + 1,), float(intercept_init)))

    def forward(self, mz: torch.Tensor, charge: torch.Tensor) -> torch.Tensor:
        z = charge.clamp(0, self.slopes.shape[0] - 1)
        return self.slopes[z] * torch.sqrt(mz.clamp(min=0.0)) + self.intercepts[z]


class IonMobilityHead(nn.Module):
    """Predict inverse ion mobility (1/K0) as ``(mean, std)``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.max_charge = cfg.max_charge
        self.physics = SqrtMzProjection(cfg.max_charge)
        self.charge_emb = nn.Embedding(cfg.max_charge + 1, d)
        self.net = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 2),  # correction to the mean, log-std
        )

    def forward(
        self,
        latent: torch.Tensor,        # (B, 1+L, d)
        charge: torch.Tensor,        # (B,) long
        precursor_mz: torch.Tensor,  # (B,) float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        glob = latent[:, 0]                                       # (B, d)
        h = torch.cat([glob, self.charge_emb(charge.clamp(0, self.max_charge))], dim=-1)
        correction, log_std = self.net(h).unbind(dim=-1)          # (B,), (B,)
        mean = self.physics(precursor_mz, charge) + correction
        std = F.softplus(log_std) + 1e-4
        return mean, std
