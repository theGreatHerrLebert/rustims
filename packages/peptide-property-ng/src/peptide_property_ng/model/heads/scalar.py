"""Scalar heads — retention time and precursor charge.

Both read only the encoder's global token (index 0). That token aggregates the
peptide sequence plus instrument / acquisition-mode conditioning — but *not*
charge, m/z or collision energy (those are kept out of the encoder). So the
charge head never sees its own label: no leakage, one encoder pass.
"""
from __future__ import annotations

import torch
from torch import nn

from peptide_property_ng.model.config import ModelConfig


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.LayerNorm(d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_hidden // 2),
        nn.GELU(),
        nn.Linear(d_hidden // 2, d_out),
    )


class RetentionTimeHead(nn.Module):
    """Predict (per-dataset min-max normalised) retention time."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = _mlp(cfg.d_model, cfg.d_model, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent[:, 0]).squeeze(-1)  # (B,)


class ChargeHead(nn.Module):
    """Predict precursor-charge logits — class ``j`` corresponds to charge ``j + 1``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.max_charge = cfg.max_charge
        self.net = _mlp(cfg.d_model, cfg.d_model, cfg.max_charge)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent[:, 0])  # (B, max_charge) logits
