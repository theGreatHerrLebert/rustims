"""Fragment-indexed intensity head.

Unlike Prosit's fixed 174-value vector (29 positions x 6 channels -> a hard
30-residue cap), this head predicts one intensity per *cleavage site*. A peptide
of length L has L-1 sites; the site representation is built from the two
flanking residue outputs, so the output is ``(batch, L-1, n_ion_channels)`` and
the peptide length is bounded only by the encoder's ``max_seq_len``.

Charge and collision energy are conditioned here (head-level) via FiLM — they
are deliberately kept out of the shared encoder (see config.py).
"""
from __future__ import annotations

import torch
from torch import nn

from peptide_property_ng.model.config import ModelConfig


class IntensityHead(nn.Module):
    """Predict per-cleavage-site fragment intensities in ``[0, 1]``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.charge_emb = nn.Embedding(cfg.max_charge + 1, d)
        self.ce_proj = nn.Linear(1, d)

        self.site_in = nn.Linear(2 * d, d)
        # FiLM modulation from (charge, CE); zero-init -> starts as identity.
        self.film = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, cfg.n_ion_channels),
        )
        self.max_charge = cfg.max_charge

    def forward(
        self,
        latent: torch.Tensor,        # (B, 1+L, d) — index 0 is the global token
        charge: torch.Tensor,        # (B,) long
        collision_energy: torch.Tensor,  # (B,) float (0.0 when unknown)
    ) -> torch.Tensor:
        residues = latent[:, 1:]                       # (B, L, d)
        sites = torch.cat([residues[:, :-1], residues[:, 1:]], dim=-1)  # (B, L-1, 2d)
        h = self.site_in(sites)                        # (B, L-1, d)

        cond = self.charge_emb(charge.clamp(0, self.max_charge))
        cond = cond + self.ce_proj(collision_energy.unsqueeze(-1).float())  # (B, d)
        scale, shift = self.film(cond).chunk(2, dim=-1)                      # (B, d) each
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]

        return torch.sigmoid(self.mlp(h))              # (B, L-1, n_ion_channels)
