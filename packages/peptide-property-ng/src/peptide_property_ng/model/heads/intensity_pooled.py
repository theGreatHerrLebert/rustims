"""V4-compatible pooled intensity head — Prosit-174 ablation.

Mirrors ``imspy_predictors.models.heads.IntensityHead``: attention-pool the
residue latents, concat with charge / CE / instrument side-embeddings, MLP →
sigmoid → reshape to ``(B, max_sites, n_ion_channels)``.

The hypothesis (codex review): a global head that predicts the *whole* Prosit-174
vector can model cross-fragment competition ("this peptide's base peak should
dominate, suppress the rest"), which the per-cleavage-site FiLM head in
``intensity.py`` cannot express except through encoder context. If swapping
heads on the same pretrained encoder closes the remaining SA gap to v4 (~0.83),
the gap is head topology, not capacity.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from peptide_property_ng.model.config import ModelConfig


# Prosit-174 ceiling — fragments per peptide cap (29 = 30 aa - 1 cleavage site).
# This is the public-corpus ground truth shape; even though ``cfg.max_seq_len``
# allows longer peptides in the encoder, intensity supervision is bounded here.
_MAX_SITES = 29


class PooledIntensityHead(nn.Module):
    """Predict ``(B, 29, n_ion_channels)`` via attention-pool + side-conditioned MLP."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.max_sites = _MAX_SITES
        self.n_ion = cfg.n_ion_channels
        self.num_outputs = self.max_sites * self.n_ion
        self.max_charge = cfg.max_charge

        # Single-head self-attention over the residue latents.
        self.attention = nn.MultiheadAttention(
            embed_dim=d, num_heads=1, batch_first=True, dropout=cfg.dropout,
        )

        # Side embeddings — match v4's d_model // 4 sizing.
        side = max(d // 4, 16)
        self.charge_embed = nn.Linear(cfg.max_charge + 1, side)
        self.ce_embed = nn.Linear(1, side)
        # Instrument conditioning lives in the head (v4 parity) on top of the
        # encoder's global-token conditioning, not as a replacement.
        self.instrument_embed = nn.Embedding(cfg.n_instruments, side)

        in_dim = d + 3 * side
        hidden = max(d * 2, 512)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, self.num_outputs),
        )

    def forward(
        self,
        latent: torch.Tensor,            # (B, 1+L, d) — index 0 is the global token
        charge: torch.Tensor,            # (B,) long
        collision_energy: torch.Tensor,  # (B,) float
        instrument: torch.Tensor | None = None,    # (B,) long
        padding_mask: torch.Tensor | None = None,  # (B, L) bool, True at pad
    ) -> torch.Tensor:
        residues = latent[:, 1:]                       # (B, L, d)
        B = residues.size(0)
        device = residues.device

        # Self-attention pool, then mean over valid residues.
        pooled, _ = self.attention(
            residues, residues, residues, key_padding_mask=padding_mask,
        )
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()
            denom = valid.sum(dim=1).clamp(min=1.0)
            pooled = (pooled * valid).sum(dim=1) / denom    # (B, d)
        else:
            pooled = pooled.mean(dim=1)

        # Side-info embeddings (v4 layout).
        chg = charge.view(-1).long().clamp(0, self.max_charge)
        chg_onehot = F.one_hot(chg, num_classes=self.max_charge + 1).float()
        chg_emb = self.charge_embed(chg_onehot)
        ce_emb = self.ce_embed(collision_energy.view(-1, 1).float())
        if instrument is None:
            instrument = torch.zeros(B, dtype=torch.long, device=device)
        inst_emb = self.instrument_embed(instrument.view(-1).long())

        features = torch.cat([pooled, chg_emb, ce_emb, inst_emb], dim=-1)
        out = torch.sigmoid(self.fc(features))           # (B, max_sites * n_ion)
        return out.view(B, self.max_sites, self.n_ion)
